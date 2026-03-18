#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from blender_temp.cmd.main import collect_image_paths, load_images
from blender_temp.gaussian_sr import (
    AppearanceConfig,
    CameraInit,
    DensityControlConfig,
    FieldConfig,
    ObservationConfig,
    PoseFreeGaussianConfig,
    PoseFreeGaussianSR,
    RenderConfig,
    TrainConfig,
    apply_density_control,
    should_run_density_control_for_stage,
)
from blender_temp.gaussian_sr.density_control import DensityViewCoverage, DensityViewObservation
from blender_temp.gaussian_sr.debug_checkpoint import restore_module_from_debug_checkpoint
from blender_temp.gaussian_sr.pipeline import _META_STAT_KEYS
from blender_temp.gaussian_sr.observation_model import observation_render_size
from blender_temp.gaussian_sr.warp_gsplat_autograd import clear_warp_launch_cache


def _load_config(data: dict) -> PoseFreeGaussianConfig:
    return PoseFreeGaussianConfig(
        camera=CameraInit(**data["camera"]),
        render=RenderConfig(**data["render"]),
        observation=ObservationConfig(**data["observation"]),
        appearance=AppearanceConfig(**data["appearance"]),
        density=DensityControlConfig(**data["density"]),
        field=FieldConfig(**data["field"]),
        train=TrainConfig(**data["train"]),
    )


def _freeze_module(module: torch.nn.Module | None) -> None:
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad_(False)


def _nonfinite_index_report(module: torch.nn.Module) -> dict[str, list[list[int]]]:
    report: dict[str, list[list[int]]] = {}
    for name, param in module.named_parameters():
        if param.grad is None:
            continue
        mask = ~torch.isfinite(param.grad)
        if not bool(mask.any()):
            continue
        coords = mask.nonzero(as_tuple=False)[:8]
        report[name] = coords.detach().cpu().tolist()
    return report


def _restore_rng_state(payload: dict[str, object]) -> bool:
    rng_state = payload.get("rng_state")
    if not isinstance(rng_state, dict):
        return False
    torch_state = rng_state.get("torch")
    if isinstance(torch_state, torch.Tensor):
        torch.set_rng_state(torch_state.cpu())
    cuda_state = rng_state.get("cuda")
    if isinstance(cuda_state, list) and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([
            state.cpu() if isinstance(state, torch.Tensor) else state for state in cuda_state
        ])
    python_random = rng_state.get("python_random")
    if python_random is not None:
        random.setstate(python_random)
    return True


def _optimizer_to_device(opt: torch.optim.Optimizer, device: torch.device) -> None:
    for state in opt.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device)


def _run_replay(
    *,
    checkpoint_path: Path,
    input_dir: Path,
    steps: int,
    freeze_camera: bool,
    freeze_residual: bool,
    backward_impl: str | None,
    view_ids: list[int] | None,
    min_log_scale: float | None,
    simulate_verbose_progress: bool,
) -> dict[str, object]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    cfg = _load_config(payload["config"])
    active_min_log_scale = cfg.field.min_log_scale if min_log_scale is None else float(min_log_scale)

    image_paths = collect_image_paths(input_dir)
    images = load_images(image_paths, torch.device("cuda"))

    pipeline = PoseFreeGaussianSR.from_images(images, intrinsics=None, config=cfg).to(images.device)
    restore_module_from_debug_checkpoint(pipeline, checkpoint_path)
    pipeline.train()
    pipeline.field_model.enforce_scale_floor(active_min_log_scale)
    if backward_impl is not None:
        original_renderer_config = pipeline._renderer_config

        def _patched_renderer_config():
            render_cfg = original_renderer_config()
            render_cfg.backward_impl = backward_impl
            return render_cfg

        pipeline._renderer_config = _patched_renderer_config

    if freeze_camera:
        _freeze_module(pipeline.camera_model)
    if freeze_residual:
        _freeze_module(pipeline.residual_head)

    opt = pipeline._make_optimizer(cfg.train)
    restored_optimizer_state = False
    optimizer_state_dict = payload.get("optimizer_state_dict")
    if isinstance(optimizer_state_dict, dict):
        opt.load_state_dict(optimizer_state_dict)
        _optimizer_to_device(opt, images.device)
        restored_optimizer_state = True
    restored_rng_state = _restore_rng_state(payload)

    stage_idx = int(payload["stage_index"])
    total_stages = len(cfg.train.stage_scales)
    stage_scale = float(cfg.train.stage_scales[stage_idx])
    stage_h = max(1, int(round(pipeline.train_height * stage_scale)))
    stage_w = max(1, int(round(pipeline.train_width * stage_scale)))
    render_h, render_w = observation_render_size(stage_h, stage_w, cfg.observation)
    stage_targets = (
        images
        if (stage_h == pipeline.train_height and stage_w == pipeline.train_width)
        else torch.nn.functional.interpolate(images, size=(stage_h, stage_w), mode="area")
    )
    stage_alpha = 0.0 if total_stages == 1 else stage_idx / (total_stages - 1)
    active_view_ids = list(range(pipeline.num_views)) if not view_ids else list(view_ids)
    stage_steps = int(cfg.train.steps_per_stage[stage_idx])
    checkpoint_step_index = int(payload["step_index"])
    debug_progress_enabled = bool(simulate_verbose_progress)
    timing_enabled = bool(simulate_verbose_progress)
    if checkpoint_step_index + steps >= stage_steps:
        raise ValueError("Replay across a stage boundary is not supported; reduce --steps for this checkpoint.")

    last_loss = None
    last_density_event: dict[str, object] | None = None
    for replay_step in range(steps):
        global_step = int(payload["global_step"]) + replay_step + 1
        density_due = should_run_density_control_for_stage(
            global_step,
            cfg.density,
            stage_idx,
            total_stages,
        )
        opt.zero_grad(set_to_none=True)
        intr = pipeline._scale_intrinsics(render_h, render_w)
        R_all, t_all = pipeline.camera_model.world_to_camera()

        photo_loss = images.new_tensor(0.0)
        rendered_any = False
        stats_accum: dict[str, torch.Tensor] | None = None
        per_view_observations: list[DensityViewObservation] = []
        for view_idx in active_view_ids:
            tgt = stage_targets[view_idx]
            if (
                timing_enabled
                or debug_progress_enabled
                or (density_due and cfg.density.weak_view_reseed_budget_per_view > 0)
            ):
                render = pipeline.render_with_pose(
                    R_all[view_idx],
                    t_all[view_idx],
                    render_h,
                    render_w,
                    return_aux=True,
                    stats_mode="meta",
                    return_prepared=density_due,
                    profile=timing_enabled,
                )
                pred, photo_term = pipeline._observe_and_photometric_loss(render["rgb"], tgt)
                if not torch.isfinite(render["rgb"]).all():
                    return {
                        "status": "nonfinite_render",
                        "global_step": global_step,
                        "view_index": view_idx,
                        "density_due": bool(density_due),
                    }
                if not torch.isfinite(pred).all():
                    return {
                        "status": "nonfinite_prediction",
                        "global_step": global_step,
                        "view_index": view_idx,
                        "density_due": bool(density_due),
                    }
                photo_loss = photo_loss + photo_term
                if density_due:
                    if timing_enabled:
                        pipeline._sync_for_timing(images.device)
                    residual_map = pipeline._residual_map_for_render(pred, tgt, render_h, render_w)
                    residual_stats = pipeline._render_stats_from_prepared(
                        render["_prepared_visibility"],
                        render["_opacity"],
                        residual_map=residual_map,
                    )
                    if timing_enabled:
                        pipeline._sync_for_timing(images.device)
                    stats_accum = pipeline._accumulate_render_stats(stats_accum, residual_stats)
                    render_summary = pipeline._render_stats_summary(render["render_stats"])
                    target_render = (
                        tgt
                        if tuple(tgt.shape[-2:]) == (render_h, render_w)
                        else torch.nn.functional.interpolate(
                            tgt.unsqueeze(0),
                            size=(render_h, render_w),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(0)
                    )
                    per_view_observations.append(
                        DensityViewObservation(
                            coverage=DensityViewCoverage(
                                view_index=int(view_idx),
                                visible_count=int(render_summary["visible_count"]),
                                intersection_count=int(render_summary["intersection_count"]),
                                render_width=int(render_summary["render_width"]),
                                render_height=int(render_summary["render_height"]),
                            ),
                            render_stats=residual_stats,
                            residual_map=residual_map.detach(),
                            target_rgb=target_render.detach(),
                            pred_rgb=pred.detach(),
                            R_cw=R_all[view_idx].detach(),
                            t_cw=t_all[view_idx].detach(),
                            intrinsics=torch.cat(tuple(t.detach() for t in intr)),
                        )
                    )
                else:
                    stats_accum = pipeline._accumulate_render_stats(stats_accum, render["render_stats"])
            else:
                if density_due:
                    train_view = pipeline._training_view_forward_density(
                        intr,
                        R_all[view_idx],
                        t_all[view_idx],
                        tgt,
                        render_h,
                        render_w,
                    )
                    meta_offset = 3 + len(_META_STAT_KEYS)
                    residual_stats = pipeline._density_stats_dict(train_view[meta_offset:])
                    stats_accum = pipeline._accumulate_render_stats(stats_accum, residual_stats)
                else:
                    train_view = pipeline._training_view_forward(
                        intr,
                        R_all[view_idx],
                        t_all[view_idx],
                        tgt,
                        render_h,
                        render_w,
                    )
                    meta_offset = 3 + len(_META_STAT_KEYS)
                    stats_accum = pipeline._accumulate_render_stats(
                        stats_accum,
                        pipeline._meta_dict(train_view[3:meta_offset]),
                    )
                photo_term = train_view[0].clone()
                rgb_finite = bool(train_view[1].item())
                pred_finite = bool(train_view[2].item())
                if not rgb_finite:
                    return {
                        "status": "nonfinite_render",
                        "global_step": global_step,
                        "view_index": view_idx,
                        "density_due": bool(density_due),
                    }
                if not pred_finite:
                    return {
                        "status": "nonfinite_prediction",
                        "global_step": global_step,
                        "view_index": view_idx,
                        "density_due": bool(density_due),
                    }
                photo_loss = photo_loss + photo_term
                if density_due:
                    render_summary = pipeline._render_stats_summary(pipeline._meta_dict(train_view[3:meta_offset]))
                    per_view_observations.append(
                        DensityViewObservation(
                            coverage=DensityViewCoverage(
                                view_index=int(view_idx),
                                visible_count=int(render_summary["visible_count"]),
                                intersection_count=int(render_summary["intersection_count"]),
                                render_width=int(render_summary["render_width"]),
                                render_height=int(render_summary["render_height"]),
                            ),
                            render_stats=residual_stats,
                            target_rgb=tgt.detach(),
                            R_cw=R_all[view_idx].detach(),
                            t_cw=t_all[view_idx].detach(),
                            intrinsics=torch.cat(tuple(t.detach() for t in intr)),
                        )
                    )
            rendered_any = True

        photo_loss = photo_loss / float(len(active_view_ids))
        if not rendered_any:
            raise RuntimeError("No views were rendered during replay step.")
        reg_loss = pipeline._regularization_impl(None, stage_alpha)
        loss = photo_loss + reg_loss
        if timing_enabled:
            pipeline._sync_for_timing(images.device)
        loss.backward()
        if timing_enabled:
            pipeline._sync_for_timing(images.device)
        if stage_idx == (total_stages - 1):
            for param in pipeline.camera_model.parameters():
                param.grad = None

        gradient_stats = pipeline._gradient_stats_report()
        gradient_nonfinite = {
            name: int(stats["nonfinite"]) for name, stats in gradient_stats.items() if int(stats["nonfinite"]) > 0
        }
        if gradient_nonfinite:
            return {
                "status": "nonfinite_gradients",
                "global_step": global_step,
                "density_due": bool(density_due),
                "gradient_nonfinite": gradient_nonfinite,
                "gradient_stats": gradient_stats,
                "gradient_indices": _nonfinite_index_report(pipeline),
                "loss": float(loss.detach().item()),
                "photo_loss": float(photo_loss.detach().item()),
                "reg_loss": float(reg_loss.detach().item()),
            }

        if cfg.train.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(pipeline.parameters(), cfg.train.grad_clip)
        if timing_enabled:
            pipeline._sync_for_timing(images.device)
        opt.step()
        if timing_enabled:
            pipeline._sync_for_timing(images.device)
        pipeline.field_model.enforce_protection(
            global_step,
            float(cfg.density.weak_view_reseed_min_opacity),
        )
        pipeline.field_model.enforce_scale_floor(active_min_log_scale)

        parameter_nonfinite = pipeline._parameter_nonfinite_report()
        if parameter_nonfinite:
            return {
                "status": "nonfinite_parameters",
                "global_step": global_step,
                "density_due": bool(density_due),
                "parameter_nonfinite": parameter_nonfinite,
                "loss": float(loss.detach().item()),
                "photo_loss": float(photo_loss.detach().item()),
                "reg_loss": float(reg_loss.detach().item()),
            }

        last_density_event = None
        if density_due:
            density_event = apply_density_control(
                pipeline.field_model,
                cfg.density,
                global_step,
                stage_index=stage_idx,
                total_stages=total_stages,
                render_stats=stats_accum,
                per_view_observations=per_view_observations,
            )
            last_density_event = {
                "ran": bool(density_event.ran),
                "changed": bool(density_event.changed),
                "before": int(density_event.before),
                "after": int(density_event.after),
                "pruned": int(density_event.pruned),
                "split": int(density_event.split),
                "cloned": int(density_event.cloned),
                "reseeded": int(density_event.reseeded),
            }
            if density_event.changed:
                clear_warp_launch_cache()
                pipeline.field_model.enforce_scale_floor(active_min_log_scale)
                opt = pipeline._rebuild_optimizer_after_density(
                    opt,
                    cfg.train,
                    density_event.survivor_sources,
                    int(density_event.appended_count),
                )
                pipeline._debug_optimizer = opt

        last_loss = float(loss.detach().item())

    return {
        "status": "ok",
        "checkpoint": str(checkpoint_path),
        "steps": int(steps),
        "start_global_step": int(payload["global_step"]),
        "end_global_step": int(payload["global_step"]) + int(steps),
        "freeze_camera": bool(freeze_camera),
        "freeze_residual": bool(freeze_residual),
        "backward_impl": backward_impl or "default",
        "view_ids": active_view_ids,
        "min_log_scale": active_min_log_scale,
        "restored_optimizer_state": restored_optimizer_state,
        "restored_rng_state": restored_rng_state,
        "simulate_verbose_progress": bool(simulate_verbose_progress),
        "config": asdict(cfg),
        "last_loss": last_loss,
        "last_density_event": last_density_event,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--freeze-camera", action="store_true")
    parser.add_argument("--freeze-residual", action="store_true")
    parser.add_argument("--backward-impl", choices=("warp_tape", "reference", "hybrid"))
    parser.add_argument("--view-ids", type=str, default="")
    parser.add_argument("--min-log-scale", type=float)
    parser.add_argument("--simulate-verbose-progress", action="store_true")
    args = parser.parse_args()

    view_ids = [int(x) for x in args.view_ids.split(",") if x.strip()] if args.view_ids else None
    result = _run_replay(
        checkpoint_path=args.checkpoint,
        input_dir=args.input_dir,
        steps=args.steps,
        freeze_camera=args.freeze_camera,
        freeze_residual=args.freeze_residual,
        backward_impl=args.backward_impl,
        view_ids=view_ids,
        min_log_scale=args.min_log_scale,
        simulate_verbose_progress=args.simulate_verbose_progress,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
