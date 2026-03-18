#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import random
from pathlib import Path
import sys

import torch
import warp as wp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from blender_temp.cmd.main import _restore_rng_state, collect_image_paths, load_images
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
    render_stats_prepared_warp,
    render_values_warp,
)
from blender_temp.gaussian_sr.benchmarking import (
    aggregate_step_metrics,
    compare_render_summary,
    select_compare_views,
    summarize_render_output,
)
from blender_temp.gaussian_sr.debug_checkpoint import load_debug_checkpoint, restore_module_from_debug_checkpoint
from blender_temp.gaussian_sr.pipeline import set_torch_compile_enabled


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


def _cpu_clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    return value


def _limit_steps_to_target(
    cfg: PoseFreeGaussianConfig,
    *,
    resume_stage_index: int,
    resume_step_index: int,
    resume_global_step: int,
    target_global_step: int,
) -> None:
    if target_global_step < resume_global_step:
        raise ValueError("target_global_step must be >= resume_global_step")
    steps_per_stage = list(int(step) for step in cfg.train.steps_per_stage)
    additional_steps = int(target_global_step - resume_global_step)
    steps_per_stage[resume_stage_index] = max(
        resume_step_index + 1,
        resume_step_index + 1 + additional_steps,
    )
    for stage_offset in range(resume_stage_index + 1, len(steps_per_stage)):
        steps_per_stage[stage_offset] = 0
    cfg.train.steps_per_stage = tuple(steps_per_stage)


def _snapshot_dynamo_counters() -> dict[str, dict[str, int]]:
    try:
        from torch._dynamo.utils import counters
    except Exception:
        return {}
    snapshot: dict[str, dict[str, int]] = {}
    for group, counter in counters.items():
        snapshot[group] = {key: int(value) for key, value in counter.items()}
    return snapshot


def _counter_delta(
    before: dict[str, dict[str, int]],
    after: dict[str, dict[str, int]],
) -> dict[str, dict[str, int]]:
    delta: dict[str, dict[str, int]] = {}
    for group in sorted(set(before) | set(after)):
        group_before = before.get(group, {})
        group_after = after.get(group, {})
        group_delta: dict[str, int] = {}
        for key in sorted(set(group_before) | set(group_after)):
            value = int(group_after.get(key, 0) - group_before.get(key, 0))
            if value != 0:
                group_delta[key] = value
        if group_delta:
            delta[group] = group_delta
    return delta


def _parse_compare_views(arg: str | None, num_views: int) -> tuple[int, ...]:
    if arg is None or arg.strip() == "":
        return select_compare_views(num_views)
    indices: list[int] = []
    for token in arg.split(","):
        idx = int(token.strip())
        idx = max(0, min(idx, num_views - 1))
        if idx not in indices:
            indices.append(idx)
    return tuple(indices)


def _apply_config_overrides(cfg: PoseFreeGaussianConfig, args: argparse.Namespace) -> dict[str, int | float | bool]:
    applied: dict[str, int | float | bool] = {}

    if args.ordinary_step_view_batch is not None:
        cfg.train.ordinary_step_view_batch = int(args.ordinary_step_view_batch)
        applied["ordinary_step_view_batch"] = int(cfg.train.ordinary_step_view_batch)

    if args.final_stage_views_per_microbatch is not None:
        cfg.train.final_stage_views_per_microbatch = int(args.final_stage_views_per_microbatch)
        applied["final_stage_views_per_microbatch"] = int(cfg.train.final_stage_views_per_microbatch)

    if args.final_stage_max_steps is not None:
        cfg.train.final_stage_max_steps = int(args.final_stage_max_steps)
        applied["final_stage_max_steps"] = int(cfg.train.final_stage_max_steps)

    if args.final_stage_early_stop_patience is not None:
        cfg.train.final_stage_early_stop_patience = int(args.final_stage_early_stop_patience)
        applied["final_stage_early_stop_patience"] = int(cfg.train.final_stage_early_stop_patience)

    if args.final_stage_early_stop_min_step is not None:
        cfg.train.final_stage_early_stop_min_step = int(args.final_stage_early_stop_min_step)
        applied["final_stage_early_stop_min_step"] = int(cfg.train.final_stage_early_stop_min_step)

    if args.final_stage_early_stop_loss_delta is not None:
        cfg.train.final_stage_early_stop_loss_delta = float(args.final_stage_early_stop_loss_delta)
        applied["final_stage_early_stop_loss_delta"] = float(cfg.train.final_stage_early_stop_loss_delta)

    if args.freeze_after_stable_events is not None:
        cfg.density.freeze_after_stable_events = int(args.freeze_after_stable_events)
        applied["freeze_after_stable_events"] = int(cfg.density.freeze_after_stable_events)

    if args.freeze_min_visible_fraction is not None:
        cfg.density.freeze_min_visible_fraction = float(args.freeze_min_visible_fraction)
        applied["freeze_min_visible_fraction"] = float(cfg.density.freeze_min_visible_fraction)

    if args.freeze_min_intersection_fraction is not None:
        cfg.density.freeze_min_intersection_fraction = float(args.freeze_min_intersection_fraction)
        applied["freeze_min_intersection_fraction"] = float(cfg.density.freeze_min_intersection_fraction)

    if args.helion_runtime_autotune is not None:
        cfg.render.helion_runtime_autotune = bool(args.helion_runtime_autotune)
        applied["helion_runtime_autotune"] = bool(cfg.render.helion_runtime_autotune)

    if args.helion_static_shapes is not None:
        cfg.render.helion_static_shapes = bool(args.helion_static_shapes)
        applied["helion_static_shapes"] = bool(cfg.render.helion_static_shapes)

    return applied


def _profiler_activities(device: torch.device) -> list[torch.profiler.ProfilerActivity]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda" and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return activities


def _write_pytorch_profile(profiler: torch.profiler.profile, output_dir: Path, *, row_limit: int) -> dict[str, str]:
    profile_dir = output_dir / "pytorch_profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    trace_path = profile_dir / "trace.json"
    profiler.export_chrome_trace(str(trace_path))

    cuda_table = profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=row_limit)
    cpu_table = profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=row_limit)
    (profile_dir / "summary_cuda.txt").write_text(cuda_table, encoding="utf-8")
    (profile_dir / "summary_cpu.txt").write_text(cpu_table, encoding="utf-8")
    return {
        "trace": str(trace_path),
        "summary_cuda": str((profile_dir / "summary_cuda.txt").resolve()),
        "summary_cpu": str((profile_dir / "summary_cpu.txt").resolve()),
    }


def _resize_target(target: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    if target.shape[-2:] == (out_h, out_w):
        return target
    return torch.nn.functional.interpolate(
        target.unsqueeze(0),
        size=(out_h, out_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def _stage_render_size(
    pipeline: PoseFreeGaussianSR,
    cfg: PoseFreeGaussianConfig,
    resume_stage_index: int,
) -> tuple[float, int, int]:
    if 0 <= resume_stage_index < len(cfg.train.stage_scales):
        scale = float(cfg.train.stage_scales[resume_stage_index])
    else:
        scale = 1.0
    out_h = max(1, int(round(pipeline.train_height * scale)))
    out_w = max(1, int(round(pipeline.train_width * scale)))
    return scale, out_h, out_w


def _top_kernel_rows(kernel_totals: dict[str, float], limit: int = 10) -> list[dict[str, float | str]]:
    rows = sorted(kernel_totals.items(), key=lambda item: item[1], reverse=True)
    return [{"name": name, "elapsed_ms": float(elapsed)} for name, elapsed in rows[:limit]]


def _profile_warp_render_path(
    pipeline: PoseFreeGaussianSR,
    images: torch.Tensor,
    cfg: PoseFreeGaussianConfig,
    *,
    resume_stage_index: int,
    view_indices: tuple[int, ...],
    repeats: int,
    output_dir: Path,
) -> dict[str, object]:
    scale, out_h, out_w = _stage_render_size(pipeline, cfg, resume_stage_index)
    render_intr = pipeline._scale_intrinsics(out_h, out_w)
    R_all, t_all = pipeline.camera_model.world_to_camera()
    renderer_cfg = pipeline._renderer_config()

    profile_payload: dict[str, object] = {
        "stage_index": int(resume_stage_index),
        "render_scale": float(scale),
        "render_height": int(out_h),
        "render_width": int(out_w),
        "repeats": int(repeats),
        "views": [],
    }

    for view_index in view_indices:
        target = _resize_target(images[view_index], out_h, out_w)
        op_elapsed_ms: dict[str, list[float]] = defaultdict(list)
        op_kernel_ms: dict[str, dict[str, float]] = {
            "render_values_warp": defaultdict(float),
            "render_stats_prepared_warp": defaultdict(float),
        }
        visible_count = 0
        intersection_count = 0
        gaussian_count = 0

        for _ in range(max(1, repeats)):
            field, viewmat, K, means3d, quat, scale_render, opacity, values, background, active_count = (
                pipeline._prepare_render_payload_eager(
                    render_intr,
                    R_all[view_index],
                    t_all[view_index],
                )
            )

            with wp.ScopedTimer(
                "render_values_warp",
                print=False,
                synchronize=True,
                cuda_filter=wp.TIMING_KERNEL,
            ) as render_timer:
                packed_hwc, prepared = render_values_warp(
                    means=means3d.contiguous(),
                    quat=quat.contiguous(),
                    scale=scale_render.contiguous(),
                    values=values.contiguous(),
                    opacity=opacity.contiguous(),
                    background=background.contiguous(),
                    viewmat=viewmat.contiguous(),
                    K=K.contiguous(),
                    width=out_w,
                    height=out_h,
                    cfg=renderer_cfg,
                    return_prepared=True,
                    active_count=active_count,
                )

            op_elapsed_ms["render_values_warp"].append(float(render_timer.elapsed))
            for item in render_timer.timing_results:
                op_kernel_ms["render_values_warp"][item.name] += float(item.elapsed)

            packed = packed_hwc.permute(2, 0, 1).contiguous()
            rgb, _latent, _alpha = pipeline._postprocess_rgb(packed, out_h, out_w)
            residual_map = pipeline._residual_map_for_render(rgb, target, out_h, out_w)

            with wp.ScopedTimer(
                "render_stats_prepared_warp",
                print=False,
                synchronize=True,
                cuda_filter=wp.TIMING_KERNEL,
            ) as stats_timer:
                stats = render_stats_prepared_warp(
                    prepared,
                    opacity=opacity.contiguous(),
                    cfg=renderer_cfg,
                    residual_map=residual_map.contiguous(),
                    active_count=prepared.gaussian_count,
                )

            op_elapsed_ms["render_stats_prepared_warp"].append(float(stats_timer.elapsed))
            for item in stats_timer.timing_results:
                op_kernel_ms["render_stats_prepared_warp"][item.name] += float(item.elapsed)

            meta = prepared.meta()
            visible_count = int(meta["meta_visible_count"].item())
            intersection_count = int(meta["meta_intersection_count"].item())
            gaussian_count = int(meta["meta_gaussian_count"].item())
            _ = field

        view_payload = {
            "view_index": int(view_index),
            "visible_count": int(visible_count),
            "intersection_count": int(intersection_count),
            "gaussian_count": int(gaussian_count),
            "ops": {},
        }
        for op_name, values_ms in op_elapsed_ms.items():
            kernel_totals = op_kernel_ms[op_name]
            view_payload["ops"][op_name] = {
                "avg_elapsed_ms": float(sum(values_ms) / len(values_ms)),
                "min_elapsed_ms": float(min(values_ms)),
                "max_elapsed_ms": float(max(values_ms)),
                "top_kernels": _top_kernel_rows(kernel_totals),
            }
        profile_payload["views"].append(view_payload)

    warp_profile_path = output_dir / "warp_profile.json"
    warp_profile_path.write_text(json.dumps(profile_payload, indent=2), encoding="utf-8")
    return {
        "json": str(warp_profile_path.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--timed-steps", type=int, default=30)
    parser.add_argument("--compare-views", type=str, default=None)
    parser.add_argument("--baseline-dir", type=Path, default=None)
    parser.add_argument("--disable-torch-compile", action="store_true")
    parser.add_argument("--renderer-backend", choices=("warp", "helion"), default=None)
    parser.add_argument("--renderer-backward-impl", choices=("hybrid", "reference", "warp_tape"))
    parser.add_argument("--scenario-name", type=str, default="quick_benchmark")
    parser.add_argument("--run-until-stage-end", action="store_true")
    parser.add_argument("--ordinary-step-view-batch", type=int, default=None)
    parser.add_argument("--final-stage-views-per-microbatch", type=int, default=None)
    parser.add_argument("--final-stage-max-steps", type=int, default=None)
    parser.add_argument("--final-stage-early-stop-patience", type=int, default=None)
    parser.add_argument("--final-stage-early-stop-min-step", type=int, default=None)
    parser.add_argument("--final-stage-early-stop-loss-delta", type=float, default=None)
    parser.add_argument("--freeze-after-stable-events", type=int, default=None)
    parser.add_argument("--freeze-min-visible-fraction", type=float, default=None)
    parser.add_argument("--freeze-min-intersection-fraction", type=float, default=None)
    parser.add_argument("--helion-runtime-autotune", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--helion-static-shapes", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--profile-pytorch", action="store_true")
    parser.add_argument("--profile-warp", action="store_true")
    parser.add_argument("--skip-compare-renders", action="store_true")
    parser.add_argument("--pytorch-row-limit", type=int, default=50)
    parser.add_argument("--warp-profile-repeats", type=int, default=5)
    parser.add_argument("--warp-profile-views", type=str, default=None)
    args = parser.parse_args()

    payload = load_debug_checkpoint(args.checkpoint)
    config_payload = payload.get("config")
    if not isinstance(config_payload, dict):
        raise SystemExit("Checkpoint is missing an embedded config")
    cfg = _load_config(config_payload)
    applied_overrides = _apply_config_overrides(cfg, args)
    if args.renderer_backend is not None:
        cfg.render.backend = str(args.renderer_backend)
        applied_overrides["renderer_backend"] = str(cfg.render.backend)

    set_torch_compile_enabled(not args.disable_torch_compile)
    device = torch.device(args.device)
    images = load_images(collect_image_paths(args.input_dir.resolve()), device)
    pipeline = PoseFreeGaussianSR.from_images(images, intrinsics=None, config=cfg).to(device)
    restore_module_from_debug_checkpoint(pipeline, args.checkpoint)
    _restore_rng_state(payload)

    if args.renderer_backward_impl is not None or args.renderer_backend is not None:
        original_renderer_config = pipeline._renderer_config

        def _patched_renderer_config():
            render_cfg = original_renderer_config()
            if args.renderer_backward_impl is not None:
                render_cfg.backward_impl = args.renderer_backward_impl
            if args.renderer_backend is not None:
                render_cfg.backend = str(args.renderer_backend)
            return render_cfg

        pipeline._renderer_config = _patched_renderer_config

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    compare_dir = output_dir / "compare_views"
    compare_dir.mkdir(parents=True, exist_ok=True)

    resume_stage_index = int(payload.get("stage_index", -1))
    resume_step_index = int(payload.get("step_index", -1))
    resume_global_step = int(payload.get("global_step", -1))
    if resume_global_step < 0 and resume_stage_index > 0:
        resume_global_step = int(sum(int(s) for s in cfg.train.steps_per_stage[:resume_stage_index])) - 1

    total_requested_steps = int(args.warmup_steps) + int(args.timed_steps)
    if args.run_until_stage_end:
        if resume_stage_index < 0:
            raise SystemExit("--run-until-stage-end requires a valid checkpoint stage_index")
        target_global_step = resume_global_step + total_requested_steps
    else:
        target_global_step = resume_global_step + total_requested_steps
        _limit_steps_to_target(
            cfg,
            resume_stage_index=resume_stage_index,
            resume_step_index=resume_step_index,
            resume_global_step=resume_global_step,
            target_global_step=target_global_step,
        )

    optimizer_state_dict = payload.get("optimizer_state_dict")
    if not isinstance(optimizer_state_dict, dict):
        optimizer_state_dict = None

    progress_events: list[dict] = []
    density_events: list[dict] = []
    dynamo_before = _snapshot_dynamo_counters()
    profiler_artifacts: dict[str, str] | None = None
    if args.profile_pytorch:
        with torch.profiler.profile(
            activities=_profiler_activities(device),
            record_shapes=False,
            profile_memory=True,
            with_stack=False,
            acc_events=True,
        ) as profiler:
            history = pipeline.fit(
                images,
                verbose_progress=False,
                synchronize_progress_timing=True,
                optimizer_state_dict=optimizer_state_dict,
                resume_stage_index=resume_stage_index,
                resume_step_index=resume_step_index,
                resume_global_step=resume_global_step,
                progress_event_callback=progress_events.append,
                density_event_callback=density_events.append,
            )
        profiler_artifacts = _write_pytorch_profile(
            profiler,
            output_dir,
            row_limit=max(1, int(args.pytorch_row_limit)),
        )
    else:
        history = pipeline.fit(
            images,
            verbose_progress=False,
            synchronize_progress_timing=True,
            optimizer_state_dict=optimizer_state_dict,
            resume_stage_index=resume_stage_index,
            resume_step_index=resume_step_index,
            resume_global_step=resume_global_step,
            progress_event_callback=progress_events.append,
            density_event_callback=density_events.append,
        )
    dynamo_after = _snapshot_dynamo_counters()

    warmup_start = resume_global_step + 1
    warmup_end = resume_global_step + int(args.warmup_steps)
    timed_start = warmup_end + 1
    timed_end = target_global_step

    step_end_events = [
        event
        for event in progress_events
        if event.get("event") == "step_end" and timed_start <= int(event["global_step"]) <= timed_end
    ]
    ordinary_steps = [event for event in step_end_events if not bool(event.get("density_due", False))]
    density_steps = [event for event in step_end_events if bool(event.get("density_due", False))]

    compare_views = _parse_compare_views(args.compare_views, pipeline.num_views)
    warp_profile_views = _parse_compare_views(args.warp_profile_views, pipeline.num_views)
    render_summaries: list[dict[str, float | int]] = []
    compare_results: list[dict[str, object]] = []
    baseline_dir = None if args.baseline_dir is None else args.baseline_dir.resolve()
    if not args.skip_compare_renders:
        for view_index in compare_views:
            render = pipeline.render_view(view_index, return_aux=True, stats_mode="meta")
            rgb = render["rgb"].detach().cpu()
            summary = summarize_render_output(view_index, render["rgb"], render.get("render_stats"))
            render_summaries.append(summary)
            torch.save(
                {
                    "summary": summary,
                    "rgb": _cpu_clone(rgb),
                },
                compare_dir / f"view_{view_index:02d}.pt",
            )
            if baseline_dir is not None:
                baseline_payload = torch.load(
                    baseline_dir / "compare_views" / f"view_{view_index:02d}.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                baseline_summary = baseline_payload["summary"]
                baseline_rgb = baseline_payload["rgb"].to(dtype=rgb.dtype)
                l1 = float((rgb - baseline_rgb).abs().mean().item())
                compare_results.append(compare_render_summary(summary, baseline_summary, l1=l1))

    warp_profile_artifacts: dict[str, str] | None = None
    if args.profile_warp and cfg.render.backend != "warp":
        raise SystemExit("--profile-warp is only supported with --renderer-backend warp")
    if args.profile_warp:
        warp_profile_artifacts = _profile_warp_render_path(
            pipeline,
            images,
            cfg,
            resume_stage_index=resume_stage_index,
            view_indices=warp_profile_views,
            repeats=max(1, int(args.warp_profile_repeats)),
            output_dir=output_dir,
        )

    metrics = {
        "scenario_name": args.scenario_name,
        "renderer_backend": str(cfg.render.backend),
        "checkpoint": str(args.checkpoint.resolve()),
        "applied_overrides": applied_overrides,
        "warmup_steps": int(args.warmup_steps),
        "timed_steps": int(args.timed_steps),
        "warmup_window": {
            "global_step_start": int(warmup_start),
            "global_step_end": int(warmup_end),
        },
        "timed_window": {
            "global_step_start": int(timed_start),
            "global_step_end": int(timed_end),
        },
        "history_lengths": {key: len(value) for key, value in history.items()},
        "progress_event_count": int(len(progress_events)),
        "ordinary_step_metrics": aggregate_step_metrics(ordinary_steps),
        "density_step_metrics": aggregate_step_metrics(density_steps),
        "timed_window_metrics": aggregate_step_metrics(step_end_events),
        "dynamo_counter_delta": _counter_delta(dynamo_before, dynamo_after),
        "compare_views": list(compare_views),
        "render_summaries": render_summaries,
        "density_event_count": len(density_events),
        "pytorch_profile_artifacts": profiler_artifacts,
        "warp_profile_artifacts": warp_profile_artifacts,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    compare_payload = {
        "baseline_dir": None if baseline_dir is None else str(baseline_dir),
        "results": compare_results,
        "passed": all(bool(result["passed"]) for result in compare_results) if compare_results else None,
    }
    (output_dir / "compare.json").write_text(json.dumps(compare_payload, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "metrics_path": str((output_dir / "metrics.json").resolve()),
                "compare_path": str((output_dir / "compare.json").resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
