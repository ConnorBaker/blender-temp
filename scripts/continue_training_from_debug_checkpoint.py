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

from blender_temp.cmd.main import collect_image_paths, load_images, _restore_rng_state, _DebugRunObserver
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
)
from blender_temp.gaussian_sr.debug_checkpoint import load_debug_checkpoint, restore_module_from_debug_checkpoint
from blender_temp.gaussian_sr.pipeline import NonFiniteTrainingError, set_torch_compile_enabled


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


def _checkpoint_path(output_dir: Path, reason: str, stage_index: int, step_index: int, global_step: int) -> Path:
    safe_global_step = max(int(global_step), 0)
    stage_tag = f"stage{int(stage_index) + 1}"
    step_tag = f"step{int(step_index) + 1:04d}"
    return output_dir / "checkpoints" / f"{stage_tag}_{step_tag}_{reason}_g{safe_global_step:05d}.pt"


def _save_checkpoint(
    *,
    output_dir: Path,
    pipeline: PoseFreeGaussianSR,
    cfg: PoseFreeGaussianConfig,
    reason: str,
    stage_index: int,
    step_index: int,
    global_step: int,
    extra: dict[str, object] | None = None,
) -> Path:
    path = _checkpoint_path(output_dir, reason, stage_index, step_index, global_step)
    path.parent.mkdir(parents=True, exist_ok=True)
    optimizer = pipeline._debug_optimizer
    payload: dict[str, object] = {
        "config": asdict(cfg),
        "pipeline_state_dict": _cpu_clone(pipeline.state_dict()),
        "stage_index": int(stage_index),
        "step_index": int(step_index),
        "global_step": int(global_step),
        "optimizer_state_dict": None if optimizer is None else _cpu_clone(optimizer.state_dict()),
        "rng_state": {
            "torch": torch.get_rng_state().cpu().clone(),
            "cuda": [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else [],
            "python_random": random.getstate(),
        },
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--renderer-backward-impl", choices=("hybrid", "reference", "warp_tape"))
    parser.add_argument("--target-global-step", type=int, default=None)
    parser.add_argument("--enable-progress-log", action="store_true")
    parser.add_argument("--enable-density-log", action="store_true")
    parser.add_argument("--enable-progress-observer", action="store_true")
    parser.add_argument("--enable-density-observer", action="store_true")
    parser.add_argument("--verbose-progress", action="store_true")
    args = parser.parse_args()

    payload = load_debug_checkpoint(args.checkpoint)
    config_payload = payload.get("config")
    if not isinstance(config_payload, dict):
        raise SystemExit("Checkpoint is missing an embedded config")
    cfg = _load_config(config_payload)

    set_torch_compile_enabled(False)
    device = torch.device(args.device)
    images = load_images(collect_image_paths(args.input_dir.resolve()), device)
    pipeline = PoseFreeGaussianSR.from_images(images, intrinsics=None, config=cfg).to(device)
    restore_module_from_debug_checkpoint(pipeline, args.checkpoint)
    _restore_rng_state(payload)

    if args.renderer_backward_impl is not None:
        original_renderer_config = pipeline._renderer_config

        def _patched_renderer_config():
            render_cfg = original_renderer_config()
            render_cfg.backward_impl = args.renderer_backward_impl
            return render_cfg

        pipeline._renderer_config = _patched_renderer_config

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    def _on_density_event(event: dict) -> None:
        _save_checkpoint(
            output_dir=output_dir,
            pipeline=pipeline,
            cfg=cfg,
            reason="density",
            stage_index=int(event.get("stage_index", -1)),
            step_index=int(event.get("step_index", -1)),
            global_step=int(event.get("global_step", -1)),
            extra={"density_event": event},
        )

    resume_stage_index = int(payload.get("stage_index", -1))
    resume_step_index = int(payload.get("step_index", -1))
    resume_global_step = int(payload.get("global_step", -1))
    if resume_global_step < 0 and resume_stage_index > 0:
        resume_global_step = int(sum(int(s) for s in cfg.train.steps_per_stage[:resume_stage_index])) - 1
    if args.target_global_step is not None:
        target_global_step = int(args.target_global_step)
        if target_global_step < resume_global_step:
            raise SystemExit(
                f"--target-global-step ({target_global_step}) must be >= resume global step ({resume_global_step})"
            )
        steps_per_stage = list(int(s) for s in cfg.train.steps_per_stage)
        steps_per_stage[resume_stage_index] = max(
            resume_step_index + 1,
            (target_global_step - sum(steps_per_stage[:resume_stage_index])) + 1,
        )
        for stage_offset in range(resume_stage_index + 1, len(steps_per_stage)):
            steps_per_stage[stage_offset] = 0
        cfg.train.steps_per_stage = tuple(steps_per_stage)

    optimizer_state_dict = payload.get("optimizer_state_dict")
    if not isinstance(optimizer_state_dict, dict):
        optimizer_state_dict = None

    history_path = output_dir / "history.json"
    progress_log_path = output_dir / "progress.jsonl"
    density_event_log_path = output_dir / "density_events.jsonl"
    for log_path in (progress_log_path, density_event_log_path):
        if log_path.exists():
            log_path.unlink()
    debug_observer = None
    if args.enable_progress_observer or args.enable_density_observer:
        debug_observer = _DebugRunObserver(
            pipeline=pipeline,
            cfg=cfg,
            output_dir=output_dir,
            progress_log_path=progress_log_path,
            preview_every=0,
            diagnostic_meta_every=0,
            stage2_checkpoint_every=0,
            stage2_checkpoint_start_global_step=0,
            collapse_action="off",
            collapse_density_threshold=0.0,
            collapse_view_threshold=0,
            collapse_view_persistence=1,
        )
    try:
        fit_kwargs: dict[str, object] = {
            "verbose_progress": bool(args.verbose_progress),
            "synchronize_progress_timing": True,
            "optimizer_state_dict": optimizer_state_dict,
            "resume_stage_index": resume_stage_index,
            "resume_step_index": resume_step_index,
            "resume_global_step": resume_global_step,
        }
        if args.enable_progress_log:
            fit_kwargs["progress_log_path"] = progress_log_path
        if args.enable_density_log:
            fit_kwargs["density_event_log_path"] = density_event_log_path
        if args.enable_progress_observer and debug_observer is not None:
            fit_kwargs["progress_event_callback"] = debug_observer.on_progress
        if args.enable_density_observer and debug_observer is not None:
            fit_kwargs["density_event_callback"] = debug_observer.on_density_event
        history = pipeline.fit(images, **fit_kwargs)
    except NonFiniteTrainingError as exc:
        failure_path = _save_checkpoint(
            output_dir=output_dir,
            pipeline=pipeline,
            cfg=cfg,
            reason="nonfinite",
            stage_index=int(exc.payload.get("stage_index", -1)),
            step_index=int(exc.payload.get("step_index", -1)),
            global_step=int(exc.payload.get("global_step", -1)),
            extra={"nonfinite": exc.payload},
        )
        print(
            json.dumps(
                {
                    "status": "nonfinite",
                    "checkpoint": str(failure_path),
                    "payload": exc.payload,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1

    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    final_stage_index = len(cfg.train.steps_per_stage) - 1
    final_step_index = int(cfg.train.steps_per_stage[-1]) - 1
    total_steps = sum(int(s) for s in cfg.train.steps_per_stage)
    final_global_step = total_steps - 1
    final_path = _save_checkpoint(
        output_dir=output_dir,
        pipeline=pipeline,
        cfg=cfg,
        reason="final",
        stage_index=final_stage_index,
        step_index=final_step_index,
        global_step=final_global_step,
        extra={"history_path": str(history_path)},
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "history_path": str(history_path),
                "final_checkpoint": str(final_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
