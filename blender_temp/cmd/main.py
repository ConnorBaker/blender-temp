import json
import logging
import random
import sys
import time
from argparse import ArgumentParser, Namespace
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import torch
from torchvision.io import decode_image
from torchvision.transforms import v2
from torchvision.utils import save_image

from blender_temp.gaussian_sr.debug_checkpoint import load_debug_checkpoint, restore_module_from_debug_checkpoint
from blender_temp.gaussian_sr.pipeline import NonFiniteTrainingError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", stream=sys.stderr)
LOGGER = logging.getLogger(__name__)

IMAGE_EXTENSIONS: set[str] = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}


@dataclass(frozen=True)
class RunSafetyIssue:
    severity: Literal["warning", "error"]
    code: str
    message: str


@dataclass
class _StageProfilerSession:
    stage_index: int
    steps: int
    trace_dir: Path
    summary_path: Path
    schedule_kwargs: dict[str, int]
    profiler: object


class _PerStagePyTorchProfiler:
    def __init__(self, *, output_dir: Path, device: torch.device) -> None:
        self.output_dir = output_dir
        self.trace_root = output_dir / "pytorch-profile"
        self.trace_root.mkdir(parents=True, exist_ok=True)
        self.summary_path = output_dir / "pytorch-profile-summary.txt"
        self.sort_by = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
        self.activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            self.activities.append(torch.profiler.ProfilerActivity.CUDA)
        self.current: _StageProfilerSession | None = None
        self.records: list[dict[str, object]] = []

    def _start_stage(self, *, stage_index: int, steps: int) -> None:
        self._stop_stage()
        stage_dir = self.trace_root / f"stage{stage_index + 1}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        schedule_kwargs = pytorch_profiler_schedule_kwargs(steps)
        profiler = torch.profiler.profile(
            activities=self.activities,
            schedule=torch.profiler.schedule(**schedule_kwargs),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            acc_events=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(stage_dir)),
        )
        profiler.start()
        self.current = _StageProfilerSession(
            stage_index=stage_index,
            steps=steps,
            trace_dir=stage_dir,
            summary_path=self.trace_root / f"stage{stage_index + 1}-summary.txt",
            schedule_kwargs=schedule_kwargs,
            profiler=profiler,
        )
        LOGGER.info(
            "PyTorch profiler stage %d enabled: wait=%d warmup=%d active=%d trace_dir=%s",
            stage_index + 1,
            schedule_kwargs["wait"],
            schedule_kwargs["warmup"],
            schedule_kwargs["active"],
            stage_dir,
        )

    def _stop_stage(self) -> None:
        if self.current is None:
            return
        session = self.current
        profiler = session.profiler
        table = "Profiler summary unavailable."
        try:
            profiler.stop()
            table = profiler.key_averages().table(sort_by=self.sort_by, row_limit=50)
        except AssertionError as exc:
            table = f"Profiler summary unavailable: {exc}"
        session.summary_path.write_text(table, encoding="utf-8")
        self.records.append({
            "stage_index": session.stage_index,
            "steps": session.steps,
            "trace_dir": session.trace_dir,
            "summary_path": session.summary_path,
            "schedule": session.schedule_kwargs,
            "table": table,
        })
        self.current = None

    def on_progress(self, event: dict) -> None:
        event_name = event.get("event")
        if event_name == "stage_start":
            self._start_stage(
                stage_index=int(event["stage_index"]),
                steps=int(event["steps"]),
            )
        elif event_name == "stage_end":
            self._stop_stage()

    def on_step(self, *_args) -> None:
        if self.current is not None:
            self.current.profiler.step()

    def finish(self) -> None:
        self._stop_stage()
        parts: list[str] = []
        for record in self.records:
            schedule = record["schedule"]
            parts.append(
                "\n".join([
                    f"[stage {int(record['stage_index']) + 1}]",
                    f"steps={int(record['steps'])}",
                    (f"wait={int(schedule['wait'])} warmup={int(schedule['warmup'])} active={int(schedule['active'])}"),
                    f"trace_dir={record['trace_dir']}",
                    f"summary_path={record['summary_path']}",
                    "",
                    str(record["table"]),
                ])
            )
        self.summary_path.write_text("\n\n".join(parts), encoding="utf-8")


@dataclass
class _DebugRunObserver:
    pipeline: object
    cfg: object
    output_dir: Path
    progress_log_path: Path
    preview_every: int
    diagnostic_meta_every: int
    stage2_checkpoint_every: int
    stage2_checkpoint_start_global_step: int
    collapse_action: Literal["off", "warn", "abort"]
    collapse_density_threshold: float
    collapse_view_threshold: int
    collapse_view_persistence: int
    checkpoint_dir: Path = field(init=False)
    preview_dir: Path = field(init=False)
    last_safe_checkpoint_path: Path | None = None
    warned_tokens: set[str] = field(default_factory=set)
    view_low_visibility_streaks: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.preview_dir = self.output_dir / "debug-renders"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if self.preview_every > 0 or self.diagnostic_meta_every > 0:
            self.preview_dir.mkdir(parents=True, exist_ok=True)

    def _timestamp_utc(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def _append_debug_event(self, event: dict[str, object]) -> None:
        payload = {"timestamp_utc": self._timestamp_utc(), **event}
        with self.progress_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")

    def _cpu_clone(self, value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: self._cpu_clone(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._cpu_clone(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._cpu_clone(item) for item in value)
        return value

    def _meta_summary(self, meta: dict[str, torch.Tensor]) -> dict[str, int]:
        summary: dict[str, int] = {}
        for key in (
            "meta_gaussian_count",
            "meta_visible_count",
            "meta_intersection_count",
            "meta_tile_count",
            "meta_tiles_x",
            "meta_tiles_y",
            "meta_render_width",
            "meta_render_height",
        ):
            value = meta.get(key)
            if value is not None:
                summary[key.removeprefix("meta_")] = int(value.item())
        return summary

    def _checkpoint_path(self, reason: str, stage_index: int, step_index: int, global_step: int) -> Path:
        stage_tag = f"stage{stage_index + 1}"
        step_tag = "start" if step_index < 0 else f"step{step_index + 1:04d}"
        safe_global_step = max(int(global_step), 0)
        return self.checkpoint_dir / f"{stage_tag}_{step_tag}_{reason}_g{safe_global_step:05d}.pt"

    def save_checkpoint(
        self,
        *,
        reason: str,
        stage_index: int,
        step_index: int,
        global_step: int,
        extra: dict[str, object] | None = None,
        mark_safe: bool = False,
    ) -> Path:
        path = self._checkpoint_path(reason, stage_index, step_index, global_step)
        state_dict = {name: tensor.detach().cpu() for name, tensor in self.pipeline.state_dict().items()}
        payload = {
            "reason": reason,
            "timestamp_utc": self._timestamp_utc(),
            "stage_index": int(stage_index),
            "step_index": int(step_index),
            "global_step": int(global_step),
            "num_gaussians": int(self.pipeline.field_model.num_gaussians),
            "config": asdict(self.cfg),
            "pipeline_state_dict": state_dict,
        }
        optimizer = getattr(self.pipeline, "_debug_optimizer", None)
        if optimizer is not None:
            payload["optimizer_state_dict"] = self._cpu_clone(optimizer.state_dict())
        payload["rng_state"] = {
            "torch": torch.get_rng_state().cpu().clone(),
            "cuda": [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else [],
            "python_random": random.getstate(),
        }
        if extra:
            payload["extra"] = extra
        torch.save(payload, path)
        LOGGER.info("Saved debug checkpoint: %s", path)
        self._append_debug_event({
            "event": "debug_checkpoint",
            "reason": reason,
            "stage_index": int(stage_index),
            "step_index": int(step_index),
            "global_step": int(global_step),
            "path": str(path),
            "num_gaussians": int(self.pipeline.field_model.num_gaussians),
        })
        if mark_safe:
            self.last_safe_checkpoint_path = path
        return path

    def _pipeline_device(self) -> torch.device:
        try:
            return next(self.pipeline.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _restore_checkpoint_state(self, path: Path) -> None:
        _payload, skipped = restore_module_from_debug_checkpoint(self.pipeline, path)
        if skipped:
            LOGGER.info("Skipped %d checkpoint entries during rollback restore", len(skipped))

    def _is_safe_density_event(self, event: dict[str, object]) -> bool:
        summary = event.get("summary", {})
        if not isinstance(summary, dict):
            return True
        visibility_mean = float(summary.get("visibility_mean", float("inf")))
        if visibility_mean < self.collapse_density_threshold:
            return False
        weak_view_indices = summary.get("weak_view_indices", [])
        if isinstance(weak_view_indices, list) and weak_view_indices:
            return False
        view_coverages = summary.get("view_coverages", [])
        if not isinstance(view_coverages, list):
            return True
        min_visible = int(self.cfg.density.min_view_visible_gaussians)
        min_intersections = int(self.cfg.density.min_view_intersection_count)
        for coverage in view_coverages:
            if not isinstance(coverage, dict):
                continue
            if int(coverage.get("visible_count", 0)) < min_visible:
                return False
            if int(coverage.get("intersection_count", 0)) < min_intersections:
                return False
        return True

    def save_previews(
        self,
        *,
        reason: str,
        stage_index: int,
        step_index: int,
        global_step: int,
        out_h: int,
        out_w: int,
    ) -> Path:
        stage_tag = f"stage{stage_index + 1}"
        step_tag = f"step{step_index + 1:04d}"
        preview_path = self.preview_dir / f"{stage_tag}_{step_tag}_{reason}_g{global_step:05d}"
        preview_path.mkdir(parents=True, exist_ok=True)
        manifest: list[dict[str, object]] = []
        was_training = bool(self.pipeline.training)
        self.pipeline.eval()
        try:
            with torch.no_grad():
                for view_index in range(int(self.pipeline.num_views)):
                    render = self.pipeline.render_view(
                        view_index=view_index,
                        out_size=(out_h, out_w),
                        return_aux=True,
                        stats_mode="meta",
                    )
                    rgb = render["rgb"].detach().clone().cpu().clamp(0.0, 1.0)
                    image_path = preview_path / f"view{view_index}.png"
                    save_image(rgb, image_path)
                    stats = render.get("render_stats")
                    summary = self._meta_summary(stats) if isinstance(stats, dict) else {}
                    manifest.append({"view_index": int(view_index), "path": str(image_path), **summary})
        finally:
            if was_training:
                self.pipeline.train()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        summary_path = preview_path / "summary.json"
        summary_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        LOGGER.info("Saved debug previews: %s", preview_path)
        self._append_debug_event({
            "event": "debug_preview",
            "reason": reason,
            "stage_index": int(stage_index),
            "step_index": int(step_index),
            "global_step": int(global_step),
            "path": str(preview_path),
            "render_height": int(out_h),
            "render_width": int(out_w),
        })
        return preview_path

    def emit_diagnostic_visibility(
        self,
        *,
        stage_index: int,
        step_index: int,
        global_step: int,
        out_h: int,
        out_w: int,
    ) -> None:
        was_training = bool(self.pipeline.training)
        self.pipeline.eval()
        try:
            with torch.no_grad():
                for view_index in range(int(self.pipeline.num_views)):
                    meta = self.pipeline.render_view_meta(view_index=view_index, out_size=(out_h, out_w))
                    summary = self._meta_summary(meta)
                    event = {
                        "event": "diagnostic_view",
                        "stage_index": int(stage_index),
                        "step_index": int(step_index),
                        "global_step": int(global_step),
                        "view_index": int(view_index),
                        **summary,
                    }
                    self._append_debug_event(event)
                    self._observe_view_visibility(
                        view_index=int(view_index),
                        visible_count=int(summary.get("visible_count", 0)),
                        intersection_count=int(summary.get("intersection_count", 0)),
                        stage_index=stage_index,
                        step_index=step_index,
                        global_step=global_step,
                        source="diagnostic_view",
                        render_size=(out_h, out_w),
                    )
        finally:
            if was_training:
                self.pipeline.train()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _signal_collapse(
        self,
        *,
        token: str,
        message: str,
        stage_index: int,
        step_index: int,
        global_step: int,
        render_size: tuple[int, int] | None = None,
    ) -> None:
        if token in self.warned_tokens:
            return
        self.warned_tokens.add(token)
        extra = {"message": message}
        self.save_checkpoint(
            reason="collapse",
            stage_index=stage_index,
            step_index=step_index,
            global_step=global_step,
            extra=extra,
        )
        if render_size is not None:
            self.save_previews(
                reason="collapse",
                stage_index=stage_index,
                step_index=step_index,
                global_step=global_step,
                out_h=int(render_size[0]),
                out_w=int(render_size[1]),
            )
        self._append_debug_event({
            "event": "collapse_signal",
            "stage_index": int(stage_index),
            "step_index": int(step_index),
            "global_step": int(global_step),
            "message": message,
        })
        if self.collapse_action == "warn":
            LOGGER.warning("%s", message)
        elif self.collapse_action == "abort":
            rollback_preview_path: Path | None = None
            rollback_path = self.last_safe_checkpoint_path
            if rollback_path is not None:
                self._restore_checkpoint_state(rollback_path)
                if render_size is not None:
                    rollback_preview_path = self.save_previews(
                        reason="rollback",
                        stage_index=stage_index,
                        step_index=step_index,
                        global_step=global_step,
                        out_h=int(render_size[0]),
                        out_w=int(render_size[1]),
                    )
                self._append_debug_event({
                    "event": "rollback_restore",
                    "stage_index": int(stage_index),
                    "step_index": int(step_index),
                    "global_step": int(global_step),
                    "checkpoint_path": str(rollback_path),
                    "preview_path": None if rollback_preview_path is None else str(rollback_preview_path),
                })
                raise RuntimeError(
                    f"{message} Restored last safe checkpoint {rollback_path}."
                    + ("" if rollback_preview_path is None else f" Rollback previews: {rollback_preview_path}.")
                )
            raise RuntimeError(message)

    def _observe_view_visibility(
        self,
        *,
        view_index: int,
        visible_count: int,
        intersection_count: int,
        stage_index: int,
        step_index: int,
        global_step: int,
        source: str,
        render_size: tuple[int, int],
    ) -> None:
        if self.collapse_action == "off":
            return
        low_visibility = visible_count <= self.collapse_view_threshold or intersection_count <= 0
        next_streak = self.view_low_visibility_streaks.get(view_index, 0) + 1 if low_visibility else 0
        self.view_low_visibility_streaks[view_index] = next_streak
        if next_streak < self.collapse_view_persistence:
            return
        self._signal_collapse(
            token=f"{source}:view{view_index}",
            message=(
                f"Visibility collapse detected for view {view_index} via {source}: "
                f"visible_count={visible_count}, intersection_count={intersection_count}, "
                f"streak={next_streak}."
            ),
            stage_index=stage_index,
            step_index=step_index,
            global_step=global_step,
            render_size=render_size,
        )

    def on_progress(self, event: dict[str, object]) -> None:
        event_name = str(event.get("event", ""))
        stage_index = int(event.get("stage_index", -1))
        step_index = int(event.get("step_index", -1))
        global_step = int(event.get("global_step", step_index))
        if event_name == "stage_start":
            self.save_checkpoint(
                reason="stage_start",
                stage_index=stage_index,
                step_index=-1,
                global_step=global_step,
                mark_safe=True,
            )
            return
        if event_name == "view_end":
            render_h = int(event.get("render_height", 0))
            render_w = int(event.get("render_width", 0))
            self._observe_view_visibility(
                view_index=int(event["view_index"]),
                visible_count=int(event.get("visible_count", 0)),
                intersection_count=int(event.get("intersection_count", 0)),
                stage_index=stage_index,
                step_index=step_index,
                global_step=global_step,
                source="view_end",
                render_size=(render_h, render_w),
            )
            return
        if event_name != "step_end" or stage_index < 0:
            return
        render_h = int(event.get("render_height", 0))
        render_w = int(event.get("render_width", 0))
        if self.stage2_checkpoint_every > 0 and stage_index == 1:
            if (
                global_step >= self.stage2_checkpoint_start_global_step
                and (step_index + 1) % self.stage2_checkpoint_every == 0
            ):
                self.save_checkpoint(
                    reason="periodic",
                    stage_index=stage_index,
                    step_index=step_index,
                    global_step=global_step,
                )
        if self.preview_every > 0 and stage_index == len(self.cfg.train.stage_scales) - 1:
            if (step_index + 1) % self.preview_every == 0:
                self.save_previews(
                    reason="periodic",
                    stage_index=stage_index,
                    step_index=step_index,
                    global_step=global_step,
                    out_h=render_h,
                    out_w=render_w,
                )
        if self.diagnostic_meta_every > 0 and stage_index == len(self.cfg.train.stage_scales) - 1:
            if (step_index + 1) % self.diagnostic_meta_every == 0:
                self.emit_diagnostic_visibility(
                    stage_index=stage_index,
                    step_index=step_index,
                    global_step=global_step,
                    out_h=render_h,
                    out_w=render_w,
                )

    def on_density_event(self, event: dict[str, object]) -> None:
        stage_index = int(event.get("stage_index", -1))
        step_index = int(event.get("step_index", -1))
        global_step = int(event.get("global_step", step_index))
        self.save_checkpoint(
            reason="density",
            stage_index=stage_index,
            step_index=step_index,
            global_step=global_step,
            extra={"density_event": event},
            mark_safe=self._is_safe_density_event(event),
        )
        if self.collapse_action == "off":
            return
        summary = event.get("summary", {})
        if not isinstance(summary, dict):
            return
        visibility_mean = float(summary.get("visibility_mean", float("nan")))
        if not (visibility_mean < self.collapse_density_threshold):
            return
        if int(event.get("reseeded", 0)) > 0:
            return
        render_h = int(self.pipeline.train_height)
        render_w = int(self.pipeline.train_width)
        self._signal_collapse(
            token=f"density:{stage_index}:{step_index}",
            message=(
                f"Density visibility collapsed at stage {stage_index + 1} step {step_index + 1}: "
                f"visibility_mean={visibility_mean:.6f} < {self.collapse_density_threshold:.6f}."
            ),
            stage_index=stage_index,
            step_index=step_index,
            global_step=global_step,
            render_size=(render_h, render_w),
        )


def setup_argparse() -> ArgumentParser:
    parser = ArgumentParser(description="Joint multi-frame super-resolution.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing input images to super-resolve",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to write super-resolved output",
        default=Path("output"),
    )
    parser.add_argument(
        "--resume-debug-checkpoint",
        type=Path,
        help="Resume training from a saved debug checkpoint instead of initializing a fresh scene",
        default=None,
    )
    parser.add_argument(
        "--scale",
        type=float,
        help="Output scale factor; accepts any positive float",
        default=2.0,
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Torch device to use",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Maximum number of input frames to load",
        default=None,
    )
    parser.add_argument(
        "--anchor-stride",
        type=int,
        help="Stride for the anchor lattice used to initialize Gaussians",
        default=1,
    )
    parser.add_argument(
        "--view-batch-size",
        type=int,
        help="Number of views to render per optimizer step; use 0 to render all views",
        default=0,
    )
    parser.add_argument(
        "--radius-clip-px",
        type=float,
        help="Cull Gaussians whose projected screen-space radius exceeds this many pixels; 0 disables clipping",
        default=0.0,
    )
    parser.add_argument(
        "--disable-density-control-final-stage",
        action="store_true",
        help="Disable density control during the final training stage",
    )
    parser.add_argument(
        "--verbose-progress",
        action="store_true",
        help="Print per-view and per-step progress with timings and memory usage",
    )
    parser.add_argument(
        "--profile-pytorch",
        action="store_true",
        help="Capture separate short PyTorch profiler traces for each training stage",
    )
    parser.add_argument(
        "--disable-torch-compile",
        action="store_true",
        help="Disable torch.compile and run the training loop eagerly",
    )
    parser.add_argument(
        "--renderer-backward-impl",
        choices=("hybrid", "reference", "warp_tape"),
        help="Override the renderer backward implementation",
        default=None,
    )
    parser.add_argument(
        "--debug-preview-every",
        type=int,
        help="During the final stage, save preview renders for all views every N steps; defaults to 50 in verbose mode",
        default=None,
    )
    parser.add_argument(
        "--diagnostic-meta-every",
        type=int,
        help="When torch.compile is enabled, run a meta-only visibility diagnostic every N final-stage steps; defaults to 50",
        default=None,
    )
    parser.add_argument(
        "--debug-stage2-checkpoint-every",
        type=int,
        help="During stage 2, save a rolling debug checkpoint every N steps; 0 disables this",
        default=0,
    )
    parser.add_argument(
        "--debug-stage2-checkpoint-start-global-step",
        type=int,
        help="Only start the periodic stage-2 checkpoint cadence at or after this global step",
        default=0,
    )
    parser.add_argument(
        "--collapse-action",
        choices=("off", "warn", "abort"),
        help="How to respond when visibility collapse is detected",
        default="abort",
    )
    parser.add_argument(
        "--collapse-density-threshold",
        type=float,
        help="Warn or abort when density visibility_mean falls below this threshold",
        default=5.0,
    )
    parser.add_argument(
        "--collapse-view-threshold",
        type=int,
        help="Treat a view as collapsed when visible_count is at or below this threshold",
        default=8,
    )
    parser.add_argument(
        "--collapse-view-persistence",
        type=int,
        help="Number of consecutive low-visibility samples before triggering the collapse action",
        default=3,
    )
    parser.add_argument(
        "--disable-runtime-observer",
        action="store_true",
        help="Disable debug observer callbacks and run logs to match the minimal stable training path",
    )
    parser.add_argument(
        "--allow-unsafe-config",
        action="store_true",
        help="Proceed even when preflight safety checks flag a high-risk configuration",
    )
    return parser


def collect_image_paths(input_dir: Path) -> list[Path]:
    """Collect and sort image file paths from a directory."""
    paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    return paths


def load_images(image_paths: list[Path], device: torch.device) -> torch.Tensor:
    """Load images from disk and stack into a single (N, C, H, W) float32 tensor."""
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    tensors: list[torch.Tensor] = []
    for path in image_paths:
        img = decode_image(str(path))
        img = transform(img)
        tensors.append(img)

    # Verify all images have the same spatial dimensions
    shapes = {t.shape for t in tensors}
    if len(shapes) > 1:
        LOGGER.error("Images have inconsistent shapes: %s", shapes)
        raise SystemExit(1)

    batch = torch.stack(tensors).to(device)
    return batch


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


def assess_run_safety(
    *,
    num_views: int,
    scale: float,
    anchor_stride: int,
    view_batch_size: int,
    radius_clip_px: float,
    disable_density_control_final_stage: bool,
) -> list[RunSafetyIssue]:
    issues: list[RunSafetyIssue] = []
    effective_view_batch = num_views if view_batch_size <= 0 else min(view_batch_size, num_views)

    if num_views > 1 and effective_view_batch < num_views:
        issues.append(
            RunSafetyIssue(
                severity="warning",
                code="partial_view_batch",
                message=(
                    "Training will optimize only a subset of views per step; held-out views can lose visibility "
                    "before they are sampled again."
                ),
            )
        )
    if num_views > 1 and effective_view_batch == 1:
        issues.append(
            RunSafetyIssue(
                severity="warning",
                code="single_view_steps",
                message=(
                    "`view_batch_size=1` with multiple views is brittle for pose-free optimization and can collapse "
                    "coverage for non-anchor views."
                ),
            )
        )
    if radius_clip_px > 0.0 and scale > 1:
        issues.append(
            RunSafetyIssue(
                severity="warning",
                code="scaled_radius_clip",
                message=(
                    "`radius_clip_px` is applied in output pixels; exporting above training resolution makes the "
                    "clip stricter and can cull views that still rendered at 1x."
                ),
            )
        )
    if disable_density_control_final_stage and num_views > 1 and effective_view_batch < num_views:
        issues.append(
            RunSafetyIssue(
                severity="warning",
                code="no_final_density_recovery",
                message=(
                    "Disabling final-stage density control removes the last densification pass that could recover "
                    "coverage for held-out views."
                ),
            )
        )
    if anchor_stride <= 1 and num_views > 1 and effective_view_batch < num_views:
        issues.append(
            RunSafetyIssue(
                severity="warning",
                code="dense_anchor_lattice",
                message=(
                    "A very dense anchor lattice with partial-view training increases optimization pressure on a "
                    "single sampled view and raises instability risk."
                ),
            )
        )

    high_risk_black_output = (
        num_views >= 3
        and effective_view_batch == 1
        and radius_clip_px > 0.0
        and scale > 1
        and disable_density_control_final_stage
    )
    if high_risk_black_output:
        issues.append(
            RunSafetyIssue(
                severity="error",
                code="high_risk_black_output",
                message=(
                    "This combination is high risk for black outputs: multi-view training with `view_batch_size=1`, "
                    "upscaled export, active `radius_clip_px`, and final-stage density control disabled."
                ),
            )
        )

    return issues


def format_scale_tag(scale: float) -> str:
    return format(scale, "g")


def reset_run_logs(output_dir: Path) -> tuple[Path, Path]:
    progress_log_path = output_dir / "progress.jsonl"
    density_event_log_path = output_dir / "density_events.jsonl"
    for path in (progress_log_path, density_event_log_path):
        if path.exists():
            path.unlink()
            LOGGER.info("Removed stale run log %s", path)
    return progress_log_path, density_event_log_path


def pytorch_profiler_schedule_kwargs(total_steps: int) -> dict[str, int]:
    total = max(int(total_steps), 1)
    wait = 1 if total >= 5 else 0
    remaining = max(total - wait, 1)
    warmup = 1 if remaining >= 3 else 0
    active = max(min(total - wait - warmup, 3), 1)
    return {
        "wait": wait,
        "warmup": warmup,
        "active": active,
        "repeat": 1,
    }


def configure_tf32_backends() -> None:
    # `torch.backends.fp32_precision` trips torch.compile in Torch 2.10 because
    # Inductor still reads the legacy cuBLAS TF32 flag on some paths.
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True


def configure_torch_compile_runtime() -> None:
    compiler_config = getattr(getattr(torch, "compiler", None), "config", None)
    if compiler_config is not None:
        compiler_config.recompile_limit = 32


def fit_with_optional_pytorch_profiler(
    pipeline,
    images: torch.Tensor,
    *,
    output_dir: Path,
    profile_pytorch: bool,
    total_steps: int,
    fit_kwargs: dict[str, object],
) -> dict:
    if not profile_pytorch:
        return pipeline.fit(images, **fit_kwargs)

    profiler = _PerStagePyTorchProfiler(output_dir=output_dir, device=images.device)
    existing_progress_callback = fit_kwargs.get("progress_event_callback")
    existing_step_callback = fit_kwargs.get("step_callback")

    def _progress_callback(event: dict) -> None:
        profiler.on_progress(event)
        if callable(existing_progress_callback):
            existing_progress_callback(event)

    def _step_callback(stage_idx: int, step_idx: int, global_step: int) -> None:
        profiler.on_step(stage_idx, step_idx, global_step)
        if callable(existing_step_callback):
            existing_step_callback(stage_idx, step_idx, global_step)

    fit_kwargs = dict(fit_kwargs)
    fit_kwargs["progress_event_callback"] = _progress_callback
    fit_kwargs["step_callback"] = _step_callback
    try:
        history = pipeline.fit(images, **fit_kwargs)
    finally:
        profiler.finish()
        LOGGER.info("Wrote per-stage PyTorch profiler summaries to %s", profiler.summary_path)
    return history


def main() -> None:
    parser = setup_argparse()
    args: Namespace = parser.parse_args()

    input_dir: Path = args.input_dir.resolve()
    output_dir: Path = args.output_dir.resolve()
    resume_debug_checkpoint: Path | None = (
        args.resume_debug_checkpoint.resolve() if args.resume_debug_checkpoint is not None else None
    )
    scale: float = float(args.scale)
    device = torch.device(args.device)
    max_frames: int | None = args.max_frames
    anchor_stride: int = args.anchor_stride
    view_batch_size: int = args.view_batch_size
    radius_clip_px: float = float(args.radius_clip_px)
    disable_density_control_final_stage: bool = bool(args.disable_density_control_final_stage)
    verbose_progress: bool = bool(args.verbose_progress)
    profile_pytorch: bool = bool(args.profile_pytorch)
    disable_torch_compile: bool = bool(args.disable_torch_compile)
    renderer_backward_impl: str | None = args.renderer_backward_impl
    debug_preview_every_arg: int | None = args.debug_preview_every
    diagnostic_meta_every_arg: int | None = args.diagnostic_meta_every
    debug_stage2_checkpoint_every: int = int(args.debug_stage2_checkpoint_every)
    debug_stage2_checkpoint_start_global_step: int = int(args.debug_stage2_checkpoint_start_global_step)
    collapse_action: Literal["off", "warn", "abort"] = args.collapse_action
    collapse_density_threshold: float = float(args.collapse_density_threshold)
    collapse_view_threshold: int = int(args.collapse_view_threshold)
    collapse_view_persistence: int = int(args.collapse_view_persistence)
    disable_runtime_observer: bool = bool(args.disable_runtime_observer)
    allow_unsafe_config: bool = bool(args.allow_unsafe_config)

    if not input_dir.is_dir():
        LOGGER.error("Input directory not found: %s", input_dir)
        raise SystemExit(1)
    if resume_debug_checkpoint is not None and not resume_debug_checkpoint.is_file():
        LOGGER.error("Resume checkpoint not found: %s", resume_debug_checkpoint)
        raise SystemExit(1)
    if max_frames is not None and max_frames <= 0:
        LOGGER.error("--max-frames must be a positive integer")
        raise SystemExit(1)
    if anchor_stride <= 0:
        LOGGER.error("--anchor-stride must be a positive integer")
        raise SystemExit(1)
    if view_batch_size < 0:
        LOGGER.error("--view-batch-size must be non-negative; use 0 to render all views")
        raise SystemExit(1)
    if radius_clip_px < 0.0:
        LOGGER.error("--radius-clip-px must be non-negative; use 0 to disable clipping")
        raise SystemExit(1)
    if scale <= 0.0:
        LOGGER.error("--scale must be a positive float")
        raise SystemExit(1)
    if debug_preview_every_arg is not None and debug_preview_every_arg < 0:
        LOGGER.error("--debug-preview-every must be non-negative")
        raise SystemExit(1)
    if diagnostic_meta_every_arg is not None and diagnostic_meta_every_arg < 0:
        LOGGER.error("--diagnostic-meta-every must be non-negative")
        raise SystemExit(1)
    if debug_stage2_checkpoint_every < 0:
        LOGGER.error("--debug-stage2-checkpoint-every must be non-negative")
        raise SystemExit(1)
    if debug_stage2_checkpoint_start_global_step < 0:
        LOGGER.error("--debug-stage2-checkpoint-start-global-step must be non-negative")
        raise SystemExit(1)
    if collapse_density_threshold < 0.0:
        LOGGER.error("--collapse-density-threshold must be non-negative")
        raise SystemExit(1)
    if collapse_view_threshold < 0:
        LOGGER.error("--collapse-view-threshold must be non-negative")
        raise SystemExit(1)
    if collapse_view_persistence <= 0:
        LOGGER.error("--collapse-view-persistence must be positive")
        raise SystemExit(1)

    configure_tf32_backends()
    if not disable_torch_compile:
        configure_torch_compile_runtime()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect images
    image_paths = collect_image_paths(input_dir)
    if not image_paths:
        LOGGER.error("No images found in %s", input_dir)
        raise SystemExit(1)
    if max_frames is not None:
        original_count = len(image_paths)
        image_paths = image_paths[:max_frames]
        LOGGER.info("Limiting input frames to %d of %d", len(image_paths), original_count)

    resume_payload: dict[str, object] | None = None
    if resume_debug_checkpoint is not None:
        resume_payload = load_debug_checkpoint(resume_debug_checkpoint)
        LOGGER.info("Resuming from debug checkpoint: %s", resume_debug_checkpoint)

    safety_anchor_stride = anchor_stride
    safety_view_batch_size = view_batch_size
    safety_radius_clip_px = radius_clip_px
    safety_disable_density_control_final_stage = disable_density_control_final_stage
    if resume_payload is not None:
        config_payload = resume_payload.get("config")
        if not isinstance(config_payload, dict):
            LOGGER.error("Resume checkpoint is missing an embedded config: %s", resume_debug_checkpoint)
            raise SystemExit(1)
        try:
            safety_anchor_stride = int(config_payload["field"]["anchor_stride"])
            safety_view_batch_size = int(config_payload["train"]["view_batch_size"])
            safety_radius_clip_px = float(config_payload["render"]["radius_clip_px"])
            safety_disable_density_control_final_stage = bool(config_payload["density"]["disable_final_stage"])
        except (KeyError, TypeError, ValueError) as exc:
            LOGGER.error("Resume checkpoint has an invalid embedded config: %s", exc)
            raise SystemExit(1)

    safety_issues = assess_run_safety(
        num_views=len(image_paths),
        scale=scale,
        anchor_stride=safety_anchor_stride,
        view_batch_size=safety_view_batch_size,
        radius_clip_px=safety_radius_clip_px,
        disable_density_control_final_stage=safety_disable_density_control_final_stage,
    )
    for issue in safety_issues:
        if issue.severity == "warning":
            LOGGER.warning("[%s] %s", issue.code, issue.message)
    safety_errors = [issue for issue in safety_issues if issue.severity == "error"]
    if safety_errors and not allow_unsafe_config:
        for issue in safety_errors:
            LOGGER.error("[%s] %s", issue.code, issue.message)
        LOGGER.error("Refusing to run with a high-risk configuration. Re-run with --allow-unsafe-config to override.")
        raise SystemExit(1)
    if safety_errors and allow_unsafe_config:
        LOGGER.warning(
            "Proceeding despite %d high-risk configuration issue(s) because --allow-unsafe-config was set.",
            len(safety_errors),
        )

    LOGGER.info("Found %d images in %s", len(image_paths), input_dir)
    LOGGER.info("Device: %s", device)
    LOGGER.info("Scale factor: %sx", format_scale_tag(scale))
    LOGGER.info("Anchor stride: %d", anchor_stride)
    LOGGER.info("View batch size: %s", "all" if view_batch_size == 0 else view_batch_size)
    LOGGER.info("Radius clip px: %s", radius_clip_px if radius_clip_px > 0.0 else "disabled")
    LOGGER.info("Torch compile: %s", "disabled" if disable_torch_compile else "enabled")
    LOGGER.info("Renderer backward: %s", renderer_backward_impl or "config default")
    LOGGER.info(
        "Density control final stage: %s",
        "disabled" if disable_density_control_final_stage else "enabled",
    )
    if profile_pytorch and verbose_progress:
        LOGGER.warning("--profile-pytorch disables --verbose-progress to avoid sync-heavy traces.")
        verbose_progress = False

    debug_preview_every = (
        50 if debug_preview_every_arg is None and verbose_progress else int(debug_preview_every_arg or 0)
    )
    diagnostic_meta_every = (
        0
        if disable_torch_compile or verbose_progress
        else (50 if diagnostic_meta_every_arg is None else int(diagnostic_meta_every_arg))
    )
    LOGGER.info(
        "Debug preview cadence: %s",
        "disabled" if debug_preview_every <= 0 else f"every {debug_preview_every} final-stage steps",
    )
    LOGGER.info(
        "Diagnostic visibility cadence: %s",
        "disabled" if diagnostic_meta_every <= 0 else f"every {diagnostic_meta_every} final-stage steps",
    )
    LOGGER.info(
        "Stage-2 debug checkpoint cadence: %s",
        (
            "disabled"
            if debug_stage2_checkpoint_every <= 0
            else f"every {debug_stage2_checkpoint_every} stage-2 steps from global step {debug_stage2_checkpoint_start_global_step}"
        ),
    )
    LOGGER.info(
        "Collapse detector: action=%s density_threshold=%.3f view_threshold=%d persistence=%d",
        collapse_action,
        collapse_density_threshold,
        collapse_view_threshold,
        collapse_view_persistence,
    )

    # Load all images into a batch tensor
    LOGGER.info("Loading images...")
    batch = load_images(image_paths, device)
    LOGGER.info("Loaded batch: shape=%s, dtype=%s", batch.shape, batch.dtype)

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
    from blender_temp.gaussian_sr.pipeline import set_torch_compile_enabled

    images = batch
    set_torch_compile_enabled(not disable_torch_compile)

    if resume_payload is None:
        cfg = PoseFreeGaussianConfig()
        cfg.camera.learn_intrinsics = False
        cfg.train.stage_scales = (0.25, 0.5, 1.0)
        cfg.train.steps_per_stage = (250, 500, 1000)
        cfg.field.anchor_stride = anchor_stride
        cfg.field.feature_dim = 8
        cfg.train.view_batch_size = view_batch_size
        cfg.render.radius_clip_px = radius_clip_px
        cfg.density.disable_final_stage = disable_density_control_final_stage
    else:
        config_payload = resume_payload["config"]
        assert isinstance(config_payload, dict)
        cfg = PoseFreeGaussianConfig(
            camera=CameraInit(**config_payload["camera"]),
            render=RenderConfig(**config_payload["render"]),
            observation=ObservationConfig(**config_payload["observation"]),
            appearance=AppearanceConfig(**config_payload["appearance"]),
            density=DensityControlConfig(**config_payload["density"]),
            field=FieldConfig(**config_payload["field"]),
            train=TrainConfig(**config_payload["train"]),
        )
        LOGGER.info("Using the training config embedded in the resume checkpoint.")

    intrinsics = None
    anchor_h = (images.shape[-2] + cfg.field.anchor_stride - 1) // cfg.field.anchor_stride
    anchor_w = (images.shape[-1] + cfg.field.anchor_stride - 1) // cfg.field.anchor_stride
    estimated_gaussians = int(anchor_h * anchor_w)
    view_batch_desc = "all" if cfg.train.view_batch_size <= 0 else str(cfg.train.view_batch_size)
    LOGGER.info(
        "Init config: frames=%d resolution=%dx%d anchor_stride=%d est_gaussians=%d view_batch_size=%s",
        images.shape[0],
        images.shape[-2],
        images.shape[-1],
        cfg.field.anchor_stride,
        estimated_gaussians,
        view_batch_desc,
    )
    if estimated_gaussians > 250_000:
        LOGGER.warning(
            "Large initial Gaussian count (%d). Expect slow steps. Consider larger anchor_stride and/or fewer frames per step.",
            estimated_gaussians,
        )
    if cfg.train.view_batch_size <= 0 and images.shape[0] > 1:
        LOGGER.warning("All %d frames will be rendered in every training step.", images.shape[0])

    LOGGER.info("Initializing pose-free Gaussian scene...")
    pipeline = PoseFreeGaussianSR.from_images(images, intrinsics=intrinsics, config=cfg).to(device)
    if resume_debug_checkpoint is not None:
        _payload, skipped = restore_module_from_debug_checkpoint(pipeline, resume_debug_checkpoint)
        if skipped:
            LOGGER.info("Skipped %d checkpoint entries during resume restore", len(skipped))
        restored_rng_state = _restore_rng_state(resume_payload or {})
        LOGGER.info("Restored RNG state from checkpoint: %s", "yes" if restored_rng_state else "no")
    if renderer_backward_impl is not None:
        original_renderer_config = pipeline._renderer_config

        def _patched_renderer_config():
            render_cfg = original_renderer_config()
            render_cfg.backward_impl = renderer_backward_impl
            return render_cfg

        pipeline._renderer_config = _patched_renderer_config
    LOGGER.info("Training scene model...")
    observer_requested = (
        debug_preview_every > 0
        or diagnostic_meta_every > 0
        or debug_stage2_checkpoint_every > 0
        or collapse_action != "off"
    )
    progress_log_path: Path | None = None
    density_event_log_path: Path | None = None
    debug_observer: _DebugRunObserver | None = None
    if disable_runtime_observer:
        if observer_requested:
            LOGGER.warning(
                "Runtime observer disabled; ignoring collapse detection, debug previews, diagnostic visibility, and periodic debug checkpoints."
            )
        LOGGER.info("Runtime observer: disabled")
    else:
        progress_log_path, density_event_log_path = reset_run_logs(output_dir)
        LOGGER.info("Progress log: %s", progress_log_path)
        debug_observer = _DebugRunObserver(
            pipeline=pipeline,
            cfg=cfg,
            output_dir=output_dir,
            progress_log_path=progress_log_path,
            preview_every=debug_preview_every,
            diagnostic_meta_every=diagnostic_meta_every,
            stage2_checkpoint_every=debug_stage2_checkpoint_every,
            stage2_checkpoint_start_global_step=debug_stage2_checkpoint_start_global_step,
            collapse_action=collapse_action,
            collapse_density_threshold=collapse_density_threshold,
            collapse_view_threshold=collapse_view_threshold,
            collapse_view_persistence=collapse_view_persistence,
        )
        LOGGER.info("Debug checkpoints dir: %s", debug_observer.checkpoint_dir)
        if debug_preview_every > 0 or diagnostic_meta_every > 0:
            LOGGER.info("Debug render dir: %s", debug_observer.preview_dir)

    resume_stage_index = 0
    resume_step_index = -1
    resume_global_step = -1
    resume_optimizer_state_dict: dict[str, object] | None = None
    if resume_payload is not None:
        resume_stage_index = int(resume_payload.get("stage_index", -1))
        resume_step_index = int(resume_payload.get("step_index", -1))
        resume_global_step = int(resume_payload.get("global_step", -1))
        if resume_stage_index < 0:
            LOGGER.error("Resume checkpoint has an invalid stage_index: %s", resume_stage_index)
            raise SystemExit(1)
        if resume_global_step < 0 and resume_stage_index > 0:
            resume_global_step = int(sum(int(s) for s in cfg.train.steps_per_stage[:resume_stage_index])) - 1
        optimizer_state_dict = resume_payload.get("optimizer_state_dict")
        if isinstance(optimizer_state_dict, dict):
            resume_optimizer_state_dict = optimizer_state_dict
        LOGGER.info(
            "Continuing from stage=%d step=%d global_step=%d",
            resume_stage_index,
            resume_step_index,
            resume_global_step,
        )

    fit_kwargs: dict[str, object] = {
        "verbose_progress": verbose_progress,
        "synchronize_progress_timing": verbose_progress,
        "optimizer_state_dict": resume_optimizer_state_dict,
        "resume_stage_index": resume_stage_index,
        "resume_step_index": resume_step_index,
        "resume_global_step": resume_global_step,
    }
    if density_event_log_path is not None:
        fit_kwargs["density_event_log_path"] = density_event_log_path
    if progress_log_path is not None:
        fit_kwargs["progress_log_path"] = progress_log_path
    if debug_observer is not None:
        fit_kwargs["density_event_callback"] = debug_observer.on_density_event
        fit_kwargs["progress_event_callback"] = debug_observer.on_progress
    if resume_stage_index >= len(cfg.train.steps_per_stage):
        remaining_total_steps = 0
    else:
        remaining_in_stage = max(0, int(cfg.train.steps_per_stage[resume_stage_index]) - max(0, resume_step_index + 1))
        remaining_later = sum(int(s) for s in cfg.train.steps_per_stage[resume_stage_index + 1 :])
        remaining_total_steps = remaining_in_stage + remaining_later
    try:
        history = fit_with_optional_pytorch_profiler(
            pipeline,
            images,
            output_dir=output_dir,
            profile_pytorch=profile_pytorch,
            total_steps=remaining_total_steps,
            fit_kwargs=fit_kwargs,
        )
    except NonFiniteTrainingError as exc:
        payload = exc.payload
        if debug_observer is not None:
            debug_observer.save_checkpoint(
                reason="nonfinite",
                stage_index=int(payload.get("stage_index", -1)),
                step_index=int(payload.get("step_index", -1)),
                global_step=int(payload.get("global_step", -1)),
                extra={"nonfinite": payload},
            )
        raise

    history_path = output_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    LOGGER.info("Wrote training history to %s", history_path)

    render_scale = float(scale)
    scale_tag = format_scale_tag(scale)
    with torch.no_grad():
        anchor_render = (
            pipeline
            .render_view(view_index=0, scale=render_scale, return_aux=False)["rgb"]
            .detach()
            .clone()
            .cpu()
            .clamp(0.0, 1.0)
        )
    anchor_pt_path = output_dir / f"render_view0_{scale_tag}x.pt"
    anchor_png_path = output_dir / f"render_view0_{scale_tag}x.png"
    torch.save(anchor_render, anchor_pt_path)
    save_image(anchor_render, anchor_png_path)
    LOGGER.info("Wrote anchor render to %s and %s", anchor_pt_path, anchor_png_path)

    if images.shape[0] > 1:
        with torch.no_grad():
            secondary_render = (
                pipeline
                .render_view(view_index=1, scale=render_scale, return_aux=False)["rgb"]
                .detach()
                .clone()
                .cpu()
                .clamp(0.0, 1.0)
            )
        secondary_pt_path = output_dir / f"render_view1_{scale_tag}x.pt"
        secondary_png_path = output_dir / f"render_view1_{scale_tag}x.png"
        torch.save(secondary_render, secondary_pt_path)
        save_image(secondary_render, secondary_png_path)
        LOGGER.info("Wrote secondary render to %s and %s", secondary_pt_path, secondary_png_path)


if __name__ == "__main__":
    main()
