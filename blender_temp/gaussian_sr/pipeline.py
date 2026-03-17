import gc
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from .camera import LearnableCameraBundle, LearnableSharedIntrinsics
from .density_control import (
    DensityControlResult,
    DensityViewCoverage,
    DensityViewObservation,
    apply_density_control,
    should_run_density_control_for_stage,
)
from .density_logging import emit_density_event
from .field import CanonicalGaussianField, ScaleAwareResidualHead
from .image_utils import charbonnier, estimate_translation_bootstrap, ssim_value, tv_loss_grid
from .math_utils import default_intrinsics
from .observation_model import observation_render_size, render_observe_rgb
from .posefree_config import PoseFreeGaussianConfig, TrainConfig
from .progress_logging import emit_progress_event
from .warp_gsplat_contracts import RasterConfig
from .gsplat_renderer import (
    PreparedVisibility,
    clear_renderer_caches,
    reserve_renderer_intersection_capacity,
    render_projection_meta,
    render_visibility_meta,
    render_stats_prepared,
    render_stats,
    render_values,
)


_META_STAT_KEYS = (
    "meta_gaussian_count",
    "meta_visible_count",
    "meta_intersection_count",
    "meta_tile_count",
    "meta_tiles_x",
    "meta_tiles_y",
    "meta_render_width",
    "meta_render_height",
)
_DENSITY_STAT_KEYS = ("contrib", "transmittance", "hits", "residual", "error_map")
_TORCH_COMPILE_ENABLED = True


class NonFiniteTrainingError(FloatingPointError):
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        super().__init__(f"Non-finite state detected: {payload}")


@dataclass(frozen=True)
class ProjectionPreflightError(RuntimeError):
    payload: dict[str, object]

    def __post_init__(self) -> None:
        RuntimeError.__init__(self, str(self.payload.get("message", "Projection preflight failed.")))


def set_torch_compile_enabled(enabled: bool) -> None:
    global _TORCH_COMPILE_ENABLED
    _TORCH_COMPILE_ENABLED = bool(enabled)


def _make_compile_disabled(callable_obj):
    compiler = getattr(torch, "compiler", None)
    disable = getattr(compiler, "disable", None)
    if callable(disable):
        try:
            return disable(callable_obj)
        except Exception:
            return callable_obj
    dynamo = getattr(torch, "_dynamo", None)
    disable = getattr(dynamo, "disable", None)
    if callable(disable):
        try:
            return disable(callable_obj)
        except Exception:
            return callable_obj
    return callable_obj


def _clear_compiled_cuda_state() -> None:
    if not torch.cuda.is_available():
        return
    try:
        from torch._inductor.cudagraph_trees import reset_cudagraph_trees
    except Exception:
        reset_cudagraph_trees = None
    if callable(reset_cudagraph_trees):
        try:
            reset_cudagraph_trees()
        except Exception:
            pass
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def _make_optional_compiled(callable_obj, name: str):
    if not _TORCH_COMPILE_ENABLED or not hasattr(torch, "compile"):
        return callable_obj
    try:
        # These compiled regions intentionally graph-break around the Warp entrypoints
        # wrapped by torch.compiler.disable(). fullgraph=True would raise on those
        # breaks instead of compiling the surrounding train-step code.
        compiled_obj = torch.compile(
            callable_obj,
            dynamic=None,
            fullgraph=False,
            mode="max-autotune-no-cudagraphs",
        )
    except Exception:
        return callable_obj

    state = {"compiled": compiled_obj, "warned": False}

    def wrapped(*args, **kwargs):
        compiled = state["compiled"]
        if compiled is None:
            return callable_obj(*args, **kwargs)
        try:
            return compiled(*args, **kwargs)
        except Exception as exc:
            if isinstance(exc, torch.OutOfMemoryError):
                _clear_compiled_cuda_state()
            if not state["warned"]:
                warnings.warn(
                    f"torch.compile fallback for {name}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                state["warned"] = True
            state["compiled"] = None
            return callable_obj(*args, **kwargs)

    return wrapped


def _should_clear_renderer_cache(backend: str) -> bool:
    return str(backend) == "warp"


def _ordinary_step_view_batch_size(train_cfg: TrainConfig, num_views: int) -> int:
    if num_views <= 0:
        return 0
    explicit_batch = int(train_cfg.view_batch_size)
    if explicit_batch > 0:
        return min(num_views, explicit_batch)
    if num_views <= 4:
        return num_views
    configured_batch = int(getattr(train_cfg, "ordinary_step_view_batch", 0))
    if configured_batch > 0:
        return min(num_views, configured_batch)
    return 4


def _round_robin_view_ids(num_views: int, batch_size: int, step_seed: int) -> list[int]:
    if num_views <= 0:
        return []
    batch = max(1, min(num_views, int(batch_size)))
    if batch == num_views:
        return list(range(num_views))
    start = (int(step_seed) * batch) % num_views
    return [int((start + offset) % num_views) for offset in range(batch)]


def _final_stage_microbatch_size(train_cfg: TrainConfig, views_in_step: int) -> int:
    if views_in_step <= 0:
        return 1
    configured = int(getattr(train_cfg, "final_stage_views_per_microbatch", 0))
    if configured <= 0:
        configured = 2
    return max(1, min(views_in_step, configured))


def _density_event_is_stable_for_freeze(
    density_event: DensityControlResult,
    cfg,
) -> bool:
    if not density_event.ran:
        return False
    debug = density_event.debug_dict()
    if debug.get("weak_view_indices") or debug.get("reseed_view_indices"):
        return False
    visible_fraction = [float(value) for value in debug.get("visible_fraction_of_best", [])]
    intersection_fraction = [float(value) for value in debug.get("intersection_fraction_of_best", [])]
    if not visible_fraction or not intersection_fraction:
        return False
    return min(visible_fraction) >= float(cfg.freeze_min_visible_fraction) and min(intersection_fraction) >= float(
        cfg.freeze_min_intersection_fraction
    )


def _effective_stage_steps(train_cfg: TrainConfig, stage_idx: int, total_stages: int) -> int:
    steps = int(train_cfg.steps_per_stage[stage_idx])
    if stage_idx == (total_stages - 1):
        final_stage_max_steps = int(getattr(train_cfg, "final_stage_max_steps", 0))
        if final_stage_max_steps > 0:
            steps = min(steps, final_stage_max_steps)
    return steps


def _should_early_stop_final_stage(
    train_cfg: TrainConfig,
    loss_window: deque[float],
    *,
    step_index: int,
    density_frozen: bool,
) -> bool:
    patience = int(getattr(train_cfg, "final_stage_early_stop_patience", 0))
    if patience <= 0:
        return False
    if not density_frozen:
        return False
    min_step = int(getattr(train_cfg, "final_stage_early_stop_min_step", 0))
    if (step_index + 1) < min_step:
        return False
    if len(loss_window) < patience:
        return False
    loss_delta = float(getattr(train_cfg, "final_stage_early_stop_loss_delta", 0.0))
    return (max(loss_window) - min(loss_window)) <= loss_delta


def _format_projection_preflight_message(
    *,
    reason: str,
    stage_index: int,
    step_index: int,
    global_step: int,
    record: dict[str, object],
) -> str:
    estimated_sort_buffer_bytes = int(record["estimated_sort_buffer_bytes"])
    budget = record.get("sort_buffer_budget_bytes")
    estimated_gib = float(estimated_sort_buffer_bytes) / float(1024**3)
    budget_text = "unbounded"
    if budget is not None:
        budget_int = int(budget)
        budget_text = f"{budget_int} ({float(budget_int) / float(1024**3):.2f} GiB)"
    reason_text = "stage entry" if reason == "stage_entry" else "post-density"
    step_text = "stage start" if step_index < 0 else f"step {step_index + 1}"
    return (
        f"Projection preflight failed during {reason_text}: stage {stage_index + 1}, {step_text}, "
        f"global_step={global_step}, view={int(record['view_index'])}, "
        f"render={int(record['render_width'])}x{int(record['render_height'])}, "
        f"N={int(record['gaussian_count'])}, visible={int(record['visible_count'])}, "
        f"M={int(record['intersection_count'])}, sort_mode={record['sort_mode']}, "
        f"estimated_sort_bytes={estimated_sort_buffer_bytes} ({estimated_gib:.2f} GiB), "
        f"budget={budget_text}. "
        "Recommended fixes, in order: "
        "1) lower projected footprint with `radius_clip_px`; "
        "2) reduce initial or learned scale growth; "
        "3) freeze or prune before the next stage or after density growth; "
        "4) only then increase `anchor_stride`."
    )


class PoseFreeGaussianSR(nn.Module):
    def __init__(
        self,
        image_shape: Sequence[int],
        intrinsics_init: Tensor,
        num_views: int,
        config: PoseFreeGaussianConfig | None = None,
        anchor_rgb: Tensor | None = None,
        init_shifts_px: Tensor | None = None,
    ):
        super().__init__()
        self.config = config or PoseFreeGaussianConfig()

        if len(image_shape) != 4:
            raise ValueError("image_shape must be [V, 3, H, W]")
        _, c, h, w = image_shape
        if c != 3:
            raise ValueError("Only RGB images are supported in this reference implementation.")
        if anchor_rgb is None:
            raise ValueError("anchor_rgb is required for field initialization")

        self.train_height = int(h)
        self.train_width = int(w)
        self.num_views = int(num_views)

        self.intrinsics = LearnableSharedIntrinsics(
            initial=intrinsics_init,
            learn_intrinsics=self.config.camera.learn_intrinsics,
        )
        self.field_model = CanonicalGaussianField(
            anchor_rgb, intrinsics_init, self.config.field, self.config.appearance
        )
        self.camera_model = LearnableCameraBundle(
            num_views=num_views,
            fx=float(intrinsics_init[0].item()),
            fy=float(intrinsics_init[1].item()),
            init_shifts_px=init_shifts_px,
            device=anchor_rgb.device,
            dtype=anchor_rgb.dtype,
        )
        self.residual_head = (
            ScaleAwareResidualHead(
                feature_dim=self.config.field.feature_dim,
                hidden_dim=self.config.field.residual_hidden_dim,
                residual_scale=self.config.field.residual_scale,
            )
            if self.config.field.use_residual_head
            else None
        )
        self._prepare_render_payload = self._prepare_render_payload_eager
        self._residual_head_forward = self.residual_head if self.residual_head is not None else None
        self._postprocess_rgb = self._postprocess_rgb_eager
        self._observe_and_photometric_loss = self._observe_and_photometric_loss_eager
        self._regularization_impl = self._regularization
        self._render_values_with_meta = _make_compile_disabled(self._render_values_with_meta_eager)
        self._render_density_stats = _make_compile_disabled(self._render_density_stats_eager)
        self._training_view_forward = _make_optional_compiled(
            self._training_view_forward_eager,
            "training_view_forward",
        )
        self._training_view_forward_density = _make_optional_compiled(
            self._training_view_forward_density_eager,
            "training_view_forward_density",
        )
        self._train_step_all_views = _make_optional_compiled(
            self._train_step_all_views_eager,
            "train_step_all_views",
        )
        self._train_step_all_views_density = _make_optional_compiled(
            self._train_step_all_views_density_eager,
            "train_step_all_views_density",
        )
        self._debug_optimizer: torch.optim.Optimizer | None = None

    @classmethod
    def from_images(
        cls,
        images: Tensor,
        intrinsics: Tensor | None = None,
        config: PoseFreeGaussianConfig | None = None,
    ) -> "PoseFreeGaussianSR":
        if images.dim() != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [V, 3, H, W]")
        config = config or PoseFreeGaussianConfig()
        device = images.device
        dtype = images.dtype
        _, _, h, w = images.shape

        if intrinsics is None:
            intrinsics_init = default_intrinsics(h, w, device, dtype, config.camera)
        else:
            intrinsics_init = intrinsics.to(device=device, dtype=dtype)

        shifts = estimate_translation_bootstrap(images) if config.train.use_phasecorr_init else None

        return cls(
            image_shape=tuple(images.shape),
            intrinsics_init=intrinsics_init,
            num_views=images.shape[0],
            config=config,
            anchor_rgb=images[0],
            init_shifts_px=shifts,
        )

    def _make_optimizer(self, train_cfg: TrainConfig) -> torch.optim.Optimizer:
        # Per-parameter learning rates following 3DGS (Kerbl et al., 2023).
        # Each gaussian attribute gets its own LR; position LR is decayed
        # separately via a scheduler (see _make_position_lr_scheduler).
        fp = self.field_model.optimizer_param_dict()
        param_groups: list[dict[str, object]] = [
            {"params": [fp["means3d"]], "lr": train_cfg.lr_position, "name": "position"},
            {"params": [fp["quat_raw"]], "lr": train_cfg.lr_rotation, "name": "rotation"},
            {"params": [fp["log_scale"]], "lr": train_cfg.lr_scaling, "name": "scaling"},
            {"params": [fp["opacity_logit"]], "lr": train_cfg.lr_opacity, "name": "opacity"},
            {"params": [fp["rgb_logit"]], "lr": train_cfg.lr_color, "name": "color"},
            {"params": [fp["latent"]], "lr": train_cfg.lr_latent, "name": "latent"},
        ]
        if "sh_coeffs" in fp:
            param_groups.append({"params": [fp["sh_coeffs"]], "lr": train_cfg.lr_sh, "name": "sh"})
        param_groups.append({"params": list(self.camera_model.parameters()), "lr": train_cfg.lr_camera, "name": "camera"})
        if self.config.camera.learn_intrinsics:
            param_groups.append({"params": list(self.intrinsics.parameters()), "lr": train_cfg.lr_camera, "name": "intrinsics"})
        if self.residual_head is not None:
            param_groups.append({"params": list(self.residual_head.parameters()), "lr": train_cfg.lr_residual, "name": "residual"})
        # Adam epsilon=1e-15 follows 3DGS; much smaller than PyTorch default
        # (1e-8).  This prevents the adaptive LR from being dominated by
        # epsilon when second-moment estimates are very small.
        return torch.optim.Adam(param_groups, eps=1e-15)

    @staticmethod
    def _make_position_lr_scheduler(
        opt: torch.optim.Optimizer,
        lr_init: float,
        lr_final: float,
        max_steps: int,
    ) -> torch.optim.lr_scheduler.LambdaLR:
        """Log-linear position LR decay, constant for all other param groups.

        Follows 3DGS (Kerbl et al., 2023) which decays only position LR via
        ``lr = lr_init^(1-t) * lr_final^t`` where ``t = step / max_steps``.
        Borrowed originally from Plenoxels (Fridovich-Keil et al., 2022).
        """
        import math

        position_idx: int | None = None
        for i, pg in enumerate(opt.param_groups):
            if pg.get("name") == "position":
                position_idx = i
                break
        log_ratio = math.log(lr_final / lr_init) if lr_init > 0 and lr_final > 0 else 0.0

        def _lr_lambda(step: int, group_idx: int = 0) -> float:
            if group_idx != position_idx or max_steps <= 0:
                return 1.0
            t = min(step / max_steps, 1.0)
            return math.exp(log_ratio * t)

        return torch.optim.lr_scheduler.LambdaLR(
            opt, [lambda step, gi=i: _lr_lambda(step, gi) for i in range(len(opt.param_groups))]
        )

    def _rebuild_optimizer_after_density(
        self,
        old_opt: torch.optim.Optimizer,
        train_cfg: TrainConfig,
        survivor_sources: Tensor | None,
        appended_count: int,
    ) -> torch.optim.Optimizer:
        """Zero Adam momentum/variance for newly activated gaussian rows.

        Following 3DGS (Kerbl et al., 2023), which zero-initializes optimizer
        state for new gaussians created during density control
        (``cat_tensors_to_optimizer`` appends zeros to ``exp_avg`` and
        ``exp_avg_sq``).  Our fixed-capacity approach pre-allocates parameter
        tensors, so we zero the Adam state for the newly activated tail rows.
        """
        del train_cfg, survivor_sources
        if appended_count <= 0:
            return old_opt
        new_active = self.field_model.num_gaussians
        new_start = new_active - appended_count
        for pg in old_opt.param_groups:
            if pg.get("name") in ("camera", "intrinsics", "residual"):
                continue
            for param in pg["params"]:
                state = old_opt.state.get(param)
                if state is None:
                    continue
                for key in ("exp_avg", "exp_avg_sq"):
                    buf = state.get(key)
                    if buf is not None:
                        buf[new_start:new_active].zero_()
        return old_opt

    def _scale_intrinsics(self, out_h: int, out_w: int) -> Tensor:
        scale_x = out_w / self.train_width
        scale_y = out_h / self.train_height
        return self.intrinsics.get(scale_x=scale_x, scale_y=scale_y)

    def _renderer_config(self) -> RasterConfig:
        render_cfg = self.config.render
        return RasterConfig(
            backend=render_cfg.backend,
            tile_size=render_cfg.tile_size,
            near_plane=render_cfg.near,
            far_plane=render_cfg.far,
            eps2d=render_cfg.eps2d,
            radius_clip=render_cfg.radius_clip_px,
            max_sort_buffer_bytes=render_cfg.max_sort_buffer_bytes,
            rasterize_mode="antialiased" if render_cfg.antialiased_opacity else "classic",
            alpha_min=render_cfg.alpha_threshold,
            transmittance_eps=render_cfg.transmittance_threshold,
            backward_impl=render_cfg.backward_impl,
            background_rgb=render_cfg.bg_color,
            helion_static_shapes=render_cfg.helion_static_shapes,
            helion_runtime_autotune=render_cfg.helion_runtime_autotune,
        )

    def _viewmat_from_pose(self, R_cw: Tensor, t_cw: Tensor) -> Tensor:
        viewmat = torch.eye(4, device=R_cw.device, dtype=R_cw.dtype)
        viewmat[:3, :3] = R_cw
        viewmat[:3, 3] = t_cw
        return viewmat

    def _K_from_intrinsics(self, intrinsics: Tensor) -> Tensor:
        fx, fy, cx, cy = intrinsics.unbind(dim=0)
        K = torch.eye(3, device=intrinsics.device, dtype=intrinsics.dtype)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy
        return K

    def _packed_values(self, rgb: Tensor, latent: Tensor) -> Tensor:
        alpha_one = torch.ones(rgb.shape[0], 1, device=rgb.device, dtype=rgb.dtype)
        packed = torch.cat((rgb, latent, alpha_one), dim=-1)
        # Downcast packed values to the configured render dtype (FP32 or BF16).
        # Field outputs (rgb from sigmoid, latent, alpha=1) are bounded in [0,1]
        # or moderate magnitude, so BF16 is safe.  The rasterization kernel loads
        # these per-Gaussian per-pixel; halving the dtype halves the bandwidth
        # of the main memory bottleneck.  The .to() is a no-op when already FP32.
        render_dtype = self.config.precision.resolve_values_dtype()
        return packed.to(render_dtype) if render_dtype != packed.dtype else packed

    def _packed_background(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        # Background must match values dtype for consistency in the kernel's
        # output blending (accum + trans * bg).to(values.dtype).
        render_dtype = self.config.precision.resolve_values_dtype()
        c = 3 + self.config.field.feature_dim + 1
        bg = torch.zeros(c, device=device, dtype=render_dtype)
        bg[:3] = torch.tensor(self.config.render.bg_color, device=device, dtype=render_dtype)
        return bg

    def _render_inputs(self, field: dict[str, Tensor | None], R_cw: Tensor, t_cw: Tensor, intrinsics: Tensor):
        viewmat = self._viewmat_from_pose(R_cw, t_cw)
        K = self._K_from_intrinsics(intrinsics)
        means3d = field["means3d"]
        quat = field["quat"]
        scale = field["scale"]
        opacity = field["opacity"]
        rgb = field["rgb"]
        latent = field["latent"]
        assert (
            means3d is not None
            and quat is not None
            and scale is not None
            and opacity is not None
            and rgb is not None
            and latent is not None
        )
        ac = field.get("active_count")
        active_count = ac if isinstance(ac, Tensor) else means3d.shape[0]
        values = self._packed_values(rgb, latent)
        background = self._packed_background(values.device, values.dtype)
        return viewmat, K, means3d, quat, scale, opacity, values, background, active_count

    def _prepare_render_payload_eager(
        self,
        base_intrinsics: Tensor,
        render_intrinsics: Tensor,
        R_cw: Tensor,
        t_cw: Tensor,
    ):
        field = self.field_model(base_intrinsics, R_cw=R_cw, t_cw=t_cw, padded=True)
        viewmat, K, means3d, quat, scale, opacity, values, background, active_count = self._render_inputs(
            field,
            R_cw,
            t_cw,
            render_intrinsics,
        )
        return field, viewmat, K, means3d, quat, scale, opacity, values, background, active_count

    def _render_stats_with_pose(
        self,
        field: dict[str, Tensor | None],
        R_cw: Tensor,
        t_cw: Tensor,
        intrinsics: Tensor,
        out_h: int,
        out_w: int,
        residual_map: Tensor | None = None,
    ) -> dict[str, Tensor]:
        viewmat, K, means3d, quat, scale, opacity, values, background, active_count = self._render_inputs(
            field, R_cw, t_cw, intrinsics
        )
        return render_stats(
            means=means3d.contiguous(),
            quat=quat.contiguous(),
            scale=scale.contiguous(),
            opacity=opacity.contiguous(),
            viewmat=viewmat.contiguous(),
            K=K.contiguous(),
            width=out_w,
            height=out_h,
            cfg=self._renderer_config(),
            residual_map=None if residual_map is None else residual_map.contiguous(),
            active_count=active_count,
        )

    def _render_stats_from_prepared(
        self,
        prepared: PreparedVisibility,
        opacity: Tensor,
        residual_map: Tensor | None = None,
    ) -> dict[str, Tensor]:
        return render_stats_prepared(
            prepared,
            opacity=opacity.contiguous(),
            cfg=self._renderer_config(),
            residual_map=None if residual_map is None else residual_map.contiguous(),
            active_count=prepared.gaussian_count,
        )

    def _meta_tuple(self, meta: dict[str, Tensor]) -> tuple[Tensor, ...]:
        return tuple(meta[key] for key in _META_STAT_KEYS)

    def _meta_dict(self, values: Sequence[Tensor]) -> dict[str, Tensor]:
        return {key: value for key, value in zip(_META_STAT_KEYS, values, strict=True)}

    def _density_stats_dict(self, values: Sequence[Tensor]) -> dict[str, Tensor]:
        return {key: value for key, value in zip(_DENSITY_STAT_KEYS, values, strict=True)}

    def _render_values_with_meta_eager(
        self,
        means: Tensor,
        quat: Tensor,
        scale: Tensor,
        values: Tensor,
        opacity: Tensor,
        background: Tensor,
        viewmat: Tensor,
        K: Tensor,
        out_h: int,
        out_w: int,
        active_count: int | Tensor,
    ) -> tuple[Tensor, PreparedVisibility, tuple[Tensor, ...]]:
        packed_hwc, prepared = render_values(
            means=means.contiguous(),
            quat=quat.contiguous(),
            scale=scale.contiguous(),
            values=values.contiguous(),
            opacity=opacity.contiguous(),
            background=background.contiguous(),
            viewmat=viewmat.contiguous(),
            K=K.contiguous(),
            width=out_w,
            height=out_h,
            cfg=self._renderer_config(),
            return_prepared=True,
            active_count=active_count,
        )
        return packed_hwc, prepared, self._meta_tuple(prepared.meta())

    def _render_density_stats_eager(
        self,
        prepared: PreparedVisibility,
        opacity: Tensor,
        residual_map: Tensor,
    ) -> tuple[Tensor, ...]:
        stats = render_stats_prepared(
            prepared,
            opacity=opacity.contiguous(),
            cfg=self._renderer_config(),
            residual_map=residual_map.contiguous(),
            active_count=prepared.gaussian_count,
        )
        return tuple(stats[key].detach() for key in _DENSITY_STAT_KEYS)

    def _residual_map_for_render(self, pred_observed: Tensor, target: Tensor, render_h: int, render_w: int) -> Tensor:
        residual = (pred_observed - target).abs().mean(dim=0, keepdim=True)
        if residual.shape[-2:] != (render_h, render_w):
            residual = torch.nn.functional.interpolate(
                residual.unsqueeze(0),
                size=(render_h, render_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return residual[0]

    def _postprocess_rgb_eager(
        self,
        packed: Tensor,
        out_h: int,
        out_w: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        rgb = packed[:3]
        latent = packed[3 : 3 + self.config.field.feature_dim]
        alpha = packed[-1:].clamp(0.0, 1.0)

        if self._residual_head_forward is not None:
            scale_x = out_w / self.train_width
            scale_y = out_h / self.train_height
            residual = self._residual_head_forward(latent, scale_x=scale_x, scale_y=scale_y)
            rgb = (rgb + alpha * residual).clamp(0.0, 1.0)
        else:
            rgb = rgb.clamp(0.0, 1.0)
        return rgb, latent, alpha

    def _regularization_field(self, field: dict[str, Tensor | None]) -> dict[str, Tensor | None]:
        depth_map = field.get("depth_map")
        opacity = field.get("opacity")
        scale = field.get("scale")
        return {
            "depth_map": None if depth_map is None else depth_map.clone(),
            "opacity": None if opacity is None else opacity.clone(),
            "scale": None if scale is None else scale.clone(),
        }

    def _observe_and_photometric_loss_eager(self, rgb: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        pred = render_observe_rgb(
            rgb,
            int(target.shape[-2]),
            int(target.shape[-1]),
            self.config.observation,
            layout="chw",
        )
        return pred, self._photometric_loss(pred, target)

    def _training_view_forward_eager(
        self,
        base_intrinsics: Tensor,
        render_intrinsics: Tensor,
        R_cw: Tensor,
        t_cw: Tensor,
        target: Tensor,
        out_h: int,
        out_w: int,
    ) -> tuple[Tensor, ...]:
        _field, viewmat, K, means3d, quat, scale, opacity, values, background, active_count = (
            self._prepare_render_payload(
                base_intrinsics,
                render_intrinsics,
                R_cw,
                t_cw,
            )
        )
        packed_hwc, _prepared, meta = self._render_values_with_meta(
            means3d,
            quat,
            scale,
            values,
            opacity,
            background,
            viewmat,
            K,
            out_h,
            out_w,
            active_count,
        )
        packed = packed_hwc.permute(2, 0, 1).contiguous()
        rgb, _latent, _alpha = self._postprocess_rgb(packed, out_h, out_w)
        pred, photo_term = self._observe_and_photometric_loss(rgb, target)
        return (photo_term, torch.isfinite(rgb).all(), torch.isfinite(pred).all(), *meta)

    def _training_view_forward_density_eager(
        self,
        base_intrinsics: Tensor,
        render_intrinsics: Tensor,
        R_cw: Tensor,
        t_cw: Tensor,
        target: Tensor,
        out_h: int,
        out_w: int,
    ) -> tuple[Tensor, ...]:
        _field, viewmat, K, means3d, quat, scale, opacity, values, background, active_count = (
            self._prepare_render_payload(
                base_intrinsics,
                render_intrinsics,
                R_cw,
                t_cw,
            )
        )
        packed_hwc, prepared, meta = self._render_values_with_meta(
            means3d,
            quat,
            scale,
            values,
            opacity,
            background,
            viewmat,
            K,
            out_h,
            out_w,
            active_count,
        )
        packed = packed_hwc.permute(2, 0, 1).contiguous()
        rgb, _latent, _alpha = self._postprocess_rgb(packed, out_h, out_w)
        pred, photo_term = self._observe_and_photometric_loss(rgb, target)
        residual_map = self._residual_map_for_render(pred, target, out_h, out_w)
        density_stats = self._render_density_stats(prepared, opacity, residual_map)
        return (photo_term, torch.isfinite(rgb).all(), torch.isfinite(pred).all(), *meta, *density_stats)

    def _train_step_all_views_eager(
        self,
        opt: torch.optim.Optimizer,
        stage_targets: Tensor,
        out_h: int,
        out_w: int,
        stage_alpha: float,
        grad_clip: float,
    ) -> tuple[Tensor, ...]:
        opt.zero_grad(set_to_none=True)
        base_intrinsics = self.intrinsics.get()
        render_intrinsics = self._scale_intrinsics(out_h, out_w)
        R_all, t_all = self.camera_model.world_to_camera()
        photo_loss = stage_targets.new_tensor(0.0)
        rgb_finite = torch.ones((), device=stage_targets.device, dtype=torch.bool)
        pred_finite = torch.ones((), device=stage_targets.device, dtype=torch.bool)
        for view_index in range(self.num_views):
            train_view = self._training_view_forward_eager(
                base_intrinsics,
                render_intrinsics,
                R_all[view_index],
                t_all[view_index],
                stage_targets[view_index],
                out_h,
                out_w,
            )
            photo_loss = photo_loss + train_view[0]
            rgb_finite = rgb_finite & train_view[1]
            pred_finite = pred_finite & train_view[2]
        photo_loss = photo_loss / float(max(self.num_views, 1))
        reg_loss = self._regularization(None, stage_alpha)
        loss = photo_loss + reg_loss
        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
        opt.step()
        return loss.detach(), photo_loss.detach(), reg_loss.detach(), rgb_finite.detach(), pred_finite.detach()

    def _train_step_all_views_density_eager(
        self,
        opt: torch.optim.Optimizer,
        stage_targets: Tensor,
        out_h: int,
        out_w: int,
        stage_alpha: float,
        grad_clip: float,
    ) -> tuple[Tensor, ...]:
        opt.zero_grad(set_to_none=True)
        base_intrinsics = self.intrinsics.get()
        render_intrinsics = self._scale_intrinsics(out_h, out_w)
        R_all, t_all = self.camera_model.world_to_camera()
        photo_loss = stage_targets.new_tensor(0.0)
        rgb_finite = torch.ones((), device=stage_targets.device, dtype=torch.bool)
        pred_finite = torch.ones((), device=stage_targets.device, dtype=torch.bool)
        stats_accum: dict[str, Tensor] | None = None
        meta_offset = 3 + len(_META_STAT_KEYS)
        for view_index in range(self.num_views):
            train_view = self._training_view_forward_density_eager(
                base_intrinsics,
                render_intrinsics,
                R_all[view_index],
                t_all[view_index],
                stage_targets[view_index],
                out_h,
                out_w,
            )
            photo_loss = photo_loss + train_view[0]
            rgb_finite = rgb_finite & train_view[1]
            pred_finite = pred_finite & train_view[2]
            stats_accum = self._accumulate_render_stats(stats_accum, self._density_stats_dict(train_view[meta_offset:]))
        photo_loss = photo_loss / float(max(self.num_views, 1))
        reg_loss = self._regularization(None, stage_alpha)
        loss = photo_loss + reg_loss
        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
        opt.step()
        assert stats_accum is not None
        return (
            loss.detach(),
            photo_loss.detach(),
            reg_loss.detach(),
            rgb_finite.detach(),
            pred_finite.detach(),
            *(stats_accum[key].detach() for key in _DENSITY_STAT_KEYS),
        )

    def render_with_pose(
        self,
        R_cw: Tensor,
        t_cw: Tensor,
        out_h: int,
        out_w: int,
        return_aux: bool = True,
        stats_mode: str = "full",
        return_prepared: bool = False,
        profile: bool = False,
    ) -> dict[str, object]:
        if stats_mode not in {"meta", "full"}:
            raise ValueError(f"stats_mode must be 'meta' or 'full', got {stats_mode!r}")
        intr = self._scale_intrinsics(out_h, out_w)
        timing_device = R_cw.device
        field_s = 0.0
        render_s = 0.0
        aux_stats_s = 0.0
        base_intr = self.intrinsics.get()
        if profile:
            self._sync_for_timing(timing_device)
            t0 = time.perf_counter()
        with torch.profiler.record_function("pipeline.render.prepare_field"):
            field, viewmat, K, means3d, quat, scale, opacity, values, background, active_count = (
                self._prepare_render_payload(
                    base_intr,
                    intr,
                    R_cw,
                    t_cw,
                )
            )
        means_render = means3d.contiguous()
        quat_render = quat.contiguous()
        scale_render = scale.contiguous()
        opacity_render = opacity.contiguous()
        values_render = values.contiguous()
        background_render = background.contiguous()
        viewmat_render = viewmat.contiguous()
        K_render = K.contiguous()
        if profile:
            self._sync_for_timing(timing_device)
            field_s = time.perf_counter() - t0
            t0 = time.perf_counter()
        renderer_cfg = self._renderer_config()
        with torch.profiler.record_function("pipeline.render.warp"):
            render_result = render_values(
                means=means_render,
                quat=quat_render,
                scale=scale_render,
                values=values_render,
                opacity=opacity_render,
                background=background_render,
                viewmat=viewmat_render,
                K=K_render,
                width=out_w,
                height=out_h,
                cfg=renderer_cfg,
                return_prepared=return_aux,
                active_count=active_count,
            )
        if return_aux:
            packed_hwc, prepared = render_result
        else:
            packed_hwc = render_result
            prepared = None
        if profile:
            self._sync_for_timing(timing_device)
            render_s = time.perf_counter() - t0
        with torch.profiler.record_function("pipeline.render.postprocess"):
            packed = packed_hwc.permute(2, 0, 1).contiguous()
            rgb, latent, alpha = self._postprocess_rgb(packed, out_h, out_w)

        result: dict[str, object] = {"rgb": rgb}
        if return_aux:
            if profile:
                self._sync_for_timing(timing_device)
                t0 = time.perf_counter()
            assert prepared is not None
            with torch.profiler.record_function("pipeline.render.aux_stats"):
                if stats_mode == "meta":
                    render_stats = prepared.meta()
                else:
                    render_stats = render_stats_prepared(
                        prepared,
                        opacity=opacity_render,
                        cfg=renderer_cfg,
                        active_count=active_count,
                    )
            if profile:
                self._sync_for_timing(timing_device)
                aux_stats_s = time.perf_counter() - t0
            result.update({
                "alpha": alpha,
                "latent": latent,
                "field": self._regularization_field(field),
                "packed": packed,
                "render_stats": render_stats,
            })
            if return_prepared:
                result["_prepared_visibility"] = prepared
                result["_opacity"] = opacity_render
        if profile:
            result["profile"] = {
                "field_s": field_s,
                "render_s": render_s,
                "aux_stats_s": aux_stats_s,
            }
        return result

    def render_view(
        self,
        view_index: int,
        scale: float | None = None,
        out_size: tuple[int, int] | None = None,
        return_aux: bool = True,
        stats_mode: str = "full",
    ) -> dict[str, Tensor | dict[str, Tensor | None] | dict[str, Tensor]]:
        if out_size is None:
            s = 1.0 if scale is None else float(scale)
            out_h = max(1, int(round(self.train_height * s)))
            out_w = max(1, int(round(self.train_width * s)))
        else:
            out_h, out_w = int(out_size[0]), int(out_size[1])

        R_all, t_all = self.camera_model.world_to_camera()
        return self.render_with_pose(
            R_all[view_index],
            t_all[view_index],
            out_h,
            out_w,
            return_aux=return_aux,
            stats_mode=stats_mode,
        )

    def render_view_meta(
        self,
        view_index: int,
        scale: float | None = None,
        out_size: tuple[int, int] | None = None,
    ) -> dict[str, Tensor]:
        if out_size is None:
            s = 1.0 if scale is None else float(scale)
            out_h = max(1, int(round(self.train_height * s)))
            out_w = max(1, int(round(self.train_width * s)))
        else:
            out_h, out_w = int(out_size[0]), int(out_size[1])

        base_intrinsics = self.intrinsics.get()
        render_intrinsics = self._scale_intrinsics(out_h, out_w)
        R_all, t_all = self.camera_model.world_to_camera()
        _field, viewmat, K, means3d, quat, scale_render, _opacity, _values, _background, active_count = (
            self._prepare_render_payload_eager(
                base_intrinsics,
                render_intrinsics,
                R_all[view_index],
                t_all[view_index],
            )
        )
        return render_visibility_meta(
            means=means3d.contiguous(),
            quat=quat.contiguous(),
            scale=scale_render.contiguous(),
            viewmat=viewmat.contiguous(),
            K=K.contiguous(),
            width=out_w,
            height=out_h,
            cfg=self._renderer_config(),
            active_count=active_count,
        )

    def _projection_preflight_for_views(
        self,
        view_ids: Sequence[int],
        render_h: int,
        render_w: int,
    ) -> list[dict[str, object]]:
        base_intrinsics = self.intrinsics.get()
        render_intrinsics = self._scale_intrinsics(render_h, render_w)
        R_all, t_all = self.camera_model.world_to_camera()
        renderer_cfg = self._renderer_config()
        records: list[dict[str, object]] = []
        for view_index in view_ids:
            _field, viewmat, K, means3d, quat, scale_render, _opacity, _values, _background, active_count = (
                self._prepare_render_payload_eager(
                    base_intrinsics,
                    render_intrinsics,
                    R_all[view_index],
                    t_all[view_index],
                )
            )
            meta = render_projection_meta(
                means=means3d.contiguous(),
                quat=quat.contiguous(),
                scale=scale_render.contiguous(),
                viewmat=viewmat.contiguous(),
                K=K.contiguous(),
                width=render_w,
                height=render_h,
                cfg=renderer_cfg,
                active_count=active_count,
            )
            records.append({
                "view_index": int(view_index),
                **meta,
            })
        return records

    def _run_projection_preflight(
        self,
        *,
        view_ids: Sequence[int],
        render_h: int,
        render_w: int,
        stage_index: int,
        step_index: int,
        global_step: int,
        reason: str,
        progress_event_callback: Callable[[dict], None] | None = None,
        fail_on_error: bool,
    ) -> dict[str, object]:
        if not self.field_model.means3d.is_cuda:
            records = [
                {
                    "view_index": int(view_index),
                    "gaussian_count": int(self.field_model.num_gaussians),
                    "visible_count": 0,
                    "intersection_count": 0,
                    "tile_count": 0,
                    "tiles_x": 0,
                    "tiles_y": 0,
                    "render_width": int(render_w),
                    "render_height": int(render_h),
                    "sort_mode": "cpu_skip",
                    "estimated_sort_buffer_bytes": 0,
                    "sort_buffer_budget_bytes": 0,
                    "sort_buffer_within_budget": True,
                }
                for view_index in view_ids
            ]
            summary = {
                "reason": reason,
                "stage_index": int(stage_index),
                "step_index": int(step_index),
                "global_step": int(global_step),
                "render_height": int(render_h),
                "render_width": int(render_w),
                "unsafe_view_count": 0,
                "worst_view_index": int(records[0]["view_index"]) if records else -1,
                "max_intersection_count": 0,
                "max_estimated_sort_buffer_bytes": 0,
                "all_within_budget": True,
                "views": records,
            }
            return summary
        records = self._projection_preflight_for_views(view_ids, render_h, render_w)
        unsafe_records = [record for record in records if not bool(record["sort_buffer_within_budget"])]
        worst_record = max(records, key=lambda record: int(record["estimated_sort_buffer_bytes"]))
        summary = {
            "reason": reason,
            "stage_index": int(stage_index),
            "step_index": int(step_index),
            "global_step": int(global_step),
            "render_height": int(render_h),
            "render_width": int(render_w),
            "unsafe_view_count": int(len(unsafe_records)),
            "worst_view_index": int(worst_record["view_index"]),
            "max_intersection_count": int(worst_record["intersection_count"]),
            "max_estimated_sort_buffer_bytes": int(worst_record["estimated_sort_buffer_bytes"]),
            "all_within_budget": bool(not unsafe_records),
            "views": records,
        }
        for record in records:
            emit_progress_event(
                {
                    "event": "projection_preflight",
                    "reason": reason,
                    "stage_index": int(stage_index),
                    "step_index": int(step_index),
                    "global_step": int(global_step),
                    **record,
                },
                callback=progress_event_callback,
            )
        if unsafe_records and fail_on_error:
            offending = max(unsafe_records, key=lambda record: int(record["estimated_sort_buffer_bytes"]))
            raise ProjectionPreflightError({
                "reason": reason,
                "stage_index": int(stage_index),
                "step_index": int(step_index),
                "global_step": int(global_step),
                "record": offending,
                "message": _format_projection_preflight_message(
                    reason=reason,
                    stage_index=stage_index,
                    step_index=step_index,
                    global_step=global_step,
                    record=offending,
                ),
            })
        return summary

    def _reserve_helion_intersection_capacity_from_preflight(self, summary: dict[str, object]) -> None:
        if str(self.config.render.backend) != "helion":
            return
        required_count = int(summary.get("max_intersection_count", 0))
        if required_count <= 0:
            return
        reserve_renderer_intersection_capacity(
            backend="helion",
            device=self.field_model.means3d.device,
            width=int(summary["render_width"]),
            height=int(summary["render_height"]),
            required_count=required_count,
        )

    def preflight_training_stages(self) -> list[dict[str, object]]:
        results: list[dict[str, object]] = []
        if _should_clear_renderer_cache(self.config.render.backend):
            clear_renderer_caches(backend=self.config.render.backend)
        total_stages = len(self.config.train.stage_scales)
        for stage_idx, stage_scale in enumerate(self.config.train.stage_scales):
            stage_h = max(1, int(round(self.train_height * float(stage_scale))))
            stage_w = max(1, int(round(self.train_width * float(stage_scale))))
            render_h, render_w = observation_render_size(stage_h, stage_w, self.config.observation)
            summary = self._run_projection_preflight(
                view_ids=list(range(self.num_views)),
                render_h=render_h,
                render_w=render_w,
                stage_index=stage_idx,
                step_index=-1,
                global_step=-1,
                reason="manual_preflight",
                progress_event_callback=None,
                fail_on_error=False,
            )
            self._reserve_helion_intersection_capacity_from_preflight(summary)
            summary["stage_scale"] = float(stage_scale)
            summary["total_stages"] = int(total_stages)
            results.append(summary)
        return results

    def _photometric_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        l1 = charbonnier(pred - target).mean()
        if self.config.train.photometric_ssim_weight > 0.0:
            ssim_l = 1.0 - ssim_value(pred, target)
        else:
            ssim_l = pred.new_tensor(0.0)
        return self.config.train.photometric_l1_weight * l1 + self.config.train.photometric_ssim_weight * ssim_l

    def _regularization(self, field: dict[str, Tensor | None] | None, stage_alpha: float) -> Tensor:
        opacity = None if field is None else field.get("opacity")
        scale = None if field is None else field.get("scale")
        if opacity is None:
            opacity = torch.sigmoid(self.field_model.opacity_logit[:, 0])
        if scale is None:
            scale = torch.exp(self.field_model.log_scale)
        depth_tv = self.field_model.seed_depth_tv()
        opacity_reg = opacity.mean()
        scale_reg = scale.mean()
        pose_reg_weight = (
            1.0 - stage_alpha
        ) * self.config.train.pose_weight_stage0 + stage_alpha * self.config.train.pose_weight_final
        pose_reg = self.camera_model.pose_regularizer()
        return (
            self.config.train.depth_tv_weight * depth_tv
            + self.config.train.opacity_weight * opacity_reg
            + self.config.train.scale_weight * scale_reg
            + pose_reg_weight * pose_reg
        )

    def _accumulate_render_stats(
        self, accum: dict[str, Tensor] | None, render_stats: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        render_stats = {k: v for k, v in render_stats.items() if not k.startswith("meta_")}
        if accum is None:
            return {k: v.detach().clone() for k, v in render_stats.items()}
        return {k: accum[k] + render_stats[k].detach() for k in accum.keys()}

    def _sync_for_timing(self, device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device=device)

    def _cudagraph_mark_step_begin(self) -> None:
        compiler = getattr(torch, "compiler", None)
        fn = getattr(compiler, "cudagraph_mark_step_begin", None)
        if callable(fn):
            fn()

    def _memory_snapshot(self, device: torch.device) -> dict[str, float]:
        if device.type != "cuda":
            return {
                "cuda_alloc_gib": 0.0,
                "cuda_reserved_gib": 0.0,
                "cuda_max_alloc_gib": 0.0,
            }
        denom = float(1024**3)
        return {
            "cuda_alloc_gib": float(torch.cuda.memory_allocated(device) / denom),
            "cuda_reserved_gib": float(torch.cuda.memory_reserved(device) / denom),
            "cuda_max_alloc_gib": float(torch.cuda.max_memory_allocated(device) / denom),
        }

    def _render_stats_summary(self, render_stats: dict[str, Tensor]) -> dict[str, int]:
        def _scalar(name: str) -> int:
            value = render_stats.get(name)
            if value is None:
                return 0
            return int(value.item())

        return {
            "gaussian_count": _scalar("meta_gaussian_count"),
            "visible_count": _scalar("meta_visible_count"),
            "intersection_count": _scalar("meta_intersection_count"),
            "tile_count": _scalar("meta_tile_count"),
            "tiles_x": _scalar("meta_tiles_x"),
            "tiles_y": _scalar("meta_tiles_y"),
            "render_width": _scalar("meta_render_width"),
            "render_height": _scalar("meta_render_height"),
        }

    def _format_eta(self, seconds: float) -> str:
        seconds_i = max(int(round(seconds)), 0)
        hours = seconds_i // 3600
        minutes = (seconds_i % 3600) // 60
        secs = seconds_i % 60
        if hours > 0:
            return f"{hours}h{minutes:02d}m{secs:02d}s"
        if minutes > 0:
            return f"{minutes}m{secs:02d}s"
        return f"{secs}s"

    def _timestamp(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    def _progress_print(self, message: str) -> None:
        print(f"[{self._timestamp()}] {message}", flush=True)

    def _tensor_nonfinite_count(self, value: Tensor | None) -> int:
        if value is None:
            return 0
        finite = torch.isfinite(value)
        return int((~finite).sum().item())

    def _monitored_parameter_items(self) -> list[tuple[str, Tensor]]:
        items: list[tuple[str, Tensor]] = [
            ("means3d", self.field_model.means3d),
            ("quat_raw", self.field_model.quat_raw),
            ("log_scale", self.field_model.log_scale),
            ("opacity_logit", self.field_model.opacity_logit),
            ("rgb_logit", self.field_model.rgb_logit),
            ("latent", self.field_model.latent),
            ("camera_pose", self.camera_model.pose_rest),
            ("intrinsics_log_fx", self.intrinsics.log_fx),
            ("intrinsics_log_fy", self.intrinsics.log_fy),
            ("intrinsics_cx", self.intrinsics.cx),
            ("intrinsics_cy", self.intrinsics.cy),
        ]
        if getattr(self.field_model, "sh_coeffs", None) is not None:
            items.append(("sh_coeffs", self.field_model.sh_coeffs))
        if self.residual_head is not None:
            for name, param in self.residual_head.named_parameters():
                items.append((f"residual_head.{name}", param))
        return items

    def _parameter_nonfinite_report(self) -> dict[str, int]:
        report = {name: self._tensor_nonfinite_count(param) for name, param in self._monitored_parameter_items()}
        return {k: v for k, v in report.items() if v > 0}

    def _gradient_stats_report(self) -> dict[str, dict[str, float | int]]:
        report: dict[str, dict[str, float | int]] = {}
        for name, param in self._monitored_parameter_items():
            grad = param.grad
            if grad is None:
                continue
            grad_detached = grad.detach()
            finite_mask = torch.isfinite(grad_detached)
            nonfinite_count = int((~finite_mask).sum().item())
            finite_vals = grad_detached[finite_mask]
            if finite_vals.numel() > 0:
                abs_vals = finite_vals.abs()
                max_abs = float(abs_vals.max().item())
                mean_abs = float(abs_vals.mean().item())
                l2_norm = float(torch.linalg.vector_norm(finite_vals).item())
            else:
                max_abs = float("nan")
                mean_abs = float("nan")
                l2_norm = float("nan")
            report[name] = {
                "nonfinite": nonfinite_count,
                "max_abs": max_abs,
                "mean_abs": mean_abs,
                "l2_norm": l2_norm,
            }
        return report

    def _gradient_nonfinite_report(self) -> dict[str, int]:
        report = self._gradient_stats_report()
        return {name: int(stats["nonfinite"]) for name, stats in report.items() if int(stats["nonfinite"]) > 0}

    def _raise_nonfinite(
        self,
        kind: str,
        stage_idx: int,
        step_idx: int,
        global_step: int,
        extra: dict[str, object] | None = None,
    ) -> None:
        payload: dict[str, object] = {
            "kind": kind,
            "stage_index": int(stage_idx),
            "step_index": int(step_idx),
            "global_step": int(global_step),
            "nonfinite_params": self._parameter_nonfinite_report(),
        }
        if extra:
            payload.update(extra)
        raise NonFiniteTrainingError(payload)

    def fit(
        self,
        images: Tensor,
        train_cfg: TrainConfig | None = None,
        density_event_log_path: str | Path | None = None,
        density_event_callback: Callable[[dict], None] | None = None,
        verbose_progress: bool = False,
        progress_log_path: str | Path | None = None,
        progress_event_callback: Callable[[dict], None] | None = None,
        synchronize_progress_timing: bool = False,
        step_callback: Callable[[int, int, int], None] | None = None,
        optimizer_state_dict: dict[str, object] | None = None,
        resume_stage_index: int = 0,
        resume_step_index: int = -1,
        resume_global_step: int = -1,
    ) -> dict[str, list]:
        train_cfg = train_cfg or self.config.train
        if images.shape[0] != self.num_views:
            raise ValueError("Number of images does not match pipeline initialization.")
        if images.shape[-2] != self.train_height or images.shape[-1] != self.train_width:
            raise ValueError("Training images must match initialization resolution.")

        total_stages = len(train_cfg.stage_scales)
        if len(train_cfg.steps_per_stage) != total_stages:
            raise ValueError("steps_per_stage must have the same length as stage_scales.")
        if resume_stage_index < 0 or resume_stage_index > total_stages:
            raise ValueError(f"resume_stage_index must be in [0, {total_stages}], got {resume_stage_index}")
        if resume_step_index < -1:
            raise ValueError(f"resume_step_index must be >= -1, got {resume_step_index}")
        if resume_stage_index < total_stages and resume_step_index >= int(
            train_cfg.steps_per_stage[resume_stage_index]
        ):
            raise ValueError("resume_step_index must be smaller than the number of steps in the resumed stage")

        opt = self._make_optimizer(train_cfg)
        if optimizer_state_dict is not None:
            opt.load_state_dict(optimizer_state_dict)
            for state in opt.state.values():
                for key, value in list(state.items()):
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device=images.device)
        self._debug_optimizer = opt
        # Position LR scheduler: will be (re-)created at each stage start.
        lr_scheduler: torch.optim.lr_scheduler.LambdaLR | None = None
        history: dict[str, list] = {"loss": [], "photo": [], "reg": [], "num_gaussians": [], "density_events": []}

        progress_enabled = bool(
            verbose_progress or progress_log_path is not None or progress_event_callback is not None
        )
        debug_progress_enabled = bool(verbose_progress)
        timing_enabled = bool(synchronize_progress_timing or verbose_progress)
        run_start = time.perf_counter()

        def _emit_progress(event: dict) -> None:
            if not progress_enabled:
                return
            payload = {
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "elapsed_s": float(time.perf_counter() - run_start),
                **event,
            }
            emit_progress_event(
                payload,
                jsonl_path=progress_log_path,
                callback=progress_event_callback,
            )

        global_step = int(resume_global_step)
        for stage_idx, stage_scale in enumerate(train_cfg.stage_scales):
            if stage_idx < resume_stage_index:
                continue
            is_final_stage = stage_idx == (total_stages - 1)
            final_stage_density_stable_events = 0
            final_stage_density_frozen = False
            steps = _effective_stage_steps(train_cfg, stage_idx, total_stages)
            step_start_index = max(0, resume_step_index + 1) if stage_idx == resume_stage_index else 0
            if step_start_index >= int(steps):
                continue
            # Reset optimizer at stage transitions to clear stale Adam momentum/variance,
            # and scale LR down proportionally to the resolution increase.
            if stage_idx > 0 and stage_idx != resume_stage_index:
                opt = self._make_optimizer(train_cfg)
                lr_scale = float(train_cfg.stage_scales[0]) / float(stage_scale)
                for pg in opt.param_groups:
                    pg["lr"] = pg["lr"] * lr_scale
                self._debug_optimizer = opt
            # Position LR exponential decay following 3DGS / Plenoxels.
            # The position group's base LR (after stage scaling) decays
            # log-linearly to lr_position_final over this stage's steps.
            pos_lr_now = train_cfg.lr_position
            if stage_idx > 0:
                pos_lr_now *= float(train_cfg.stage_scales[0]) / float(stage_scale)
            lr_scheduler = self._make_position_lr_scheduler(
                opt,
                lr_init=pos_lr_now,
                lr_final=train_cfg.lr_position_final,
                max_steps=int(steps),
            )
            final_stage_loss_window: deque[float] = deque(
                maxlen=max(1, int(getattr(train_cfg, "final_stage_early_stop_patience", 0)))
            )
            final_stage_early_stopped = False
            if _should_clear_renderer_cache(self.config.render.backend):
                clear_renderer_caches(backend=self.config.render.backend)
            stage_h = max(1, int(round(self.train_height * float(stage_scale))))
            stage_w = max(1, int(round(self.train_width * float(stage_scale))))
            render_h, render_w = observation_render_size(stage_h, stage_w, self.config.observation)
            stage_targets = (
                images
                if (stage_h == self.train_height and stage_w == self.train_width)
                else torch.nn.functional.interpolate(images, size=(stage_h, stage_w), mode="area")
            )
            stage_alpha = 0.0 if total_stages == 1 else stage_idx / (total_stages - 1)
            stage_preflight = self._run_projection_preflight(
                view_ids=list(range(self.num_views)),
                render_h=render_h,
                render_w=render_w,
                stage_index=stage_idx,
                step_index=-1,
                global_step=global_step,
                reason="stage_entry",
                progress_event_callback=progress_event_callback,
                fail_on_error=True,
            )
            self._reserve_helion_intersection_capacity_from_preflight(stage_preflight)
            stage_start_time = time.perf_counter()

            _emit_progress({
                "event": "stage_start",
                "stage_index": int(stage_idx),
                "stage_scale": float(stage_scale),
                "steps": int(steps),
                "resume_step_index": int(resume_step_index) if stage_idx == resume_stage_index else -1,
                "resume_global_step": int(global_step) if stage_idx == resume_stage_index else -1,
                "stage_height": int(stage_h),
                "stage_width": int(stage_w),
                "render_height": int(render_h),
                "render_width": int(render_w),
                "num_gaussians": int(self.field_model.num_gaussians),
                "preflight_unsafe_view_count": int(stage_preflight["unsafe_view_count"]),
                "preflight_worst_view_index": int(stage_preflight["worst_view_index"]),
                "preflight_max_intersection_count": int(stage_preflight["max_intersection_count"]),
                "preflight_max_estimated_sort_buffer_bytes": int(stage_preflight["max_estimated_sort_buffer_bytes"]),
            })
            if verbose_progress:
                self._progress_print(
                    f"[stage-start] stage={stage_idx + 1}/{total_stages} scale={stage_scale:.2f} "
                    f"stage={stage_h}x{stage_w} render={render_h}x{render_w} "
                    f"N={self.field_model.num_gaussians} "
                    f"preflight_worst_view={int(stage_preflight['worst_view_index'])} "
                    f"preflight_max_M={int(stage_preflight['max_intersection_count'])} "
                    f"preflight_sort={float(int(stage_preflight['max_estimated_sort_buffer_bytes'])) / float(1024**3):.2f}GiB"
                    + (
                        f" resume_step={step_start_index + 1}"
                        if stage_idx == resume_stage_index and step_start_index > 0
                        else ""
                    )
                )

            for step in range(step_start_index, steps):
                with torch.profiler.record_function("pipeline.step"):
                    step_start_time = time.perf_counter()
                    if not timing_enabled:
                        self._cudagraph_mark_step_begin()
                    density_scheduled = should_run_density_control_for_stage(
                        global_step,
                        self.config.density,
                        stage_idx,
                        total_stages,
                    )
                    density_due = bool(density_scheduled and not (is_final_stage and final_stage_density_frozen))
                    R_all, t_all = self.camera_model.world_to_camera()
                    if density_due:
                        view_ids = list(range(self.num_views))
                    else:
                        ordinary_batch = _ordinary_step_view_batch_size(train_cfg, self.num_views)
                        view_ids = _round_robin_view_ids(self.num_views, ordinary_batch, global_step)
                    final_stage_microbatch = (
                        _final_stage_microbatch_size(train_cfg, len(view_ids)) if is_final_stage else 1
                    )
                    full_view_batch = len(view_ids) == self.num_views
                    use_compiled_train_step = bool(
                        not debug_progress_enabled and not timing_enabled and full_view_batch and not is_final_stage
                    )

                    _emit_progress({
                        "event": "step_start",
                        "stage_index": int(stage_idx),
                        "step_index": int(step),
                        "global_step": int(global_step),
                        "steps_in_stage": int(steps),
                        "views_in_step": int(len(view_ids)),
                        "view_microbatch_size": int(final_stage_microbatch),
                        "render_height": int(render_h),
                        "render_width": int(render_w),
                        "density_due": bool(density_due),
                        "density_scheduled": bool(density_scheduled),
                        "density_frozen": bool(final_stage_density_frozen),
                        "num_gaussians": int(self.field_model.num_gaussians),
                    })
                    if verbose_progress:
                        self._progress_print(
                            f"[step-start] stage={stage_idx + 1}/{total_stages} "
                            f"step={step + 1}/{steps} global={global_step} "
                            f"views={len(view_ids)} render={render_h}x{render_w} "
                            f"N={self.field_model.num_gaussians} density_due={int(density_due)} "
                            f"density_frozen={int(final_stage_density_frozen)} "
                            f"microbatch={final_stage_microbatch}"
                        )

                    photo_loss = images.new_tensor(0.0)
                    rendered_any = False
                    stats_accum: dict[str, Tensor] | None = None
                    per_view_observations: list[DensityViewObservation] = []
                    field_total_s = 0.0
                    render_total_s = 0.0
                    aux_stats_total_s = 0.0
                    residual_stats_total_s = 0.0
                    view_total_s = 0.0
                    backward_s = 0.0
                    opt_step_s = 0.0
                    final_stage_photo_microbatch = images.new_tensor(0.0) if is_final_stage else None
                    final_stage_photo_count = 0

                    if use_compiled_train_step and not density_due:
                        with torch.profiler.record_function("pipeline.train_step_compiled"):
                            step_out = self._train_step_all_views(
                                opt,
                                stage_targets,
                                render_h,
                                render_w,
                                stage_alpha,
                                train_cfg.grad_clip,
                            )
                        loss = step_out[0].clone()
                        photo_loss = step_out[1].clone()
                        reg_loss = step_out[2].clone()
                        if lr_scheduler is not None:
                            lr_scheduler.step()
                        if not bool(step_out[3].item()):
                            self._raise_nonfinite("render_rgb", stage_idx, step, global_step)
                        if not bool(step_out[4].item()):
                            self._raise_nonfinite("observed_prediction", stage_idx, step, global_step)
                        rendered_any = True
                    else:
                        opt.zero_grad(set_to_none=True)
                        base_intr = self.intrinsics.get()
                        intr = self._scale_intrinsics(render_h, render_w)

                        for view_pos, v in enumerate(view_ids):
                            with torch.profiler.record_function("pipeline.view"):
                                if verbose_progress:
                                    self._progress_print(
                                        f"[view-start] stage={stage_idx + 1}/{total_stages} "
                                        f"step={step + 1}/{steps} view={view_pos + 1}/{len(view_ids)} "
                                        f"vid={v} render={render_h}x{render_w} N={self.field_model.num_gaussians}"
                                    )
                                _emit_progress({
                                    "event": "view_start",
                                    "stage_index": int(stage_idx),
                                    "step_index": int(step),
                                    "global_step": int(global_step),
                                    "view_index": int(v),
                                    "view_ordinal": int(view_pos),
                                    "views_in_step": int(len(view_ids)),
                                    "num_gaussians": int(self.field_model.num_gaussians),
                                })
                                view_start = time.perf_counter()
                                render_profile: dict[str, float] = {}
                                residual_stats_s = 0.0
                                tgt = stage_targets[v]
                                view_R = R_all[v].detach() if stage_idx == (total_stages - 1) else R_all[v]
                                view_t = t_all[v].detach() if stage_idx == (total_stages - 1) else t_all[v]
                                view_intr = intr.detach() if stage_idx == (total_stages - 1) else intr
                                view_base_intr = base_intr.detach() if stage_idx == (total_stages - 1) else base_intr
                                if (
                                    timing_enabled
                                    or debug_progress_enabled
                                    or (density_due and self.config.density.weak_view_reseed_budget_per_view > 0)
                                ):
                                    render = self.render_with_pose(
                                        view_R,
                                        view_t,
                                        render_h,
                                        render_w,
                                        return_aux=True,
                                        stats_mode="meta",
                                        return_prepared=density_due,
                                        profile=timing_enabled,
                                    )
                                    with torch.profiler.record_function("pipeline.observe_and_photometric_loss"):
                                        pred, photo_term = self._observe_and_photometric_loss(render["rgb"], tgt)
                                    if not torch.isfinite(render["rgb"]).all():
                                        self._raise_nonfinite(
                                            "render_rgb",
                                            stage_idx,
                                            step,
                                            global_step,
                                            {
                                                "view_index": int(v),
                                                "view_ordinal": int(view_pos),
                                            },
                                        )
                                    if not torch.isfinite(pred).all():
                                        self._raise_nonfinite(
                                            "observed_prediction",
                                            stage_idx,
                                            step,
                                            global_step,
                                            {
                                                "view_index": int(v),
                                                "view_ordinal": int(view_pos),
                                            },
                                        )
                                    if is_final_stage:
                                        photo_loss = photo_loss + photo_term.detach()
                                        assert final_stage_photo_microbatch is not None
                                        final_stage_photo_microbatch = final_stage_photo_microbatch + photo_term
                                        final_stage_photo_count += 1
                                        if (
                                            final_stage_photo_count >= final_stage_microbatch
                                            or view_pos == len(view_ids) - 1
                                        ):
                                            with torch.profiler.record_function("pipeline.photo_backward"):
                                                (final_stage_photo_microbatch / max(len(view_ids), 1)).backward()
                                            final_stage_photo_microbatch = images.new_tensor(0.0)
                                            final_stage_photo_count = 0
                                    else:
                                        photo_loss = photo_loss + photo_term
                                    render_profile = (
                                        render.get("profile", {}) if isinstance(render.get("profile"), dict) else {}
                                    )
                                    field_total_s += float(render_profile.get("field_s", 0.0))
                                    render_total_s += float(render_profile.get("render_s", 0.0))
                                    aux_stats_total_s += float(render_profile.get("aux_stats_s", 0.0))
                                    if density_due:
                                        if timing_enabled:
                                            self._sync_for_timing(images.device)
                                        residual_stats_start = time.perf_counter()
                                        with torch.profiler.record_function("pipeline.residual_stats"):
                                            residual_map = self._residual_map_for_render(pred, tgt, render_h, render_w)
                                            residual_stats = self._render_stats_from_prepared(
                                                render["_prepared_visibility"],
                                                render["_opacity"],
                                                residual_map=residual_map,
                                            )
                                        if timing_enabled:
                                            self._sync_for_timing(images.device)
                                            residual_stats_s = time.perf_counter() - residual_stats_start
                                        stats_accum = self._accumulate_render_stats(stats_accum, residual_stats)
                                    else:
                                        stats_accum = self._accumulate_render_stats(stats_accum, render["render_stats"])
                                    render_summary = self._render_stats_summary(render["render_stats"])
                                    if density_due:
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
                                                    view_index=int(v),
                                                    visible_count=int(render_summary["visible_count"]),
                                                    intersection_count=int(render_summary["intersection_count"]),
                                                    render_width=int(render_summary["render_width"]),
                                                    render_height=int(render_summary["render_height"]),
                                                ),
                                                render_stats=residual_stats,
                                                residual_map=residual_map.detach(),
                                                target_rgb=target_render.detach(),
                                                pred_rgb=pred.detach(),
                                                R_cw=view_R.detach(),
                                                t_cw=view_t.detach(),
                                                intrinsics=view_intr.detach(),
                                            )
                                        )
                                else:
                                    with torch.profiler.record_function("pipeline.training_view_forward"):
                                        if density_due:
                                            train_view = self._training_view_forward_density(
                                                view_base_intr,
                                                view_intr,
                                                view_R,
                                                view_t,
                                                tgt,
                                                render_h,
                                                render_w,
                                            )
                                            meta_offset = 3 + len(_META_STAT_KEYS)
                                            residual_stats = self._density_stats_dict(train_view[meta_offset:])
                                            stats_accum = self._accumulate_render_stats(stats_accum, residual_stats)
                                        else:
                                            train_view = self._training_view_forward(
                                                view_base_intr,
                                                view_intr,
                                                view_R,
                                                view_t,
                                                tgt,
                                                render_h,
                                                render_w,
                                            )
                                            meta_offset = 3 + len(_META_STAT_KEYS)
                                            stats_accum = self._accumulate_render_stats(
                                                stats_accum,
                                                self._meta_dict(train_view[3:meta_offset]),
                                            )
                                    photo_term = train_view[0].clone()
                                    rgb_finite = bool(train_view[1].item())
                                    pred_finite = bool(train_view[2].item())
                                    if not rgb_finite:
                                        self._raise_nonfinite(
                                            "render_rgb",
                                            stage_idx,
                                            step,
                                            global_step,
                                            {
                                                "view_index": int(v),
                                                "view_ordinal": int(view_pos),
                                            },
                                        )
                                    if not pred_finite:
                                        self._raise_nonfinite(
                                            "observed_prediction",
                                            stage_idx,
                                            step,
                                            global_step,
                                            {
                                                "view_index": int(v),
                                                "view_ordinal": int(view_pos),
                                            },
                                        )
                                    if is_final_stage:
                                        photo_loss = photo_loss + photo_term.detach()
                                        assert final_stage_photo_microbatch is not None
                                        final_stage_photo_microbatch = final_stage_photo_microbatch + photo_term
                                        final_stage_photo_count += 1
                                        if (
                                            final_stage_photo_count >= final_stage_microbatch
                                            or view_pos == len(view_ids) - 1
                                        ):
                                            with torch.profiler.record_function("pipeline.photo_backward"):
                                                (final_stage_photo_microbatch / max(len(view_ids), 1)).backward()
                                            final_stage_photo_microbatch = images.new_tensor(0.0)
                                            final_stage_photo_count = 0
                                    else:
                                        photo_loss = photo_loss + photo_term
                                    render_summary = self._render_stats_summary(
                                        self._meta_dict(train_view[3:meta_offset])
                                    )
                                    if density_due:
                                        per_view_observations.append(
                                            DensityViewObservation(
                                                coverage=DensityViewCoverage(
                                                    view_index=int(v),
                                                    visible_count=int(render_summary["visible_count"]),
                                                    intersection_count=int(render_summary["intersection_count"]),
                                                    render_width=int(render_summary["render_width"]),
                                                    render_height=int(render_summary["render_height"]),
                                                ),
                                                render_stats=residual_stats,
                                                target_rgb=tgt.detach(),
                                                R_cw=view_R.detach(),
                                                t_cw=view_t.detach(),
                                                intrinsics=view_intr.detach(),
                                            )
                                        )
                                rendered_any = True
                                residual_stats_total_s += residual_stats_s
                                view_elapsed_s = time.perf_counter() - view_start
                                view_total_s += view_elapsed_s
                                memory_snapshot = self._memory_snapshot(images.device)
                                cumulative_photo = float(photo_loss.detach().item())
                                _emit_progress({
                                    "event": "view_end",
                                    "stage_index": int(stage_idx),
                                    "step_index": int(step),
                                    "global_step": int(global_step),
                                    "view_index": int(v),
                                    "view_ordinal": int(view_pos),
                                    "views_in_step": int(len(view_ids)),
                                    "field_s": float(render_profile.get("field_s", 0.0)),
                                    "render_s": float(render_profile.get("render_s", 0.0)),
                                    "aux_stats_s": float(render_profile.get("aux_stats_s", 0.0)),
                                    "residual_stats_s": float(residual_stats_s),
                                    "view_total_s": float(view_elapsed_s),
                                    "cumulative_photo_loss": cumulative_photo,
                                    **render_summary,
                                    **memory_snapshot,
                                })
                                if verbose_progress:
                                    self._progress_print(
                                        f"[view] stage={stage_idx + 1}/{total_stages} step={step + 1}/{steps} "
                                        f"view={view_pos + 1}/{len(view_ids)} vid={v} "
                                        f"N={render_summary['gaussian_count']} visible={render_summary['visible_count']} "
                                        f"M={render_summary['intersection_count']} "
                                        f"tiles={render_summary['tiles_x']}x{render_summary['tiles_y']} "
                                        f"field={float(render_profile.get('field_s', 0.0)):.2f}s "
                                        f"render={float(render_profile.get('render_s', 0.0)):.2f}s "
                                        f"aux={float(render_profile.get('aux_stats_s', 0.0)):.2f}s "
                                        f"residual={residual_stats_s:.2f}s "
                                        f"photo_cum={cumulative_photo:.6f} "
                                        f"cuda_alloc={memory_snapshot['cuda_alloc_gib']:.2f}GiB "
                                        f"cuda_max={memory_snapshot['cuda_max_alloc_gib']:.2f}GiB"
                                    )

                        photo_loss = photo_loss / max(len(view_ids), 1)
                        if not rendered_any:
                            raise RuntimeError("No views were rendered during training step.")
                        reg_loss = self._regularization_impl(None, stage_alpha)
                        if not torch.isfinite(photo_loss):
                            self._raise_nonfinite("photo_loss", stage_idx, step, global_step)
                        if not torch.isfinite(reg_loss):
                            self._raise_nonfinite("reg_loss", stage_idx, step, global_step)
                        loss = photo_loss + reg_loss.detach() if is_final_stage else photo_loss + reg_loss
                        if not torch.isfinite(loss):
                            self._raise_nonfinite("loss", stage_idx, step, global_step)

                        backward_start = time.perf_counter()
                        if timing_enabled:
                            self._sync_for_timing(images.device)
                            backward_start = time.perf_counter()
                        with torch.profiler.record_function("pipeline.backward"):
                            if is_final_stage:
                                reg_loss.backward()
                            else:
                                loss.backward()
                        if timing_enabled:
                            self._sync_for_timing(images.device)
                            backward_s = time.perf_counter() - backward_start
                        if is_final_stage:
                            for param in self.camera_model.parameters():
                                param.grad = None
                        if train_cfg.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(self.parameters(), train_cfg.grad_clip)
                        opt_step_start = time.perf_counter()
                        if timing_enabled:
                            self._sync_for_timing(images.device)
                            opt_step_start = time.perf_counter()
                        with torch.profiler.record_function("pipeline.optimizer_step"):
                            opt.step()
                        if lr_scheduler is not None:
                            lr_scheduler.step()
                        if timing_enabled:
                            self._sync_for_timing(images.device)
                            opt_step_s = time.perf_counter() - opt_step_start

                    self.field_model.enforce_protection(
                        global_step,
                        float(self.config.density.weak_view_reseed_min_opacity),
                    )
                    self.field_model.enforce_scale_floor()

                    if not torch.isfinite(photo_loss):
                        self._raise_nonfinite("photo_loss", stage_idx, step, global_step)
                    if not torch.isfinite(reg_loss):
                        self._raise_nonfinite("reg_loss", stage_idx, step, global_step)
                    if not torch.isfinite(loss):
                        self._raise_nonfinite("loss", stage_idx, step, global_step)

                    gradient_stats = self._gradient_stats_report()
                    gradient_nonfinite = {
                        name: int(stats["nonfinite"])
                        for name, stats in gradient_stats.items()
                        if int(stats["nonfinite"]) > 0
                    }
                    _emit_progress({
                        "event": "gradient_stats",
                        "stage_index": int(stage_idx),
                        "step_index": int(step),
                        "global_step": int(global_step),
                        "gradient_stats": gradient_stats,
                    })
                    if gradient_nonfinite:
                        self._raise_nonfinite(
                            "gradients_after_backward",
                            stage_idx,
                            step,
                            global_step,
                            {
                                "nonfinite_grads": gradient_nonfinite,
                                "gradient_stats": gradient_stats,
                            },
                        )
                    nonfinite_params = self._parameter_nonfinite_report()
                    if nonfinite_params:
                        self._raise_nonfinite(
                            "parameters_after_opt_step",
                            stage_idx,
                            step,
                            global_step,
                            {"nonfinite_params": nonfinite_params},
                        )

                    density_event = DensityControlResult.skipped(self.field_model.num_gaussians)
                    density_s = 0.0
                    if density_due:
                        density_start = time.perf_counter()
                        with torch.profiler.record_function("pipeline.density_control"):
                            density_event = apply_density_control(
                                self.field_model,
                                self.config.density,
                                global_step,
                                stage_index=stage_idx,
                                total_stages=total_stages,
                                render_stats=stats_accum,
                                per_view_observations=per_view_observations,
                            )
                        density_s = time.perf_counter() - density_start
                    if density_event.changed:
                        if _should_clear_renderer_cache(self.config.render.backend):
                            clear_renderer_caches(backend=self.config.render.backend)
                        self.field_model.enforce_scale_floor()
                        opt = self._rebuild_optimizer_after_density(
                            opt,
                            train_cfg,
                            density_event.survivor_sources,
                            int(density_event.appended_count),
                        )
                        self._debug_optimizer = opt
                    if density_event.ran:
                        density_summary = density_event.debug_dict()
                        if is_final_stage and int(self.config.density.freeze_after_stable_events) > 0:
                            if _density_event_is_stable_for_freeze(density_event, self.config.density):
                                final_stage_density_stable_events += 1
                            else:
                                final_stage_density_stable_events = 0
                            if not final_stage_density_frozen and final_stage_density_stable_events >= int(
                                self.config.density.freeze_after_stable_events
                            ):
                                final_stage_density_frozen = True
                                _emit_progress({
                                    "event": "density_frozen",
                                    "stage_index": int(stage_idx),
                                    "step_index": int(step),
                                    "global_step": int(global_step),
                                    "stable_events": int(final_stage_density_stable_events),
                                    "num_gaussians": int(self.field_model.num_gaussians),
                                })
                        event_record = {
                            "global_step": int(global_step),
                            "stage_index": int(stage_idx),
                            "step_index": int(step),
                            "summary": density_summary,
                            "pruned": int(density_event.pruned),
                            "split": int(density_event.split),
                            "cloned": int(density_event.cloned),
                            "reseeded": int(density_event.reseeded),
                            "before": int(density_event.before),
                            "after": int(density_event.after),
                            "prune_protected": bool(density_summary.get("prune_protected", False)),
                            "coverage_weights": list(density_summary.get("coverage_weights", [])),
                            "visible_fraction_of_best": list(density_summary.get("visible_fraction_of_best", [])),
                            "intersection_fraction_of_best": list(
                                density_summary.get("intersection_fraction_of_best", [])
                            ),
                            "weak_view_indices": list(density_summary.get("weak_view_indices", [])),
                            "reseed_view_indices": list(density_summary.get("reseed_view_indices", [])),
                            "view_coverages": list(density_summary.get("view_coverages", [])),
                        }
                        history["density_events"].append(event_record)
                        emit_density_event(
                            event_record,
                            jsonl_path=density_event_log_path,
                            callback=density_event_callback,
                        )
                    if density_event.changed:
                        post_density_preflight = self._run_projection_preflight(
                            view_ids=list(range(self.num_views)),
                            render_h=render_h,
                            render_w=render_w,
                            stage_index=stage_idx,
                            step_index=step,
                            global_step=global_step,
                            reason="post_density",
                            progress_event_callback=progress_event_callback,
                            fail_on_error=True,
                        )
                        self._reserve_helion_intersection_capacity_from_preflight(post_density_preflight)

                    step_total_s = time.perf_counter() - step_start_time
                    stage_elapsed_s = time.perf_counter() - stage_start_time
                    completed_stage_steps = max(1, int(step - step_start_index + 1))
                    eta_stage_s = (stage_elapsed_s / float(completed_stage_steps)) * float(steps - step - 1)
                    memory_snapshot = self._memory_snapshot(images.device)
                    _emit_progress({
                        "event": "step_end",
                        "stage_index": int(stage_idx),
                        "step_index": int(step),
                        "global_step": int(global_step),
                        "render_height": int(render_h),
                        "render_width": int(render_w),
                        "steps_in_stage": int(steps),
                        "views_in_step": int(len(view_ids)),
                        "view_microbatch_size": int(final_stage_microbatch),
                        "density_due": bool(density_due),
                        "density_scheduled": bool(density_scheduled),
                        "density_frozen": bool(final_stage_density_frozen),
                        "loss": float(loss.detach().item()),
                        "photo_loss": float(photo_loss.detach().item()),
                        "reg_loss": float(reg_loss.detach().item()),
                        "num_gaussians": int(self.field_model.num_gaussians),
                        "view_total_s": float(view_total_s),
                        "field_total_s": float(field_total_s),
                        "render_total_s": float(render_total_s),
                        "aux_stats_total_s": float(aux_stats_total_s),
                        "residual_stats_total_s": float(residual_stats_total_s),
                        "backward_s": float(backward_s),
                        "opt_step_s": float(opt_step_s),
                        "density_s": float(density_s),
                        "step_total_s": float(step_total_s),
                        "eta_stage_s": float(eta_stage_s),
                        **memory_snapshot,
                    })

                    history["loss"].append(float(loss.detach().item()))
                    history["photo"].append(float(photo_loss.detach().item()))
                    history["reg"].append(float(reg_loss.detach().item()))
                    history["num_gaussians"].append(float(self.field_model.num_gaussians))

                    if verbose_progress or (
                        train_cfg.print_every > 0 and ((step + 1) % train_cfg.print_every == 0 or step == 0)
                    ):
                        density_msg = ""
                        if density_event.changed:
                            density_msg = (
                                f"  density(pruned={density_event.pruned},"
                                f" split={density_event.split},"
                                f" cloned={density_event.cloned}, N={density_event.after})"
                            )
                        self._progress_print(
                            f"[stage {stage_idx + 1}/{total_stages} | scale={stage_scale:.2f} | render={render_h}x{render_w}] "
                            f"step {step + 1:05d}/{steps:05d}  "
                            f"loss={history['loss'][-1]:.6f}  "
                            f"photo={history['photo'][-1]:.6f}  "
                            f"reg={history['reg'][-1]:.6f}  "
                            f"N={self.field_model.num_gaussians}  "
                            f"views_in_step={len(view_ids)}  "
                            f"microbatch={final_stage_microbatch}  "
                            f"views={view_total_s:.2f}s(field={field_total_s:.2f}s render={render_total_s:.2f}s aux={aux_stats_total_s:.2f}s residual={residual_stats_total_s:.2f}s)  "
                            f"backward={backward_s:.2f}s  "
                            f"opt={opt_step_s:.2f}s  "
                            f"density={density_s:.2f}s  "
                            f"total={step_total_s:.2f}s  "
                            f"eta={self._format_eta(eta_stage_s)}  "
                            f"cuda_alloc={memory_snapshot['cuda_alloc_gib']:.2f}GiB  "
                            f"cuda_max={memory_snapshot['cuda_max_alloc_gib']:.2f}GiB"
                            f"{density_msg}"
                        )
                    if density_event.ran:
                        debug = density_event.debug_dict()
                        split_top = ",".join(str(x["index"]) for x in debug.get("split_top", []))
                        clone_top = ",".join(str(x["index"]) for x in debug.get("clone_top", []))
                        self._progress_print(
                            f"[density step={global_step}] "
                            f"vis_mean={debug.get('visibility_mean', 0.0):.6f} "
                            f"vis_max={debug.get('visibility_max', 0.0):.6f} "
                            f"res_mean={debug.get('residual_mean', 0.0):.6f} "
                            f"res_max={debug.get('residual_max', 0.0):.6f} "
                            f"peak_mean={debug.get('peak_error_mean', 0.0):.6f} "
                            f"peak_max={debug.get('peak_error_max', 0.0):.6f} "
                            f"trans_mean={debug.get('transmittance_mean', 0.0):.6f} "
                            f"split_top=[{split_top}] "
                            f"clone_top=[{clone_top}]"
                        )
                    if step_callback is not None:
                        step_callback(stage_idx, step, global_step)
                    if is_final_stage:
                        final_stage_loss_window.append(float(loss.detach().item()))
                    global_step += 1
                    if is_final_stage and _should_early_stop_final_stage(
                        train_cfg,
                        final_stage_loss_window,
                        step_index=step,
                        density_frozen=final_stage_density_frozen,
                    ):
                        final_stage_early_stopped = True
                        _emit_progress({
                            "event": "final_stage_early_stop",
                            "stage_index": int(stage_idx),
                            "step_index": int(step),
                            "global_step": int(global_step - 1),
                            "stable_window": list(final_stage_loss_window),
                            "num_gaussians": int(self.field_model.num_gaussians),
                        })
                        if verbose_progress:
                            self._progress_print(
                                f"[final-stage-early-stop] stage={stage_idx + 1}/{total_stages} "
                                f"step={step + 1}/{steps} global={global_step - 1} "
                                f"window={list(final_stage_loss_window)} "
                                f"N={self.field_model.num_gaussians}"
                            )
                        break

            _emit_progress({
                "event": "stage_end",
                "stage_index": int(stage_idx),
                "stage_scale": float(stage_scale),
                "steps": int(steps),
                "stage_elapsed_s": float(time.perf_counter() - stage_start_time),
                "num_gaussians": int(self.field_model.num_gaussians),
                "early_stopped": bool(final_stage_early_stopped),
            })

        return history


def fit_posefree_gaussian_scene(
    images: Tensor,
    intrinsics: Tensor | None = None,
    config: PoseFreeGaussianConfig | None = None,
    density_event_log_path: str | Path | None = None,
    density_event_callback: Callable[[dict], None] | None = None,
    verbose_progress: bool = False,
    progress_log_path: str | Path | None = None,
    progress_event_callback: Callable[[dict], None] | None = None,
    synchronize_progress_timing: bool = False,
    step_callback: Callable[[int, int, int], None] | None = None,
) -> tuple[PoseFreeGaussianSR, dict[str, list]]:
    pipeline = PoseFreeGaussianSR.from_images(images, intrinsics=intrinsics, config=config)
    history = pipeline.fit(
        images,
        density_event_log_path=density_event_log_path,
        density_event_callback=density_event_callback,
        verbose_progress=verbose_progress,
        progress_log_path=progress_log_path,
        progress_event_callback=progress_event_callback,
        synchronize_progress_timing=synchronize_progress_timing,
        step_callback=step_callback,
    )
    return pipeline, history


def render_arbitrary_scale(
    pipeline: PoseFreeGaussianSR,
    view_index: int = 0,
    scale: float = 4.0,
) -> Tensor:
    render = pipeline.render_view(view_index=view_index, scale=scale, return_aux=False)
    return render["rgb"]  # type: ignore[return-value]


__all__ = [
    "PoseFreeGaussianSR",
    "fit_posefree_gaussian_scene",
    "render_arbitrary_scale",
]
