from __future__ import annotations

from torch import Tensor

from .helion_gsplat_renderer import (
    clear_helion_kernel_cache,
    render_gaussians_helion,
    render_projection_meta_helion,
    render_stats_helion,
    render_stats_prepared_helion,
    render_values_helion,
    render_visibility_meta_helion,
    reserve_helion_intersection_capacity,
    prepare_visibility_helion,
)
from .warp_gsplat_autograd import PreparedVisibility
from .warp_gsplat_contracts import RasterConfig
from .warp_gsplat_renderer import (
    clear_warp_launch_cache,
    render_gaussians_warp,
    render_projection_meta_warp,
    render_stats_prepared_warp,
    render_stats_warp,
    render_values_warp,
    render_visibility_meta_warp,
    prepare_visibility_warp,
)


def clear_renderer_caches(*, backend: str | None = None) -> None:
    if backend in (None, "warp"):
        clear_warp_launch_cache()
    if backend in (None, "helion"):
        clear_helion_kernel_cache()


def reserve_renderer_intersection_capacity(
    *,
    backend: str,
    device,
    width: int,
    height: int,
    required_count: int,
) -> int | None:
    if backend == "helion":
        return reserve_helion_intersection_capacity(
            device=device,
            width=width,
            height=height,
            required_count=required_count,
        )
    return None


def prepare_visibility(
    means: Tensor,
    quat: Tensor,
    scale: Tensor,
    viewmat: Tensor,
    K: Tensor,
    width: int,
    height: int,
    cfg: RasterConfig | None = None,
    requires_grad: bool = False,
    active_count: int | None = None,
) -> PreparedVisibility:
    cfg = cfg or RasterConfig()
    if cfg.backend == "helion":
        return prepare_visibility_helion(
            means=means,
            quat=quat,
            scale=scale,
            viewmat=viewmat,
            K=K,
            width=width,
            height=height,
            cfg=cfg,
            requires_grad=requires_grad,
            active_count=active_count,
        )
    return prepare_visibility_warp(
        means=means,
        quat=quat,
        scale=scale,
        viewmat=viewmat,
        K=K,
        width=width,
        height=height,
        cfg=cfg,
        requires_grad=requires_grad,
        active_count=active_count,
    )


def render_values(*args, cfg: RasterConfig | None = None, **kwargs):
    cfg = cfg or RasterConfig()
    if cfg.backend == "helion":
        return render_values_helion(*args, cfg=cfg, **kwargs)
    return render_values_warp(*args, cfg=cfg, **kwargs)


def render_stats_prepared(prepared: PreparedVisibility, opacity: Tensor, *, cfg: RasterConfig | None = None, **kwargs):
    cfg = cfg or RasterConfig()
    if cfg.backend == "helion":
        return render_stats_prepared_helion(prepared, opacity, cfg=cfg, **kwargs)
    return render_stats_prepared_warp(prepared, opacity, cfg=cfg, **kwargs)


def render_stats(opacity: Tensor, *, cfg: RasterConfig | None = None, **kwargs):
    cfg = cfg or RasterConfig()
    if cfg.backend == "helion":
        return render_stats_helion(opacity, cfg=cfg, **kwargs)
    return render_stats_warp(opacity, cfg=cfg, **kwargs)


def render_visibility_meta(*, cfg: RasterConfig | None = None, **kwargs):
    cfg = cfg or RasterConfig()
    if cfg.backend == "helion":
        return render_visibility_meta_helion(cfg=cfg, **kwargs)
    return render_visibility_meta_warp(cfg=cfg, **kwargs)


def render_projection_meta(*, cfg: RasterConfig | None = None, **kwargs):
    cfg = cfg or RasterConfig()
    if cfg.backend == "helion":
        return render_projection_meta_helion(cfg=cfg, **kwargs)
    return render_projection_meta_warp(cfg=cfg, **kwargs)


def render_gaussians(*args, cfg: RasterConfig | None = None, **kwargs):
    cfg = cfg or RasterConfig()
    if cfg.backend == "helion":
        return render_gaussians_helion(*args, cfg=cfg, **kwargs)
    return render_gaussians_warp(*args, cfg=cfg, **kwargs)


__all__ = [
    "PreparedVisibility",
    "RasterConfig",
    "clear_renderer_caches",
    "reserve_renderer_intersection_capacity",
    "prepare_visibility",
    "render_values",
    "render_stats_prepared",
    "render_stats",
    "render_visibility_meta",
    "render_projection_meta",
    "render_gaussians",
]
