import torch
from torch import Tensor

from ..posefree_config import DensityControlConfig
from .types import (
    DensityDebugEntry,
    DensityDebugSummary,
    NormalizedRenderStats,
    ViewAwareDensityContext,
)


def _topk_debug(indices: Tensor, score: Tensor, stats: NormalizedRenderStats, topk: int) -> list[DensityDebugEntry]:
    if indices.numel() == 0 or topk <= 0:
        return []
    vals = score[indices]
    order = torch.argsort(vals, descending=True)
    sel = indices[order[:topk]]
    out: list[DensityDebugEntry] = []
    for i in sel.tolist():
        peak_bin = int(stats.error_map[i].argmax().item()) if stats.error_map.ndim == 2 else 0
        out.append(
            DensityDebugEntry(
                index=int(i),
                score=float(score[i].item()),
                peak_bin=peak_bin,
                peak_error=float(stats.peak_error[i].item()),
                residual=float(stats.avg_residual[i].item()),
                visibility=float(stats.contrib[i].item()),
            )
        )
    return out


def build_density_debug_summary(
    opacity: Tensor,
    grad: Tensor,
    scale: Tensor,
    stats: NormalizedRenderStats,
    view_ctx: ViewAwareDensityContext | None,
    split_idx: Tensor,
    clone_idx: Tensor,
    split_score: Tensor,
    clone_score: Tensor,
    cfg: DensityControlConfig,
) -> DensityDebugSummary:
    return DensityDebugSummary(
        visibility_mean=float(stats.contrib.mean().item()),
        visibility_max=float(stats.contrib.max().item()),
        residual_mean=float(stats.avg_residual.mean().item()),
        residual_max=float(stats.avg_residual.max().item()),
        peak_error_mean=float(stats.peak_error.mean().item()),
        peak_error_max=float(stats.peak_error.max().item()),
        transmittance_mean=float(stats.avg_trans.mean().item()),
        screen_error_bins=int(stats.error_map.shape[-1]) if stats.error_map.ndim == 2 else 0,
        gradient_mean=float(grad.mean().item()),
        scale_mean=float(scale.mean().item()),
        opacity_mean=float(opacity.mean().item()),
        prune_protected=bool(view_ctx is not None and not view_ctx.can_prune),
        coverage_weights=([float(value.item()) for value in view_ctx.coverage_weights] if view_ctx is not None else []),
        visible_fraction_of_best=(
            [float(value.item()) for value in view_ctx.visible_fraction_of_best] if view_ctx is not None else []
        ),
        intersection_fraction_of_best=(
            [float(value.item()) for value in view_ctx.intersection_fraction_of_best] if view_ctx is not None else []
        ),
        weak_view_indices=list(view_ctx.weak_view_indices) if view_ctx is not None else [],
        reseed_view_indices=list(view_ctx.reseed_view_indices) if view_ctx is not None else [],
        view_coverages=list(view_ctx.view_coverages) if view_ctx is not None else [],
        split_top=_topk_debug(split_idx, split_score, stats, int(cfg.debug_topk)),
        clone_top=_topk_debug(clone_idx, clone_score, stats, int(cfg.debug_topk)),
    )
