import torch
from torch import Tensor

from ..posefree_config import DensityControlConfig
from .scoring import _combine_density_score, _density_score_terms, _quantile_threshold
from .types import NormalizedRenderStats


def compute_prune_keep_mask(opacity: Tensor, visibility: Tensor, cfg: DensityControlConfig) -> Tensor:
    keep = (opacity >= float(cfg.opacity_prune_threshold)) | (visibility >= float(cfg.prune_visibility_threshold))
    min_keep = min(int(cfg.min_gaussians), opacity.shape[0])
    if int(keep.sum().item()) < min_keep:
        priority = opacity + visibility
        topk = torch.topk(priority, k=min_keep, largest=True).indices
        keep = torch.zeros_like(keep, dtype=torch.bool)
        keep[topk] = True
    return keep


def _topk_candidates(mask: Tensor, score: Tensor, k: int) -> Tensor:
    idx = mask.nonzero(as_tuple=False).squeeze(-1)
    if idx.numel() == 0 or k <= 0:
        return torch.empty(0, device=score.device, dtype=torch.long)
    if idx.numel() > k:
        local = torch.topk(score[idx], k=k, largest=True).indices
        idx = idx[local]
    return idx.to(torch.long)


def select_split_indices(
    opacity: Tensor,
    grad_score: Tensor,
    scale: Tensor,
    stats: NormalizedRenderStats,
    cfg: DensityControlConfig,
    current_count: int,
    score: Tensor | None = None,
    weak_visibility: Tensor | None = None,
    weak_trans: Tensor | None = None,
) -> Tensor:
    remaining = max(int(cfg.max_gaussians) - int(current_count), 0)
    limit = min(int(cfg.split_topk), remaining)
    if limit <= 0:
        return torch.empty(0, device=opacity.device, dtype=torch.long)

    large_thresh = _quantile_threshold(scale, float(cfg.split_scale_quantile))
    mask = (
        (opacity >= float(cfg.densify_opacity_min))
        & (grad_score >= float(cfg.grad_threshold))
        & (scale >= large_thresh)
        & (stats.contrib >= float(cfg.densify_visibility_threshold))
        & (stats.avg_trans >= float(cfg.split_transmittance_threshold))
    )
    if weak_visibility is not None:
        mask = mask & (weak_visibility >= float(cfg.densify_visibility_threshold))
    if weak_trans is not None:
        mask = mask & (weak_trans >= float(cfg.split_transmittance_threshold))
    split_score = (
        score
        if score is not None
        else _combine_density_score(
            cfg,
            _density_score_terms(grad_score, scale, stats, cfg),
            use_inverse_scale=False,
        )
    )
    return _topk_candidates(mask, split_score, limit)


def select_clone_indices(
    opacity: Tensor,
    grad_score: Tensor,
    scale: Tensor,
    stats: NormalizedRenderStats,
    cfg: DensityControlConfig,
    current_count: int,
    exclude: Tensor | None = None,
    score: Tensor | None = None,
    weak_visibility: Tensor | None = None,
    weak_trans: Tensor | None = None,
) -> Tensor:
    remaining = max(int(cfg.max_gaussians) - int(current_count), 0)
    limit = min(int(cfg.clone_topk), remaining)
    if limit <= 0:
        return torch.empty(0, device=opacity.device, dtype=torch.long)

    small_thresh = _quantile_threshold(scale, float(cfg.clone_scale_quantile))
    mask = (
        (opacity >= float(cfg.densify_opacity_min))
        & (grad_score >= float(cfg.grad_threshold))
        & (scale <= small_thresh)
        & (stats.contrib >= float(cfg.densify_visibility_threshold))
        & (stats.avg_trans >= float(cfg.clone_transmittance_threshold))
    )
    if weak_visibility is not None:
        mask = mask & (weak_visibility >= float(cfg.densify_visibility_threshold))
    if weak_trans is not None:
        mask = mask & (weak_trans >= float(cfg.clone_transmittance_threshold))
    if exclude is not None and exclude.numel() > 0:
        mask = mask.clone()
        mask[exclude] = False

    clone_score = (
        score
        if score is not None
        else _combine_density_score(
            cfg,
            _density_score_terms(grad_score, scale, stats, cfg),
            use_inverse_scale=True,
        )
    )
    return _topk_candidates(mask, clone_score, limit)
