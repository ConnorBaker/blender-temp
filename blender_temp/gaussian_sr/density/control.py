from collections.abc import Mapping, Sequence
from dataclasses import replace

import torch
from torch import Tensor

from ..field import CanonicalGaussianField
from ..posefree_config import DensityControlConfig
from .coverage import _view_aware_context
from .debug import build_density_debug_summary
from .reseeding import _reseed_for_observation
from .scoring import _combine_density_score, _density_score_terms, _norm, gradient_score, scale_score
from .selection import compute_prune_keep_mask, select_clone_indices, select_split_indices
from .types import (
    DensityControlResult,
    DensityScoreTerms,
    DensityViewObservation,
    NormalizedRenderStats,
    ViewAwareDensityContext,
    normalize_render_stats,
)


def should_run_density_control(step: int, cfg: DensityControlConfig) -> bool:
    if not cfg.enabled:
        return False
    if step < int(cfg.start_step):
        return False
    every = int(cfg.every_steps)
    if every <= 0:
        return False
    return (step - int(cfg.start_step)) % every == 0


def should_run_density_control_for_stage(
    step: int,
    cfg: DensityControlConfig,
    stage_index: int,
    total_stages: int,
) -> bool:
    if cfg.disable_final_stage and stage_index == (max(int(total_stages), 1) - 1):
        return False
    if not cfg.enabled:
        return False
    if step < int(cfg.start_step):
        return False

    every = int(cfg.every_steps)
    is_final_stage = stage_index == (max(int(total_stages), 1) - 1)
    if is_final_stage and cfg.final_stage_every_steps is not None:
        every = int(cfg.final_stage_every_steps)
    if every <= 0:
        return False
    return (step - int(cfg.start_step)) % every == 0


def apply_density_control(
    field_model: CanonicalGaussianField,
    cfg: DensityControlConfig,
    step: int,
    stage_index: int | None = None,
    total_stages: int | None = None,
    render_stats: Mapping[str, Tensor] | None = None,
    per_view_observations: Sequence[DensityViewObservation] | None = None,
) -> DensityControlResult:
    should_run = (
        should_run_density_control_for_stage(step, cfg, stage_index, total_stages)
        if stage_index is not None and total_stages is not None
        else should_run_density_control(step, cfg)
    )
    if not should_run:
        return DensityControlResult.skipped(field_model.num_gaussians)

    allow_reseed = stage_index is None or total_stages is None or stage_index == (max(int(total_stages), 1) - 1)

    before = field_model.num_gaussians
    device = field_model.means3d.device
    dtype = field_model.means3d.dtype
    opacity = torch.sigmoid(field_model.opacity_logit.detach()[:before, 0])
    grad = gradient_score(field_model)
    scale = scale_score(field_model)
    source_indices = torch.arange(before, device=device, dtype=torch.long)
    view_ctx = _view_aware_context(
        per_view_observations,
        count=before,
        device=device,
        dtype=dtype,
        cfg=cfg,
    )
    if view_ctx is None:
        stats = normalize_render_stats(render_stats, before, device, dtype)
        score_terms = _density_score_terms(grad, scale, stats, cfg)
    else:
        stats = view_ctx.weighted_stats
        score_terms = _density_score_terms(grad, scale, stats, cfg)
        score_terms = DensityScoreTerms(
            grad=score_terms.grad,
            visibility=score_terms.visibility,
            min_visibility=_norm(view_ctx.min_contrib),
            residual=score_terms.residual,
            peak_error=score_terms.peak_error,
            trans=score_terms.trans,
            scale=score_terms.scale,
            inv_scale=score_terms.inv_scale,
        )

    split_score = _combine_density_score(cfg, score_terms, use_inverse_scale=False)
    clone_score = _combine_density_score(cfg, score_terms, use_inverse_scale=True)
    rescue_only = view_ctx is not None and bool(view_ctx.weak_view_indices)
    selection_cfg = (
        replace(
            cfg,
            split_topk=min(int(cfg.split_topk), int(cfg.weak_view_split_topk)),
            clone_topk=min(int(cfg.clone_topk), int(cfg.weak_view_clone_topk)),
        )
        if rescue_only
        else cfg
    )

    if view_ctx is not None and not view_ctx.can_prune:
        keep = torch.ones_like(opacity, dtype=torch.bool)
    else:
        keep = compute_prune_keep_mask(opacity, stats.contrib, cfg)
        keep = keep | field_model.protected_mask(step)
    pruned = int((~keep).sum().item())
    if pruned > 0:
        field_model.prune_keep_mask(keep)
        source_indices = source_indices[keep]
        opacity = opacity[keep]
        grad = grad[keep]
        scale = scale[keep]
        stats = stats.masked(keep)
        split_score = split_score[keep]
        clone_score = clone_score[keep]
        if view_ctx is not None:
            view_ctx = replace(
                view_ctx,
                weighted_stats=view_ctx.weighted_stats.masked(keep),
                weak_view_stats=view_ctx.weak_view_stats.masked(keep),
                min_contrib=view_ctx.min_contrib[keep],
                min_trans=view_ctx.min_trans[keep],
            )

    survivor_sources = source_indices.clone()

    split_idx = select_split_indices(
        opacity,
        grad,
        scale,
        stats,
        selection_cfg,
        field_model.num_gaussians,
        score=split_score,
        weak_visibility=(
            None if view_ctx is None or not view_ctx.weak_view_indices else view_ctx.weak_view_stats.contrib
        ),
        weak_trans=(None if view_ctx is None or not view_ctx.weak_view_indices else view_ctx.weak_view_stats.avg_trans),
    )
    split_count = int(split_idx.numel())
    if split_count > 0:
        split_count = int(
            field_model.split_gaussians(
                split_idx,
                shrink_factor=float(cfg.split_shrink_factor),
                offset_scale=float(cfg.split_offset_scale),
            )
        )

    clone_idx = select_clone_indices(
        opacity,
        grad,
        scale,
        stats,
        selection_cfg,
        field_model.num_gaussians,
        exclude=split_idx,
        score=clone_score,
        weak_visibility=(
            None if view_ctx is None or not view_ctx.weak_view_indices else view_ctx.weak_view_stats.contrib
        ),
        weak_trans=(None if view_ctx is None or not view_ctx.weak_view_indices else view_ctx.weak_view_stats.avg_trans),
    )
    clone_count = int(clone_idx.numel())
    if clone_count > 0:
        clone_count = int(field_model.clone_gaussians(clone_idx, jitter_scale=float(cfg.clone_jitter_scale)))

    reseeded = 0
    if allow_reseed and view_ctx is not None and view_ctx.reseed_observations:
        for observation in view_ctx.reseed_observations:
            reseeded += int(_reseed_for_observation(field_model, observation, cfg, step=step))

    after = field_model.num_gaussians
    debug = build_density_debug_summary(
        opacity,
        grad,
        scale,
        stats,
        view_ctx,
        split_idx,
        clone_idx,
        split_score,
        clone_score,
        cfg,
    )
    return DensityControlResult(
        ran=True,
        changed=bool(pruned > 0 or split_count > 0 or clone_count > 0 or reseeded > 0),
        pruned=pruned,
        split=split_count,
        cloned=clone_count,
        before=before,
        after=after,
        appended_count=int(after - survivor_sources.shape[0]),
        reseeded=reseeded,
        survivor_sources=survivor_sources,
        debug=debug,
    )
