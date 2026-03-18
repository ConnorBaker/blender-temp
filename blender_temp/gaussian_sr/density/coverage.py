from collections.abc import Sequence

import torch
from torch import Tensor

from ..posefree_config import DensityControlConfig
from .types import (
    DensityViewCoverage,
    DensityViewObservation,
    NormalizedRenderStats,
    ViewAwareDensityContext,
    normalize_render_stats,
)


def _coverage_weight_tensor(
    view_coverages: Sequence[DensityViewCoverage],
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if not view_coverages:
        return torch.ones(0, device=device, dtype=dtype)
    visible = torch.tensor(
        [max(int(coverage.visible_count), 0) for coverage in view_coverages],
        device=device,
        dtype=dtype,
    )
    safe_visible = visible.clamp_min(1.0)
    weights = safe_visible.max().clamp_min(1.0) / safe_visible
    return weights / weights.mean().clamp_min(1.0e-8)


def _coverage_fraction_tensors(
    view_coverages: Sequence[DensityViewCoverage],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    if not view_coverages:
        empty = torch.ones(0, device=device, dtype=dtype)
        return empty, empty
    visible = torch.tensor(
        [max(int(coverage.visible_count), 0) for coverage in view_coverages],
        device=device,
        dtype=dtype,
    )
    intersections = torch.tensor(
        [max(int(coverage.intersection_count), 0) for coverage in view_coverages],
        device=device,
        dtype=dtype,
    )
    visible_fraction = visible / visible.max().clamp_min(1.0)
    intersection_fraction = intersections / intersections.max().clamp_min(1.0)
    return visible_fraction, intersection_fraction


def _weak_view_indices(
    view_coverages: Sequence[DensityViewCoverage],
    *,
    visible_fraction_of_best: Tensor,
    intersection_fraction_of_best: Tensor,
    error_fraction_of_worst: Tensor | None,
    cfg: DensityControlConfig,
) -> tuple[int, ...]:
    min_visible_abs = int(cfg.min_view_visible_gaussians)
    min_intersections_abs = int(cfg.min_view_intersection_count)
    min_visible_rel = max(float(cfg.min_view_visible_fraction_of_best), float(cfg.coverage_floor_visible_fraction))
    min_intersections_rel = max(
        float(cfg.min_view_intersection_fraction_of_best),
        float(cfg.coverage_floor_intersection_fraction),
    )
    weak_views: list[int] = []
    for idx, coverage in enumerate(view_coverages):
        visible_too_low = (
            coverage.visible_count < min_visible_abs or float(visible_fraction_of_best[idx].item()) < min_visible_rel
        )
        intersections_too_low = (
            coverage.intersection_count < min_intersections_abs
            or float(intersection_fraction_of_best[idx].item()) < min_intersections_rel
        )
        error_high = True
        if error_fraction_of_worst is not None and error_fraction_of_worst.numel() > idx:
            error_high = float(error_fraction_of_worst[idx].item()) >= float(cfg.weak_view_error_fraction_of_worst)
        if (visible_too_low or intersections_too_low) and error_high:
            weak_views.append(int(coverage.view_index))
    return tuple(weak_views)


def _reseed_view_indices(
    view_coverages: Sequence[DensityViewCoverage],
    *,
    visible_fraction_of_best: Tensor,
    intersection_fraction_of_best: Tensor,
    error_fraction_of_worst: Tensor | None,
    weak_view_indices: Sequence[int],
    cfg: DensityControlConfig,
) -> tuple[int, ...]:
    if int(cfg.weak_view_reseed_budget_per_view) <= 0:
        return tuple()

    trigger_rel = max(
        float(cfg.weak_view_reseed_trigger_fraction_of_best),
        float(cfg.min_view_visible_fraction_of_best),
        float(cfg.min_view_intersection_fraction_of_best),
        float(cfg.coverage_floor_visible_fraction),
        float(cfg.coverage_floor_intersection_fraction),
    )
    reseed_views: list[int] = []
    seen: set[int] = set()
    for view_index in weak_view_indices:
        view_id = int(view_index)
        reseed_views.append(view_id)
        seen.add(view_id)
    for idx, coverage in enumerate(view_coverages):
        view_id = int(coverage.view_index)
        if view_id in seen:
            continue
        visible_lagging = float(visible_fraction_of_best[idx].item()) < trigger_rel
        intersections_lagging = float(intersection_fraction_of_best[idx].item()) < trigger_rel
        error_high = True
        if error_fraction_of_worst is not None and error_fraction_of_worst.numel() > idx:
            error_high = float(error_fraction_of_worst[idx].item()) >= float(cfg.weak_view_error_fraction_of_worst)
        if (visible_lagging or intersections_lagging) and error_high:
            reseed_views.append(view_id)
            seen.add(view_id)
    return tuple(reseed_views)


def _weighted_render_stats(
    stats_list: Sequence[NormalizedRenderStats],
    weights: Tensor,
    *,
    count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> NormalizedRenderStats:
    if not stats_list:
        return NormalizedRenderStats.zeros(count, device, dtype)
    if weights.numel() == 0:
        weights = torch.ones(len(stats_list), device=device, dtype=dtype)
    w = weights / weights.sum().clamp_min(1.0e-8)
    contrib = torch.zeros(count, device=device, dtype=dtype)
    hits = torch.zeros(count, device=device, dtype=dtype)
    avg_trans = torch.zeros(count, device=device, dtype=dtype)
    avg_contrib = torch.zeros(count, device=device, dtype=dtype)
    residual = torch.zeros(count, device=device, dtype=dtype)
    avg_residual = torch.zeros(count, device=device, dtype=dtype)
    peak_error = torch.zeros(count, device=device, dtype=dtype)
    error_map = torch.zeros(count, stats_list[0].error_map.shape[-1], device=device, dtype=dtype)
    for idx, stats in enumerate(stats_list):
        weight = w[idx]
        contrib = contrib + weight * stats.contrib
        hits = hits + weight * stats.hits
        avg_trans = avg_trans + weight * stats.avg_trans
        avg_contrib = avg_contrib + weight * stats.avg_contrib
        residual = residual + weight * stats.residual
        avg_residual = avg_residual + weight * stats.avg_residual
        peak_error = peak_error + weight * stats.peak_error
        error_map = error_map + weight * stats.error_map
    return NormalizedRenderStats(
        contrib=contrib,
        hits=hits,
        avg_trans=avg_trans,
        avg_contrib=avg_contrib,
        residual=residual,
        avg_residual=avg_residual,
        error_map=error_map,
        peak_error=peak_error,
    )


def _view_aware_context(
    per_view_observations: Sequence[DensityViewObservation] | None,
    *,
    count: int,
    device: torch.device,
    dtype: torch.dtype,
    cfg: DensityControlConfig,
) -> ViewAwareDensityContext | None:
    if not per_view_observations:
        return None
    sorted_observations = sorted(per_view_observations, key=lambda obs: obs.coverage.view_index)
    view_coverages = [obs.coverage for obs in sorted_observations]
    normalized_stats = [normalize_render_stats(obs.render_stats, count, device, dtype) for obs in sorted_observations]
    view_errors = torch.tensor(
        [
            float(obs.residual_map.mean().item())
            if obs.residual_map is not None and obs.residual_map.numel() > 0
            else 0.0
            for obs in sorted_observations
        ],
        device=device,
        dtype=dtype,
    )
    error_fraction_of_worst = None
    if view_errors.numel() > 0 and bool((view_errors > 0).any().item()):
        error_fraction_of_worst = view_errors / view_errors.max().clamp_min(1.0e-8)
    coverage_weights = _coverage_weight_tensor(view_coverages, device, dtype)
    visible_fraction_of_best, intersection_fraction_of_best = _coverage_fraction_tensors(
        view_coverages,
        device,
        dtype,
    )
    weak_indices = _weak_view_indices(
        view_coverages,
        visible_fraction_of_best=visible_fraction_of_best,
        intersection_fraction_of_best=intersection_fraction_of_best,
        error_fraction_of_worst=error_fraction_of_worst,
        cfg=cfg,
    )
    reseed_indices = _reseed_view_indices(
        view_coverages,
        visible_fraction_of_best=visible_fraction_of_best,
        intersection_fraction_of_best=intersection_fraction_of_best,
        error_fraction_of_worst=error_fraction_of_worst,
        weak_view_indices=weak_indices,
        cfg=cfg,
    )
    weak_positions = [idx for idx, coverage in enumerate(view_coverages) if coverage.view_index in weak_indices]
    reseed_positions = [idx for idx, coverage in enumerate(view_coverages) if coverage.view_index in reseed_indices]
    weighted_stats = _weighted_render_stats(
        normalized_stats,
        coverage_weights,
        count=count,
        device=device,
        dtype=dtype,
    )
    weak_view_stats = _weighted_render_stats(
        [normalized_stats[idx] for idx in weak_positions],
        torch.ones(len(weak_positions), device=device, dtype=dtype),
        count=count,
        device=device,
        dtype=dtype,
    )
    contrib_stack = torch.stack([stats.contrib for stats in normalized_stats], dim=0)
    trans_stack = torch.stack([stats.avg_trans for stats in normalized_stats], dim=0)
    return ViewAwareDensityContext(
        weighted_stats=weighted_stats,
        weak_view_stats=weak_view_stats,
        min_contrib=contrib_stack.min(dim=0).values,
        min_trans=trans_stack.min(dim=0).values,
        coverage_weights=coverage_weights,
        visible_fraction_of_best=visible_fraction_of_best,
        intersection_fraction_of_best=intersection_fraction_of_best,
        weak_view_indices=weak_indices,
        reseed_view_indices=reseed_indices,
        can_prune=len(weak_indices) == 0,
        view_coverages=view_coverages,
        weak_observations=[sorted_observations[idx] for idx in weak_positions],
        reseed_observations=[sorted_observations[idx] for idx in reseed_positions],
    )
