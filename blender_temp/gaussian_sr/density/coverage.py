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
    if not view_coverages:
        return ()
    min_visible_abs = int(cfg.min_view_visible_gaussians)
    min_intersections_abs = int(cfg.min_view_intersection_count)
    min_visible_rel = max(float(cfg.min_view_visible_fraction_of_best), float(cfg.coverage_floor_visible_fraction))
    min_intersections_rel = max(
        float(cfg.min_view_intersection_fraction_of_best),
        float(cfg.coverage_floor_intersection_fraction),
    )
    device = visible_fraction_of_best.device
    V = len(view_coverages)
    visible_counts = torch.tensor([c.visible_count for c in view_coverages], device=device, dtype=torch.long)
    intersection_counts = torch.tensor([c.intersection_count for c in view_coverages], device=device, dtype=torch.long)
    visible_too_low = (visible_counts < min_visible_abs) | (visible_fraction_of_best < min_visible_rel)
    intersections_too_low = (intersection_counts < min_intersections_abs) | (
        intersection_fraction_of_best < min_intersections_rel
    )
    error_high = torch.ones(V, dtype=torch.bool, device=device)
    if error_fraction_of_worst is not None:
        n = min(error_fraction_of_worst.numel(), V)
        error_high[:n] = error_fraction_of_worst[:n] >= float(cfg.weak_view_error_fraction_of_worst)
    mask = (visible_too_low | intersections_too_low) & error_high
    return tuple(int(view_coverages[i].view_index) for i in mask.nonzero(as_tuple=False).squeeze(-1).tolist())


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
        return ()
    if not view_coverages:
        return tuple(int(v) for v in weak_view_indices)

    trigger_rel = max(
        float(cfg.weak_view_reseed_trigger_fraction_of_best),
        float(cfg.min_view_visible_fraction_of_best),
        float(cfg.min_view_intersection_fraction_of_best),
        float(cfg.coverage_floor_visible_fraction),
        float(cfg.coverage_floor_intersection_fraction),
    )
    reseed_views: list[int] = [int(v) for v in weak_view_indices]
    seen: set[int] = set(reseed_views)
    device = visible_fraction_of_best.device
    V = len(view_coverages)
    visible_lagging = visible_fraction_of_best < trigger_rel
    intersections_lagging = intersection_fraction_of_best < trigger_rel
    error_high = torch.ones(V, dtype=torch.bool, device=device)
    if error_fraction_of_worst is not None:
        n = min(error_fraction_of_worst.numel(), V)
        error_high[:n] = error_fraction_of_worst[:n] >= float(cfg.weak_view_error_fraction_of_worst)
    lagging_mask = (visible_lagging | intersections_lagging) & error_high
    for idx in lagging_mask.nonzero(as_tuple=False).squeeze(-1).tolist():
        view_id = int(view_coverages[idx].view_index)
        if view_id not in seen:
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
    w = weights / weights.sum().clamp_min(1.0e-8)  # [V]
    contrib_stack = torch.stack([s.contrib for s in stats_list], dim=0)  # [V, N]
    hits_stack = torch.stack([s.hits for s in stats_list], dim=0)
    avg_trans_stack = torch.stack([s.avg_trans for s in stats_list], dim=0)
    avg_contrib_stack = torch.stack([s.avg_contrib for s in stats_list], dim=0)
    residual_stack = torch.stack([s.residual for s in stats_list], dim=0)
    avg_residual_stack = torch.stack([s.avg_residual for s in stats_list], dim=0)
    peak_error_stack = torch.stack([s.peak_error for s in stats_list], dim=0)
    error_map_stack = torch.stack([s.error_map for s in stats_list], dim=0)  # [V, N, B]
    w1 = w[:, None]  # [V, 1] for broadcasting over [V, N]
    return NormalizedRenderStats(
        contrib=(w1 * contrib_stack).sum(dim=0),
        hits=(w1 * hits_stack).sum(dim=0),
        avg_trans=(w1 * avg_trans_stack).sum(dim=0),
        avg_contrib=(w1 * avg_contrib_stack).sum(dim=0),
        residual=(w1 * residual_stack).sum(dim=0),
        avg_residual=(w1 * avg_residual_stack).sum(dim=0),
        peak_error=(w1 * peak_error_stack).sum(dim=0),
        error_map=(w[:, None, None] * error_map_stack).sum(dim=0),
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
