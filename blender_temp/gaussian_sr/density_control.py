from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import torch
from torch import Tensor

from .field import CanonicalGaussianField
from .posefree_config import DensityControlConfig


@dataclass(frozen=True, slots=True)
class NormalizedRenderStats:
    contrib: Tensor
    hits: Tensor
    avg_trans: Tensor
    avg_contrib: Tensor
    residual: Tensor
    avg_residual: Tensor
    error_map: Tensor
    peak_error: Tensor

    @classmethod
    def zeros(cls, count: int, device: torch.device, dtype: torch.dtype) -> "NormalizedRenderStats":
        zeros = torch.zeros(count, device=device, dtype=dtype)
        zero_map = torch.zeros(count, 1, device=device, dtype=dtype)
        return cls(
            contrib=zeros,
            hits=zeros,
            avg_trans=zeros,
            avg_contrib=zeros,
            residual=zeros,
            avg_residual=zeros,
            error_map=zero_map,
            peak_error=zeros,
        )

    @classmethod
    def from_render_stats(
        cls,
        render_stats: Mapping[str, Tensor] | None,
        count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "NormalizedRenderStats":
        if render_stats is None:
            return cls.zeros(count, device, dtype)

        contrib = render_stats["contrib"].to(device=device, dtype=dtype)
        hits = render_stats["hits"].to(device=device, dtype=dtype)
        trans_sum = render_stats["transmittance"].to(device=device, dtype=dtype)
        residual = render_stats["residual"].to(device=device, dtype=dtype)
        error_map = render_stats["error_map"].to(device=device, dtype=dtype)
        safe_hits = hits.clamp_min(1.0)
        safe_contrib = contrib.clamp_min(1.0e-8)
        peak_error = error_map.max(dim=1).values if error_map.numel() > 0 else torch.zeros_like(contrib)
        return cls(
            contrib=contrib,
            hits=hits,
            avg_trans=trans_sum / safe_hits,
            avg_contrib=contrib / safe_hits,
            residual=residual,
            avg_residual=residual / safe_contrib,
            error_map=error_map,
            peak_error=peak_error,
        )

    def masked(self, keep: Tensor) -> "NormalizedRenderStats":
        return type(self)(
            contrib=self.contrib[keep],
            hits=self.hits[keep],
            avg_trans=self.avg_trans[keep],
            avg_contrib=self.avg_contrib[keep],
            residual=self.residual[keep],
            avg_residual=self.avg_residual[keep],
            error_map=self.error_map[keep],
            peak_error=self.peak_error[keep],
        )


@dataclass(frozen=True, slots=True)
class DensityViewCoverage:
    view_index: int
    visible_count: int
    intersection_count: int
    render_width: int
    render_height: int

    def to_dict(self) -> dict[str, int]:
        return {
            "view_index": self.view_index,
            "visible_count": self.visible_count,
            "intersection_count": self.intersection_count,
            "render_width": self.render_width,
            "render_height": self.render_height,
        }


@dataclass(frozen=True, slots=True)
class DensityViewObservation:
    coverage: DensityViewCoverage
    render_stats: Mapping[str, Tensor] | None
    residual_map: Tensor | None = None
    target_rgb: Tensor | None = None
    pred_rgb: Tensor | None = None
    R_cw: Tensor | None = None
    t_cw: Tensor | None = None
    intrinsics: Tensor | None = None


@dataclass(frozen=True, slots=True)
class DensityScoreTerms:
    grad: Tensor
    visibility: Tensor
    min_visibility: Tensor
    residual: Tensor
    peak_error: Tensor
    trans: Tensor
    scale: Tensor
    inv_scale: Tensor


@dataclass(frozen=True, slots=True)
class DensityDebugEntry:
    index: int
    score: float
    peak_bin: int
    peak_error: float
    residual: float
    visibility: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "index": self.index,
            "score": self.score,
            "peak_bin": self.peak_bin,
            "peak_error": self.peak_error,
            "residual": self.residual,
            "visibility": self.visibility,
        }


@dataclass(frozen=True, slots=True)
class DensityDebugSummary:
    visibility_mean: float
    visibility_max: float
    residual_mean: float
    residual_max: float
    peak_error_mean: float
    peak_error_max: float
    transmittance_mean: float
    screen_error_bins: int
    gradient_mean: float
    scale_mean: float
    opacity_mean: float
    prune_protected: bool
    coverage_weights: list[float]
    visible_fraction_of_best: list[float]
    intersection_fraction_of_best: list[float]
    weak_view_indices: list[int]
    reseed_view_indices: list[int]
    view_coverages: list[DensityViewCoverage]
    split_top: list[DensityDebugEntry]
    clone_top: list[DensityDebugEntry]

    def to_dict(self) -> dict[str, object]:
        return {
            "visibility_mean": self.visibility_mean,
            "visibility_max": self.visibility_max,
            "residual_mean": self.residual_mean,
            "residual_max": self.residual_max,
            "peak_error_mean": self.peak_error_mean,
            "peak_error_max": self.peak_error_max,
            "transmittance_mean": self.transmittance_mean,
            "screen_error_bins": self.screen_error_bins,
            "gradient_mean": self.gradient_mean,
            "scale_mean": self.scale_mean,
            "opacity_mean": self.opacity_mean,
            "prune_protected": self.prune_protected,
            "coverage_weights": self.coverage_weights,
            "visible_fraction_of_best": self.visible_fraction_of_best,
            "intersection_fraction_of_best": self.intersection_fraction_of_best,
            "weak_view_indices": self.weak_view_indices,
            "reseed_view_indices": self.reseed_view_indices,
            "view_coverages": [coverage.to_dict() for coverage in self.view_coverages],
            "split_top": [entry.to_dict() for entry in self.split_top],
            "clone_top": [entry.to_dict() for entry in self.clone_top],
        }


@dataclass(frozen=True, slots=True)
class DensityControlResult:
    ran: bool
    changed: bool
    pruned: int
    split: int
    cloned: int
    before: int
    after: int
    appended_count: int = 0
    reseeded: int = 0
    survivor_sources: Tensor | None = None
    debug: DensityDebugSummary | None = None

    @classmethod
    def skipped(cls, gaussian_count: int) -> "DensityControlResult":
        return cls(
            ran=False,
            changed=False,
            pruned=0,
            split=0,
            cloned=0,
            before=gaussian_count,
            after=gaussian_count,
            appended_count=0,
            reseeded=0,
            survivor_sources=None,
            debug=None,
        )

    def debug_dict(self) -> dict[str, object]:
        if self.debug is None:
            return {}
        return self.debug.to_dict()


@dataclass(frozen=True, slots=True)
class ViewAwareDensityContext:
    weighted_stats: NormalizedRenderStats
    weak_view_stats: NormalizedRenderStats
    min_contrib: Tensor
    min_trans: Tensor
    coverage_weights: Tensor
    visible_fraction_of_best: Tensor
    intersection_fraction_of_best: Tensor
    weak_view_indices: tuple[int, ...]
    reseed_view_indices: tuple[int, ...]
    can_prune: bool
    view_coverages: list[DensityViewCoverage]
    weak_observations: list[DensityViewObservation]
    reseed_observations: list[DensityViewObservation]


def gradient_score(field_model: CanonicalGaussianField) -> Tensor:
    n = field_model.num_gaussians
    device = field_model.means3d.device
    dtype = field_model.means3d.dtype
    score = torch.zeros(n, device=device, dtype=dtype)
    for param in (field_model.means3d, field_model.log_scale, field_model.opacity_logit):
        if param.grad is None:
            continue
        score = score + param.grad.detach()[:n].reshape(n, -1).norm(dim=1)
    return score


def scale_score(field_model: CanonicalGaussianField) -> Tensor:
    n = field_model.num_gaussians
    return torch.exp(field_model.log_scale.detach()[:n]).amax(dim=1)


def normalize_render_stats(
    render_stats: Mapping[str, Tensor] | None,
    count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> NormalizedRenderStats:
    return NormalizedRenderStats.from_render_stats(render_stats, count, device, dtype)


def compute_prune_keep_mask(opacity: Tensor, visibility: Tensor, cfg: DensityControlConfig) -> Tensor:
    keep = (opacity >= float(cfg.opacity_prune_threshold)) | (visibility >= float(cfg.prune_visibility_threshold))
    min_keep = min(int(cfg.min_gaussians), opacity.shape[0])
    if int(keep.sum().item()) < min_keep:
        priority = opacity + visibility
        topk = torch.topk(priority, k=min_keep, largest=True).indices
        keep = torch.zeros_like(keep, dtype=torch.bool)
        keep[topk] = True
    return keep


def _norm(x: Tensor) -> Tensor:
    return x / x.mean().clamp_min(1.0e-8)


def _quantile_threshold(values: Tensor, q: float) -> Tensor:
    return torch.quantile(values, values.new_tensor(float(q)))


def _density_score_terms(
    grad_score: Tensor,
    scale: Tensor,
    stats: NormalizedRenderStats,
    cfg: DensityControlConfig,
) -> DensityScoreTerms:
    inv_scale = scale.mean().clamp_min(1.0e-8) / scale.clamp_min(1.0e-8)
    visibility_term = stats.avg_contrib if bool(cfg.use_normalized_density_scores) else stats.contrib
    residual_term = stats.avg_residual if bool(cfg.use_normalized_density_scores) else stats.residual
    return DensityScoreTerms(
        grad=_norm(grad_score),
        visibility=_norm(visibility_term),
        min_visibility=_norm(visibility_term),
        residual=_norm(residual_term),
        peak_error=_norm(stats.peak_error),
        trans=_norm(stats.avg_trans),
        scale=_norm(scale),
        inv_scale=_norm(inv_scale),
    )


def _combine_density_score(
    cfg: DensityControlConfig,
    terms: DensityScoreTerms,
    *,
    use_inverse_scale: bool,
) -> Tensor:
    scale_term = terms.inv_scale if use_inverse_scale else terms.scale
    return (
        float(cfg.gradient_score_weight) * terms.grad
        + float(cfg.visibility_score_weight) * terms.visibility
        + float(cfg.min_view_score_weight) * terms.min_visibility
        + float(cfg.residual_score_weight) * terms.residual
        + float(cfg.screen_error_peak_weight) * terms.peak_error
        + float(cfg.transmittance_score_weight) * terms.trans
        + float(cfg.scale_score_weight) * scale_term
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
    weak_view_indices = _weak_view_indices(
        view_coverages,
        visible_fraction_of_best=visible_fraction_of_best,
        intersection_fraction_of_best=intersection_fraction_of_best,
        error_fraction_of_worst=error_fraction_of_worst,
        cfg=cfg,
    )
    reseed_view_indices = _reseed_view_indices(
        view_coverages,
        visible_fraction_of_best=visible_fraction_of_best,
        intersection_fraction_of_best=intersection_fraction_of_best,
        error_fraction_of_worst=error_fraction_of_worst,
        weak_view_indices=weak_view_indices,
        cfg=cfg,
    )
    weak_positions = [idx for idx, coverage in enumerate(view_coverages) if coverage.view_index in weak_view_indices]
    reseed_positions = [
        idx for idx, coverage in enumerate(view_coverages) if coverage.view_index in reseed_view_indices
    ]
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
        weak_view_indices=weak_view_indices,
        reseed_view_indices=reseed_view_indices,
        can_prune=len(weak_view_indices) == 0,
        view_coverages=view_coverages,
        weak_observations=[sorted_observations[idx] for idx in weak_positions],
        reseed_observations=[sorted_observations[idx] for idx in reseed_positions],
    )


def _topk_candidates(mask: Tensor, score: Tensor, k: int) -> Tensor:
    idx = mask.nonzero(as_tuple=False).squeeze(-1)
    if idx.numel() == 0 or k <= 0:
        return torch.empty(0, device=score.device, dtype=torch.long)
    if idx.numel() > k:
        local = torch.topk(score[idx], k=k, largest=True).indices
        idx = idx[local]
    return idx.to(torch.long)


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


def _select_reseed_pixels(
    flat_residual: Tensor,
    *,
    width: int,
    budget: int,
    residual_quantile: float,
    match_radius_px: float,
) -> Tensor:
    if budget <= 0 or flat_residual.numel() <= 0:
        return torch.empty(0, device=flat_residual.device, dtype=torch.long)

    if flat_residual.numel() == 1:
        return torch.zeros(1, device=flat_residual.device, dtype=torch.long)

    threshold = torch.quantile(flat_residual, flat_residual.new_tensor(residual_quantile))
    candidate_idx = (flat_residual >= threshold).nonzero(as_tuple=False).squeeze(-1)
    if candidate_idx.numel() == 0:
        candidate_idx = torch.topk(flat_residual, k=min(budget, flat_residual.numel()), largest=True).indices

    candidate_score = flat_residual[candidate_idx]
    pool = min(int(candidate_idx.numel()), max(int(budget), int(budget) * 8))
    if candidate_idx.numel() > pool:
        top = torch.topk(candidate_score, k=pool, largest=True).indices
        candidate_idx = candidate_idx[top]
        candidate_score = candidate_score[top]

    order = torch.argsort(candidate_score, descending=True)
    candidate_idx = candidate_idx[order]
    if match_radius_px <= 0.0 or candidate_idx.numel() <= budget:
        return candidate_idx[:budget].to(torch.long)

    px = (candidate_idx % width).tolist()
    py = (candidate_idx // width).tolist()
    radius2 = float(match_radius_px) * float(match_radius_px)
    keep_positions: list[int] = []
    for pos, (x, y) in enumerate(zip(px, py)):
        if all(((x - px[prev]) ** 2 + (y - py[prev]) ** 2) > radius2 for prev in keep_positions):
            keep_positions.append(pos)
            if len(keep_positions) >= budget:
                break
    if not keep_positions:
        return candidate_idx[:budget].to(torch.long)
    keep = torch.tensor(keep_positions, device=flat_residual.device, dtype=torch.long)
    return candidate_idx[keep].to(torch.long)


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


def _camera_depths(means3d: Tensor, R_cw: Tensor, t_cw: Tensor) -> Tensor:
    return means3d @ R_cw.transpose(0, 1)[:, 2] + t_cw[2]


def _reseed_for_observation(
    field_model: CanonicalGaussianField,
    observation: DensityViewObservation,
    cfg: DensityControlConfig,
    step: int,
) -> int:
    budget = int(cfg.weak_view_reseed_budget_per_view)
    if budget <= 0:
        return 0
    if (
        observation.residual_map is None
        or observation.target_rgb is None
        or observation.R_cw is None
        or observation.t_cw is None
        or observation.intrinsics is None
    ):
        return 0

    residual_map = observation.residual_map.detach()
    target_rgb = observation.target_rgb.detach()
    if residual_map.numel() <= 0 or target_rgb.numel() <= 0:
        return 0

    flat = residual_map.reshape(-1)
    pred_rgb = observation.pred_rgb.detach() if observation.pred_rgb is not None else None
    if (
        pred_rgb is not None
        and pred_rgb.shape == target_rgb.shape
        and float(cfg.weak_view_reseed_target_luma_min) >= 0.0
    ):
        target_luma = 0.2126 * target_rgb[0] + 0.7152 * target_rgb[1] + 0.0722 * target_rgb[2]
        pred_luma = 0.2126 * pred_rgb[0] + 0.7152 * pred_rgb[1] + 0.0722 * pred_rgb[2]
        positive_miss = (target_luma - pred_luma).clamp_min(0.0)
        if float(cfg.weak_view_reseed_target_luma_min) > 0.0:
            positive_miss = positive_miss * (target_luma >= float(cfg.weak_view_reseed_target_luma_min))
        guided = residual_map * positive_miss
        if bool((guided > 0).any().item()):
            flat = guided.reshape(-1)
    if flat.numel() <= 0:
        return 0
    h, w = residual_map.shape
    cand = _select_reseed_pixels(
        flat,
        width=w,
        budget=budget,
        residual_quantile=float(cfg.weak_view_reseed_residual_quantile),
        match_radius_px=float(cfg.weak_view_reseed_match_radius_px),
    )
    if cand.numel() == 0:
        return 0

    R_cw = observation.R_cw.detach()
    t_cw = observation.t_cw.detach()
    intr = observation.intrinsics.detach()
    fx, fy, cx, cy = intr.unbind(dim=0)

    means = field_model.means3d.detach()[: field_model.num_gaussians]
    cam_pts_all = means @ R_cw.transpose(0, 1) + t_cw.unsqueeze(0)
    cam_depths = cam_pts_all[:, 2]
    visible_mask = cam_depths > field_model.field_cfg.min_depth
    if observation.render_stats is not None and "contrib" in observation.render_stats:
        contrib = observation.render_stats["contrib"].to(device=means.device)
        if contrib.numel() == cam_depths.numel():
            visible_mask = visible_mask & (contrib > 0)
    if visible_mask.any():
        fallback_depth = cam_depths[visible_mask].median().clamp_min(field_model.field_cfg.min_depth)
    else:
        global_valid = cam_depths > field_model.field_cfg.min_depth
        if global_valid.any():
            fallback_depth = cam_depths[global_valid].median().clamp_min(field_model.field_cfg.min_depth)
        else:
            fallback_depth = means.new_tensor(field_model.field_cfg.init_depth)

    py = torch.div(cand, w, rounding_mode="floor")
    px = cand - py * w
    u = px.to(dtype=means.dtype) + 0.5
    v = py.to(dtype=means.dtype) + 0.5
    depth_init = torch.full_like(u, float(fallback_depth.item()))
    if visible_mask.any():
        cand_uv = torch.stack((u, v), dim=-1)
        visible_pts = cam_pts_all[visible_mask]
        visible_depths = cam_depths[visible_mask]
        visible_xy = visible_pts[:, :2] / visible_depths[:, None].clamp_min(field_model.field_cfg.min_depth)
        visible_uv = torch.stack(
            (
                fx * visible_xy[:, 0] + cx,
                fy * visible_xy[:, 1] + cy,
            ),
            dim=-1,
        )
        in_bounds = (
            (visible_uv[:, 0] >= 0.0)
            & (visible_uv[:, 0] <= float(w - 1))
            & (visible_uv[:, 1] >= 0.0)
            & (visible_uv[:, 1] <= float(h - 1))
        )
        if in_bounds.any():
            visible_uv = visible_uv[in_bounds]
            visible_depths = visible_depths[in_bounds]
        if visible_uv.numel() > 0:
            diff = cand_uv[:, None, :] - visible_uv[None, :, :]
            dist2 = (diff * diff).sum(dim=-1)
            nearest = dist2.argmin(dim=1)
            nearest_depth = visible_depths[nearest]
            nearest_dist2 = dist2.gather(1, nearest[:, None]).squeeze(1)
            depth_match_radius_px = float(cfg.weak_view_reseed_depth_match_radius_px)
            if depth_match_radius_px > 0.0:
                use_nearest = nearest_dist2 <= (depth_match_radius_px * depth_match_radius_px)
                depth_init[use_nearest] = nearest_depth[use_nearest]
            else:
                depth_init = nearest_depth
    x = (u - cx) / fx.clamp_min(1.0)
    y = (v - cy) / fy.clamp_min(1.0)
    cam_pts = torch.stack((x * depth_init, y * depth_init, depth_init), dim=-1)
    world_pts = (cam_pts - t_cw.unsqueeze(0)) @ R_cw

    rgb = target_rgb[:, py.long(), px.long()].permute(1, 0).contiguous()
    sx = field_model.field_cfg.init_scale_xy * depth_init / max(float(fx.item()), 1.0)
    sy = field_model.field_cfg.init_scale_xy * depth_init / max(float(fy.item()), 1.0)
    sz = field_model.field_cfg.init_scale_z * depth_init
    scale = torch.stack((sx, sy, sz), dim=-1)
    opacity = torch.full(
        (world_pts.shape[0], 1),
        float(cfg.weak_view_reseed_init_opacity),
        device=world_pts.device,
        dtype=world_pts.dtype,
    ).clamp_(1.0e-4, 1.0 - 1.0e-4)
    uv = torch.stack((u, v), dim=-1)
    protect_until_step = torch.full(
        (world_pts.shape[0],),
        int(step) + max(int(cfg.weak_view_reseed_protect_steps), 0),
        device=world_pts.device,
        dtype=torch.long,
    )
    appended = field_model.append_gaussians(
        means3d=world_pts,
        rgb=rgb,
        uv=uv,
        scale=scale,
        opacity=opacity,
        protect_until_step=protect_until_step,
    )
    return int(appended)


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
            view_ctx = ViewAwareDensityContext(
                weighted_stats=view_ctx.weighted_stats.masked(keep),
                weak_view_stats=view_ctx.weak_view_stats.masked(keep),
                min_contrib=view_ctx.min_contrib[keep],
                min_trans=view_ctx.min_trans[keep],
                coverage_weights=view_ctx.coverage_weights,
                visible_fraction_of_best=view_ctx.visible_fraction_of_best,
                intersection_fraction_of_best=view_ctx.intersection_fraction_of_best,
                weak_view_indices=view_ctx.weak_view_indices,
                reseed_view_indices=view_ctx.reseed_view_indices,
                can_prune=view_ctx.can_prune,
                view_coverages=view_ctx.view_coverages,
                weak_observations=view_ctx.weak_observations,
                reseed_observations=view_ctx.reseed_observations,
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


__all__ = [
    "DensityControlResult",
    "DensityViewCoverage",
    "DensityViewObservation",
    "NormalizedRenderStats",
    "apply_density_control",
    "build_density_debug_summary",
    "compute_prune_keep_mask",
    "gradient_score",
    "normalize_render_stats",
    "scale_score",
    "select_clone_indices",
    "select_split_indices",
    "should_run_density_control",
    "should_run_density_control_for_stage",
]
