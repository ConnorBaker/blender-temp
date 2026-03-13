from collections.abc import Mapping, Sequence
from dataclasses import dataclass

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
        peak_error = error_map.max(dim=1).values if error_map.numel() > 0 else torch.zeros_like(contrib)
        return cls(
            contrib=contrib,
            hits=hits,
            avg_trans=trans_sum / safe_hits,
            avg_contrib=contrib / safe_hits,
            residual=residual,
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
    can_prune: bool
    view_coverages: list[DensityViewCoverage]


def gradient_score(field_model: CanonicalGaussianField) -> Tensor:
    n = field_model.num_gaussians
    device = field_model.depth_raw.device
    dtype = field_model.depth_raw.dtype
    score = torch.zeros(n, device=device, dtype=dtype)
    for param in (field_model.depth_raw, field_model.xyz_offset):
        if param.grad is None:
            continue
        score = score + param.grad.detach().reshape(n, -1).norm(dim=1)
    return score


def scale_score(field_model: CanonicalGaussianField) -> Tensor:
    return torch.exp(field_model.log_scale.detach()).amax(dim=1)


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


def _density_score_terms(grad_score: Tensor, scale: Tensor, stats: NormalizedRenderStats) -> DensityScoreTerms:
    inv_scale = scale.mean().clamp_min(1.0e-8) / scale.clamp_min(1.0e-8)
    return DensityScoreTerms(
        grad=_norm(grad_score),
        visibility=_norm(stats.contrib),
        min_visibility=_norm(stats.contrib),
        residual=_norm(stats.residual),
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
    cfg: DensityControlConfig,
) -> tuple[int, ...]:
    min_visible_abs = int(cfg.min_view_visible_gaussians)
    min_intersections_abs = int(cfg.min_view_intersection_count)
    min_visible_rel = float(cfg.min_view_visible_fraction_of_best)
    min_intersections_rel = float(cfg.min_view_intersection_fraction_of_best)
    weak_views: list[int] = []
    for idx, coverage in enumerate(view_coverages):
        visible_too_low = (
            coverage.visible_count < min_visible_abs
            or float(visible_fraction_of_best[idx].item()) < min_visible_rel
        )
        intersections_too_low = (
            coverage.intersection_count < min_intersections_abs
            or float(intersection_fraction_of_best[idx].item()) < min_intersections_rel
        )
        if visible_too_low or intersections_too_low:
            weak_views.append(int(coverage.view_index))
    return tuple(weak_views)


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
    peak_error = torch.zeros(count, device=device, dtype=dtype)
    error_map = torch.zeros(count, stats_list[0].error_map.shape[-1], device=device, dtype=dtype)
    for idx, stats in enumerate(stats_list):
        weight = w[idx]
        contrib = contrib + weight * stats.contrib
        hits = hits + weight * stats.hits
        avg_trans = avg_trans + weight * stats.avg_trans
        avg_contrib = avg_contrib + weight * stats.avg_contrib
        residual = residual + weight * stats.residual
        peak_error = peak_error + weight * stats.peak_error
        error_map = error_map + weight * stats.error_map
    return NormalizedRenderStats(
        contrib=contrib,
        hits=hits,
        avg_trans=avg_trans,
        avg_contrib=avg_contrib,
        residual=residual,
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
    normalized_stats = [
        normalize_render_stats(obs.render_stats, count, device, dtype) for obs in sorted_observations
    ]
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
        cfg=cfg,
    )
    weak_positions = [idx for idx, coverage in enumerate(view_coverages) if coverage.view_index in weak_view_indices]
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
        can_prune=len(weak_view_indices) == 0,
        view_coverages=view_coverages,
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
                residual=float(stats.residual[i].item()),
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
        residual_mean=float(stats.residual.mean().item()),
        residual_max=float(stats.residual.max().item()),
        peak_error_mean=float(stats.peak_error.mean().item()),
        peak_error_max=float(stats.peak_error.max().item()),
        transmittance_mean=float(stats.avg_trans.mean().item()),
        screen_error_bins=int(stats.error_map.shape[-1]) if stats.error_map.ndim == 2 else 0,
        gradient_mean=float(grad.mean().item()),
        scale_mean=float(scale.mean().item()),
        opacity_mean=float(opacity.mean().item()),
        prune_protected=bool(view_ctx is not None and not view_ctx.can_prune),
        coverage_weights=(
            [float(value.item()) for value in view_ctx.coverage_weights]
            if view_ctx is not None
            else []
        ),
        visible_fraction_of_best=(
            [float(value.item()) for value in view_ctx.visible_fraction_of_best]
            if view_ctx is not None
            else []
        ),
        intersection_fraction_of_best=(
            [float(value.item()) for value in view_ctx.intersection_fraction_of_best]
            if view_ctx is not None
            else []
        ),
        weak_view_indices=list(view_ctx.weak_view_indices) if view_ctx is not None else [],
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
            _density_score_terms(grad_score, scale, stats),
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
            _density_score_terms(grad_score, scale, stats),
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
    return should_run_density_control(step, cfg)


def apply_density_control(
    field_model: CanonicalGaussianField,
    cfg: DensityControlConfig,
    step: int,
    render_stats: Mapping[str, Tensor] | None = None,
    per_view_observations: Sequence[DensityViewObservation] | None = None,
) -> DensityControlResult:
    if not should_run_density_control(step, cfg):
        return DensityControlResult.skipped(field_model.num_gaussians)

    before = field_model.num_gaussians
    device = field_model.depth_raw.device
    dtype = field_model.depth_raw.dtype
    opacity = torch.sigmoid(field_model.opacity_logit.detach().view(-1))
    grad = gradient_score(field_model)
    scale = scale_score(field_model)
    view_ctx = _view_aware_context(
        per_view_observations,
        count=before,
        device=device,
        dtype=dtype,
        cfg=cfg,
    )
    if view_ctx is None:
        stats = normalize_render_stats(render_stats, before, device, dtype)
        score_terms = _density_score_terms(grad, scale, stats)
    else:
        stats = view_ctx.weighted_stats
        score_terms = _density_score_terms(grad, scale, stats)
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

    if view_ctx is not None and not view_ctx.can_prune:
        keep = torch.ones_like(opacity, dtype=torch.bool)
    else:
        keep = compute_prune_keep_mask(opacity, stats.contrib, cfg)
    pruned = int((~keep).sum().item())
    if pruned > 0:
        field_model.prune_keep_mask(keep)
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
                can_prune=view_ctx.can_prune,
                view_coverages=view_ctx.view_coverages,
            )

    split_idx = select_split_indices(
        opacity,
        grad,
        scale,
        stats,
        cfg,
        field_model.num_gaussians,
        score=split_score,
        weak_visibility=(
            None
            if view_ctx is None or not view_ctx.weak_view_indices
            else view_ctx.weak_view_stats.contrib
        ),
        weak_trans=(
            None
            if view_ctx is None or not view_ctx.weak_view_indices
            else view_ctx.weak_view_stats.avg_trans
        ),
    )
    split_count = int(split_idx.numel())
    if split_count > 0:
        field_model.split_gaussians(
            split_idx,
            shrink_factor=float(cfg.split_shrink_factor),
            offset_scale=float(cfg.split_offset_scale),
        )

    clone_idx = select_clone_indices(
        opacity,
        grad,
        scale,
        stats,
        cfg,
        field_model.num_gaussians,
        exclude=split_idx,
        score=clone_score,
        weak_visibility=(
            None
            if view_ctx is None or not view_ctx.weak_view_indices
            else view_ctx.weak_view_stats.contrib
        ),
        weak_trans=(
            None
            if view_ctx is None or not view_ctx.weak_view_indices
            else view_ctx.weak_view_stats.avg_trans
        ),
    )
    clone_count = int(clone_idx.numel())
    if clone_count > 0:
        field_model.clone_gaussians(clone_idx, jitter_scale=float(cfg.clone_jitter_scale))

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
        changed=bool(pruned > 0 or split_count > 0 or clone_count > 0),
        pruned=pruned,
        split=split_count,
        cloned=clone_count,
        before=before,
        after=after,
        debug=debug,
    )


__all__ = [
    "DensityControlResult",
    "DensityDebugEntry",
    "DensityDebugSummary",
    "DensityViewCoverage",
    "DensityViewObservation",
    "DensityScoreTerms",
    "NormalizedRenderStats",
    "gradient_score",
    "scale_score",
    "normalize_render_stats",
    "build_density_debug_summary",
    "compute_prune_keep_mask",
    "select_split_indices",
    "select_clone_indices",
    "should_run_density_control",
    "should_run_density_control_for_stage",
    "apply_density_control",
]
