from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor


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


def normalize_render_stats(
    render_stats: Mapping[str, Tensor] | None,
    count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> NormalizedRenderStats:
    return NormalizedRenderStats.from_render_stats(render_stats, count, device, dtype)
