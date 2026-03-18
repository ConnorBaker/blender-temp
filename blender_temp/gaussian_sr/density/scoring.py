import torch
from torch import Tensor

from ..field import CanonicalGaussianField
from ..posefree_config import DensityControlConfig
from .types import DensityScoreTerms, NormalizedRenderStats


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
