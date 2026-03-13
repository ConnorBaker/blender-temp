import math

import torch
from torch import Tensor

from .posefree_config import AppearanceConfig

_MAX_SH_DEGREE = 2

# Real SH normalization constants for the basis order used below.
# These are written in closed form to make their origin explicit instead of
# leaving unexplained decimal literals in the basis evaluation.
_SH_C0 = math.sqrt(1.0 / (4.0 * math.pi))
_SH_C1 = math.sqrt(3.0 / (4.0 * math.pi))
_SH_C2_0 = math.sqrt(15.0 / (4.0 * math.pi))
_SH_C2_1 = math.sqrt(5.0 / (16.0 * math.pi))
_SH_C2_2 = math.sqrt(15.0 / (16.0 * math.pi))


def _validate_sh_degree(degree: int) -> int:
    degree = int(degree)
    if degree < 0 or degree > _MAX_SH_DEGREE:
        raise ValueError(f"Only SH degrees 0-{_MAX_SH_DEGREE} are supported in this implementation")
    return degree


def _validate_view_dependent_inputs(
    base_rgb_logits: Tensor,
    means3d: Tensor,
    sh_coeffs: Tensor,
    sh_degree: int,
) -> None:
    if base_rgb_logits.ndim != 2 or base_rgb_logits.shape[1] != 3:
        raise ValueError(f"base_rgb_logits must have shape [N, 3], got {tuple(base_rgb_logits.shape)}")
    if means3d.ndim != 2 or means3d.shape[1] != 3:
        raise ValueError(f"means3d must have shape [N, 3], got {tuple(means3d.shape)}")
    if means3d.shape[0] != base_rgb_logits.shape[0]:
        raise ValueError(f"means3d length mismatch: expected {base_rgb_logits.shape[0]}, got {means3d.shape[0]}")
    expected_shape = (base_rgb_logits.shape[0], 3, num_sh_bases(sh_degree) - 1)
    if tuple(sh_coeffs.shape) != expected_shape:
        raise ValueError(f"sh_coeffs must have shape {expected_shape}, got {tuple(sh_coeffs.shape)}")


def num_sh_bases(degree: int) -> int:
    """Return the number of real spherical-harmonic basis functions up to `degree`."""
    degree = _validate_sh_degree(degree)
    return (degree + 1) * (degree + 1)


def _camera_center_world(R_cw: Tensor, t_cw: Tensor) -> Tensor:
    """Compute the camera center in world coordinates from a world-to-camera pose."""
    return -(R_cw.transpose(0, 1) @ t_cw)


def _viewdirs_from_pose(means3d: Tensor, R_cw: Tensor, t_cw: Tensor) -> Tensor:
    """Return normalized world-space directions from each Gaussian toward the camera center."""
    center = _camera_center_world(R_cw, t_cw)[None, :]
    dirs = center - means3d
    return dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)


def sh_basis(view_dirs: Tensor, degree: int) -> Tensor:
    """Evaluate the real SH basis up to `degree` for normalized view directions.

    The basis order is:
    `Y00, Y1-1, Y10, Y11, Y2-2, Y2-1, Y20, Y21, Y22`.
    This matches the layout used by the learned appearance residual coefficients.
    """
    degree = _validate_sh_degree(degree)
    if view_dirs.ndim != 2 or view_dirs.shape[1] != 3:
        raise ValueError(f"view_dirs must have shape [N, 3], got {tuple(view_dirs.shape)}")

    x = view_dirs[:, 0]
    y = view_dirs[:, 1]
    z = view_dirs[:, 2]

    basis = [torch.full_like(x, _SH_C0)]
    if degree >= 1:
        basis.extend([
            -_SH_C1 * y,
            _SH_C1 * z,
            -_SH_C1 * x,
        ])
    if degree >= 2:
        basis.extend([
            _SH_C2_0 * x * y,
            -_SH_C2_0 * y * z,
            _SH_C2_1 * (3.0 * z * z - 1.0),
            -_SH_C2_0 * x * z,
            _SH_C2_2 * (x * x - y * y),
        ])
    return torch.stack(basis, dim=-1)


def apply_view_dependent_rgb(
    base_rgb_logits: Tensor,
    sh_coeffs: Tensor | None,
    means3d: Tensor,
    R_cw: Tensor | None,
    t_cw: Tensor | None,
    cfg: AppearanceConfig,
) -> Tensor:
    """Apply the configured appearance model to produce bounded RGB values.

    In `"constant"` mode this returns the sigmoid of the learned base RGB logits.
    In `"sh"` mode this adds a real spherical-harmonic directional residual before
    the final sigmoid. The learned SH coefficients exclude the DC term because the
    base RGB logits already represent the view-independent color.
    """
    if cfg.mode == "constant" or sh_coeffs is None or cfg.sh_degree <= 0 or R_cw is None or t_cw is None:
        return torch.sigmoid(base_rgb_logits)

    _validate_view_dependent_inputs(base_rgb_logits, means3d, sh_coeffs, cfg.sh_degree)
    dirs = _viewdirs_from_pose(means3d, R_cw, t_cw)
    directional = sh_basis(dirs, cfg.sh_degree)[:, 1:]
    residual = torch.einsum("nck,nk->nc", sh_coeffs, directional)
    return torch.sigmoid(base_rgb_logits + float(cfg.residual_scale) * residual)


__all__ = [
    "num_sh_bases",
    "sh_basis",
    "apply_view_dependent_rgb",
]
