import pytest
import torch
from hypothesis import given
from torch.testing import assert_close

from blender_temp.gaussian_sr.appearance import apply_view_dependent_rgb, num_sh_bases, sh_basis
from blender_temp.gaussian_sr.math_utils import pose_vec_to_rt
from blender_temp.gaussian_sr.posefree_config import AppearanceConfig

from .strategies import DEFAULT_SETTINGS, appearance_inputs


@DEFAULT_SETTINGS
@given(inputs=appearance_inputs())
def test_constant_mode_is_pose_independent(inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
    base_logits, sh_coeffs, means = inputs
    cfg = AppearanceConfig(mode="constant")
    r_a = torch.eye(3, dtype=torch.float32)
    t_a = torch.zeros(3, dtype=torch.float32)
    r_b = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
    t_b = torch.tensor([0.25, -0.5, 0.75], dtype=torch.float32)

    out_a = apply_view_dependent_rgb(base_logits, sh_coeffs, means, r_a, t_a, cfg)
    out_b = apply_view_dependent_rgb(base_logits, sh_coeffs, means, r_b, t_b, cfg)

    assert_close(out_a, out_b, atol=1.0e-6, rtol=1.0e-6)


@DEFAULT_SETTINGS
@given(inputs=appearance_inputs(sh_degree=2))
def test_zero_sh_coefficients_reduce_to_sigmoid_base_color(
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    base_logits, _, means = inputs
    cfg = AppearanceConfig(mode="sh", sh_degree=2)
    sh_coeffs = torch.zeros(base_logits.shape[0], 3, num_sh_bases(cfg.sh_degree) - 1, dtype=torch.float32)
    out = apply_view_dependent_rgb(base_logits, sh_coeffs, means, torch.eye(3), torch.zeros(3), cfg)

    assert_close(out, torch.sigmoid(base_logits), atol=1.0e-6, rtol=1.0e-6)


@DEFAULT_SETTINGS
@given(inputs=appearance_inputs(sh_degree=2))
def test_view_dependent_rgb_output_is_finite_and_bounded(
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    base_logits, sh_coeffs, means = inputs
    cfg = AppearanceConfig(mode="sh", sh_degree=2)
    out = apply_view_dependent_rgb(base_logits, sh_coeffs, means, torch.eye(3), torch.zeros(3), cfg)

    assert torch.isfinite(out).all()
    assert ((out >= 0.0) & (out <= 1.0)).all()


def test_view_dependent_rgb_pose_path_has_finite_gradients_at_identity_pose() -> None:
    count = 8
    cfg = AppearanceConfig(mode="sh", sh_degree=2)
    base_logits = torch.zeros(count, 3, dtype=torch.float32)
    sh_coeffs = torch.zeros(count, 3, num_sh_bases(cfg.sh_degree) - 1, dtype=torch.float32)
    means = torch.randn(count, 3, dtype=torch.float32)
    means[:, 2] = means[:, 2].abs() + 0.5
    xi = torch.zeros(1, 6, dtype=torch.float32, requires_grad=True)

    r, t = pose_vec_to_rt(xi)
    out = apply_view_dependent_rgb(base_logits, sh_coeffs, means, r[0], t[0], cfg)
    loss = out.sum()
    loss.backward()

    assert xi.grad is not None
    assert torch.isfinite(xi.grad).all()


def test_sh_basis_output_width_matches_degree() -> None:
    dirs = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=torch.float32)

    assert sh_basis(dirs, 0).shape == (2, 1)
    assert sh_basis(dirs, 1).shape == (2, 4)
    assert sh_basis(dirs, 2).shape == (2, 9)


def test_view_dependent_rgb_rejects_mismatched_sh_coeff_shape() -> None:
    count = 4
    cfg = AppearanceConfig(mode="sh", sh_degree=2)
    base_logits = torch.zeros(count, 3, dtype=torch.float32)
    means = torch.zeros(count, 3, dtype=torch.float32)
    bad_coeffs = torch.zeros(count, 3, num_sh_bases(cfg.sh_degree), dtype=torch.float32)

    with pytest.raises(ValueError, match="sh_coeffs must have shape"):
        apply_view_dependent_rgb(base_logits, bad_coeffs, means, torch.eye(3), torch.zeros(3), cfg)
