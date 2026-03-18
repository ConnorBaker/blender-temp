import torch
from hypothesis import given, strategies as st

from blender_temp.gaussian_sr.field import CanonicalGaussianField, ScaleAwareResidualHead
from blender_temp.gaussian_sr.image_utils import pixel_grid
from blender_temp.gaussian_sr.posefree_config import AppearanceConfig, FieldConfig

from .strategies import DEFAULT_SETTINGS, chw_images


def _make_intrinsics(height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    focal = torch.tensor([float(max(width, 1)), float(max(height, 1))], dtype=torch.float32)
    principal = torch.tensor([(width - 1.0) * 0.5, (height - 1.0) * 0.5], dtype=torch.float32)
    return focal, principal


def _make_field(
    anchor_rgb: torch.Tensor, stride: int, feature_dim: int
) -> tuple[CanonicalGaussianField, tuple[torch.Tensor, torch.Tensor]]:
    field_cfg = FieldConfig(anchor_stride=stride, feature_dim=feature_dim)
    appearance_cfg = AppearanceConfig(mode="constant")
    focal, principal = _make_intrinsics(anchor_rgb.shape[-2], anchor_rgb.shape[-1])
    field = CanonicalGaussianField(anchor_rgb, focal, principal, field_cfg, appearance_cfg)
    return field, (focal, principal)


def _assert_aligned_lengths(field: CanonicalGaussianField) -> None:
    n = field.num_gaussians
    assert field.uv[:n].shape[0] == n
    assert field.seed_id[:n].shape[0] == n
    assert field.protect_until_step[:n].shape[0] == n
    assert field.means3d[:n].shape[0] == n
    assert field.quat_raw[:n].shape[0] == n
    assert field.log_scale[:n].shape[0] == n
    assert field.opacity_logit[:n].shape[0] == n
    assert field.rgb_logit[:n].shape[0] == n
    assert field.latent[:n].shape[0] == n
    if field.sh_coeffs is not None:
        assert field.sh_coeffs[:n].shape[0] == n


@DEFAULT_SETTINGS
@given(
    anchor_rgb=chw_images(min_side=2, max_side=6, min_value=0.05, max_value=0.95),
    stride=st.integers(min_value=1, max_value=2),
    feature_dim=st.integers(min_value=1, max_value=4),
)
def test_gaussian_params_are_finite_and_well_formed(
    anchor_rgb: torch.Tensor,
    stride: int,
    feature_dim: int,
) -> None:
    field, intrinsics = _make_field(anchor_rgb, stride=stride, feature_dim=feature_dim)
    params = field.gaussian_params(intrinsics)
    n = field.num_gaussians

    assert params["means3d"].shape == (n, 3)
    assert params["quat"].shape == (n, 4)
    assert params["scale"].shape == (n, 3)
    assert params["opacity"].shape == (n,)
    assert params["rgb"].shape == (n, 3)
    assert params["latent"].shape == (n, feature_dim)
    assert torch.isfinite(params["means3d"]).all()
    assert torch.isfinite(params["quat"]).all()
    assert torch.isfinite(params["scale"]).all()
    assert torch.isfinite(params["opacity"]).all()
    assert torch.isfinite(params["rgb"]).all()
    assert torch.isfinite(params["latent"]).all()
    assert (params["scale"] > 0.0).all()
    assert ((params["opacity"] >= 0.0) & (params["opacity"] <= 1.0)).all()
    assert ((params["rgb"] >= 0.0) & (params["rgb"] <= 1.0)).all()


@DEFAULT_SETTINGS
@given(
    anchor_rgb=chw_images(min_side=2, max_side=6, min_value=0.05, max_value=0.95),
    feature_dim=st.integers(min_value=1, max_value=4),
    data=st.data(),
)
def test_prune_keep_mask_preserves_internal_alignment(
    anchor_rgb: torch.Tensor,
    feature_dim: int,
    data: st.DataObject,
) -> None:
    field, intrinsics = _make_field(anchor_rgb, stride=1, feature_dim=feature_dim)
    n = field.num_gaussians
    keep_indices = data.draw(st.lists(st.integers(min_value=0, max_value=n - 1), min_size=1, max_size=n, unique=True))
    keep_mask = torch.zeros(n, dtype=torch.bool)
    keep_mask[keep_indices] = True

    field.prune_keep_mask(keep_mask)
    _assert_aligned_lengths(field)
    assert field.num_gaussians == len(keep_indices)
    params = field.gaussian_params(intrinsics)
    assert torch.isfinite(params["means3d"]).all()


@DEFAULT_SETTINGS
@given(
    anchor_rgb=chw_images(min_side=2, max_side=6, min_value=0.05, max_value=0.95),
    feature_dim=st.integers(min_value=1, max_value=4),
    data=st.data(),
)
def test_clone_gaussians_increases_count_and_keeps_finite_state(
    anchor_rgb: torch.Tensor,
    feature_dim: int,
    data: st.DataObject,
) -> None:
    field, intrinsics = _make_field(anchor_rgb, stride=1, feature_dim=feature_dim)
    before = field.num_gaussians
    clone_indices = data.draw(
        st.lists(st.integers(min_value=0, max_value=before - 1), min_size=1, max_size=min(before, 4), unique=True)
    )

    field.clone_gaussians(torch.tensor(clone_indices, dtype=torch.long))
    _assert_aligned_lengths(field)
    assert field.num_gaussians == before + len(clone_indices)
    params = field.gaussian_params(intrinsics)
    assert torch.isfinite(params["means3d"]).all()


@DEFAULT_SETTINGS
@given(
    anchor_rgb=chw_images(min_side=2, max_side=6, min_value=0.05, max_value=0.95),
    feature_dim=st.integers(min_value=1, max_value=4),
    data=st.data(),
)
def test_split_gaussians_increases_count_and_keeps_finite_state(
    anchor_rgb: torch.Tensor,
    feature_dim: int,
    data: st.DataObject,
) -> None:
    field, intrinsics = _make_field(anchor_rgb, stride=1, feature_dim=feature_dim)
    before = field.num_gaussians
    split_indices = data.draw(
        st.lists(st.integers(min_value=0, max_value=before - 1), min_size=1, max_size=min(before, 4), unique=True)
    )

    field.split_gaussians(torch.tensor(split_indices, dtype=torch.long))
    _assert_aligned_lengths(field)
    assert field.num_gaussians == before + len(split_indices)
    params = field.gaussian_params(intrinsics)
    assert torch.isfinite(params["means3d"]).all()


def test_scale_aware_residual_head_matches_legacy_pointwise_mlp() -> None:
    class _LegacyResidualHead(torch.nn.Module):
        def __init__(self, feature_dim: int, hidden_dim: int, residual_scale: float) -> None:
            super().__init__()
            self.residual_scale = residual_scale
            self.net = torch.nn.Sequential(
                torch.nn.Linear(feature_dim + 4, hidden_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden_dim, 3),
            )

        def forward(self, latent_map: torch.Tensor, scale_x: float, scale_y: float) -> torch.Tensor:
            f_dim, h, w = latent_map.shape
            coords = pixel_grid(h, w, latent_map.device, latent_map.dtype, normalized=True).view(-1, 2)
            scale_token = latent_map.new_tensor([
                torch.log(latent_map.new_tensor(scale_x)),
                torch.log(latent_map.new_tensor(scale_y)),
            ])
            scale_token = scale_token.expand(coords.shape[0], 2)
            feats = latent_map.permute(1, 2, 0).reshape(-1, f_dim)
            residual = self.net(torch.cat((feats, coords, scale_token), dim=-1)).view(h, w, 3).permute(2, 0, 1)
            return self.residual_scale * torch.tanh(residual)

    feature_dim = 3
    hidden_dim = 7
    residual_scale = 0.15
    legacy = _LegacyResidualHead(feature_dim, hidden_dim, residual_scale)
    current = ScaleAwareResidualHead(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        residual_scale=residual_scale,
    )

    with torch.no_grad():
        current.net[0].weight.copy_(legacy.net[0].weight.view(hidden_dim, feature_dim + 4, 1, 1))
        current.net[0].bias.copy_(legacy.net[0].bias)
        current.net[2].weight.copy_(legacy.net[2].weight.view(hidden_dim, hidden_dim, 1, 1))
        current.net[2].bias.copy_(legacy.net[2].bias)
        current.net[4].weight.copy_(legacy.net[4].weight.view(3, hidden_dim, 1, 1))
        current.net[4].bias.copy_(legacy.net[4].bias)

    latent = torch.randn(feature_dim, 5, 4, dtype=torch.float32)
    expected = legacy(latent, scale_x=1.5, scale_y=0.75)
    actual = current(latent, scale_x=1.5, scale_y=0.75)

    torch.testing.assert_close(actual, expected, atol=1.0e-6, rtol=1.0e-6)


def test_density_mutations_preserve_parameter_identity_with_fixed_capacity() -> None:
    anchor_rgb = torch.full((3, 4, 4), 0.5, dtype=torch.float32)
    field_cfg = FieldConfig(anchor_stride=2, feature_dim=2, gaussian_capacity=8)
    focal, principal = _make_intrinsics(anchor_rgb.shape[-2], anchor_rgb.shape[-1])
    field = CanonicalGaussianField(
        anchor_rgb,
        focal,
        principal,
        field_cfg,
        AppearanceConfig(mode="constant"),
    )

    param_ids_before = {
        "means3d": id(field.means3d),
        "quat_raw": id(field.quat_raw),
        "log_scale": id(field.log_scale),
        "opacity_logit": id(field.opacity_logit),
        "rgb_logit": id(field.rgb_logit),
        "latent": id(field.latent),
    }
    capacity_before = field.gaussian_capacity

    field.clone_gaussians(torch.tensor([0], dtype=torch.long))
    field.split_gaussians(torch.tensor([1], dtype=torch.long))
    keep = torch.ones(field.num_gaussians, dtype=torch.bool)
    keep[-1] = False
    field.prune_keep_mask(keep)

    assert field.gaussian_capacity == capacity_before
    assert param_ids_before == {
        "means3d": id(field.means3d),
        "quat_raw": id(field.quat_raw),
        "log_scale": id(field.log_scale),
        "opacity_logit": id(field.opacity_logit),
        "rgb_logit": id(field.rgb_logit),
        "latent": id(field.latent),
    }
