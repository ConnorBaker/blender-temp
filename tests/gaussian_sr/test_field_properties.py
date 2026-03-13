import torch
from hypothesis import given, strategies as st

from blender_temp.gaussian_sr.field import CanonicalGaussianField
from blender_temp.gaussian_sr.posefree_config import AppearanceConfig, FieldConfig

from .strategies import DEFAULT_SETTINGS, chw_images


def _make_intrinsics(height: int, width: int) -> torch.Tensor:
    return torch.tensor(
        [
            float(max(width, 1)),
            float(max(height, 1)),
            (width - 1.0) * 0.5,
            (height - 1.0) * 0.5,
        ],
        dtype=torch.float32,
    )


def _make_field(anchor_rgb: torch.Tensor, stride: int, feature_dim: int) -> tuple[CanonicalGaussianField, torch.Tensor]:
    field_cfg = FieldConfig(anchor_stride=stride, feature_dim=feature_dim)
    appearance_cfg = AppearanceConfig(mode="constant")
    intrinsics = _make_intrinsics(anchor_rgb.shape[-2], anchor_rgb.shape[-1])
    field = CanonicalGaussianField(anchor_rgb, intrinsics, field_cfg, appearance_cfg)
    return field, intrinsics


def _assert_aligned_lengths(field: CanonicalGaussianField) -> None:
    n = field.num_gaussians
    assert field.uv.shape[0] == n
    assert field.depth_raw.shape[0] == n
    assert field.xyz_offset.shape[0] == n
    assert field.quat_raw.shape[0] == n
    assert field.log_scale.shape[0] == n
    assert field.opacity_logit.shape[0] == n
    assert field.rgb_logit.shape[0] == n
    assert field.latent.shape[0] == n
    if field.sh_coeffs is not None:
        assert field.sh_coeffs.shape[0] == n


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
