import torch
from hypothesis import given, strategies as st
from torch.testing import assert_close

from blender_temp.gaussian_sr.observation_model import (
    apply_observation_model,
    area_downsample_chw,
    area_downsample_hwc,
    observation_render_size,
)
from blender_temp.gaussian_sr.posefree_config import ObservationConfig

from .strategies import DEFAULT_SETTINGS, chw_images, unit_float32


@DEFAULT_SETTINGS
@given(
    h=st.integers(min_value=1, max_value=8),
    w=st.integers(min_value=1, max_value=8),
    out_h=st.integers(min_value=1, max_value=8),
    out_w=st.integers(min_value=1, max_value=8),
    value=unit_float32,
)
def test_area_downsample_chw_preserves_constant_images(
    h: int,
    w: int,
    out_h: int,
    out_w: int,
    value: float,
) -> None:
    image = torch.full((3, h, w), float(value), dtype=torch.float32)
    out = area_downsample_chw(image, out_h, out_w)
    expected = torch.full((3, out_h, out_w), float(value), dtype=torch.float32)
    assert_close(out, expected, atol=1.0e-6, rtol=0.0)


@DEFAULT_SETTINGS
@given(
    h=st.integers(min_value=1, max_value=8),
    w=st.integers(min_value=1, max_value=8),
    out_h=st.integers(min_value=1, max_value=8),
    out_w=st.integers(min_value=1, max_value=8),
    value=unit_float32,
)
def test_area_downsample_hwc_preserves_constant_images(
    h: int,
    w: int,
    out_h: int,
    out_w: int,
    value: float,
) -> None:
    image = torch.full((h, w, 3), float(value), dtype=torch.float32)
    out = area_downsample_hwc(image, out_h, out_w)
    expected = torch.full((out_h, out_w, 3), float(value), dtype=torch.float32)
    assert_close(out, expected, atol=1.0e-6, rtol=0.0)


@DEFAULT_SETTINGS
@given(
    target_h=st.integers(min_value=1, max_value=32),
    target_w=st.integers(min_value=1, max_value=32),
    supersample_factor=st.floats(min_value=1.0, max_value=4.0, allow_nan=False, allow_infinity=False, width=32),
)
def test_supersample_render_size_is_never_smaller(
    target_h: int,
    target_w: int,
    supersample_factor: float,
) -> None:
    cfg = ObservationConfig(mode="supersample_area", supersample_factor=float(supersample_factor))
    render_h, render_w = observation_render_size(target_h, target_w, cfg)
    assert render_h >= target_h
    assert render_w >= target_w


@DEFAULT_SETTINGS
@given(
    image=chw_images(min_side=1, max_side=8),
    out_h=st.integers(min_value=1, max_value=8),
    out_w=st.integers(min_value=1, max_value=8),
)
def test_apply_observation_model_area_returns_requested_shape(
    image: torch.Tensor,
    out_h: int,
    out_w: int,
) -> None:
    cfg = ObservationConfig(mode="area")
    out = apply_observation_model(image, out_h, out_w, cfg, layout="chw")
    assert out.shape == (image.shape[0], out_h, out_w)


@DEFAULT_SETTINGS
@given(image=chw_images(min_side=1, max_side=8))
def test_identity_observation_returns_input_when_shape_matches(image: torch.Tensor) -> None:
    cfg = ObservationConfig(mode="identity")
    out = apply_observation_model(image, image.shape[-2], image.shape[-1], cfg, layout="chw")
    assert_close(out, image, atol=0.0, rtol=0.0)
