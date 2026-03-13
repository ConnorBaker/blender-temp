import math

import pytest
import torch
from hypothesis import given, strategies as st

from blender_temp.gaussian_sr.warp_gsplat_contracts import (
    _assert_1d,
    _assert_cuda_float32_contiguous,
    _assert_same_len_1d,
    estimate_buffer_bytes_for_example,
    estimate_intersections,
    estimate_tiles,
)

from .strategies import DEFAULT_SETTINGS


@DEFAULT_SETTINGS
@given(
    width=st.integers(min_value=1, max_value=2048),
    height=st.integers(min_value=1, max_value=2048),
    tile_size=st.integers(min_value=1, max_value=128),
)
def test_estimate_tiles_matches_ceil_division(width: int, height: int, tile_size: int) -> None:
    tiles_x, tiles_y, tile_count = estimate_tiles(width, height, tile_size)

    assert tiles_x == math.ceil(width / tile_size)
    assert tiles_y == math.ceil(height / tile_size)
    assert tile_count == tiles_x * tiles_y


@DEFAULT_SETTINGS
@given(
    n=st.integers(min_value=0, max_value=10000),
    k=st.floats(min_value=0.0, max_value=32.0, allow_nan=False, allow_infinity=False, width=32),
)
def test_estimate_intersections_is_ceiling_product(n: int, k: float) -> None:
    out = estimate_intersections(n, float(k))
    assert out == math.ceil(n * float(k))
    assert out >= 0


@DEFAULT_SETTINGS
@given(
    n=st.integers(min_value=1, max_value=1000),
    k=st.floats(min_value=0.0, max_value=8.0, allow_nan=False, allow_infinity=False, width=32),
    width=st.integers(min_value=1, max_value=128),
    height=st.integers(min_value=1, max_value=128),
    tile_size=st.integers(min_value=1, max_value=32),
    channels_a=st.integers(min_value=1, max_value=4),
    channels_b=st.integers(min_value=5, max_value=8),
)
def test_buffer_estimate_grows_with_channel_count(
    n: int,
    k: float,
    width: int,
    height: int,
    tile_size: int,
    channels_a: int,
    channels_b: int,
) -> None:
    small = estimate_buffer_bytes_for_example(n, float(k), width, height, tile_size, channels=channels_a)
    large = estimate_buffer_bytes_for_example(n, float(k), width, height, tile_size, channels=channels_b)

    assert large["render_values_float32"] >= small["render_values_float32"]
    assert large["background_float32"] >= small["background_float32"]
    assert large["out_values_float32"] >= small["out_values_float32"]
    assert large["total_bytes_estimate_no_grads"] >= small["total_bytes_estimate_no_grads"]


def test_assert_same_len_1d_accepts_matching_inputs() -> None:
    xs = [torch.zeros(4), torch.ones(4)]
    out = _assert_same_len_1d(xs, ["a", "b"])
    assert out == 4


def test_assert_same_len_1d_rejects_mismatch() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        _assert_same_len_1d([torch.zeros(4), torch.ones(5)], ["a", "b"])


def test_assert_1d_rejects_non_vector() -> None:
    with pytest.raises(ValueError, match="must be 1D"):
        _assert_1d(torch.zeros(2, 2), "x")


def test_assert_cuda_float32_contiguous_rejects_cpu_tensor() -> None:
    with pytest.raises(ValueError, match="must be on CUDA"):
        _assert_cuda_float32_contiguous(torch.zeros(3, dtype=torch.float32), "x")
