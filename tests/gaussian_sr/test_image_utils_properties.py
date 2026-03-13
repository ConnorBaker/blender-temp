import torch
from hypothesis import given
from torch.testing import assert_close

from blender_temp.gaussian_sr.image_utils import charbonnier, make_gaussian_kernel, pixel_grid, ssim_value

from .strategies import DEFAULT_SETTINGS, chw_images, finite_vectors, same_shape_chw_image_pairs


@DEFAULT_SETTINGS
@given(x=finite_vectors())
def test_charbonnier_is_even_nonnegative_and_finite(x: torch.Tensor) -> None:
    y = charbonnier(x)
    y_neg = charbonnier(-x)

    assert torch.isfinite(y).all()
    assert (y >= 0.0).all()
    assert_close(y, y_neg, atol=1.0e-6, rtol=1.0e-6)


@DEFAULT_SETTINGS
@given(pair=same_shape_chw_image_pairs())
def test_ssim_is_symmetric_and_bounded(pair: tuple[torch.Tensor, torch.Tensor]) -> None:
    x, y = pair
    sxy = ssim_value(x, y)
    syx = ssim_value(y, x)

    assert torch.isfinite(sxy)
    assert 0.0 <= float(sxy.item()) <= 1.0
    assert_close(sxy, syx, atol=1.0e-5, rtol=1.0e-5)


@DEFAULT_SETTINGS
@given(x=chw_images(min_side=3, max_side=8))
def test_ssim_of_identical_images_is_one(x: torch.Tensor) -> None:
    out = ssim_value(x, x, window_size=3, sigma=1.0)
    expected = torch.tensor(1.0, dtype=x.dtype)
    assert_close(out, expected, atol=1.0e-4, rtol=1.0e-4)


def test_pixel_grid_reuses_cached_tensor() -> None:
    grid_a = pixel_grid(4, 5, device=torch.device("cpu"), dtype=torch.float32, normalized=True)
    grid_b = pixel_grid(4, 5, device=torch.device("cpu"), dtype=torch.float32, normalized=True)

    assert grid_a.data_ptr() == grid_b.data_ptr()
    assert_close(grid_a, grid_b, atol=0.0, rtol=0.0)


def test_gaussian_kernel_reuses_cached_tensor() -> None:
    kernel_a = make_gaussian_kernel(5, 1.5, 3, device=torch.device("cpu"), dtype=torch.float32)
    kernel_b = make_gaussian_kernel(5, 1.5, 3, device=torch.device("cpu"), dtype=torch.float32)

    assert kernel_a.data_ptr() == kernel_b.data_ptr()
    assert_close(kernel_a, kernel_b, atol=0.0, rtol=0.0)
