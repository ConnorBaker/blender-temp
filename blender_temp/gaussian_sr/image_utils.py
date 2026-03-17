import math

import torch
import torch.nn.functional as F
from torch import Tensor

_PIXEL_GRID_CACHE: dict[tuple[int, int, bool, str, int | None, str], Tensor] = {}
_GAUSSIAN_KERNEL_CACHE: dict[tuple[int, float, int, str, int | None, str], Tensor] = {}


def _device_dtype_key(device: torch.device, dtype: torch.dtype) -> tuple[str, int | None, str]:
    return (device.type, device.index, str(dtype))


def _cache_allowed() -> bool:
    compiler = getattr(torch, "compiler", None)
    if compiler is not None and hasattr(compiler, "is_compiling") and compiler.is_compiling():
        return False
    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None and hasattr(dynamo, "is_compiling") and dynamo.is_compiling():
        return False
    return True


def pixel_grid(
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    normalized: bool = False,
) -> Tensor:
    use_cache = _cache_allowed()
    key = (int(height), int(width), bool(normalized), *_device_dtype_key(device, dtype))
    if use_cache:
        cached = _PIXEL_GRID_CACHE.get(key)
        if cached is not None:
            return cached

    ys = torch.arange(height, device=device, dtype=dtype)
    xs = torch.arange(width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    if normalized:
        xx = (xx + 0.5) / width * 2.0 - 1.0
        yy = (yy + 0.5) / height * 2.0 - 1.0
    grid = torch.stack((xx, yy), dim=-1).contiguous()
    if use_cache:
        _PIXEL_GRID_CACHE[key] = grid
    return grid


def downsample_image(img: Tensor, scale: float) -> Tensor:
    if scale == 1.0:
        return img
    h = max(1, int(round(img.shape[-2] * scale)))
    w = max(1, int(round(img.shape[-1] * scale)))
    return F.interpolate(img[None], size=(h, w), mode="area")[0]


def downsample_batch(images: Tensor, scale: float) -> Tensor:
    if scale == 1.0:
        return images
    h = max(1, int(round(images.shape[-2] * scale)))
    w = max(1, int(round(images.shape[-1] * scale)))
    return F.interpolate(images, size=(h, w), mode="area")


def make_gaussian_kernel(
    window_size: int,
    sigma: float,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    use_cache = _cache_allowed()
    key = (int(window_size), float(sigma), int(channels), *_device_dtype_key(device, dtype))
    if use_cache:
        cached = _GAUSSIAN_KERNEL_CACHE.get(key)
        if cached is not None:
            return cached

    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel = kernel_2d.expand(channels, 1, window_size, window_size).contiguous()
    if use_cache:
        _GAUSSIAN_KERNEL_CACHE[key] = kernel
    return kernel


def ssim_value(x: Tensor, y: Tensor, window_size: int = 11, sigma: float = 1.5) -> Tensor:
    assert x.shape == y.shape and x.dim() == 3
    c = x.shape[0]
    kernel = make_gaussian_kernel(window_size, sigma, c, x.device, x.dtype)
    pad = window_size // 2

    x_b = x.unsqueeze(0)
    y_b = y.unsqueeze(0)

    mu_x = F.conv2d(x_b, kernel, padding=pad, groups=c)
    mu_y = F.conv2d(y_b, kernel, padding=pad, groups=c)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x_b * x_b, kernel, padding=pad, groups=c) - mu_x2
    sigma_y2 = F.conv2d(y_b * y_b, kernel, padding=pad, groups=c) - mu_y2
    sigma_xy = F.conv2d(x_b * y_b, kernel, padding=pad, groups=c) - mu_xy

    c1 = 0.01**2
    c2 = 0.03**2
    luminance = (2.0 * mu_xy + c1) / (mu_x2 + mu_y2 + c1)
    contrast_structure = (2.0 * sigma_xy + c2) / (sigma_x2 + sigma_y2 + c2).clamp_min(torch.finfo(x.dtype).tiny)
    ssim_map = luminance * contrast_structure
    return ssim_map.mean().clamp(0.0, 1.0)


def charbonnier(x: Tensor, eps: float = 1.0e-3) -> Tensor:
    return torch.hypot(x, x.new_tensor(eps))


def tv_loss_grid(x: Tensor) -> Tensor:
    if x.dim() not in (2, 3):
        raise ValueError(f"Unsupported tensor rank for tv_loss_grid: {x.dim()}")
    dx = torch.diff(x, dim=-1)
    dy = torch.diff(x, dim=-2)
    return dx.abs().mean() + dy.abs().mean()


def estimate_phase_correlation_shift(ref_rgb: Tensor, src_rgb: Tensor) -> Tensor:
    ref = ref_rgb.mean(dim=0)
    src = src_rgb.mean(dim=0)
    f_ref = torch.fft.rfft2(ref)
    f_src = torch.fft.rfft2(src)
    cross = f_ref * f_src.conj()
    cross = cross / cross.abs().clamp_min(1.0e-8)
    corr = torch.fft.irfft2(cross, s=ref.shape)
    peak = corr.argmax()
    h, w = ref.shape
    peak_y, peak_x = torch.unravel_index(peak, (h, w))
    peak_y = peak_y.to(ref.dtype)
    peak_x = peak_x.to(ref.dtype)

    if peak_x > w // 2:
        peak_x = peak_x - w
    if peak_y > h // 2:
        peak_y = peak_y - h

    return torch.stack((peak_x, peak_y))


def estimate_translation_bootstrap(images: Tensor) -> Tensor:
    ref = images[0]
    shifts = []
    for v in range(images.shape[0]):
        if v == 0:
            shifts.append(torch.zeros(2, device=images.device, dtype=images.dtype))
        else:
            shifts.append(estimate_phase_correlation_shift(ref, images[v]))
    return torch.stack(shifts, dim=0)


__all__ = [
    "pixel_grid",
    "downsample_image",
    "downsample_batch",
    "make_gaussian_kernel",
    "ssim_value",
    "charbonnier",
    "tv_loss_grid",
    "estimate_phase_correlation_shift",
    "estimate_translation_bootstrap",
]
