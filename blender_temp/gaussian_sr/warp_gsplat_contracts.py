import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor


@dataclass(frozen=True)
class DataContracts:
    images_lr: str = "float32 cuda [V,3,H,W] contiguous"
    viewmats: str = "float32 cuda [V,4,4] contiguous"
    Ks: str = "float32 cuda [V,3,3] contiguous"

    means: str = "float32 cuda [N,3] contiguous"
    quat: str = "float32 cuda [N,4] contiguous (wxyz)"
    scale: str = "float32 cuda [N,3] contiguous"
    opacity: str = "float32 cuda [N] contiguous"
    values: str = "float32 cuda [N,C] contiguous"
    background: str = "float32 cuda [C] contiguous"
    out_values: str = "float32 cuda [H,W,C] contiguous"


@dataclass
class RasterConfig:
    tile_size: int = 16
    near_plane: float = 0.01
    far_plane: float = 1.0e10
    eps2d: float = 0.3
    radius_clip: float = 0.0
    max_sort_buffer_bytes: int | None = 2 * 1024 * 1024 * 1024
    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    alpha_min: float = 1.0e-6
    transmittance_eps: float = 1.0e-4
    clamp_alpha_max: float = 0.999
    depth_scale: float = 1.0e6
    sort_mode: Literal["auto", "warp_radix", "torch_sort"] = "auto"
    background_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0)
    record_projection_and_rasterize_on_tape: bool = True
    max_pixels_per_launch: int | None = None


def estimate_intersections(N: int, k: float) -> int:
    return int(math.ceil(N * k))


def estimate_tiles(width: int, height: int, tile_size: int) -> tuple[int, int, int]:
    tiles_x = (width + tile_size - 1) // tile_size
    tiles_y = (height + tile_size - 1) // tile_size
    return tiles_x, tiles_y, tiles_x * tiles_y


def estimate_buffer_bytes_for_example(
    N: int,
    k: float,
    width: int,
    height: int,
    tile_size: int,
    channels: int = 3,
    use_warp_radix: bool = True,
) -> dict[str, int]:
    M = estimate_intersections(N, k)
    _, _, tile_count = estimate_tiles(width, height, tile_size)

    keys_len = (2 * M) if use_warp_radix else M
    vals_len = (2 * M) if use_warp_radix else M

    sz: dict[str, int] = {}
    sz["M_intersections"] = M
    sz["tile_count"] = tile_count
    sz["keys_int64"] = keys_len * 8
    sz["values_int32"] = vals_len * 4
    sz["xys_u_float32"] = N * 4
    sz["xys_v_float32"] = N * 4
    sz["conic_a_float32"] = N * 4
    sz["conic_b_float32"] = N * 4
    sz["conic_c_float32"] = N * 4
    sz["rho_float32"] = N * 4
    sz["radius_int32"] = N * 4
    sz["num_tiles_hit_int32"] = N * 4
    sz["cum_tiles_hit_int32"] = N * 4
    sz["tile_min_xy_int32"] = 2 * N * 4
    sz["tile_max_xy_int32"] = 2 * N * 4
    sz["depth_key_int32"] = N * 4
    sz["render_values_float32"] = N * channels * 4
    sz["background_float32"] = channels * 4
    sz["tile_bins_start_int32"] = tile_count * 4
    sz["tile_bins_end_int32"] = tile_count * 4
    sz["out_values_float32"] = height * width * channels * 4
    sz["total_bytes_estimate_no_grads"] = sum(v for k, v in sz.items() if k.endswith(("int64", "int32", "float32")))
    return sz


KERNEL_MAPPING_TABLE = r"""
| Stage | gsplat concept | Warp component | Priority | Brief pseudocode |
|------:|----------------|----------------|----------|------------------|
| 1 | ProjectGaussians (Jacobian/EWA) | project_gaussians_kernel | High | xys, conic, radius, tile bbox, num_tiles |
| 2 | map_gaussian_to_intersects | map_to_intersects_kernel | High | write keys=(tile<<32)|depth, values=gid |
| 3 | sort intersections | warp.utils.radix_sort_pairs | High | stable radix sort (requires >=2*M capacity) |
| 3b | fallback sort | torch.sort | Medium | keys_sorted, idx = torch.sort(keys) |
| 4 | get_tile_bin_edges | get_tile_bin_edges_kernel | High | boundary detection by tile id |
| 5 | rasterize forward | rasterize_values_kernel | High | per pixel alpha composite over C channels |
| 6 | rasterize backward | wp.Tape OR custom kernel | Medium | tape.backward(grads={out: grad_out}) |
"""


MERMAID_PORT_FLOWCHART = r"""
```mermaid
flowchart TD
  A[Inputs: PyTorch tensors (LR images, viewmats, Ks, Gaussians)] --> B[Warp forward: project_gaussians]
  B --> C[num_tiles_hit, tile bounds, xys, conics]
  C --> D[Warp utils: array_scan (inclusive) -> cum_tiles_hit]
  D --> E[Warp forward: map_to_intersects -> packed keys + gaussian ids]
  E --> F{Sort mode}
  F -->|warp_radix| G[warp.utils.radix_sort_pairs(keys, vals), requires >=2*M]
  F -->|torch_sort| H[torch.sort(keys) + gather(vals)]
  G --> I[get_tile_bin_edges (Warp)]
  H --> I
  I --> J[Warp forward: rasterize_values]
  J --> K[Output tensor (H,W,C)]
  K --> L[Backward: wp.Tape.backward(grads={out: dL/dout})]
  L --> M[PyTorch optimizer step]
```
"""


def _assert_cuda_float32_contiguous(x: Tensor, name: str, shape: Sequence[int] | None = None) -> None:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x)}")
    if not x.is_cuda:
        raise ValueError(f"{name} must be on CUDA")
    if x.dtype != torch.float32:
        raise ValueError(f"{name} must be float32, got {x.dtype}")
    if not x.is_contiguous():
        raise ValueError(f"{name} must be contiguous()")
    if shape is not None and list(x.shape) != list(shape):
        raise ValueError(f"{name} must have shape {shape}, got {tuple(x.shape)}")


def _assert_1d(x: Tensor, name: str) -> None:
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {tuple(x.shape)}")


def _assert_same_len_1d(xs: Sequence[Tensor], names: Sequence[str]) -> int:
    if len(xs) != len(names):
        raise ValueError("internal error: length mismatch")
    n = xs[0].shape[0]
    for t, nm in zip(xs, names):
        _assert_1d(t, nm)
        if t.shape[0] != n:
            raise ValueError(f"{nm} length mismatch: expected {n}, got {t.shape[0]}")
    return int(n)


__all__ = [
    "DataContracts",
    "RasterConfig",
    "estimate_intersections",
    "estimate_tiles",
    "estimate_buffer_bytes_for_example",
    "KERNEL_MAPPING_TABLE",
    "MERMAID_PORT_FLOWCHART",
    "_assert_cuda_float32_contiguous",
    "_assert_1d",
    "_assert_same_len_1d",
]
