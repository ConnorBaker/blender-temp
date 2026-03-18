from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Any

import helion  # type: ignore[import-untyped]
import helion.language as hl  # type: ignore[import-untyped]
import torch
from torch import Tensor

from .reference_renderer import (
    project_gaussians_reference,
    render_values_from_prepared_reference,
    render_values_reference,
)
from .renderer_host_prep import prepare_visibility_from_projection, projection_meta_from_projection
from .warp_gsplat_autograd import PreparedVisibility
from .warp_gsplat_contracts import RasterConfig


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


_REDUCED_PRECISION_DTYPES = frozenset({torch.float32, torch.bfloat16})


def _validate_helion_render_inputs(
    means: Tensor,
    quat: Tensor,
    scale: Tensor,
    values: Tensor,
    opacity: Tensor,
    background: Tensor,
    viewmat: Tensor,
    K: Tensor,
) -> None:
    all_tensors = {
        "means": means,
        "quat": quat,
        "scale": scale,
        "values": values,
        "opacity": opacity,
        "background": background,
        "viewmat": viewmat,
        "K": K,
    }
    for name, tensor in all_tensors.items():
        if tensor.device.type != "cuda":
            raise ValueError(f"{name} must be on CUDA for Helion backend")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous() for Helion backend")

    # Projection inputs must stay FP32: quaternion cancellation, determinant
    # subtraction, Jacobian overflow, and sub-pixel xys precision (BF16 ULP
    # at 1920 is ~8) all require full precision.  viewmat/K participate in
    # the same projection math.
    fp32_only = {"means": means, "quat": quat, "scale": scale, "viewmat": viewmat, "K": K}
    for name, tensor in fp32_only.items():
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be float32 for Helion backend, got {tensor.dtype}")

    # values and background may be FP32 or BF16.  These tensors have bounded
    # ranges ([0,1] for RGB, moderate for latent) and are loaded into FP32
    # arithmetic via type promotion from FP32 pixel coordinates and FP32
    # accumulators, so BF16 storage is safe.
    # opacity may arrive as FP32 (from field sigmoid) even when values is BF16;
    # the forward impl downcasts it to match values before calling the kernel.
    reduced_ok = {"values": values, "opacity": opacity, "background": background}
    for name, tensor in reduced_ok.items():
        if tensor.dtype not in _REDUCED_PRECISION_DTYPES:
            raise ValueError(
                f"{name} must be float32 or bfloat16 for Helion backend, got {tensor.dtype}"
            )
    # Consistency: values and background must share the same dtype so that the
    # kernel output dtype and background blending (accum + trans * bg) are
    # consistent.  opacity is allowed to differ since it's downcast later.
    if values.dtype != background.dtype:
        raise ValueError(
            f"values and background must share the same dtype, "
            f"got values={values.dtype}, background={background.dtype}"
        )

    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError(f"means must have shape [N, 3], got {tuple(means.shape)}")
    if quat.ndim != 2 or quat.shape[1] != 4:
        raise ValueError(f"quat must have shape [N, 4], got {tuple(quat.shape)}")
    if scale.ndim != 2 or scale.shape[1] != 3:
        raise ValueError(f"scale must have shape [N, 3], got {tuple(scale.shape)}")
    if values.ndim != 2:
        raise ValueError(f"values must have shape [N, C], got {tuple(values.shape)}")
    if opacity.ndim != 1 or opacity.shape[0] != means.shape[0]:
        raise ValueError(f"opacity must have shape [N], got {tuple(opacity.shape)}")
    if values.shape[0] != means.shape[0]:
        raise ValueError("values length must match means length")
    if background.ndim != 1 or background.shape[0] != values.shape[1]:
        raise ValueError("background length must match values channels")
    if viewmat.shape != (4, 4):
        raise ValueError(f"viewmat must have shape [4,4], got {tuple(viewmat.shape)}")
    if K.shape != (3, 3):
        raise ValueError(f"K must have shape [3,3], got {tuple(K.shape)}")


# ---------------------------------------------------------------------------
# Helion config / kernel cache management
# ---------------------------------------------------------------------------


def _helion_config_dir() -> Path:
    raw = os.environ.get("BLENDER_TEMP_HELION_CONFIG_DIR")
    if raw is not None and raw.strip() != "":
        return Path(raw).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "configs" / "helion"


@lru_cache(maxsize=None)
def _load_saved_helion_configs(kernel_name: str) -> tuple[helion.Config, ...]:
    config_dir = _helion_config_dir()
    if not config_dir.exists():
        return ()
    paths = sorted(config_dir.glob(f"{kernel_name}*.json"))
    return tuple(helion.Config.load(path) for path in paths)


def _kernel_kwargs(static_shapes: bool, runtime_autotune: bool, *, kernel_name: str) -> dict[str, object]:
    kwargs: dict[str, object] = {"static_shapes": bool(static_shapes)}
    saved_configs = _load_saved_helion_configs(kernel_name)
    if len(saved_configs) == 1:
        kwargs["config"] = saved_configs[0]
    elif len(saved_configs) > 1:
        kwargs["configs"] = list(saved_configs)
    elif not runtime_autotune:
        kwargs["autotune_effort"] = "none"
    return kwargs


def _helion_raster_chunk_size() -> int:
    raw = os.environ.get("BLENDER_TEMP_HELION_RASTER_CHUNK", "4")
    value = int(raw)
    if value <= 0:
        raise ValueError(f"BLENDER_TEMP_HELION_RASTER_CHUNK must be > 0, got {value}")
    return value


def _helion_raster_backward_chunk_size() -> int:
    raw = os.environ.get("BLENDER_TEMP_HELION_RASTER_BACKWARD_CHUNK", "1")
    value = int(raw)
    if value <= 0:
        raise ValueError(f"BLENDER_TEMP_HELION_RASTER_BACKWARD_CHUNK must be > 0, got {value}")
    return value


def clear_helion_kernel_cache() -> None:
    """Clear all cached Helion kernel compilations and saved configs."""
    _load_saved_helion_configs.cache_clear()
    _make_batched_raster_forward_kernel.cache_clear()
    _make_visibility_stats_kernel.cache_clear()
    _make_raster_backward_kernel.cache_clear()


# ---------------------------------------------------------------------------
# Intersection capacity reservation
# ---------------------------------------------------------------------------

_HELION_INTERSECTION_CAPACITY: dict[tuple[str, int, int], int] = {}


def _helion_intersection_capacity(device: torch.device, width: int, height: int) -> int | None:
    key = (str(device), int(width), int(height))
    return _HELION_INTERSECTION_CAPACITY.get(key)


def reserve_helion_intersection_capacity(
    *,
    device: torch.device | str,
    width: int,
    height: int,
    required_count: int,
) -> int:
    """Pre-reserve intersection buffer capacity for the Helion rasterizer.

    Returns the new capacity (which may be larger than *required_count* if a
    previous reservation was bigger).
    """
    dev_str = str(device)
    key = (dev_str, int(width), int(height))
    existing = _HELION_INTERSECTION_CAPACITY.get(key, 0)
    new_cap = max(existing, int(required_count))
    _HELION_INTERSECTION_CAPACITY[key] = new_cap
    return new_cap


# ---------------------------------------------------------------------------
# Padding helpers  (stabilise PreparedVisibility for fixed-capacity kernels)
# ---------------------------------------------------------------------------


def _pad_rows(t: Tensor, target_rows: int) -> Tensor:
    """Pad the first dimension of *t* with zeros to *target_rows*."""
    current = t.shape[0]
    if current >= target_rows:
        return t
    pad_shape = (target_rows - current, *t.shape[1:])
    return torch.cat([t, torch.zeros(pad_shape, device=t.device, dtype=t.dtype)], dim=0)


def _pad_vector(t: Tensor, target_len: int) -> Tensor:
    """Pad a 1-D tensor with zeros to *target_len*."""
    current = t.shape[0]
    if current >= target_len:
        return t
    return torch.cat([t, torch.zeros(target_len - current, device=t.device, dtype=t.dtype)], dim=0)


def _stabilize_prepared_visibility(prepared: PreparedVisibility) -> PreparedVisibility:
    """Ensure the prepared visibility has stable (padded) tensor sizes.

    If a reservation was made via ``reserve_helion_intersection_capacity``, the
    sorted_vals buffer is padded up to that capacity so that Helion can keep
    compiled kernels with static shapes across frames.
    """
    reserved = _helion_intersection_capacity(prepared.device, prepared.width, prepared.height)
    if reserved is None or reserved <= prepared.intersection_capacity:
        return prepared
    new_sorted = _pad_vector(prepared.sorted_vals, reserved)
    return PreparedVisibility(
        xys=prepared.xys,
        conic=prepared.conic,
        rho=prepared.rho,
        num_tiles_hit=prepared.num_tiles_hit,
        tile_start=prepared.tile_start,
        tile_end=prepared.tile_end,
        sorted_vals=new_sorted,
        width=prepared.width,
        height=prepared.height,
        tile_size=prepared.tile_size,
        tiles_x=prepared.tiles_x,
        tiles_y=prepared.tiles_y,
        tile_count=prepared.tile_count,
        gaussian_count_value=prepared.gaussian_count,
        intersection_count_value=prepared.intersection_count,
    )


# ---------------------------------------------------------------------------
# Batched raster forward kernel (V views in one launch)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _make_batched_raster_forward_kernel(*, static_shapes: bool, runtime_autotune: bool):
    kwargs = _kernel_kwargs(
        static_shapes=static_shapes,
        runtime_autotune=runtime_autotune,
        kernel_name="batched_raster_forward",
    )

    @helion.kernel(**kwargs)
    def batched_raster_forward(
        tile_start: Tensor,       # [V, T] per-view tile starts
        tile_end: Tensor,         # [V, T] per-view tile ends
        sorted_vals: Tensor,      # [V, M_max] per-view intersection lists
        pixel_x: Tensor,          # [W] shared pixel coords
        pixel_y: Tensor,          # [H]
        xys: Tensor,              # [V, N, 2] per-view projected positions (FP32)
        conic: Tensor,            # [V, N, 3] per-view conics
        rho: Tensor,              # [V, N] per-view rho
        values_flat: Tensor,      # [V, N*C] per-view flattened values
        opacity: Tensor,          # [V, N] per-view opacity
        background: Tensor,       # [C] shared background
        total_tiles: hl.constexpr,
        tiles_per_view: hl.constexpr,
        tiles_x: hl.constexpr,
        tile_size: hl.constexpr,
        antialiased_flag: hl.constexpr,
        background_is_zero_flag: hl.constexpr,
        chunk_size_flag: hl.constexpr,
        alpha_min: hl.constexpr,
        trans_eps: hl.constexpr,
        clamp_alpha_max: hl.constexpr,
        pixels_per_view: hl.constexpr,
    ) -> tuple[Tensor, Tensor]:
        height = pixel_y.size(0)
        width = pixel_x.size(0)
        channels = background.size(0)
        pixels_per_tile = tile_size * tile_size
        tiles_per_view = hl.specialize(tiles_per_view)
        tiles_x = hl.specialize(tiles_x)
        tile_size = hl.specialize(tile_size)
        chunk_size_static = hl.specialize(chunk_size_flag)
        antialiased = hl.specialize(antialiased_flag)
        background_is_zero = hl.specialize(background_is_zero_flag)
        pixels_per_view = hl.specialize(pixels_per_view)
        num_views = tile_start.size(0)

        out_flat = torch.empty(
            [num_views * height * width * channels],
            device=values_flat.device, dtype=values_flat.dtype,
        )
        final_T_flat = torch.empty(
            [num_views * height * width],
            device=values_flat.device, dtype=torch.float32,
        )

        for tile_tid in hl.tile(total_tiles):
            view_id = tile_tid.index // tiles_per_view
            local_tid = tile_tid.index - view_id * tiles_per_view
            screen_tile_y = local_tid // tiles_x
            screen_tile_x = local_tid - screen_tile_y * tiles_x

            starts = hl.load(tile_start, [view_id, local_tid])
            ends = hl.load(tile_end, [view_id, local_tid])
            nnz = ends - starts
            max_nnz = nnz.amax()

            for tile_pix in hl.tile(pixels_per_tile):
                py_off = tile_pix.index // tile_size
                px_off = tile_pix.index - py_off * tile_size
                py_idx = screen_tile_y[:, None] * tile_size + py_off[None, :]
                px_idx = screen_tile_x[:, None] * tile_size + px_off[None, :]
                pixel_valid = (py_idx < height) & (px_idx < width)
                fy = hl.load(pixel_y, [py_idx], extra_mask=pixel_valid).to(torch.float32)
                fx = hl.load(pixel_x, [px_idx], extra_mask=pixel_valid).to(torch.float32)
                pixel_flat = py_idx * width + px_idx
                trans = hl.full([tile_tid, tile_pix], 1.0, dtype=torch.float32)

                for tile_c in hl.tile(0, channels):
                    channel_valid = tile_c.index < channels
                    accum = hl.zeros([tile_tid, tile_pix, tile_c], dtype=torch.float32)
                    trans_local = trans

                    for tile_k in hl.tile(0, max_nnz, block_size=chunk_size_static):
                        for k_inner in hl.static_range(chunk_size_static):
                            k_abs = tile_k.begin + k_inner
                            valid_k = k_abs < nnz
                            intersect_idx = starts + k_abs
                            gid = hl.load(sorted_vals, [view_id, intersect_idx], extra_mask=valid_k).to(torch.int64)
                            gid = torch.where(valid_k, gid, 0)

                            xy0 = hl.load(xys, [view_id, gid, 0], extra_mask=valid_k).to(torch.float32)
                            xy1 = hl.load(xys, [view_id, gid, 1], extra_mask=valid_k).to(torch.float32)
                            conic0 = hl.load(conic, [view_id, gid, 0], extra_mask=valid_k)
                            conic1 = hl.load(conic, [view_id, gid, 1], extra_mask=valid_k)
                            conic2 = hl.load(conic, [view_id, gid, 2], extra_mask=valid_k)
                            rho_k = hl.load(rho, [view_id, gid], extra_mask=valid_k)
                            opacity_k = hl.load(opacity, [view_id, gid], extra_mask=valid_k)

                            dx = fx - xy0[:, None]
                            dy = fy - xy1[:, None]
                            sigma = (
                                0.5 * (conic0[:, None] * dx * dx + conic2[:, None] * dy * dy)
                                + conic1[:, None] * dx * dy
                            )
                            alpha = opacity_k[:, None] * torch.exp(-sigma)
                            if antialiased != 0:
                                alpha = alpha * rho_k[:, None]
                            valid_mask = pixel_valid & valid_k[:, None]
                            alpha = torch.where(valid_mask, alpha, 0.0)
                            alpha = torch.where(alpha > clamp_alpha_max, clamp_alpha_max, alpha)
                            accepted = valid_mask & (alpha >= alpha_min) & (trans_local >= trans_eps)
                            weight = torch.where(accepted, trans_local * alpha, 0.0)

                            value_mask = valid_k[:, None] & channel_valid[None, :]
                            value_idx = gid[:, None] * channels + tile_c.index[None, :]
                            view_val_offset = view_id[:, None] + hl.zeros([tile_tid, tile_c], dtype=torch.int64)
                            value_k = hl.load(values_flat, [view_val_offset, value_idx], extra_mask=value_mask)
                            value_k = torch.where(value_mask, value_k, 0.0)
                            accum = accum + weight[:, :, None] * value_k[:, None, :]
                            trans_local = torch.where(accepted, trans_local * (1.0 - alpha), trans_local)

                    view_pixel_offset = view_id[:, None, None] * pixels_per_view
                    out_idx = view_pixel_offset * channels + pixel_flat[:, :, None] * channels + tile_c.index[None, None, :]
                    out_mask = pixel_valid[:, :, None] & channel_valid[None, None, :]
                    if background_is_zero != 0:
                        out_chunk = accum.to(values_flat.dtype)
                    else:
                        bg = hl.load(background, [tile_c.index], extra_mask=channel_valid)
                        bg = torch.where(channel_valid, bg, 0.0)
                        out_chunk = (accum + trans_local[:, :, None] * bg[None, None, :]).to(values_flat.dtype)
                    hl.store(out_flat, [out_idx], out_chunk, extra_mask=out_mask)
                    trans = trans_local

                ft_idx = view_id[:, None] * pixels_per_view + pixel_flat
                hl.store(final_T_flat, [ft_idx], trans, extra_mask=pixel_valid)

        return out_flat, final_T_flat

    return batched_raster_forward


# ---------------------------------------------------------------------------
# Visibility stats kernel
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _make_visibility_stats_kernel(*, static_shapes: bool, runtime_autotune: bool):
    kwargs = _kernel_kwargs(
        static_shapes=static_shapes,
        runtime_autotune=runtime_autotune,
        kernel_name="visibility_stats",
    )

    @helion.kernel(**kwargs)
    def visibility_stats(
        tile_start: Tensor,       # [V, T]
        tile_end: Tensor,         # [V, T]
        sorted_vals: Tensor,      # [V, M_max]
        pixel_x: Tensor,          # [W]
        pixel_y: Tensor,          # [H]
        xys: Tensor,              # [V, N, 2]
        conic: Tensor,            # [V, N, 3]
        rho: Tensor,              # [V, N]
        opacity: Tensor,          # [V, N]
        residual_map: Tensor,     # [V, H, W]
        total_tiles: hl.constexpr,
        tiles_per_view: hl.constexpr,
        tiles_x: hl.constexpr,
        tile_size: hl.constexpr,
        antialiased_flag: hl.constexpr,
        chunk_size_flag: hl.constexpr,
        error_bins_x: hl.constexpr,
        error_bins_y: hl.constexpr,
        alpha_min: hl.constexpr,
        trans_eps: hl.constexpr,
        clamp_alpha_max: hl.constexpr,
        gaussian_count: hl.constexpr,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        height = pixel_y.size(0)
        width = pixel_x.size(0)
        pixels_per_tile = tile_size * tile_size
        tiles_per_view = hl.specialize(tiles_per_view)
        tiles_x = hl.specialize(tiles_x)
        tile_size = hl.specialize(tile_size)
        antialiased = hl.specialize(antialiased_flag)
        chunk_size_static = hl.specialize(chunk_size_flag)
        gaussian_count = hl.specialize(gaussian_count)

        # Shared output buffers — atomics sum across all V views.
        contrib = torch.zeros([gaussian_count], device=opacity.device, dtype=torch.float32)
        trans = torch.zeros_like(contrib)
        hits = torch.zeros_like(contrib)
        residual = torch.zeros_like(contrib)
        error_map = torch.zeros([gaussian_count, error_bins_x * error_bins_y], device=opacity.device, dtype=torch.float32)

        for tile_tid in hl.tile(total_tiles):
            view_id = tile_tid.index // tiles_per_view
            local_tid = tile_tid.index - view_id * tiles_per_view
            screen_tile_y = local_tid // tiles_x
            screen_tile_x = local_tid - screen_tile_y * tiles_x

            starts = hl.load(tile_start, [view_id, local_tid])
            ends = hl.load(tile_end, [view_id, local_tid])
            nnz = ends - starts
            max_nnz = nnz.amax()

            for tile_pix in hl.tile(pixels_per_tile):
                py_off = tile_pix.index // tile_size
                px_off = tile_pix.index - py_off * tile_size
                py_idx = screen_tile_y[:, None] * tile_size + py_off[None, :]
                px_idx = screen_tile_x[:, None] * tile_size + px_off[None, :]
                pixel_valid = (py_idx < height) & (px_idx < width)
                fy = hl.load(pixel_y, [py_idx], extra_mask=pixel_valid).to(torch.float32)
                fx = hl.load(pixel_x, [px_idx], extra_mask=pixel_valid).to(torch.float32)
                pixel_residual = hl.load(residual_map, [view_id, py_idx, px_idx], extra_mask=pixel_valid).to(torch.float32)
                bx = (px_idx * error_bins_x) // width
                by = (py_idx * error_bins_y) // height
                bin_idx = by * error_bins_x + bx

                T = hl.full([tile_tid, tile_pix], 1.0, dtype=torch.float32)
                for tile_k in hl.tile(0, max_nnz, block_size=chunk_size_static):
                    for k_inner in hl.static_range(chunk_size_static):
                        k_abs = tile_k.begin + k_inner
                        valid_k = k_abs < nnz
                        intersect_idx = starts + k_abs
                        gid = hl.load(sorted_vals, [view_id, intersect_idx], extra_mask=valid_k).to(torch.int64)
                        gid = torch.where(valid_k, gid, 0)

                        xy0 = hl.load(xys, [view_id, gid, 0], extra_mask=valid_k).to(torch.float32)
                        xy1 = hl.load(xys, [view_id, gid, 1], extra_mask=valid_k).to(torch.float32)
                        conic0 = hl.load(conic, [view_id, gid, 0], extra_mask=valid_k)
                        conic1 = hl.load(conic, [view_id, gid, 1], extra_mask=valid_k)
                        conic2 = hl.load(conic, [view_id, gid, 2], extra_mask=valid_k)
                        rho_k = hl.load(rho, [view_id, gid], extra_mask=valid_k)
                        opacity_k = hl.load(opacity, [view_id, gid], extra_mask=valid_k)

                        dx = fx - xy0[:, None]
                        dy = fy - xy1[:, None]
                        sigma = (
                            0.5 * (conic0[:, None] * dx * dx + conic2[:, None] * dy * dy)
                            + conic1[:, None] * dx * dy
                        )
                        alpha = opacity_k[:, None] * torch.exp(-sigma)
                        if antialiased != 0:
                            alpha = alpha * rho_k[:, None]
                        valid_mask = pixel_valid & valid_k[:, None]
                        alpha = torch.where(valid_mask, alpha, 0.0)
                        alpha = torch.where(alpha > clamp_alpha_max, clamp_alpha_max, alpha)
                        accepted = valid_mask & (alpha >= alpha_min) & (T >= trans_eps)
                        w_i = torch.where(accepted, T * alpha, 0.0)
                        trans_i = torch.where(accepted, T, 0.0)
                        hit_i = torch.where(accepted, 1.0, 0.0)
                        residual_i = w_i * pixel_residual
                        reduced_w_i = torch.sum(w_i, dim=-1)
                        reduced_trans_i = torch.sum(trans_i, dim=-1)
                        reduced_hit_i = torch.sum(hit_i, dim=-1)
                        reduced_residual_i = torch.sum(residual_i, dim=-1)
                        hl.atomic_add(contrib, [gid], reduced_w_i)
                        hl.atomic_add(trans, [gid], reduced_trans_i)
                        hl.atomic_add(hits, [gid], reduced_hit_i)
                        hl.atomic_add(residual, [gid], reduced_residual_i)
                        gid_idx = gid[:, None] + hl.zeros([tile_tid, tile_pix], dtype=torch.int64)
                        hl.atomic_add(error_map, [gid_idx, bin_idx], residual_i)
                        T = torch.where(accepted, T * (1.0 - alpha), T)
        return contrib, trans, hits, residual, error_map

    return visibility_stats


# ---------------------------------------------------------------------------
# Raster backward kernel
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _make_raster_backward_kernel(*, static_shapes: bool, runtime_autotune: bool):
    kwargs = _kernel_kwargs(
        static_shapes=static_shapes,
        runtime_autotune=runtime_autotune,
        kernel_name="raster_backward",
    )

    @helion.kernel(**kwargs)
    def raster_backward(
        tile_start: Tensor,       # [V, T]
        tile_end: Tensor,         # [V, T]
        sorted_vals: Tensor,      # [V, M_max]
        pixel_x: Tensor,          # [W]
        pixel_y: Tensor,          # [H]
        xys: Tensor,              # [V, N, 2]
        conic: Tensor,            # [V, N, 3]
        rho: Tensor,              # [V, N]
        values_flat: Tensor,      # [V, N*C]
        opacity: Tensor,          # [V, N]
        background: Tensor,       # [C]
        final_T_flat: Tensor,     # [V, H*W]
        grad_out_flat: Tensor,    # [V, H*W*C]
        total_tiles: hl.constexpr,
        tiles_per_view: hl.constexpr,
        tiles_x: hl.constexpr,
        tile_size: hl.constexpr,
        antialiased_flag: hl.constexpr,
        background_is_zero_flag: hl.constexpr,
        chunk_size_flag: hl.constexpr,
        alpha_min: hl.constexpr,
        trans_eps: hl.constexpr,
        clamp_alpha_max: hl.constexpr,
        gaussian_count: hl.constexpr,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        height = pixel_y.size(0)
        width = pixel_x.size(0)
        channels = background.size(0)
        pixels_per_tile = tile_size * tile_size
        tiles_per_view = hl.specialize(tiles_per_view)
        tiles_x = hl.specialize(tiles_x)
        tile_size = hl.specialize(tile_size)
        chunk_size_static = hl.specialize(chunk_size_flag)
        antialiased = hl.specialize(antialiased_flag)
        background_is_zero = hl.specialize(background_is_zero_flag)
        gaussian_count = hl.specialize(gaussian_count)

        # Gradient buffers: FP32, shared across all views.
        # Atomic adds from V views naturally sum the multi-view gradients.
        grad_xys_flat = torch.zeros([gaussian_count * 2], device=xys.device, dtype=torch.float32)
        grad_conic_flat = torch.zeros([gaussian_count * 3], device=conic.device, dtype=torch.float32)
        grad_rho = torch.zeros([gaussian_count], device=rho.device, dtype=torch.float32)
        grad_values_flat = torch.zeros([gaussian_count * channels], device=values_flat.device, dtype=torch.float32)
        grad_opacity = torch.zeros([gaussian_count], device=opacity.device, dtype=torch.float32)

        for tile_tid in hl.tile(total_tiles):
            view_id = tile_tid.index // tiles_per_view
            local_tid = tile_tid.index - view_id * tiles_per_view
            screen_tile_y = local_tid // tiles_x
            screen_tile_x = local_tid - screen_tile_y * tiles_x

            starts = hl.load(tile_start, [view_id, local_tid])
            ends = hl.load(tile_end, [view_id, local_tid])
            nnz = ends - starts
            max_nnz = nnz.amax()

            for tile_pix in hl.tile(pixels_per_tile):
                py_off = tile_pix.index // tile_size
                px_off = tile_pix.index - py_off * tile_size
                py_idx = screen_tile_y[:, None] * tile_size + py_off[None, :]
                px_idx = screen_tile_x[:, None] * tile_size + px_off[None, :]
                pixel_valid = (py_idx < height) & (px_idx < width)
                fy = hl.load(pixel_y, [py_idx], extra_mask=pixel_valid).to(torch.float32)
                fx = hl.load(pixel_x, [px_idx], extra_mask=pixel_valid).to(torch.float32)
                pixel_flat = py_idx * width + px_idx

                final_T_pix = hl.load(final_T_flat, [view_id, pixel_flat], extra_mask=pixel_valid).to(torch.float32)

                suffix_dot = hl.zeros([tile_tid, tile_pix], dtype=torch.float32)
                suffix_trans = hl.full([tile_tid, tile_pix], 1.0, dtype=torch.float32)

                if background_is_zero == 0:
                    for c in range(channels):
                        go_idx = pixel_flat * channels + c
                        go_c = hl.load(grad_out_flat, [view_id, go_idx], extra_mask=pixel_valid)
                        bg_c = background[c]
                        suffix_dot = suffix_dot + go_c * bg_c

                # Iterate gaussians in REVERSE order
                for tile_k in hl.tile(0, max_nnz, block_size=chunk_size_static):
                    for k_inner in hl.static_range(chunk_size_static):
                        k_abs = tile_k.begin + k_inner
                        valid_k = k_abs < nnz
                        reverse_idx = starts + (nnz - 1 - k_abs)
                        gid = hl.load(sorted_vals, [view_id, reverse_idx], extra_mask=valid_k).to(torch.int64)
                        gid = torch.where(valid_k, gid, 0)

                        xy0 = hl.load(xys, [view_id, gid, 0], extra_mask=valid_k).to(torch.float32)
                        xy1 = hl.load(xys, [view_id, gid, 1], extra_mask=valid_k).to(torch.float32)
                        conic0 = hl.load(conic, [view_id, gid, 0], extra_mask=valid_k)
                        conic1 = hl.load(conic, [view_id, gid, 1], extra_mask=valid_k)
                        conic2 = hl.load(conic, [view_id, gid, 2], extra_mask=valid_k)
                        rho_k = hl.load(rho, [view_id, gid], extra_mask=valid_k)
                        opacity_k = hl.load(opacity, [view_id, gid], extra_mask=valid_k)

                        dx = fx - xy0[:, None]
                        dy = fy - xy1[:, None]
                        sigma = (
                            0.5 * (conic0[:, None] * dx * dx + conic2[:, None] * dy * dy)
                            + conic1[:, None] * dx * dy
                        )
                        weight_exp = torch.exp(-sigma)
                        alpha_base = opacity_k[:, None] * weight_exp
                        alpha_unclamped = alpha_base
                        if antialiased != 0:
                            alpha_unclamped = alpha_unclamped * rho_k[:, None]

                        valid_mask = pixel_valid & valid_k[:, None]
                        alpha_unclamped = torch.where(valid_mask, alpha_unclamped, 0.0)
                        clamped = alpha_unclamped > clamp_alpha_max
                        alpha = torch.where(clamped, clamp_alpha_max, alpha_unclamped)
                        accepted = valid_mask & (alpha >= alpha_min)

                        # Recover T_i from final_T and suffix_trans
                        denom = (1.0 - alpha) * suffix_trans
                        safe_denom = torch.where(denom.abs() < 1.0e-12, 1.0e-12, denom)
                        T_i = final_T_pix / safe_denom
                        T_i = torch.where(accepted, T_i, 0.0)
                        weight_i = T_i * alpha

                        # dot(grad_out, value) for this gaussian
                        gid_2d = gid[:, None] + hl.zeros([tile_tid, tile_pix], dtype=torch.int64)
                        view_id_2d = view_id[:, None] + hl.zeros([tile_tid, tile_pix], dtype=torch.int64)
                        dot_grad_value = hl.zeros([tile_tid, tile_pix], dtype=torch.float32)
                        for c in range(channels):
                            go_idx = pixel_flat * channels + c
                            go_c = hl.load(grad_out_flat, [view_id_2d, go_idx], extra_mask=pixel_valid)
                            go_c = torch.where(pixel_valid, go_c, 0.0)
                            val_idx = gid_2d * channels + c
                            view_val_2d = view_id_2d
                            v_c = hl.load(values_flat, [view_val_2d, val_idx], extra_mask=valid_mask)
                            v_c = torch.where(valid_mask, v_c, 0.0)
                            dot_grad_value = dot_grad_value + go_c * v_c

                            gv_contrib = torch.where(accepted, weight_i * go_c, 0.0)
                            reduced_gv = torch.sum(gv_contrib, dim=-1)
                            hl.atomic_add(grad_values_flat, [gid * channels + c], reduced_gv)

                        # dL/dalpha
                        dL_dalpha = torch.where(accepted, T_i * (dot_grad_value - suffix_dot), 0.0)
                        unclamped_accepted = accepted & ~clamped
                        dL_dalpha_base = dL_dalpha

                        if antialiased != 0:
                            grad_rho_contrib = torch.where(unclamped_accepted, dL_dalpha * alpha_base, 0.0)
                            reduced_grad_rho = torch.sum(grad_rho_contrib, dim=-1)
                            hl.atomic_add(grad_rho, [gid], reduced_grad_rho)
                            dL_dalpha_base = torch.where(unclamped_accepted, dL_dalpha_base * rho_k[:, None], dL_dalpha_base)

                        grad_opacity_contrib = torch.where(unclamped_accepted, dL_dalpha_base * weight_exp, 0.0)
                        reduced_grad_opacity = torch.sum(grad_opacity_contrib, dim=-1)
                        hl.atomic_add(grad_opacity, [gid], reduced_grad_opacity)

                        dL_dsigma = torch.where(unclamped_accepted, -dL_dalpha_base * alpha_base, 0.0)

                        dL_dx = -dL_dsigma * (conic0[:, None] * dx + conic1[:, None] * dy)
                        dL_dy = -dL_dsigma * (conic2[:, None] * dy + conic1[:, None] * dx)
                        reduced_dL_dx = torch.sum(dL_dx, dim=-1)
                        reduced_dL_dy = torch.sum(dL_dy, dim=-1)
                        hl.atomic_add(grad_xys_flat, [gid * 2], reduced_dL_dx)
                        hl.atomic_add(grad_xys_flat, [gid * 2 + 1], reduced_dL_dy)

                        dL_dA = dL_dsigma * 0.5 * dx * dx
                        dL_dB = dL_dsigma * dx * dy
                        dL_dC = dL_dsigma * 0.5 * dy * dy
                        reduced_dL_dA = torch.sum(dL_dA, dim=-1)
                        reduced_dL_dB = torch.sum(dL_dB, dim=-1)
                        reduced_dL_dC = torch.sum(dL_dC, dim=-1)
                        hl.atomic_add(grad_conic_flat, [gid * 3], reduced_dL_dA)
                        hl.atomic_add(grad_conic_flat, [gid * 3 + 1], reduced_dL_dB)
                        hl.atomic_add(grad_conic_flat, [gid * 3 + 2], reduced_dL_dC)

                        # Update suffix accumulators
                        suffix_dot = torch.where(
                            accepted,
                            alpha * dot_grad_value + (1.0 - alpha) * suffix_dot,
                            suffix_dot,
                        )
                        suffix_trans = torch.where(accepted, suffix_trans * (1.0 - alpha), suffix_trans)

        return (
            grad_xys_flat.view(gaussian_count, 2),
            grad_conic_flat.view(gaussian_count, 3),
            grad_rho,
            grad_values_flat.view(gaussian_count, channels),
            grad_opacity,
        )

    return raster_backward


# ---------------------------------------------------------------------------
# Forward implementation (shared by custom_op and autograd.Function)
# ---------------------------------------------------------------------------


def _helion_rasterize_from_projection_forward_impl(
    projection,
    values: Tensor,
    opacity: Tensor,
    background: Tensor,
    width: int,
    height: int,
    cfg: RasterConfig,
) -> tuple[Tensor, PreparedVisibility, Tensor, Tensor]:
    """Sort + Helion forward rasterization from a pre-computed projection.

    Use this when projection has already been computed (e.g. via batched
    projection for multiple views).  Returns ``(out, prepared, final_T, stop_idx)``.
    """
    prepared = prepare_visibility_from_projection(projection, width=int(width), height=int(height), cfg=cfg)
    prepared = _stabilize_prepared_visibility(prepared)

    return _helion_rasterize_prepared_forward_impl(
        prepared, values, opacity, background, width, height, cfg,
    )


def _helion_rasterize_prepared_forward_impl(
    prepared: PreparedVisibility,
    values: Tensor,
    opacity: Tensor,
    background: Tensor,
    width: int,
    height: int,
    cfg: RasterConfig,
) -> tuple[Tensor, PreparedVisibility, Tensor, Tensor]:
    """Helion forward rasterization from prepared visibility.

    Returns ``(out, prepared, final_T, stop_idx)``.
    """

    # Downcast projection outputs (conic, rho) and opacity to match values
    # dtype when running in mixed precision.  Conic and rho have bounded
    # magnitude and are consumed via type promotion from FP32 dx/dy in the
    # kernel, so BF16 is safe.  Opacity is [0,1] from sigmoid.
    # xys stays FP32 (sub-pixel precision; BF16 ULP at 1920 ≈ 8).
    values_dtype = values.dtype
    if values_dtype != torch.float32:
        prepared = PreparedVisibility(
            xys=prepared.xys,  # stays FP32: sub-pixel precision required
            conic=prepared.conic.to(values_dtype),
            rho=prepared.rho.to(values_dtype),
            num_tiles_hit=prepared.num_tiles_hit,
            tile_start=prepared.tile_start,
            tile_end=prepared.tile_end,
            sorted_vals=prepared.sorted_vals,
            width=prepared.width,
            height=prepared.height,
            tile_size=prepared.tile_size,
            tiles_x=prepared.tiles_x,
            tiles_y=prepared.tiles_y,
            tile_count=prepared.tile_count,
            gaussian_count_value=prepared.gaussian_count,
            intersection_count_value=prepared.intersection_count,
        )
        # opacity may arrive as FP32 (from field sigmoid output); downcast
        # to match values dtype.  Opacity is bounded [0,1], safe for BF16.
        if opacity.dtype != values_dtype:
            opacity = opacity.to(values_dtype)

    # Route through the batched kernel with V=1.
    out_list, final_T_list = _helion_batched_rasterize_forward_impl(
        [prepared], [values], [opacity], background, width, height, cfg,
    )
    out = out_list[0]
    final_T = final_T_list[0]
    # stop_idx: the batched kernel doesn't produce stop_idx, compute from tile_end
    stop_idx = torch.zeros(int(height), int(width), device=out.device, dtype=torch.int32)
    return out, prepared, final_T, stop_idx


def _helion_rasterize_values_forward_impl(
    means: Tensor,
    quat: Tensor,
    scale: Tensor,
    values: Tensor,
    opacity: Tensor,
    background: Tensor,
    viewmat: Tensor,
    K: Tensor,
    width: int,
    height: int,
    cfg: RasterConfig,
) -> tuple[Tensor, PreparedVisibility, Tensor, Tensor]:
    """Run projection + Helion forward rasterization.

    Returns ``(out, prepared, final_T, stop_idx)``.
    """
    projection = project_gaussians_reference(
        means=means,
        quat=quat,
        scale=scale,
        viewmat=viewmat,
        K=K,
        width=int(width),
        height=int(height),
        cfg=cfg,
    )
    return _helion_rasterize_from_projection_forward_impl(
        projection, values, opacity, background, width, height, cfg,
    )


def _helion_batched_rasterize_forward_impl(
    prepared_list: list[PreparedVisibility],
    values_list: list[Tensor],
    opacity_list: list[Tensor],
    background: Tensor,
    width: int,
    height: int,
    cfg: RasterConfig,
) -> tuple[list[Tensor], list[Tensor]]:
    """Run batched Helion forward rasterization for V views in one kernel.

    Args:
        prepared_list: Per-view PreparedVisibility (from projection + sort).
        values_list: Per-view packed values [N, C] tensors.
        opacity_list: Per-view opacity [N] tensors.
        background: Shared background [C].
        width, height: Render dimensions (same for all views).
        cfg: Raster config.

    Returns:
        (out_list, final_T_list) — per-view output images [H, W, C] and
        transmittance maps [H, W].
    """
    num_views = len(prepared_list)
    assert num_views > 0
    tile_size = int(cfg.tile_size)
    tiles_x = (int(width) + tile_size - 1) // tile_size
    tiles_y = (int(height) + tile_size - 1) // tile_size
    tiles_per_view = tiles_x * tiles_y
    total_tiles = num_views * tiles_per_view
    pixels_per_view = int(height) * int(width)
    channels = int(background.shape[0])

    # Stack per-view data into [V, ...] tensors
    max_intersections = max(p.sorted_vals.shape[0] for p in prepared_list)
    max_gaussians = max(v.shape[0] for v in values_list)

    tile_start = torch.stack([p.tile_start for p in prepared_list])   # [V, T]
    tile_end = torch.stack([p.tile_end for p in prepared_list])       # [V, T]
    sorted_vals = torch.stack([
        _pad_vector(p.sorted_vals, max_intersections) for p in prepared_list
    ])  # [V, M_max]
    xys = torch.stack([
        _pad_rows(p.xys, max_gaussians) for p in prepared_list
    ])  # [V, N, 2]
    conic = torch.stack([
        _pad_rows(p.conic, max_gaussians) for p in prepared_list
    ])  # [V, N, 3]
    rho_stacked = torch.stack([
        _pad_vector(p.rho, max_gaussians) for p in prepared_list
    ])  # [V, N]

    values_flat = torch.stack([
        _pad_rows(v.contiguous(), max_gaussians).view(-1)
        for v in values_list
    ])  # [V, N*C]
    opacity_stacked = torch.stack([
        _pad_vector(o, max_gaussians) for o in opacity_list
    ])  # [V, N]

    # Downcast to match values dtype
    values_dtype = values_list[0].dtype
    if values_dtype != torch.float32:
        conic = conic.to(values_dtype)
        rho_stacked = rho_stacked.to(values_dtype)
        opacity_stacked = opacity_stacked.to(values_dtype)

    kernel = _make_batched_raster_forward_kernel(
        static_shapes=cfg.helion_static_shapes,
        runtime_autotune=cfg.helion_runtime_autotune,
    )
    out_flat, final_T_flat = kernel(
        tile_start,
        tile_end,
        sorted_vals,
        torch.arange(int(width), device=background.device, dtype=torch.float32) + 0.5,
        torch.arange(int(height), device=background.device, dtype=torch.float32) + 0.5,
        xys.contiguous(),
        conic.contiguous(),
        rho_stacked,
        values_flat,
        opacity_stacked,
        background,
        total_tiles,
        tiles_per_view,
        tiles_x,
        tile_size,
        int(cfg.rasterize_mode == "antialiased"),
        int(bool(torch.count_nonzero(background).item() == 0)),
        _helion_raster_chunk_size(),
        float(cfg.alpha_min),
        float(cfg.transmittance_eps),
        float(cfg.clamp_alpha_max),
        pixels_per_view,
    )

    # Split per-view results
    out_per_view = out_flat.view(num_views, int(height), int(width), channels)
    final_T_per_view = final_T_flat.view(num_views, int(height), int(width))

    return (
        [out_per_view[v] for v in range(num_views)],
        [final_T_per_view[v] for v in range(num_views)],
    )


# ---------------------------------------------------------------------------
# Backward implementation (shared by custom_op and autograd.Function)
# ---------------------------------------------------------------------------


def _helion_rasterize_values_backward_impl(
    grad_out: Tensor,
    *,
    means: Tensor,
    quat: Tensor,
    scale: Tensor,
    values: Tensor,
    opacity: Tensor,
    background: Tensor,
    viewmat: Tensor,
    K: Tensor,
    xys: Tensor,
    conic: Tensor,
    rho: Tensor,
    num_tiles_hit: Tensor,
    tile_start: Tensor,
    tile_end: Tensor,
    sorted_vals: Tensor,
    final_T: Tensor,
    stop_idx: Tensor,
    width: int,
    height: int,
    cfg: RasterConfig,
    needs_input_grad: tuple[bool, ...],
) -> tuple[Tensor | None, ...]:
    """Compute backward pass. Returns grads for (means, quat, scale, values,
    opacity, background, viewmat, K)."""

    tile_size = int(cfg.tile_size)
    tiles_x = (int(width) + tile_size - 1) // tile_size
    tiles_y = (int(height) + tile_size - 1) // tile_size

    prepared = PreparedVisibility(
        xys=xys,
        conic=conic,
        rho=rho,
        num_tiles_hit=num_tiles_hit,
        tile_start=tile_start,
        tile_end=tile_end,
        sorted_vals=sorted_vals,
        width=int(width),
        height=int(height),
        tile_size=tile_size,
        tiles_x=tiles_x,
        tiles_y=tiles_y,
        tile_count=tiles_x * tiles_y,
    )

    if cfg.backward_impl == "warp_tape":
        raise NotImplementedError("Helion backend does not support backward_impl='warp_tape'")

    if cfg.backward_impl == "reference":
        with torch.enable_grad():
            means_req = means.detach().requires_grad_(True)
            quat_req = quat.detach().requires_grad_(True)
            scale_req = scale.detach().requires_grad_(True)
            values_req = values.detach().requires_grad_(True)
            opacity_req = opacity.detach().requires_grad_(True)
            background_req = background.detach().requires_grad_(True)
            viewmat_req = viewmat.detach().requires_grad_(True)
            K_req = K.detach().requires_grad_(True)
            ref_out = render_values_reference(
                means=means_req,
                quat=quat_req,
                scale=scale_req,
                values=values_req,
                opacity=opacity_req,
                background=background_req,
                viewmat=viewmat_req,
                K=K_req,
                width=int(width),
                height=int(height),
                cfg=cfg,
            )
            grads = torch.autograd.grad(
                outputs=ref_out,
                inputs=(means_req, quat_req, scale_req, values_req, opacity_req, background_req, viewmat_req, K_req),
                grad_outputs=grad_out,
                allow_unused=True,
            )
        masked = tuple(grad if need else None for grad, need in zip(grads, needs_input_grad[:8], strict=True))
        return (*masked, None, None, None, None)

    if cfg.backward_impl == "helion":
        # --- Helion native backward: rasterization gradients ---
        bwd_kernel = _make_raster_backward_kernel(
            static_shapes=cfg.helion_static_shapes,
            runtime_autotune=cfg.helion_runtime_autotune,
        )
        grad_xys, grad_conic, grad_rho, grad_values, grad_opacity = bwd_kernel(
            prepared.tile_start,
            prepared.tile_end,
            prepared.sorted_vals,
            torch.arange(int(width), device=values.device, dtype=torch.float32) + 0.5,
            torch.arange(int(height), device=values.device, dtype=torch.float32) + 0.5,
            prepared.xys.contiguous(),
            prepared.conic.contiguous(),
            prepared.rho,
            values.contiguous(),
            opacity,
            background,
            final_T.view(-1),
            grad_out.contiguous().view(-1),
            prepared.tiles_x,
            prepared.tile_size,
            int(cfg.rasterize_mode == "antialiased"),
            int(bool(torch.count_nonzero(background).item() == 0)),
            _helion_raster_backward_chunk_size(),
            float(cfg.alpha_min),
            float(cfg.transmittance_eps),
            float(cfg.clamp_alpha_max),
        )

        # --- Projection backward via autograd ---
        with torch.enable_grad():
            means_req = means.detach().requires_grad_(True)
            quat_req = quat.detach().requires_grad_(True)
            scale_req = scale.detach().requires_grad_(True)
            viewmat_req = viewmat.detach().requires_grad_(True)
            K_req = K.detach().requires_grad_(True)
            proj = project_gaussians_reference(
                means=means_req,
                quat=quat_req,
                scale=scale_req,
                viewmat=viewmat_req,
                K=K_req,
                width=int(width),
                height=int(height),
                cfg=cfg,
            )
            grad_means, grad_quat, grad_scale, grad_viewmat, grad_K = torch.autograd.grad(
                outputs=(proj.xys, proj.conic, proj.rho),
                inputs=(means_req, quat_req, scale_req, viewmat_req, K_req),
                grad_outputs=(grad_xys, grad_conic, grad_rho),
                allow_unused=True,
            )

        # Compute grad_background from final_T if needed
        grad_background: Tensor | None = None
        if needs_input_grad[5]:
            # grad_background[c] = sum over pixels of final_T[p] * grad_out[p, c]
            grad_background = (final_T.view(-1, 1) * grad_out.view(-1, background.shape[0])).sum(dim=0)

        return (
            grad_means if needs_input_grad[0] else None,
            grad_quat if needs_input_grad[1] else None,
            grad_scale if needs_input_grad[2] else None,
            grad_values if needs_input_grad[3] else None,
            grad_opacity if needs_input_grad[4] else None,
            grad_background,
            grad_viewmat if needs_input_grad[6] else None,
            grad_K if needs_input_grad[7] else None,
            None,
            None,
            None,
            None,
        )

    raise NotImplementedError(f"Helion backend does not support backward_impl={cfg.backward_impl!r}")


# ---------------------------------------------------------------------------
# Custom op (torch.library.custom_op)
# ---------------------------------------------------------------------------


@torch.library.custom_op("blender_temp::helion_rasterize_values", mutates_args=())
def helion_rasterize_values_op(
    means: Tensor,
    quat: Tensor,
    scale: Tensor,
    values: Tensor,
    opacity: Tensor,
    background: Tensor,
    viewmat: Tensor,
    K: Tensor,
    width: int,
    height: int,
    backward_impl: str,
    rasterize_mode: str,
    tile_size: int,
    near_plane: float,
    far_plane: float,
    eps2d: float,
    radius_clip: float,
    alpha_min: float,
    transmittance_eps: float,
    clamp_alpha_max: float,
    helion_static_shapes: bool,
    helion_runtime_autotune: bool,
) -> Tensor:
    cfg = RasterConfig(
        backend="helion",
        backward_impl=backward_impl,
        rasterize_mode=rasterize_mode,
        tile_size=tile_size,
        near_plane=near_plane,
        far_plane=far_plane,
        eps2d=eps2d,
        radius_clip=radius_clip,
        alpha_min=alpha_min,
        transmittance_eps=transmittance_eps,
        clamp_alpha_max=clamp_alpha_max,
        helion_static_shapes=helion_static_shapes,
        helion_runtime_autotune=helion_runtime_autotune,
    )
    out, _prepared, _final_T, _stop_idx = _helion_rasterize_values_forward_impl(
        means, quat, scale, values, opacity, background, viewmat, K, width, height, cfg,
    )
    return out


@helion_rasterize_values_op.register_fake
def _helion_rasterize_values_fake(
    means: Tensor,
    quat: Tensor,
    scale: Tensor,
    values: Tensor,
    opacity: Tensor,
    background: Tensor,
    viewmat: Tensor,
    K: Tensor,
    width: int,
    height: int,
    backward_impl: str,
    rasterize_mode: str,
    tile_size: int,
    near_plane: float,
    far_plane: float,
    eps2d: float,
    radius_clip: float,
    alpha_min: float,
    transmittance_eps: float,
    clamp_alpha_max: float,
    helion_static_shapes: bool,
    helion_runtime_autotune: bool,
) -> Tensor:
    channels = values.shape[1]
    return values.new_empty((height, width, channels))


def _helion_rasterize_values_setup_context(
    ctx: Any,
    inputs: tuple,
    output: Tensor,
) -> None:
    (
        means, quat, scale, values, opacity, background,
        viewmat, K, width, height,
        backward_impl, rasterize_mode, tile_size,
        near_plane, far_plane, eps2d, radius_clip,
        alpha_min, transmittance_eps, clamp_alpha_max,
        helion_static_shapes, helion_runtime_autotune,
    ) = inputs
    cfg = RasterConfig(
        backend="helion",
        backward_impl=backward_impl,
        rasterize_mode=rasterize_mode,
        tile_size=tile_size,
        near_plane=near_plane,
        far_plane=far_plane,
        eps2d=eps2d,
        radius_clip=radius_clip,
        alpha_min=alpha_min,
        transmittance_eps=transmittance_eps,
        clamp_alpha_max=clamp_alpha_max,
        helion_static_shapes=helion_static_shapes,
        helion_runtime_autotune=helion_runtime_autotune,
    )
    # Re-run forward to capture intermediate state for backward
    _out, prepared, final_T, stop_idx = _helion_rasterize_values_forward_impl(
        means, quat, scale, values, opacity, background, viewmat, K, width, height, cfg,
    )
    ctx.cfg = cfg
    ctx.width = int(width)
    ctx.height = int(height)
    ctx.save_for_backward(
        means, quat, scale, values, opacity, background, viewmat, K,
        prepared.xys, prepared.conic, prepared.rho, prepared.num_tiles_hit,
        prepared.tile_start, prepared.tile_end, prepared.sorted_vals,
        final_T, stop_idx,
    )


def _helion_rasterize_values_backward(
    ctx: Any,
    grad_out: Tensor,
) -> tuple[Tensor | None, ...]:
    (
        means, quat, scale, values, opacity, background, viewmat, K,
        xys, conic, rho, num_tiles_hit,
        tile_start, tile_end, sorted_vals,
        final_T, stop_idx,
    ) = ctx.saved_tensors
    result = _helion_rasterize_values_backward_impl(
        grad_out,
        means=means,
        quat=quat,
        scale=scale,
        values=values,
        opacity=opacity,
        background=background,
        viewmat=viewmat,
        K=K,
        xys=xys,
        conic=conic,
        rho=rho,
        num_tiles_hit=num_tiles_hit,
        tile_start=tile_start,
        tile_end=tile_end,
        sorted_vals=sorted_vals,
        final_T=final_T,
        stop_idx=stop_idx,
        width=ctx.width,
        height=ctx.height,
        cfg=ctx.cfg,
        needs_input_grad=ctx.needs_input_grad,
    )
    # result has 12 elements: 8 input grads + 4 Nones for (width, height, cfg, return_prepared)
    # custom_op expects grads for all 22 params
    grads_8 = result[:8]
    # Pad with None for the remaining 14 non-Tensor params
    return (*grads_8, *((None,) * 14))


helion_rasterize_values_op.register_autograd(
    _helion_rasterize_values_backward,
    setup_context=_helion_rasterize_values_setup_context,
)


# ---------------------------------------------------------------------------
# autograd.Function wrapper (for render_values_helion with return_prepared)
# ---------------------------------------------------------------------------


class _HelionRasterizePreparedFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx: Any,
        means: Tensor,
        quat: Tensor,
        scale: Tensor,
        values: Tensor,
        opacity: Tensor,
        background: Tensor,
        viewmat: Tensor,
        K: Tensor,
        width: int,
        height: int,
        cfg: RasterConfig,
        return_prepared: bool,
    ) -> Tensor | tuple[Tensor, PreparedVisibility]:
        out, prepared, final_T, stop_idx = _helion_rasterize_values_forward_impl(
            means, quat, scale, values, opacity, background, viewmat, K,
            int(width), int(height), cfg,
        )
        ctx.cfg = cfg
        ctx.width = int(width)
        ctx.height = int(height)
        ctx.return_prepared = bool(return_prepared)
        ctx.save_for_backward(
            means, quat, scale, values, opacity, background, viewmat, K,
            prepared.xys, prepared.conic, prepared.rho, prepared.num_tiles_hit,
            prepared.tile_start, prepared.tile_end, prepared.sorted_vals,
            final_T, stop_idx,
        )
        if return_prepared:
            return out, prepared
        return out

    @staticmethod
    def backward(  # pyrefly: ignore [bad-override]
        ctx: Any,
        *grad_outputs: Tensor,
    ) -> tuple[Tensor | None, ...]:
        grad_out = grad_outputs[0]
        (
            means, quat, scale, values, opacity, background, viewmat, K,
            xys, conic, rho, num_tiles_hit,
            tile_start, tile_end, sorted_vals,
            final_T, stop_idx,
        ) = ctx.saved_tensors
        result = _helion_rasterize_values_backward_impl(
            grad_out,
            means=means,
            quat=quat,
            scale=scale,
            values=values,
            opacity=opacity,
            background=background,
            viewmat=viewmat,
            K=K,
            xys=xys,
            conic=conic,
            rho=rho,
            num_tiles_hit=num_tiles_hit,
            tile_start=tile_start,
            tile_end=tile_end,
            sorted_vals=sorted_vals,
            final_T=final_T,
            stop_idx=stop_idx,
            width=ctx.width,
            height=ctx.height,
            cfg=ctx.cfg,
            needs_input_grad=ctx.needs_input_grad,
        )
        return result


# ---------------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------------


def prepare_visibility_helion(
    means: Tensor,
    quat: Tensor,
    scale: Tensor,
    viewmat: Tensor,
    K: Tensor,
    width: int,
    height: int,
    cfg: RasterConfig | None = None,
    requires_grad: bool = False,
    active_count: int | None = None,
) -> PreparedVisibility:
    del requires_grad
    if cfg is None:
        cfg = RasterConfig(backend="helion")
    if active_count is not None:
        count = max(0, min(int(active_count), int(means.shape[0])))
        means = means[:count]
        quat = quat[:count]
        scale = scale[:count]
    projection = project_gaussians_reference(
        means=means,
        quat=quat,
        scale=scale,
        viewmat=viewmat,
        K=K,
        width=int(width),
        height=int(height),
        cfg=cfg,
    )
    prepared = prepare_visibility_from_projection(projection, width=int(width), height=int(height), cfg=cfg)
    return _stabilize_prepared_visibility(prepared)


def render_values_helion(
    means: Tensor,
    quat: Tensor,
    scale: Tensor,
    values: Tensor,
    opacity: Tensor,
    viewmat: Tensor,
    K: Tensor,
    width: int,
    height: int,
    cfg: RasterConfig | None = None,
    background: Tensor | None = None,
    return_prepared: bool = False,
    active_count: int | None = None,
) -> Tensor | tuple[Tensor, PreparedVisibility]:
    if cfg is None:
        cfg = RasterConfig(backend="helion")
    if background is None:
        background = torch.zeros(values.shape[1], device=values.device, dtype=values.dtype)
    if active_count is not None:
        count = max(0, min(int(active_count), int(means.shape[0])))
        means = means[:count]
        quat = quat[:count]
        scale = scale[:count]
        values = values[:count]
        opacity = opacity[:count]
    _validate_helion_render_inputs(means, quat, scale, values, opacity, background, viewmat, K)
    return _HelionRasterizePreparedFunction.apply(
        means,
        quat,
        scale,
        values,
        opacity,
        background,
        viewmat,
        K,
        int(width),
        int(height),
        cfg,
        bool(return_prepared),
    )


def render_stats_prepared_helion(
    prepared: PreparedVisibility,
    opacity: Tensor,
    *,
    cfg: RasterConfig | None = None,
    residual_map: Tensor | None = None,
    screen_error_bins: int = 4,
    include_details: bool = True,
    active_count: int | None = None,
) -> dict[str, Tensor]:
    if cfg is None:
        cfg = RasterConfig(backend="helion")
    if active_count is not None:
        opacity = opacity[: int(active_count)]
    if int(opacity.shape[0]) != prepared.gaussian_count:
        raise ValueError(f"opacity length mismatch: expected {prepared.gaussian_count}, got {opacity.shape[0]}")
    meta = prepared.meta()
    if not include_details:
        return meta
    contrib = torch.zeros(prepared.gaussian_count, device=prepared.device, dtype=torch.float32)
    trans = torch.zeros_like(contrib)
    hits = torch.zeros_like(contrib)
    residual = torch.zeros_like(contrib)
    bins = max(int(screen_error_bins), 1)
    error_map = torch.zeros(prepared.gaussian_count, bins * bins, device=prepared.device, dtype=torch.float32)
    if prepared.intersection_count <= 0:
        return {
            "contrib": contrib,
            "transmittance": trans,
            "hits": hits,
            "residual": residual,
            "error_map": error_map,
            **meta,
        }
    if residual_map is None:
        residual_map = torch.zeros(prepared.height, prepared.width, device=prepared.device, dtype=torch.float32)
    kernel = _make_visibility_stats_kernel(
        static_shapes=cfg.helion_static_shapes,
        runtime_autotune=cfg.helion_runtime_autotune,
    )
    tiles_per_view = prepared.tile_count
    contrib, trans, hits, residual, error_map = kernel(
        prepared.tile_start.unsqueeze(0),
        prepared.tile_end.unsqueeze(0),
        prepared.sorted_vals.unsqueeze(0),
        torch.arange(prepared.width, device=prepared.device, dtype=torch.float32) + 0.5,
        torch.arange(prepared.height, device=prepared.device, dtype=torch.float32) + 0.5,
        prepared.xys.contiguous().unsqueeze(0),
        prepared.conic.contiguous().unsqueeze(0),
        prepared.rho.unsqueeze(0),
        opacity.contiguous().unsqueeze(0),
        residual_map.contiguous().unsqueeze(0),
        tiles_per_view,  # total_tiles (V=1)
        tiles_per_view,
        prepared.tiles_x,
        prepared.tile_size,
        int(cfg.rasterize_mode == "antialiased"),
        _helion_raster_chunk_size(),
        bins,
        bins,
        float(cfg.alpha_min),
        float(cfg.transmittance_eps),
        float(cfg.clamp_alpha_max),
        prepared.gaussian_count,
    )
    return {
        "contrib": contrib,
        "transmittance": trans,
        "hits": hits,
        "residual": residual,
        "error_map": error_map,
        **meta,
    }


def render_stats_helion(
    opacity: Tensor,
    *,
    means: Tensor | None = None,
    quat: Tensor | None = None,
    scale: Tensor | None = None,
    viewmat: Tensor | None = None,
    K: Tensor | None = None,
    width: int | None = None,
    height: int | None = None,
    cfg: RasterConfig | None = None,
    residual_map: Tensor | None = None,
    screen_error_bins: int = 4,
    include_details: bool = True,
    prepared: PreparedVisibility | None = None,
    active_count: int | None = None,
) -> dict[str, Tensor]:
    if cfg is None:
        cfg = RasterConfig(backend="helion")
    if prepared is None:
        if (
            means is None
            or quat is None
            or scale is None
            or viewmat is None
            or K is None
            or width is None
            or height is None
        ):
            raise ValueError("render_stats_helion requires either prepared visibility or full render inputs")
        prepared = prepare_visibility_helion(
            means=means,
            quat=quat,
            scale=scale,
            viewmat=viewmat,
            K=K,
            width=int(width),
            height=int(height),
            cfg=cfg,
            active_count=active_count,
        )
    return render_stats_prepared_helion(
        prepared,
        opacity,
        cfg=cfg,
        residual_map=residual_map,
        screen_error_bins=screen_error_bins,
        include_details=include_details,
        active_count=active_count,
    )


def render_visibility_meta_helion(
    *,
    prepared: PreparedVisibility | None = None,
    means: Tensor | None = None,
    quat: Tensor | None = None,
    scale: Tensor | None = None,
    viewmat: Tensor | None = None,
    K: Tensor | None = None,
    width: int | None = None,
    height: int | None = None,
    cfg: RasterConfig | None = None,
    active_count: int | None = None,
) -> dict[str, Tensor]:
    if cfg is None:
        cfg = RasterConfig(backend="helion")
    if prepared is None:
        if (
            means is None
            or quat is None
            or scale is None
            or viewmat is None
            or K is None
            or width is None
            or height is None
        ):
            raise ValueError("render_visibility_meta_helion requires prepared visibility or full render inputs")
        prepared = prepare_visibility_helion(
            means=means,
            quat=quat,
            scale=scale,
            viewmat=viewmat,
            K=K,
            width=int(width),
            height=int(height),
            cfg=cfg,
            active_count=active_count,
        )
    return prepared.meta()


def render_projection_meta_helion(
    *,
    means: Tensor,
    quat: Tensor,
    scale: Tensor,
    viewmat: Tensor,
    K: Tensor,
    width: int,
    height: int,
    cfg: RasterConfig | None = None,
    active_count: int | None = None,
) -> dict[str, int | bool | str | None]:
    if cfg is None:
        cfg = RasterConfig(backend="helion")
    if active_count is not None:
        count = max(0, min(int(active_count), int(means.shape[0])))
        means = means[:count]
        quat = quat[:count]
        scale = scale[:count]
    projection = project_gaussians_reference(
        means=means,
        quat=quat,
        scale=scale,
        viewmat=viewmat,
        K=K,
        width=int(width),
        height=int(height),
        cfg=cfg,
    )
    return projection_meta_from_projection(projection, width=int(width), height=int(height), cfg=cfg)


def render_gaussians_helion(
    means: Tensor,
    quat: Tensor,
    scale: Tensor,
    color_r: Tensor,
    color_g: Tensor,
    color_b: Tensor,
    opacity: Tensor,
    viewmat: Tensor,
    K: Tensor,
    width: int,
    height: int,
    cfg: RasterConfig | None = None,
) -> Tensor:
    values = torch.stack((color_r, color_g, color_b), dim=-1).contiguous()
    background = None
    if cfg is not None:
        background = torch.tensor(cfg.background_rgb, device=values.device, dtype=values.dtype)
    return render_values_helion(
        means=means,
        quat=quat,
        scale=scale,
        values=values,
        opacity=opacity,
        viewmat=viewmat,
        K=K,
        width=width,
        height=height,
        cfg=cfg,
        background=background,
    )


__all__ = [
    "clear_helion_kernel_cache",
    "helion_rasterize_values_op",
    "prepare_visibility_helion",
    "render_gaussians_helion",
    "render_projection_meta_helion",
    "render_stats_helion",
    "render_stats_prepared_helion",
    "render_values_helion",
    "render_visibility_meta_helion",
    "reserve_helion_intersection_capacity",
]
