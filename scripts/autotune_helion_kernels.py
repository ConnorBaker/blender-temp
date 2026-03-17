#!/usr/bin/env python3
"""Offline autotuning for Helion rasterization kernels.

Generates optimized kernel configs and saves them to configs/helion/ so that
training runs don't need to autotune at startup.

Usage:
    # Tune for default 1080p FP32 (the common case):
    python scripts/autotune_helion_kernels.py

    # Tune for BF16:
    python scripts/autotune_helion_kernels.py --dtype bfloat16

    # Tune for a different resolution:
    python scripts/autotune_helion_kernels.py --height 720 --width 1280

    # Tune only the forward kernel:
    python scripts/autotune_helion_kernels.py --kernels forward
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Ensure the project root is on sys.path so that `blender_temp` is importable
# when this script is invoked directly (e.g. `python scripts/autotune_helion_kernels.py`).
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from torch import Tensor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", stream=sys.stderr)
LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Synthetic scene generation
# ---------------------------------------------------------------------------


def make_synthetic_scene(
    *,
    n_gaussians: int,
    height: int,
    width: int,
    channels: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Tensor | int | float]:
    """Create a synthetic Gaussian splatting scene for autotuning.

    The scene is designed to exercise realistic code paths: Gaussians are
    placed in front of the camera so they actually get rasterised, producing
    a non-trivial intersection list.
    """
    torch.manual_seed(0)

    means = torch.randn(n_gaussians, 3, device=device) * 0.5
    means[:, 2] = means[:, 2].abs() + 1.0  # in front of camera
    quat = torch.randn(n_gaussians, 4, device=device)
    quat = quat / quat.norm(dim=-1, keepdim=True)
    scale = torch.ones(n_gaussians, 3, device=device) * 0.05

    # values/opacity/background use the target dtype (FP32 or BF16).
    values = torch.rand(n_gaussians, channels, device=device, dtype=dtype)
    opacity = torch.rand(n_gaussians, device=device, dtype=dtype) * 0.8 + 0.1
    background = torch.zeros(channels, device=device, dtype=dtype)
    background[:3] = 0.5

    viewmat = torch.eye(4, device=device, dtype=torch.float32)
    K = torch.zeros(3, 3, device=device, dtype=torch.float32)
    K[0, 0] = max(width, height)  # fx
    K[1, 1] = max(width, height)  # fy
    K[0, 2] = width / 2.0
    K[1, 2] = height / 2.0
    K[2, 2] = 1.0

    return {
        "means": means,
        "quat": quat,
        "scale": scale,
        "values": values,
        "opacity": opacity,
        "background": background,
        "viewmat": viewmat,
        "K": K,
        "width": width,
        "height": height,
        "channels": channels,
    }


# ---------------------------------------------------------------------------
# Autotuning drivers
# ---------------------------------------------------------------------------


def autotune_forward_kernel(
    scene: dict,
    *,
    config_dir: Path,
    cfg,
    screen_error_bins: int,
) -> Path | None:
    """Autotune the raster forward kernel and save the config."""
    from blender_temp.gaussian_sr.helion_gsplat_renderer import (
        _helion_raster_chunk_size,
        _make_raster_forward_kernel,
        _stabilize_prepared_visibility,
    )
    from blender_temp.gaussian_sr.reference_renderer import project_gaussians_reference
    from blender_temp.gaussian_sr.renderer_host_prep import prepare_visibility_from_projection
    from blender_temp.gaussian_sr.warp_gsplat_autograd import PreparedVisibility

    width, height = scene["width"], scene["height"]
    channels = scene["channels"]
    values = scene["values"]
    values_dtype = values.dtype

    # Run projection to get realistic prepared visibility data.
    projection = project_gaussians_reference(
        means=scene["means"], quat=scene["quat"], scale=scene["scale"],
        viewmat=scene["viewmat"], K=scene["K"],
        width=width, height=height, cfg=cfg,
    )
    prepared = prepare_visibility_from_projection(projection, width=width, height=height, cfg=cfg)
    prepared = _stabilize_prepared_visibility(prepared)

    # Downcast conic/rho to match values dtype (mirrors forward_impl).
    if values_dtype != torch.float32:
        prepared = PreparedVisibility(
            xys=prepared.xys,
            conic=prepared.conic.to(values_dtype),
            rho=prepared.rho.to(values_dtype),
            num_tiles_hit=prepared.num_tiles_hit,
            tile_start=prepared.tile_start,
            tile_end=prepared.tile_end,
            sorted_vals=prepared.sorted_vals,
            width=prepared.width, height=prepared.height,
            tile_size=prepared.tile_size, tiles_x=prepared.tiles_x,
            tiles_y=prepared.tiles_y, tile_count=prepared.tile_count,
            gaussian_count_value=prepared.gaussian_count,
            intersection_count_value=prepared.intersection_count,
        )

    opacity = scene["opacity"]
    if values_dtype != torch.float32 and opacity.dtype != values_dtype:
        opacity = opacity.to(values_dtype)

    device = values.device
    pixel_x = torch.arange(width, device=device, dtype=torch.float32) + 0.5
    pixel_y = torch.arange(height, device=device, dtype=torch.float32) + 0.5

    bg_is_zero = int(bool(torch.count_nonzero(scene["background"]).item() == 0))
    args = (
        prepared.tile_start, prepared.tile_end, prepared.sorted_vals,
        pixel_x, pixel_y,
        prepared.xys.contiguous(), prepared.conic.contiguous(), prepared.rho,
        values.contiguous(), opacity,
        scene["background"],
        prepared.tiles_x, prepared.tile_size,
        int(cfg.rasterize_mode == "antialiased"),
        bg_is_zero,
        _helion_raster_chunk_size(),
        float(cfg.alpha_min), float(cfg.transmittance_eps), float(cfg.clamp_alpha_max),
    )

    LOGGER.info("Autotuning raster_forward (h=%d w=%d c=%d dtype=%s)...", height, width, channels, values_dtype)
    intersection_count = prepared.intersection_count
    LOGGER.info("  Scene: %d Gaussians, %d intersections", prepared.gaussian_count, intersection_count)

    kernel = _make_raster_forward_kernel(static_shapes=True, runtime_autotune=True)
    t0 = time.perf_counter()
    best_config = kernel.autotune(args)
    elapsed = time.perf_counter() - t0
    LOGGER.info("  Best config found in %.1fs", elapsed)

    dtype_tag = "bf16" if values_dtype == torch.bfloat16 else "fp32"
    filename = f"raster_forward_h{height}_w{width}_c{channels}_{dtype_tag}.json"
    out_path = config_dir / filename
    best_config.save(str(out_path))
    LOGGER.info("  Saved: %s", out_path)
    return out_path


def autotune_visibility_stats_kernel(
    scene: dict,
    *,
    config_dir: Path,
    cfg,
    screen_error_bins: int,
) -> Path | None:
    """Autotune the visibility stats kernel and save the config."""
    from blender_temp.gaussian_sr.helion_gsplat_renderer import (
        _helion_raster_chunk_size,
        _make_visibility_stats_kernel,
        _stabilize_prepared_visibility,
    )
    from blender_temp.gaussian_sr.reference_renderer import project_gaussians_reference
    from blender_temp.gaussian_sr.renderer_host_prep import prepare_visibility_from_projection
    from blender_temp.gaussian_sr.warp_gsplat_autograd import PreparedVisibility

    width, height = scene["width"], scene["height"]
    values_dtype = scene["values"].dtype

    projection = project_gaussians_reference(
        means=scene["means"], quat=scene["quat"], scale=scene["scale"],
        viewmat=scene["viewmat"], K=scene["K"],
        width=width, height=height, cfg=cfg,
    )
    prepared = prepare_visibility_from_projection(projection, width=width, height=height, cfg=cfg)
    prepared = _stabilize_prepared_visibility(prepared)

    if values_dtype != torch.float32:
        prepared = PreparedVisibility(
            xys=prepared.xys,
            conic=prepared.conic.to(values_dtype),
            rho=prepared.rho.to(values_dtype),
            num_tiles_hit=prepared.num_tiles_hit,
            tile_start=prepared.tile_start,
            tile_end=prepared.tile_end,
            sorted_vals=prepared.sorted_vals,
            width=prepared.width, height=prepared.height,
            tile_size=prepared.tile_size, tiles_x=prepared.tiles_x,
            tiles_y=prepared.tiles_y, tile_count=prepared.tile_count,
            gaussian_count_value=prepared.gaussian_count,
            intersection_count_value=prepared.intersection_count,
        )

    opacity = scene["opacity"]
    if values_dtype != torch.float32 and opacity.dtype != values_dtype:
        opacity = opacity.to(values_dtype)

    device = scene["values"].device
    pixel_x = torch.arange(width, device=device, dtype=torch.float32) + 0.5
    pixel_y = torch.arange(height, device=device, dtype=torch.float32) + 0.5
    residual_map = torch.rand(height, width, device=device, dtype=torch.float32)
    bins = max(int(screen_error_bins), 1)

    args = (
        prepared.tile_start, prepared.tile_end, prepared.sorted_vals,
        pixel_x, pixel_y,
        prepared.xys.contiguous(), prepared.conic.contiguous(), prepared.rho,
        opacity.contiguous(), residual_map.contiguous(),
        prepared.tiles_x, prepared.tile_size,
        int(cfg.rasterize_mode == "antialiased"),
        _helion_raster_chunk_size(),
        bins, bins,
        float(cfg.alpha_min), float(cfg.transmittance_eps), float(cfg.clamp_alpha_max),
    )

    LOGGER.info("Autotuning visibility_stats (h=%d w=%d bins=%d dtype=%s)...", height, width, bins, values_dtype)

    kernel = _make_visibility_stats_kernel(static_shapes=True, runtime_autotune=True)
    t0 = time.perf_counter()
    best_config = kernel.autotune(args)
    elapsed = time.perf_counter() - t0
    LOGGER.info("  Best config found in %.1fs", elapsed)

    dtype_tag = "bf16" if values_dtype == torch.bfloat16 else "fp32"
    filename = f"visibility_stats_h{height}_w{width}_bins{bins}_{dtype_tag}.json"
    out_path = config_dir / filename
    best_config.save(str(out_path))
    LOGGER.info("  Saved: %s", out_path)
    return out_path


def autotune_backward_kernel(
    scene: dict,
    *,
    config_dir: Path,
    cfg,
    screen_error_bins: int,
) -> Path | None:
    """Autotune the raster backward kernel and save the config."""
    from blender_temp.gaussian_sr.helion_gsplat_renderer import (
        _helion_raster_backward_chunk_size,
        _helion_raster_chunk_size,
        _make_raster_backward_kernel,
        _make_raster_forward_kernel,
        _stabilize_prepared_visibility,
    )
    from blender_temp.gaussian_sr.reference_renderer import project_gaussians_reference
    from blender_temp.gaussian_sr.renderer_host_prep import prepare_visibility_from_projection
    from blender_temp.gaussian_sr.warp_gsplat_autograd import PreparedVisibility

    width, height = scene["width"], scene["height"]
    channels = scene["channels"]
    values = scene["values"]
    values_dtype = values.dtype

    projection = project_gaussians_reference(
        means=scene["means"], quat=scene["quat"], scale=scene["scale"],
        viewmat=scene["viewmat"], K=scene["K"],
        width=width, height=height, cfg=cfg,
    )
    prepared = prepare_visibility_from_projection(projection, width=width, height=height, cfg=cfg)
    prepared = _stabilize_prepared_visibility(prepared)

    if values_dtype != torch.float32:
        prepared = PreparedVisibility(
            xys=prepared.xys,
            conic=prepared.conic.to(values_dtype),
            rho=prepared.rho.to(values_dtype),
            num_tiles_hit=prepared.num_tiles_hit,
            tile_start=prepared.tile_start,
            tile_end=prepared.tile_end,
            sorted_vals=prepared.sorted_vals,
            width=prepared.width, height=prepared.height,
            tile_size=prepared.tile_size, tiles_x=prepared.tiles_x,
            tiles_y=prepared.tiles_y, tile_count=prepared.tile_count,
            gaussian_count_value=prepared.gaussian_count,
            intersection_count_value=prepared.intersection_count,
        )

    opacity = scene["opacity"]
    if values_dtype != torch.float32 and opacity.dtype != values_dtype:
        opacity = opacity.to(values_dtype)

    device = values.device
    pixel_x = torch.arange(width, device=device, dtype=torch.float32) + 0.5
    pixel_y = torch.arange(height, device=device, dtype=torch.float32) + 0.5

    # Run forward to get final_T for the backward.
    fwd_kernel = _make_raster_forward_kernel(static_shapes=True, runtime_autotune=False)
    bg_is_zero = int(bool(torch.count_nonzero(scene["background"]).item() == 0))
    _out, final_T, _stop_idx = fwd_kernel(
        prepared.tile_start, prepared.tile_end, prepared.sorted_vals,
        pixel_x, pixel_y,
        prepared.xys.contiguous(), prepared.conic.contiguous(), prepared.rho,
        values.contiguous(), opacity,
        scene["background"],
        prepared.tiles_x, prepared.tile_size,
        int(cfg.rasterize_mode == "antialiased"),
        bg_is_zero,
        _helion_raster_chunk_size(),
        float(cfg.alpha_min), float(cfg.transmittance_eps), float(cfg.clamp_alpha_max),
    )

    # Synthetic grad_out in the same dtype as values.
    grad_out = torch.rand(height, width, channels, device=device, dtype=values_dtype)

    args = (
        prepared.tile_start, prepared.tile_end, prepared.sorted_vals,
        pixel_x, pixel_y,
        prepared.xys.contiguous(), prepared.conic.contiguous(), prepared.rho,
        values.contiguous(), opacity,
        scene["background"],
        final_T.view(-1),
        grad_out.contiguous().view(-1),
        prepared.tiles_x, prepared.tile_size,
        int(cfg.rasterize_mode == "antialiased"),
        bg_is_zero,
        _helion_raster_backward_chunk_size(),
        float(cfg.alpha_min), float(cfg.transmittance_eps), float(cfg.clamp_alpha_max),
    )

    LOGGER.info("Autotuning raster_backward (h=%d w=%d c=%d dtype=%s)...", height, width, channels, values_dtype)

    kernel = _make_raster_backward_kernel(static_shapes=True, runtime_autotune=True)
    t0 = time.perf_counter()
    best_config = kernel.autotune(args)
    elapsed = time.perf_counter() - t0
    LOGGER.info("  Best config found in %.1fs", elapsed)

    dtype_tag = "bf16" if values_dtype == torch.bfloat16 else "fp32"
    filename = f"raster_backward_h{height}_w{width}_c{channels}_{dtype_tag}.json"
    out_path = config_dir / filename
    best_config.save(str(out_path))
    LOGGER.info("  Saved: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline autotuning for Helion rasterization kernels.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--height", type=int, default=1080, help="Render height (default: 1080)")
    parser.add_argument("--width", type=int, default=1920, help="Render width (default: 1920)")
    parser.add_argument("--channels", type=int, default=12, help="Value channels (default: 12 = 3 RGB + 8 latent + 1 alpha)")
    parser.add_argument("--n-gaussians", type=int, default=65536, help="Number of Gaussians (default: 65536)")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32", help="Values dtype")
    parser.add_argument("--screen-error-bins", type=int, default=4, help="Screen error bins for visibility stats")
    parser.add_argument(
        "--kernels",
        nargs="+",
        choices=("forward", "visibility", "backward"),
        default=["forward", "visibility", "backward"],
        help="Which kernels to tune (default: all)",
    )
    parser.add_argument("--config-dir", type=Path, default=None, help="Output directory for configs")
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    # Resolve config output directory.
    if args.config_dir is not None:
        config_dir = args.config_dir.resolve()
    else:
        config_dir = Path(__file__).resolve().parents[1] / "configs" / "helion"
    config_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Config output directory: %s", config_dir)

    # Import after torch init to avoid slow startup on import-time Warp init.
    from blender_temp.gaussian_sr.helion_gsplat_renderer import clear_helion_kernel_cache
    from blender_temp.gaussian_sr.warp_gsplat_contracts import RasterConfig

    # Clear any cached kernels so factories create fresh ones with the right
    # autotune settings (the factories use @lru_cache internally).
    clear_helion_kernel_cache()

    cfg = RasterConfig(
        backend="helion",
        rasterize_mode="antialiased",
        helion_static_shapes=True,
        helion_runtime_autotune=True,
    )

    scene = make_synthetic_scene(
        n_gaussians=args.n_gaussians,
        height=args.height,
        width=args.width,
        channels=args.channels,
        dtype=dtype,
        device=device,
    )

    tuned = []
    if "forward" in args.kernels:
        p = autotune_forward_kernel(scene, config_dir=config_dir, cfg=cfg, screen_error_bins=args.screen_error_bins)
        if p:
            tuned.append(p)

    if "visibility" in args.kernels:
        p = autotune_visibility_stats_kernel(scene, config_dir=config_dir, cfg=cfg, screen_error_bins=args.screen_error_bins)
        if p:
            tuned.append(p)

    if "backward" in args.kernels:
        p = autotune_backward_kernel(scene, config_dir=config_dir, cfg=cfg, screen_error_bins=args.screen_error_bins)
        if p:
            tuned.append(p)

    LOGGER.info("Autotuning complete. %d configs saved:", len(tuned))
    for p in tuned:
        LOGGER.info("  %s", p)


if __name__ == "__main__":
    main()
