#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from blender_temp.cmd.main import _restore_rng_state, collect_image_paths, load_images
from blender_temp.gaussian_sr import PoseFreeGaussianSR
from blender_temp.gaussian_sr.debug_checkpoint import load_debug_checkpoint, restore_module_from_debug_checkpoint
from blender_temp.gaussian_sr.helion_gsplat_renderer import (
    _helion_raster_chunk_size,
    _make_raster_forward_kernel,
    _make_visibility_stats_kernel,
)
from blender_temp.gaussian_sr.pipeline import set_torch_compile_enabled
from scripts.run_quick_benchmark import _load_config


def _render_size(pipeline: PoseFreeGaussianSR, stage_scale: float) -> tuple[int, int]:
    out_h = max(1, int(round(pipeline.train_height * stage_scale)))
    out_w = max(1, int(round(pipeline.train_width * stage_scale)))
    return out_h, out_w


def _prepare_example(
    *,
    checkpoint: Path,
    input_dir: Path,
    device: torch.device,
    view_index: int,
) -> tuple[PoseFreeGaussianSR, dict[str, object]]:
    payload = load_debug_checkpoint(checkpoint)
    cfg = _load_config(payload["config"])
    set_torch_compile_enabled(False)
    images = load_images(collect_image_paths(input_dir.resolve()), device)
    pipeline = PoseFreeGaussianSR.from_images(images, intrinsics=None, config=cfg).to(device)
    restore_module_from_debug_checkpoint(pipeline, checkpoint)
    _restore_rng_state(payload)

    stage_index = int(payload.get("stage_index", len(cfg.train.stage_scales) - 1))
    stage_index = max(0, min(stage_index, len(cfg.train.stage_scales) - 1))
    stage_scale = float(cfg.train.stage_scales[stage_index])
    out_h, out_w = _render_size(pipeline, stage_scale)

    base_intr = pipeline.intrinsics.get()
    render_intr = pipeline._scale_intrinsics(out_h, out_w)
    R_all, t_all = pipeline.camera_model.world_to_camera()
    field, viewmat, K, means3d, quat, scale, opacity, values, background, active_count = (
        pipeline._prepare_render_payload_eager(
            base_intr,
            render_intr,
            R_all[view_index],
            t_all[view_index],
        )
    )
    del field

    renderer_cfg = pipeline._renderer_config()
    renderer_cfg.backend = "helion"
    renderer_cfg.helion_runtime_autotune = True

    from blender_temp.gaussian_sr.reference_renderer import project_gaussians_reference
    from blender_temp.gaussian_sr.renderer_host_prep import prepare_visibility_from_projection

    projection = project_gaussians_reference(
        means=means3d.contiguous(),
        quat=quat.contiguous(),
        scale=scale.contiguous(),
        viewmat=viewmat.contiguous(),
        K=K.contiguous(),
        width=out_w,
        height=out_h,
        cfg=renderer_cfg,
    )
    prepared = prepare_visibility_from_projection(projection, width=out_w, height=out_h, cfg=renderer_cfg)
    residual_map = torch.linspace(
        0.0,
        1.0,
        steps=prepared.height * prepared.width,
        device=device,
        dtype=torch.float32,
    ).view(prepared.height, prepared.width)

    example = {
        "cfg": renderer_cfg,
        "prepared": prepared,
        "values": values.contiguous(),
        "opacity": opacity.contiguous(),
        "background": background.contiguous(),
        "pixel_x": torch.arange(prepared.width, device=device, dtype=torch.float32) + 0.5,
        "pixel_y": torch.arange(prepared.height, device=device, dtype=torch.float32) + 0.5,
        "residual_map": residual_map.contiguous(),
    }
    return pipeline, example


def tune_raster_forward(example: dict[str, object], output_path: Path) -> None:
    cfg = example["cfg"]
    prepared = example["prepared"]
    assert hasattr(prepared, "tile_start")
    kernel = _make_raster_forward_kernel(
        static_shapes=cfg.helion_static_shapes,
        runtime_autotune=True,
    )
    args = (
        prepared.tile_start,
        prepared.tile_end,
        prepared.sorted_vals,
        example["pixel_x"],
        example["pixel_y"],
        prepared.xys.contiguous(),
        prepared.conic.contiguous(),
        prepared.rho,
        example["values"],
        example["opacity"],
        example["background"],
        prepared.tiles_x,
        prepared.tile_size,
        int(cfg.rasterize_mode == "antialiased"),
        int(bool(torch.count_nonzero(example["background"]).item() == 0)),
        _helion_raster_chunk_size(),
        float(cfg.alpha_min),
        float(cfg.transmittance_eps),
        float(cfg.clamp_alpha_max),
    )
    best = kernel.autotune(args, force=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best.save(output_path)
    print(f"saved raster_forward config to {output_path}")


def tune_visibility_stats(example: dict[str, object], output_path: Path) -> None:
    cfg = example["cfg"]
    prepared = example["prepared"]
    kernel = _make_visibility_stats_kernel(
        static_shapes=cfg.helion_static_shapes,
        runtime_autotune=True,
    )
    bins = 4
    args = (
        prepared.tile_start,
        prepared.tile_end,
        prepared.sorted_vals,
        example["pixel_x"],
        example["pixel_y"],
        prepared.xys.contiguous(),
        prepared.conic.contiguous(),
        prepared.rho,
        example["opacity"],
        example["residual_map"],
        prepared.tiles_x,
        prepared.tile_size,
        int(cfg.rasterize_mode == "antialiased"),
        _helion_raster_chunk_size(),
        bins,
        bins,
        float(cfg.alpha_min),
        float(cfg.transmittance_eps),
        float(cfg.clamp_alpha_max),
    )
    best = kernel.autotune(args, force=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best.save(output_path)
    print(f"saved visibility_stats config to {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("configs/helion"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--view-index", type=int, default=0)
    parser.add_argument("--kernel", choices=("raster_forward", "visibility_stats", "all"), default="all")
    args = parser.parse_args()

    device = torch.device(args.device)
    _pipeline, example = _prepare_example(
        checkpoint=args.checkpoint.resolve(),
        input_dir=args.input_dir.resolve(),
        device=device,
        view_index=int(args.view_index),
    )

    cfg = example["cfg"]
    prepared = example["prepared"]
    channels = int(example["values"].shape[1])
    base = args.output_dir.resolve()

    if args.kernel in ("raster_forward", "all"):
        path = base / f"raster_forward_h{prepared.height}_w{prepared.width}_c{channels}.json"
        tune_raster_forward(example, path)
    if args.kernel in ("visibility_stats", "all"):
        path = base / f"visibility_stats_h{prepared.height}_w{prepared.width}_bins4.json"
        tune_visibility_stats(example, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
