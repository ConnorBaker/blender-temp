"""Blender render script — run inside Blender's Python interpreter.

Usage:
    blender --background -noaudio --python-exit-code 1 --python render.py -- [OPTIONS]
"""

import math
import os
import random
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import bpy
from mathutils import Euler, Vector


def setup_argparse() -> ArgumentParser:
    parser = ArgumentParser(description="Render a Blender scene to an image.")
    parser.add_argument(
        "--blend-file",
        type=Path,
        help="Path to the .blend file to render",
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path for the rendered output directory (images are written here)",
        default=Path("render"),
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["CYCLES", "BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"],
        help="Render engine to use",
        default="CYCLES",
    )
    parser.add_argument(
        "--frame",
        type=int,
        help="Frame number to render",
        default=1,
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of render samples",
        default=None,
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        help="Minimum number of render samples before adaptive sampling can stop",
        default=None,
    )
    parser.add_argument(
        "--noise-threshold",
        type=float,
        help="Adaptive sampling noise threshold (0 for automatic)",
        default=None,
    )
    parser.add_argument(
        "--resolution-x",
        type=int,
        help="Horizontal resolution in pixels",
        default=None,
    )
    parser.add_argument(
        "--resolution-y",
        type=int,
        help="Vertical resolution in pixels",
        default=None,
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["PNG", "JPEG", "OPEN_EXR", "TIFF", "BMP"],
        help="Output image format",
        default="PNG",
        dest="image_format",
    )

    # Curves
    parser.add_argument(
        "--curves-shape",
        type=str,
        choices=["RIBBONS", "THICK"],
        help="Curves rendering shape: RIBBONS (Rounded Ribbons) or THICK (3D Curves)",
        default="THICK",
    )

    # Light Paths - Max Bounces
    parser.add_argument("--total-bounces", type=int, help="Total maximum light bounces", default=None)
    parser.add_argument("--diffuse-bounces", type=int, help="Maximum diffuse bounces", default=None)
    parser.add_argument("--glossy-bounces", type=int, help="Maximum glossy bounces", default=None)
    parser.add_argument("--transmission-bounces", type=int, help="Maximum transmission bounces", default=None)
    parser.add_argument("--volume-bounces", type=int, help="Maximum volume bounces", default=None)
    parser.add_argument("--transparent-max-bounces", type=int, help="Maximum transparent bounces", default=None)

    # Light Paths - Clamping
    parser.add_argument("--clamp-direct", type=float, help="Direct light clamping (0 to disable)", default=None)
    parser.add_argument("--clamp-indirect", type=float, help="Indirect light clamping (0 to disable)", default=None)

    # Light Paths - Filter Glossy
    parser.add_argument(
        "--filter-glossy",
        type=float,
        help="Filter glossy threshold to reduce noise from caustics (0 to disable)",
        default=None,
    )

    # Light Paths - Caustics
    parser.add_argument(
        "--no-caustics-reflective",
        action="store_false",
        dest="caustics_reflective",
        help="Disable reflective caustics",
    )
    parser.add_argument(
        "--no-caustics-refractive",
        action="store_false",
        dest="caustics_refractive",
        help="Disable refractive caustics",
    )

    # Sampling - Denoise
    parser.add_argument("--denoise", action="store_true", default=None, help="Enable render denoising")
    parser.add_argument("--no-denoise", action="store_false", dest="denoise")

    # GPU / Device
    parser.add_argument(
        "--device",
        type=str,
        choices=["CPU", "GPU"],
        help="Compute device for Cycles",
        default=None,
    )
    parser.add_argument(
        "--gpu-backend",
        type=str,
        choices=["CUDA", "OPTIX", "HIP", "ONEAPI"],
        help="GPU compute backend",
        default=None,
    )

    # Multi-camera jitter
    parser.add_argument(
        "--num-cameras",
        type=int,
        help="Number of cameras to render from (1 = original camera only)",
        default=1,
    )
    parser.add_argument(
        "--jitter-position",
        type=float,
        help="Maximum position jitter per axis (in Blender units)",
        default=0.5,
    )
    parser.add_argument(
        "--jitter-rotation",
        type=float,
        help="Maximum rotation jitter per axis (in degrees)",
        default=5.0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for deterministic jitter",
        default=0,
    )

    return parser


def configure_scene(args: Namespace) -> None:
    scene = bpy.context.scene

    # Engine
    scene.render.engine = args.engine

    # Sampling
    if args.max_samples is not None:
        scene.cycles.samples = args.max_samples
        scene.eevee.taa_render_samples = args.max_samples

    if args.min_samples is not None:
        scene.cycles.adaptive_min_samples = args.min_samples

    if args.noise_threshold is not None:
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.adaptive_threshold = args.noise_threshold

    # Resolution
    scene.render.resolution_percentage = 100
    if args.resolution_x is not None:
        scene.render.resolution_x = args.resolution_x
    if args.resolution_y is not None:
        scene.render.resolution_y = args.resolution_y

    # Output format
    scene.render.image_settings.file_format = args.image_format
    scene.render.image_settings.color_mode = "RGB"

    # Curves
    scene.cycles_curves.shape = args.curves_shape

    # Light Paths - Max Bounces
    if args.total_bounces is not None:
        scene.cycles.max_bounces = args.total_bounces
    if args.diffuse_bounces is not None:
        scene.cycles.diffuse_bounces = args.diffuse_bounces
    if args.glossy_bounces is not None:
        scene.cycles.glossy_bounces = args.glossy_bounces
    if args.transmission_bounces is not None:
        scene.cycles.transmission_bounces = args.transmission_bounces
    if args.volume_bounces is not None:
        scene.cycles.volume_bounces = args.volume_bounces
    if args.transparent_max_bounces is not None:
        scene.cycles.transparent_max_bounces = args.transparent_max_bounces

    # Light Paths - Clamping
    if args.clamp_direct is not None:
        scene.cycles.clamp_direct = args.clamp_direct
    if args.clamp_indirect is not None:
        scene.cycles.clamp_indirect = args.clamp_indirect

    # Light Paths - Filter Glossy
    if args.filter_glossy is not None:
        scene.cycles.blur_glossy = args.filter_glossy

    # Light Paths - Caustics
    scene.cycles.caustics_reflective = args.caustics_reflective
    scene.cycles.caustics_refractive = args.caustics_refractive

    # Sampling - Denoise
    if args.denoise is not None:
        scene.cycles.use_denoising = args.denoise
        bpy.context.view_layer.cycles.use_denoising = args.denoise

    # GPU / Device
    if args.device is not None:
        scene.cycles.device = args.device

    if args.gpu_backend is not None:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = args.gpu_backend
        prefs.get_devices()
        for dev in prefs.devices:
            dev.use = dev.type == args.gpu_backend


def render_cameras(args: Namespace) -> None:
    scene = bpy.context.scene
    output_base = str(args.output.resolve()) + os.sep
    frame = args.frame
    num_cameras = args.num_cameras
    jitter_pos = args.jitter_position
    jitter_rot_rad = math.radians(args.jitter_rotation)
    rng = random.Random(args.seed)

    base_cam = scene.camera
    base_loc = base_cam.location.copy()
    base_rot = base_cam.rotation_euler.copy()

    scene.frame_set(frame)

    for i in range(num_cameras):
        if i == 0:
            # First camera: use original position unmodified
            base_cam.location = base_loc
            base_cam.rotation_euler = base_rot
        else:
            base_cam.location = Vector((
                base_loc.x + rng.uniform(-jitter_pos, jitter_pos),
                base_loc.y + rng.uniform(-jitter_pos, jitter_pos),
                base_loc.z + rng.uniform(-jitter_pos, jitter_pos),
            ))
            base_cam.rotation_euler = Euler((
                base_rot.x + rng.uniform(-jitter_rot_rad, jitter_rot_rad),
                base_rot.y + rng.uniform(-jitter_rot_rad, jitter_rot_rad),
                base_rot.z + rng.uniform(-jitter_rot_rad, jitter_rot_rad),
            ))

        if num_cameras == 1:
            scene.render.filepath = f"{output_base}{frame:04d}"
        else:
            scene.render.filepath = f"{output_base}cam{i:04d}_frame{frame:04d}"

        print(f"Rendering camera {i + 1}/{num_cameras}: {scene.render.filepath}", flush=True)
        bpy.ops.render.render(write_still=True)

    # Restore original camera transform
    base_cam.location = base_loc
    base_cam.rotation_euler = base_rot


def main() -> None:
    # Blender puts its own args before "--"; ours come after.
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []

    parser = setup_argparse()
    args = parser.parse_args(argv)

    blend_file = args.blend_file.resolve()
    if not blend_file.is_file():
        print(f"Error: blend file not found: {blend_file}", file=sys.stderr)
        sys.exit(1)

    # Ensure output directory exists
    args.output.resolve().mkdir(parents=True, exist_ok=True)

    # Open the blend file
    print(f"Opening blend file: {blend_file}", flush=True)
    bpy.ops.wm.open_mainfile(filepath=str(blend_file))

    # Register render progress handler
    def render_stats_handler(stats: str) -> None:
        print(stats, flush=True)

    bpy.app.handlers.render_stats.append(render_stats_handler)

    # Configure scene settings
    configure_scene(args)

    # Log configuration
    scene = bpy.context.scene
    print(f"Engine: {scene.render.engine}", flush=True)
    print(f"Resolution: {scene.render.resolution_x}x{scene.render.resolution_y}", flush=True)
    print(
        f"Cameras: {args.num_cameras} (seed={args.seed}, jitter_pos={args.jitter_position:.2f}, jitter_rot={args.jitter_rotation:.1f}°)",
        flush=True,
    )

    # Render
    render_cameras(args)

    print("Render complete.", flush=True)


main()
