#!/usr/bin/env python3
"""Comparative benchmark: Helion vs Warp Gaussian splatting renderer."""

import argparse
import json
import sys
import time

import torch

from blender_temp.gaussian_sr import RasterConfig, render_values
from blender_temp.gaussian_sr.reference_renderer import render_values_reference


def time_cuda_call(fn, *args, warmup: int = 5, repeats: int = 20, **kwargs) -> dict[str, float]:
    """Time a CUDA function using torch.cuda.Event."""
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    timings = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))

    timings.sort()
    return {
        "mean_ms": sum(timings) / len(timings),
        "median_ms": timings[len(timings) // 2],
        "min_ms": timings[0],
        "max_ms": timings[-1],
        "p95_ms": timings[int(len(timings) * 0.95)],
        "std_ms": (sum((t - sum(timings) / len(timings)) ** 2 for t in timings) / len(timings)) ** 0.5,
    }


def make_synthetic_scene(n: int, width: int, height: int, channels: int, device: torch.device):
    """Create a synthetic scene with n Gaussians."""
    means = torch.stack(
        [
            torch.linspace(-0.5, 0.5, n, device=device),
            torch.linspace(-0.3, 0.3, n, device=device),
            torch.full((n,), 3.0, device=device),
        ],
        dim=-1,
    ).float()
    quat = torch.zeros(n, 4, device=device, dtype=torch.float32)
    quat[:, 0] = 1.0
    scale = torch.full((n, 3), 0.05, device=device, dtype=torch.float32)
    values = torch.rand(n, channels, device=device, dtype=torch.float32)
    opacity = torch.full((n,), 0.5, device=device, dtype=torch.float32)
    background = torch.zeros(channels, device=device, dtype=torch.float32)
    viewmat = torch.eye(4, device=device, dtype=torch.float32)
    K = torch.tensor(
        [
            [float(width), 0.0, width / 2.0],
            [0.0, float(height), height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=torch.float32,
    )
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
    }


def benchmark_forward(scene, cfg):
    def run():
        render_values(
            means=scene["means"],
            quat=scene["quat"],
            scale=scene["scale"],
            values=scene["values"],
            opacity=scene["opacity"],
            background=scene["background"],
            viewmat=scene["viewmat"],
            K=scene["K"],
            width=scene["width"],
            height=scene["height"],
            cfg=cfg,
        )

    return time_cuda_call(run)


def benchmark_backward(scene, cfg):
    def run():
        means = scene["means"].detach().requires_grad_(True)
        values = scene["values"].detach().requires_grad_(True)
        opacity = scene["opacity"].detach().requires_grad_(True)
        out = render_values(
            means=means,
            quat=scene["quat"],
            scale=scene["scale"],
            values=values,
            opacity=opacity,
            background=scene["background"],
            viewmat=scene["viewmat"],
            K=scene["K"],
            width=scene["width"],
            height=scene["height"],
            cfg=cfg,
        )
        loss = out.sum()
        loss.backward()

    return time_cuda_call(run, warmup=3, repeats=10)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Helion vs Warp renderers")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    parser.add_argument("--backends", nargs="+", default=["warp", "helion"], choices=["warp", "helion"])
    parser.add_argument("--gaussians", nargs="+", type=int, default=[1000, 10000, 32000])
    parser.add_argument("--resolutions", nargs="+", default=["256x256", "540x960", "1080x1920"])
    parser.add_argument("--channels", nargs="+", type=int, default=[3, 12])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)

    results = {
        "gpu": gpu_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "benchmarks": [],
    }

    for n in args.gaussians:
        for res_str in args.resolutions:
            height, width = map(int, res_str.split("x"))
            for channels in args.channels:
                scene = make_synthetic_scene(n, width, height, channels, device)
                for backend in args.backends:
                    backward_impl = "hybrid" if backend == "warp" else "reference"
                    cfg = RasterConfig(
                        backend=backend,
                        backward_impl=backward_impl,
                        helion_runtime_autotune=False,
                    )
                    entry = {
                        "backend": backend,
                        "gaussians": n,
                        "resolution": res_str,
                        "channels": channels,
                    }
                    try:
                        print(f"  {backend} N={n} {res_str} C={channels} forward...", end="", flush=True)
                        entry["forward"] = benchmark_forward(scene, cfg)
                        print(f" {entry['forward']['median_ms']:.2f}ms", flush=True)

                        print(f"  {backend} N={n} {res_str} C={channels} backward...", end="", flush=True)
                        entry["backward"] = benchmark_backward(scene, cfg)
                        print(f" {entry['backward']['median_ms']:.2f}ms", flush=True)
                    except Exception as e:
                        entry["error"] = str(e)
                        print(f" ERROR: {e}", flush=True)

                    results["benchmarks"].append(entry)
                    torch.cuda.empty_cache()

    output = json.dumps(results, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"\nResults written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
