from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

_STEP_METRIC_KEYS = (
    "step_total_s",
    "view_total_s",
    "field_total_s",
    "render_total_s",
    "aux_stats_total_s",
    "residual_stats_total_s",
    "backward_s",
    "opt_step_s",
    "density_s",
    "cuda_alloc_gib",
    "cuda_reserved_gib",
    "cuda_max_alloc_gib",
)


def select_compare_views(num_views: int) -> tuple[int, ...]:
    if num_views <= 0:
        return ()
    candidates = (0, num_views // 2, num_views - 1)
    ordered: list[int] = []
    seen: set[int] = set()
    for raw_idx in candidates:
        idx = max(0, min(int(raw_idx), num_views - 1))
        if idx not in seen:
            ordered.append(idx)
            seen.add(idx)
    return tuple(ordered)


def summarize_render_output(
    view_index: int,
    rgb: Tensor,
    render_stats: Mapping[str, Tensor] | None,
) -> dict[str, float | int]:
    stats = render_stats or {}

    def _int_stat(name: str) -> int:
        value = stats.get(name)
        if value is None:
            return 0
        return int(value.item())

    return {
        "view_index": int(view_index),
        "visible_count": _int_stat("meta_visible_count"),
        "intersection_count": _int_stat("meta_intersection_count"),
        "gaussian_count": _int_stat("meta_gaussian_count"),
        "mean_gray": float(rgb.detach().mean().item()),
    }


def compare_render_summary(
    current: Mapping[str, float | int],
    baseline: Mapping[str, float | int],
    *,
    visible_fraction_tolerance: float = 0.10,
    intersection_fraction_tolerance: float = 0.10,
    mean_gray_tolerance: float = 0.03,
    l1: float | None = None,
) -> dict[str, object]:
    current_visible = int(current["visible_count"])
    baseline_visible = max(int(baseline["visible_count"]), 1)
    current_intersection = int(current["intersection_count"])
    baseline_intersection = max(int(baseline["intersection_count"]), 1)
    mean_gray_abs_diff = abs(float(current["mean_gray"]) - float(baseline["mean_gray"]))
    visible_fraction_diff = abs(current_visible - baseline_visible) / float(baseline_visible)
    intersection_fraction_diff = abs(current_intersection - baseline_intersection) / float(baseline_intersection)
    passed = (
        visible_fraction_diff <= visible_fraction_tolerance
        and intersection_fraction_diff <= intersection_fraction_tolerance
        and mean_gray_abs_diff <= mean_gray_tolerance
    )
    result: dict[str, object] = {
        "view_index": int(current["view_index"]),
        "passed": bool(passed),
        "visible_fraction_diff": float(visible_fraction_diff),
        "intersection_fraction_diff": float(intersection_fraction_diff),
        "mean_gray_abs_diff": float(mean_gray_abs_diff),
        "thresholds": {
            "visible_fraction_tolerance": float(visible_fraction_tolerance),
            "intersection_fraction_tolerance": float(intersection_fraction_tolerance),
            "mean_gray_tolerance": float(mean_gray_tolerance),
        },
    }
    if l1 is not None:
        result["l1"] = float(l1)
    return result


def aggregate_step_metrics(events: Sequence[Mapping[str, object]]) -> dict[str, float | int]:
    count = len(events)
    if count == 0:
        return {"count": 0}
    metrics: dict[str, float | int] = {"count": int(count)}
    for key in _STEP_METRIC_KEYS:
        values = [float(event[key]) for event in events if key in event]
        if values:
            metrics[key] = float(sum(values) / len(values))
    if "num_gaussians" in events[0]:
        metrics["num_gaussians_start"] = int(events[0]["num_gaussians"])
    if "num_gaussians" in events[-1]:
        metrics["num_gaussians_end"] = int(events[-1]["num_gaussians"])
    if "global_step" in events[0]:
        metrics["global_step_start"] = int(events[0]["global_step"])
    if "global_step" in events[-1]:
        metrics["global_step_end"] = int(events[-1]["global_step"])
    return metrics


def time_cuda_call(
    fn: object,
    *args: object,
    warmup: int = 5,
    repeats: int = 20,
    **kwargs: object,
) -> dict[str, float]:
    """Time a CUDA function using torch.cuda.Event for accurate GPU timing."""
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    timings: list[float] = []
    for _ in range(repeats):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        fn(*args, **kwargs)
        end_event.record()
        torch.cuda.synchronize()
        timings.append(start_event.elapsed_time(end_event))
    timings.sort()
    mean = sum(timings) / len(timings)
    return {
        "mean_ms": mean,
        "median_ms": timings[len(timings) // 2],
        "min_ms": timings[0],
        "max_ms": timings[-1],
        "p95_ms": timings[int(len(timings) * 0.95)],
        "std_ms": (sum((t - mean) ** 2 for t in timings) / len(timings)) ** 0.5,
    }


def measure_peak_memory(fn: object, *args: object, **kwargs: object) -> dict[str, float]:
    """Measure peak GPU memory during a function call."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    fn(*args, **kwargs)
    torch.cuda.synchronize()
    return {
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024 * 1024),
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / (1024 * 1024),
    }


__all__ = [
    "aggregate_step_metrics",
    "compare_render_summary",
    "measure_peak_memory",
    "select_compare_views",
    "summarize_render_output",
    "time_cuda_call",
]
