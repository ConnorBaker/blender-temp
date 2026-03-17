import torch

from blender_temp.gaussian_sr.benchmarking import (
    aggregate_step_metrics,
    compare_render_summary,
    select_compare_views,
    summarize_render_output,
)


def test_select_compare_views_deduplicates_small_view_counts() -> None:
    assert select_compare_views(1) == (0,)
    assert select_compare_views(2) == (0, 1)
    assert select_compare_views(3) == (0, 1, 2)
    assert select_compare_views(8) == (0, 4, 7)


def test_summarize_render_output_reads_meta_stats_and_gray_mean() -> None:
    rgb = torch.full((3, 4, 5), 0.25, dtype=torch.float32)
    summary = summarize_render_output(
        2,
        rgb,
        {
            "meta_visible_count": torch.tensor(12, dtype=torch.int64),
            "meta_intersection_count": torch.tensor(34, dtype=torch.int64),
            "meta_gaussian_count": torch.tensor(56, dtype=torch.int64),
        },
    )

    assert summary == {
        "view_index": 2,
        "visible_count": 12,
        "intersection_count": 34,
        "gaussian_count": 56,
        "mean_gray": 0.25,
    }


def test_compare_render_summary_enforces_thresholds() -> None:
    baseline = {
        "view_index": 1,
        "visible_count": 100,
        "intersection_count": 200,
        "gaussian_count": 400,
        "mean_gray": 0.50,
    }
    current = {
        "view_index": 1,
        "visible_count": 95,
        "intersection_count": 190,
        "gaussian_count": 410,
        "mean_gray": 0.52,
    }
    failed = {
        "view_index": 1,
        "visible_count": 70,
        "intersection_count": 190,
        "gaussian_count": 410,
        "mean_gray": 0.60,
    }

    assert compare_render_summary(current, baseline, l1=0.01)["passed"] is True
    assert compare_render_summary(failed, baseline)["passed"] is False


def test_aggregate_step_metrics_averages_numeric_step_fields() -> None:
    events = [
        {"global_step": 10, "num_gaussians": 100, "step_total_s": 1.0, "render_total_s": 0.4},
        {"global_step": 11, "num_gaussians": 120, "step_total_s": 3.0, "render_total_s": 0.8},
    ]

    metrics = aggregate_step_metrics(events)

    assert metrics["count"] == 2
    assert metrics["global_step_start"] == 10
    assert metrics["global_step_end"] == 11
    assert metrics["num_gaussians_start"] == 100
    assert metrics["num_gaussians_end"] == 120
    assert metrics["step_total_s"] == 2.0
    assert metrics["render_total_s"] == 0.6000000000000001
