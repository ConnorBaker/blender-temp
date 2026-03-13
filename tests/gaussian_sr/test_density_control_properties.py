import torch
from hypothesis import given, strategies as st
from torch.testing import assert_close

from blender_temp.gaussian_sr.density_control import (
    apply_density_control,
    build_density_debug_summary,
    compute_prune_keep_mask,
    NormalizedRenderStats,
    normalize_render_stats,
    select_clone_indices,
    should_run_density_control,
    should_run_density_control_for_stage,
)
from blender_temp.gaussian_sr.field import CanonicalGaussianField
from blender_temp.gaussian_sr.posefree_config import AppearanceConfig, DensityControlConfig, FieldConfig

from .strategies import DEFAULT_SETTINGS, chw_images


def _make_test_field(anchor_rgb: torch.Tensor, stride: int = 2, feature_dim: int = 2) -> CanonicalGaussianField:
    intrinsics = torch.tensor(
        [
            float(anchor_rgb.shape[-1]),
            float(anchor_rgb.shape[-2]),
            (anchor_rgb.shape[-1] - 1.0) * 0.5,
            (anchor_rgb.shape[-2] - 1.0) * 0.5,
        ],
        dtype=torch.float32,
    )
    return CanonicalGaussianField(
        anchor_rgb=anchor_rgb,
        intrinsics=intrinsics,
        field_cfg=FieldConfig(anchor_stride=stride, feature_dim=feature_dim),
        appearance_cfg=AppearanceConfig(mode="constant"),
    )


@DEFAULT_SETTINGS
@given(count=st.integers(min_value=1, max_value=16))
def test_normalize_render_stats_none_returns_zero_shapes(count: int) -> None:
    stats = normalize_render_stats(None, count, device=torch.device("cpu"), dtype=torch.float32)

    assert stats.contrib.shape == (count,)
    assert stats.hits.shape == (count,)
    assert stats.avg_trans.shape == (count,)
    assert stats.avg_contrib.shape == (count,)
    assert stats.residual.shape == (count,)
    assert stats.error_map.shape == (count, 1)
    assert stats.peak_error.shape == (count,)
    for value in (
        stats.contrib,
        stats.hits,
        stats.avg_trans,
        stats.avg_contrib,
        stats.residual,
        stats.error_map,
        stats.peak_error,
    ):
        assert torch.isfinite(value).all()
        assert_close(value, torch.zeros_like(value), atol=0.0, rtol=0.0)


def test_normalize_render_stats_computes_expected_averages_and_peaks() -> None:
    render_stats = {
        "contrib": torch.tensor([2.0, 9.0], dtype=torch.float32),
        "hits": torch.tensor([2.0, 3.0], dtype=torch.float32),
        "transmittance": torch.tensor([4.0, 12.0], dtype=torch.float32),
        "residual": torch.tensor([0.5, 1.5], dtype=torch.float32),
        "error_map": torch.tensor([[0.1, 0.4, 0.2], [0.3, 0.2, 0.9]], dtype=torch.float32),
    }

    stats = normalize_render_stats(render_stats, 2, device=torch.device("cpu"), dtype=torch.float32)

    assert_close(stats.avg_trans, torch.tensor([2.0, 4.0], dtype=torch.float32), atol=1.0e-6, rtol=1.0e-6)
    assert_close(stats.avg_contrib, torch.tensor([1.0, 3.0], dtype=torch.float32), atol=1.0e-6, rtol=1.0e-6)
    assert_close(stats.peak_error, torch.tensor([0.4, 0.9], dtype=torch.float32), atol=1.0e-6, rtol=1.0e-6)


def test_compute_prune_keep_mask_enforces_minimum_keep() -> None:
    cfg = DensityControlConfig(min_gaussians=2, opacity_prune_threshold=0.9, prune_visibility_threshold=0.9)
    opacity = torch.tensor([0.05, 0.10, 0.20, 0.15], dtype=torch.float32)
    visibility = torch.tensor([0.00, 0.01, 0.00, 0.20], dtype=torch.float32)

    keep = compute_prune_keep_mask(opacity, visibility, cfg)

    expected = torch.tensor([False, False, True, True])
    assert torch.equal(keep, expected)


@DEFAULT_SETTINGS
@given(
    step=st.integers(min_value=0, max_value=200),
    start_step=st.integers(min_value=0, max_value=50),
    every_steps=st.integers(min_value=1, max_value=20),
)
def test_should_run_density_control_matches_schedule(step: int, start_step: int, every_steps: int) -> None:
    cfg = DensityControlConfig(enabled=True, start_step=start_step, every_steps=every_steps)
    expected = step >= start_step and ((step - start_step) % every_steps == 0)
    assert should_run_density_control(step, cfg) is expected


def test_should_run_density_control_for_stage_disables_last_stage_when_requested() -> None:
    cfg = DensityControlConfig(enabled=True, disable_final_stage=True, start_step=0, every_steps=1)

    assert should_run_density_control_for_stage(0, cfg, stage_index=0, total_stages=3) is True
    assert should_run_density_control_for_stage(1, cfg, stage_index=1, total_stages=3) is True
    assert should_run_density_control_for_stage(2, cfg, stage_index=2, total_stages=3) is False


def test_select_clone_indices_respects_exclude_and_capacity() -> None:
    cfg = DensityControlConfig(
        clone_topk=3,
        max_gaussians=6,
        grad_threshold=0.1,
        densify_opacity_min=0.1,
        densify_visibility_threshold=0.0,
        clone_transmittance_threshold=0.0,
        clone_scale_quantile=0.75,
    )
    opacity = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float32)
    grad = torch.tensor([2.0, 4.0, 3.0, 1.0], dtype=torch.float32)
    scale = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
    stats = NormalizedRenderStats(
        contrib=torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32),
        hits=torch.ones(4, dtype=torch.float32),
        avg_trans=torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32),
        avg_contrib=torch.ones(4, dtype=torch.float32),
        residual=torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        error_map=torch.zeros(4, 1, dtype=torch.float32),
        peak_error=torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
    )
    exclude = torch.tensor([1], dtype=torch.long)

    out = select_clone_indices(opacity, grad, scale, stats, cfg, current_count=4, exclude=exclude)

    assert out.numel() <= 2
    assert out.dtype == torch.long
    assert int(out.numel()) > 0
    assert 1 not in out.tolist()
    assert all(0 <= int(i) < 4 for i in out.tolist())


def test_build_density_debug_summary_reports_peak_bins() -> None:
    cfg = DensityControlConfig(debug_topk=2)
    opacity = torch.tensor([0.2, 0.6], dtype=torch.float32)
    grad = torch.tensor([1.0, 2.0], dtype=torch.float32)
    scale = torch.tensor([0.3, 0.4], dtype=torch.float32)
    stats = NormalizedRenderStats(
        contrib=torch.tensor([2.0, 4.0], dtype=torch.float32),
        hits=torch.tensor([1.0, 1.0], dtype=torch.float32),
        avg_trans=torch.tensor([0.2, 0.3], dtype=torch.float32),
        avg_contrib=torch.tensor([2.0, 4.0], dtype=torch.float32),
        residual=torch.tensor([0.5, 1.0], dtype=torch.float32),
        error_map=torch.tensor([[0.1, 0.9], [0.2, 0.3]], dtype=torch.float32),
        peak_error=torch.tensor([0.9, 0.3], dtype=torch.float32),
    )
    split_idx = torch.tensor([0, 1], dtype=torch.long)
    clone_idx = torch.tensor([1], dtype=torch.long)
    split_score = torch.tensor([0.7, 0.8], dtype=torch.float32)
    clone_score = torch.tensor([0.2, 0.4], dtype=torch.float32)

    summary = build_density_debug_summary(
        opacity, grad, scale, stats, split_idx, clone_idx, split_score, clone_score, cfg
    )

    assert summary.screen_error_bins == 2
    assert len(summary.split_top) == 2
    assert summary.split_top[0].peak_bin in (0, 1)
    assert summary.clone_top[0].index == 1


@DEFAULT_SETTINGS
@given(anchor_rgb=chw_images(min_side=4, max_side=6, min_value=0.1, max_value=0.9))
def test_apply_density_control_can_clone_with_real_field(anchor_rgb: torch.Tensor) -> None:
    field = _make_test_field(anchor_rgb, stride=2, feature_dim=2)
    n = field.num_gaussians
    field.depth_raw.grad = torch.ones_like(field.depth_raw)
    field.xyz_offset.grad = torch.zeros_like(field.xyz_offset)

    cfg = DensityControlConfig(
        enabled=True,
        start_step=0,
        every_steps=1,
        min_gaussians=1,
        max_gaussians=n + 2,
        opacity_prune_threshold=0.0,
        prune_visibility_threshold=0.0,
        densify_opacity_min=0.0,
        densify_visibility_threshold=0.0,
        grad_threshold=0.0,
        split_topk=0,
        clone_topk=1,
        clone_scale_quantile=1.0,
        clone_transmittance_threshold=0.0,
    )
    render_stats = {
        "contrib": torch.ones(n, dtype=torch.float32),
        "hits": torch.ones(n, dtype=torch.float32),
        "transmittance": torch.ones(n, dtype=torch.float32),
        "residual": torch.ones(n, dtype=torch.float32),
        "error_map": torch.zeros(n, 1, dtype=torch.float32),
    }

    event = apply_density_control(field, cfg, step=0, render_stats=render_stats)

    assert event.ran is True
    assert event.changed is True
    assert event.pruned == 0
    assert event.split == 0
    assert event.cloned == 1
    assert event.after == event.before + 1
