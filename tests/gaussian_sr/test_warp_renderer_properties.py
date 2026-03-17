import pytest
import torch
from hypothesis import given, strategies as st
from torch.testing import assert_close

from blender_temp.gaussian_sr import RasterConfig, render_stats_prepared_warp, render_stats_warp, render_values_warp
from blender_temp.gaussian_sr.warp_gsplat_renderer import render_projection_meta_warp, render_visibility_meta_warp
from blender_temp.gaussian_sr.warp_runtime import _WARP_AVAILABLE

CUDA_WARP_AVAILABLE = bool(_WARP_AVAILABLE and torch.cuda.is_available())


def _make_visible_scene(n: int, width: int, height: int, device: torch.device) -> dict[str, torch.Tensor]:
    dtype = torch.float32
    means = torch.stack(
        (
            torch.linspace(-0.2, 0.2, n, device=device, dtype=dtype),
            torch.linspace(0.15, -0.15, n, device=device, dtype=dtype),
            torch.full((n,), 2.0, device=device, dtype=dtype),
        ),
        dim=-1,
    )
    quat = torch.stack(
        (
            torch.ones(n, device=device, dtype=dtype),
            torch.zeros(n, device=device, dtype=dtype),
            torch.zeros(n, device=device, dtype=dtype),
            torch.zeros(n, device=device, dtype=dtype),
        ),
        dim=-1,
    )
    scale = torch.full((n, 3), 0.05, device=device, dtype=dtype)
    viewmat = torch.eye(4, device=device, dtype=dtype)
    k = torch.tensor(
        [
            [float(width), 0.0, (width - 1.0) * 0.5],
            [0.0, float(height), (height - 1.0) * 0.5],
            [0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    )
    return {
        "means": means.contiguous(),
        "quat": quat.contiguous(),
        "scale": scale.contiguous(),
        "viewmat": viewmat.contiguous(),
        "K": k.contiguous(),
    }


renderer_settings = st.integers(min_value=1, max_value=4)


@pytest.mark.skipif(not CUDA_WARP_AVAILABLE, reason="requires CUDA + NVIDIA Warp")
@given(
    n=renderer_settings,
    width=st.integers(min_value=4, max_value=8),
    height=st.integers(min_value=4, max_value=8),
    channels=st.integers(min_value=1, max_value=5),
)
def test_zero_opacity_renders_background_exactly(
    n: int,
    width: int,
    height: int,
    channels: int,
) -> None:
    device = torch.device("cuda")
    scene = _make_visible_scene(n, width, height, device)
    values = torch.rand(n, channels, device=device, dtype=torch.float32)
    opacity = torch.zeros(n, device=device, dtype=torch.float32)
    background = torch.linspace(0.1, 0.9, channels, device=device, dtype=torch.float32)

    out = render_values_warp(
        means=scene["means"],
        quat=scene["quat"],
        scale=scene["scale"],
        values=values.contiguous(),
        opacity=opacity.contiguous(),
        background=background.contiguous(),
        viewmat=scene["viewmat"],
        K=scene["K"],
        width=width,
        height=height,
        cfg=RasterConfig(),
    )

    expected = background.view(1, 1, channels).expand(height, width, channels)
    assert_close(out, expected, atol=1.0e-6, rtol=1.0e-6)


@pytest.mark.skipif(not CUDA_WARP_AVAILABLE, reason="requires CUDA + NVIDIA Warp")
@given(
    n=renderer_settings,
    width=st.integers(min_value=4, max_value=8),
    height=st.integers(min_value=4, max_value=8),
)
def test_renderer_stats_residual_identities_hold_for_constant_residual_map(
    n: int,
    width: int,
    height: int,
) -> None:
    device = torch.device("cuda")
    scene = _make_visible_scene(n, width, height, device)
    opacity = torch.full((n,), 0.2, device=device, dtype=torch.float32)
    residual_map = torch.ones(height, width, device=device, dtype=torch.float32)

    stats = render_stats_warp(
        means=scene["means"],
        quat=scene["quat"],
        scale=scene["scale"],
        opacity=opacity.contiguous(),
        viewmat=scene["viewmat"],
        K=scene["K"],
        width=width,
        height=height,
        residual_map=residual_map.contiguous(),
        cfg=RasterConfig(),
    )

    assert torch.isfinite(stats["contrib"]).all()
    assert torch.isfinite(stats["residual"]).all()
    assert torch.isfinite(stats["error_map"]).all()
    assert_close(stats["residual"], stats["contrib"], atol=1.0e-5, rtol=1.0e-5)
    assert_close(stats["error_map"].sum(dim=1), stats["residual"], atol=1.0e-5, rtol=1.0e-5)
    visible = int(stats["meta_visible_count"].item())
    intersections = int(stats["meta_intersection_count"].item())
    assert 0 <= visible <= n
    assert intersections >= visible


@pytest.mark.skipif(not CUDA_WARP_AVAILABLE, reason="requires CUDA + NVIDIA Warp")
@given(
    n=renderer_settings,
    width=st.integers(min_value=4, max_value=8),
    height=st.integers(min_value=4, max_value=8),
)
def test_zero_residual_map_produces_zero_residual_stats(
    n: int,
    width: int,
    height: int,
) -> None:
    device = torch.device("cuda")
    scene = _make_visible_scene(n, width, height, device)
    opacity = torch.full((n,), 0.2, device=device, dtype=torch.float32)
    residual_map = torch.zeros(height, width, device=device, dtype=torch.float32)

    stats = render_stats_warp(
        means=scene["means"],
        quat=scene["quat"],
        scale=scene["scale"],
        opacity=opacity.contiguous(),
        viewmat=scene["viewmat"],
        K=scene["K"],
        width=width,
        height=height,
        residual_map=residual_map.contiguous(),
        cfg=RasterConfig(),
    )

    assert_close(stats["residual"], torch.zeros_like(stats["residual"]), atol=1.0e-6, rtol=0.0)
    assert_close(stats["error_map"], torch.zeros_like(stats["error_map"]), atol=1.0e-6, rtol=0.0)


@pytest.mark.skipif(not CUDA_WARP_AVAILABLE, reason="requires CUDA + NVIDIA Warp")
def test_renderer_returns_finite_gradients_for_viewmat_and_intrinsics() -> None:
    device = torch.device("cuda")
    width = 8
    height = 6
    scene = _make_visible_scene(n=3, width=width, height=height, device=device)
    values = torch.rand(3, 3, device=device, dtype=torch.float32, requires_grad=True)
    opacity = torch.full((3,), 0.2, device=device, dtype=torch.float32, requires_grad=True)
    viewmat = scene["viewmat"].clone().detach().requires_grad_(True)
    K = scene["K"].clone().detach().requires_grad_(True)

    out = render_values_warp(
        means=scene["means"],
        quat=scene["quat"],
        scale=scene["scale"],
        values=values,
        opacity=opacity,
        background=torch.zeros(3, device=device, dtype=torch.float32),
        viewmat=viewmat,
        K=K,
        width=width,
        height=height,
        cfg=RasterConfig(),
    )

    loss = out.sum()
    loss.backward()

    assert values.grad is not None
    assert opacity.grad is not None
    assert viewmat.grad is not None
    assert K.grad is not None
    assert torch.isfinite(values.grad).all()
    assert torch.isfinite(opacity.grad).all()
    assert torch.isfinite(viewmat.grad).all()
    assert torch.isfinite(K.grad).all()


@pytest.mark.skipif(not CUDA_WARP_AVAILABLE, reason="requires CUDA + NVIDIA Warp")
def test_prepared_visibility_matches_direct_aux_stats() -> None:
    device = torch.device("cuda")
    width = 8
    height = 6
    channels = 4
    scene = _make_visible_scene(n=3, width=width, height=height, device=device)
    values = torch.rand(3, channels, device=device, dtype=torch.float32)
    opacity = torch.full((3,), 0.2, device=device, dtype=torch.float32)
    background = torch.linspace(0.1, 0.9, channels, device=device, dtype=torch.float32)
    residual_map = torch.rand(height, width, device=device, dtype=torch.float32)

    out_direct = render_values_warp(
        means=scene["means"],
        quat=scene["quat"],
        scale=scene["scale"],
        values=values,
        opacity=opacity,
        background=background,
        viewmat=scene["viewmat"],
        K=scene["K"],
        width=width,
        height=height,
        cfg=RasterConfig(),
    )
    out_prepared, prepared = render_values_warp(
        means=scene["means"],
        quat=scene["quat"],
        scale=scene["scale"],
        values=values,
        opacity=opacity,
        background=background,
        viewmat=scene["viewmat"],
        K=scene["K"],
        width=width,
        height=height,
        cfg=RasterConfig(),
        return_prepared=True,
    )
    direct_stats = render_stats_warp(
        means=scene["means"],
        quat=scene["quat"],
        scale=scene["scale"],
        opacity=opacity,
        viewmat=scene["viewmat"],
        K=scene["K"],
        width=width,
        height=height,
        residual_map=residual_map,
        cfg=RasterConfig(),
    )
    prepared_stats = render_stats_prepared_warp(
        prepared,
        opacity,
        residual_map=residual_map,
        cfg=RasterConfig(),
    )

    assert_close(out_prepared, out_direct, atol=1.0e-6, rtol=1.0e-6)
    for key in ("contrib", "transmittance", "hits", "residual", "error_map"):
        assert_close(prepared_stats[key], direct_stats[key], atol=1.0e-6, rtol=1.0e-6)
    for key in (
        "meta_gaussian_count",
        "meta_visible_count",
        "meta_intersection_count",
        "meta_tile_count",
        "meta_tiles_x",
        "meta_tiles_y",
        "meta_render_width",
        "meta_render_height",
    ):
        assert_close(prepared_stats[key], direct_stats[key], atol=0.0, rtol=0.0)


@pytest.mark.skipif(not CUDA_WARP_AVAILABLE, reason="requires CUDA + NVIDIA Warp")
def test_prepared_stats_launch_cache_accepts_grad_tracking_inputs() -> None:
    device = torch.device("cuda")
    width = 8
    height = 6
    scene = _make_visible_scene(n=3, width=width, height=height, device=device)
    values = torch.rand(3, 4, device=device, dtype=torch.float32, requires_grad=True)
    opacity = torch.full((3,), 0.2, device=device, dtype=torch.float32, requires_grad=True)

    rendered, prepared = render_values_warp(
        means=scene["means"],
        quat=scene["quat"],
        scale=scene["scale"],
        values=values,
        opacity=opacity,
        background=torch.zeros(4, device=device, dtype=torch.float32),
        viewmat=scene["viewmat"],
        K=scene["K"],
        width=width,
        height=height,
        cfg=RasterConfig(),
        return_prepared=True,
    )
    residual_map = rendered[..., 0].contiguous()

    stats = render_stats_prepared_warp(
        prepared,
        opacity,
        residual_map=residual_map,
        cfg=RasterConfig(),
    )

    assert torch.isfinite(stats["contrib"]).all()
    assert torch.isfinite(stats["residual"]).all()
    assert torch.isfinite(stats["error_map"]).all()


@pytest.mark.skipif(not CUDA_WARP_AVAILABLE, reason="requires CUDA + NVIDIA Warp")
def test_projection_meta_matches_visibility_meta_counts_and_budget() -> None:
    device = torch.device("cuda")
    width = 8
    height = 6
    scene = _make_visible_scene(n=3, width=width, height=height, device=device)
    cfg = RasterConfig(max_sort_buffer_bytes=128 * 1024 * 1024)

    projection_meta = render_projection_meta_warp(
        means=scene["means"],
        quat=scene["quat"],
        scale=scene["scale"],
        viewmat=scene["viewmat"],
        K=scene["K"],
        width=width,
        height=height,
        cfg=cfg,
    )
    visibility_meta = render_visibility_meta_warp(
        means=scene["means"],
        quat=scene["quat"],
        scale=scene["scale"],
        viewmat=scene["viewmat"],
        K=scene["K"],
        width=width,
        height=height,
        cfg=cfg,
    )

    assert projection_meta["gaussian_count"] == int(visibility_meta["meta_gaussian_count"].item())
    assert projection_meta["visible_count"] == int(visibility_meta["meta_visible_count"].item())
    assert projection_meta["intersection_count"] == int(visibility_meta["meta_intersection_count"].item())
    assert projection_meta["tile_count"] == int(visibility_meta["meta_tile_count"].item())
    assert projection_meta["tiles_x"] == int(visibility_meta["meta_tiles_x"].item())
    assert projection_meta["tiles_y"] == int(visibility_meta["meta_tiles_y"].item())
    assert projection_meta["render_width"] == int(visibility_meta["meta_render_width"].item())
    assert projection_meta["render_height"] == int(visibility_meta["meta_render_height"].item())
    assert projection_meta["sort_mode"] in {"warp_radix", "torch_sort"}
    assert int(projection_meta["estimated_sort_buffer_bytes"]) >= 0
    assert bool(projection_meta["sort_buffer_within_budget"]) is True
    assert int(projection_meta["torch_sort_buffer_bytes"]) == int(projection_meta["intersection_count"]) * 12
    assert int(projection_meta["warp_sort_buffer_bytes"]) == int(projection_meta["intersection_count"]) * 24
