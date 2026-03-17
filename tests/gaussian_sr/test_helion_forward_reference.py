import pytest
import torch
from torch.testing import assert_close

from blender_temp.gaussian_sr import (
    RasterConfig,
    render_stats_prepared_helion,
    render_stats_prepared_warp,
    render_values_helion,
    render_values_warp,
)
from blender_temp.gaussian_sr.reference_renderer import render_values_reference


CUDA_AVAILABLE = bool(torch.cuda.is_available())


def _base_intrinsics(width: int, height: int, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [
            [12.0, 0.0, width / 2.0],
            [0.0, 12.0, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=torch.float32,
    )


def _identity_viewmat(device: torch.device) -> torch.Tensor:
    return torch.eye(4, device=device, dtype=torch.float32)


def _scene_single(device: torch.device) -> dict[str, torch.Tensor | int]:
    width = 16
    height = 16
    return {
        "means": torch.tensor([[0.0, 0.0, 3.0]], device=device, dtype=torch.float32),
        "quat": torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=torch.float32),
        "scale": torch.tensor([[0.28, 0.24, 0.18]], device=device, dtype=torch.float32),
        "values": torch.tensor([[0.75, 0.30, 0.15]], device=device, dtype=torch.float32),
        "opacity": torch.tensor([0.62], device=device, dtype=torch.float32),
        "viewmat": _identity_viewmat(device),
        "K": _base_intrinsics(width, height, device),
        "width": width,
        "height": height,
    }


def _scene_overlap2(device: torch.device) -> dict[str, torch.Tensor | int]:
    width = 16
    height = 16
    return {
        "means": torch.tensor(
            [
                [-0.08, 0.02, 3.0],
                [0.06, -0.01, 3.08],
            ],
            device=device,
            dtype=torch.float32,
        ),
        "quat": torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            device=device,
            dtype=torch.float32,
        ),
        "scale": torch.tensor(
            [
                [0.32, 0.28, 0.18],
                [0.30, 0.30, 0.20],
            ],
            device=device,
            dtype=torch.float32,
        ),
        "values": torch.tensor(
            [
                [0.85, 0.20, 0.20],
                [0.20, 0.65, 0.85],
            ],
            device=device,
            dtype=torch.float32,
        ),
        "opacity": torch.tensor([0.58, 0.46], device=device, dtype=torch.float32),
        "viewmat": _identity_viewmat(device),
        "K": _base_intrinsics(width, height, device),
        "width": width,
        "height": height,
    }


SCENES = {
    "single": _scene_single,
    "overlap2": _scene_overlap2,
}


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA")
@pytest.mark.parametrize("scene_name", ["single", "overlap2"])
def test_helion_forward_matches_reference(scene_name: str) -> None:
    device = torch.device("cuda")
    scene = SCENES[scene_name](device)
    cfg = RasterConfig(backend="helion")
    cfg.helion_runtime_autotune = False
    background = torch.zeros(scene["values"].shape[1], device=device, dtype=torch.float32)

    ref = render_values_reference(
        means=scene["means"],
        quat=scene["quat"],
        scale=scene["scale"],
        values=scene["values"],
        opacity=scene["opacity"],
        background=background,
        viewmat=scene["viewmat"],
        K=scene["K"],
        width=int(scene["width"]),
        height=int(scene["height"]),
        cfg=cfg,
    )
    helion = render_values_helion(
        means=scene["means"],
        quat=scene["quat"],
        scale=scene["scale"],
        values=scene["values"],
        opacity=scene["opacity"],
        background=background,
        viewmat=scene["viewmat"],
        K=scene["K"],
        width=int(scene["width"]),
        height=int(scene["height"]),
        cfg=cfg,
    )

    assert_close(helion, ref, atol=1.0e-5, rtol=1.0e-5)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA")
@pytest.mark.parametrize("scene_name", ["single", "overlap2"])
def test_helion_stats_matches_warp(scene_name: str) -> None:
    device = torch.device("cuda")
    scene = SCENES[scene_name](device)
    warp_cfg = RasterConfig(backend="warp")
    helion_cfg = RasterConfig(backend="helion")
    helion_cfg.helion_runtime_autotune = False
    background = torch.zeros(scene["values"].shape[1], device=device, dtype=torch.float32)
    residual_map = torch.linspace(
        0.0,
        1.0,
        steps=int(scene["height"]) * int(scene["width"]),
        device=device,
        dtype=torch.float32,
    ).view(int(scene["height"]), int(scene["width"]))

    _, prepared = render_values_warp(
        means=scene["means"],
        quat=scene["quat"],
        scale=scene["scale"],
        values=scene["values"],
        opacity=scene["opacity"],
        background=background,
        viewmat=scene["viewmat"],
        K=scene["K"],
        width=int(scene["width"]),
        height=int(scene["height"]),
        cfg=warp_cfg,
        return_prepared=True,
    )
    warp_stats = render_stats_prepared_warp(
        prepared,
        opacity=scene["opacity"],
        cfg=warp_cfg,
        residual_map=residual_map,
    )
    helion_stats = render_stats_prepared_helion(
        prepared,
        opacity=scene["opacity"],
        cfg=helion_cfg,
        residual_map=residual_map,
    )

    for key in ("contrib", "transmittance", "hits", "residual", "error_map"):
        assert_close(helion_stats[key], warp_stats[key], atol=1.0e-4, rtol=1.0e-4)
