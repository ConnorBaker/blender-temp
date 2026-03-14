import pytest
import torch
from torch.testing import assert_close

from blender_temp.gaussian_sr import RasterConfig, render_values_warp
from blender_temp.gaussian_sr.reference_renderer import (
    project_gaussians_reference,
    render_values_from_prepared_reference,
    render_values_reference,
)
from blender_temp.gaussian_sr.warp_runtime import _WARP_AVAILABLE

CUDA_WARP_AVAILABLE = bool(_WARP_AVAILABLE and torch.cuda.is_available())


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


def _scene_single(device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    width = 16
    height = 16
    scene = {
        "means": torch.tensor([[0.0, 0.0, 3.0]], device=device, dtype=torch.float32),
        "quat": torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=torch.float32),
        "scale": torch.tensor([[0.28, 0.24, 0.18]], device=device, dtype=torch.float32),
        "values": torch.tensor([[0.75, 0.30, 0.15]], device=device, dtype=torch.float32),
        "opacity": torch.tensor([0.62], device=device, dtype=torch.float32),
        "viewmat": _identity_viewmat(device),
        "K": _base_intrinsics(width, height, device),
        "width": torch.tensor(width, device=device, dtype=torch.int32),
        "height": torch.tensor(height, device=device, dtype=torch.int32),
    }
    target_scene = {
        "means": scene["means"].clone(),
        "quat": scene["quat"].clone(),
        "scale": scene["scale"].clone(),
        "values": scene["values"].clone(),
        "opacity": scene["opacity"].clone(),
        "viewmat": scene["viewmat"].clone(),
        "K": scene["K"].clone(),
        "width": scene["width"].clone(),
        "height": scene["height"].clone(),
    }
    target_scene["means"][0, 0] += 0.05
    target_scene["means"][0, 2] += 0.08
    target_scene["opacity"][0] = 0.70
    target_scene["K"][0, 0] += 0.35
    return scene, target_scene


def _scene_overlap2(device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    width = 16
    height = 16
    scene = {
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
        "width": torch.tensor(width, device=device, dtype=torch.int32),
        "height": torch.tensor(height, device=device, dtype=torch.int32),
    }
    target_scene = {
        "means": scene["means"].clone(),
        "quat": scene["quat"].clone(),
        "scale": scene["scale"].clone(),
        "values": scene["values"].clone(),
        "opacity": scene["opacity"].clone(),
        "viewmat": scene["viewmat"].clone(),
        "K": scene["K"].clone(),
        "width": scene["width"].clone(),
        "height": scene["height"].clone(),
    }
    target_scene["means"][0, 0] += 0.04
    target_scene["means"][1, 1] -= 0.05
    target_scene["means"][1, 2] += 0.06
    target_scene["opacity"][0] = 0.66
    target_scene["opacity"][1] = 0.40
    target_scene["K"][0, 0] += 0.30
    target_scene["K"][1, 2] -= 0.25
    return scene, target_scene


def _clone_scene(
    scene: dict[str, torch.Tensor],
    *,
    requires_grad: bool,
    grad_keys: set[str] | None = None,
) -> dict[str, torch.Tensor]:
    if grad_keys is None:
        grad_keys = {"means", "opacity", "K"}
    out: dict[str, torch.Tensor] = {}
    for key, value in scene.items():
        clone = value.clone().detach().contiguous()
        if key in grad_keys:
            clone.requires_grad_(requires_grad)
        out[key] = clone
    return out


def _render_warp(scene: dict[str, torch.Tensor], cfg: RasterConfig) -> torch.Tensor:
    return render_values_warp(
        means=scene["means"],
        quat=scene["quat"],
        scale=scene["scale"],
        values=scene["values"],
        opacity=scene["opacity"],
        background=torch.zeros(scene["values"].shape[1], device=scene["values"].device, dtype=torch.float32),
        viewmat=scene["viewmat"],
        K=scene["K"],
        width=int(scene["width"].item()),
        height=int(scene["height"].item()),
        cfg=cfg,
    )


def _render_reference(scene: dict[str, torch.Tensor], cfg: RasterConfig) -> torch.Tensor:
    return render_values_reference(
        means=scene["means"],
        quat=scene["quat"],
        scale=scene["scale"],
        values=scene["values"],
        opacity=scene["opacity"],
        background=torch.zeros(scene["values"].shape[1], device=scene["values"].device, dtype=torch.float32),
        viewmat=scene["viewmat"],
        K=scene["K"],
        width=int(scene["width"].item()),
        height=int(scene["height"].item()),
        cfg=cfg,
    )


def _loss(render_fn, scene: dict[str, torch.Tensor], target: torch.Tensor, cfg: RasterConfig) -> torch.Tensor:
    out = render_fn(scene, cfg)
    return torch.mean((out - target) ** 2)


SCENE_BUILDERS = {
    "single": _scene_single,
    "overlap2": _scene_overlap2,
}


@pytest.mark.skipif(not CUDA_WARP_AVAILABLE, reason="requires CUDA + NVIDIA Warp")
@pytest.mark.parametrize("scene_name", ["single", "overlap2"])
def test_reference_backward_mode_matches_reference_backward(scene_name: str) -> None:
    device = torch.device("cuda")
    cfg = RasterConfig(backward_impl="reference")
    scene, target_scene = SCENE_BUILDERS[scene_name](device)
    with torch.no_grad():
        target = _render_reference(target_scene, cfg)

    ref_scene = _clone_scene(scene, requires_grad=True, grad_keys={"means", "values", "opacity", "K"})
    fast_scene = _clone_scene(scene, requires_grad=True, grad_keys={"means", "values", "opacity", "K"})

    ref_loss = _loss(_render_reference, ref_scene, target, cfg)
    fast_loss = _loss(_render_warp, fast_scene, target, cfg)
    ref_loss.backward()
    fast_loss.backward()

    assert ref_scene["means"].grad is not None
    assert fast_scene["means"].grad is not None
    assert ref_scene["values"].grad is not None
    assert fast_scene["values"].grad is not None
    assert ref_scene["opacity"].grad is not None
    assert fast_scene["opacity"].grad is not None
    assert ref_scene["K"].grad is not None
    assert fast_scene["K"].grad is not None

    assert_close(fast_scene["means"].grad, ref_scene["means"].grad, atol=5.0e-5, rtol=5.0e-2)
    assert_close(fast_scene["values"].grad, ref_scene["values"].grad, atol=5.0e-5, rtol=5.0e-2)
    assert_close(fast_scene["opacity"].grad, ref_scene["opacity"].grad, atol=5.0e-5, rtol=5.0e-2)
    assert_close(fast_scene["K"].grad, ref_scene["K"].grad, atol=5.0e-5, rtol=5.0e-2)


@pytest.mark.skipif(not CUDA_WARP_AVAILABLE, reason="requires CUDA + NVIDIA Warp")
@pytest.mark.parametrize("scene_name", ["single", "overlap2"])
def test_reference_renderer_matches_warp_forward(scene_name: str) -> None:
    device = torch.device("cuda")
    cfg = RasterConfig()
    scene, _target_scene = SCENE_BUILDERS[scene_name](device)

    ref = _render_reference(scene, cfg)
    fast = _render_warp(scene, cfg)

    assert_close(fast, ref, atol=1.0e-5, rtol=1.0e-5)


@pytest.mark.skipif(not CUDA_WARP_AVAILABLE, reason="requires CUDA + NVIDIA Warp")
@pytest.mark.parametrize("scene_name", ["single", "overlap2"])
def test_reference_projection_matches_warp_prepared_state(scene_name: str) -> None:
    device = torch.device("cuda")
    cfg = RasterConfig()
    scene, _target_scene = SCENE_BUILDERS[scene_name](device)

    _out, prepared = render_values_warp(
        means=scene["means"],
        quat=scene["quat"],
        scale=scene["scale"],
        values=scene["values"],
        opacity=scene["opacity"],
        background=torch.zeros(scene["values"].shape[1], device=device, dtype=torch.float32),
        viewmat=scene["viewmat"],
        K=scene["K"],
        width=int(scene["width"].item()),
        height=int(scene["height"].item()),
        cfg=cfg,
        return_prepared=True,
    )
    ref = project_gaussians_reference(
        means=scene["means"],
        quat=scene["quat"],
        scale=scene["scale"],
        viewmat=scene["viewmat"],
        K=scene["K"],
        width=int(scene["width"].item()),
        height=int(scene["height"].item()),
        cfg=cfg,
    )

    assert_close(prepared.xys, ref.xys, atol=1.0e-5, rtol=1.0e-5)
    assert_close(prepared.conic, ref.conic, atol=1.0e-5, rtol=1.0e-5)
    assert_close(prepared.rho, ref.rho, atol=1.0e-5, rtol=1.0e-5)
    assert_close(prepared.num_tiles_hit, ref.num_tiles_hit)


@pytest.mark.skipif(not CUDA_WARP_AVAILABLE, reason="requires CUDA + NVIDIA Warp")
@pytest.mark.parametrize("scene_name", ["single", "overlap2"])
def test_reference_raster_from_prepared_matches_warp_forward(scene_name: str) -> None:
    device = torch.device("cuda")
    cfg = RasterConfig()
    scene, _target_scene = SCENE_BUILDERS[scene_name](device)
    background = torch.zeros(scene["values"].shape[1], device=device, dtype=torch.float32)

    fast, prepared = render_values_warp(
        means=scene["means"],
        quat=scene["quat"],
        scale=scene["scale"],
        values=scene["values"],
        opacity=scene["opacity"],
        background=background,
        viewmat=scene["viewmat"],
        K=scene["K"],
        width=int(scene["width"].item()),
        height=int(scene["height"].item()),
        cfg=cfg,
        return_prepared=True,
    )
    ref = render_values_from_prepared_reference(
        prepared=prepared,
        values=scene["values"],
        opacity=scene["opacity"],
        background=background,
        cfg=cfg,
    )

    assert_close(fast, ref, atol=1.0e-5, rtol=1.0e-5)


@pytest.mark.skipif(not CUDA_WARP_AVAILABLE, reason="requires CUDA + NVIDIA Warp")
@pytest.mark.parametrize("scene_name", ["single", "overlap2"])
def test_reference_renderer_matches_warp_backward(scene_name: str) -> None:
    device = torch.device("cuda")
    cfg = RasterConfig(backward_impl="hybrid")
    scene, target_scene = SCENE_BUILDERS[scene_name](device)
    with torch.no_grad():
        target = _render_reference(target_scene, cfg)

    ref_scene = _clone_scene(scene, requires_grad=True)
    fast_scene = _clone_scene(scene, requires_grad=True)

    ref_loss = _loss(_render_reference, ref_scene, target, cfg)
    fast_loss = _loss(_render_warp, fast_scene, target, cfg)
    ref_loss.backward()
    fast_loss.backward()

    assert ref_scene["means"].grad is not None
    assert fast_scene["means"].grad is not None
    assert ref_scene["opacity"].grad is not None
    assert fast_scene["opacity"].grad is not None
    assert ref_scene["K"].grad is not None
    assert fast_scene["K"].grad is not None

    assert_close(fast_scene["means"].grad, ref_scene["means"].grad, atol=5.0e-5, rtol=5.0e-2)
    assert_close(fast_scene["opacity"].grad, ref_scene["opacity"].grad, atol=5.0e-5, rtol=5.0e-2)
    assert_close(fast_scene["K"].grad, ref_scene["K"].grad, atol=5.0e-5, rtol=5.0e-2)


@pytest.mark.skipif(not CUDA_WARP_AVAILABLE, reason="requires CUDA + NVIDIA Warp")
@pytest.mark.parametrize("scene_name", ["single", "overlap2"])
def test_reference_raster_from_prepared_matches_warp_backward_for_values_and_opacity(scene_name: str) -> None:
    device = torch.device("cuda")
    cfg = RasterConfig(backward_impl="hybrid")
    base_scene, target_scene = SCENE_BUILDERS[scene_name](device)
    background = torch.zeros(base_scene["values"].shape[1], device=device, dtype=torch.float32)

    with torch.no_grad():
        _out, prepared = render_values_warp(
            means=base_scene["means"],
            quat=base_scene["quat"],
            scale=base_scene["scale"],
            values=base_scene["values"],
            opacity=base_scene["opacity"],
            background=background,
            viewmat=base_scene["viewmat"],
            K=base_scene["K"],
            width=int(base_scene["width"].item()),
            height=int(base_scene["height"].item()),
            cfg=cfg,
            return_prepared=True,
        )
        target = render_values_from_prepared_reference(
            prepared=prepared,
            values=target_scene["values"],
            opacity=target_scene["opacity"],
            background=background,
            cfg=cfg,
        )

    fast_scene = _clone_scene(base_scene, requires_grad=True, grad_keys={"values", "opacity"})
    ref_scene = _clone_scene(base_scene, requires_grad=True, grad_keys={"values", "opacity"})

    fast_out = render_values_warp(
        means=fast_scene["means"],
        quat=fast_scene["quat"],
        scale=fast_scene["scale"],
        values=fast_scene["values"],
        opacity=fast_scene["opacity"],
        background=background,
        viewmat=fast_scene["viewmat"],
        K=fast_scene["K"],
        width=int(fast_scene["width"].item()),
        height=int(fast_scene["height"].item()),
        cfg=cfg,
    )
    ref_out = render_values_from_prepared_reference(
        prepared=prepared,
        values=ref_scene["values"],
        opacity=ref_scene["opacity"],
        background=background,
        cfg=cfg,
    )
    fast_loss = torch.mean((fast_out - target) ** 2)
    ref_loss = torch.mean((ref_out - target) ** 2)
    fast_loss.backward()
    ref_loss.backward()

    assert fast_scene["values"].grad is not None
    assert ref_scene["values"].grad is not None
    assert fast_scene["opacity"].grad is not None
    assert ref_scene["opacity"].grad is not None
    assert_close(fast_scene["values"].grad, ref_scene["values"].grad, atol=5.0e-5, rtol=5.0e-2)
    assert_close(fast_scene["opacity"].grad, ref_scene["opacity"].grad, atol=5.0e-5, rtol=5.0e-2)


@pytest.mark.skipif(not CUDA_WARP_AVAILABLE, reason="requires CUDA + NVIDIA Warp")
@pytest.mark.xfail(strict=True, reason="Warp Tape raster backward is still known to drift from reference")
@pytest.mark.parametrize("scene_name", ["single", "overlap2"])
def test_warp_tape_renderer_backward_drift_is_tracked(scene_name: str) -> None:
    device = torch.device("cuda")
    cfg = RasterConfig(backward_impl="warp_tape")
    scene, target_scene = SCENE_BUILDERS[scene_name](device)
    with torch.no_grad():
        target = _render_reference(target_scene, cfg)

    ref_scene = _clone_scene(scene, requires_grad=True)
    fast_scene = _clone_scene(scene, requires_grad=True)

    ref_loss = _loss(_render_reference, ref_scene, target, RasterConfig(backward_impl="reference"))
    fast_loss = _loss(_render_warp, fast_scene, target, cfg)
    ref_loss.backward()
    fast_loss.backward()

    assert ref_scene["means"].grad is not None
    assert fast_scene["means"].grad is not None
    assert ref_scene["opacity"].grad is not None
    assert fast_scene["opacity"].grad is not None
    assert ref_scene["K"].grad is not None
    assert fast_scene["K"].grad is not None

    assert_close(fast_scene["means"].grad, ref_scene["means"].grad, atol=5.0e-5, rtol=5.0e-2)
    assert_close(fast_scene["opacity"].grad, ref_scene["opacity"].grad, atol=5.0e-5, rtol=5.0e-2)
    assert_close(fast_scene["K"].grad, ref_scene["K"].grad, atol=5.0e-5, rtol=5.0e-2)


@pytest.mark.skipif(not CUDA_WARP_AVAILABLE, reason="requires CUDA + NVIDIA Warp")
@pytest.mark.xfail(strict=True, reason="Warp Tape raster backward is still known to drift from reference")
@pytest.mark.parametrize("scene_name", ["single", "overlap2"])
def test_warp_tape_prepared_raster_backward_drift_is_tracked(scene_name: str) -> None:
    device = torch.device("cuda")
    cfg = RasterConfig(backward_impl="warp_tape")
    base_scene, target_scene = SCENE_BUILDERS[scene_name](device)
    background = torch.zeros(base_scene["values"].shape[1], device=device, dtype=torch.float32)

    with torch.no_grad():
        _out, prepared = render_values_warp(
            means=base_scene["means"],
            quat=base_scene["quat"],
            scale=base_scene["scale"],
            values=base_scene["values"],
            opacity=base_scene["opacity"],
            background=background,
            viewmat=base_scene["viewmat"],
            K=base_scene["K"],
            width=int(base_scene["width"].item()),
            height=int(base_scene["height"].item()),
            cfg=cfg,
            return_prepared=True,
        )
        target = render_values_from_prepared_reference(
            prepared=prepared,
            values=target_scene["values"],
            opacity=target_scene["opacity"],
            background=background,
            cfg=RasterConfig(backward_impl="reference"),
        )

    fast_scene = _clone_scene(base_scene, requires_grad=True, grad_keys={"values", "opacity"})
    ref_scene = _clone_scene(base_scene, requires_grad=True, grad_keys={"values", "opacity"})

    fast_out = render_values_warp(
        means=fast_scene["means"],
        quat=fast_scene["quat"],
        scale=fast_scene["scale"],
        values=fast_scene["values"],
        opacity=fast_scene["opacity"],
        background=background,
        viewmat=fast_scene["viewmat"],
        K=fast_scene["K"],
        width=int(fast_scene["width"].item()),
        height=int(fast_scene["height"].item()),
        cfg=cfg,
    )
    ref_out = render_values_from_prepared_reference(
        prepared=prepared,
        values=ref_scene["values"],
        opacity=ref_scene["opacity"],
        background=background,
        cfg=RasterConfig(backward_impl="reference"),
    )
    fast_loss = torch.mean((fast_out - target) ** 2)
    ref_loss = torch.mean((ref_out - target) ** 2)
    fast_loss.backward()
    ref_loss.backward()

    assert fast_scene["values"].grad is not None
    assert ref_scene["values"].grad is not None
    assert fast_scene["opacity"].grad is not None
    assert ref_scene["opacity"].grad is not None
    assert_close(fast_scene["values"].grad, ref_scene["values"].grad, atol=5.0e-5, rtol=5.0e-2)
    assert_close(fast_scene["opacity"].grad, ref_scene["opacity"].grad, atol=5.0e-5, rtol=5.0e-2)
