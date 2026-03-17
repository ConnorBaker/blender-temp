import pytest
import torch
from torch.testing import assert_close

from blender_temp.gaussian_sr import RasterConfig, render_values
from blender_temp.gaussian_sr.reference_renderer import render_values_reference


CUDA_AVAILABLE = torch.cuda.is_available()


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
    target_scene = {key: value.clone() for key, value in scene.items()}
    target_scene["means"][0, 0] += 0.05
    target_scene["means"][0, 2] += 0.08
    target_scene["opacity"][0] = 0.70
    target_scene["K"][0, 0] += 0.35
    return scene, target_scene


def _scene_overlap2(device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    width = 16
    height = 16
    scene = {
        "means": torch.tensor([[-0.08, 0.02, 3.0], [0.06, -0.01, 3.08]], device=device, dtype=torch.float32),
        "quat": torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], device=device, dtype=torch.float32),
        "scale": torch.tensor([[0.32, 0.28, 0.18], [0.30, 0.30, 0.20]], device=device, dtype=torch.float32),
        "values": torch.tensor([[0.85, 0.20, 0.20], [0.20, 0.65, 0.85]], device=device, dtype=torch.float32),
        "opacity": torch.tensor([0.58, 0.46], device=device, dtype=torch.float32),
        "viewmat": _identity_viewmat(device),
        "K": _base_intrinsics(width, height, device),
        "width": torch.tensor(width, device=device, dtype=torch.int32),
        "height": torch.tensor(height, device=device, dtype=torch.int32),
    }
    target_scene = {key: value.clone() for key, value in scene.items()}
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
        grad_keys = {"means", "opacity", "values", "K"}
    out: dict[str, torch.Tensor] = {}
    for key, value in scene.items():
        clone = value.clone().detach().contiguous()
        if key in grad_keys:
            clone.requires_grad_(requires_grad)
        out[key] = clone
    return out


def _render_helion(scene: dict[str, torch.Tensor], cfg: RasterConfig) -> torch.Tensor:
    return render_values(
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


SCENE_BUILDERS = {
    "single": _scene_single,
    "overlap2": _scene_overlap2,
}


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA")
@pytest.mark.parametrize("scene_name", ["single", "overlap2"])
def test_helion_forward_matches_reference(scene_name: str) -> None:
    device = torch.device("cuda")
    cfg = RasterConfig(backend="helion", backward_impl="helion", helion_runtime_autotune=False)
    scene, _target_scene = SCENE_BUILDERS[scene_name](device)
    out = _render_helion(scene, cfg)
    ref = _render_reference(scene, cfg)
    assert_close(out, ref, atol=5.0e-5, rtol=5.0e-2)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA")
@pytest.mark.parametrize("scene_name", ["single", "overlap2"])
def test_helion_native_backward_matches_reference(scene_name: str) -> None:
    """Test that backward_impl='helion' matches backward_impl='reference'."""
    device = torch.device("cuda")
    helion_cfg = RasterConfig(backend="helion", backward_impl="helion", helion_runtime_autotune=False)
    ref_cfg = RasterConfig(backend="helion", backward_impl="reference", helion_runtime_autotune=False)
    scene, target_scene = SCENE_BUILDERS[scene_name](device)
    with torch.no_grad():
        target = _render_reference(target_scene, ref_cfg)

    ref_scene = _clone_scene(scene, requires_grad=True)
    helion_scene = _clone_scene(scene, requires_grad=True)

    ref_loss = torch.mean((_render_helion(ref_scene, ref_cfg) - target) ** 2)
    helion_loss = torch.mean((_render_helion(helion_scene, helion_cfg) - target) ** 2)
    ref_loss.backward()
    helion_loss.backward()

    for key in ("means", "values", "opacity", "K"):
        ref_grad = ref_scene[key].grad
        helion_grad = helion_scene[key].grad
        assert ref_grad is not None, f"reference {key}.grad is None"
        assert helion_grad is not None, f"helion {key}.grad is None"
        assert_close(helion_grad, ref_grad, atol=5.0e-5, rtol=5.0e-2)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA")
@pytest.mark.parametrize("scene_name", ["single", "overlap2"])
def test_helion_native_backward_forward_matches(scene_name: str) -> None:
    """Test that backward_impl='helion' forward output matches other impls."""
    device = torch.device("cuda")
    helion_cfg = RasterConfig(backend="helion", backward_impl="helion", helion_runtime_autotune=False)
    ref_cfg = RasterConfig(backend="helion", backward_impl="reference", helion_runtime_autotune=False)
    scene, _target_scene = SCENE_BUILDERS[scene_name](device)

    out_helion = _render_helion(scene, helion_cfg)
    out_ref = _render_helion(scene, ref_cfg)
    assert_close(out_helion, out_ref, atol=5.0e-5, rtol=5.0e-2)
