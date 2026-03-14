#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
import sys
from dataclasses import dataclass

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from blender_temp.gaussian_sr.warp_gsplat_renderer import RasterConfig, render_values_warp


@dataclass
class Scene:
    means: torch.Tensor
    quat: torch.Tensor
    scale: torch.Tensor
    values: torch.Tensor
    opacity: torch.Tensor
    viewmat: torch.Tensor
    K: torch.Tensor
    width: int
    height: int


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


def _scene_single(device: torch.device) -> tuple[Scene, Scene]:
    width = 16
    height = 16
    means = torch.tensor([[0.0, 0.0, 3.0]], device=device, dtype=torch.float32)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
    scale = torch.tensor([[0.28, 0.24, 0.18]], device=device, dtype=torch.float32)
    values = torch.tensor([[0.75, 0.30, 0.15]], device=device, dtype=torch.float32)
    opacity = torch.tensor([0.62], device=device, dtype=torch.float32)
    viewmat = _identity_viewmat(device)
    K = _base_intrinsics(width, height, device)

    ref_means = means.clone()
    ref_means[0, 0] += 0.05
    ref_means[0, 2] += 0.08
    ref_opacity = opacity.clone()
    ref_opacity[0] = 0.70
    ref_K = K.clone()
    ref_K[0, 0] += 0.35

    return (
        Scene(means, quat, scale, values, opacity, viewmat, K, width, height),
        Scene(
            ref_means, quat.clone(), scale.clone(), values.clone(), ref_opacity, viewmat.clone(), ref_K, width, height
        ),
    )


def _scene_overlap2(device: torch.device) -> tuple[Scene, Scene]:
    width = 16
    height = 16
    means = torch.tensor(
        [
            [-0.08, 0.02, 3.0],
            [0.06, -0.01, 3.08],
        ],
        device=device,
        dtype=torch.float32,
    )
    quat = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        device=device,
        dtype=torch.float32,
    )
    scale = torch.tensor(
        [
            [0.32, 0.28, 0.18],
            [0.30, 0.30, 0.20],
        ],
        device=device,
        dtype=torch.float32,
    )
    values = torch.tensor(
        [
            [0.85, 0.20, 0.20],
            [0.20, 0.65, 0.85],
        ],
        device=device,
        dtype=torch.float32,
    )
    opacity = torch.tensor([0.58, 0.46], device=device, dtype=torch.float32)
    viewmat = _identity_viewmat(device)
    K = _base_intrinsics(width, height, device)

    ref_means = means.clone()
    ref_means[0, 0] += 0.04
    ref_means[1, 1] -= 0.05
    ref_means[1, 2] += 0.06
    ref_opacity = opacity.clone()
    ref_opacity[0] = 0.66
    ref_opacity[1] = 0.40
    ref_K = K.clone()
    ref_K[0, 0] += 0.30
    ref_K[1, 2] -= 0.25

    return (
        Scene(means, quat, scale, values, opacity, viewmat, K, width, height),
        Scene(
            ref_means, quat.clone(), scale.clone(), values.clone(), ref_opacity, viewmat.clone(), ref_K, width, height
        ),
    )


def _clone_scene(scene: Scene, *, requires_grad: bool) -> Scene:
    def clone(t: torch.Tensor) -> torch.Tensor:
        out = t.clone().detach().contiguous()
        out.requires_grad_(requires_grad)
        return out

    return Scene(
        means=clone(scene.means),
        quat=scene.quat.clone().detach().contiguous(),
        scale=scene.scale.clone().detach().contiguous(),
        values=scene.values.clone().detach().contiguous(),
        opacity=clone(scene.opacity),
        viewmat=scene.viewmat.clone().detach().contiguous(),
        K=clone(scene.K),
        width=scene.width,
        height=scene.height,
    )


def _render_loss(scene: Scene, target: torch.Tensor, cfg: RasterConfig) -> torch.Tensor:
    out = render_values_warp(
        means=scene.means,
        quat=scene.quat,
        scale=scene.scale,
        values=scene.values,
        opacity=scene.opacity,
        viewmat=scene.viewmat,
        K=scene.K,
        width=scene.width,
        height=scene.height,
        cfg=cfg,
    )
    return torch.mean((out - target) ** 2)


def _finite_difference(
    scene: Scene,
    target: torch.Tensor,
    cfg: RasterConfig,
    tensor_name: str,
    index: tuple[int, ...],
    eps: float,
) -> float:
    plus = _clone_scene(scene, requires_grad=False)
    minus = _clone_scene(scene, requires_grad=False)
    plus_t = getattr(plus, tensor_name)
    minus_t = getattr(minus, tensor_name)
    plus_t[index] += eps
    minus_t[index] -= eps
    if tensor_name == "opacity":
        plus_t.clamp_(1.0e-4, 1.0 - 1.0e-4)
        minus_t.clamp_(1.0e-4, 1.0 - 1.0e-4)
    loss_plus = float(_render_loss(plus, target, cfg).item())
    loss_minus = float(_render_loss(minus, target, cfg).item())
    return (loss_plus - loss_minus) / (2.0 * eps)


def _relative_error(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1.0e-8)
    return abs(a - b) / denom


def run_case(name: str, scene_builder, eps_by_name: dict[str, float]) -> None:
    device = torch.device("cuda:0")
    cfg = RasterConfig()

    scene, reference_scene = scene_builder(device)
    with torch.no_grad():
        target = render_values_warp(
            means=reference_scene.means,
            quat=reference_scene.quat,
            scale=reference_scene.scale,
            values=reference_scene.values,
            opacity=reference_scene.opacity,
            viewmat=reference_scene.viewmat,
            K=reference_scene.K,
            width=reference_scene.width,
            height=reference_scene.height,
            cfg=cfg,
        )

    grad_scene = _clone_scene(scene, requires_grad=True)
    loss = _render_loss(grad_scene, target, cfg)
    loss.backward()

    checks = [
        ("means", (0, 0)),
        ("means", (0, 2)),
        ("opacity", (0,)),
        ("K", (0, 0)),
        ("K", (0, 2)),
    ]
    if int(scene.means.shape[0]) > 1:
        checks.extend([
            ("means", (1, 1)),
            ("opacity", (1,)),
        ])

    print(f"=== {name}")
    print(f"loss={float(loss.item()):.8f}")
    print("param,index,autograd,numeric,abs_err,rel_err")
    for tensor_name, index in checks:
        grad_tensor = getattr(grad_scene, tensor_name).grad
        assert grad_tensor is not None
        autograd_grad = float(grad_tensor[index].item())
        numeric_grad = _finite_difference(
            scene=scene,
            target=target,
            cfg=cfg,
            tensor_name=tensor_name,
            index=index,
            eps=eps_by_name.get(tensor_name, 1.0e-3),
        )
        abs_err = abs(autograd_grad - numeric_grad)
        rel_err = _relative_error(autograd_grad, numeric_grad)
        print(f"{tensor_name},{index},{autograd_grad:.8e},{numeric_grad:.8e},{abs_err:.8e},{rel_err:.8e}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--means-eps", type=float, default=1.0e-3)
    parser.add_argument("--opacity-eps", type=float, default=1.0e-3)
    parser.add_argument("--k-eps", type=float, default=1.0e-2)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this gradcheck.")

    torch.manual_seed(0)
    eps_by_name = {
        "means": float(args.means_eps),
        "opacity": float(args.opacity_eps),
        "K": float(args.k_eps),
    }
    run_case("single", _scene_single, eps_by_name)
    run_case("overlap2", _scene_overlap2, eps_by_name)


if __name__ == "__main__":
    main()
