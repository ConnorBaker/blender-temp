import pytest
import torch

from blender_temp.gaussian_sr import PoseFreeGaussianConfig, PoseFreeGaussianSR
from blender_temp.gaussian_sr.warp_runtime import _WARP_AVAILABLE

CUDA_WARP_AVAILABLE = bool(_WARP_AVAILABLE and torch.cuda.is_available())


def _problem_config() -> PoseFreeGaussianConfig:
    cfg = PoseFreeGaussianConfig()
    cfg.camera.learn_intrinsics = False
    cfg.train.stage_scales = (0.25,)
    cfg.train.steps_per_stage = (1,)
    cfg.field.anchor_stride = 1
    cfg.field.feature_dim = 8
    cfg.train.print_every = 0
    return cfg


@pytest.mark.skipif(not CUDA_WARP_AVAILABLE, reason="requires CUDA + NVIDIA Warp")
def test_single_view_synthetic_problem_config_stays_finite_for_one_step() -> None:
    device = torch.device("cuda")
    images = torch.zeros((1, 3, 16, 16), device=device, dtype=torch.float32)
    pipeline = PoseFreeGaussianSR.from_images(images, intrinsics=None, config=_problem_config()).to(device)

    history = pipeline.fit(images)

    assert len(history["loss"]) == 1
    assert torch.isfinite(torch.tensor(history["loss"], dtype=torch.float32)).all()
    assert torch.isfinite(torch.tensor(history["photo"], dtype=torch.float32)).all()
    assert torch.isfinite(torch.tensor(history["reg"], dtype=torch.float32)).all()


@pytest.mark.skipif(not CUDA_WARP_AVAILABLE, reason="requires CUDA + NVIDIA Warp")
def test_two_view_synthetic_problem_config_stays_finite_for_one_step() -> None:
    device = torch.device("cuda")
    images = torch.zeros((2, 3, 16, 16), device=device, dtype=torch.float32)
    pipeline = PoseFreeGaussianSR.from_images(images, intrinsics=None, config=_problem_config()).to(device)

    history = pipeline.fit(images)

    assert len(history["loss"]) == 1
    assert torch.isfinite(torch.tensor(history["loss"], dtype=torch.float32)).all()
    assert torch.isfinite(torch.tensor(history["photo"], dtype=torch.float32)).all()
    assert torch.isfinite(torch.tensor(history["reg"], dtype=torch.float32)).all()
