from pathlib import Path

import pytest
import torch

from blender_temp.cmd.main import collect_image_paths, load_images
from blender_temp.gaussian_sr import PoseFreeGaussianConfig, PoseFreeGaussianSR
from blender_temp.gaussian_sr.warp_runtime import _WARP_AVAILABLE

CUDA_WARP_AVAILABLE = bool(_WARP_AVAILABLE and torch.cuda.is_available())
RENDER_DIR = Path(__file__).resolve().parents[2] / "render_1920_1080"


@pytest.mark.skipif(not CUDA_WARP_AVAILABLE, reason="requires CUDA + NVIDIA Warp")
@pytest.mark.skipif(not RENDER_DIR.is_dir(), reason="requires local render_1920_1080 dataset")
def test_real_blender_frames_stay_finite_for_one_step() -> None:
    device = torch.device("cuda")
    image_paths = collect_image_paths(RENDER_DIR)[:3]
    if len(image_paths) < 3:
        pytest.skip("requires at least 3 input frames in render_1920_1080")

    images = load_images(image_paths, device)

    cfg = PoseFreeGaussianConfig()
    cfg.camera.learn_intrinsics = False
    cfg.train.stage_scales = (0.25,)
    cfg.train.steps_per_stage = (1,)
    cfg.field.anchor_stride = 1
    cfg.field.feature_dim = 8

    pipeline = PoseFreeGaussianSR.from_images(images, intrinsics=None, config=cfg).to(device)

    history = pipeline.fit(images)

    assert len(history["loss"]) == 1
    assert torch.isfinite(torch.tensor(history["loss"], dtype=torch.float32)).all()
    assert torch.isfinite(torch.tensor(history["photo"], dtype=torch.float32)).all()
    assert torch.isfinite(torch.tensor(history["reg"], dtype=torch.float32)).all()
