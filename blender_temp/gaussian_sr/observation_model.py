import math

import torch
import torch.nn.functional as F
from torch import Tensor

from .posefree_config import ObservationConfig


def area_downsample_chw(image: Tensor, out_h: int, out_w: int) -> Tensor:
    if image.ndim != 3:
        raise ValueError(f"Expected CHW image, got shape {tuple(image.shape)}")
    if image.shape[-2] == out_h and image.shape[-1] == out_w:
        return image
    return F.interpolate(image.unsqueeze(0), size=(out_h, out_w), mode="area").squeeze(0)


def area_downsample_hwc(image: Tensor, out_h: int, out_w: int) -> Tensor:
    if image.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape {tuple(image.shape)}")
    if image.shape[0] == out_h and image.shape[1] == out_w:
        return image
    chw = image.permute(2, 0, 1).contiguous()
    down = F.interpolate(chw.unsqueeze(0), size=(out_h, out_w), mode="area").squeeze(0)
    return down.permute(1, 2, 0).contiguous()


def observation_render_size(target_h: int, target_w: int, cfg: ObservationConfig) -> tuple[int, int]:
    if cfg.mode in ("identity", "area"):
        return target_h, target_w
    if cfg.mode == "supersample_area":
        s = max(float(cfg.supersample_factor), 1.0)
        return max(1, int(math.ceil(target_h * s))), max(1, int(math.ceil(target_w * s)))
    raise ValueError(f"Unsupported observation mode: {cfg.mode}")


def observe_rgb(image: Tensor, out_h: int, out_w: int, layout: str = "chw") -> Tensor:
    if layout == "chw":
        return area_downsample_chw(image, out_h, out_w)
    if layout == "hwc":
        return area_downsample_hwc(image, out_h, out_w)
    raise ValueError(f"Unsupported layout: {layout}")


def apply_observation_model(
    image: Tensor, target_h: int, target_w: int, cfg: ObservationConfig, layout: str = "chw"
) -> Tensor:
    if cfg.mode == "identity":
        if layout == "chw" and image.shape[-2:] == (target_h, target_w):
            return image
        if layout == "hwc" and image.shape[:2] == (target_h, target_w):
            return image
        raise ValueError("identity observation requires render size to match target size")
    if cfg.mode in ("area", "supersample_area"):
        return observe_rgb(image, target_h, target_w, layout=layout)
    raise ValueError(f"Unsupported observation mode: {cfg.mode}")


def render_observe_rgb(
    rendered: Tensor, target_h: int, target_w: int, cfg: ObservationConfig, layout: str = "chw"
) -> Tensor:
    return apply_observation_model(rendered, target_h, target_w, cfg, layout=layout)


__all__ = [
    "area_downsample_chw",
    "area_downsample_hwc",
    "observation_render_size",
    "observe_rgb",
    "apply_observation_model",
    "render_observe_rgb",
]
