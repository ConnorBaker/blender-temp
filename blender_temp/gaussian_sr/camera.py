import math

import torch
import torch.nn as nn
from torch import Tensor

from .math_utils import pose_vec_to_rt


class LearnableSharedIntrinsics(nn.Module):
    def __init__(self, initial: Tensor, learn_intrinsics: bool):
        super().__init__()
        fx, fy, cx, cy = initial.tolist()
        register = self.register_parameter if learn_intrinsics else self.register_buffer
        wrap = nn.Parameter if learn_intrinsics else (lambda x: x)

        def _register_scalar(name: str, value: float) -> None:
            register(name, wrap(torch.tensor(value, device=initial.device, dtype=initial.dtype)))

        _register_scalar("log_fx", math.log(fx))
        _register_scalar("log_fy", math.log(fy))
        _register_scalar("cx", cx)
        _register_scalar("cy", cy)

    def get(self, scale_x: float = 1.0, scale_y: float = 1.0) -> Tensor:
        fx = torch.exp(self.log_fx) * scale_x
        fy = torch.exp(self.log_fy) * scale_y
        cx = (self.cx + 0.5) * scale_x - 0.5
        cy = (self.cy + 0.5) * scale_y - 0.5
        return torch.stack((fx, fy, cx, cy), dim=0)


class LearnableCameraBundle(nn.Module):
    def __init__(
        self,
        num_views: int,
        fx: float,
        fy: float,
        init_shifts_px: Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        if num_views < 1:
            raise ValueError("num_views must be >= 1")

        init_pose = torch.zeros(num_views - 1, 6, device=device, dtype=dtype)
        if num_views > 1 and init_shifts_px is not None:
            init_pose[:, 3] = -init_shifts_px[1:, 0] / max(fx, 1.0)
            init_pose[:, 4] = -init_shifts_px[1:, 1] / max(fy, 1.0)

        self.pose_rest = nn.Parameter(init_pose)

    def world_to_camera(self) -> tuple[Tensor, Tensor]:
        if self.pose_rest.numel() == 0:
            r = torch.eye(3, device=self.pose_rest.device, dtype=self.pose_rest.dtype)[None]
            t = torch.zeros(1, 3, device=self.pose_rest.device, dtype=self.pose_rest.dtype)
            return r, t

        zeros = torch.zeros(1, 6, device=self.pose_rest.device, dtype=self.pose_rest.dtype)
        xi = torch.cat((zeros, self.pose_rest), dim=0)
        r, t = pose_vec_to_rt(xi)
        return r, t

    def pose_regularizer(self) -> Tensor:
        if self.pose_rest.numel() == 0:
            return self.pose_rest.new_tensor(0.0)
        return (self.pose_rest * self.pose_rest).mean()


__all__ = [
    "LearnableSharedIntrinsics",
    "LearnableCameraBundle",
]
