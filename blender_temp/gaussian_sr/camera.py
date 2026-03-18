import torch
import torch.nn as nn
from torch import Tensor

from .math_utils import so3_exp_map


class LearnableSharedIntrinsics(nn.Module):
    def __init__(
        self,
        focal: Tensor,
        principal: Tensor,
        *,
        learn: bool,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.log_focal = nn.Parameter(
            torch.log(focal.to(device=device, dtype=dtype)),
            requires_grad=learn,
        )
        self.principal = nn.Parameter(
            principal.to(device=device, dtype=dtype),
            requires_grad=learn,
        )

    def get(self, scale: Tensor) -> tuple[Tensor, Tensor]:
        """Return (focal, principal) scaled by ``scale`` ([sx, sy] tensor)."""
        focal = torch.exp(self.log_focal) * scale
        pp = (self.principal + 0.5) * scale - 0.5
        return focal, pp


class LearnableCameraBundle(nn.Module):
    """Per-view camera poses.  View 0 is always identity; views 1..V-1
    have learnable rotation (so3) and translation parameters."""

    def __init__(
        self,
        num_views: int,
        *,
        focal: Tensor | None = None,
        init_shifts_px: Tensor | None = None,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        if num_views < 1:
            raise ValueError("num_views must be >= 1")
        if num_views > 1 and focal is None:
            raise ValueError("focal is required for multi-view camera bundle")

        rest = num_views - 1
        learn = rest > 0
        init_omega = torch.zeros(rest, 3, device=device, dtype=dtype)
        init_trans = torch.zeros(rest, 3, device=device, dtype=dtype)
        if learn and init_shifts_px is not None:
            init_trans[:, 0] = -init_shifts_px[1:, 0] / focal[0].clamp(min=1.0)
            init_trans[:, 1] = -init_shifts_px[1:, 1] / focal[1].clamp(min=1.0)

        self.omega_rest = nn.Parameter(init_omega, requires_grad=learn)
        self.trans_rest = nn.Parameter(init_trans, requires_grad=learn)
        self.register_buffer("_identity_R", torch.eye(3, device=device, dtype=dtype)[None])
        self.register_buffer("_zero_t", torch.zeros(1, 3, device=device, dtype=dtype))

    def world_to_camera(self) -> tuple[Tensor, Tensor]:
        R = torch.cat((self._identity_R, so3_exp_map(self.omega_rest)), dim=0)
        t = torch.cat((self._zero_t, self.trans_rest), dim=0)
        return R, t

    def pose_regularizer(self) -> Tensor:
        total = torch.square(self.omega_rest).sum() + torch.square(self.trans_rest).sum()
        n = self.omega_rest.numel() + self.trans_rest.numel()
        return total / max(n, 1)


__all__ = [
    "LearnableSharedIntrinsics",
    "LearnableCameraBundle",
]
