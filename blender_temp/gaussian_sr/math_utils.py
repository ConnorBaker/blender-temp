import math

import torch
from torch import Tensor

from .posefree_config import CameraInit


def inverse_sigmoid(x: Tensor, eps: float = 1.0e-6) -> Tensor:
    return torch.special.logit(x, eps=eps)


def softplus_inverse(y: Tensor) -> Tensor:
    return y + torch.log(-torch.expm1(-y))


def skew(v: Tensor) -> Tensor:
    vx, vy, vz = v.unbind(dim=-1)
    o = torch.zeros_like(vx)
    return torch.stack(
        (
            torch.stack((o, -vz, vy), dim=-1),
            torch.stack((vz, o, -vx), dim=-1),
            torch.stack((-vy, vx, o), dim=-1),
        ),
        dim=-2,
    )


def so3_exp_map(omega: Tensor) -> Tensor:
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True)
    theta2 = theta * theta
    k = skew(omega)

    eye = torch.eye(3, device=omega.device, dtype=omega.dtype).expand(*omega.shape[:-1], 3, 3)
    small = theta <= 1.0e-5
    theta4 = theta2 * theta2
    a_small = 1.0 - theta2 / 6.0 + theta4 / 120.0
    b_small = 0.5 - theta2 / 24.0 + theta4 / 720.0

    # Avoid boolean index_put() and data-dependent control flow so torch.compile
    # can keep this function in-graph. The masked assignment path lowers through
    # nonzero(), which currently forces a graph break in Inductor.
    theta_safe = torch.where(small, torch.ones_like(theta), theta)
    theta2_safe = torch.where(small, torch.ones_like(theta2), theta2)
    a_large = torch.sin(theta_safe) / theta_safe
    b_large = (1.0 - torch.cos(theta_safe)) / theta2_safe
    a = torch.where(small, a_small, a_large)
    b = torch.where(small, b_small, b_large)

    a = a.unsqueeze(-1)
    b = b.unsqueeze(-1)
    return eye + a * k + b * (k @ k)


def pose_vec_to_rt(xi: Tensor) -> tuple[Tensor, Tensor]:
    omega = xi[..., :3]
    trans = xi[..., 3:]
    r = so3_exp_map(omega)
    return r, trans


def normalize_quaternion(q: Tensor, eps: float = 1.0e-8) -> Tensor:
    return torch.nn.functional.normalize(q, dim=-1, eps=eps)


def quaternion_to_matrix(q: Tensor) -> Tensor:
    q = normalize_quaternion(q)
    w, x, y, z = q.unbind(dim=-1)

    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    return torch.stack(
        (
            torch.stack((ww + xx - yy - zz, 2.0 * (xy - wz), 2.0 * (xz + wy)), dim=-1),
            torch.stack((2.0 * (xy + wz), ww - xx + yy - zz, 2.0 * (yz - wx)), dim=-1),
            torch.stack((2.0 * (xz - wy), 2.0 * (yz + wx), ww - xx - yy + zz), dim=-1),
        ),
        dim=-2,
    )


def covariance_from_quat_scale(quat: Tensor, scale: Tensor) -> Tensor:
    r = quaternion_to_matrix(quat)
    rs = r * scale.unsqueeze(-2)
    return rs @ rs.transpose(-1, -2)


def default_intrinsics(
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    init: CameraInit,
) -> tuple[Tensor, Tensor]:
    if init.fx is None or init.fy is None:
        f = 0.5 * width / math.tan(0.5 * math.radians(init.default_fov_degrees))
        fx = f if init.fx is None else init.fx
        fy = f if init.fy is None else init.fy
    else:
        fx = init.fx
        fy = init.fy
    cx = (width - 1.0) * 0.5 if init.cx is None else init.cx
    cy = (height - 1.0) * 0.5 if init.cy is None else init.cy
    focal = torch.tensor([fx, fy], device=device, dtype=dtype)
    principal = torch.tensor([cx, cy], device=device, dtype=dtype)
    return focal, principal


__all__ = [
    "inverse_sigmoid",
    "softplus_inverse",
    "skew",
    "so3_exp_map",
    "pose_vec_to_rt",
    "normalize_quaternion",
    "quaternion_to_matrix",
    "covariance_from_quat_scale",
    "default_intrinsics",
]
