import math
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .appearance import apply_view_dependent_rgb, num_sh_bases
from .image_utils import pixel_grid
from .math_utils import inverse_sigmoid, normalize_quaternion, quaternion_to_matrix, softplus_inverse
from .posefree_config import AppearanceConfig, FieldConfig


class ScaleAwareResidualHead(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 64, residual_scale: float = 0.05):
        super().__init__()
        self.residual_scale = residual_scale
        self.net = nn.Sequential(
            nn.Linear(feature_dim + 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(
        self,
        latent_map: Tensor,
        scale_x: float,
        scale_y: float,
        chunk: int = 262144,
    ) -> Tensor:
        f_dim, h, w = latent_map.shape
        coords = pixel_grid(h, w, latent_map.device, latent_map.dtype, normalized=True).view(-1, 2)
        scale_token = latent_map.new_tensor([math.log(scale_x), math.log(scale_y)]).expand(coords.shape[0], 2)
        feats = latent_map.permute(1, 2, 0).reshape(-1, f_dim)
        inp = torch.cat((feats, coords, scale_token), dim=-1)

        out_chunks: list[Tensor] = []
        for start in range(0, inp.shape[0], chunk):
            pred = self.net(inp[start : start + chunk])
            out_chunks.append(pred)
        residual = torch.cat(out_chunks, dim=0).view(h, w, 3).permute(2, 0, 1)
        return self.residual_scale * torch.tanh(residual)


class CanonicalGaussianField(nn.Module):
    def __init__(
        self,
        anchor_rgb: Tensor,
        intrinsics: Tensor,
        field_cfg: FieldConfig,
        appearance_cfg: AppearanceConfig,
    ):
        super().__init__()
        if anchor_rgb.dim() != 3 or anchor_rgb.shape[0] != 3:
            raise ValueError("anchor_rgb must have shape [3, H, W]")

        device = anchor_rgb.device
        dtype = anchor_rgb.dtype
        _, h, w = anchor_rgb.shape
        s = field_cfg.anchor_stride

        ys = torch.arange(0, h, s, device=device, dtype=dtype)
        xs = torch.arange(0, w, s, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        uv = torch.stack((xx + 0.5, yy + 0.5), dim=-1)
        gh, gw = uv.shape[:2]
        uv_flat = uv.view(-1, 2)

        anchor_colors = anchor_rgb[:, ::s, ::s].permute(1, 2, 0).contiguous().view(-1, 3)
        init_depth = torch.full((gh * gw, 1), field_cfg.init_depth, device=device, dtype=dtype)
        depth_raw = softplus_inverse(init_depth - field_cfg.min_depth)

        fx, fy, _, _ = intrinsics
        sx = field_cfg.init_scale_xy * field_cfg.init_depth / max(float(fx.item()), 1.0) * s
        sy = field_cfg.init_scale_xy * field_cfg.init_depth / max(float(fy.item()), 1.0) * s
        sz = field_cfg.init_scale_z * field_cfg.init_depth
        init_scale = torch.tensor([sx, sy, sz], device=device, dtype=dtype).expand(gh * gw, 3)
        log_scale = torch.log(init_scale)

        quat = torch.zeros(gh * gw, 4, device=device, dtype=dtype)
        quat[:, 0] = 1.0

        self.gh = gh
        self.gw = gw
        self.feature_dim = field_cfg.feature_dim
        self.field_cfg = field_cfg
        self.appearance_cfg = appearance_cfg

        self.register_buffer("uv", uv_flat)
        self.depth_raw = nn.Parameter(depth_raw)
        self.xyz_offset = nn.Parameter(torch.zeros(gh * gw, 3, device=device, dtype=dtype))
        self.quat_raw = nn.Parameter(quat)
        self.log_scale = nn.Parameter(log_scale)
        self.opacity_logit = nn.Parameter(
            torch.full(
                (gh * gw, 1),
                inverse_sigmoid(torch.tensor(field_cfg.init_opacity, device=device, dtype=dtype)),
                device=device,
                dtype=dtype,
            )
        )
        self.rgb_logit = nn.Parameter(inverse_sigmoid(anchor_colors))
        self.latent = nn.Parameter(torch.zeros(gh * gw, field_cfg.feature_dim, device=device, dtype=dtype))
        if appearance_cfg.mode == "sh" and appearance_cfg.sh_degree > 0:
            basis_count = num_sh_bases(appearance_cfg.sh_degree) - 1
            self.sh_coeffs = nn.Parameter(torch.zeros(gh * gw, 3, basis_count, device=device, dtype=dtype))
        else:
            self.register_parameter("sh_coeffs", None)

    @property
    def num_gaussians(self) -> int:
        return int(self.depth_raw.shape[0])

    def parameters_for_optimizer(self) -> Iterable[nn.Parameter]:
        yield self.depth_raw
        yield self.xyz_offset
        yield self.quat_raw
        yield self.log_scale
        yield self.opacity_logit
        yield self.rgb_logit
        yield self.latent
        if self.sh_coeffs is not None:
            yield self.sh_coeffs

    def _replace_param(self, name: str, value: Tensor | None) -> None:
        if value is None:
            self.register_parameter(name, None)
        else:
            setattr(self, name, nn.Parameter(value.contiguous()))

    def _replace_buffer(self, name: str, value: Tensor) -> None:
        self._buffers[name] = value.contiguous()

    def prune_keep_mask(self, keep_mask: Tensor) -> None:
        if keep_mask.ndim != 1 or keep_mask.shape[0] != self.num_gaussians:
            raise ValueError("keep_mask must be [N]")
        if keep_mask.dtype != torch.bool:
            keep_mask = keep_mask.to(dtype=torch.bool)
        if int(keep_mask.sum().item()) <= 0:
            raise ValueError("prune_keep_mask would remove all gaussians")

        self._replace_buffer("uv", self.uv.detach()[keep_mask])
        self._replace_param("depth_raw", self.depth_raw.detach()[keep_mask])
        self._replace_param("xyz_offset", self.xyz_offset.detach()[keep_mask])
        self._replace_param("quat_raw", self.quat_raw.detach()[keep_mask])
        self._replace_param("log_scale", self.log_scale.detach()[keep_mask])
        self._replace_param("opacity_logit", self.opacity_logit.detach()[keep_mask])
        self._replace_param("rgb_logit", self.rgb_logit.detach()[keep_mask])
        self._replace_param("latent", self.latent.detach()[keep_mask])
        if self.sh_coeffs is not None:
            self._replace_param("sh_coeffs", self.sh_coeffs.detach()[keep_mask])

    def clone_gaussians(self, indices: Tensor, jitter_scale: float = 0.25) -> None:
        if indices.numel() == 0:
            return
        idx = indices.to(device=self.depth_raw.device, dtype=torch.long).view(-1)
        if idx.max().item() >= self.num_gaussians or idx.min().item() < 0:
            raise ValueError("clone_gaussians indices out of range")

        base_uv = self.uv.detach()
        base_depth_raw = self.depth_raw.detach()
        base_xyz_offset = self.xyz_offset.detach()
        base_quat_raw = self.quat_raw.detach()
        base_log_scale = self.log_scale.detach()
        base_opacity_logit = self.opacity_logit.detach().clone()
        base_rgb_logit = self.rgb_logit.detach()
        base_latent = self.latent.detach()
        base_sh = None if self.sh_coeffs is None else self.sh_coeffs.detach()

        src_opacity = torch.sigmoid(base_opacity_logit[idx]) * 0.5
        src_opacity = src_opacity.clamp(1.0e-4, 1.0 - 1.0e-4)
        base_opacity_logit[idx] = inverse_sigmoid(src_opacity)
        clone_opacity_logit = inverse_sigmoid(src_opacity)

        clone_uv = base_uv[idx].clone()
        clone_depth_raw = base_depth_raw[idx].clone()
        clone_xyz_offset = base_xyz_offset[idx].clone()
        clone_quat_raw = base_quat_raw[idx].clone()
        clone_log_scale = base_log_scale[idx].clone()
        clone_rgb_logit = base_rgb_logit[idx].clone()
        clone_latent = base_latent[idx].clone()
        clone_sh = None if base_sh is None else base_sh[idx].clone()

        if jitter_scale > 0.0:
            scale = torch.exp(clone_log_scale)
            clone_xyz_offset = clone_xyz_offset + torch.randn_like(clone_xyz_offset) * scale * float(jitter_scale)

        self._replace_buffer("uv", torch.cat((base_uv, clone_uv), dim=0))
        self._replace_param("depth_raw", torch.cat((base_depth_raw, clone_depth_raw), dim=0))
        self._replace_param("xyz_offset", torch.cat((base_xyz_offset, clone_xyz_offset), dim=0))
        self._replace_param("quat_raw", torch.cat((base_quat_raw, clone_quat_raw), dim=0))
        self._replace_param("log_scale", torch.cat((base_log_scale, clone_log_scale), dim=0))
        self._replace_param("opacity_logit", torch.cat((base_opacity_logit, clone_opacity_logit), dim=0))
        self._replace_param("rgb_logit", torch.cat((base_rgb_logit, clone_rgb_logit), dim=0))
        self._replace_param("latent", torch.cat((base_latent, clone_latent), dim=0))
        if base_sh is not None and clone_sh is not None:
            self._replace_param("sh_coeffs", torch.cat((base_sh, clone_sh), dim=0))

    def split_gaussians(self, indices: Tensor, shrink_factor: float = 0.8, offset_scale: float = 0.75) -> None:
        if indices.numel() == 0:
            return
        idx = indices.to(device=self.depth_raw.device, dtype=torch.long).view(-1)
        if idx.max().item() >= self.num_gaussians or idx.min().item() < 0:
            raise ValueError("split_gaussians indices out of range")

        base_uv = self.uv.detach()
        base_depth_raw = self.depth_raw.detach()
        base_xyz_offset = self.xyz_offset.detach().clone()
        base_quat_raw = self.quat_raw.detach()
        base_log_scale = self.log_scale.detach().clone()
        base_opacity_logit = self.opacity_logit.detach().clone()
        base_rgb_logit = self.rgb_logit.detach()
        base_latent = self.latent.detach()
        base_sh = None if self.sh_coeffs is None else self.sh_coeffs.detach()

        orig_xyz = base_xyz_offset[idx].clone()
        orig_log_scale = base_log_scale[idx].clone()
        rot = quaternion_to_matrix(normalize_quaternion(base_quat_raw[idx]))
        scales = torch.exp(orig_log_scale)
        axis_idx = scales.argmax(dim=1)
        batch = torch.arange(idx.shape[0], device=idx.device)
        axes = rot[batch, :, axis_idx]
        max_scale = scales.gather(1, axis_idx[:, None]).squeeze(1)
        delta = axes * (max_scale * float(offset_scale)).unsqueeze(1)

        shrink_log = math.log(max(float(shrink_factor), 1.0e-4))
        new_log_scale = orig_log_scale + shrink_log
        new_opacity = (torch.sigmoid(base_opacity_logit[idx]) * 0.5).clamp(1.0e-4, 1.0 - 1.0e-4)
        new_opacity_logit = inverse_sigmoid(new_opacity)

        base_xyz_offset[idx] = orig_xyz - delta
        base_log_scale[idx] = new_log_scale
        base_opacity_logit[idx] = new_opacity_logit

        child_uv = base_uv[idx].clone()
        child_depth_raw = base_depth_raw[idx].clone()
        child_xyz_offset = orig_xyz + delta
        child_quat_raw = base_quat_raw[idx].clone()
        child_log_scale = new_log_scale.clone()
        child_opacity_logit = new_opacity_logit.clone()
        child_rgb_logit = base_rgb_logit[idx].clone()
        child_latent = base_latent[idx].clone()
        child_sh = None if base_sh is None else base_sh[idx].clone()

        self._replace_buffer("uv", torch.cat((base_uv, child_uv), dim=0))
        self._replace_param("depth_raw", torch.cat((base_depth_raw, child_depth_raw), dim=0))
        self._replace_param("xyz_offset", torch.cat((base_xyz_offset, child_xyz_offset), dim=0))
        self._replace_param("quat_raw", torch.cat((base_quat_raw, child_quat_raw), dim=0))
        self._replace_param("log_scale", torch.cat((base_log_scale, child_log_scale), dim=0))
        self._replace_param("opacity_logit", torch.cat((base_opacity_logit, child_opacity_logit), dim=0))
        self._replace_param("rgb_logit", torch.cat((base_rgb_logit, child_rgb_logit), dim=0))
        self._replace_param("latent", torch.cat((base_latent, child_latent), dim=0))
        if base_sh is not None and child_sh is not None:
            self._replace_param("sh_coeffs", torch.cat((base_sh, child_sh), dim=0))

    def gaussian_params(
        self,
        intrinsics: Tensor,
        R_cw: Tensor | None = None,
        t_cw: Tensor | None = None,
    ) -> dict[str, Tensor | None]:
        fx, fy, cx, cy = intrinsics.unbind(dim=0)

        depth = F.softplus(self.depth_raw) + self.field_cfg.min_depth
        u = self.uv[:, 0:1]
        v = self.uv[:, 1:2]
        x = (u - cx) / fx
        y = (v - cy) / fy
        ray = torch.cat((x, y, torch.ones_like(x)), dim=-1)

        means3d = ray * depth + self.xyz_offset
        quat = normalize_quaternion(self.quat_raw)
        scale = torch.exp(self.log_scale)
        opacity = torch.sigmoid(self.opacity_logit[:, 0])
        rgb = apply_view_dependent_rgb(self.rgb_logit, self.sh_coeffs, means3d, R_cw, t_cw, self.appearance_cfg)
        depth_values = depth[:, 0]
        depth_map = depth.view(self.gh, self.gw) if depth.shape[0] == self.gh * self.gw else None
        return {
            "means3d": means3d,
            "quat": quat,
            "scale": scale,
            "opacity": opacity,
            "rgb": rgb,
            "latent": self.latent,
            "depth_values": depth_values,
            "depth_map": depth_map,
        }

    def forward(
        self,
        intrinsics: Tensor,
        R_cw: Tensor | None = None,
        t_cw: Tensor | None = None,
    ) -> dict[str, Tensor | None]:
        return self.gaussian_params(intrinsics, R_cw=R_cw, t_cw=t_cw)


__all__ = [
    "ScaleAwareResidualHead",
    "CanonicalGaussianField",
]
