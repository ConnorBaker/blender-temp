import math
from collections.abc import Iterable

import torch
import torch.nn as nn
from torch import Tensor

from .appearance import apply_view_dependent_rgb, num_sh_bases
from .image_utils import pixel_grid
from .math_utils import inverse_sigmoid, normalize_quaternion, quaternion_to_matrix
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

        residual_flat = torch.empty((inp.shape[0], 3), device=inp.device, dtype=inp.dtype)
        for start in range(0, inp.shape[0], chunk):
            end = min(start + chunk, int(inp.shape[0]))
            residual_flat[start:end] = self.net(inp[start:end])
        residual = residual_flat.view(h, w, 3).permute(2, 0, 1)
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

        fx, fy, cx, cy = intrinsics
        x = (uv_flat[:, 0:1] - cx) / fx
        y = (uv_flat[:, 1:2] - cy) / fy
        ray = torch.cat((x, y, torch.ones_like(x)), dim=-1)
        means3d = ray * init_depth

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
        self.register_buffer("seed_id", torch.arange(gh * gw, device=device, dtype=torch.long))
        self.register_buffer("protect_until_step", torch.full((gh * gw,), -1, device=device, dtype=torch.long))
        self.means3d = nn.Parameter(means3d)
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
        return int(self.means3d.shape[0])

    def parameters_for_optimizer(self) -> Iterable[nn.Parameter]:
        yield self.means3d
        yield self.quat_raw
        yield self.log_scale
        yield self.opacity_logit
        yield self.rgb_logit
        yield self.latent
        if self.sh_coeffs is not None:
            yield self.sh_coeffs

    def optimizer_param_dict(self) -> dict[str, nn.Parameter]:
        params: dict[str, nn.Parameter] = {
            "means3d": self.means3d,
            "quat_raw": self.quat_raw,
            "log_scale": self.log_scale,
            "opacity_logit": self.opacity_logit,
            "rgb_logit": self.rgb_logit,
            "latent": self.latent,
        }
        if self.sh_coeffs is not None:
            params["sh_coeffs"] = self.sh_coeffs
        return params

    def _replace_param(self, name: str, value: Tensor | None) -> None:
        if value is None:
            self.register_parameter(name, None)
        else:
            setattr(self, name, nn.Parameter(value.contiguous()))

    def _replace_buffer(self, name: str, value: Tensor) -> None:
        self._buffers[name] = value.contiguous()

    def _append_gaussians(
        self,
        *,
        means3d: Tensor,
        quat_raw: Tensor,
        log_scale: Tensor,
        opacity_logit: Tensor,
        rgb_logit: Tensor,
        latent: Tensor,
        uv: Tensor | None = None,
        seed_id: Tensor | None = None,
        protect_until_step: Tensor | None = None,
        sh_coeffs: Tensor | None = None,
    ) -> None:
        if means3d.numel() == 0:
            return
        if uv is None:
            uv = torch.zeros(means3d.shape[0], 2, device=means3d.device, dtype=means3d.dtype)
        if seed_id is None:
            seed_id = torch.full((means3d.shape[0],), -1, device=means3d.device, dtype=torch.long)
        if protect_until_step is None:
            protect_until_step = torch.full((means3d.shape[0],), -1, device=means3d.device, dtype=torch.long)

        self._replace_buffer("uv", torch.cat((self.uv.detach(), uv.detach()), dim=0))
        self._replace_buffer(
            "seed_id", torch.cat((self.seed_id.detach(), seed_id.detach().to(dtype=torch.long)), dim=0)
        )
        self._replace_buffer(
            "protect_until_step",
            torch.cat((self.protect_until_step.detach(), protect_until_step.detach().to(dtype=torch.long)), dim=0),
        )
        self._replace_param("means3d", torch.cat((self.means3d.detach(), means3d.detach()), dim=0))
        self._replace_param("quat_raw", torch.cat((self.quat_raw.detach(), quat_raw.detach()), dim=0))
        self._replace_param("log_scale", torch.cat((self.log_scale.detach(), log_scale.detach()), dim=0))
        self._replace_param("opacity_logit", torch.cat((self.opacity_logit.detach(), opacity_logit.detach()), dim=0))
        self._replace_param("rgb_logit", torch.cat((self.rgb_logit.detach(), rgb_logit.detach()), dim=0))
        self._replace_param("latent", torch.cat((self.latent.detach(), latent.detach()), dim=0))
        if self.sh_coeffs is not None:
            if sh_coeffs is None:
                sh_coeffs = torch.zeros(
                    means3d.shape[0],
                    self.sh_coeffs.shape[1],
                    self.sh_coeffs.shape[2],
                    device=means3d.device,
                    dtype=means3d.dtype,
                )
            self._replace_param("sh_coeffs", torch.cat((self.sh_coeffs.detach(), sh_coeffs.detach()), dim=0))

    def append_gaussians(
        self,
        *,
        means3d: Tensor,
        rgb: Tensor,
        uv: Tensor | None = None,
        scale: Tensor | None = None,
        quat_raw: Tensor | None = None,
        opacity: Tensor | None = None,
        latent: Tensor | None = None,
        sh_coeffs: Tensor | None = None,
        seed_id: Tensor | None = None,
        protect_until_step: Tensor | None = None,
    ) -> None:
        count = int(means3d.shape[0])
        if count <= 0:
            return
        device = means3d.device
        dtype = means3d.dtype
        if scale is None:
            scale = means3d.new_tensor([
                self.field_cfg.init_scale_xy,
                self.field_cfg.init_scale_xy,
                self.field_cfg.init_scale_z,
            ]).expand(count, 3)
        if quat_raw is None:
            quat_raw = torch.zeros(count, 4, device=device, dtype=dtype)
            quat_raw[:, 0] = 1.0
        if opacity is None:
            opacity = torch.full((count, 1), self.field_cfg.init_opacity, device=device, dtype=dtype)
        elif opacity.ndim == 1:
            opacity = opacity[:, None]
        if latent is None:
            latent = torch.zeros(count, self.feature_dim, device=device, dtype=dtype)
        opacity = opacity.clamp(1.0e-4, 1.0 - 1.0e-4)
        rgb = rgb.clamp(1.0e-4, 1.0 - 1.0e-4)
        self._append_gaussians(
            means3d=means3d,
            quat_raw=quat_raw,
            log_scale=torch.log(scale.clamp_min(1.0e-8)),
            opacity_logit=inverse_sigmoid(opacity),
            rgb_logit=inverse_sigmoid(rgb),
            latent=latent,
            uv=uv,
            seed_id=seed_id,
            protect_until_step=protect_until_step,
            sh_coeffs=sh_coeffs,
        )

    def prune_keep_mask(self, keep_mask: Tensor) -> None:
        if keep_mask.ndim != 1 or keep_mask.shape[0] != self.num_gaussians:
            raise ValueError("keep_mask must be [N]")
        if keep_mask.dtype != torch.bool:
            keep_mask = keep_mask.to(dtype=torch.bool)
        if int(keep_mask.sum().item()) <= 0:
            raise ValueError("prune_keep_mask would remove all gaussians")

        self._replace_buffer("uv", self.uv.detach()[keep_mask])
        self._replace_buffer("seed_id", self.seed_id.detach()[keep_mask])
        self._replace_buffer("protect_until_step", self.protect_until_step.detach()[keep_mask])
        self._replace_param("means3d", self.means3d.detach()[keep_mask])
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
        idx = indices.to(device=self.means3d.device, dtype=torch.long).view(-1)
        if idx.max().item() >= self.num_gaussians or idx.min().item() < 0:
            raise ValueError("clone_gaussians indices out of range")

        base_means3d = self.means3d.detach()
        base_quat_raw = self.quat_raw.detach()
        base_log_scale = self.log_scale.detach()
        base_opacity_logit = self.opacity_logit.detach().clone()
        base_rgb_logit = self.rgb_logit.detach()
        base_latent = self.latent.detach()
        base_uv = self.uv.detach()
        base_protect_until_step = self.protect_until_step.detach()
        base_sh = None if self.sh_coeffs is None else self.sh_coeffs.detach()

        src_opacity = torch.sigmoid(base_opacity_logit[idx]) * 0.5
        src_opacity = src_opacity.clamp(1.0e-4, 1.0 - 1.0e-4)
        base_opacity_logit[idx] = inverse_sigmoid(src_opacity)
        clone_opacity_logit = inverse_sigmoid(src_opacity)

        clone_means3d = base_means3d[idx].clone()
        clone_quat_raw = base_quat_raw[idx].clone()
        clone_log_scale = base_log_scale[idx].clone()
        clone_rgb_logit = base_rgb_logit[idx].clone()
        clone_latent = base_latent[idx].clone()
        clone_uv = base_uv[idx].clone()
        clone_protect_until_step = base_protect_until_step[idx].clone()
        clone_sh = None if base_sh is None else base_sh[idx].clone()

        if jitter_scale > 0.0:
            scale = torch.exp(clone_log_scale)
            clone_means3d = clone_means3d + torch.randn_like(clone_means3d) * scale * float(jitter_scale)

        self._replace_param("means3d", torch.cat((base_means3d, clone_means3d), dim=0))
        self._replace_param("quat_raw", torch.cat((base_quat_raw, clone_quat_raw), dim=0))
        self._replace_param("log_scale", torch.cat((base_log_scale, clone_log_scale), dim=0))
        self._replace_param("opacity_logit", torch.cat((base_opacity_logit, clone_opacity_logit), dim=0))
        self._replace_param("rgb_logit", torch.cat((base_rgb_logit, clone_rgb_logit), dim=0))
        self._replace_param("latent", torch.cat((base_latent, clone_latent), dim=0))
        self._replace_buffer("uv", torch.cat((base_uv, clone_uv), dim=0))
        self._replace_buffer(
            "protect_until_step",
            torch.cat((base_protect_until_step, clone_protect_until_step), dim=0),
        )
        self._replace_buffer(
            "seed_id",
            torch.cat(
                (self.seed_id.detach(), torch.full((idx.shape[0],), -1, device=idx.device, dtype=torch.long)), dim=0
            ),
        )
        if base_sh is not None and clone_sh is not None:
            self._replace_param("sh_coeffs", torch.cat((base_sh, clone_sh), dim=0))

    def split_gaussians(self, indices: Tensor, shrink_factor: float = 0.8, offset_scale: float = 0.75) -> None:
        if indices.numel() == 0:
            return
        idx = indices.to(device=self.means3d.device, dtype=torch.long).view(-1)
        if idx.max().item() >= self.num_gaussians or idx.min().item() < 0:
            raise ValueError("split_gaussians indices out of range")

        base_means3d = self.means3d.detach().clone()
        base_quat_raw = self.quat_raw.detach()
        base_log_scale = self.log_scale.detach().clone()
        base_opacity_logit = self.opacity_logit.detach().clone()
        base_rgb_logit = self.rgb_logit.detach()
        base_latent = self.latent.detach()
        base_uv = self.uv.detach()
        base_protect_until_step = self.protect_until_step.detach()
        base_sh = None if self.sh_coeffs is None else self.sh_coeffs.detach()

        orig_means3d = base_means3d[idx].clone()
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

        base_means3d[idx] = orig_means3d - delta
        base_log_scale[idx] = new_log_scale
        base_opacity_logit[idx] = new_opacity_logit

        child_means3d = orig_means3d + delta
        child_quat_raw = base_quat_raw[idx].clone()
        child_log_scale = new_log_scale.clone()
        child_opacity_logit = new_opacity_logit.clone()
        child_rgb_logit = base_rgb_logit[idx].clone()
        child_latent = base_latent[idx].clone()
        child_uv = base_uv[idx].clone()
        child_protect_until_step = base_protect_until_step[idx].clone()
        child_sh = None if base_sh is None else base_sh[idx].clone()

        self._replace_param("means3d", torch.cat((base_means3d, child_means3d), dim=0))
        self._replace_param("quat_raw", torch.cat((base_quat_raw, child_quat_raw), dim=0))
        self._replace_param("log_scale", torch.cat((base_log_scale, child_log_scale), dim=0))
        self._replace_param("opacity_logit", torch.cat((base_opacity_logit, child_opacity_logit), dim=0))
        self._replace_param("rgb_logit", torch.cat((base_rgb_logit, child_rgb_logit), dim=0))
        self._replace_param("latent", torch.cat((base_latent, child_latent), dim=0))
        self._replace_buffer("uv", torch.cat((base_uv, child_uv), dim=0))
        self._replace_buffer(
            "protect_until_step",
            torch.cat((base_protect_until_step, child_protect_until_step), dim=0),
        )
        self._replace_buffer(
            "seed_id",
            torch.cat(
                (self.seed_id.detach(), torch.full((idx.shape[0],), -1, device=idx.device, dtype=torch.long)), dim=0
            ),
        )
        if base_sh is not None and child_sh is not None:
            self._replace_param("sh_coeffs", torch.cat((base_sh, child_sh), dim=0))

    def seed_depth_tv(self) -> Tensor:
        if self.seed_id.numel() == 0:
            return self.means3d.new_tensor(0.0)
        keep = self.seed_id >= 0
        if int(keep.sum().item()) <= 1:
            return self.means3d.new_tensor(0.0)

        sid = self.seed_id[keep]
        depth = self.means3d[keep, 2]
        grid_depth = torch.zeros(self.gh * self.gw, device=depth.device, dtype=depth.dtype)
        grid_valid = torch.zeros(self.gh * self.gw, device=depth.device, dtype=torch.bool)
        grid_depth[sid] = depth
        grid_valid[sid] = True
        grid_depth = grid_depth.view(self.gh, self.gw)
        grid_valid = grid_valid.view(self.gh, self.gw)

        total = self.means3d.new_tensor(0.0)
        count = 0
        if self.gw > 1:
            mask_x = grid_valid[:, 1:] & grid_valid[:, :-1]
            if mask_x.any():
                total = total + (grid_depth[:, 1:] - grid_depth[:, :-1]).abs()[mask_x].mean()
                count += 1
        if self.gh > 1:
            mask_y = grid_valid[1:, :] & grid_valid[:-1, :]
            if mask_y.any():
                total = total + (grid_depth[1:, :] - grid_depth[:-1, :]).abs()[mask_y].mean()
                count += 1
        if count == 0:
            return self.means3d.new_tensor(0.0)
        return total / float(count)

    def protected_mask(self, step: int) -> Tensor:
        if self.protect_until_step.numel() == 0:
            return torch.zeros(self.num_gaussians, device=self.means3d.device, dtype=torch.bool)
        return self.protect_until_step >= int(step)

    def enforce_protection(self, step: int, min_opacity: float) -> None:
        if self.num_gaussians <= 0 or min_opacity <= 0.0:
            return
        mask = self.protected_mask(step)
        if not mask.any():
            return
        opacity_floor = float(min(max(min_opacity, 1.0e-4), 1.0 - 1.0e-4))
        floor_logit = inverse_sigmoid(self.opacity_logit.new_tensor(opacity_floor))
        with torch.no_grad():
            self.opacity_logit[mask] = torch.maximum(self.opacity_logit[mask], floor_logit)

    def enforce_scale_floor(self, min_log_scale: float | None = None) -> None:
        if self.num_gaussians <= 0:
            return
        floor = self.field_cfg.min_log_scale if min_log_scale is None else float(min_log_scale)
        with torch.no_grad():
            self.log_scale.clamp_(min=floor)

    def gaussian_params(
        self,
        intrinsics: Tensor,
        R_cw: Tensor | None = None,
        t_cw: Tensor | None = None,
    ) -> dict[str, Tensor | None]:
        means3d = self.means3d
        quat = normalize_quaternion(self.quat_raw)
        scale = torch.exp(self.log_scale)
        opacity = torch.sigmoid(self.opacity_logit[:, 0])
        rgb = apply_view_dependent_rgb(self.rgb_logit, self.sh_coeffs, means3d, R_cw, t_cw, self.appearance_cfg)
        depth_values = means3d[:, 2]
        return {
            "means3d": means3d,
            "quat": quat,
            "scale": scale,
            "opacity": opacity,
            "rgb": rgb,
            "latent": self.latent,
            "depth_values": depth_values,
            "depth_map": None,
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
