import math
import warnings
from collections.abc import Iterable

import torch
import torch.nn as nn
from torch import Tensor

from .appearance import apply_view_dependent_rgb, num_sh_bases
from .fixed_capacity import append_rows_in_place, available_capacity, compact_rows_in_place, resolve_capacity
from .image_utils import pixel_grid
from .math_utils import inverse_sigmoid, normalize_quaternion, quaternion_to_matrix
from .posefree_config import AppearanceConfig, FieldConfig


class ScaleAwareResidualHead(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 64, residual_scale: float = 0.05):
        super().__init__()
        self.residual_scale = residual_scale
        self.net = nn.Sequential(
            nn.Conv2d(feature_dim + 4, hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, 3, kernel_size=1),
        )

    def forward(
        self,
        latent_map: Tensor,
        scale_x: float,
        scale_y: float,
        chunk: int = 262144,
    ) -> Tensor:
        del chunk
        f_dim, h, w = latent_map.shape
        coords = pixel_grid(h, w, latent_map.device, latent_map.dtype, normalized=True).permute(2, 0, 1).contiguous()
        scale_token = latent_map.new_tensor([math.log(scale_x), math.log(scale_y)]).view(2, 1, 1).expand(2, h, w)
        inp = torch.cat((latent_map.view(f_dim, h, w), coords, scale_token), dim=0).unsqueeze(0)
        residual = self.net(inp).squeeze(0)
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
        initial_count = int(gh * gw)
        capacity = resolve_capacity(initial_count, int(field_cfg.gaussian_capacity))

        anchor_colors = anchor_rgb[:, ::s, ::s].permute(1, 2, 0).contiguous().view(-1, 3)
        init_depth = torch.full((initial_count, 1), field_cfg.init_depth, device=device, dtype=dtype)

        fx, fy, cx, cy = intrinsics
        x = (uv_flat[:, 0:1] - cx) / fx
        y = (uv_flat[:, 1:2] - cy) / fy
        ray = torch.cat((x, y, torch.ones_like(x)), dim=-1)
        means3d = ray * init_depth

        sx = field_cfg.init_scale_xy * field_cfg.init_depth / max(float(fx.item()), 1.0) * s
        sy = field_cfg.init_scale_xy * field_cfg.init_depth / max(float(fy.item()), 1.0) * s
        sz = field_cfg.init_scale_z * field_cfg.init_depth
        init_scale = torch.tensor([sx, sy, sz], device=device, dtype=dtype).expand(initial_count, 3)
        log_scale = torch.log(init_scale)

        quat = torch.zeros(initial_count, 4, device=device, dtype=dtype)
        quat[:, 0] = 1.0

        self.gh = gh
        self.gw = gw
        self.feature_dim = field_cfg.feature_dim
        self.field_cfg = field_cfg
        self.appearance_cfg = appearance_cfg

        self.register_buffer("active_count", torch.tensor(initial_count, device=device, dtype=torch.long))
        self.register_buffer("uv", self._capacity_buffer(uv_flat, capacity))
        self.register_buffer(
            "seed_id",
            self._capacity_buffer(
                torch.arange(initial_count, device=device, dtype=torch.long),
                capacity,
                fill_value=-1,
                dtype=torch.long,
            ),
        )
        self.register_buffer(
            "protect_until_step",
            self._capacity_buffer(
                torch.full((initial_count,), -1, device=device, dtype=torch.long),
                capacity,
                fill_value=-1,
                dtype=torch.long,
            ),
        )
        self.means3d = nn.Parameter(self._capacity_buffer(means3d, capacity))
        self.quat_raw = nn.Parameter(self._capacity_buffer(quat, capacity))
        self.log_scale = nn.Parameter(self._capacity_buffer(log_scale, capacity))
        self.opacity_logit = nn.Parameter(
            self._capacity_buffer(
                torch.full(
                    (initial_count, 1),
                    inverse_sigmoid(torch.tensor(field_cfg.init_opacity, device=device, dtype=dtype)),
                    device=device,
                    dtype=dtype,
                ),
                capacity,
            )
        )
        self.rgb_logit = nn.Parameter(self._capacity_buffer(inverse_sigmoid(anchor_colors), capacity))
        self.latent = nn.Parameter(torch.zeros(capacity, field_cfg.feature_dim, device=device, dtype=dtype))
        if appearance_cfg.mode == "sh" and appearance_cfg.sh_degree > 0:
            basis_count = num_sh_bases(appearance_cfg.sh_degree) - 1
            self.sh_coeffs = nn.Parameter(torch.zeros(capacity, 3, basis_count, device=device, dtype=dtype))
        else:
            self.register_parameter("sh_coeffs", None)

    def _capacity_buffer(
        self,
        value: Tensor,
        capacity: int,
        *,
        fill_value: float | int = 0.0,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        out_dtype = value.dtype if dtype is None else dtype
        storage = torch.full(
            (capacity, *value.shape[1:]),
            fill_value,
            device=value.device,
            dtype=out_dtype,
        )
        if value.numel() > 0:
            storage[: value.shape[0]].copy_(value.to(dtype=out_dtype))
        return storage.contiguous()

    def _active_rows(self, tensor: Tensor) -> Tensor:
        return tensor[: self.num_gaussians]

    def _set_active_count(self, value: int) -> None:
        self.active_count.fill_(int(value))

    @property
    def num_gaussians(self) -> int:
        return int(self.active_count.item())

    @property
    def gaussian_capacity(self) -> int:
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

    def _warn_overflow(self, op_name: str, dropped: int) -> None:
        if dropped <= 0:
            return
        warnings.warn(
            f"{op_name} dropped {dropped} gaussian(s) because fixed capacity {self.gaussian_capacity} is full",
            RuntimeWarning,
            stacklevel=2,
        )

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
    ) -> int:
        if means3d.numel() == 0:
            return 0
        count = int(means3d.shape[0])
        if uv is None:
            uv = torch.zeros(count, 2, device=means3d.device, dtype=means3d.dtype)
        if seed_id is None:
            seed_id = torch.full((count,), -1, device=means3d.device, dtype=torch.long)
        if protect_until_step is None:
            protect_until_step = torch.full((count,), -1, device=means3d.device, dtype=torch.long)

        rows: dict[str, Tensor] = {
            "uv": self.uv,
            "seed_id": self.seed_id,
            "protect_until_step": self.protect_until_step,
            "means3d": self.means3d,
            "quat_raw": self.quat_raw,
            "log_scale": self.log_scale,
            "opacity_logit": self.opacity_logit,
            "rgb_logit": self.rgb_logit,
            "latent": self.latent,
        }
        new_rows: dict[str, Tensor] = {
            "uv": uv.detach(),
            "seed_id": seed_id.detach().to(dtype=torch.long),
            "protect_until_step": protect_until_step.detach().to(dtype=torch.long),
            "means3d": means3d.detach(),
            "quat_raw": quat_raw.detach(),
            "log_scale": log_scale.detach(),
            "opacity_logit": opacity_logit.detach(),
            "rgb_logit": rgb_logit.detach(),
            "latent": latent.detach(),
        }
        if self.sh_coeffs is not None:
            if sh_coeffs is None:
                sh_coeffs = torch.zeros(
                    count,
                    self.sh_coeffs.shape[1],
                    self.sh_coeffs.shape[2],
                    device=means3d.device,
                    dtype=means3d.dtype,
                )
            rows["sh_coeffs"] = self.sh_coeffs
            new_rows["sh_coeffs"] = sh_coeffs.detach()

        with torch.no_grad():
            result = append_rows_in_place(
                rows,
                active_count=self.num_gaussians,
                new_rows=new_rows,
                overflow_policy=self.field_cfg.overflow_policy,
            )
            self._set_active_count(result.new_active_count)
        self._warn_overflow("append_gaussians", int(result.dropped))
        return int(result.appended)

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
    ) -> int:
        count = int(means3d.shape[0])
        if count <= 0:
            return 0
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
        return self._append_gaussians(
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
        n = self.num_gaussians
        if keep_mask.ndim != 1 or keep_mask.shape[0] != n:
            raise ValueError("keep_mask must be [N]")
        if keep_mask.dtype != torch.bool:
            keep_mask = keep_mask.to(dtype=torch.bool)
        if int(keep_mask.sum().item()) <= 0:
            raise ValueError("prune_keep_mask would remove all gaussians")

        rows: dict[str, Tensor] = {
            "uv": self.uv,
            "seed_id": self.seed_id,
            "protect_until_step": self.protect_until_step,
            "means3d": self.means3d,
            "quat_raw": self.quat_raw,
            "log_scale": self.log_scale,
            "opacity_logit": self.opacity_logit,
            "rgb_logit": self.rgb_logit,
            "latent": self.latent,
        }
        if self.sh_coeffs is not None:
            rows["sh_coeffs"] = self.sh_coeffs
        with torch.no_grad():
            kept = compact_rows_in_place(rows, active_count=n, keep_mask=keep_mask.to(device=self.means3d.device))
            self._set_active_count(kept)

    def clone_gaussians(self, indices: Tensor, jitter_scale: float = 0.25) -> int:
        if indices.numel() == 0:
            return 0
        idx = indices.to(device=self.means3d.device, dtype=torch.long).view(-1)
        n = self.num_gaussians
        if idx.max().item() >= n or idx.min().item() < 0:
            raise ValueError("clone_gaussians indices out of range")

        free = available_capacity(self.gaussian_capacity, n)
        if idx.numel() > free:
            if self.field_cfg.overflow_policy == "abort":
                raise RuntimeError("clone_gaussians exceeded fixed-capacity storage")
            self._warn_overflow("clone_gaussians", int(idx.numel() - free))
            idx = idx[:free]
        if idx.numel() == 0:
            return 0

        with torch.no_grad():
            src_opacity = torch.sigmoid(self.opacity_logit[idx]) * 0.5
            src_opacity = src_opacity.clamp(1.0e-4, 1.0 - 1.0e-4)
            self.opacity_logit[idx] = inverse_sigmoid(src_opacity)

            clone_means3d = self.means3d[idx].detach().clone()
            clone_quat_raw = self.quat_raw[idx].detach().clone()
            clone_log_scale = self.log_scale[idx].detach().clone()
            clone_rgb_logit = self.rgb_logit[idx].detach().clone()
            clone_latent = self.latent[idx].detach().clone()
            clone_uv = self.uv[idx].detach().clone()
            clone_protect_until_step = self.protect_until_step[idx].detach().clone()
            clone_sh = None if self.sh_coeffs is None else self.sh_coeffs[idx].detach().clone()

            if jitter_scale > 0.0:
                scale = torch.exp(clone_log_scale)
                clone_means3d = clone_means3d + torch.randn_like(clone_means3d) * scale * float(jitter_scale)

            return self._append_gaussians(
                means3d=clone_means3d,
                quat_raw=clone_quat_raw,
                log_scale=clone_log_scale,
                opacity_logit=inverse_sigmoid(src_opacity),
                rgb_logit=clone_rgb_logit,
                latent=clone_latent,
                uv=clone_uv,
                protect_until_step=clone_protect_until_step,
                sh_coeffs=clone_sh,
            )

    def split_gaussians(self, indices: Tensor, shrink_factor: float = 0.8, offset_scale: float = 0.75) -> int:
        if indices.numel() == 0:
            return 0
        idx = indices.to(device=self.means3d.device, dtype=torch.long).view(-1)
        n = self.num_gaussians
        if idx.max().item() >= n or idx.min().item() < 0:
            raise ValueError("split_gaussians indices out of range")

        free = available_capacity(self.gaussian_capacity, n)
        if idx.numel() > free:
            if self.field_cfg.overflow_policy == "abort":
                raise RuntimeError("split_gaussians exceeded fixed-capacity storage")
            self._warn_overflow("split_gaussians", int(idx.numel() - free))
            idx = idx[:free]
        if idx.numel() == 0:
            return 0

        with torch.no_grad():
            orig_means3d = self.means3d[idx].detach().clone()
            orig_log_scale = self.log_scale[idx].detach().clone()
            rot = quaternion_to_matrix(normalize_quaternion(self.quat_raw[idx].detach()))
            scales = torch.exp(orig_log_scale)
            axis_idx = scales.argmax(dim=1)
            batch = torch.arange(idx.shape[0], device=idx.device)
            axes = rot[batch, :, axis_idx]
            max_scale = scales.gather(1, axis_idx[:, None]).squeeze(1)
            delta = axes * (max_scale * float(offset_scale)).unsqueeze(1)

            shrink_log = math.log(max(float(shrink_factor), 1.0e-4))
            new_log_scale = orig_log_scale + shrink_log
            new_opacity = (torch.sigmoid(self.opacity_logit[idx]) * 0.5).clamp(1.0e-4, 1.0 - 1.0e-4)
            new_opacity_logit = inverse_sigmoid(new_opacity)

            self.means3d[idx] = orig_means3d - delta
            self.log_scale[idx] = new_log_scale
            self.opacity_logit[idx] = new_opacity_logit

            child_means3d = orig_means3d + delta
            child_quat_raw = self.quat_raw[idx].detach().clone()
            child_rgb_logit = self.rgb_logit[idx].detach().clone()
            child_latent = self.latent[idx].detach().clone()
            child_uv = self.uv[idx].detach().clone()
            child_protect_until_step = self.protect_until_step[idx].detach().clone()
            child_sh = None if self.sh_coeffs is None else self.sh_coeffs[idx].detach().clone()

            return self._append_gaussians(
                means3d=child_means3d,
                quat_raw=child_quat_raw,
                log_scale=new_log_scale,
                opacity_logit=new_opacity_logit,
                rgb_logit=child_rgb_logit,
                latent=child_latent,
                uv=child_uv,
                protect_until_step=child_protect_until_step,
                sh_coeffs=child_sh,
            )

    def seed_depth_tv(self) -> Tensor:
        n = self.num_gaussians
        if n <= 0:
            return self.means3d.new_tensor(0.0)
        keep = self.seed_id[:n] >= 0
        if int(keep.sum().item()) <= 1:
            return self.means3d.new_tensor(0.0)

        sid = self.seed_id[:n][keep]
        depth = self.means3d[:n][keep, 2]
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
        n = self.num_gaussians
        if n <= 0:
            return torch.zeros(0, device=self.means3d.device, dtype=torch.bool)
        return self.protect_until_step[:n] >= int(step)

    def enforce_protection(self, step: int, min_opacity: float) -> None:
        n = self.num_gaussians
        if n <= 0 or min_opacity <= 0.0:
            return
        mask = self.protected_mask(step)
        if not mask.any():
            return
        opacity_floor = float(min(max(min_opacity, 1.0e-4), 1.0 - 1.0e-4))
        floor_logit = inverse_sigmoid(self.opacity_logit.new_tensor(opacity_floor))
        idx = mask.nonzero(as_tuple=False).squeeze(-1)
        with torch.no_grad():
            self.opacity_logit[idx] = torch.maximum(self.opacity_logit[idx], floor_logit)

    def enforce_scale_floor(self, min_log_scale: float | None = None) -> None:
        n = self.num_gaussians
        if n <= 0:
            return
        floor = self.field_cfg.min_log_scale if min_log_scale is None else float(min_log_scale)
        with torch.no_grad():
            self.log_scale[:n].clamp_(min=floor)

    def gaussian_params(
        self,
        intrinsics: Tensor,
        R_cw: Tensor | None = None,
        t_cw: Tensor | None = None,
        padded: bool = False,
    ) -> dict[str, Tensor | None]:
        del intrinsics
        n = self.active_count  # keep as tensor to avoid graph break from .item()
        if padded:
            means3d = self.means3d
            quat = normalize_quaternion(self.quat_raw)
            scale = torch.exp(self.log_scale)
            opacity = torch.sigmoid(self.opacity_logit[:, 0])
            rgb = apply_view_dependent_rgb(self.rgb_logit, self.sh_coeffs, means3d, R_cw, t_cw, self.appearance_cfg)
            latent = self.latent
        else:
            means3d = self.means3d[:n]
            quat = normalize_quaternion(self.quat_raw[:n])
            scale = torch.exp(self.log_scale[:n])
            opacity = torch.sigmoid(self.opacity_logit[:n, 0])
            sh_coeffs = None if self.sh_coeffs is None else self.sh_coeffs[:n]
            rgb = apply_view_dependent_rgb(self.rgb_logit[:n], sh_coeffs, means3d, R_cw, t_cw, self.appearance_cfg)
            latent = self.latent[:n]
        return {
            "means3d": means3d,
            "quat": quat,
            "scale": scale,
            "opacity": opacity,
            "rgb": rgb,
            "latent": latent,
            "depth_values": means3d[:, 2],
            "depth_map": None,
            "active_count": n.clone(),
        }

    def forward(
        self,
        intrinsics: Tensor,
        R_cw: Tensor | None = None,
        t_cw: Tensor | None = None,
        padded: bool = False,
    ) -> dict[str, Tensor | None]:
        return self.gaussian_params(intrinsics, R_cw=R_cw, t_cw=t_cw, padded=padded)


__all__ = [
    "ScaleAwareResidualHead",
    "CanonicalGaussianField",
]
