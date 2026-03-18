import math
import warnings
from collections.abc import Iterable

import torch
import torch.nn as nn
from torch import Tensor

from .appearance import apply_view_dependent_rgb, num_sh_bases
from .fixed_capacity import append_rows_in_place, available_capacity, compact_rows_in_place, resolve_capacity
from .math_utils import inverse_sigmoid, normalize_quaternion, quaternion_to_matrix
from .posefree_config import AppearanceConfig, FieldConfig

_INT_BUFFER_KEYS = frozenset({"seed_id", "protect_until_step"})


class CanonicalGaussianField(nn.Module):
    def __init__(
        self,
        anchor_rgb: Tensor,
        focal: Tensor,
        principal: Tensor,
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

        fx, fy = focal
        cx, cy = principal
        safe_fx = fx.clamp_min(1.0)
        safe_fy = fy.clamp_min(1.0)
        x = (uv_flat[:, 0:1] - cx) / safe_fx
        y = (uv_flat[:, 1:2] - cy) / safe_fy
        ray = torch.cat((x, y, torch.ones_like(x)), dim=-1)
        means3d = ray * init_depth

        init_scale_xy = field_cfg.init_scale_xy * field_cfg.init_depth
        sx = init_scale_xy * s / safe_fx
        sy = init_scale_xy * s / safe_fy
        sz = anchor_rgb.new_tensor(field_cfg.init_scale_z * field_cfg.init_depth)
        init_scale = torch.stack((sx, sy, sz)).expand(initial_count, 3)
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

    # -- Capacity helpers --

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

    def _storage_rows(self) -> dict[str, Tensor]:
        """Return a dict mapping each storage field name to its backing tensor."""
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
        return rows

    def _snapshot_rows(self, idx: Tensor) -> dict[str, Tensor]:
        """Detach and clone all storage rows at the given indices."""
        return {k: v[idx].detach().clone() for k, v in self._storage_rows().items()}

    def _set_active_count(self, value: int) -> None:
        self.active_count.fill_(int(value))

    @property
    def num_gaussians(self) -> int:
        return int(self.active_count.item())

    @property
    def gaussian_capacity(self) -> int:
        return self.means3d.shape[0]

    # -- Optimizer support --

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

    # -- Mutation primitives --

    def _warn_overflow(self, op_name: str, dropped: int) -> None:
        if dropped <= 0:
            return
        warnings.warn(
            f"{op_name} dropped {dropped} gaussian(s) because fixed capacity {self.gaussian_capacity} is full",
            RuntimeWarning,
            stacklevel=2,
        )

    def _append_rows(self, new_rows: dict[str, Tensor]) -> int:
        """Append rows to storage. Detaches all values. Fills missing buffer keys with defaults."""
        first = next(iter(new_rows.values()), None)
        if first is None or first.numel() == 0:
            return 0
        count = first.shape[0]
        device = self.means3d.device
        dtype = self.means3d.dtype
        new_rows.setdefault("uv", torch.zeros(count, 2, device=device, dtype=dtype))
        new_rows.setdefault("seed_id", torch.full((count,), -1, device=device, dtype=torch.long))
        new_rows.setdefault("protect_until_step", torch.full((count,), -1, device=device, dtype=torch.long))
        if self.sh_coeffs is not None:
            new_rows.setdefault(
                "sh_coeffs",
                torch.zeros(count, self.sh_coeffs.shape[1], self.sh_coeffs.shape[2], device=device, dtype=dtype),
            )
        detached = {
            k: v.detach().to(dtype=torch.long) if k in _INT_BUFFER_KEYS else v.detach() for k, v in new_rows.items()
        }
        with torch.no_grad():
            result = append_rows_in_place(
                self._storage_rows(),
                active_count=self.num_gaussians,
                new_rows=detached,
                overflow_policy=self.field_cfg.overflow_policy,
            )
            self._set_active_count(result.new_active_count)
        self._warn_overflow("_append_rows", result.dropped)
        return result.appended

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
        count = means3d.shape[0]
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
        new_rows: dict[str, Tensor] = {
            "means3d": means3d,
            "quat_raw": quat_raw,
            "log_scale": torch.log(scale.clamp_min(1.0e-8)),
            "opacity_logit": inverse_sigmoid(opacity),
            "rgb_logit": inverse_sigmoid(rgb),
            "latent": latent,
        }
        if uv is not None:
            new_rows["uv"] = uv
        if seed_id is not None:
            new_rows["seed_id"] = seed_id
        if protect_until_step is not None:
            new_rows["protect_until_step"] = protect_until_step
        if sh_coeffs is not None:
            new_rows["sh_coeffs"] = sh_coeffs
        return self._append_rows(new_rows)

    def prune_keep_mask(self, keep_mask: Tensor) -> None:
        n = self.num_gaussians
        if keep_mask.ndim != 1 or keep_mask.shape[0] != n:
            raise ValueError("keep_mask must be [N]")
        if keep_mask.dtype != torch.bool:
            keep_mask = keep_mask.to(dtype=torch.bool)
        if int(keep_mask.sum()) <= 0:
            raise ValueError("prune_keep_mask would remove all gaussians")
        with torch.no_grad():
            kept = compact_rows_in_place(
                self._storage_rows(), active_count=n, keep_mask=keep_mask.to(device=self.means3d.device)
            )
            self._set_active_count(kept)

    def _check_indices(self, indices: Tensor, op_name: str) -> Tensor:
        """Validate and truncate indices for clone/split, handling overflow policy."""
        idx = indices.to(device=self.means3d.device, dtype=torch.long).view(-1)
        n = self.num_gaussians
        if idx.max() >= n or idx.min() < 0:
            raise ValueError(f"{op_name} indices out of range")
        free = available_capacity(self.gaussian_capacity, n)
        if idx.numel() > free:
            if self.field_cfg.overflow_policy == "abort":
                raise RuntimeError(f"{op_name} exceeded fixed-capacity storage")
            self._warn_overflow(op_name, idx.numel() - free)
            idx = idx[:free]
        return idx

    def clone_gaussians(self, indices: Tensor, jitter_scale: float = 0.25) -> int:
        if indices.numel() == 0:
            return 0
        idx = self._check_indices(indices, "clone_gaussians")
        if idx.numel() == 0:
            return 0

        with torch.no_grad():
            src_opacity = (torch.sigmoid(self.opacity_logit[idx]) * 0.5).clamp(1.0e-4, 1.0 - 1.0e-4)
            self.opacity_logit[idx] = inverse_sigmoid(src_opacity)

            cloned = self._snapshot_rows(idx)
            cloned["seed_id"] = torch.full((idx.numel(),), -1, device=idx.device, dtype=torch.long)
            cloned["opacity_logit"] = inverse_sigmoid(src_opacity)

            if jitter_scale > 0.0:
                scale = torch.exp(cloned["log_scale"])
                cloned["means3d"] = cloned["means3d"] + torch.randn_like(cloned["means3d"]) * scale * jitter_scale

            return self._append_rows(cloned)

    def split_gaussians(self, indices: Tensor, shrink_factor: float = 0.8, offset_scale: float = 0.75) -> int:
        if indices.numel() == 0:
            return 0
        idx = self._check_indices(indices, "split_gaussians")
        if idx.numel() == 0:
            return 0

        with torch.no_grad():
            child = self._snapshot_rows(idx)
            child["seed_id"] = torch.full((idx.numel(),), -1, device=idx.device, dtype=torch.long)

            rot = quaternion_to_matrix(normalize_quaternion(child["quat_raw"]))
            scales = torch.exp(child["log_scale"])
            axis_idx = scales.argmax(dim=1)
            batch = torch.arange(idx.shape[0], device=idx.device)
            axes = rot[batch, :, axis_idx]
            max_scale = scales.gather(1, axis_idx[:, None]).squeeze(1)
            delta = axes * (max_scale * offset_scale).unsqueeze(1)

            shrink_log = math.log(max(shrink_factor, 1.0e-4))
            new_log_scale = child["log_scale"] + shrink_log
            new_opacity_logit = inverse_sigmoid(
                (torch.sigmoid(self.opacity_logit[idx]) * 0.5).clamp(1.0e-4, 1.0 - 1.0e-4)
            )

            # Update parent in-place
            self.means3d[idx] = child["means3d"] - delta
            self.log_scale[idx] = new_log_scale
            self.opacity_logit[idx] = new_opacity_logit

            # Setup child
            child["means3d"] = child["means3d"] + delta
            child["log_scale"] = new_log_scale
            child["opacity_logit"] = new_opacity_logit

            return self._append_rows(child)

    # -- Regularization --

    def seed_depth_tv(self) -> Tensor:
        n = self.num_gaussians
        if n <= 0:
            return self.means3d.new_tensor(0.0)
        keep = self.seed_id[:n] >= 0
        if int(keep.sum()) <= 1:
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
        return self.protect_until_step[:n] >= step

    def enforce_protection(self, step: int, min_opacity: float) -> None:
        n = self.num_gaussians
        if n <= 0 or min_opacity <= 0.0:
            return
        mask = self.protected_mask(step)
        if not mask.any():
            return
        opacity_floor = min(max(min_opacity, 1.0e-4), 1.0 - 1.0e-4)
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

    # -- Parameter extraction --

    def gaussian_params(
        self,
        R_cw: Tensor | None = None,
        t_cw: Tensor | None = None,
        padded: bool = False,
    ) -> dict[str, Tensor | None]:
        n = self.active_count  # keep as tensor to avoid graph break from .item()
        s = slice(None) if padded else slice(None, n)
        means3d = self.means3d[s]
        quat = normalize_quaternion(self.quat_raw[s])
        scale = torch.exp(self.log_scale[s])
        opacity = torch.sigmoid(self.opacity_logit[s, 0])
        sh_coeffs = None if self.sh_coeffs is None else self.sh_coeffs[s]
        rgb = apply_view_dependent_rgb(self.rgb_logit[s], sh_coeffs, means3d, R_cw, t_cw, self.appearance_cfg)
        latent = self.latent[s]
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
        R_cw: Tensor | None = None,
        t_cw: Tensor | None = None,
        padded: bool = False,
    ) -> dict[str, Tensor | None]:
        return self.gaussian_params(R_cw=R_cw, t_cw=t_cw, padded=padded)


__all__ = ["CanonicalGaussianField"]
