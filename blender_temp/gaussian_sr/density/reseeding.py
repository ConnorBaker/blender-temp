import torch
from torch import Tensor

from ..field import CanonicalGaussianField
from ..posefree_config import DensityControlConfig
from .types import DensityViewObservation


def _select_reseed_pixels(
    flat_residual: Tensor,
    *,
    width: int,
    budget: int,
    residual_quantile: float,
    match_radius_px: float,
) -> Tensor:
    if budget <= 0 or flat_residual.numel() <= 0:
        return torch.empty(0, device=flat_residual.device, dtype=torch.long)

    if flat_residual.numel() == 1:
        return torch.zeros(1, device=flat_residual.device, dtype=torch.long)

    threshold = torch.quantile(flat_residual, flat_residual.new_tensor(residual_quantile))
    candidate_idx = (flat_residual >= threshold).nonzero(as_tuple=False).squeeze(-1)
    if candidate_idx.numel() == 0:
        candidate_idx = torch.topk(flat_residual, k=min(budget, flat_residual.numel()), largest=True).indices

    candidate_score = flat_residual[candidate_idx]
    pool = min(int(candidate_idx.numel()), max(int(budget), int(budget) * 8))
    if candidate_idx.numel() > pool:
        top = torch.topk(candidate_score, k=pool, largest=True).indices
        candidate_idx = candidate_idx[top]
        candidate_score = candidate_score[top]

    order = torch.argsort(candidate_score, descending=True)
    candidate_idx = candidate_idx[order]
    if match_radius_px <= 0.0 or candidate_idx.numel() <= budget:
        return candidate_idx[:budget].to(torch.long)

    px = (candidate_idx % width).tolist()
    py = (candidate_idx // width).tolist()
    radius2 = float(match_radius_px) * float(match_radius_px)
    keep_positions: list[int] = []
    for pos, (x, y) in enumerate(zip(px, py)):
        if all(((x - px[prev]) ** 2 + (y - py[prev]) ** 2) > radius2 for prev in keep_positions):
            keep_positions.append(pos)
            if len(keep_positions) >= budget:
                break
    if not keep_positions:
        return candidate_idx[:budget].to(torch.long)
    keep = torch.tensor(keep_positions, device=flat_residual.device, dtype=torch.long)
    return candidate_idx[keep].to(torch.long)


def _camera_depths(means3d: Tensor, R_cw: Tensor, t_cw: Tensor) -> Tensor:
    return means3d @ R_cw.transpose(0, 1)[:, 2] + t_cw[2]


def _reseed_for_observation(
    field_model: CanonicalGaussianField,
    observation: DensityViewObservation,
    cfg: DensityControlConfig,
    step: int,
) -> int:
    budget = int(cfg.weak_view_reseed_budget_per_view)
    if budget <= 0:
        return 0
    if (
        observation.residual_map is None
        or observation.target_rgb is None
        or observation.R_cw is None
        or observation.t_cw is None
        or observation.intrinsics is None
    ):
        return 0

    residual_map = observation.residual_map.detach()
    target_rgb = observation.target_rgb.detach()
    if residual_map.numel() <= 0 or target_rgb.numel() <= 0:
        return 0

    flat = residual_map.reshape(-1)
    pred_rgb = observation.pred_rgb.detach() if observation.pred_rgb is not None else None
    if (
        pred_rgb is not None
        and pred_rgb.shape == target_rgb.shape
        and float(cfg.weak_view_reseed_target_luma_min) >= 0.0
    ):
        target_luma = 0.2126 * target_rgb[0] + 0.7152 * target_rgb[1] + 0.0722 * target_rgb[2]
        pred_luma = 0.2126 * pred_rgb[0] + 0.7152 * pred_rgb[1] + 0.0722 * pred_rgb[2]
        positive_miss = (target_luma - pred_luma).clamp_min(0.0)
        if float(cfg.weak_view_reseed_target_luma_min) > 0.0:
            positive_miss = positive_miss * (target_luma >= float(cfg.weak_view_reseed_target_luma_min))
        guided = residual_map * positive_miss
        if bool((guided > 0).any().item()):
            flat = guided.reshape(-1)
    if flat.numel() <= 0:
        return 0
    h, w = residual_map.shape
    cand = _select_reseed_pixels(
        flat,
        width=w,
        budget=budget,
        residual_quantile=float(cfg.weak_view_reseed_residual_quantile),
        match_radius_px=float(cfg.weak_view_reseed_match_radius_px),
    )
    if cand.numel() == 0:
        return 0

    R_cw = observation.R_cw.detach()
    t_cw = observation.t_cw.detach()
    intr = observation.intrinsics.detach()
    fx, fy, cx, cy = intr.unbind(dim=0)

    means = field_model.means3d.detach()[: field_model.num_gaussians]
    cam_pts_all = means @ R_cw.transpose(0, 1) + t_cw.unsqueeze(0)
    cam_depths = cam_pts_all[:, 2]
    visible_mask = cam_depths > field_model.field_cfg.min_depth
    if observation.render_stats is not None and "contrib" in observation.render_stats:
        contrib = observation.render_stats["contrib"].to(device=means.device)
        if contrib.numel() == cam_depths.numel():
            visible_mask = visible_mask & (contrib > 0)
    if visible_mask.any():
        fallback_depth = cam_depths[visible_mask].median().clamp_min(field_model.field_cfg.min_depth)
    else:
        global_valid = cam_depths > field_model.field_cfg.min_depth
        if global_valid.any():
            fallback_depth = cam_depths[global_valid].median().clamp_min(field_model.field_cfg.min_depth)
        else:
            fallback_depth = means.new_tensor(field_model.field_cfg.init_depth)

    py = torch.div(cand, w, rounding_mode="floor")
    px = cand - py * w
    u = px.to(dtype=means.dtype) + 0.5
    v = py.to(dtype=means.dtype) + 0.5
    depth_init = torch.full_like(u, float(fallback_depth.item()))
    if visible_mask.any():
        cand_uv = torch.stack((u, v), dim=-1)
        visible_pts = cam_pts_all[visible_mask]
        visible_depths = cam_depths[visible_mask]
        visible_xy = visible_pts[:, :2] / visible_depths[:, None].clamp_min(field_model.field_cfg.min_depth)
        visible_uv = torch.stack(
            (
                fx * visible_xy[:, 0] + cx,
                fy * visible_xy[:, 1] + cy,
            ),
            dim=-1,
        )
        in_bounds = (
            (visible_uv[:, 0] >= 0.0)
            & (visible_uv[:, 0] <= float(w - 1))
            & (visible_uv[:, 1] >= 0.0)
            & (visible_uv[:, 1] <= float(h - 1))
        )
        if in_bounds.any():
            visible_uv = visible_uv[in_bounds]
            visible_depths = visible_depths[in_bounds]
        if visible_uv.numel() > 0:
            diff = cand_uv[:, None, :] - visible_uv[None, :, :]
            dist2 = (diff * diff).sum(dim=-1)
            nearest = dist2.argmin(dim=1)
            nearest_depth = visible_depths[nearest]
            nearest_dist2 = dist2.gather(1, nearest[:, None]).squeeze(1)
            depth_match_radius_px = float(cfg.weak_view_reseed_depth_match_radius_px)
            if depth_match_radius_px > 0.0:
                use_nearest = nearest_dist2 <= (depth_match_radius_px * depth_match_radius_px)
                depth_init[use_nearest] = nearest_depth[use_nearest]
            else:
                depth_init = nearest_depth
    x = (u - cx) / fx.clamp_min(1.0)
    y = (v - cy) / fy.clamp_min(1.0)
    cam_pts = torch.stack((x * depth_init, y * depth_init, depth_init), dim=-1)
    world_pts = (cam_pts - t_cw.unsqueeze(0)) @ R_cw

    rgb = target_rgb[:, py.long(), px.long()].permute(1, 0).contiguous()
    sx = field_model.field_cfg.init_scale_xy * depth_init / max(float(fx.item()), 1.0)
    sy = field_model.field_cfg.init_scale_xy * depth_init / max(float(fy.item()), 1.0)
    sz = field_model.field_cfg.init_scale_z * depth_init
    scale = torch.stack((sx, sy, sz), dim=-1)
    opacity = torch.full(
        (world_pts.shape[0], 1),
        float(cfg.weak_view_reseed_init_opacity),
        device=world_pts.device,
        dtype=world_pts.dtype,
    ).clamp_(1.0e-4, 1.0 - 1.0e-4)
    uv = torch.stack((u, v), dim=-1)
    protect_until_step = torch.full(
        (world_pts.shape[0],),
        int(step) + max(int(cfg.weak_view_reseed_protect_steps), 0),
        device=world_pts.device,
        dtype=torch.long,
    )
    appended = field_model.append_gaussians(
        means3d=world_pts,
        rgb=rgb,
        uv=uv,
        scale=scale,
        opacity=opacity,
        protect_until_step=protect_until_step,
    )
    return int(appended)
