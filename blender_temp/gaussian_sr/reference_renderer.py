from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .warp_gsplat_contracts import RasterConfig


@dataclass(frozen=True)
class ReferenceProjection:
    xys: Tensor
    conic: Tensor
    rho: Tensor
    radius: Tensor
    tile_min: Tensor
    tile_max: Tensor
    num_tiles_hit: Tensor
    depth_key: Tensor


def _quat_normalize(quat: Tensor) -> Tensor:
    return quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(1.0e-20)


def _quat_to_rot_cols(quat: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    c0 = torch.stack(
        (
            1.0 - 2.0 * (yy + zz),
            2.0 * (xy + wz),
            2.0 * (xz - wy),
        ),
        dim=-1,
    )
    c1 = torch.stack(
        (
            2.0 * (xy - wz),
            1.0 - 2.0 * (xx + zz),
            2.0 * (yz + wx),
        ),
        dim=-1,
    )
    c2 = torch.stack(
        (
            2.0 * (xz + wy),
            2.0 * (yz - wx),
            1.0 - 2.0 * (xx + yy),
        ),
        dim=-1,
    )
    return c0, c1, c2


def project_gaussians_reference(
    means: Tensor,
    quat: Tensor,
    scale: Tensor,
    viewmat: Tensor,
    K: Tensor,
    width: int,
    height: int,
    cfg: RasterConfig | None = None,
) -> ReferenceProjection:
    if cfg is None:
        cfg = RasterConfig()

    dtype = means.dtype
    device = means.device
    r = viewmat[:3, :3]
    t = viewmat[:3, 3]
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    cam = means @ r.transpose(0, 1) + t.unsqueeze(0)
    x = cam[:, 0]
    y = cam[:, 1]
    z = cam[:, 2]

    depth_valid = (z > float(cfg.near_plane)) & (z < float(cfg.far_plane))
    safe_x = torch.where(depth_valid, x, torch.zeros_like(x))
    safe_y = torch.where(depth_valid, y, torch.zeros_like(y))
    safe_z = torch.where(depth_valid, z, torch.ones_like(z))
    inv_z = 1.0 / safe_z
    u = fx * safe_x * inv_z + cx
    v = fy * safe_y * inv_z + cy

    qn = _quat_normalize(quat)
    c0, c1, c2 = _quat_to_rot_cols(qn)
    t0v = c0 @ r.transpose(0, 1)
    t1v = c1 @ r.transpose(0, 1)
    t2v = c2 @ r.transpose(0, 1)

    s = scale.clamp_min(1.0e-8)
    s0 = s[:, 0] * s[:, 0]
    s1 = s[:, 1] * s[:, 1]
    s2 = s[:, 2] * s[:, 2]

    cov3d = (
        s0[:, None, None] * torch.einsum("ni,nj->nij", t0v, t0v)
        + s1[:, None, None] * torch.einsum("ni,nj->nij", t1v, t1v)
        + s2[:, None, None] * torch.einsum("ni,nj->nij", t2v, t2v)
    )
    c00 = cov3d[:, 0, 0]
    c01 = cov3d[:, 0, 1]
    c02 = cov3d[:, 0, 2]
    c11 = cov3d[:, 1, 1]
    c12 = cov3d[:, 1, 2]
    c22 = cov3d[:, 2, 2]

    inv_z2 = inv_z * inv_z
    j00 = fx * inv_z
    j02 = -fx * safe_x * inv_z2
    j11 = fy * inv_z
    j12 = -fy * safe_y * inv_z2

    s00 = j00 * j00 * c00 + 2.0 * j00 * j02 * c02 + j02 * j02 * c22
    s11 = j11 * j11 * c11 + 2.0 * j11 * j12 * c12 + j12 * j12 * c22
    s01 = j00 * (c01 * j11 + c02 * j12) + j02 * (c12 * j11 + c22 * j12)

    det_noeps = (s00 * s11 - s01 * s01).clamp_min(1.0e-20)
    s00e = s00 + float(cfg.eps2d)
    s11e = s11 + float(cfg.eps2d)
    s01e = s01
    det_eps = (s00e * s11e - s01e * s01e).clamp_min(1.0e-20)

    a = s11e / det_eps
    b = -s01e / det_eps
    c = s00e / det_eps
    rho = torch.sqrt(det_noeps / det_eps)

    trace = s00e + s11e
    disc = ((s00e - s11e) * (s00e - s11e) + 4.0 * s01e * s01e).clamp_min(0.0).sqrt()
    lam = 0.5 * (trace + disc)
    radius = torch.ceil(3.0 * lam.clamp_min(0.0).sqrt()).to(torch.int32)
    radius_f = radius.to(dtype=dtype)

    depth_key = (z * float(cfg.depth_scale)).clamp(0.0, 2147483647.0).trunc().to(torch.int32)
    xys = torch.stack((u, v), dim=-1)
    conic = torch.stack((a, b, c), dim=-1)
    xys = torch.where(depth_valid.unsqueeze(-1), xys, torch.zeros_like(xys))
    conic = torch.where(depth_valid.unsqueeze(-1), conic, torch.zeros_like(conic))
    rho = torch.where(depth_valid, rho, torch.ones_like(rho))
    radius = torch.where(depth_valid, radius, torch.zeros_like(radius))
    depth_key = torch.where(depth_valid, depth_key, torch.zeros_like(depth_key))

    tiles_x = (int(width) + int(cfg.tile_size) - 1) // int(cfg.tile_size)
    tiles_y = (int(height) + int(cfg.tile_size) - 1) // int(cfg.tile_size)

    valid = depth_valid
    if float(cfg.radius_clip) > 0.0:
        valid = valid & (radius.to(dtype) < float(cfg.radius_clip))
    valid = valid & ~(
        (u + radius_f < 0.0) | (u - radius_f >= float(width)) | (v + radius_f < 0.0) | (v - radius_f >= float(height))
    )

    # Compute tile bounds unconditionally (no data-dependent branch) so that
    # this function is compatible with torch.vmap.  Invalid Gaussians are
    # masked to (0,0)/(−1,−1) which yields num_tiles_hit == 0.
    tile_min_x = torch.floor((u - radius_f) / float(cfg.tile_size)).to(torch.int32).clamp(0, tiles_x - 1)
    tile_min_y = torch.floor((v - radius_f) / float(cfg.tile_size)).to(torch.int32).clamp(0, tiles_y - 1)
    tile_max_x = torch.floor((u + radius_f) / float(cfg.tile_size)).to(torch.int32).clamp(0, tiles_x - 1)
    tile_max_y = torch.floor((v + radius_f) / float(cfg.tile_size)).to(torch.int32).clamp(0, tiles_y - 1)

    zero_i32 = torch.zeros_like(radius)
    neg1_i32 = torch.full_like(radius, -1)
    tile_min = torch.stack(
        (torch.where(valid, tile_min_x, zero_i32), torch.where(valid, tile_min_y, zero_i32)),
        dim=-1,
    )
    tile_max = torch.stack(
        (torch.where(valid, tile_max_x, neg1_i32), torch.where(valid, tile_max_y, neg1_i32)),
        dim=-1,
    )
    num_tiles_hit = torch.where(
        valid,
        ((tile_max_x - tile_min_x + 1) * (tile_max_y - tile_min_y + 1)).to(torch.int32),
        zero_i32,
    )

    return ReferenceProjection(
        xys=xys,
        conic=conic,
        rho=rho,
        radius=radius,
        tile_min=tile_min,
        tile_max=tile_max,
        num_tiles_hit=num_tiles_hit,
        depth_key=depth_key,
    )


def render_values_reference(
    means: Tensor,
    quat: Tensor,
    scale: Tensor,
    values: Tensor,
    opacity: Tensor,
    viewmat: Tensor,
    K: Tensor,
    width: int,
    height: int,
    cfg: RasterConfig | None = None,
    background: Tensor | None = None,
) -> Tensor:
    if cfg is None:
        cfg = RasterConfig()
    if background is None:
        background = torch.zeros(values.shape[1], device=values.device, dtype=values.dtype)

    projection = project_gaussians_reference(
        means=means,
        quat=quat,
        scale=scale,
        viewmat=viewmat,
        K=K,
        width=width,
        height=height,
        cfg=cfg,
    )

    visible = (projection.num_tiles_hit > 0).nonzero(as_tuple=False).squeeze(-1)
    if visible.numel() == 0:
        return background.view(1, 1, -1).expand(height, width, -1).clone()

    order = torch.argsort(projection.depth_key.index_select(0, visible), stable=True)
    gids = visible.index_select(0, order)

    out = values.new_empty((height, width, values.shape[1]))
    antialiased = cfg.rasterize_mode == "antialiased"
    tile_size = int(cfg.tile_size)
    for py in range(int(height)):
        tile_y = py // tile_size
        fy = float(py) + 0.5
        for px in range(int(width)):
            tile_x = px // tile_size
            fx = float(px) + 0.5
            accum = values.new_zeros((values.shape[1],))
            trans = values.new_tensor(1.0)
            for gid in gids.tolist():
                tile_min = projection.tile_min[gid]
                tile_max = projection.tile_max[gid]
                if tile_x < int(tile_min[0].item()) or tile_x > int(tile_max[0].item()):
                    continue
                if tile_y < int(tile_min[1].item()) or tile_y > int(tile_max[1].item()):
                    continue

                xy = projection.xys[gid]
                conic = projection.conic[gid]
                dx = values.new_tensor(fx) - xy[0]
                dy = values.new_tensor(fy) - xy[1]
                sigma = 0.5 * (conic[0] * dx * dx + conic[2] * dy * dy) + conic[1] * dx * dy
                weight = torch.exp(-sigma)
                alpha = opacity[gid] * weight
                if antialiased:
                    alpha = alpha * projection.rho[gid]
                alpha = alpha.clamp_max(float(cfg.clamp_alpha_max))
                if float(alpha.detach().item()) < float(cfg.alpha_min):
                    continue
                accum = accum + trans * alpha * values[gid]
                trans = trans * (1.0 - alpha)
                if float(trans.detach().item()) < float(cfg.transmittance_eps):
                    break
            out[py, px] = accum + trans * background
    return out


def render_values_from_prepared_reference(
    *,
    prepared,
    values: Tensor,
    opacity: Tensor,
    background: Tensor | None = None,
    cfg: RasterConfig | None = None,
) -> Tensor:
    if cfg is None:
        cfg = RasterConfig()
    if background is None:
        background = torch.zeros(values.shape[1], device=values.device, dtype=values.dtype)

    height = int(prepared.height)
    width = int(prepared.width)
    channels = int(values.shape[1])
    out = values.new_empty((height, width, channels))
    antialiased = cfg.rasterize_mode == "antialiased"
    tile_size = int(prepared.tile_size)
    for py in range(height):
        tile_y = py // tile_size
        fy = float(py) + 0.5
        for px in range(width):
            tile_x = px // tile_size
            tile_id = tile_y * int(prepared.tiles_x) + tile_x
            start = int(prepared.tile_start[tile_id].item())
            end = int(prepared.tile_end[tile_id].item())
            accum = values.new_zeros((channels,))
            trans = values.new_tensor(1.0)
            for idx in range(start, end):
                gid = int(prepared.sorted_vals[idx].item())
                xy = prepared.xys[gid]
                conic = prepared.conic[gid]
                dx = values.new_tensor(float(px) + 0.5) - xy[0]
                dy = values.new_tensor(fy) - xy[1]
                sigma = 0.5 * (conic[0] * dx * dx + conic[2] * dy * dy) + conic[1] * dx * dy
                weight = torch.exp(-sigma)
                alpha = opacity[gid] * weight
                if antialiased:
                    alpha = alpha * prepared.rho[gid]
                alpha = alpha.clamp_max(float(cfg.clamp_alpha_max))
                if float(alpha.detach().item()) < float(cfg.alpha_min):
                    continue
                accum = accum + trans * alpha * values[gid]
                trans = trans * (1.0 - alpha)
                if float(trans.detach().item()) < float(cfg.transmittance_eps):
                    break
            out[py, px] = accum + trans * background
    return out


def project_gaussians_batched(
    means: Tensor,
    quat: Tensor,
    scale: Tensor,
    viewmats: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    cfg: RasterConfig | None = None,
) -> ReferenceProjection:
    """Project N Gaussians through V views using torch.vmap.

    Args:
        means: Gaussian centers [N, 3]
        quat: Gaussian quaternions [N, 4]
        scale: Gaussian scales [N, 3]
        viewmats: Per-view 4x4 viewmats [V, 4, 4]
        Ks: Per-view 3x3 intrinsics [V, 3, 3]
        width: Render width
        height: Render height
        cfg: Raster config (uses defaults if None)

    Returns:
        ReferenceProjection with all fields having shape [V, N, ...].
        project_gaussians_reference is vmap-compatible (no .item() calls
        or data-dependent control flow), so this is a thin vmap wrapper.
    """
    if cfg is None:
        cfg = RasterConfig()

    def _project_one_view(viewmat: Tensor, K: Tensor) -> tuple[Tensor, ...]:
        proj = project_gaussians_reference(
            means=means,
            quat=quat,
            scale=scale,
            viewmat=viewmat,
            K=K,
            width=width,
            height=height,
            cfg=cfg,
        )
        return (
            proj.xys,
            proj.conic,
            proj.rho,
            proj.radius,
            proj.tile_min,
            proj.tile_max,
            proj.num_tiles_hit,
            proj.depth_key,
        )

    xys, conic, rho, radius, tile_min, tile_max, num_tiles_hit, depth_key = torch.vmap(
        _project_one_view,
        in_dims=(0, 0),
    )(viewmats, Ks)

    return ReferenceProjection(
        xys=xys,
        conic=conic,
        rho=rho,
        radius=radius,
        tile_min=tile_min,
        tile_max=tile_max,
        num_tiles_hit=num_tiles_hit,
        depth_key=depth_key,
    )


__all__ = [
    "ReferenceProjection",
    "project_gaussians_reference",
    "project_gaussians_batched",
    "render_values_from_prepared_reference",
    "render_values_reference",
]
