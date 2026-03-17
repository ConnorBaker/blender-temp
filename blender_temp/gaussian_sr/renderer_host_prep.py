from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .reference_renderer import ReferenceProjection
from .warp_gsplat_autograd import PreparedVisibility
from .warp_gsplat_contracts import RasterConfig, estimate_tiles


@dataclass(frozen=True)
class PreparedProjection:
    projection: ReferenceProjection
    prepared: PreparedVisibility


def _torch_sort_bytes(intersection_count: int) -> int:
    return int(intersection_count) * (8 + 4)


def projection_meta_from_projection(
    projection: ReferenceProjection,
    *,
    width: int,
    height: int,
    cfg: RasterConfig,
) -> dict[str, int | bool | str | None]:
    gaussian_count = int(projection.xys.shape[0])
    visible_count = int((projection.num_tiles_hit > 0).sum(dtype=torch.int64).item())
    intersection_count = int(projection.num_tiles_hit.sum(dtype=torch.int64).item()) if gaussian_count > 0 else 0
    tiles_x, tiles_y, tile_count = estimate_tiles(int(width), int(height), int(cfg.tile_size))
    sort_bytes = _torch_sort_bytes(intersection_count)
    budget = cfg.max_sort_buffer_bytes
    return {
        "gaussian_count": gaussian_count,
        "visible_count": visible_count,
        "intersection_count": intersection_count,
        "tile_count": tile_count,
        "tiles_x": tiles_x,
        "tiles_y": tiles_y,
        "render_width": int(width),
        "render_height": int(height),
        "sort_mode": "torch_sort",
        "estimated_sort_buffer_bytes": int(sort_bytes),
        "warp_sort_buffer_bytes": int(sort_bytes * 2),
        "torch_sort_buffer_bytes": int(sort_bytes),
        "sort_buffer_budget_bytes": None if budget is None else int(budget),
        "sort_buffer_within_budget": bool(budget is None or sort_bytes <= budget),
    }


def prepare_visibility_from_projection(
    projection: ReferenceProjection,
    *,
    width: int,
    height: int,
    cfg: RasterConfig,
) -> PreparedVisibility:
    meta = projection_meta_from_projection(projection, width=width, height=height, cfg=cfg)
    if not bool(meta["sort_buffer_within_budget"]):
        raise RuntimeError(
            "Projected intersection sort buffer exceeds configured budget: "
            f"M={int(meta['intersection_count'])} "
            f"estimated_sort_bytes={int(meta['estimated_sort_buffer_bytes'])} "
            f"({float(int(meta['estimated_sort_buffer_bytes'])) / float(1024 ** 3):.2f} GiB) "
            f"budget={int(meta['sort_buffer_budget_bytes'])} "
            f"({float(int(meta['sort_buffer_budget_bytes'])) / float(1024 ** 3):.2f} GiB)"
        )

    device = projection.xys.device
    gaussian_count = int(projection.xys.shape[0])
    tile_size = int(cfg.tile_size)
    tiles_x = int(meta["tiles_x"])
    tiles_y = int(meta["tiles_y"])
    tile_count = int(meta["tile_count"])
    num_tiles_hit = projection.num_tiles_hit.to(dtype=torch.int64)
    intersection_count = int(meta["intersection_count"])

    if gaussian_count == 0 or intersection_count == 0:
        empty = torch.zeros((0,), device=device, dtype=torch.int32)
        tile_start = torch.zeros((tile_count,), device=device, dtype=torch.int32)
        tile_end = torch.zeros((tile_count,), device=device, dtype=torch.int32)
        return PreparedVisibility(
            xys=projection.xys.contiguous(),
            conic=projection.conic.contiguous(),
            rho=projection.rho.contiguous(),
            num_tiles_hit=projection.num_tiles_hit.contiguous(),
            tile_start=tile_start,
            tile_end=tile_end,
            sorted_vals=empty,
            width=int(width),
            height=int(height),
            tile_size=tile_size,
            tiles_x=tiles_x,
            tiles_y=tiles_y,
            tile_count=tile_count,
        )

    offsets = torch.cumsum(num_tiles_hit, dim=0) - num_tiles_hit
    gids = torch.repeat_interleave(
        torch.arange(gaussian_count, device=device, dtype=torch.int64),
        num_tiles_hit,
    )
    local_offsets = torch.arange(intersection_count, device=device, dtype=torch.int64) - torch.repeat_interleave(
        offsets, num_tiles_hit
    )
    tile_min_x = projection.tile_min[:, 0].to(dtype=torch.int64).index_select(0, gids)
    tile_min_y = projection.tile_min[:, 1].to(dtype=torch.int64).index_select(0, gids)
    span_x = (
        projection.tile_max[:, 0].to(dtype=torch.int64) - projection.tile_min[:, 0].to(dtype=torch.int64) + 1
    ).index_select(0, gids)
    tile_x = tile_min_x + torch.remainder(local_offsets, span_x)
    tile_y = tile_min_y + torch.div(local_offsets, span_x, rounding_mode="floor")
    tile_ids = tile_y * int(tiles_x) + tile_x
    depth_key = projection.depth_key.to(dtype=torch.int64).index_select(0, gids) & 0xFFFFFFFF
    keys = (tile_ids << 32) | depth_key
    sorted_keys, order = torch.sort(keys)
    sorted_vals = gids.index_select(0, order).to(dtype=torch.int32)
    sorted_tile_ids = torch.bitwise_right_shift(sorted_keys, 32).to(dtype=torch.int64)

    tile_start = torch.zeros((tile_count,), device=device, dtype=torch.int32)
    tile_end = torch.zeros((tile_count,), device=device, dtype=torch.int32)
    boundaries = torch.ones((intersection_count,), device=device, dtype=torch.bool)
    if intersection_count > 1:
        boundaries[1:] = sorted_tile_ids[1:] != sorted_tile_ids[:-1]
    starts = boundaries.nonzero(as_tuple=False).squeeze(-1)
    tiles = sorted_tile_ids.index_select(0, starts).to(dtype=torch.long)
    tile_start.index_copy_(0, tiles, starts.to(dtype=torch.int32))
    ends = torch.empty_like(starts)
    if starts.numel() > 1:
        ends[:-1] = starts[1:]
    ends[-1] = intersection_count
    tile_end.index_copy_(0, tiles, ends.to(dtype=torch.int32))

    return PreparedVisibility(
        xys=projection.xys.contiguous(),
        conic=projection.conic.contiguous(),
        rho=projection.rho.contiguous(),
        num_tiles_hit=projection.num_tiles_hit.contiguous(),
        tile_start=tile_start,
        tile_end=tile_end,
        sorted_vals=sorted_vals.contiguous(),
        width=int(width),
        height=int(height),
        tile_size=tile_size,
        tiles_x=tiles_x,
        tiles_y=tiles_y,
        tile_count=tile_count,
    )


__all__ = [
    "PreparedProjection",
    "prepare_visibility_from_projection",
    "projection_meta_from_projection",
]
