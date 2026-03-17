import os
from dataclasses import dataclass

import torch
from torch import Tensor

from . import warp_gsplat_kernels as warp_gsplat_kernel_module
from .warp_gsplat_contracts import (
    DataContracts,
    KERNEL_MAPPING_TABLE,
    MERMAID_PORT_FLOWCHART,
    RasterConfig,
    _assert_1d,
    _assert_cuda_float32_contiguous,
    estimate_buffer_bytes_for_example,
    estimate_intersections,
    estimate_tiles,
)
from .warp_gsplat_kernels import (
    get_tile_bin_edges_kernel,
    map_to_intersects_kernel,
    specialize_project_kernel,
    specialize_raster_kernels,
    specialize_visibility_stats_kernel,
)
from .warp_runtime import require_warp, wp

try:
    from .reference_renderer import project_gaussians_reference, render_values_reference

    def _project_gaussians_reference_local(
        means: Tensor,
        quat: Tensor,
        scale: Tensor,
        viewmat: Tensor,
        K: Tensor,
        width: int,
        height: int,
        cfg: RasterConfig,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
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
        return (
            projection.xys,
            projection.conic,
            projection.rho,
            projection.num_tiles_hit,
            projection.tile_min,
            projection.tile_max,
            projection.depth_key,
        )

except ImportError:

    def _ref_quat_normalize(quat: Tensor) -> Tensor:
        return quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(1.0e-20)

    def _ref_quat_to_rot_cols(quat: Tensor) -> tuple[Tensor, Tensor, Tensor]:
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

        c0 = torch.stack((1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy)), dim=-1)
        c1 = torch.stack((2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx)), dim=-1)
        c2 = torch.stack((2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (xx + yy)), dim=-1)
        return c0, c1, c2

    def _project_gaussians_reference_local(
        means: Tensor,
        quat: Tensor,
        scale: Tensor,
        viewmat: Tensor,
        K: Tensor,
        width: int,
        height: int,
        cfg: RasterConfig,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
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
        safe_z = z.clamp_min(1.0e-20)
        inv_z = 1.0 / safe_z
        u = fx * x * inv_z + cx
        v = fy * y * inv_z + cy

        qn = _ref_quat_normalize(quat)
        c0, c1, c2 = _ref_quat_to_rot_cols(qn)
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
        j02 = -fx * x * inv_z2
        j11 = fy * inv_z
        j12 = -fy * y * inv_z2
        s00 = j00 * j00 * c00 + 2.0 * j00 * j02 * c02 + j02 * j02 * c22
        s11 = j11 * j11 * c11 + 2.0 * j11 * j12 * c12 + j12 * j12 * c22
        s01 = j00 * (c01 * j11 + c02 * j12) + j02 * (c12 * j11 + c22 * j12)

        det_noeps = (s00 * s11 - s01 * s01).clamp_min(1.0e-20)
        s00e = s00 + float(cfg.eps2d)
        s11e = s11 + float(cfg.eps2d)
        det_eps = (s00e * s11e - s01 * s01).clamp_min(1.0e-20)
        a = s11e / det_eps
        b = -s01 / det_eps
        c = s00e / det_eps
        rho = torch.sqrt(det_noeps / det_eps)

        trace = s00e + s11e
        disc = ((s00e - s11e) * (s00e - s11e) + 4.0 * s01 * s01).clamp_min(0.0).sqrt()
        lam = 0.5 * (trace + disc)
        radius = torch.ceil(3.0 * lam.clamp_min(0.0).sqrt()).to(torch.int32)
        radius_f = radius.to(dtype=means.dtype)

        tiles_x = (int(width) + int(cfg.tile_size) - 1) // int(cfg.tile_size)
        tiles_y = (int(height) + int(cfg.tile_size) - 1) // int(cfg.tile_size)
        tile_min = torch.zeros((means.shape[0], 2), device=means.device, dtype=torch.int32)
        tile_max = torch.full((means.shape[0], 2), -1, device=means.device, dtype=torch.int32)
        depth_key = (z * float(cfg.depth_scale)).clamp(0.0, 2147483647.0).trunc().to(torch.int32)

        valid = (z > float(cfg.near_plane)) & (z < float(cfg.far_plane))
        if float(cfg.radius_clip) > 0.0:
            valid = valid & (radius.to(means.dtype) < float(cfg.radius_clip))
        valid = valid & ~(
            (u + radius_f < 0.0)
            | (u - radius_f >= float(width))
            | (v + radius_f < 0.0)
            | (v - radius_f >= float(height))
        )
        num_tiles_hit = torch.zeros((means.shape[0],), device=means.device, dtype=torch.int32)
        if bool(valid.any().item()):
            tile_min_f = torch.stack(
                (
                    torch.floor((u - radius_f) / float(cfg.tile_size)),
                    torch.floor((v - radius_f) / float(cfg.tile_size)),
                ),
                dim=-1,
            )
            tile_max_f = torch.stack(
                (
                    torch.floor((u + radius_f) / float(cfg.tile_size)),
                    torch.floor((v + radius_f) / float(cfg.tile_size)),
                ),
                dim=-1,
            )
            tile_min_valid = tile_min_f[valid].to(torch.int32)
            tile_max_valid = tile_max_f[valid].to(torch.int32)
            tile_min_valid[:, 0].clamp_(0, tiles_x - 1)
            tile_min_valid[:, 1].clamp_(0, tiles_y - 1)
            tile_max_valid[:, 0].clamp_(0, tiles_x - 1)
            tile_max_valid[:, 1].clamp_(0, tiles_y - 1)
            tile_min[valid] = tile_min_valid
            tile_max[valid] = tile_max_valid
            num_tiles_hit[valid] = (
                (tile_max_valid[:, 0] - tile_min_valid[:, 0] + 1) * (tile_max_valid[:, 1] - tile_min_valid[:, 1] + 1)
            ).to(torch.int32)
        xys = torch.stack((u, v), dim=-1)
        conic = torch.stack((a, b, c), dim=-1)
        return xys, conic, rho, num_tiles_hit, tile_min, tile_max, depth_key

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

        xys, conic, rho, num_tiles_hit, tile_min, tile_max, depth_key = _project_gaussians_reference_local(
            means, quat, scale, viewmat, K, width, height, cfg
        )
        visible = (num_tiles_hit > 0).nonzero(as_tuple=False).squeeze(-1)
        if visible.numel() == 0:
            return background.view(1, 1, -1).expand(height, width, -1).clone()

        order = torch.argsort(depth_key.index_select(0, visible), stable=True)
        gids = visible.index_select(0, order)
        out = values.new_empty((height, width, values.shape[1]))
        antialiased = cfg.rasterize_mode == "antialiased"
        tile_size = int(cfg.tile_size)
        tiles_x = (int(width) + tile_size - 1) // tile_size
        for py in range(int(height)):
            tile_y = py // tile_size
            fy = float(py) + 0.5
            for px in range(int(width)):
                tile_x = px // tile_size
                accum = values.new_zeros((values.shape[1],))
                trans = values.new_tensor(1.0)
                for gid in gids.tolist():
                    tmin = tile_min[gid]
                    tmax = tile_max[gid]
                    if tile_x < int(tmin[0].item()) or tile_x > int(tmax[0].item()):
                        continue
                    if tile_y < int(tmin[1].item()) or tile_y > int(tmax[1].item()):
                        continue
                    xy = xys[gid]
                    abc = conic[gid]
                    dx = values.new_tensor(float(px) + 0.5) - xy[0]
                    dy = values.new_tensor(fy) - xy[1]
                    sigma = 0.5 * (abc[0] * dx * dx + abc[2] * dy * dy) + abc[1] * dx * dy
                    w = torch.exp(-sigma)
                    alpha = opacity[gid] * w
                    if antialiased:
                        alpha = alpha * rho[gid]
                    alpha = alpha.clamp_max(float(cfg.clamp_alpha_max))
                    if float(alpha.detach().item()) < float(cfg.alpha_min):
                        continue
                    accum = accum + trans * alpha * values[gid]
                    trans = trans * (1.0 - alpha)
                    if float(trans.detach().item()) < float(cfg.transmittance_eps):
                        break
                out[py, px] = accum + trans * background
        return out


_LOADED_KERNEL_MODULES: set[tuple[str, int]] = set()
_SCRATCH_TENSOR_CACHE: dict[tuple[str, tuple[int, ...], str, int | None, str], Tensor] = {}


def _warp_block_dim() -> int:
    raw = os.environ.get("BLENDER_TEMP_WARP_BLOCK_DIM", "608").strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid BLENDER_TEMP_WARP_BLOCK_DIM={raw!r}") from exc
    if value <= 0:
        raise ValueError(f"BLENDER_TEMP_WARP_BLOCK_DIM must be > 0, got {value}")
    return value


def _warp_project_block_dim() -> int:
    raw = os.environ.get("BLENDER_TEMP_WARP_PROJECT_BLOCK_DIM", "256").strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid BLENDER_TEMP_WARP_PROJECT_BLOCK_DIM={raw!r}") from exc
    if value <= 0:
        raise ValueError(f"BLENDER_TEMP_WARP_PROJECT_BLOCK_DIM must be > 0, got {value}")
    return value


@dataclass(frozen=True)
class PreparedVisibility:
    xys: Tensor
    conic: Tensor
    rho: Tensor
    num_tiles_hit: Tensor
    tile_start: Tensor
    tile_end: Tensor
    sorted_vals: Tensor
    width: int
    height: int
    tile_size: int
    tiles_x: int
    tiles_y: int
    tile_count: int
    gaussian_count_value: int | None = None
    intersection_count_value: int | None = None

    @property
    def device(self) -> torch.device:
        return self.xys.device

    @property
    def gaussian_count(self) -> int:
        if self.gaussian_count_value is not None:
            return int(self.gaussian_count_value)
        return int(self.xys.shape[0])

    @property
    def intersection_count(self) -> int:
        if self.intersection_count_value is not None:
            return int(self.intersection_count_value)
        return int(self.sorted_vals.shape[0])

    @property
    def gaussian_capacity(self) -> int:
        return int(self.xys.shape[0])

    @property
    def intersection_capacity(self) -> int:
        return int(self.sorted_vals.shape[0])

    def meta(self) -> dict[str, Tensor]:
        device = self.device
        visible_count = (self.num_tiles_hit > 0).sum(dtype=torch.int64)
        return {
            "meta_gaussian_count": torch.tensor(self.gaussian_count, device=device, dtype=torch.int64),
            "meta_visible_count": visible_count,
            "meta_intersection_count": torch.tensor(self.intersection_count, device=device, dtype=torch.int64),
            "meta_tile_count": torch.tensor(self.tile_count, device=device, dtype=torch.int64),
            "meta_tiles_x": torch.tensor(self.tiles_x, device=device, dtype=torch.int64),
            "meta_tiles_y": torch.tensor(self.tiles_y, device=device, dtype=torch.int64),
            "meta_render_width": torch.tensor(self.width, device=device, dtype=torch.int64),
            "meta_render_height": torch.tensor(self.height, device=device, dtype=torch.int64),
        }


class _KernelLaunchCache:
    def __init__(self) -> None:
        self._commands: dict[tuple[str, str, int], wp.Launch] = {}

    def clear(self) -> None:
        self._commands.clear()

    @staticmethod
    def _recordable_arg(value: object) -> object:
        if isinstance(value, torch.Tensor):
            return value.detach()
        return value

    def launch(
        self,
        kernel,
        *,
        dim: int | tuple[int, ...],
        inputs: list[object],
        outputs: list[object],
        device: str,
        stream,
    ) -> None:
        recordable_inputs = [self._recordable_arg(value) for value in inputs]
        recordable_outputs = [self._recordable_arg(value) for value in outputs]
        wp.launch(
            kernel,
            dim=dim,
            inputs=recordable_inputs,
            outputs=recordable_outputs,
            device=device,
            stream=stream,
            block_dim=_warp_block_dim(),
        )


_LAUNCH_CACHE = _KernelLaunchCache()


def clear_warp_launch_cache() -> None:
    _LAUNCH_CACHE.clear()
    _SCRATCH_TENSOR_CACHE.clear()


def _device_str(device: torch.device) -> str:
    if device.type != "cuda":
        raise ValueError("Warp gsplat renderer requires CUDA tensors.")
    return f"cuda:{device.index}" if device.index is not None else "cuda:0"


def _warp_stream(device: torch.device):
    return wp.stream_from_torch(device) if device.type == "cuda" else None


def _ensure_warp_ready(device: torch.device) -> str:
    require_warp()
    device_str = _device_str(device)
    if os.environ.get("BLENDER_TEMP_WARP_DISABLE_MEMPOOL", ""):
        try:
            if wp.is_mempool_enabled(device_str):
                wp.set_mempool_enabled(device_str, False)
        except Exception:
            pass
    block_dim = _warp_block_dim()
    key = (device_str, block_dim)
    if key not in _LOADED_KERNEL_MODULES:
        wp.load_module(module=warp_gsplat_kernel_module, device=device_str, block_dim=block_dim)
        _LOADED_KERNEL_MODULES.add(key)
    return device_str


def _scratch_tensor(
    tag: str,
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    zero: bool = False,
) -> Tensor:
    key = (tag, shape, device.type, device.index, str(dtype))
    cached = _SCRATCH_TENSOR_CACHE.get(key)
    if cached is None:
        cached = torch.empty(shape, device=device, dtype=dtype)
        _SCRATCH_TENSOR_CACHE[key] = cached
    if zero:
        cached.zero_()
    return cached


def _torch_sort_pairs(keys: Tensor, vals: Tensor) -> tuple[Tensor, Tensor]:
    sorted_keys, idx = torch.sort(keys)
    sorted_vals = vals.index_select(0, idx)
    return sorted_keys, sorted_vals


def _sort_buffer_bytes(M: int, *, use_warp_radix: bool) -> int:
    length = (2 * M) if use_warp_radix else M
    return length * 8 + length * 4


def _select_sort_mode_and_bytes(
    M: int,
    cfg: RasterConfig,
) -> tuple[bool, str, int, int, int]:
    use_warp_radix = cfg.sort_mode in ("auto", "warp_radix")
    warp_sort_bytes = _sort_buffer_bytes(M, use_warp_radix=True)
    torch_sort_bytes = _sort_buffer_bytes(M, use_warp_radix=False)
    budget = cfg.max_sort_buffer_bytes
    if budget is not None and use_warp_radix and warp_sort_bytes > budget and cfg.sort_mode == "auto":
        use_warp_radix = False
    chosen_mode = "warp_radix" if use_warp_radix else "torch_sort"
    chosen_bytes = warp_sort_bytes if use_warp_radix else torch_sort_bytes
    return use_warp_radix, chosen_mode, chosen_bytes, warp_sort_bytes, torch_sort_bytes


def _validate_vec_input(t: Tensor, name: str, width: int) -> int:
    _assert_cuda_float32_contiguous(t, name)
    if t.ndim != 2 or t.shape[1] != width:
        raise ValueError(f"{name} must have shape [N,{width}], got {tuple(t.shape)}")
    return int(t.shape[0])


def _validate_render_inputs(
    means: Tensor,
    quat: Tensor,
    scale: Tensor,
    opacity: Tensor,
    viewmat: Tensor,
    K: Tensor,
) -> int:
    N = _validate_vec_input(means, "means", 3)
    if _validate_vec_input(quat, "quat", 4) != N:
        raise ValueError(f"quat length mismatch: expected {N}, got {quat.shape[0]}")
    if _validate_vec_input(scale, "scale", 3) != N:
        raise ValueError(f"scale length mismatch: expected {N}, got {scale.shape[0]}")
    _assert_cuda_float32_contiguous(opacity, "opacity")
    _assert_1d(opacity, "opacity")
    if int(opacity.shape[0]) != N:
        raise ValueError(f"opacity length mismatch: expected {N}, got {opacity.shape[0]}")
    _assert_cuda_float32_contiguous(viewmat, "viewmat", shape=(4, 4))
    _assert_cuda_float32_contiguous(K, "K", shape=(3, 3))
    return N


def _wrap_input_tensor(t: Tensor, dtype, *, requires_grad: bool):
    return wp.from_torch(t.detach(), dtype=dtype, requires_grad=requires_grad and t.requires_grad)


def _wrap_output_tensor(t: Tensor, dtype, *, requires_grad: bool = False):
    return wp.from_torch(t, dtype=dtype, requires_grad=requires_grad)


def _allocate_visibility_state(
    means: Tensor,
    quat: Tensor,
    scale: Tensor,
    viewmat: Tensor,
    K: Tensor,
    width: int,
    height: int,
    cfg: RasterConfig,
    *,
    requires_grad: bool,
    reuse_scratch: bool = False,
) -> dict[str, object]:
    device = means.device
    device_str = _ensure_warp_ready(device)
    tile_size = int(cfg.tile_size)
    tiles_x, tiles_y, tile_count = estimate_tiles(width, height, tile_size)
    N = int(means.shape[0])

    def _alloc(tag: str, shape: tuple[int, ...], dtype: torch.dtype) -> Tensor:
        if reuse_scratch:
            return _scratch_tensor(tag, shape, device=device, dtype=dtype)
        return torch.empty(shape, device=device, dtype=dtype)

    xys_t = _alloc("xys", (N, 2), torch.float32)
    conic_t = _alloc("conic", (N, 3), torch.float32)
    rho_t = _alloc("rho", (N,), torch.float32)
    radius_t = _alloc("radius", (N,), torch.int32)
    num_tiles_hit_t = _alloc("num_tiles_hit", (N,), torch.int32)
    cum_tiles_hit_t = _alloc("cum_tiles_hit", (N,), torch.int32)
    tile_min_t = _alloc("tile_min", (N, 2), torch.int32)
    tile_max_t = _alloc("tile_max", (N, 2), torch.int32)
    depth_key_t = _alloc("depth_key", (N,), torch.int32)

    return {
        "device": device,
        "device_str": device_str,
        "requires_grad": requires_grad,
        "reuse_scratch": reuse_scratch,
        "width": int(width),
        "height": int(height),
        "tile_size": tile_size,
        "tiles_x": tiles_x,
        "tiles_y": tiles_y,
        "tile_count": tile_count,
        "gaussian_count": N,
        "means_t": means,
        "quat_t": quat,
        "scale_t": scale,
        "viewmat_t": viewmat,
        "K_t": K,
        "means_wp": _wrap_input_tensor(means, wp.vec3f, requires_grad=requires_grad),
        "quat_wp": _wrap_input_tensor(quat, wp.vec4f, requires_grad=requires_grad),
        "scale_wp": _wrap_input_tensor(scale, wp.vec3f, requires_grad=requires_grad),
        "viewmat_wp": _wrap_input_tensor(viewmat, wp.float32, requires_grad=requires_grad),
        "K_wp": _wrap_input_tensor(K, wp.float32, requires_grad=requires_grad),
        "xys_t": xys_t,
        "conic_t": conic_t,
        "rho_t": rho_t,
        "radius_t": radius_t,
        "num_tiles_hit_t": num_tiles_hit_t,
        "cum_tiles_hit_t": cum_tiles_hit_t,
        "tile_min_t": tile_min_t,
        "tile_max_t": tile_max_t,
        "depth_key_t": depth_key_t,
        "xys_wp": _wrap_output_tensor(xys_t, wp.vec2f, requires_grad=requires_grad),
        "conic_wp": _wrap_output_tensor(conic_t, wp.vec3f, requires_grad=requires_grad),
        "rho_wp": _wrap_output_tensor(rho_t, wp.float32, requires_grad=requires_grad),
        "radius_wp": _wrap_output_tensor(radius_t, wp.int32),
        "num_tiles_hit_wp": _wrap_output_tensor(num_tiles_hit_t, wp.int32),
        "cum_tiles_hit_wp": _wrap_output_tensor(cum_tiles_hit_t, wp.int32),
        "tile_min_wp": _wrap_output_tensor(tile_min_t, wp.vec2i),
        "tile_max_wp": _wrap_output_tensor(tile_max_t, wp.vec2i),
        "depth_key_wp": _wrap_output_tensor(depth_key_t, wp.int32),
    }


def _launch_project(state: dict[str, object], cfg: RasterConfig, *, stream) -> None:
    project_kernel = specialize_project_kernel(
        tile_size=int(state["tile_size"]),
        near_plane=float(cfg.near_plane),
        far_plane=float(cfg.far_plane),
        eps2d=float(cfg.eps2d),
        radius_clip=float(cfg.radius_clip),
        depth_scale=float(cfg.depth_scale),
    )
    inputs = [
        state["means_wp"],
        state["quat_wp"],
        state["scale_wp"],
        state["viewmat_wp"],
        state["K_wp"],
        int(state["width"]),
        int(state["height"]),
        int(state["tiles_x"]),
        int(state["tiles_y"]),
    ]
    outputs = [
        state["xys_wp"],
        state["conic_wp"],
        state["rho_wp"],
        state["radius_wp"],
        state["num_tiles_hit_wp"],
        state["tile_min_wp"],
        state["tile_max_wp"],
        state["depth_key_wp"],
    ]
    wp.launch(
        project_kernel,
        dim=int(state["gaussian_count"]),
        inputs=inputs,
        outputs=outputs,
        device=str(state["device_str"]),
        stream=stream,
        block_dim=_warp_project_block_dim(),
    )


def _prepare_sorted_intersections(
    state: dict[str, object],
    cfg: RasterConfig,
    *,
    stream,
) -> PreparedVisibility:
    wp.utils.array_scan(state["num_tiles_hit_wp"], state["cum_tiles_hit_wp"], inclusive=True)
    N = int(state["gaussian_count"])
    cum_tiles_hit_t: Tensor = state["cum_tiles_hit_t"]  # type: ignore[assignment]
    M = int(cum_tiles_hit_t[-1].item()) if N > 0 else 0

    if M <= 0:
        empty_i32 = torch.empty(0, device=state["device"], dtype=torch.int32)  # type: ignore[arg-type]
        prepared = PreparedVisibility(
            xys=state["xys_t"],  # type: ignore[arg-type]
            conic=state["conic_t"],  # type: ignore[arg-type]
            rho=state["rho_t"],  # type: ignore[arg-type]
            num_tiles_hit=state["num_tiles_hit_t"],  # type: ignore[arg-type]
            tile_start=empty_i32,
            tile_end=empty_i32,
            sorted_vals=empty_i32,
            width=int(state["width"]),
            height=int(state["height"]),
            tile_size=int(state["tile_size"]),
            tiles_x=int(state["tiles_x"]),
            tiles_y=int(state["tiles_y"]),
            tile_count=int(state["tile_count"]),
        )
        state["prepared"] = prepared
        state["tile_start_wp"] = None
        state["tile_end_wp"] = None
        state["sorted_vals_wp"] = None
        return prepared

    use_warp_radix, chosen_mode, chosen_bytes, _warp_sort_bytes, _torch_sort_bytes = _select_sort_mode_and_bytes(M, cfg)
    budget = cfg.max_sort_buffer_bytes
    if budget is not None:
        if chosen_bytes > budget:
            gib = float(chosen_bytes) / float(1024**3)
            budget_gib = float(budget) / float(1024**3)
            raise RuntimeError(
                "Projected intersection sort buffer exceeds configured budget: "
                f"M={M}, sort_mode={chosen_mode}, "
                f"estimated_sort_bytes={chosen_bytes} ({gib:.2f} GiB), "
                f"budget={budget} ({budget_gib:.2f} GiB). "
                "Recommended fixes, in order: "
                "1) lower projected footprint with `radius_clip_px`; "
                "2) reduce initial or learned scale growth; "
                "3) freeze or prune before the next stage or after density growth; "
                "4) only then increase `anchor_stride`."
            )

    pair_len = 2 * M if use_warp_radix else M
    reuse_scratch = bool(state.get("reuse_scratch", False))
    device = state["device"]  # type: ignore[assignment]
    if reuse_scratch:
        keys_t = _scratch_tensor("sort_keys", (pair_len,), device=device, dtype=torch.int64)
        vals_t = _scratch_tensor("sort_vals", (pair_len,), device=device, dtype=torch.int32)
    else:
        keys_t = torch.empty(pair_len, device=device, dtype=torch.int64)
        vals_t = torch.empty(pair_len, device=device, dtype=torch.int32)

    _LAUNCH_CACHE.launch(
        map_to_intersects_kernel,
        dim=N,
        inputs=[
            state["num_tiles_hit_t"],
            state["cum_tiles_hit_t"],
            state["tile_min_t"],
            state["tile_max_t"],
            state["depth_key_t"],
            int(state["tiles_x"]),
        ],
        outputs=[keys_t, vals_t],
        device=str(state["device_str"]),
        stream=stream,
    )

    did_warp_sort = False
    sorted_keys_t = keys_t[:M]
    sorted_vals_t = vals_t[:M]
    if cfg.sort_mode in ("auto", "warp_radix"):
        try:
            wp.utils.radix_sort_pairs(
                _wrap_output_tensor(keys_t, wp.int64),
                _wrap_output_tensor(vals_t, wp.int32),
                count=M,
            )
            did_warp_sort = True
        except Exception:
            did_warp_sort = False
            if cfg.sort_mode == "warp_radix":
                raise

    if not did_warp_sort:
        sorted_keys_t, sorted_vals_t = _torch_sort_pairs(keys_t[:M].contiguous(), vals_t[:M].contiguous())

    tile_shape = (int(state["tile_count"]),)
    if reuse_scratch:
        tile_start_t = _scratch_tensor("tile_start", tile_shape, device=device, dtype=torch.int32, zero=True)
        tile_end_t = _scratch_tensor("tile_end", tile_shape, device=device, dtype=torch.int32, zero=True)
    else:
        tile_start_t = torch.zeros(tile_shape, device=device, dtype=torch.int32)
        tile_end_t = torch.zeros(tile_shape, device=device, dtype=torch.int32)
    _LAUNCH_CACHE.launch(
        get_tile_bin_edges_kernel,
        dim=M,
        inputs=[sorted_keys_t, M],
        outputs=[tile_start_t, tile_end_t],
        device=str(state["device_str"]),
        stream=stream,
    )

    prepared = PreparedVisibility(
        xys=state["xys_t"],  # type: ignore[arg-type]
        conic=state["conic_t"],  # type: ignore[arg-type]
        rho=state["rho_t"],  # type: ignore[arg-type]
        num_tiles_hit=state["num_tiles_hit_t"],  # type: ignore[arg-type]
        tile_start=tile_start_t,
        tile_end=tile_end_t,
        sorted_vals=sorted_vals_t,
        width=int(state["width"]),
        height=int(state["height"]),
        tile_size=int(state["tile_size"]),
        tiles_x=int(state["tiles_x"]),
        tiles_y=int(state["tiles_y"]),
        tile_count=int(state["tile_count"]),
    )
    state["prepared"] = prepared
    state["tile_start_wp"] = _wrap_output_tensor(tile_start_t, wp.int32)
    state["tile_end_wp"] = _wrap_output_tensor(tile_end_t, wp.int32)
    state["sorted_vals_wp"] = _wrap_output_tensor(sorted_vals_t, wp.int32)
    return prepared


class WarpGSplatRasterizeValues(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        means: Tensor,
        quat: Tensor,
        scale: Tensor,
        values: Tensor,
        opacity: Tensor,
        background: Tensor,
        viewmat: Tensor,
        K: Tensor,
        width: int,
        height: int,
        cfg: RasterConfig,
        return_prepared: bool,
        background_is_zero: bool,
    ):
        N = _validate_render_inputs(means, quat, scale, opacity, viewmat, K)
        _assert_cuda_float32_contiguous(values, "values")
        if values.ndim != 2 or values.shape[0] != N:
            raise ValueError(f"values must be [N,C], got {tuple(values.shape)}")
        _assert_cuda_float32_contiguous(background, "background")
        if background.ndim != 1 or background.shape[0] != values.shape[1]:
            raise ValueError(f"background must be [C], got {tuple(background.shape)}")

        use_reference_backward = cfg.backward_impl == "reference"
        use_hybrid_backward = cfg.backward_impl == "hybrid"
        needs_projection_grads = any(t.requires_grad for t in (means, quat, scale, viewmat, K))
        requires_projection_tape = use_hybrid_backward and needs_projection_grads
        requires_tape = (
            (not use_reference_backward)
            and (not use_hybrid_backward)
            and cfg.record_projection_and_rasterize_on_tape
            and any(t.requires_grad for t in (means, quat, scale, values, opacity, viewmat, K))
        )
        state_requires_grad = requires_tape or requires_projection_tape
        state = _allocate_visibility_state(
            means,
            quat,
            scale,
            viewmat,
            K,
            int(width),
            int(height),
            cfg,
            requires_grad=state_requires_grad,
            reuse_scratch=not state_requires_grad and not return_prepared,
        )
        device = means.device
        stream = _warp_stream(device)
        channels = int(values.shape[1])
        antialiased = cfg.rasterize_mode == "antialiased"
        rasterize_values_kernel, rasterize_values_backward_kernel, value_vec_dtype = specialize_raster_kernels(
            channels,
            tile_size=int(state["tile_size"]),
            antialiased=antialiased,
            background_is_zero=background_is_zero,
        )
        out_values_t = torch.zeros((height * width, channels), device=device, dtype=torch.float32)
        final_T_t = torch.empty(height * width, device=device, dtype=torch.float32)
        stop_idx_t = torch.empty(height * width, device=device, dtype=torch.int32)
        out_values_wp = _wrap_output_tensor(out_values_t, value_vec_dtype, requires_grad=requires_tape)
        final_T_wp = _wrap_output_tensor(final_T_t, wp.float32, requires_grad=False)
        stop_idx_wp = _wrap_output_tensor(stop_idx_t, wp.int32, requires_grad=False)
        wp_values = _wrap_input_tensor(values, value_vec_dtype, requires_grad=requires_tape)
        wp_opacity = _wrap_input_tensor(opacity, wp.float32, requires_grad=requires_tape)
        wp_background = _wrap_input_tensor(background.view(-1), wp.float32, requires_grad=False)

        tape = wp.Tape() if requires_tape else None
        projection_tape = wp.Tape() if requires_projection_tape else None
        with wp.ScopedStream(stream, sync_enter=False, sync_exit=False):
            if tape is not None:
                with tape:
                    _launch_project(state, cfg, stream=stream)
            elif projection_tape is not None:
                with projection_tape:
                    _launch_project(state, cfg, stream=stream)
            else:
                _launch_project(state, cfg, stream=stream)

            prepared = _prepare_sorted_intersections(state, cfg, stream=stream)

            ctx.backward_impl = cfg.backward_impl
            ctx.cfg = cfg
            ctx.width = int(width)
            ctx.height = int(height)
            ctx.reference_tensors = (
                means.detach(),
                quat.detach(),
                scale.detach(),
                values.detach(),
                opacity.detach(),
                background.detach(),
                viewmat.detach(),
                K.detach(),
            )
            ctx.reference_requires_grad = (
                means.requires_grad,
                quat.requires_grad,
                scale.requires_grad,
                values.requires_grad,
                opacity.requires_grad,
                viewmat.requires_grad,
                K.requires_grad,
            )

            if prepared.intersection_count <= 0:
                out_t = out_values_t.view(height, width, channels)
                if not background_is_zero:
                    out_t = out_t + background.view(1, 1, channels)
                ctx.tape = None
                ctx.saved = None
                if not return_prepared:
                    return out_t
                aux_outputs = (
                    prepared.xys,
                    prepared.conic,
                    prepared.rho,
                    prepared.num_tiles_hit,
                    prepared.tile_start,
                    prepared.tile_end,
                    prepared.sorted_vals,
                )
                ctx.mark_non_differentiable(*aux_outputs)
                return (out_t, *aux_outputs)

            raster_inputs = [
                state["tile_start_wp"],
                state["tile_end_wp"],
                state["sorted_vals_wp"],
                state["xys_wp"],
                state["conic_wp"],
                state["rho_wp"],
                wp_values,
                wp_opacity,
                int(width),
                int(height),
                int(state["tiles_x"]),
                float(cfg.alpha_min),
                float(cfg.transmittance_eps),
                float(cfg.clamp_alpha_max),
                wp_background,
            ]
            if tape is not None:
                with tape:
                    wp.launch(
                        rasterize_values_kernel,
                        dim=int(height) * int(width),
                        inputs=raster_inputs,
                        outputs=[out_values_wp, final_T_wp, stop_idx_wp],
                        device=str(state["device_str"]),
                        stream=stream,
                        block_dim=_warp_block_dim(),
                    )
            else:
                wp.launch(
                    rasterize_values_kernel,
                    dim=int(height) * int(width),
                    inputs=raster_inputs,
                    outputs=[out_values_wp, final_T_wp, stop_idx_wp],
                    device=str(state["device_str"]),
                    stream=stream,
                    block_dim=_warp_block_dim(),
                )

        ctx.tape = tape
        ctx.projection_tape = projection_tape
        ctx.saved = (
            state["means_wp"],
            state["quat_wp"],
            state["scale_wp"],
            state["viewmat_wp"],
            state["K_wp"],
            wp_values,
            wp_opacity,
            state["xys_wp"],
            state["conic_wp"],
            state["rho_wp"],
            state["tile_start_wp"],
            state["tile_end_wp"],
            state["sorted_vals_wp"],
            out_values_wp,
            final_T_wp,
            stop_idx_wp,
            background_is_zero,
        )
        ctx.value_shape = tuple(values.shape)
        out_t = out_values_t.view(height, width, channels)
        if not return_prepared:
            return out_t
        aux_outputs = (
            prepared.xys,
            prepared.conic,
            prepared.rho,
            prepared.num_tiles_hit,
            prepared.tile_start,
            prepared.tile_end,
            prepared.sorted_vals,
        )
        ctx.mark_non_differentiable(*aux_outputs)
        return (out_t, *aux_outputs)

    @staticmethod
    def backward(ctx, grad_out: Tensor, *_grad_aux):  # type: ignore[override]
        if getattr(ctx, "backward_impl", "warp_tape") == "reference":
            if grad_out is None:
                return (None,) * 13
            if not grad_out.is_contiguous():
                grad_out = grad_out.contiguous()

            (
                means_src,
                quat_src,
                scale_src,
                values_src,
                opacity_src,
                background_src,
                viewmat_src,
                K_src,
            ) = ctx.reference_tensors
            (
                means_req,
                quat_req,
                scale_req,
                values_req,
                opacity_req,
                viewmat_req,
                K_req,
            ) = ctx.reference_requires_grad

            def _clone_for_grad(src: Tensor, requires_grad: bool) -> Tensor:
                out = src.clone().detach().contiguous()
                out.requires_grad_(requires_grad)
                return out

            with torch.enable_grad():
                means = _clone_for_grad(means_src, means_req)
                quat = _clone_for_grad(quat_src, quat_req)
                scale = _clone_for_grad(scale_src, scale_req)
                values = _clone_for_grad(values_src, values_req)
                opacity = _clone_for_grad(opacity_src, opacity_req)
                viewmat = _clone_for_grad(viewmat_src, viewmat_req)
                K = _clone_for_grad(K_src, K_req)
                background = background_src.clone().detach().contiguous()
                out = render_values_reference(
                    means=means,
                    quat=quat,
                    scale=scale,
                    values=values,
                    opacity=opacity,
                    viewmat=viewmat,
                    K=K,
                    width=int(ctx.width),
                    height=int(ctx.height),
                    cfg=ctx.cfg,
                    background=background,
                )
                grad_inputs: list[Tensor] = []
                for tensor, required in (
                    (means, means_req),
                    (quat, quat_req),
                    (scale, scale_req),
                    (values, values_req),
                    (opacity, opacity_req),
                    (viewmat, viewmat_req),
                    (K, K_req),
                ):
                    if required:
                        grad_inputs.append(tensor)
                grads = torch.autograd.grad(
                    outputs=out,
                    inputs=grad_inputs,
                    grad_outputs=grad_out,
                    allow_unused=True,
                )

            grad_iter = iter(grads)

            def _next(required: bool):
                return next(grad_iter) if required else None

            return (
                _next(means_req),
                _next(quat_req),
                _next(scale_req),
                _next(values_req),
                _next(opacity_req),
                None,
                _next(viewmat_req),
                _next(K_req),
                None,
                None,
                None,
                None,
                None,
            )

        if getattr(ctx, "backward_impl", "warp_tape") == "hybrid":
            if ctx.saved is None:
                return (None,) * 13
            if grad_out is None:
                return (None,) * 13
            if not grad_out.is_contiguous():
                grad_out = grad_out.contiguous()

            (
                _wp_means,
                _wp_quat,
                _wp_scale,
                _wp_viewmat,
                _wp_K,
                wp_values,
                wp_opacity,
                xys_wp,
                conic_wp,
                rho_wp,
                tile_start_wp,
                tile_end_wp,
                sorted_vals_wp,
                _out_values,
                final_T_wp,
                stop_idx_wp,
                background_is_zero,
            ) = ctx.saved

            (
                means_src,
                quat_src,
                scale_src,
                values_src,
                opacity_src,
                background_src,
                viewmat_src,
                K_src,
            ) = ctx.reference_tensors
            (
                means_req,
                quat_req,
                scale_req,
                values_req,
                opacity_req,
                viewmat_req,
                K_req,
            ) = ctx.reference_requires_grad

            channels = int(ctx.value_shape[1])
            antialiased = ctx.cfg.rasterize_mode == "antialiased"
            _, rasterize_values_backward_kernel, value_vec_dtype = specialize_raster_kernels(
                channels,
                tile_size=int(ctx.cfg.tile_size),
                antialiased=antialiased,
                background_is_zero=background_is_zero,
            )
            gaussian_count = int(values_src.shape[0])
            device = grad_out.device
            grad_out_vec_t = grad_out.reshape(-1, channels).contiguous()
            grad_xys_t = torch.zeros((gaussian_count, 2), device=device, dtype=torch.float32)
            grad_conic_t = torch.zeros((gaussian_count, 3), device=device, dtype=torch.float32)
            grad_rho_t = torch.zeros((gaussian_count,), device=device, dtype=torch.float32)
            grad_values_t = torch.zeros((gaussian_count, channels), device=device, dtype=torch.float32)
            grad_opacity_t = torch.zeros((gaussian_count,), device=device, dtype=torch.float32)

            stream = _warp_stream(device)
            with wp.ScopedStream(stream, sync_enter=False, sync_exit=False):
                wp.launch(
                    rasterize_values_backward_kernel,
                    dim=int(ctx.width) * int(ctx.height),
                    inputs=[
                        tile_start_wp,
                        tile_end_wp,
                        sorted_vals_wp,
                        xys_wp,
                        conic_wp,
                        rho_wp,
                        wp_values,
                        wp_opacity,
                        int(ctx.width),
                        int(ctx.height),
                        int((int(ctx.width) + int(ctx.cfg.tile_size) - 1) // int(ctx.cfg.tile_size)),
                        float(ctx.cfg.alpha_min),
                        float(ctx.cfg.transmittance_eps),
                        float(ctx.cfg.clamp_alpha_max),
                        _wrap_input_tensor(background_src.view(-1).contiguous(), wp.float32, requires_grad=False),
                        final_T_wp,
                        stop_idx_wp,
                        _wrap_input_tensor(grad_out_vec_t, value_vec_dtype, requires_grad=False),
                    ],
                    outputs=[
                        _wrap_output_tensor(grad_xys_t, wp.vec2f),
                        _wrap_output_tensor(grad_conic_t, wp.vec3f),
                        _wrap_output_tensor(grad_rho_t, wp.float32),
                        _wrap_output_tensor(grad_values_t, value_vec_dtype),
                        _wrap_output_tensor(grad_opacity_t, wp.float32),
                    ],
                    device=str(device),
                    stream=stream,
                    block_dim=_warp_block_dim(),
                )

            values_grad = grad_values_t.view(ctx.value_shape) if values_req else None
            opacity_grad = grad_opacity_t if opacity_req else None

            def _clone_for_grad(src: Tensor, requires_grad: bool) -> Tensor:
                out = src.clone().detach().contiguous()
                out.requires_grad_(requires_grad)
                return out

            projection_tape = getattr(ctx, "projection_tape", None)

            def _warp_grad_or_none(wparr):
                if getattr(wparr, "grad", None) is None:
                    return None
                return wp.to_torch(wparr.grad)

            if projection_tape is not None:
                projection_tape.backward(
                    grads={
                        xys_wp: _wrap_input_tensor(grad_xys_t.contiguous(), wp.vec2f, requires_grad=False),
                        conic_wp: _wrap_input_tensor(grad_conic_t.contiguous(), wp.vec3f, requires_grad=False),
                        rho_wp: _wrap_input_tensor(grad_rho_t.contiguous(), wp.float32, requires_grad=False),
                    }
                )

            means_grad = _warp_grad_or_none(_wp_means) if means_req else None
            quat_grad = _warp_grad_or_none(_wp_quat) if quat_req else None
            scale_grad = _warp_grad_or_none(_wp_scale) if scale_req else None
            viewmat_grad = _warp_grad_or_none(_wp_viewmat) if viewmat_req else None
            K_grad = _warp_grad_or_none(_wp_K) if K_req else None

            return (
                means_grad,
                quat_grad,
                scale_grad,
                values_grad,
                opacity_grad,
                None,
                viewmat_grad,
                K_grad,
                None,
                None,
                None,
                None,
                None,
            )

        if ctx.saved is None or ctx.tape is None:
            return (None,) * 13

        (
            wp_means,
            wp_quat,
            wp_scale,
            wp_viewmat,
            wp_K,
            wp_values,
            wp_opacity,
            _xys,
            _conic,
            _rho,
            _tile_start,
            _tile_end,
            _sorted_vals,
            out_values,
            _final_T,
            _stop_idx,
            _background_is_zero,
        ) = ctx.saved

        if grad_out is None:
            return (None,) * 13
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()

        channels = int(ctx.value_shape[1])
        value_vec_dtype = wp.types.vector(length=channels, dtype=wp.float32)
        wp_grad = _wrap_input_tensor(grad_out.reshape(-1, channels).contiguous(), value_vec_dtype, requires_grad=False)
        ctx.tape.backward(grads={out_values: wp_grad})

        def _grad_or_none(wparr):
            if getattr(wparr, "grad", None) is None:
                return None
            return wp.to_torch(wparr.grad)

        values_grad = _grad_or_none(wp_values)
        if values_grad is not None:
            values_grad = values_grad.view(ctx.value_shape)

        return (
            _grad_or_none(wp_means),
            _grad_or_none(wp_quat),
            _grad_or_none(wp_scale),
            values_grad,
            _grad_or_none(wp_opacity),
            None,
            _grad_or_none(wp_viewmat),
            _grad_or_none(wp_K),
            None,
            None,
            None,
            None,
            None,
        )


def prepare_visibility_warp(
    means: Tensor,
    quat: Tensor,
    scale: Tensor,
    viewmat: Tensor,
    K: Tensor,
    width: int,
    height: int,
    cfg: RasterConfig | None = None,
    requires_grad: bool = False,
    active_count: int | None = None,
) -> PreparedVisibility:
    if cfg is None:
        cfg = RasterConfig()
    if active_count is not None:
        count = max(0, min(int(active_count), int(means.shape[0])))
        means = means[:count]
        quat = quat[:count]
        scale = scale[:count]
    _validate_render_inputs(
        means=means,
        quat=quat,
        scale=scale,
        opacity=torch.zeros(means.shape[0], device=means.device, dtype=means.dtype),
        viewmat=viewmat,
        K=K,
    )
    state = _allocate_visibility_state(
        means,
        quat,
        scale,
        viewmat,
        K,
        int(width),
        int(height),
        cfg,
        requires_grad=requires_grad,
        reuse_scratch=False,
    )
    stream = _warp_stream(means.device)
    with wp.ScopedStream(stream, sync_enter=False, sync_exit=False):
        _launch_project(state, cfg, stream=stream)
        return _prepare_sorted_intersections(state, cfg, stream=stream)


def render_values_warp(
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
    return_prepared: bool = False,
    active_count: int | None = None,
) -> Tensor | tuple[Tensor, PreparedVisibility]:
    if cfg is None:
        cfg = RasterConfig()
    # Warp kernels require FP32 for all inputs -- they have no mixed-precision
    # support.  Upcast BF16 values/opacity/background at the entry point.
    # This is a simple fallback; the Helion backend is the performance-critical
    # path where BF16 savings matter.
    if values.dtype != torch.float32:
        values = values.float()
    if opacity.dtype != torch.float32:
        opacity = opacity.float()
    background_is_zero = background is None
    if background is None:
        background = torch.zeros(values.shape[1], device=values.device, dtype=values.dtype)
    elif background.dtype != torch.float32:
        background = background.float()
    if active_count is not None:
        count = max(0, min(int(active_count), int(means.shape[0])))
        means = means[:count]
        quat = quat[:count]
        scale = scale[:count]
        values = values[:count]
        opacity = opacity[:count]

    if return_prepared:
        out, xys, conic, rho, num_tiles_hit, tile_start, tile_end, sorted_vals = WarpGSplatRasterizeValues.apply(
            means,
            quat,
            scale,
            values,
            opacity,
            background,
            viewmat,
            K,
            int(width),
            int(height),
            cfg,
            True,
            background_is_zero,
        )
        tile_size = int(cfg.tile_size)
        tiles_x, tiles_y, tile_count = estimate_tiles(int(width), int(height), tile_size)
        prepared = PreparedVisibility(
            xys=xys,
            conic=conic,
            rho=rho,
            num_tiles_hit=num_tiles_hit,
            tile_start=tile_start,
            tile_end=tile_end,
            sorted_vals=sorted_vals,
            width=int(width),
            height=int(height),
            tile_size=tile_size,
            tiles_x=tiles_x,
            tiles_y=tiles_y,
            tile_count=tile_count,
        )
        return out, prepared

    return WarpGSplatRasterizeValues.apply(
        means,
        quat,
        scale,
        values,
        opacity,
        background,
        viewmat,
        K,
        int(width),
        int(height),
        cfg,
        False,
        background_is_zero,
    )


def render_stats_prepared_warp(
    prepared: PreparedVisibility,
    opacity: Tensor,
    *,
    cfg: RasterConfig | None = None,
    residual_map: Tensor | None = None,
    screen_error_bins: int = 4,
    include_details: bool = True,
    active_count: int | None = None,
) -> dict[str, Tensor]:
    if cfg is None:
        cfg = RasterConfig()
    if active_count is not None:
        opacity = opacity[: int(active_count)]
    # Warp stats kernel requires FP32; upcast BF16 opacity at the entry point.
    if opacity.dtype != torch.float32:
        opacity = opacity.float()
    _assert_cuda_float32_contiguous(opacity, "opacity")
    _assert_1d(opacity, "opacity")
    if int(opacity.shape[0]) != prepared.gaussian_count:
        raise ValueError(f"opacity length mismatch: expected {prepared.gaussian_count}, got {opacity.shape[0]}")

    meta = prepared.meta()
    if not include_details:
        return meta

    contrib = torch.zeros(prepared.gaussian_count, device=prepared.device, dtype=torch.float32)
    trans = torch.zeros_like(contrib)
    hits = torch.zeros_like(contrib)
    residual = torch.zeros_like(contrib)
    bins = max(int(screen_error_bins), 1)
    error_map = torch.zeros(prepared.gaussian_count, bins * bins, device=prepared.device, dtype=torch.float32)
    if prepared.intersection_count <= 0:
        return {
            "contrib": contrib,
            "transmittance": trans,
            "hits": hits,
            "residual": residual,
            "error_map": error_map,
            **meta,
        }

    if residual_map is None:
        residual_map = torch.zeros(prepared.height, prepared.width, device=prepared.device, dtype=torch.float32)
    else:
        _assert_cuda_float32_contiguous(residual_map, "residual_map")
        if tuple(residual_map.shape) != (prepared.height, prepared.width):
            raise ValueError(
                f"residual_map must have shape {(prepared.height, prepared.width)}, got {tuple(residual_map.shape)}"
            )
    sorted_min = int(prepared.sorted_vals.min().item())
    sorted_max = int(prepared.sorted_vals.max().item())
    if sorted_min < 0 or sorted_max >= prepared.gaussian_count:
        raise RuntimeError(
            "PreparedVisibility contains out-of-range gaussian ids for density stats: "
            f"min_gid={sorted_min}, max_gid={sorted_max}, gaussian_count={prepared.gaussian_count}, "
            f"intersection_count={prepared.intersection_count}"
        )
    tile_start_min = int(prepared.tile_start.min().item()) if prepared.tile_start.numel() > 0 else 0
    tile_end_max = int(prepared.tile_end.max().item()) if prepared.tile_end.numel() > 0 else 0
    if tile_start_min < 0 or tile_end_max > prepared.intersection_count:
        raise RuntimeError(
            "PreparedVisibility contains out-of-range tile bins for density stats: "
            f"tile_start_min={tile_start_min}, tile_end_max={tile_end_max}, "
            f"intersection_count={prepared.intersection_count}"
        )

    device_str = _ensure_warp_ready(prepared.device)
    stream = _warp_stream(prepared.device)
    stats_kernel = specialize_visibility_stats_kernel(
        antialiased=cfg.rasterize_mode == "antialiased",
        error_bins_x=bins,
        error_bins_y=bins,
    )
    opacity_stats = opacity.detach()
    residual_stats = residual_map.detach()
    with wp.ScopedStream(stream, sync_enter=False, sync_exit=False):
        wp.launch(
            stats_kernel,
            dim=prepared.width * prepared.height,
            inputs=[
                prepared.tile_start,
                prepared.tile_end,
                prepared.sorted_vals,
                prepared.xys,
                prepared.conic,
                prepared.rho,
                opacity_stats,
                prepared.width,
                prepared.height,
                prepared.tiles_x,
                prepared.tile_size,
                residual_stats.view(-1),
                float(cfg.alpha_min),
                float(cfg.transmittance_eps),
                float(cfg.clamp_alpha_max),
            ],
            outputs=[contrib, trans, hits, residual, error_map.view(-1)],
            device=device_str,
            stream=stream,
            block_dim=_warp_block_dim(),
        )

    return {
        "contrib": contrib,
        "transmittance": trans,
        "hits": hits,
        "residual": residual,
        "error_map": error_map,
        **meta,
    }


def render_stats_warp(
    opacity: Tensor,
    *,
    means: Tensor | None = None,
    quat: Tensor | None = None,
    scale: Tensor | None = None,
    viewmat: Tensor | None = None,
    K: Tensor | None = None,
    width: int | None = None,
    height: int | None = None,
    cfg: RasterConfig | None = None,
    residual_map: Tensor | None = None,
    screen_error_bins: int = 4,
    include_details: bool = True,
    prepared: PreparedVisibility | None = None,
    active_count: int | None = None,
) -> dict[str, Tensor]:
    if cfg is None:
        cfg = RasterConfig()
    if active_count is not None:
        count = max(0, min(int(active_count), int(opacity.shape[0])))
        opacity = opacity[:count]
        if means is not None:
            means = means[:count]
        if quat is not None:
            quat = quat[:count]
        if scale is not None:
            scale = scale[:count]
    if prepared is None:
        if (
            means is None
            or quat is None
            or scale is None
            or viewmat is None
            or K is None
            or width is None
            or height is None
        ):
            raise ValueError("render_stats_warp requires either a prepared visibility object or full render inputs.")
        _validate_render_inputs(means, quat, scale, opacity, viewmat, K)
        state = _allocate_visibility_state(
            means=means,
            quat=quat,
            scale=scale,
            viewmat=viewmat,
            K=K,
            width=int(width),
            height=int(height),
            cfg=cfg,
            requires_grad=False,
            reuse_scratch=True,
        )
        stream = _warp_stream(means.device)
        with wp.ScopedStream(stream, sync_enter=False, sync_exit=False):
            _launch_project(state, cfg, stream=stream)
            prepared = _prepare_sorted_intersections(state, cfg, stream=stream)
    return render_stats_prepared_warp(
        prepared,
        opacity,
        cfg=cfg,
        residual_map=residual_map,
        screen_error_bins=screen_error_bins,
        include_details=include_details,
        active_count=active_count,
    )


def render_visibility_meta_warp(
    *,
    prepared: PreparedVisibility | None = None,
    means: Tensor | None = None,
    quat: Tensor | None = None,
    scale: Tensor | None = None,
    viewmat: Tensor | None = None,
    K: Tensor | None = None,
    width: int | None = None,
    height: int | None = None,
    cfg: RasterConfig | None = None,
    active_count: int | None = None,
) -> dict[str, Tensor]:
    if cfg is None:
        cfg = RasterConfig()
    if prepared is None:
        if active_count is not None:
            count = max(0, min(int(active_count), int(means.shape[0]) if means is not None else 0))
            if means is not None:
                means = means[:count]
            if quat is not None:
                quat = quat[:count]
            if scale is not None:
                scale = scale[:count]
        if (
            means is None
            or quat is None
            or scale is None
            or viewmat is None
            or K is None
            or width is None
            or height is None
        ):
            raise ValueError(
                "render_visibility_meta_warp requires either a prepared visibility object or full render inputs."
            )
        state = _allocate_visibility_state(
            means=means,
            quat=quat,
            scale=scale,
            viewmat=viewmat,
            K=K,
            width=int(width),
            height=int(height),
            cfg=cfg,
            requires_grad=False,
            reuse_scratch=True,
        )
        stream = _warp_stream(means.device)
        with wp.ScopedStream(stream, sync_enter=False, sync_exit=False):
            _launch_project(state, cfg, stream=stream)
            prepared = _prepare_sorted_intersections(state, cfg, stream=stream)
    return prepared.meta()


def render_projection_meta_warp(
    *,
    means: Tensor,
    quat: Tensor,
    scale: Tensor,
    viewmat: Tensor,
    K: Tensor,
    width: int,
    height: int,
    cfg: RasterConfig | None = None,
    active_count: int | None = None,
) -> dict[str, int | bool | str | None]:
    if cfg is None:
        cfg = RasterConfig()
    if active_count is not None:
        count = max(0, min(int(active_count), int(means.shape[0])))
        means = means[:count]
        quat = quat[:count]
        scale = scale[:count]
    state = _allocate_visibility_state(
        means=means,
        quat=quat,
        scale=scale,
        viewmat=viewmat,
        K=K,
        width=int(width),
        height=int(height),
        cfg=cfg,
        requires_grad=False,
        reuse_scratch=True,
    )
    stream = _warp_stream(means.device)
    with wp.ScopedStream(stream, sync_enter=False, sync_exit=False):
        _launch_project(state, cfg, stream=stream)
    num_tiles_hit_t: Tensor = state["num_tiles_hit_t"]  # type: ignore[assignment]
    gaussian_count = int(state["gaussian_count"])
    visible_count = int((num_tiles_hit_t > 0).sum(dtype=torch.int64).item())
    intersection_count = int(num_tiles_hit_t.sum(dtype=torch.int64).item()) if gaussian_count > 0 else 0
    _use_warp_radix, chosen_mode, chosen_bytes, warp_sort_bytes, torch_sort_bytes = _select_sort_mode_and_bytes(
        intersection_count,
        cfg,
    )
    budget = cfg.max_sort_buffer_bytes
    return {
        "gaussian_count": gaussian_count,
        "visible_count": visible_count,
        "intersection_count": intersection_count,
        "tile_count": int(state["tile_count"]),
        "tiles_x": int(state["tiles_x"]),
        "tiles_y": int(state["tiles_y"]),
        "render_width": int(width),
        "render_height": int(height),
        "sort_mode": chosen_mode,
        "estimated_sort_buffer_bytes": int(chosen_bytes),
        "warp_sort_buffer_bytes": int(warp_sort_bytes),
        "torch_sort_buffer_bytes": int(torch_sort_bytes),
        "sort_buffer_budget_bytes": None if budget is None else int(budget),
        "sort_buffer_within_budget": bool(budget is None or chosen_bytes <= budget),
    }


def render_gaussians_warp(
    means: Tensor,
    quat: Tensor,
    scale: Tensor,
    color_r: Tensor,
    color_g: Tensor,
    color_b: Tensor,
    opacity: Tensor,
    viewmat: Tensor,
    K: Tensor,
    width: int,
    height: int,
    cfg: RasterConfig | None = None,
) -> Tensor:
    values = torch.stack((color_r, color_g, color_b), dim=-1).contiguous()
    background = None
    if cfg is not None:
        background = torch.tensor(cfg.background_rgb, device=values.device, dtype=values.dtype)
    return render_values_warp(
        means=means,
        quat=quat,
        scale=scale,
        values=values,
        opacity=opacity,
        viewmat=viewmat,
        K=K,
        width=width,
        height=height,
        cfg=cfg,
        background=background,
    )


__all__ = [
    "DataContracts",
    "PreparedVisibility",
    "RasterConfig",
    "estimate_intersections",
    "estimate_tiles",
    "estimate_buffer_bytes_for_example",
    "KERNEL_MAPPING_TABLE",
    "MERMAID_PORT_FLOWCHART",
    "WarpGSplatRasterizeValues",
    "prepare_visibility_warp",
    "render_values_warp",
    "render_projection_meta_warp",
    "render_visibility_meta_warp",
    "render_stats_prepared_warp",
    "render_stats_warp",
    "render_gaussians_warp",
]
