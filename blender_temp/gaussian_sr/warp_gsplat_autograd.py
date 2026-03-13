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
    project_gaussians_kernel,
    rasterize_values_kernel,
    rasterize_visibility_stats_kernel,
)
from .warp_runtime import require_warp, wp

_WARP_BLOCK_DIM = 256
_LOADED_KERNEL_MODULES: set[tuple[str, int]] = set()
_SCRATCH_TENSOR_CACHE: dict[tuple[str, tuple[int, ...], str, int | None, str], Tensor] = {}


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

    @property
    def device(self) -> torch.device:
        return self.xys.device

    @property
    def gaussian_count(self) -> int:
        return int(self.xys.shape[0])

    @property
    def intersection_count(self) -> int:
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
        key = (kernel.key, device, _WARP_BLOCK_DIM)
        recordable_inputs = [self._recordable_arg(value) for value in inputs]
        recordable_outputs = [self._recordable_arg(value) for value in outputs]
        cmd = self._commands.get(key)
        if cmd is None:
            cmd = wp.launch(
                kernel,
                dim=dim,
                inputs=recordable_inputs,
                outputs=recordable_outputs,
                device=device,
                block_dim=_WARP_BLOCK_DIM,
                record_cmd=True,
            )
            self._commands[key] = cmd
        else:
            cmd.set_dim(dim)
            for index, value in enumerate((*recordable_inputs, *recordable_outputs)):
                cmd.set_param_at_index(index, value)
        cmd.launch(stream=stream)


_LAUNCH_CACHE = _KernelLaunchCache()


def _device_str(device: torch.device) -> str:
    if device.type != "cuda":
        raise ValueError("Warp gsplat renderer requires CUDA tensors.")
    return f"cuda:{device.index}" if device.index is not None else "cuda:0"


def _warp_stream(device: torch.device):
    return wp.stream_from_torch(device) if device.type == "cuda" else None


def _ensure_warp_ready(device: torch.device) -> str:
    require_warp()
    device_str = _device_str(device)
    key = (device_str, _WARP_BLOCK_DIM)
    if key not in _LOADED_KERNEL_MODULES:
        wp.load_module(module=warp_gsplat_kernel_module, device=device_str, block_dim=_WARP_BLOCK_DIM)
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
    inputs = [
        state["means_wp"],
        state["quat_wp"],
        state["scale_wp"],
        state["viewmat_wp"],
        state["K_wp"],
        int(state["width"]),
        int(state["height"]),
        int(state["tile_size"]),
        int(state["tiles_x"]),
        int(state["tiles_y"]),
        float(cfg.near_plane),
        float(cfg.far_plane),
        float(cfg.eps2d),
        float(cfg.radius_clip),
        float(cfg.depth_scale),
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
    if bool(state["requires_grad"]):
        wp.launch(
            project_gaussians_kernel,
            dim=int(state["gaussian_count"]),
            inputs=inputs,
            outputs=outputs,
            device=str(state["device_str"]),
            stream=stream,
            block_dim=_WARP_BLOCK_DIM,
        )
    else:
        _LAUNCH_CACHE.launch(
            project_gaussians_kernel,
            dim=int(state["gaussian_count"]),
            inputs=[
                state["means_t"],
                state["quat_t"],
                state["scale_t"],
                state["viewmat_t"],
                state["K_t"],
                int(state["width"]),
                int(state["height"]),
                int(state["tile_size"]),
                int(state["tiles_x"]),
                int(state["tiles_y"]),
                float(cfg.near_plane),
                float(cfg.far_plane),
                float(cfg.eps2d),
                float(cfg.radius_clip),
                float(cfg.depth_scale),
            ],
            outputs=[
                state["xys_t"],
                state["conic_t"],
                state["rho_t"],
                state["radius_t"],
                state["num_tiles_hit_t"],
                state["tile_min_t"],
                state["tile_max_t"],
                state["depth_key_t"],
            ],
            device=str(state["device_str"]),
            stream=stream,
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

    use_warp_radix = cfg.sort_mode in ("auto", "warp_radix")
    warp_sort_bytes = _sort_buffer_bytes(M, use_warp_radix=True)
    torch_sort_bytes = _sort_buffer_bytes(M, use_warp_radix=False)
    budget = cfg.max_sort_buffer_bytes
    if budget is not None:
        if use_warp_radix and warp_sort_bytes > budget and cfg.sort_mode == "auto":
            use_warp_radix = False
        chosen_bytes = warp_sort_bytes if use_warp_radix else torch_sort_bytes
        if chosen_bytes > budget:
            gib = float(chosen_bytes) / float(1024**3)
            budget_gib = float(budget) / float(1024**3)
            raise RuntimeError(
                "Projected intersection sort buffer exceeds configured budget: "
                f"M={M}, sort_mode={'warp_radix' if use_warp_radix else 'torch_sort'}, "
                f"estimated_sort_bytes={chosen_bytes} ({gib:.2f} GiB), "
                f"budget={budget} ({budget_gib:.2f} GiB). "
                "Reduce projected footprints with radius_clip_px, increase anchor_stride, "
                "use fewer views per step, or disable final-stage density control."
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
    ):
        N = _validate_render_inputs(means, quat, scale, opacity, viewmat, K)
        _assert_cuda_float32_contiguous(values, "values")
        if values.ndim != 2 or values.shape[0] != N:
            raise ValueError(f"values must be [N,C], got {tuple(values.shape)}")
        _assert_cuda_float32_contiguous(background, "background")
        if background.ndim != 1 or background.shape[0] != values.shape[1]:
            raise ValueError(f"background must be [C], got {tuple(background.shape)}")

        requires_tape = cfg.record_projection_and_rasterize_on_tape and any(
            t.requires_grad for t in (means, quat, scale, values, opacity, viewmat, K)
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
            requires_grad=requires_tape,
            reuse_scratch=not requires_tape and not return_prepared,
        )
        device = means.device
        stream = _warp_stream(device)
        channels = int(values.shape[1])
        out_values_t = torch.zeros(height * width * channels, device=device, dtype=torch.float32)
        out_values_wp = _wrap_output_tensor(out_values_t, wp.float32, requires_grad=requires_tape)
        wp_values = _wrap_input_tensor(values.view(-1), wp.float32, requires_grad=requires_tape)
        wp_opacity = _wrap_input_tensor(opacity, wp.float32, requires_grad=requires_tape)
        wp_background = _wrap_input_tensor(background.view(-1), wp.float32, requires_grad=False)

        tape = wp.Tape() if requires_tape else None
        with wp.ScopedStream(stream, sync_enter=False, sync_exit=False):
            if tape is not None:
                with tape:
                    _launch_project(state, cfg, stream=stream)
            else:
                _launch_project(state, cfg, stream=stream)

            prepared = _prepare_sorted_intersections(state, cfg, stream=stream)

            if prepared.intersection_count <= 0:
                out_t = out_values_t.view(height, width, channels) + background.view(1, 1, channels)
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

            antialiased = 1 if cfg.rasterize_mode == "antialiased" else 0
            raster_inputs = [
                state["tile_start_wp"],
                state["tile_end_wp"],
                state["sorted_vals_wp"],
                state["xys_wp"],
                state["conic_wp"],
                state["rho_wp"],
                wp_values,
                wp_opacity,
                channels,
                int(width),
                int(height),
                int(state["tiles_x"]),
                int(state["tile_size"]),
                float(cfg.alpha_min),
                float(cfg.transmittance_eps),
                float(cfg.clamp_alpha_max),
                int(antialiased),
                wp_background,
            ]
            if tape is not None:
                with tape:
                    wp.launch(
                        rasterize_values_kernel,
                        dim=height * width,
                        inputs=raster_inputs,
                        outputs=[out_values_wp],
                        device=str(state["device_str"]),
                        stream=stream,
                        block_dim=_WARP_BLOCK_DIM,
                    )
            else:
                wp.launch(
                    rasterize_values_kernel,
                    dim=height * width,
                    inputs=raster_inputs,
                    outputs=[out_values_wp],
                    device=str(state["device_str"]),
                    stream=stream,
                    block_dim=_WARP_BLOCK_DIM,
                )

        ctx.tape = tape
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
        if ctx.saved is None or ctx.tape is None:
            return (None,) * 12

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
        ) = ctx.saved

        if grad_out is None:
            return (None,) * 12
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()

        wp_grad = _wrap_input_tensor(grad_out.view(-1), wp.float32, requires_grad=False)
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
) -> PreparedVisibility:
    if cfg is None:
        cfg = RasterConfig()
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
) -> Tensor | tuple[Tensor, PreparedVisibility]:
    if cfg is None:
        cfg = RasterConfig()
    if background is None:
        background = torch.zeros(values.shape[1], device=values.device, dtype=values.dtype)

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
    )


def render_stats_prepared_warp(
    prepared: PreparedVisibility,
    opacity: Tensor,
    *,
    cfg: RasterConfig | None = None,
    residual_map: Tensor | None = None,
    screen_error_bins: int = 4,
    include_details: bool = True,
) -> dict[str, Tensor]:
    if cfg is None:
        cfg = RasterConfig()
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

    device_str = _ensure_warp_ready(prepared.device)
    stream = _warp_stream(prepared.device)
    antialiased = 1 if cfg.rasterize_mode == "antialiased" else 0
    with wp.ScopedStream(stream, sync_enter=False, sync_exit=False):
        _LAUNCH_CACHE.launch(
            rasterize_visibility_stats_kernel,
            dim=prepared.width * prepared.height,
            inputs=[
                prepared.tile_start,
                prepared.tile_end,
                prepared.sorted_vals,
                prepared.xys,
                prepared.conic,
                prepared.rho,
                opacity,
                prepared.width,
                prepared.height,
                prepared.tiles_x,
                prepared.tile_size,
                residual_map.view(-1),
                bins,
                bins,
                float(cfg.alpha_min),
                float(cfg.transmittance_eps),
                float(cfg.clamp_alpha_max),
                int(antialiased),
            ],
            outputs=[contrib, trans, hits, residual, error_map.view(-1)],
            device=device_str,
            stream=stream,
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
) -> dict[str, Tensor]:
    if cfg is None:
        cfg = RasterConfig()
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
) -> dict[str, Tensor]:
    if cfg is None:
        cfg = RasterConfig()
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
    "render_visibility_meta_warp",
    "render_stats_prepared_warp",
    "render_stats_warp",
    "render_gaussians_warp",
]
