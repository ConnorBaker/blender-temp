# Warp + PyTorch port of gsplat-style Gaussian rasterization kernels

## Executive summary

This report distills how to implement (and port) the core gsplat/ContinuousSR Gaussian rasterization pipeline **entirely in NVIDIA Warp kernels plus Warp utilities**, with **PyTorch handling parameters and optimization**, under the assumption that **camera intrinsics/extrinsics are known and fixed**. It maps the canonical gsplat stages—**projection (EWA), tile-intersection enumeration, sort-by-(tile, depth), tile-bin edge construction, and per-pixel alpha compositing**—to a concrete Warp execution plan and an implementable code design. The design follows the gsplat contract that *binning/sorting are nondifferentiable* while projection and rasterization are differentiable w.r.t. the continuous Gaussian parameters. citeturn13view0turn13view1turn12view0turn11view3turn11view5

A key engineering choice is how to replace gsplat’s CUDA kernels and helper ops without pulling in external vendor kernels: **Warp provides scan and radix-sort primitives** (notably `warp.utils.array_scan` and `warp.utils.radix_sort_pairs`) and zero-copy PyTorch interop. citeturn2view0turn0view0turn7view0 The resulting implementation records the differentiable parts (projection + rasterize) on `wp.Tape`, while treating **intersection enumeration + sorting + tile bin edges** as **constants** (no gradient), matching gsplat’s own “not differentiable” guarantees for those steps. citeturn13view1turn11view3turn12view0

A working scaffold that embodies this design (known poses, Warp radix sort, Warp tile-bin edges kernel, Warp rasterize kernel, PyTorch training loop) is provided here:

[Download updated warp_pytorch_posefree_gaussian_sr.py](sandbox:/mnt/data/warp_pytorch_posefree_gaussian_sr.py)

## Provenance and kernel decomposition from primary sources

The pipeline you want to port is explicitly described in gsplat’s documentation: `project_gaussians()` projects 3D Gaussians to 2D using an EWA/Jacobian approximation and returns screen-space means, conics, radii, and `num_tiles_hit`; `rasterize_gaussians()` then **groups gaussians by tile, sorts by increasing depth, and alpha-composites per pixel**, and is differentiable w.r.t. `xys`, `conics`, `colors`, and `opacity`. citeturn13view0turn12view0

The helper ops that make this fast—`bin_and_sort_gaussians`, `map_gaussian_to_intersects`, and `get_tile_bin_edges`—are explicitly documented as **not differentiable**. citeturn13view1turn11view3 In XingtongGe’s gsplat submodule (used by ContinuousSR/GaussianImage lineage), the Python binding confirms the same: `map_gaussian_to_intersects` and `bin_and_sort_gaussians` are “not differentiable,” and the reference implementation uses `torch.sort` on intersection IDs. citeturn11view3turn11view5

ContinuousSR itself invokes the same conceptual pipeline (in 2D form): it calls `project_gaussians_2d(...)` followed by `rasterize_gaussians_sum(...)` to produce an image, with explicit tile bounds and block sizes. citeturn11view2

On the CUDA side (nerfstudio-project/gsplat), the forward kernel shows the exact alpha/sigma computation and early termination pattern used in practice (compute `sigma`, `alpha = min(0.999, opac * exp(-sigma))`, update `T`, and terminate when transmittance becomes tiny). citeturn19view0turn19view1 The backward kernel shows a concrete VJP structure for conics and projected means (e.g., `v_sigma`, then `v_conic_local` and `v_xy_local`), and uses atomic accumulation for parameter grads. citeturn20view0turn20view1

These sources imply the minimal “port surface” you must replicate:

- **Projection kernel**: 3D → 2D EWA/Jacobian projection. citeturn13view0  
- **Intersection enumeration** (`map_gaussian_to_intersects`): produce packed keys `(tile_id << 32) | depth_key` and values `gaussian_id`. citeturn13view1turn11view3turn11view5  
- **Sort** by packed key and **derive tile bin edges**. citeturn12view0turn13view1turn11view5  
- **Rasterize** per pixel with alpha compositing; differentiable w.r.t. continuous params. citeturn12view0turn19view0turn19view1  

## Assumptions and PyTorch tensor contracts

This implementation plan assumes:

- **Poses known and fixed** (no pose estimation / BA). If poses are unknown, you must add a pose-parameterization and an outer loop (or a separate BA/SfM module); that is not implemented here. citeturn13view0  
- Images are already loaded as PyTorch tensors; any color management (linearization, exposure, etc.) is **unspecified** unless you provide it.

### Expected input tensors (explicit)

For **V** views and an LR resolution **H×W**:

- `images_lr`: `torch.float32`, CUDA, shape **[V, 3, H, W]**, range **[0,1]** (if your data is `[V, H, W, 3]`, transpose once at ingestion).
- `K`: `torch.float32`, CUDA, shape **[V, 3, 3]** (or per-view `fx,fy,cx,cy` extracted).
- `view`: `torch.float32`, CUDA, shape **[V, 4, 4]**, world→camera.

Scene Gaussians (**N** Gaussians):

- `means3d`: `torch.float32`, CUDA, shape **[N, 3]**
- `scales`: `torch.float32`, CUDA, shape **[N, 3]** (positive; typically via `exp(log_scales)`)
- `quats`: `torch.float32`, CUDA, shape **[N, 4]**, normalized quaternion `[w,x,y,z]` per gsplat convention. citeturn13view0  
- `opacity`: `torch.float32`, CUDA, shape **[N]** in **[0,1]** (typically via `sigmoid(opacity_logits)`)
- `colors`: `torch.float32` (or `float16`), CUDA, shape **[N, 3]**

### SoA layout recommendation

Even if PyTorch stores these as AoS `[N,3]`, treat them as **structure-of-arrays at the algorithm level**: do not pack mixed fields into structs for kernels that need random access. This aligns with gsplat’s separation of `xys`, `conics`, `colors`, `opacities`. citeturn12view0turn13view0

Memory-layout sketch (conceptual SoA):

```
means3d_x[N], means3d_y[N], means3d_z[N]
scales_x[N],  scales_y[N],  scales_z[N]
quat_w[N], quat_x[N], quat_y[N], quat_z[N]
opacity[N]
color_r[N], color_g[N], color_b[N]
```

Warp can consume either AoS vectors (vec3/vec4) or SoA, but SoA is often preferable when you later add culling, bucketing, or partial updates.

## Warp port plan with kernel mapping

### What must be nondifferentiable vs differentiable

gsplat makes explicit that:

- Projection is differentiable w.r.t. `(means3d, scales, quats)`. citeturn13view0  
- Rasterization is differentiable w.r.t. `(xys, conics, colors, opacity)`. citeturn12view0  
- Binning/sorting utilities are **not differentiable**. citeturn13view1turn11view3turn11view5  

A Warp+PyTorch port should mirror this by controlling what is recorded on `wp.Tape`:

- **Recorded on `wp.Tape`**: `project_gaussians`, `rasterize` (so grads flow to scene parameters).
- **Treated as constants / outside tape**: prefix-sum/scan, intersection enumeration, radix-sort, tile-bin edge construction.

### Kernel-by-kernel mapping table

The following table maps the typical gsplat kernel set (as exposed in docs and CUDA trees) to Warp kernels/utilities. File names for XingtongGe/gsplat CUDA sources are listed in its `gsplat/cuda/csrc` directory, including `forward.cu`, `backward.cu`, the 2D variants, `ext.cpp`, and helper headers. citeturn14view0turn15view0

| Source concept (gsplat) | Canonical location in gsplat lineage | Warp port component | Priority | Notes |
|---|---|---|---|---|
| Project 3D Gaussians → 2D (`project_gaussians`) | gsplat docs: Jacobian/EWA projection; differentiable w.r.t. means/scales/quats citeturn13view0 | `project_gaussians_kernel` (Warp) | High | Implement J-based covariance projection and screen-space mean. |
| Compute 2D bounds (`compute_cov2d_bounds`) | gsplat utils doc (exposed helper) citeturn13view1 | Inline Warp function in projection kernel | High | Produces conic (inverse covariance) + pixel radius. |
| Cumulative intersects (`compute_cumulative_intersects`) | gsplat utils doc: nondiff cumulative intersects citeturn13view1 | `warp.utils.array_scan(num_tiles_hit → cum_tiles_hit)` | High | `array_scan` supports int32/float32. citeturn2view0 |
| Map gaussians to intersection IDs (`map_gaussian_to_intersects`) | XingtongGe/gsplat: nondiff; exposed binding citeturn11view3turn15view0 | `map_gaussian_to_intersects_kernel` (Warp) | High | Writes packed 64-bit keys + gaussian IDs. |
| Sort intersections by key | XingtongGe/gsplat uses `torch.sort` (nondiff) citeturn11view5 | `warp.utils.radix_sort_pairs(keys, values, count)` | High | Requires arrays sized **2×count**. citeturn0view0 |
| Tile bin edges (`get_tile_bin_edges`) | gsplat utils doc: nondiff `get_tile_bin_edges` citeturn13view1 | `get_tile_bin_edges_kernel` (Warp) | High | Avoids `runlength_encode` output sizing constraints. |
| Rasterize (alpha compositing) | gsplat docs: sort+bin then compositing; differentiable w.r.t xys/conics/colors/opacity citeturn12view0 | `rasterize_kernel_rgb` (Warp) | High | Mirrors forward sigma/alpha/T update. citeturn19view0turn19view1 |
| Rasterize backward math (optional manual) | nerfstudio CUDA backward shows concrete VJP forms + atomic adds citeturn20view0turn20view1 | Either (A) Warp autodiff on Tape or (B) custom backward kernel | Medium | Manual BWD is a major engineering lift but can reduce tape overhead. |

### Why the “radix-sort bridge” is Warp-native

Warp’s radix sort primitive is explicitly designed for exactly this key/value case:

- stable, linear-time radix sort; keys can be `int64`; values must be `int32`; arrays must be sized to `2*count`. citeturn0view0  

That matches a classic 3DGS intersection list: key=(tile,depth), value=gaussian_id.

### Where Warp Tiles can help

Warp “Tiles” expose block-cooperative operations (`tile_load`, `tile_store`, `tile_reduce`, etc.) intended to build high-performance kernels with shared-memory–like semantics. citeturn9view0

For Gaussian rasterization, the clearest tile-API opportunities are:

- **Batch loading Gaussian parameters for a tile** into a block-local buffer (like the nerfstudio CUDA forward kernel does with `xy_opacity_batch` and `conic_batch` followed by `block.sync()`). citeturn19view1  
- **Reducing per-gaussian gradient contributions** within a block before a single atomic update (as in nerfstudio’s backward path). citeturn20view1  

The provided scaffold currently uses a straightforward SIMT loop per pixel (correctness-first). A tile-optimized rasterizer would restructure the kernel to launch **one thread-block per image tile**, use tile-cooperative loads for gaussian batches, and compute the per-pixel loop with reduced global memory traffic.

## Buffer layout, sizing, and example memory estimates

### Core derived quantities

Let:

- N = number of 3D Gaussians  
- k = average number of tiles hit per Gaussian  
- M = total intersections ≈ N·k  
- H×W = output resolution  
- tile_size = 16 (typical) → tiles_x = ceil(W/16), tiles_y = ceil(H/16), tile_count = tiles_x·tiles_y.

gsplat’s binning/rasterization description makes M explicit as the “total number of tile intersections.” citeturn13view1

### Required buffers (Warp radix-sort path)

Warp radix sort requires allocating `keys` and `values` arrays sized to **2·M**. citeturn0view0

#### Example: N=2,000,000; k=8; 1920×1080; tile_size=16

- M = N·k = 16,000,000  
- tiles_x = ceil(1920/16)=120; tiles_y = ceil(1080/16)=68; tile_count=8,160

| Buffer | Shape | Dtype | Size formula | Example size |
|---|---:|---:|---:|---:|
| `keys` (radix sort) | [2M] | int64 | 2M·8 | 256 MB |
| `values` (radix sort) | [2M] | int32 | 2M·4 | 128 MB |
| `xys` | [N,2] | float32 | N·2·4 | 16 MB |
| `conics` | [N,3] | float32 | N·3·4 | 24 MB |
| `num_tiles_hit` | [N] | int32 | N·4 | 8 MB |
| `cum_tiles_hit` | [N] | int32 | N·4 | 8 MB |
| `tile_min` + `tile_max` | [N,2]×2 | int32 | 2·N·2·4 | 32 MB |
| `colors` | [N,3] | float32 | N·3·4 | 24 MB |
| `opacity` | [N] | float32 | N·4 | 8 MB |
| `tile_bins` | [tile_count,2] | int32 | tile_count·2·4 | ~0.06 MB |
| `out_rgb` | [H·W,3] | float32 | H·W·3·4 | ~23.7 MB |

This excludes **gradient buffers** and any tape storage. If you mark large intermediates as `requires_grad=True`, Warp will allocate corresponding `.grad` arrays (roughly doubling those), and `wp.Tape` may retain additional temporaries depending on kernel complexity. citeturn7view0turn12view0turn13view0

### Expected computational costs and bottlenecks

- Rasterization cost is roughly proportional to total gaussian evaluations across pixels. A crude average is `M / (H·W)` gaussians per pixel (since each intersection is a gaussian affecting a tile, not a pixel; actual per-pixel work depends on gaussian footprint). The forward CUDA reference shows per-pixel loops over batches of gaussians and early termination when transmittance becomes tiny. citeturn19view0turn19view1  
- Sorting cost is O(M) and typically bandwidth-bound; Warp’s radix sort is stable and linear-time by design. citeturn0view0  
- Prefix sum/scan on `num_tiles_hit` is O(N) and supported directly. citeturn2view0  

The “big three” bottlenecks are therefore: (1) projection+tile counting (O(N)), (2) radix sort (O(M)), and (3) rasterization (O(pixel_work)). This matches gsplat’s emphasis on fast tile-based binning and per-tile processing. citeturn12view0turn13view1

## Code-level design for Warp + PyTorch integration

### What is recorded on `wp.Tape` vs treated as constant

The provided scaffold follows the recommended split:

- **On tape (differentiable)**  
  - `project_gaussians_kernel` (grads to `means3d, scales, quats`) consistent with gsplat’s differentiability statement. citeturn13view0  
  - `rasterize_kernel_rgb` (grads to `xys, conics, colors, opacity`) consistent with gsplat’s differentiability statement. citeturn12view0  

- **Off tape (constants / nondifferentiable)**  
  - `warp.utils.array_scan` for cumulative intersects. citeturn2view0turn13view1  
  - `map_gaussian_to_intersects_kernel`, `warp.utils.radix_sort_pairs`, and `get_tile_bin_edges_kernel`. This aligns with gsplat’s “not differentiable” notes for these stages. citeturn13view1turn11view3turn0view0  

### Interop patterns: `wp.from_torch`, zero-copy, and PyTorch autograd

Warp’s interoperability guide documents that Warp arrays can be created from PyTorch tensors (and vice versa), enabling zero-copy handoff for GPU data and integration with PyTorch autograd wrappers. citeturn7view0

The provided file implements a `torch.autograd.Function` that:

- In `forward`: maps input tensors → Warp arrays, launches Warp kernels, returns a PyTorch tensor output.
- In `backward`: maps incoming `grad_output` → Warp array and calls `tape.backward(...)` to populate gradients for the original inputs.

This matches the pattern recommended in Warp’s interop documentation for bridging custom Warp computation into PyTorch training loops. citeturn7view0

### Key kernel pseudocode sketches

Projection (EWA/Jacobian), aligned to gsplat’s documentation:

```text
for each gaussian i:
  t = view * [mu_i, 1]                  # camera-space mean
  if t.z < clip_thresh: mark invalid
  compute camera-space covariance Σ_cam via quaternion + scales and camera rotation
  compute Jacobian J(fx,fy,t)
  Σ' = J Σ_cam J^T                      # 2×2
  cov2d += eps2d * I
  conic = inv(cov2d)                    # upper triangular (xx,xy,yy)
  radius = bbox_sigma * sqrt(max_eigenvalue(cov2d))
  tile_bbox = floor((xy ± radius)/tile_size) clamped to screen tiles
  num_tiles_hit = tile_bbox_area
```

The Jacobian and Σ′ formula are explicitly given in gsplat’s ProjectGaussians docs. citeturn13view0

Intersection enumeration (nondifferentiable), aligned to gsplat’s “tile | depth id” description:

```text
start_i = cum_tiles_hit[i] - num_tiles_hit[i]
for ty in [tile_min_y, tile_max_y):
  for tx in [tile_min_x, tile_max_x):
    tile_id = ty*tiles_x + tx
    depth_key = quantize(depth_i)
    keys[start_i + local]   = (tile_id<<32) | depth_key
    values[start_i + local] = i
```

This matches how gsplat describes intersection IDs and gaussian IDs used for binning/sorting. citeturn11view5turn13view1

Tile bin edges (nondifferentiable), aligned to `get_tile_bin_edges` semantics:

```text
initialize tile_bins[t] = (0,0)
for i in [0..M-1]:
  tile = keys[i] >> 32
  if i==0: tile_bins[tile].start = 0
  else if tile != (keys[i-1]>>32):
     tile_bins[tile].start = i
     tile_bins[prev_tile].end = i
  if i==M-1: tile_bins[tile].end = M
```

gsplat’s docs explicitly define tile bins as `[lower, upper)` ranges over the sorted intersection list. citeturn13view1turn11view5

Rasterize (forward), aligned to gsplat’s rasterization equation and CUDA reference:

```text
for each pixel p:
  tile_id = tile(p)
  [start,end) = tile_bins[tile_id]
  T = 1
  C = 0
  for idx in start..end:
     g = values[idx]
     delta = xy_g - p
     sigma = 0.5*(A*dx^2 + C*dy^2) + B*dx*dy
     alpha = min(0.999, opacity_g * exp(-sigma))
     if alpha < threshold: continue
     C += T * alpha * color_g
     T *= (1 - alpha)
     if T < trans_thresh: break
  out = C + T*background
```

The core `sigma`, `alpha`, and `T` update structure is visible both in gsplat docs and the CUDA forward implementation. citeturn12view0turn19view0turn19view1

Rasterize (backward), if you later implement a manual backward kernel, follows the VJP structure shown in the CUDA backward source (including `v_sigma`, `v_conic_local`, and `v_xy_local`, and atomic accumulation). citeturn20view0turn20view1

## Integration steps, tile-API optimization path, and effort estimate

### Mermaid flowchart of the ported training loop

```mermaid
flowchart TD
  A[PyTorch scene params: means3d, log_scales, quats, colors, opacity_logits] --> B[Render view v]
  B --> C[Warp project_gaussians kernel]
  C --> D[num_tiles_hit]
  D --> E[Warp utils array_scan -> cum_tiles_hit]
  E --> F[Warp map_gaussian_to_intersects kernel -> keys, values]
  F --> G[Warp utils radix_sort_pairs(keys, values)]
  G --> H[Warp get_tile_bin_edges kernel -> tile_bins]
  H --> I[Warp rasterize kernel -> out_rgb]
  I --> J[PyTorch loss vs LR image]
  J --> K[loss.backward()]
  K --> L[torch.autograd.Function.backward calls wp.Tape.backward]
  L --> M[PyTorch optimizer step]
  M --> B
```

### Tile-API optimization plan

A correctness-first Warp port (like the scaffold) is SIMT per pixel. To reach gsplat-class performance, the next steps are tile-centric:

- Launch **one CUDA block per image tile** (16×16 pixels).  
- Use tile-cooperative loads (`tile_load`, indexed loads) to fetch gaussian parameters for a batch of intersections into block-local storage, analogous to the CUDA cooperative-groups approach. citeturn9view0turn19view1  
- Use `tile_reduce` / cooperative reductions to aggregate gradient contributions per gaussian before a single atomic update, analogous to CUDA’s warp-sum + atomic sequence. citeturn9view0turn20view1  

Warp Tiles are explicitly designed for this class of block-cooperative programming. citeturn9view0

### Validation and benchmarks

Minimum parity suite:

- Render correctness vs a reference (numerical):  
  - Per-view PSNR/SSIM (and optionally LPIPS if you add it in PyTorch).  
- Gradient sanity:  
  - Finite-difference checks on a tiny scene (N≈1k) for `means3d`, `colors`, `opacity` using the differentiability guarantees stated by gsplat. citeturn12view0turn13view0  
- Performance:  
  - Time each stage: projection, scan, intersection map, radix sort, tile bins kernel, rasterize. Warp’s primitives’ constraints (2·count buffers for radix sort, supported dtypes) must be respected. citeturn0view0turn2view0  

### Engineering effort estimate

These are realistic ranges for a single experienced GPU engineer familiar with 3DGS:

- **Partial port (forward path + Warp autodiff on tape, correctness-first)**: ~2–4 person-weeks.  
  - The scaffold is already close to this: projection + radix sort + rasterize + tape boundaries.  
- **Full-performance port (tile-optimized forward + manual backward kernel matching gsplat’s CUDA patterns)**: ~6–12 person-weeks.  
  - The main risk is implementing a high-throughput backward path with correct atomic behavior and acceptable memory/tape overhead, consistent with the complexity visible in the CUDA backward kernel. citeturn20view0turn20view1  

## Primary repositories and docs referenced

```text
ContinuousSR
  https://github.com/peylnog/ContinuousSR

XingtongGe gsplat submodule (commit observed in GitHub UI shows full SHA in URL)
  https://github.com/XingtongGe/gsplat
  Example commit page: https://github.com/XingtongGe/gsplat/commit/bcca3ecae966a052e3bf8dd1ff9910cf7b8f851d

nerfstudio-project gsplat
  https://github.com/nerfstudio-project/gsplat

gsplat documentation (projection / rasterization / utils)
  https://docs.gsplat.studio/

NVIDIA Warp documentation (interop, utils, tiles)
  https://nvidia.github.io/warp/
```