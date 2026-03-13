# SR / Warp / Gaussian Splatting Notes

## Status

- Started from `deep-research-report-01.md`, `deep-research-report-02.md`, `deep-research-report-03.md`.
- Next step: read both local Python implementations and compare them against the research guidance.

## Coverage log

- `deep-research-report-01.md`: 489 lines
  - read in two passes:
    - `1-260`
    - `261-489`
- `deep-research-report-02.md`: 411 lines
  - read in two passes:
    - `1-260`
    - `261-411`
- `deep-research-report-03.md`: 331 lines
  - read in two passes:
    - `1-260`
    - `261-331`
- `blender_temp/warp_pytorch_posefree_gaussian_sr.py`: 1367 lines
  - read in five passes:
    - `1-250`
    - `251-500`
    - `501-750`
    - `751-1000`
    - `1001-1367`
- `blender_temp/warp_pytorch_posefree_gaussian_sr_updated.py`: 1633 lines
  - read in seven passes:
    - `1-250`
    - `251-500`
    - `501-750`
    - `751-1000`
    - `1001-1250`
    - `1251-1500`
    - `1501-1633`

## Early research takeaways

- The core recommendation from the reports is to solve this as a continuous 3D scene-representation problem, not as independent 2D super-resolution.
- For multiple low-resolution renders with known intrinsics/extrinsics and slight camera offsets, the stable formulation is:
  - optimize a shared Gaussian-splat scene representation
  - render through an explicit low-resolution observation model
  - then render the learned scene at arbitrary output resolution
- The reports consistently treat anti-aliasing / pixel-footprint modeling as essential, not optional.
- The most relevant reference pipeline is gsplat / 3D Gaussian Splatting:
  - project Gaussians to screen space
  - enumerate tile intersections
  - sort by tile and depth
  - build per-tile ranges
  - rasterize with alpha compositing
- The reports treat projection and rasterization as differentiable, but binning / sorting / tile-range construction as nondifferentiable bookkeeping.

## Report-specific notes

### `deep-research-report-01.md`

- Broad landscape survey:
  - 2D SR methods can improve visual detail but are risky as a front-end because they can hallucinate view-inconsistent texture.
  - Multi-view LR to arbitrary-scale output is better handled by optimizing a continuous 3D representation directly.
- Strongest conceptual guidance:
  - supervise in LR space
  - model LR pixels as finite-area measurements
  - prefer pixel-footprint covariance inflation or supersample-and-downsample over naive point-sample loss
- 3DGS-style density control is presented as core functionality:
  - split / clone Gaussians where needed
  - prune low-contribution Gaussians

### `deep-research-report-02.md`

- ContinuousSR is useful as a reference for arbitrary-scale Gaussian rasterization, but it is fundamentally a 2D method, not a multi-view 3D scene method.
- The practically useful part is the gsplat-shaped execution pattern, not the learned 2D SR architecture.
- Direct code reuse is constrained by license risk:
  - ContinuousSR repo is described as CC BY-NC 4.0
- Practical porting guidance:
  - keep PyTorch for parameters / optimization
  - port projection and rasterization kernels into Warp
  - it is acceptable to keep sorting in PyTorch initially, though report 03 argues Warp utilities can cover more of the pipeline

### `deep-research-report-03.md`

- This report is the most implementation-directed.
- It lays out a fully Warp-native replacement for the gsplat pipeline:
  - Warp projection kernel
  - Warp scan
  - Warp intersection enumeration
  - Warp radix sort
  - Warp tile-bin edge kernel
  - Warp rasterizer
- It keeps the same differentiation boundary:
  - differentiable: projection and rasterization
  - nondifferentiable: intersection bookkeeping and sorting
- It also makes the memory implications explicit:
  - the intersection lists and radix-sort buffers can dominate memory footprint

## Provisional evaluation criteria for the local scripts

- Do they represent a shared 3D scene, or only warp/fuse 2D inputs?
- Do they implement an explicit LR observation model?
- Do they model pixel footprints / anti-aliasing?
- Do they use tile binning, sorting, and alpha compositing in a gsplat-like way?
- Do they distinguish differentiable rendering math from nondifferentiable bookkeeping?
- Do they include density control, culling, and practical training-loop scaffolding?

## Pending

- Reports and both scripts have now been read in full.

## What the research says, in practical terms

- The reports point to one main answer:
  - if you already have multiple LR renders of the same static scene with known or recoverable camera relationships, the right abstraction is a shared continuous 3D scene representation
  - the representation should be optimized against the LR images directly
  - arbitrary output resolution then comes from re-rendering the learned scene, not from fixed-scale 2D upsampling
- The reports are consistent that 2D SR methods are, at best, optional helpers:
  - they can denoise or clean inputs
  - they are not the core solution when the real signal is multi-view geometric parallax
- The most relevant renderer structure is the gsplat / 3DGS family:
  - project 3D Gaussians into screen-space ellipses
  - enumerate tile intersections
  - sort by tile and depth
  - build per-tile ranges
  - alpha-composite in front-to-back order
- Anti-aliasing is a first-class requirement:
  - LR pixels must be treated as finite-area measurements
  - practical approximations include covariance inflation in screen space and opacity compensation
  - more faithful alternatives include supersample-then-downsample observation models
- The reports separate the pipeline into two categories:
  - differentiable continuous math:
    - projection
    - Gaussian evaluation
    - rasterization / compositing
  - nondifferentiable bookkeeping:
    - tile-intersection enumeration
    - sort
    - tile-bin edge construction
- The reports also imply that “arbitrarily high resolution” has a hard caveat:
  - the scene representation can be rendered at arbitrary size
  - recoverable detail is still bounded by the information content of the LR views and the reconstruction prior

## Historical script-by-script reading notes

### `blender_temp/warp_pytorch_posefree_gaussian_sr.py`

- This is the more complete system-level implementation.
- It contains:
  - learnable shared intrinsics
  - learnable per-view camera poses
  - a canonical Gaussian field initialized from view 0 on an anchor lattice
  - a PyTorch-side Gaussian projection step
  - a PyTorch-side tile-bin builder
  - a Warp rasterizer wrapped in `torch.autograd.Function`
  - a multi-stage training loop
  - an optional LIIF/LMF-style residual head for scale-aware RGB refinement
- The overall design is:
  - define the scene in the canonical frame of the first image
  - optimize Gaussian parameters and poses together
  - render at arbitrary output size by rescaling intrinsics and rerunning projection/rasterization
- Relative to the research, this file is strongest on:
  - end-to-end usability
  - per-scene optimization
  - camera optimization / pose-free ambition
  - arbitrary-scale rendering as part of a full training pipeline
- Relative to the research, this file is weaker on:
  - faithful gsplat-style implementation of the full bin/sort/raster pipeline
  - explicit LR observation modeling
  - dynamic Gaussian density control

### `blender_temp/warp_pytorch_posefree_gaussian_sr_updated.py`

- This is the more renderer-centric implementation.
- It contains:
  - explicit data contracts
  - explicit buffer sizing / memory notes
  - a Warp projection kernel
  - a Warp intersection-enumeration kernel
  - Warp tile-bin edge construction
  - Warp rasterization
  - optional Warp radix sort with `torch.sort` fallback
  - a `torch.autograd.Function` wrapper that records projection and rasterization on `wp.Tape`
- The overall design is:
  - assume a known-pose single-view render call
  - expose a gsplat-like renderer as a reusable primitive
  - leave scene optimization, Gaussian initialization, and most higher-level training behavior to the caller
- Relative to the research, this file is strongest on:
  - matching the report-03 decomposition
  - making differentiability boundaries explicit
  - porting more of the gsplat pipeline into Warp
  - documenting memory costs and implementation constraints
- Relative to the research, this file is weaker on:
  - being a complete reconstruction pipeline
  - any actual pose-free functionality
  - any system for learning a scene from a stack of LR renders

## Direct implementation differences

### High-level framing

- Old script:
  - a full per-scene optimization pipeline
  - scene representation + cameras + rendering + losses + optimizer loop
- Updated script:
  - primarily a renderer / port scaffold
  - no full scene object, no initialization from images, no real reconstruction pipeline

### Camera handling

- Old script:
  - attempts pose-free optimization
  - view 0 is canonical
  - other views optimize `se(3)` parameters
  - optional phase-correlation bootstrap from image shifts
  - optionally learn shared intrinsics
- Updated script:
  - assumes known `viewmats` and `K`
  - returns no camera-learning mechanism
  - despite the filename, it is not pose-free

### Scene representation

- Old script:
  - one Gaussian per anchor-grid point derived from the first image
  - each Gaussian has depth, xyz offset, quaternion, scale, opacity, RGB, latent code
  - includes a residual head that predicts scale-aware RGB refinement from rendered latent maps
- Updated script:
  - no scene representation class
  - expects SoA Gaussian tensors as inputs
  - no latent channels, no residual head, no Gaussian initialization strategy

### Projection

- Old script:
  - projection is done in PyTorch (`project_gaussians_single_view`)
  - covariance projection and antialiased opacity compensation are handled on the PyTorch side
- Updated script:
  - projection is moved into Warp (`project_gaussians_kernel`)
  - outputs screen means, conics, AA rho, tile bounds, and depth keys directly from the kernel

### Tile binning and sorting

- Old script:
  - tile membership is built in PyTorch
  - sorting is done using stable `argsort` on depth and then tile id
  - no packed intersection keys
  - no Warp scan or Warp sort
- Updated script:
  - tile intersections are explicitly enumerated as packed `(tile_id << 32) | depth_key` keys
  - supports Warp inclusive scan
  - supports Warp radix sort
  - falls back to `torch.sort` when needed
  - builds tile start/end bins with a dedicated Warp kernel

### Rasterizer interface

- Old script:
  - Warp rasterizer consumes already-projected Gaussians and already-built tile lists
  - rasterizer input includes arbitrary per-Gaussian value channels, which enables RGB + latent + alpha packing
- Updated script:
  - Warp rasterizer consumes SoA projection outputs plus sorted Gaussian ids
  - output is RGB only
  - much closer to the canonical gsplat split between projection, bookkeeping, and rasterization

### Training loop

- Old script:
  - actually trains a scene
  - multi-stage resolution schedule
  - photometric + SSIM loss
  - depth TV / opacity / scale / pose regularization
  - view batching
- Updated script:
  - only includes an example training-loop skeleton
  - simple L1 loss
  - explicitly leaves observation-model handling as TODO
  - no scene parameter module to optimize directly

## What is missing relative to the research

### Missing in both scripts

- A proper LR observation model for multi-view super-resolution:
  - neither script fully implements “render at one scale, then integrate/downsample with a known LR kernel” as the core supervision mechanism
  - the old script partially approximates anti-aliasing, but not a full observation operator
  - the updated script explicitly leaves this as TODO in the example training loop
- 3DGS-style density control:
  - no splitting
  - no cloning
  - no pruning
  - no adaptive Gaussian count
- Strong geometry / appearance extensions:
  - no view-dependent color model such as SH coefficients
  - no depth/normal supervision path
  - no exposure compensation or per-view color correction
- Production-level performance work:
  - no tile-cooperative Warp Tiles implementation
  - no manual backward kernel
  - no real memory pooling / buffer reuse strategy
- A complete validation harness:
  - no finite-difference gradient tests
  - no renderer parity tests
  - no benchmark / profiling suite

### Missing from the old script specifically

- Warp-native implementation of the whole gsplat pipeline:
  - projection remains in PyTorch
  - tile bookkeeping remains in PyTorch
  - no Warp scan / sort / tile-bin kernels
- Dynamic scene refinement:
  - Gaussian set is fixed by the initial anchor lattice
  - this is a major departure from density-controlled 3DGS

### Missing from the updated script specifically

- The entire higher-level scene-learning system:
  - no `from_images(...)`
  - no field initialization from LR images
  - no camera model
  - no pose optimization
  - no intrinsics learning
  - no residual decoder
  - no multiview reconstruction pipeline
- In other words:
  - it is closer to a renderer backend than a successor to the old full pipeline

## Current bottom-line assessment

- If the target is:
  - “a practical end-to-end experiment for reconstructing a continuous scene from a stack of LR Blender renders,”
  - the old script is closer, because it actually defines a scene, a camera model, and a training loop.
- If the target is:
  - “a cleaner and more faithful Warp port of the gsplat renderer architecture,”
  - the updated script is closer, because it ports projection, scan/sort bookkeeping, and tile-bin construction much more explicitly.
- The updated script is therefore not a clean superset of the old one.
- It improves renderer architecture fidelity, but it drops substantial system-level functionality.
- The real missing step is to combine them:
  - keep the old script’s scene / camera / optimization pipeline
  - replace its projection + tile-bin machinery with the updated script’s Warp-native renderer pipeline
  - add the research-recommended LR observation model and density control on top

## Refactor performed

- The monolithic implementations have been split into a small package at:
  - `blender_temp/gaussian_sr/`
- The original top-level scripts have now been removed.
- Their functionality lives directly under:
  - `blender_temp/gaussian_sr/`

## New module layout

- `blender_temp/gaussian_sr/warp_runtime.py`
  - shared Warp import guard
  - `wp`, `_WARP_AVAILABLE`, `_WARP_IMPORT_ERROR`, `require_warp`
- `blender_temp/gaussian_sr/posefree_config.py`
  - `CameraInit`
  - `RenderConfig`
  - `FieldConfig`
  - `TrainConfig`
  - `PoseFreeGaussianConfig`
- `blender_temp/gaussian_sr/image_utils.py`
  - image grids
  - downsampling helpers
  - SSIM / Charbonnier / TV helpers
  - phase-correlation bootstrap helpers
- `blender_temp/gaussian_sr/math_utils.py`
  - inverse-sigmoid / softplus inverse
  - SO(3) and SE(3) helpers
  - quaternion math
  - covariance construction
  - default intrinsics
- `blender_temp/gaussian_sr/camera.py`
  - learnable shared intrinsics
  - learnable camera bundle
- `blender_temp/gaussian_sr/field.py`
  - canonical Gaussian field
  - scale-aware residual head
- `blender_temp/gaussian_sr/pipeline.py`
  - end-to-end pose-free pipeline
  - fit helpers
  - arbitrary-scale rendering helper
- `blender_temp/gaussian_sr/warp_gsplat_renderer.py`
  - updated-script-style low-level Warp gsplat renderer
  - data contracts
  - raster config
  - buffer estimates
  - sort/bin/raster path
  - interop helpers
- `blender_temp/gaussian_sr/__init__.py`
  - package-level re-exports

## Refactor rationale

- The split follows the functional seams that were already present across the two large scripts:
  - configuration
  - generic math / image utilities
  - camera parameterization
  - scene field definition
  - projection and bookkeeping
  - low-level Warp rasterization
  - end-to-end optimization pipeline
- This keeps the “full pipeline” path and the “renderer backend” path separate instead of mixing them in one file.
- Internal code and documentation now import directly from `blender_temp.gaussian_sr`.

## Remaining caveats after the refactor

- This was done without executing or testing.
- The refactor is intended to improve maintainability and make the next implementation steps clearer, not to claim runtime parity.
- The major research-recommended missing features are still missing after the split:
  - stronger view-dependent appearance model
  - validation / profiling harness

## Renderer integration follow-up

- The pose-free pipeline now uses the newer gsplat-style Warp renderer for packed output channels.
- RGB, latent features, and alpha are now rendered in a single pass through the newer backend.
- The older projected-Gaussian Warp rasterizer is no longer on the active pipeline path.
- A small observation-model module was added:
  - `blender_temp/gaussian_sr/observation_model.py`
- Current split of responsibilities:
  - generic multi-channel renderer:
    - `blender_temp/gaussian_sr/warp_gsplat_contracts.py`
    - `blender_temp/gaussian_sr/warp_gsplat_kernels.py`
    - `blender_temp/gaussian_sr/warp_gsplat_autograd.py`
    - `blender_temp/gaussian_sr/warp_gsplat_renderer.py`
  - end-to-end packed render path:
    - `blender_temp/gaussian_sr/pipeline.py`

## What this improves

- The main RGB path is now aligned with the newer renderer architecture.
- Latent/alpha auxiliaries now share the same renderer path instead of depending on the legacy one.
- The renderer internals are now split into:
  - contracts/config
  - kernels
  - autograd/orchestration
  - public wrapper
- Observation-model utilities now have a dedicated location instead of being implicit inside the training loop.
- The observation model now supports explicit render-size selection for supersample-then-downsample supervision.

## What is still unfinished after this step

- The observation model is still simple:
  - it now supports `identity`, `area`, and `supersample_area`
  - it still does not implement more faithful pixel-footprint covariance scheduling
- The old projection / old rasterizer compatibility modules have been removed from the codebase.
- Stronger view-dependent appearance and validation/profiling are still missing.

## Density-control follow-up

- A dedicated density-control layer was added:
  - `blender_temp/gaussian_sr/density_control.py`
- Configuration was added to the main pose-free config:
  - `DensityControlConfig`
  - `PoseFreeGaussianConfig.density`
- The field now supports structural mutation:
  - prune by keep-mask
  - clone selected Gaussians with jitter and reduced opacity/scale
- The pipeline now contains actual hooks:
  - after each optimizer step, optional density control can run
  - if the field topology changes, the optimizer is rebuilt
- The active pipeline now also runs a separate stats-only raster pass per rendered view:
  - renderer-derived contribution mass
  - renderer-derived hit counts
  - renderer-derived pre-contribution transmittance
  - renderer-derived residual-weighted contribution mass

## What the current density-control hooks actually do

- Pruning:
  - removes Gaussians whose opacity is low and whose rendered contribution is also low
  - preserves a minimum Gaussian count by keeping top-opacity entries
- Splitting:
  - scores Gaussians using geometry-gradient norms plus renderer-derived visibility/transmittance/residual statistics
  - treats larger, high-gradient, sufficiently visible Gaussians as split candidates
  - replaces each selected parent with a shrunken version and appends a second child displaced along its dominant local axis
- Cloning:
  - treats smaller, high-gradient, sufficiently visible Gaussians as clone candidates
  - appends jittered duplicates with shared opacity mass
- Candidate selection now uses scale-aware routing:
  - large-scale candidates route to split
  - small-scale candidates route to clone
  - both are capped by configurable per-event limits
- This is closer to a 3DGS-style densification schedule than the earlier single “clone-by-gradient” heuristic, but it is still an approximation.
- The densification decision is now informed by the rasterizer rather than only by world-scale heuristics.
- Residual attribution is now renderer-aware:
  - the stats pass accumulates per-Gaussian contribution weighted by per-pixel image residuals
  - this lets the densifier prefer Gaussians that both matter in rendering and sit under high-error pixels
  - the residual-weighted stats pass is only invoked on densification steps, keeping the extra cost bounded

## Density event debug summaries

- Each densification event now produces a compact summary containing:
  - mean/max visibility contribution
  - mean/max residual-weighted contribution
  - mean/max peak screen-space error
  - mean transmittance
  - top split candidates
  - top clone candidates
- These summaries are:
  - printed as compact log lines during training
  - stored in `history[\"density_events\"]`
- The stats path now also carries a coarse per-Gaussian screen-space error map:
  - each Gaussian accumulates residual-weighted contribution into a fixed grid of screen bins
  - the density summaries expose each top candidate's `peak_bin` and `peak_error`
- Density events can now also be emitted externally:
  - optional JSONL append path during `fit(...)`
  - optional callback for custom inspection hooks
- This gives you a lightweight inspection path without adding a separate test harness.

## View-dependent appearance follow-up

## Numerical-stability debugging follow-up

- The progress logging and fail-fast checks made the first failure boundary explicit:
  - the run is finite through the first forward/backward
  - the first bad state appears in the multi-view camera-gradient path
- A synthetic reproducer now exists that does not depend on the Blender frames:
  - `1` zero-valued view at `16x16` stays finite for one step
  - `2` zero-valued views at `16x16` reproduce the failure under the same high-density config
- After adding gradient instrumentation immediately after `loss.backward()` and before gradient clipping / `opt.step()`, the first non-finite tensor is now localized to:
  - `camera_pose.grad`
- In the two-view synthetic case, the fail-fast payload reports:
  - `kind = gradients_after_backward`
  - `nonfinite_grads = { camera_pose: 3 }`
- This materially narrows the bug:
  - it is not driven by high-resolution inputs
  - it is not driven by image content
  - it is not a whole-model optimizer explosion on the first step
  - it is specifically a multi-view camera-gradient numerical issue
- Root cause:
  - `so3_exp_map(...)` used `torch.where(...)` around the singular expressions `sin(theta) / theta` and `(1 - cos(theta)) / theta^2`
  - at zero rotation, those singular branches were still evaluated, producing `NaN` gradients even when the forward result stayed finite
- Fix:
  - `so3_exp_map(...)` now fills the small-angle and large-angle regions separately instead of evaluating the singular branch and masking it away afterward
- Post-fix validation:
  - the isolated `so3_exp_map(0)` backward path now has finite gradients
  - the two-view synthetic `16x16` pipeline case now stays finite for one step
  - the real `render_1920_1080` pipeline case now stays finite for one step
  - the exact CLI command that previously went `NaN` on step 2 remained finite through at least step 6 during a timed run
- Renderer gradient follow-up:
  - the Warp autograd wrapper had also been detaching `viewmat` and `K` with `requires_grad=False`
  - that meant camera optimization was only being supervised through the SH appearance path, not through projection geometry
  - the wrapper now records Warp-side gradients for detached `viewmat` / `K` values and returns them through the custom autograd function
  - a direct renderer test now checks finite gradients for `viewmat` and `K`
  - a pipeline-level test now checks that, even in `appearance.mode = constant`, the second-view camera pose receives finite nonzero geometry gradients

- A dedicated appearance module was added:
  - `blender_temp/gaussian_sr/appearance.py`
- Configuration was added to the main pose-free config:
  - `AppearanceConfig`
  - `PoseFreeGaussianConfig.appearance`
- The field now supports two appearance modes:
  - `constant`
  - `sh`
- In `sh` mode:
  - each Gaussian keeps its base RGB logits
  - each Gaussian also stores directional SH coefficients
  - per-view color is evaluated from the current camera direction to the Gaussian
- The implementation currently supports real SH bases up to degree 2.

## What this changes in the active pipeline

- Rendered RGB is no longer purely view-independent.
- The renderer itself remains generic and payload-based.
- View dependence is computed before rasterization as part of field evaluation, then the resulting per-view RGB is packed into the rendered channels.

## Cleanup status

- Removed from the codebase:
  - `blender_temp/gaussian_sr/projection.py`
  - `blender_temp/gaussian_sr/warp_posefree_rasterizer.py`
  - the unused private `_render_packed_with_pose(...)` helper in `pipeline.py`
  - low-level renderer demo / zero-copy / DLPack / flowchart helper exports that were not part of the active pipeline

## Main remaining caveats

- Density control now uses renderer-derived visibility statistics, residual-weighted attribution, and a coarse per-Gaussian screen-space error map, but it still does not use full-resolution or exact backward-attributed image-error diagnostics.
- The SH appearance model is limited to degree 2 and uses a sigmoid-bounded RGB parameterization.
- There is still no validation harness, renderer parity check, or profiling pass in the repo.

## Runtime debugging notes

- Running `nix run .#blender-temp -- --input-dir ./render_1920_1080 --output-dir temp-output --device cuda --scale 2` exposed several Warp 1.11 codegen constraints in `blender_temp/gaussian_sr/warp_gsplat_kernels.py`.
- The required fixes were:
  - make bit-pack / bit-unpack shifts type-stable with explicit `int64` shift counts
  - declare mutable loop accumulators as dynamic variables (`float(1.0)`, `int(0)`) instead of Python literals
- After those fixes, the command progressed past the previous kernel-compilation failures and entered the first training step on the full 8-view 1080p dataset.
- The full run is still expensive:
  - on the local RTX 4090 it reached roughly 15.6 GiB of active GPU memory during the first training step
  - the command was stopped after confirming that the earlier startup error had been resolved
- A separate PyTorch warning came from Warp's `from_torch(...)` bridge inspecting `.grad` on non-leaf tensors created by slicing/packing in the custom autograd wrapper.
- The render path now detaches those tensors before handing them to Warp:
  - this suppresses the warning
  - it does not change the Torch backward contract, because the enclosing `torch.autograd.Function` still returns gradients explicitly from `backward(...)`
- The training loop now has explicit progress instrumentation:
  - optional verbose per-view heartbeat logging
  - per-step timing breakdowns
  - CUDA memory snapshots
  - per-view renderer counters such as Gaussian count, visible count, and intersection count
  - stage/step/view JSONL progress traces
- `blender_temp/cmd/main.py` now writes `progress.jsonl` into the output directory and warns when the initial Gaussian count is likely to make steps very slow.
- Progress lines now include wall-clock timestamps, and progress JSONL events include UTC timestamps.
- The training loop now fails fast on non-finite state:
  - non-finite rendered RGB
  - non-finite observed predictions
  - non-finite photo / regularization / total loss
  - non-finite model parameters immediately after the optimizer step
- A split `pytest` + `hypothesis` property test suite was added under `tests/gaussian_sr/`.
- Coverage now includes:
  - pure utility/math invariants
  - observation-model properties
  - camera/intrinsics properties
  - appearance properties
  - field topology / finiteness properties
  - tiny-scene Warp renderer metamorphic checks
  - a bounded single-step pipeline finiteness test
- Coverage was expanded further to include:
  - density-control schedule, normalization, selection, and mutation behavior
  - JSONL logging helper behavior for density/progress events
  - renderer contract/helper utilities such as tile estimation and shape assertions
- A dataset-backed CUDA regression test was added:
  - `tests/gaussian_sr/test_pipeline_nan_regression.py`
  - it uses the local `render_1920_1080` Blender frames
  - it asserts that the current large-scene configuration fails with `FloatingPointError` after the first optimizer step
- A synthetic CUDA regression test was also added:
  - `tests/gaussian_sr/test_pipeline_synthetic_nan_regression.py`
  - it shows that the failure does not depend on the sample Blender frames
  - under the same high-risk config, `2` synthetic zero-valued views at `16x16` already trigger the first-step optimizer blow-up
  - the corresponding `1`-view case stays finite for one step
- The renderer and pipeline stability tests are guarded to run only when CUDA + Warp are available.
- The CLI now exposes the two most important runtime-throttling knobs directly:
  - `--anchor-stride`
  - `--view-batch-size`
- This makes it possible to reduce the initial Gaussian count and the number of rendered views per optimizer step without editing `blender_temp/cmd/main.py`.
- Measured comparison on `render_1920_1080` with `3` frames and `--verbose-progress`:
  - baseline config:
    - `--anchor-stride 1`
    - `--view-batch-size 0`
    - estimated Gaussians: `2,073,600`
    - stable through at least step `24`
    - typical stage-1 step time: about `3.5s` to `3.8s`
    - peak CUDA memory observed: about `4.24 GiB`
  - lighter debug config:
    - `--anchor-stride 4`
    - `--view-batch-size 1`
    - estimated Gaussians: `129,600`
    - stable through at least step `200`
    - typical stage-1 step time: about `0.16s` to `0.21s`
    - peak CUDA memory observed: about `0.40 GiB`
- Interpretation:
  - the lighter config cuts the initial Gaussian count by `16x`
  - it cuts peak observed memory by roughly `10x`
  - it cuts optimizer-step wall time by roughly `18x` to `23x`
  - because it uses one view per step instead of all views, it is a better smoke/debug mode than a like-for-like training-quality comparison
- Later full-resolution failure mode with the lighter config:
  - command:
    - `--anchor-stride 4`
    - `--view-batch-size 1`
    - `--max-frames 3`
  - it remained stable into stage 3, but then failed a few steps after a density-control event at global step `750`
  - failure:
    - Warp CUDA OOM during `_prepare_sorted_intersections(...)`
    - allocation failure occurred while creating the sort buffers for tile intersections
  - critical telemetry from the failing region:
    - stage 3 render size: `1080x1920`
    - Gaussian count: about `129,752`
    - visible Gaussians in the preceding successful step: about `110,478`
    - intersection count `M`: about `166,302,775`
  - diagnosis:
    - the bottleneck is the intersection/sort workload, not the base Gaussian count by itself
    - with Warp radix sort enabled, the code allocates `2 * M` key/value buffers before sorting
    - for `M ~= 166M`, that already implies multi-gigabyte sort storage, before accounting for other live Warp allocations and scratch space
    - the logged `torch.cuda.memory_*` numbers under-report the true peak because they only reflect PyTorch allocator usage, not Warp allocator usage
  - practical implication:
    - stage-3 full-resolution runs need an additional guard against extreme projected Gaussian footprints / intersection counts
    - the most likely controls are:
      - radius clipping in screen space
      - stronger scale control / regularization
      - more conservative densification behavior at full resolution
      - an explicit pre-sort memory budget check using actual `M`
  - implemented follow-up:
    - `radius_clip_px` is now exposed through the CLI/config path
    - the kernel-side `radius_clip_px` semantics were corrected to cull oversized projected splats rather than tiny ones
    - a pre-sort memory guard now checks the actual `M`-dependent sort-buffer estimate before allocating the intersection sort buffers
    - in `sort_mode=\"auto\"`, the renderer now prefers falling back from Warp radix sort to `torch.sort` before failing outright if the Warp radix buffers would exceed the configured budget
    - density control can now be disabled during the final training stage via config / CLI

## Warp requirement simplification

- Warp is now treated as a hard dependency rather than an optional backend.
- `blender_temp/gaussian_sr/warp_runtime.py` was simplified to:
  - import Warp directly
  - call `wp.init()` directly
  - expose `wp` without optional-import error plumbing
- `blender_temp/gaussian_sr/warp_gsplat_kernels.py` was simplified to:
  - import `wp` directly from `warp_runtime`
  - define all kernels/functions unconditionally instead of wrapping the entire file in an availability guard
- Compatibility booleans such as `_WARP_AVAILABLE` still exist only to avoid unnecessary churn in test/import surfaces, but they are now effectively constant because Warp is required.

## Current pipeline diagram

```text
CLI / main.py                                   [PyTorch]
  -> collect image paths
  -> decode images with torchvision
  -> stack batch [V, 3, H, W]
  -> build PoseFreeGaussianConfig
  -> PoseFreeGaussianSR.from_images(...)

from_images(...)                                [PyTorch]
  -> default intrinsics
  -> optional phase-correlation shift bootstrap
  -> initialize canonical Gaussian field from view 0
  -> initialize learnable camera bundle

fit(...)                                        [PyTorch orchestrating Warp]
  -> for each stage scale:
       -> area-downsample stage targets if needed                [PyTorch]
       -> choose render size from observation model              [PyTorch]
       -> for each optimizer step:
            -> choose all views or a random view subset          [PyTorch]
            -> camera_model.world_to_camera()                    [PyTorch]
            -> for each chosen view:
                 -> field_model.gaussian_params(...)             [PyTorch]
                 -> pack RGB + latent + alpha                    [PyTorch]
                 -> render_values_warp(...)                      [Mixed]
                      -> project_gaussians_kernel                [Warp]
                      -> prefix scan / intersections / bins      [Warp]
                      -> sort intersections                      [Warp or torch.sort fallback]
                      -> rasterize_values_kernel                 [Warp]
                      -> return HWC image to PyTorch             [Mixed]
                 -> optional residual head on latent map         [PyTorch]
                 -> observation model downsample                 [PyTorch]
                 -> photometric loss                             [PyTorch]
                 -> render_stats_warp(...) for aux stats         [Warp + PyTorch]
            -> regularization loss                               [PyTorch]
            -> backward through mixed graph                      [PyTorch + Warp tape]
            -> optimizer step                                    [PyTorch]
            -> optional density control                          [PyTorch]

render_view(...)                                 [PyTorch orchestrating Warp]
  -> field_model.gaussian_params(...)            [PyTorch]
  -> render_values_warp(...)                     [Mixed / Warp]
  -> optional residual head                      [PyTorch]
  -> save PNG / PT                               [PyTorch]
```

## Backend split by responsibility

- Pure PyTorch:
  - image loading / preprocessing
  - camera parameterization
  - Gaussian field parameterization
  - view-dependent appearance evaluation
  - residual head
  - observation model
  - losses and regularization
  - optimizer step
  - density control and topology mutation
- Warp:
  - Gaussian projection to screen-space conics
  - tile-intersection enumeration
  - tile-bin edge construction
  - rasterization / compositing
  - visibility / residual-weighted stats passes
- Mixed:
  - custom `torch.autograd.Function` wrapper over Warp kernels
  - Warp-side gradient recording on `wp.Tape`, returned back into PyTorch autograd
  - sorting can fall back to `torch.sort`

## What `torch.compile` can and cannot help here

### Expected limit

- `torch.compile` will not speed up the Warp kernels themselves.
- The expensive parts in the heavy configuration are still:
  - Warp projection
  - Warp bookkeeping
  - Warp rasterization
  - backward through the Warp-backed custom autograd op
- In the measured heavy run, most time was already in render + backward, so `torch.compile` will probably not produce a dramatic end-to-end speedup there.

### Good compilation candidates

- `CanonicalGaussianField.gaussian_params(...)`
  - pure tensor math
  - static shapes within a run
  - called once per rendered view
- `apply_view_dependent_rgb(...)`
  - pure tensor math
  - especially useful in SH mode
- `LearnableCameraBundle.world_to_camera(...)`
  - small, but pure tensor math
- `so3_exp_map(...)` / `pose_vec_to_rt(...)`
  - pure tensor math
- `ScaleAwareResidualHead.forward(...)`
  - standard MLP
  - likely the cleanest high-confidence compile target
- observation-model helpers
  - `area_downsample_*`
  - `render_observe_rgb(...)`
- loss / regularization helpers
  - `_photometric_loss(...)`
  - `_regularization(...)`

### Weak compilation candidates

- `fit(...)` as a whole
  - too much Python control flow
  - loops over stages and views
  - random view subsampling
  - logging / JSONL emission
  - fail-fast checks
  - density control with topology mutation
  - optimizer rebuild after pruning / split / clone
- `render_values_warp(...)`
  - the call is already a custom autograd boundary around Warp
  - `torch.compile` can only optimize around it, not inside it

### Likely graph-break / recompilation sources

- changing `view_ids` length when `view_batch_size = 0` or random subsets are used
- changing stage sizes across `(0.25, 0.5, 1.0)`
- density control changing `N`
- verbose logging and JSONL emission inside the step loop
- sorting fallback paths
- conditional residual stats pass only on densification steps

## Practical `torch.compile` strategy for this codebase

### 1. Compile leaf modules first

- Best first targets:
  - compile `ScaleAwareResidualHead`
  - compile a small pure-tensor helper used by `gaussian_params(...)`
  - compile `apply_view_dependent_rgb(...)`
- This is low-risk because it avoids the mixed Warp boundary.

### 2. Compile the pure-PyTorch pre-render block

- Split out a helper that performs:
  - camera pose construction
  - field evaluation
  - payload packing
- Keep the Warp render call outside that compiled helper.

### 3. Compile the pure-PyTorch post-render block

- Split out a helper that performs:
  - residual-head refinement
  - observation-model application
  - photometric loss
  - regularization
- Again, keep Warp render and logging outside it.

### 4. Do not compile the outer training loop first

- The outer `fit(...)` loop is the wrong first target because:
  - too many graph breaks
  - topology mutation from density control
  - varying stage sizes
  - logging dominates the Python side
- If a compiled training step is attempted later, it should be a separate minimal `train_step_core(...)` with:
  - fixed `view_batch_size`
  - logging disabled
  - density control disabled
  - static stage size

### 5. Prefer static-shape debug/training modes

- `torch.compile` will work better if:
  - `view_batch_size` is fixed and positive
  - stage size is fixed for the duration of a compiled region
  - density control is disabled inside the compiled region
- This means the light debug mode:
  - `--anchor-stride 4`
  - `--view-batch-size 1`
is a much better candidate for experimentation than the full dynamic pipeline.

## Concrete recommendation

- First implementation pass:
  - compile `ScaleAwareResidualHead`
  - compile `apply_view_dependent_rgb`
  - compile a factored pure-PyTorch `prepare_render_inputs(...)`
  - compile a factored pure-PyTorch `postprocess_and_loss(...)`
- Do not try to compile `fit(...)` monolithically.
- Expected upside:
  - modest overall speedups in the heavy Warp-dominated path
  - potentially more noticeable improvement in lighter debug configs where Python/PyTorch overhead is a larger fraction of the step

## First-pass `torch.compile` integration implemented

- The current code now performs a best-effort compile of the pure-PyTorch regions only.
- The Warp render and stats kernels are still outside the compiled regions.
- Implemented compiled/fallback wrappers:
  - `prepare_render_payload(...)`
    - performs field decode
    - builds `viewmat` and `K`
    - packs renderer payload tensors before the Warp call
  - residual-head forward path
  - post-render RGB refinement helper
  - observe + photometric-loss helper
  - regularization helper
- Fallback behavior:
  - if `torch.compile` is unavailable or a compiled callable raises at runtime, execution falls back to eager mode and emits a one-time warning for that callable
- Current compile boundary:
  - compiled:
    - field decode + pre-render payload packing
    - residual-head forward
    - post-render RGB refinement
    - observation-model application
    - photometric loss
    - regularization
  - not compiled:
    - outer `fit(...)` loop
    - logging / JSONL emission
    - density control
    - Warp-backed projection / sorting / rasterization / stats passes
- Running `nix develop .#blender-temp --command -- python3 -m pytest` now gives:
  - `23 passed`
  - `1 failed`
- The remaining failing test is the SSIM identity property:
  - `tests/gaussian_sr/test_image_utils_properties.py::test_ssim_of_identical_images_is_one`
- This appears to expose a real issue in `blender_temp/gaussian_sr/image_utils.py` rather than a test bug:
  - for identical zero images, the SSIM numerator is `c1 * c2 = 9e-8`
  - the implementation clamps the full denominator to `1e-6`
  - that forces the ratio to `0.09` instead of `1.0`
- The SSIM implementation was updated to compute the luminance and contrast/structure terms separately and only apply a tiny-value clamp to the contrast denominator.
- That preserves the identity property for low-variance images while still guarding against division by zero.

## 2026-03-13: Warp usage inventory and Helion migration assessment

### Direct Warp usage in this repository

Production code:
- `blender_temp/gaussian_sr/warp_runtime.py`
  - Imports `warp as wp`
  - Calls `wp.init()`
  - Exposes `wp` to the package surface
- `blender_temp/gaussian_sr/warp_gsplat_kernels.py`
  - Defines all renderer kernels with `@wp.func` and `@wp.kernel`
  - Uses Warp math/vector intrinsics, `wp.tid`, and `wp.atomic_add`
  - Implements:
    - Gaussian projection
    - intersection mapping
    - tile-bin edge detection
    - packed-channel rasterization
    - visibility / residual stats rasterization
- `blender_temp/gaussian_sr/warp_gsplat_autograd.py`
  - Uses Warp Torch interop: `wp.from_torch`, `wp.to_torch`
  - Allocates Warp arrays: `wp.empty`, `wp.zeros`
  - Launches kernels with `wp.launch`
  - Uses `wp.utils.array_scan`
  - Uses `wp.utils.radix_sort_pairs` with `torch.sort` fallback
  - Uses `wp.Tape` to provide the differentiable render path
- `blender_temp/gaussian_sr/__init__.py`
  - Re-exports `wp` and Warp runtime helpers

Tests:
- `tests/gaussian_sr/test_warp_renderer_properties.py`
- `tests/gaussian_sr/test_pipeline_nan_regression.py`
- `tests/gaussian_sr/test_pipeline_synthetic_nan_regression.py`
- `tests/gaussian_sr/test_pipeline_stability_properties.py`
- any test importing `_WARP_AVAILABLE` or renderer APIs is Warp-coupled by construction

Docs / packaging:
- `pyproject.toml`
  - depends on `warp-lang`
- `README.md`
  - describes Warp as the irregular raster backend
  - still mentions `warp.sparse` as a future extension

### What Warp is currently providing

The active implementation is not using Warp as a small optional accelerator. Warp is carrying the core renderer backend:
- custom GPU kernels
- Torch <-> device array bridging
- irregular prefix-sum / intersection bookkeeping
- key/value radix sort on flattened intersections
- differentiable rasterization through `wp.Tape`
- atomic accumulation for stats

The rest of the system is PyTorch:
- scene parameterization
- camera model
- losses / regularization
- density control
- optimizer
- training loop

### Helion fit against current needs

Helion is not a drop-in Warp replacement. It is a higher-level Triton DSL centered around `@helion.kernel` and tiled tensor programs.

Promising matches:
- tensor-centric kernel authoring
- atomics (`hl.atomic_add`, etc.)
- scans (`hl.associative_scan`, `hl.cumsum`)
- PyTorch-native tensor surface
- ability to drop to raw Triton via `inline_triton` / `triton_kernel`

Critical mismatches for this renderer:
- no Warp-style runtime / array API equivalent
- no direct equivalent of `wp.Tape` for this use case
- Helion docs/tutorials explicitly show manual backward kernels as the normal path for custom kernels
- no direct equivalent of `wp.utils.radix_sort_pairs`
- Helion sort support is Triton `tl.sort`-style and only along the last tensor dimension, not a general flat key/value sort primitive
- Helion tiling is loop/grid oriented, not a direct match for Warp’s explicit raster-kernel programming model

### Migration effort estimate

Effort to switch the active Warp renderer path to Helion: high.

Practical estimate:
- minimal proof-of-concept port of projection + forward raster only: ~3-7 focused engineering days
- production-equivalent forward path with bookkeeping, stats pass, and stable memory behavior: ~2-4 weeks
- full parity including gradients/backward correctness and current test coverage: ~4-8 weeks

This is closer to a backend rewrite than a library swap.

### Why the effort is high

1. `warp_gsplat_kernels.py` is a full custom kernel suite
- These kernels are already written in a CUDA-like SPMD style.
- Porting them to Helion means either:
  - rewriting them in Helion’s tiled tensor style, or
  - embedding large raw Triton fragments through Helion
- The latter reduces the benefit of “switching to Helion”.

2. `warp_gsplat_autograd.py` depends on Warp-specific runtime behavior
- explicit Warp allocations
- explicit Warp launches
- `array_scan`
- radix sort pairs
- `wp.Tape`
- Torch interop assumptions

3. Backward is the real cost center
- Today the differentiable path leans on Warp’s tape.
- Helion’s public story is not equivalent here.
- A Helion migration would likely require a custom `torch.autograd.Function` with:
  - manual forward orchestration
  - a manually implemented backward kernel suite, or
  - a different differentiable formulation

4. Sort/bookkeeping is not a clean match
- The current algorithm depends on flattening tile intersections into global key/value arrays and sorting them.
- Warp has a direct primitive for that.
- Helion does not appear to.
- This likely forces either:
  - a separate Torch/Triton sorting path
  - a redesigned binning algorithm
  - or retaining a non-Helion dependency for sorting

### Realistic migration options

Option A: do not migrate
- Keep Warp for the renderer backend.
- Continue using PyTorch for everything else.
- This is the lowest-risk path.

Option B: partial migration
- Keep the current renderer backend on Warp.
- Use Helion only for auxiliary dense kernels if profiling shows benefit.
- Example candidates:
  - observation-model experiments
  - dense post-render feature transforms
  - maybe some pre-render tensor math
- This is moderate effort and avoids rewriting the irregular renderer.

Option C: full backend rewrite
- Replace `warp_runtime.py`, `warp_gsplat_kernels.py`, and most of `warp_gsplat_autograd.py` with a Helion/Triton backend.
- Expect substantial redesign of sort + backward.
- This is only justified if there is a strategic reason to standardize on Helion/Triton.

### Recommendation

For the current codebase, a full Helion switch is not justified as a routine refactor.

Best recommendation:
- keep Warp for the active raster backend
- if Helion is desired, prototype a narrow forward-only replacement first:
  - project
  - map/bucket intersections
  - forward raster
- do not attempt the backward/stats path until the forward path is clearly superior and memory-stable

### Most likely migration boundary if attempted

If a Helion port is pursued, the cleanest boundary is:
- leave `pipeline.py`, `field.py`, `camera.py`, losses, density control unchanged
- replace only the renderer backend behind:
  - `render_values_warp(...)`
  - `render_stats_warp(...)`
- but this still requires re-implementing almost all logic now living in:
  - `warp_gsplat_kernels.py`
  - `warp_gsplat_autograd.py`
