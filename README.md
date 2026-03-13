# Blender-Temp: Pose-Free Multi-View Gaussian Super-Resolution With a Warp Renderer

This repository is a per-scene optimization system for reconstructing a shared continuous 3D scene representation from multiple low-resolution Blender renders, then re-rendering that scene at arbitrary output scale.

The implementation is a hybrid:

- PyTorch owns scene parameters, camera parameters, losses, optimization, and most training-loop orchestration.
- NVIDIA Warp owns the gsplat-style projection, visibility bookkeeping, rasterization, and visibility/statistics kernels.

This README is intentionally long and operational. The repository may be handed to an external research agent without the input images or the `temp-output*` debug directories, so this document includes:

- the current architecture and data flow
- the actual runtime behavior we observed while debugging
- the current failure mode and how far current mitigations go
- a trust map for the in-tree deep-research documents

## Current Status

The codebase is no longer a thin renderer prototype. It currently includes:

- a pose-free multi-view training pipeline
- learnable per-view camera poses
- a canonical Gaussian field initialized from view 0
- optional learnable shared intrinsics
- view-dependent SH appearance
- an area/supersample observation model
- adaptive density control with prune / split / clone
- a Warp renderer with projection, tile-intersection enumeration, sorting, tile-bin construction, rasterization, and visibility/statistics passes
- debug checkpoints, periodic preview renders, rollback on collapse, and stage-scoped profiling hooks

The main unresolved problem is not basic functionality. The pipeline trains, renders, checkpoints, and exports. The main unresolved problem is late-stage coverage collapse during full-resolution optimization, especially for non-anchor views.

In practice:

- safe/conservative runs can produce non-black output
- aggressive or unlucky runs can still collapse to effectively single-view support
- the current view-aware density policy delays collapse, but does not fully prevent it

## What This Repository Is And Is Not

This repository is:

- a static-scene, per-scene optimization pipeline
- aimed at multi-view low-resolution inputs from a renderer
- able to render at any positive output scale
- explicitly designed around Gaussian splatting rather than 2D feed-forward SR

This repository is not:

- a pretrained feed-forward SR model
- a full SfM / MVS system
- a known-pose-only reference implementation
- a production-ready robust reconstruction system

The code is explicitly pose-free: the first view is the canonical reference frame and the remaining camera poses are optimized jointly with the scene. See [blender_temp/gaussian_sr/camera.py](blender_temp/gaussian_sr/camera.py) and [blender_temp/gaussian_sr/pipeline.py](blender_temp/gaussian_sr/pipeline.py).

## High-Level Pipeline

The full end-to-end path is:

1. Load RGB images from a directory into a `[V, 3, H, W]` float32 tensor.
2. Build a pose-free scene model from the first image on an anchor lattice.
3. Initialize camera poses, optionally with a phase-correlation translation bootstrap.
4. Train in three stages at increasing spatial scales.
5. At each training step:
   - render one or more views through the Warp renderer
   - pass the rendered RGB through the observation model to compare against the low-resolution target
   - apply photometric and regularization losses
   - backpropagate through the differentiable parts of the renderer
   - periodically run density control
6. Export final renders at the requested scale.

The CLI entrypoint is [blender_temp/cmd/main.py](blender_temp/cmd/main.py). The main training system is [blender_temp/gaussian_sr/pipeline.py](blender_temp/gaussian_sr/pipeline.py).

## Runtime Environment We Actually Used

The main debugging runs summarized below were performed on:

- NVIDIA GeForce RTX 4090, 24 GiB
- CUDA Toolkit 12.8
- Warp 1.11.0
- PyTorch 2.10.0
- Python 3.13
- Nix-based environment via `nix run` / `nix develop`

Those details matter because several compiler and memory-management behaviors were specific to this stack.

## Repository Layout

The most important files are:

- [blender_temp/cmd/main.py](blender_temp/cmd/main.py): CLI, preflight warnings, debug observer, checkpointing, export, optional profiling.
- [blender_temp/gaussian_sr/pipeline.py](blender_temp/gaussian_sr/pipeline.py): training/render orchestration, loss computation, compile boundary, progress emission.
- [blender_temp/gaussian_sr/field.py](blender_temp/gaussian_sr/field.py): canonical Gaussian parameterization, residual head, prune/clone/split mutations.
- [blender_temp/gaussian_sr/camera.py](blender_temp/gaussian_sr/camera.py): learnable shared intrinsics and learnable camera bundle.
- [blender_temp/gaussian_sr/appearance.py](blender_temp/gaussian_sr/appearance.py): view-dependent SH appearance model.
- [blender_temp/gaussian_sr/observation_model.py](blender_temp/gaussian_sr/observation_model.py): LR observation model.
- [blender_temp/gaussian_sr/density_control.py](blender_temp/gaussian_sr/density_control.py): density scoring, per-view coverage logic, prune/split/clone decisions.
- [blender_temp/gaussian_sr/warp_gsplat_autograd.py](blender_temp/gaussian_sr/warp_gsplat_autograd.py): PyTorch/Warp bridge, visibility preparation, raster forward/backward wrappers.
- [blender_temp/gaussian_sr/warp_gsplat_kernels.py](blender_temp/gaussian_sr/warp_gsplat_kernels.py): Warp kernels.
- [blender_temp/gaussian_sr/debug_checkpoint.py](blender_temp/gaussian_sr/debug_checkpoint.py): reusable checkpoint restore helpers.

## Configuration That The CLI Actually Uses

The dataclass defaults in [blender_temp/gaussian_sr/posefree_config.py](blender_temp/gaussian_sr/posefree_config.py) are not the whole story. The CLI overrides several of them in [blender_temp/cmd/main.py](blender_temp/cmd/main.py):

- `camera.learn_intrinsics = False`
- `train.stage_scales = (0.25, 0.5, 1.0)`
- `train.steps_per_stage = (250, 500, 1000)`
- `field.feature_dim = 8`
- `field.anchor_stride = CLI arg`
- `train.view_batch_size = CLI arg`
- `render.radius_clip_px = CLI arg`
- `density.disable_final_stage = CLI flag`

This means a library user and a CLI user are not necessarily running the same schedule.

## Detailed Pipeline

### 1. Input Loading

Images are loaded by [load_images()](blender_temp/cmd/main.py) into a float32 CUDA tensor with shape `[V, 3, H, W]`.

Current assumptions:

- all images have identical shape
- images are RGB
- images are already in a form that can be compared directly with simple photometric losses

Not implemented:

- exposure compensation
- linear/HDR color handling
- per-view photometric normalization beyond direct RGB comparison

### 2. Intrinsics And Camera Parameterization

Shared intrinsics live in [LearnableSharedIntrinsics](blender_temp/gaussian_sr/camera.py). By default the CLI freezes them, but the model supports learning them.

Per-view extrinsics live in [LearnableCameraBundle](blender_temp/gaussian_sr/camera.py):

- view 0 is fixed as the canonical reference
- views `1..V-1` optimize a 6D pose-vector parameterization
- optional initialization comes from phase-correlation image shifts via [estimate_translation_bootstrap()](blender_temp/gaussian_sr/image_utils.py)

This is one of the biggest ways the code differs from some of the in-tree research reports: the active code is pose-free, not known-pose-only.

### 3. Canonical Gaussian Field Initialization

The scene representation is defined in [CanonicalGaussianField](blender_temp/gaussian_sr/field.py).

Initialization strategy:

- take view 0 as the canonical image
- place one Gaussian per anchor-grid sample at stride `anchor_stride`
- initialize per-Gaussian depth, 3D offset, quaternion, scale, opacity, RGB, latent features, and optional SH coefficients

The initial lattice size is:

- `ceil(H / anchor_stride) * ceil(W / anchor_stride)`

For the 1920x1080, `anchor_stride=8`, `max_frames=3` runs we debugged, this gave:

- `32400` initial Gaussians

Current field parameters:

- `depth_raw`
- `xyz_offset`
- `quat_raw`
- `log_scale`
- `opacity_logit`
- `rgb_logit`
- `latent`
- optional `sh_coeffs`

### 4. View-Dependent Appearance

Appearance is handled in [blender_temp/gaussian_sr/appearance.py](blender_temp/gaussian_sr/appearance.py).

The current options are:

- `constant`
- `sh`

The active path uses real spherical harmonics up to degree 2. The base RGB logits model the view-independent color, and SH coefficients model directional residuals.

Important note for external readers:

- view-dependent appearance is already implemented
- it is not future work

The old README and parts of the historical notes are stale on this point.

### 5. Observation Model

The observation model is in [blender_temp/gaussian_sr/observation_model.py](blender_temp/gaussian_sr/observation_model.py).

Supported modes:

- `identity`
- `area`
- `supersample_area`

The current CLI path uses `area` by default:

- render at the stage resolution
- area-downsample if needed to match the target

This is a partial implementation of the “explicit LR observation model” recommended in the research docs. What exists now is good enough for training and debugging, but it is still simpler than full pixel-footprint-aware observation modeling.

### 6. Warp Renderer

The renderer is a gsplat-style pipeline split between [blender_temp/gaussian_sr/warp_gsplat_autograd.py](blender_temp/gaussian_sr/warp_gsplat_autograd.py) and [blender_temp/gaussian_sr/warp_gsplat_kernels.py](blender_temp/gaussian_sr/warp_gsplat_kernels.py).

The active rendering stages are:

1. Project 3D Gaussians to screen space in Warp.
2. Compute tile coverage counts.
3. Prefix-scan the counts.
4. Enumerate tile intersections.
5. Sort by packed `(tile, depth)` key.
6. Build tile start/end ranges.
7. Rasterize front-to-back.
8. Optionally run a visibility/statistics pass over the prepared visibility state.

Important implementation details:

- The code uses grouped vector tensors rather than splitting `[N, 3]` or `[N, 4]` into scalar component arrays.
- The renderer can reuse a prepared visibility state for render/stat passes.
- Warp launches are recorded and reused on repeated non-grad paths.
- Warp launches use the PyTorch CUDA stream explicitly.
- Sorting can use Warp radix sort or fall back to `torch.sort`, subject to memory and safety checks.

The current differentiation boundary is:

- differentiable: projection and rasterization math
- not treated as differentiable: scan/sort/bin bookkeeping

That is aligned with gsplat practice and with the relevant research report.

### 7. Residual Head

The residual head is [ScaleAwareResidualHead](blender_temp/gaussian_sr/field.py).

It takes rendered latent feature maps plus:

- normalized pixel coordinates
- `log(scale_x)`
- `log(scale_y)`

and predicts a bounded RGB residual.

This is optional but enabled by default. It is conceptually similar to a small scale-aware continuous decoder layered on top of the Gaussian renderer.

### 8. Losses And Regularization

Photometric loss in [pipeline.py](blender_temp/gaussian_sr/pipeline.py):

- Charbonnier / L1-like term
- optional SSIM term

Regularization:

- depth TV
- mean opacity penalty
- mean scale penalty
- pose regularization with stage-dependent weight

Not implemented:

- LPIPS
- exposure compensation
- depth or normal supervision from Blender
- geometry-specific 2DGS/surfel constraints

### 9. Density Control

Density control is in [blender_temp/gaussian_sr/density_control.py](blender_temp/gaussian_sr/density_control.py).

It periodically:

- prunes weak Gaussians
- splits large important Gaussians
- clones small important Gaussians

The scoring signal mixes:

- parameter-gradient magnitude
- visibility / contribution
- minimum per-view visibility
- residual magnitude
- peak screen-space error
- average transmittance
- Gaussian scale

The current code also includes view-aware logic:

- per-view coverage summaries
- weak-view detection
- prune protection when any view is weak
- weak-view-aware split/clone filtering

This was introduced after debugging stage-3 black outputs and is central to the current troubleshooting story.

### 10. Training Loop

The main loop is [PoseFreeGaussianSR.fit()](blender_temp/gaussian_sr/pipeline.py).

Stage schedule used by the CLI:

| Stage | Scale | Steps | Example render size for 1920x1080 input |
|---|---:|---:|---:|
| 1 | 0.25 | 250 | 480x270 |
| 2 | 0.50 | 500 | 960x540 |
| 3 | 1.00 | 1000 | 1920x1080 |

At each step:

- choose all views or a subset, depending on `view_batch_size`
- render each selected view
- apply observation model and photometric loss
- accumulate optional density-control render stats
- backward
- grad clip
- optimizer step
- optionally run density control
- emit JSON progress events

There are two execution styles:

- eager/debug path with per-view observability
- compiled path that wraps stable train-step regions with `torch.compile`

### 11. Export

At the end of the CLI run, the program exports:

- `render_view0_<scale>x.(pt|png)`
- `render_view1_<scale>x.(pt|png)` if a second view exists

It does not currently export all views by default.

## torch.compile And Profiling

The pipeline defaults to `torch.compile`, but the active mode is intentionally conservative:

- `dynamic=None`
- `fullgraph=False`
- `mode="max-autotune-no-cudagraphs"`

Why:

- density control changes Gaussian count `N`
- stage transitions change spatial sizes
- Warp calls intentionally graph-break around compiled regions
- CUDA graph pools caused memory problems in earlier runs

Useful CLI switches:

- `--disable-torch-compile`
- `--profile-pytorch`

The profiler path is stage-scoped and writes per-stage traces and summaries when enabled.

## Logging, Checkpointing, And Debug Artifacts

The CLI writes:

- `progress.jsonl`
- `density_events.jsonl`
- `history.json`
- optional `checkpoints/`
- optional `debug-renders/`
- optional per-stage profiler traces

### `progress.jsonl`

This is the main execution log.

Important event types:

- `stage_start`
- `step_start`
- `view_start`
- `view_end`
- `gradient_stats`
- `step_end`
- `stage_end`
- `debug_checkpoint`
- `debug_preview`
- `diagnostic_view`
- `collapse_signal`
- `rollback_restore`

The most useful fields for debugging are:

- `photo_loss`
- `loss`
- `num_gaussians`
- `visible_count`
- `intersection_count`
- `render_total_s`
- `backward_s`
- `cuda_alloc_gib`
- `cuda_max_alloc_gib`

### `density_events.jsonl`

This records each density-control mutation.

Useful fields:

- `before`, `after`
- `pruned`, `split`, `cloned`
- `summary.visibility_mean`
- `summary.transmittance_mean`
- `summary.view_coverages`
- `summary.weak_view_indices`
- `summary.visible_fraction_of_best`
- `summary.intersection_fraction_of_best`
- `summary.prune_protected`
- `summary.split_top`
- `summary.clone_top`

### Checkpoints And Rollback

The debug observer in [blender_temp/cmd/main.py](blender_temp/cmd/main.py) can:

- save a checkpoint at each stage start
- save a checkpoint at each density event
- save preview renders periodically in the final stage
- detect visibility collapse
- restore the last safe checkpoint
- emit rollback previews before aborting

Checkpoint restoration is factored into [blender_temp/gaussian_sr/debug_checkpoint.py](blender_temp/gaussian_sr/debug_checkpoint.py).

There is not yet a first-class CLI flag to resume from a checkpoint. Resuming currently uses the Python helper directly.

## Commands We Actually Used For Debugging

Conservative CLI run:

```console
nix run .#blender-temp -- \
  --input-dir ./render_1920_1080 \
  --output-dir temp-output \
  --device cuda \
  --scale 1.0 \
  --max-frames 3 \
  --anchor-stride 8 \
  --view-batch-size 0 \
  --radius-clip-px 512
```

High-observability debug run:

```console
nix run .#blender-temp -- \
  --input-dir ./render_1920_1080 \
  --output-dir temp-output-debug \
  --device cuda \
  --scale 1.0 \
  --max-frames 3 \
  --anchor-stride 8 \
  --view-batch-size 0 \
  --radius-clip-px 512 \
  --disable-torch-compile \
  --verbose-progress
```

The second command is the most useful current debugging mode because it emits per-view visibility and timing information directly.

## Observed Runtime Behavior

This section summarizes the actual runs we used to debug the system. These `temp-output*` directories may not be included when the repository is shared, so the numbers are repeated here intentionally.

### A. Eager Debug Run (`temp-output-debug`)

Configuration:

- `max_frames=3`
- `anchor_stride=8`
- `view_batch_size=0` (all views each step)
- `radius_clip_px=512`
- `scale=1.0`
- `--disable-torch-compile`
- `--verbose-progress`

Average step timings:

| Stage | Steps | Avg step (s) | Avg render (s) | Avg backward (s) | Peak CUDA alloc (GiB) | Final Gaussian count | Final photo loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 250 | 0.1325 | 0.0545 | 0.0670 | 0.862 | 32400 | 0.198239 |
| 2 | 500 | 0.2916 | 0.1338 | 0.1420 | 2.880 | 34711 | 0.137564 |
| 3 | 1000 | 0.1493 | 0.0221 | 0.0847 | 9.129 | 35667 | 0.578366 |

Wall-clock stage durations from the log:

- stage 1: `33.22s`
- stage 2: `146.00s`
- stage 3: `160.78s`

The stage-3 averages are misleading in a useful way: stage 3 gets faster because the scene collapses and fewer Gaussians remain visible in the weak views. Lower step time there is not an improvement signal.

Density events in stage 3:

| Stage 3 density step | Gaussian count | `visibility_mean` | `transmittance_mean` | Mutation |
|---|---:|---:|---:|---|
| 1 | `34711 -> 35068` | `127.15` | `0.08869` | `split=101`, `clone=256` |
| 101 | `35068 -> 35365` | `118.59` | `0.05954` | `split=42`, `clone=256` |
| 201 | `35365 -> 35651` | `61.09` | `0.04222` | `split=34`, `clone=252` |
| 301 | `35651 -> 35664` | `2.235` | `0.003281` | `split=2`, `clone=11` |
| 401 | `35664 -> 35665` | `0.184` | `0.000224` | `split=1`, `clone=0` |
| 501 | `35665 -> 35665` | `0.109` | `0.000084` | `split=0`, `clone=0` |

The density controller stops being able to recover once the scene reaches the `step 301` to `step 401` range.

Per-view collapse snapshots from saved previews:

| Preview | View 0 visible | View 1 visible | View 2 visible |
|---|---:|---:|---:|
| Stage 3 step 50 | 8477 | 7147 | 4981 |
| Stage 3 step 150 | 6105 | 4482 | 427 |
| Stage 3 step 300 | 2880 | 48 | 3 |
| Stage 3 step 400 | 2818 | 8 | 1 |
| Stage 3 step 1000 | 2951 | 1 | 0 |

Final exported tensors from that run:

- `render_view0_1x.pt`: nonzero, sparse, `sum=17054.93`, `max=0.2426`
- `render_view1_1x.pt`: exactly zero, `sum=0.0`, `max=0.0`

So the black-output failure is real scene collapse, not a PNG-save bug.

One subtle but important point: `temp-output-debug` captured repeated collapse checkpoints and preview renders, but it did not perform a rollback restore. That happened in a later dedicated rollback run described next.

### B. Collapse / Rollback Run (`temp-output-rollback-debug2`)

This run used the debug observer with collapse detection and rollback enabled.

Wall-clock stage durations from the log:

- stage 1: `33.87s`
- stage 2: `146.10s`
- stage 3: aborted after step 279 rather than running to the nominal 1000-step end

Important stage-3 density snapshots:

| Stage 3 density step | Gaussian count | `visibility_mean` | View visible counts |
|---|---:|---:|---|
| 1 | `34790 -> 35119` | `41.57` | `10137 / 7464 / 7426` |
| 101 | `35119 -> 35435` | `34.12` | `7375 / 4935 / 2007` |
| 201 | `35435 -> 35644` | `11.17` | `5446 / 2781 / 464` |

Collapse detector trigger:

- at stage 3 step 279, preview summary showed `3626 / 66 / 8` visible Gaussians

Rollback result:

- restored the last safe checkpoint
- rollback preview showed `5620 / 2910 / 515`

This confirmed that:

- checkpointing and rollback worked
- the failure is gradual and detectable before final export
- the system can roll back to a materially healthier state

### C. Resumed Stage-3 Policy Test (`temp-output-resume-policy-v1`)

We then resumed from the stage-3 checkpoint around the first problematic density event rather than rerunning earlier stages.

Why this matters:

- it let us validate a density-policy change against the exact failure region
- it avoided wasting time replaying stable earlier stages

This resumed run used stage-1 and stage-2 zero-step placeholders and then ran only a stage-3 tail:

- stage 1 placeholder: about `13 ms`
- stage 2 placeholder: about `3 ms`
- resumed stage 3 tail: 180 steps in about `49.70s`
- exactly one density event during the resumed tail

The changed policy:

- added relative weak-view thresholds:
  - `min_view_visible_fraction_of_best = 0.20`
  - `min_view_intersection_fraction_of_best = 0.20`
- treated non-empty `weak_view_indices` as unsafe for checkpoint safety

Key comparison at the analogous full-resolution density event:

Old policy from `temp-output-rollback-debug2`:

- stage 3 step 201
- view coverage `5446 / 2781 / 464`
- `weak_view_indices = []`
- mutation `split=29`, `clone=180`

New policy from `temp-output-resume-policy-v1`:

- resumed stage 3 step 101
- view coverage `5455 / 2862 / 491`
- `weak_view_indices = [2]`
- `prune_protected = true`
- mutation `split=1`, `clone=63`

Outcome:

- the new policy delayed and softened collapse
- it did not fully prevent it

By the end of the resumed run:

- view coverage was still roughly `3512 / 102 / 12`
- photo loss was still poor (`0.543731`)

So the current view-aware density policy helps, but is not sufficient.

### D. Compiled Run Behavior (`temp-output`)

We also observed useful runtime behavior in compiled mode.

On a representative compiled run:

- stage-1 first step took `15.49s`
- subsequent early stage-1 steps dropped to about `0.106s` to `0.110s`
- stage-3 steady-state steps near the failure region were around `0.066s`
- a later density step spiked to `3.06s`

This is consistent with:

- compiler warmup
- Triton autotuning
- recompilation caused by dynamic shapes
- density-control topology changes

Average compiled step time from that partially completed run:

| Stage | Completed steps | Avg step (s) | Peak CUDA alloc (GiB) | Last photo loss |
|---|---:|---:|---:|---:|
| 1 | 250 | 0.1882 | 0.711 | 0.191531 |
| 2 | 500 | 0.5772 | 2.455 | 0.130792 |
| 3 | 601 | 0.4435 | 7.937 | 0.577192 |

The compiled path is fast enough to be useful, but it is harder to diagnose because it does not emit the same per-view detail as the eager debug path.

## Current Failure Mode

The current central failure mode is:

1. training begins with reasonable coverage in all three views
2. during stage 3, coverage concentrates into view 0
3. views 1 and 2 lose visibility and intersection support
4. density control eventually stops having enough signal to split/clone useful Gaussians for the weak views
5. one or more non-anchor outputs become all black

The failure is not caused by image export and is not primarily caused by save-time quantization.

It is a training-time scene-collapse problem.

## What We Think Is Causing The Failure

The strongest current hypotheses are:

- the optimization objective still allows the scene to sacrifice weak-view coverage while improving the dominant view
- density control only intervenes periodically, not continuously
- the current weak-view policy acts on density events, but not on the optimization objective itself
- once a view becomes weak enough, it no longer provides strong densification candidates
- stage-3 full-resolution training amplifies this because weak views have fewer surviving visible Gaussians and fewer intersections

The current implementation already rules out some earlier hypotheses:

- final export bug: fixed
- stale mixed-run logs: fixed
- density-control-final-stage flag bug: fixed
- view-batch starvation from `view_batch_size=1`: preflight checks added

## Historical Compiler / Runtime Issues

These are worth mentioning because an external researcher will not see the old logs:

- We hit `torch.compile` graph breaks in `so3_exp_map()` due to boolean masked writes lowering through `nonzero()` / `index_put`. This was rewritten to a branchless `torch.where()` form.
- We hit `torch.compile` and CUDA graph lifetime issues when small compiled helper functions returned tensors that lived across later compiled calls. The compile boundary was moved upward to stable train-step regions.
- We observed CUDA graph private-pool growth and late-stage OOM in compiled density runs. The active compile mode is now `max-autotune-no-cudagraphs` to avoid that path.
- Density-control shape changes and stage-resolution changes still cause recompilation pressure. `dynamic=None` is used rather than `dynamic=False`, but topology changes still matter.
- A dedicated `--disable-torch-compile` flag exists because the eager debug path remains essential for diagnosing collapse.

## What An External Researcher Should Focus On

If you are reading this repository to help debug or redesign it, the highest-value topics are:

1. How to preserve per-view coverage during full-resolution optimization.
2. Whether density control should become more explicitly constrained by per-view minimum coverage.
3. Whether the training loss should include a direct anti-collapse term based on per-view visibility or intersection counts.
4. Whether stage 3 should use a stricter rollback/abort policy much earlier than the current final collapse threshold.
5. Whether the observation model is still too weak for stable LR-supervised arbitrary-scale training.

Lower-priority but still relevant topics:

1. Better compile behavior under dynamic topology.
2. Better memory behavior for repeated Warp calls and large sort buffers.
3. A tile-optimized Warp raster kernel or manual backward kernel.
4. More faithful multi-scale anti-aliasing and observation modeling.

## Research Documents In This Repository: What To Trust

The in-tree research markdown is useful, but it is not the source of truth. The code is.

Also note that the reports contain browsing-era citation markup and historical provenance notes. They are useful as design notes, not as a clean or current bibliography.

### `deep-research-report-01.md`

Use it for:

- conceptual motivation
- why this should be solved as a shared continuous 3D scene problem rather than independent 2D SR
- why anti-aliasing and explicit LR observation matter
- why density control is core to a Gaussian-based approach

Do not use it as a description of the exact current implementation.

Current mismatches:

- it assumes known/fixed poses
- it discusses richer preprocessing and observation ideas than the current code actually implements
- it suggests initialization paths we do not currently use

### `deep-research-report-02.md`

Use it for:

- provenance and porting rationale around ContinuousSR / gsplat lineage
- license and reuse cautions
- understanding why a gsplat-style pipeline was a useful reference point

Do not use it as a description of the active system architecture.

It is mostly background, not a current design document.

### `deep-research-report-03.md`

This is the most relevant of the three to the current renderer.

Use it for:

- the renderer decomposition
- the differentiability boundary
- memory/buffer reasoning
- why the Warp port naturally breaks into projection, intersects, sort, tile bins, rasterization

Still treat it carefully:

- it assumes known poses, while the active code optimizes poses
- it is closer to the renderer than to the full pipeline
- it describes opportunities such as tile-optimized kernels and custom backward paths that are not yet implemented

### `sr_warp_research_notes.md`

This file should now be treated as historical notes, not as a separate authoritative document.

Useful parts:

- the later sections that say the right abstraction is shared 3D reconstruction, not 2D SR
- the summary that projection/rasterization are differentiable but sort/bin bookkeeping is not
- the discussion that anti-aliasing / finite-area LR observation is not optional

Stale parts that should mostly be ignored:

- early script-by-script comparisons against now-removed files
- old claims that density control, observation modeling, SH appearance, profiling, or validation were missing
- historical “next steps” that are already implemented

This README folds in the still-accurate parts of those notes.

## Source Of Truth Hierarchy

If the markdown and code disagree, trust them in this order:

1. current code under `blender_temp/gaussian_sr/`
2. current CLI behavior in `blender_temp/cmd/main.py`
3. this README
4. `deep-research-report-03.md`
5. `deep-research-report-01.md`
6. `deep-research-report-02.md`
7. `sr_warp_research_notes.md`

## Important Caveats For External Review

- The repository may be shared without the input images.
- The repository may be shared without any `temp-output*` directories.
- The observations in the runtime sections above are therefore critical context and are intentionally duplicated from deleted artifacts.
- The current system is not “failing to render”; it is “failing to maintain multi-view support late in training.”
- Any proposed fix should be evaluated against stage-3 per-view coverage, not only final loss.

## Immediate Open Questions

These are the current practical questions we want external input on:

1. What is the right view-aware densification/pruning policy so that weak-view coverage is preserved without freezing the scene topology completely?
2. Should per-view coverage enter the optimization loss directly rather than only density-control heuristics?
3. Is the current area-based LR observation model too weak, especially in stage 3?
4. Is the first-view anchor-lattice initialization structurally biased toward view 0 in a way that the optimizer never fully escapes?
5. Should stage 3 switch to a different curriculum, different density cadence, or stronger rollback threshold?

## Summary

The repository is a real, functioning pose-free multi-view Gaussian SR system with a Warp renderer backend. It is no longer a bare port scaffold. The dominant unsolved problem is late-stage view collapse, not basic implementation completeness.

If you are trying to help this project, focus on preserving multi-view coverage during stage-3 optimization.
