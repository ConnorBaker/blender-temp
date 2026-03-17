from dataclasses import dataclass, field as dc_field
from typing import Literal


@dataclass
class CameraInit:
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    learn_intrinsics: bool = False
    default_fov_degrees: float = 50.0


@dataclass
class RenderConfig:
    backend: Literal["warp", "helion"] = "warp"
    backward_impl: Literal["warp_tape", "reference", "hybrid", "helion"] = "hybrid"
    tile_size: int = 16
    eps2d: float = 0.30
    near: float = 1.0e-2
    far: float = 1.0e3
    radius_clip_px: float = 0.0
    max_sort_buffer_bytes: int | None = 2 * 1024 * 1024 * 1024
    alpha_threshold: float = 1.0e-4
    transmittance_threshold: float = 1.0e-4
    bbox_extent_sigma: float = 3.0
    antialiased_opacity: bool = True
    bg_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    helion_static_shapes: bool = True
    helion_runtime_autotune: bool = True


@dataclass
class ObservationConfig:
    mode: Literal["identity", "area", "supersample_area"] = "area"
    supersample_factor: float = 1.0


@dataclass
class AppearanceConfig:
    mode: Literal["constant", "sh"] = "sh"
    sh_degree: int = 2
    residual_scale: float = 0.5


@dataclass
class DensityControlConfig:
    enabled: bool = True
    disable_final_stage: bool = False
    start_step: int = 250
    every_steps: int = 100
    final_stage_every_steps: int | None = 25
    debug_topk: int = 4
    screen_error_bins: int = 4
    min_gaussians: int = 256
    max_gaussians: int = 200000
    opacity_prune_threshold: float = 0.02
    prune_visibility_threshold: float = 1.0e-5
    densify_opacity_min: float = 0.05
    densify_visibility_threshold: float = 5.0e-5
    split_transmittance_threshold: float = 5.0e-3
    clone_transmittance_threshold: float = 1.0e-3
    grad_threshold: float = 1.0e-3
    gradient_score_weight: float = 1.0
    visibility_score_weight: float = 0.75
    min_view_score_weight: float = 1.0
    residual_score_weight: float = 1.0
    screen_error_peak_weight: float = 0.5
    transmittance_score_weight: float = 0.50
    scale_score_weight: float = 0.25
    min_view_visible_gaussians: int = 64
    min_view_intersection_count: int = 1024
    min_view_visible_fraction_of_best: float = 0.50
    min_view_intersection_fraction_of_best: float = 0.50
    coverage_floor_visible_fraction: float = 0.50
    coverage_floor_intersection_fraction: float = 0.50
    use_normalized_density_scores: bool = True
    weak_view_reseed_budget_per_view: int = 256
    weak_view_reseed_trigger_fraction_of_best: float = 0.90
    weak_view_reseed_residual_quantile: float = 0.90
    weak_view_reseed_match_radius_px: float = 8.0
    weak_view_reseed_depth_match_radius_px: float = 16.0
    weak_view_reseed_init_opacity: float = 0.05
    weak_view_reseed_min_opacity: float = 0.0
    weak_view_reseed_protect_steps: int = 25
    weak_view_error_fraction_of_worst: float = 0.0
    weak_view_split_topk: int = 0
    weak_view_clone_topk: int = 0
    weak_view_reseed_target_luma_min: float = -1.0
    freeze_after_stable_events: int = 3
    freeze_min_visible_fraction: float = 0.90
    freeze_min_intersection_fraction: float = 0.95
    split_topk: int = 256
    clone_topk: int = 256
    split_scale_quantile: float = 0.7
    clone_scale_quantile: float = 0.3
    split_shrink_factor: float = 0.8
    split_offset_scale: float = 0.75
    clone_jitter_scale: float = 0.25


@dataclass
class FieldConfig:
    anchor_stride: int = 1
    feature_dim: int = 8
    gaussian_capacity: int = 65536
    overflow_policy: Literal["freeze_density_and_warn", "abort"] = "freeze_density_and_warn"
    min_depth: float = 0.05
    init_depth: float = 1.0
    init_scale_xy: float = 0.75
    init_scale_z: float = 0.10
    min_log_scale: float = -4.5
    init_opacity: float = 0.20
    use_residual_head: bool = True
    residual_hidden_dim: int = 64
    residual_scale: float = 0.05


@dataclass
class TrainConfig:
    stage_scales: tuple[float, ...] = (0.25, 0.5, 1.0)
    steps_per_stage: tuple[int, ...] = (500, 500, 1000)
    final_stage_max_steps: int = 0
    final_stage_early_stop_patience: int = 0
    final_stage_early_stop_min_step: int = 0
    final_stage_early_stop_loss_delta: float = 0.0
    # Per-parameter learning rates following 3DGS (Kerbl et al., "3D Gaussian
    # Splatting for Real-Time Radiance Field Rendering", SIGGRAPH 2023).
    # Position LR decays log-linearly; all others are constant per stage.
    # Values are higher than vanilla 3DGS (which starts from known poses)
    # because our pose-free setting needs faster initial convergence.
    lr_position: float = 1.0e-3       # 3DGS uses 1.6e-4 (known poses)
    lr_position_final: float = 1.0e-5  # 100x decay ratio matches 3DGS
    lr_opacity: float = 5.0e-2         # 3DGS default
    lr_scaling: float = 5.0e-3         # 3DGS default
    lr_rotation: float = 1.0e-3        # 3DGS default
    lr_color: float = 2.5e-3           # 3DGS features_dc default
    lr_latent: float = 2.5e-3          # Same as color features
    lr_sh: float = 1.25e-4             # 3DGS features_rest (features_dc / 20)
    lr_camera: float = 2.0e-3
    lr_residual: float = 1.0e-3
    photometric_l1_weight: float = 0.8
    photometric_ssim_weight: float = 0.2
    depth_tv_weight: float = 5.0e-3
    opacity_weight: float = 1.0e-4
    scale_weight: float = 1.0e-4
    pose_weight_stage0: float = 1.0e-3
    pose_weight_final: float = 1.0e-5
    grad_clip: float = 1.0
    use_phasecorr_init: bool = True
    view_batch_size: int = 0
    ordinary_step_view_batch: int = 2
    final_stage_views_per_microbatch: int = 2
    print_every: int = 50


@dataclass
class PoseFreeGaussianConfig:
    camera: CameraInit = dc_field(default_factory=CameraInit)
    render: RenderConfig = dc_field(default_factory=RenderConfig)
    observation: ObservationConfig = dc_field(default_factory=ObservationConfig)
    appearance: AppearanceConfig = dc_field(default_factory=AppearanceConfig)
    density: DensityControlConfig = dc_field(default_factory=DensityControlConfig)
    field: FieldConfig = dc_field(default_factory=FieldConfig)
    train: TrainConfig = dc_field(default_factory=TrainConfig)


__all__ = [
    "CameraInit",
    "RenderConfig",
    "ObservationConfig",
    "AppearanceConfig",
    "DensityControlConfig",
    "FieldConfig",
    "TrainConfig",
    "PoseFreeGaussianConfig",
]
