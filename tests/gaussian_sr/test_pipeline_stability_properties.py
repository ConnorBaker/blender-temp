import pytest
import torch
from hypothesis import given, settings

from blender_temp.gaussian_sr import PoseFreeGaussianConfig, PoseFreeGaussianSR
from blender_temp.gaussian_sr.density_control import DensityControlResult
from blender_temp.gaussian_sr.warp_runtime import _WARP_AVAILABLE
from blender_temp.gaussian_sr import pipeline as pipeline_module

from .strategies import tiny_image_batches

CUDA_WARP_AVAILABLE = bool(_WARP_AVAILABLE and torch.cuda.is_available())
GPU_STABILITY_SETTINGS = settings(deadline=None, max_examples=6)


def _all_parameters_finite(module: torch.nn.Module) -> bool:
    return all(torch.isfinite(param).all() for param in module.parameters())


def test_make_optional_compiled_uses_automatic_dynamic_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    compile_kwargs: dict[str, object] = {}

    def fake_compile(fn, **kwargs):
        compile_kwargs.update(kwargs)
        return fn

    monkeypatch.setattr(pipeline_module.torch, "compile", fake_compile)

    wrapped = pipeline_module._make_optional_compiled(lambda x: x + 1, "dummy")
    assert wrapped(1) == 2
    assert compile_kwargs["dynamic"] is None
    assert compile_kwargs["fullgraph"] is False
    assert compile_kwargs["mode"] == "max-autotune-no-cudagraphs"


def test_make_optional_compiled_respects_disabled_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    compile_calls: list[dict[str, object]] = []

    def fake_compile(fn, **kwargs):
        compile_calls.append(kwargs)
        return fn

    monkeypatch.setattr(pipeline_module.torch, "compile", fake_compile)
    monkeypatch.setattr(pipeline_module, "_TORCH_COMPILE_ENABLED", False)

    wrapped = pipeline_module._make_optional_compiled(lambda x: x + 1, "dummy")
    assert wrapped(1) == 2
    assert compile_calls == []


def test_make_optional_compiled_clears_cuda_state_after_oom(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeOOM(torch.OutOfMemoryError):
        pass

    class FakeCompiled:
        def __call__(self, *_args, **_kwargs):
            raise FakeOOM("oom")

    cleared: list[str] = []

    monkeypatch.setattr(pipeline_module.torch, "compile", lambda fn, **kwargs: FakeCompiled())
    monkeypatch.setattr(pipeline_module, "_clear_compiled_cuda_state", lambda: cleared.append("cleared"))

    wrapped = pipeline_module._make_optional_compiled(lambda x: x + 1, "dummy")
    assert wrapped(1) == 2
    assert cleared == ["cleared"]


@pytest.mark.skipif(not CUDA_WARP_AVAILABLE, reason="requires CUDA + NVIDIA Warp")
@GPU_STABILITY_SETTINGS
@given(images=tiny_image_batches(min_views=1, max_views=2, min_side=4, max_side=6))
def test_single_tiny_training_step_keeps_losses_and_parameters_finite(images: torch.Tensor) -> None:
    device = torch.device("cuda")
    images = images.to(device=device, dtype=torch.float32)

    cfg = PoseFreeGaussianConfig()
    cfg.camera.learn_intrinsics = False
    cfg.appearance.mode = "constant"
    cfg.field.use_residual_head = False
    cfg.field.anchor_stride = 2
    cfg.field.feature_dim = 2
    cfg.train.stage_scales = (1.0,)
    cfg.train.steps_per_stage = (1,)
    cfg.train.lr_field = 1.0e-4
    cfg.train.lr_camera = 1.0e-4
    cfg.train.lr_residual = 1.0e-4
    cfg.train.photometric_l1_weight = 1.0
    cfg.train.photometric_ssim_weight = 0.0
    cfg.train.grad_clip = 0.5
    cfg.train.view_batch_size = 1
    cfg.train.print_every = 0
    cfg.density.enabled = False

    pipeline = PoseFreeGaussianSR.from_images(images, intrinsics=None, config=cfg).to(device)
    history = pipeline.fit(images)

    assert len(history["loss"]) == 1
    assert len(history["photo"]) == 1
    assert len(history["reg"]) == 1
    assert torch.isfinite(torch.tensor(history["loss"], dtype=torch.float32)).all()
    assert torch.isfinite(torch.tensor(history["photo"], dtype=torch.float32)).all()
    assert torch.isfinite(torch.tensor(history["reg"], dtype=torch.float32)).all()
    assert _all_parameters_finite(pipeline)


@pytest.mark.skipif(not CUDA_WARP_AVAILABLE, reason="requires CUDA + NVIDIA Warp")
def test_constant_appearance_pipeline_produces_finite_camera_geometry_gradients() -> None:
    device = torch.device("cuda")
    images = torch.zeros((2, 3, 16, 16), device=device, dtype=torch.float32)

    cfg = PoseFreeGaussianConfig()
    cfg.camera.learn_intrinsics = False
    cfg.appearance.mode = "constant"
    cfg.field.use_residual_head = False
    cfg.field.anchor_stride = 1
    cfg.field.feature_dim = 2
    cfg.train.print_every = 0
    cfg.density.enabled = False

    pipeline = PoseFreeGaussianSR.from_images(images, intrinsics=None, config=cfg).to(device)
    opt = pipeline._make_optimizer(cfg.train)
    opt.zero_grad(set_to_none=True)

    R_all, t_all = pipeline.camera_model.world_to_camera()
    out = pipeline.render_with_pose(R_all[1], t_all[1], 4, 4, return_aux=False)
    loss = out["rgb"].sum()
    loss.backward()

    grad = pipeline.camera_model.pose_rest.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert float(grad.abs().sum().item()) > 0.0


def test_fit_skips_density_control_calls_in_final_stage_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    images = torch.zeros((2, 3, 4, 4), dtype=torch.float32)

    cfg = PoseFreeGaussianConfig()
    cfg.camera.learn_intrinsics = False
    cfg.appearance.mode = "constant"
    cfg.field.use_residual_head = False
    cfg.field.anchor_stride = 2
    cfg.field.feature_dim = 2
    cfg.train.stage_scales = (0.5, 0.75, 1.0)
    cfg.train.steps_per_stage = (1, 1, 1)
    cfg.train.photometric_l1_weight = 1.0
    cfg.train.photometric_ssim_weight = 0.0
    cfg.train.print_every = 0
    cfg.train.view_batch_size = 1
    cfg.density.enabled = True
    cfg.density.start_step = 0
    cfg.density.every_steps = 1
    cfg.density.disable_final_stage = True

    pipeline = PoseFreeGaussianSR.from_images(images, intrinsics=None, config=cfg)
    progress_events: list[dict] = []
    density_calls: list[int] = []
    render_call_modes: list[tuple[str, bool]] = []
    prepared_marker = object()

    def fake_render_with_pose(
        _R_cw: torch.Tensor,
        _t_cw: torch.Tensor,
        out_h: int,
        out_w: int,
        return_aux: bool = True,
        stats_mode: str = "full",
        return_prepared: bool = False,
        profile: bool = False,
    ) -> dict:
        del return_aux, profile
        render_call_modes.append((stats_mode, return_prepared))
        scalar = torch.sigmoid(pipeline.field_model.rgb_logit[:1]).mean()
        rgb = scalar.expand(3, out_h, out_w)
        result = {
            "rgb": rgb,
            "field": {},
            "render_stats": {
                "meta_gaussian_count": torch.tensor(pipeline.field_model.num_gaussians, dtype=torch.int64),
                "meta_visible_count": torch.tensor(1, dtype=torch.int64),
                "meta_intersection_count": torch.tensor(1, dtype=torch.int64),
                "meta_tile_count": torch.tensor(1, dtype=torch.int64),
                "meta_tiles_x": torch.tensor(1, dtype=torch.int64),
                "meta_tiles_y": torch.tensor(1, dtype=torch.int64),
                "meta_render_width": torch.tensor(out_w, dtype=torch.int64),
                "meta_render_height": torch.tensor(out_h, dtype=torch.int64),
            },
            "profile": {},
        }
        if return_prepared:
            result["_prepared_visibility"] = prepared_marker
            result["_opacity"] = torch.ones(pipeline.field_model.num_gaussians, dtype=torch.float32)
        return result

    def fake_observe_and_photometric_loss(
        rgb: torch.Tensor, _target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return rgb, rgb.mean()

    def fake_regularization(_field: dict, _stage_alpha: float) -> torch.Tensor:
        return pipeline.field_model.depth_raw.sum() * 0.0

    def fake_render_stats_from_prepared(
        prepared: object,
        opacity: torch.Tensor,
        residual_map: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        assert prepared is prepared_marker
        assert opacity.shape[0] == pipeline.field_model.num_gaussians
        del residual_map
        n = pipeline.field_model.num_gaussians
        return {
            "contrib": torch.ones(n, dtype=torch.float32),
            "hits": torch.ones(n, dtype=torch.float32),
            "transmittance": torch.ones(n, dtype=torch.float32),
            "residual": torch.zeros(n, dtype=torch.float32),
            "error_map": torch.zeros(n, 1, dtype=torch.float32),
        }

    def fake_apply_density_control(
        field_model,
        _cfg,
        step: int,
        render_stats: dict[str, torch.Tensor] | None = None,
    ) -> DensityControlResult:
        del render_stats
        density_calls.append(step)
        return DensityControlResult(
            ran=True,
            changed=False,
            pruned=0,
            split=0,
            cloned=0,
            before=field_model.num_gaussians,
            after=field_model.num_gaussians,
            debug=None,
        )

    monkeypatch.setattr(pipeline, "render_with_pose", fake_render_with_pose)
    monkeypatch.setattr(pipeline, "_observe_and_photometric_loss", fake_observe_and_photometric_loss)
    monkeypatch.setattr(pipeline, "_regularization_impl", fake_regularization)
    monkeypatch.setattr(pipeline, "_render_stats_from_prepared", fake_render_stats_from_prepared)
    monkeypatch.setattr(pipeline_module, "apply_density_control", fake_apply_density_control)

    history = pipeline.fit(images, progress_event_callback=progress_events.append)

    assert density_calls == [0, 1]
    assert render_call_modes == [("meta", True), ("meta", True), ("meta", False)]
    assert all(event["stage_index"] != 2 for event in history["density_events"])
    assert any(
        event["event"] == "step_start" and event["stage_index"] == 2 and event["density_due"] is False
        for event in progress_events
    )


def test_render_with_pose_meta_stats_prepares_once_and_skips_full_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    images = torch.zeros((1, 3, 4, 4), dtype=torch.float32)

    cfg = PoseFreeGaussianConfig()
    cfg.camera.learn_intrinsics = False
    cfg.appearance.mode = "constant"
    cfg.field.use_residual_head = False
    cfg.field.anchor_stride = 2
    cfg.field.feature_dim = 2
    cfg.train.print_every = 0
    cfg.density.enabled = False

    pipeline = PoseFreeGaussianSR.from_images(images, intrinsics=None, config=cfg)
    values_channels = 3 + cfg.field.feature_dim + 1
    call_order: list[tuple[str, object]] = []

    def fake_prepare_render_payload(
        _base_intrinsics: torch.Tensor,
        _render_intrinsics: torch.Tensor,
        _R_cw: torch.Tensor,
        _t_cw: torch.Tensor,
    ):
        n = pipeline.field_model.num_gaussians
        field = {}
        viewmat = torch.eye(4, dtype=torch.float32)
        K = torch.eye(3, dtype=torch.float32)
        means = torch.zeros((n, 3), dtype=torch.float32)
        quat = torch.zeros((n, 4), dtype=torch.float32)
        quat[:, 0] = 1.0
        scale = torch.full((n, 3), 0.1, dtype=torch.float32)
        opacity = torch.ones(n, dtype=torch.float32)
        values = torch.zeros((n, values_channels), dtype=torch.float32)
        background = torch.zeros(values_channels, dtype=torch.float32)
        return field, viewmat, K, means, quat, scale, opacity, values, background

    def fake_render_values_warp(**kwargs):
        call_order.append(("values", kwargs["return_prepared"]))
        packed = torch.zeros((kwargs["height"], kwargs["width"], values_channels), dtype=torch.float32)
        if kwargs["return_prepared"]:
            return packed, prepared_marker
        return packed

    class PreparedMarker:
        def meta(self_inner):
            call_order.append(("meta", self_inner))
            return {
                "meta_gaussian_count": torch.tensor(pipeline.field_model.num_gaussians, dtype=torch.int64),
                "meta_visible_count": torch.tensor(1, dtype=torch.int64),
                "meta_intersection_count": torch.tensor(1, dtype=torch.int64),
                "meta_tile_count": torch.tensor(1, dtype=torch.int64),
                "meta_tiles_x": torch.tensor(1, dtype=torch.int64),
                "meta_tiles_y": torch.tensor(1, dtype=torch.int64),
                "meta_render_width": torch.tensor(4, dtype=torch.int64),
                "meta_render_height": torch.tensor(4, dtype=torch.int64),
            }

    prepared_marker = PreparedMarker()

    def fail_render_stats_prepared_warp(*_args, **_kwargs):
        raise AssertionError("full render_stats_prepared_warp should not run in meta mode")

    def fake_postprocess_rgb(_packed: torch.Tensor, out_h: int, out_w: int):
        rgb = torch.zeros((3, out_h, out_w), dtype=torch.float32)
        latent = torch.zeros((cfg.field.feature_dim, out_h, out_w), dtype=torch.float32)
        alpha = torch.zeros((1, out_h, out_w), dtype=torch.float32)
        return rgb, latent, alpha

    monkeypatch.setattr(pipeline, "_prepare_render_payload", fake_prepare_render_payload)
    monkeypatch.setattr(pipeline, "_postprocess_rgb", fake_postprocess_rgb)
    monkeypatch.setattr(pipeline_module, "render_values_warp", fake_render_values_warp)
    monkeypatch.setattr(pipeline_module, "render_stats_prepared_warp", fail_render_stats_prepared_warp)

    out = pipeline.render_with_pose(
        torch.eye(3, dtype=torch.float32),
        torch.zeros(3, dtype=torch.float32),
        4,
        4,
        return_aux=True,
        stats_mode="meta",
        return_prepared=True,
    )

    assert out["_prepared_visibility"] is prepared_marker
    assert torch.equal(out["_opacity"], torch.ones(pipeline.field_model.num_gaussians, dtype=torch.float32))
    assert call_order == [("values", True), ("meta", prepared_marker)]


def test_fit_progress_callback_does_not_enable_sync_profile_and_step_callback_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    images = torch.zeros((1, 3, 4, 4), dtype=torch.float32)

    cfg = PoseFreeGaussianConfig()
    cfg.camera.learn_intrinsics = False
    cfg.appearance.mode = "constant"
    cfg.field.use_residual_head = False
    cfg.field.anchor_stride = 2
    cfg.field.feature_dim = 2
    cfg.train.stage_scales = (1.0,)
    cfg.train.steps_per_stage = (2,)
    cfg.train.photometric_l1_weight = 1.0
    cfg.train.photometric_ssim_weight = 0.0
    cfg.train.print_every = 0
    cfg.density.enabled = False

    pipeline = PoseFreeGaussianSR.from_images(images, intrinsics=None, config=cfg)
    profile_flags: list[bool] = []
    progress_events: list[dict] = []
    step_calls: list[tuple[int, int, int]] = []

    def fake_render_with_pose(
        _R_cw: torch.Tensor,
        _t_cw: torch.Tensor,
        out_h: int,
        out_w: int,
        return_aux: bool = True,
        stats_mode: str = "full",
        return_prepared: bool = False,
        profile: bool = False,
    ) -> dict:
        del return_aux, stats_mode, return_prepared
        profile_flags.append(profile)
        rgb = torch.zeros((3, out_h, out_w), dtype=torch.float32)
        return {
            "rgb": rgb,
            "field": {},
            "render_stats": {
                "meta_gaussian_count": torch.tensor(pipeline.field_model.num_gaussians, dtype=torch.int64),
                "meta_visible_count": torch.tensor(1, dtype=torch.int64),
                "meta_intersection_count": torch.tensor(1, dtype=torch.int64),
                "meta_tile_count": torch.tensor(1, dtype=torch.int64),
                "meta_tiles_x": torch.tensor(1, dtype=torch.int64),
                "meta_tiles_y": torch.tensor(1, dtype=torch.int64),
                "meta_render_width": torch.tensor(out_w, dtype=torch.int64),
                "meta_render_height": torch.tensor(out_h, dtype=torch.int64),
            },
        }

    def fake_observe_and_photometric_loss(
        rgb: torch.Tensor, _target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return rgb, rgb.mean()

    def fake_regularization(_field: dict, _stage_alpha: float) -> torch.Tensor:
        return pipeline.field_model.depth_raw.sum() * 0.0

    monkeypatch.setattr(pipeline, "render_with_pose", fake_render_with_pose)
    monkeypatch.setattr(pipeline, "_observe_and_photometric_loss", fake_observe_and_photometric_loss)
    monkeypatch.setattr(pipeline, "_regularization_impl", fake_regularization)

    history = pipeline.fit(
        images,
        progress_event_callback=progress_events.append,
        step_callback=lambda stage_idx, step_idx, global_step: step_calls.append((stage_idx, step_idx, global_step)),
    )

    assert len(history["loss"]) == 2
    assert profile_flags == [False, False]
    assert step_calls == [(0, 0, 0), (0, 1, 1)]
    assert any(event["event"] == "step_end" for event in progress_events)


def test_regularization_without_render_field_stays_finite_after_density_growth() -> None:
    images = torch.zeros((1, 3, 4, 4), dtype=torch.float32)

    cfg = PoseFreeGaussianConfig()
    cfg.camera.learn_intrinsics = False
    cfg.appearance.mode = "constant"
    cfg.field.use_residual_head = False
    cfg.field.anchor_stride = 2
    cfg.field.feature_dim = 2
    cfg.train.print_every = 0
    cfg.density.enabled = False

    pipeline = PoseFreeGaussianSR.from_images(images, intrinsics=None, config=cfg)
    pipeline.field_model.clone_gaussians(torch.tensor([0], dtype=torch.long))

    reg = pipeline._regularization(None, stage_alpha=0.5)

    assert reg.ndim == 0
    assert torch.isfinite(reg)


def test_rebuild_optimizer_after_density_resets_all_optimizer_state() -> None:
    images = torch.zeros((2, 3, 4, 4), dtype=torch.float32)

    cfg = PoseFreeGaussianConfig()
    cfg.camera.learn_intrinsics = False
    cfg.appearance.mode = "constant"
    cfg.field.use_residual_head = False
    cfg.field.anchor_stride = 2
    cfg.field.feature_dim = 2
    cfg.train.print_every = 0
    cfg.density.enabled = False

    pipeline = PoseFreeGaussianSR.from_images(images, intrinsics=None, config=cfg)
    opt = pipeline._make_optimizer(cfg.train)

    for param in opt.param_groups[0]["params"]:
        opt.state[param] = {
            "step": torch.tensor(7.0),
            "exp_avg": torch.full_like(param, 3.0),
            "exp_avg_sq": torch.full_like(param, 5.0),
        }

    camera_param = opt.param_groups[1]["params"][0]
    camera_state = {
        "step": torch.tensor(11.0),
        "exp_avg": torch.full_like(camera_param, 2.0),
        "exp_avg_sq": torch.full_like(camera_param, 4.0),
    }
    opt.state[camera_param] = {
        "step": camera_state["step"].clone(),
        "exp_avg": camera_state["exp_avg"].clone(),
        "exp_avg_sq": camera_state["exp_avg_sq"].clone(),
    }

    before = pipeline.field_model.num_gaussians
    pipeline.field_model.clone_gaussians(torch.tensor([0], dtype=torch.long))
    rebuilt = pipeline._rebuild_optimizer_after_density(
        opt,
        cfg.train,
        survivor_sources=torch.arange(before, dtype=torch.long),
        appended_count=1,
    )

    for group in rebuilt.param_groups:
        for param in group["params"]:
            assert not rebuilt.state.get(param, {})
