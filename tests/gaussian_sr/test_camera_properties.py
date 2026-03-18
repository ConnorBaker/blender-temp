import torch
from hypothesis import given, strategies as st
from torch.testing import assert_close

from blender_temp.gaussian_sr.camera import LearnableCameraBundle, LearnableSharedIntrinsics

from .strategies import DEFAULT_SETTINGS, finite_float32, positive_float32


@DEFAULT_SETTINGS
@given(
    fx=positive_float32,
    fy=positive_float32,
    cx=finite_float32,
    cy=finite_float32,
    scale_x=positive_float32,
    scale_y=positive_float32,
)
def test_shared_intrinsics_scaling_matches_pixel_center_formula(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    scale_x: float,
    scale_y: float,
) -> None:
    focal = torch.tensor([fx, fy], dtype=torch.float32)
    principal = torch.tensor([cx, cy], dtype=torch.float32)
    scale = torch.tensor([scale_x, scale_y], dtype=torch.float32)
    intr = LearnableSharedIntrinsics(focal, principal, learn=False, device=torch.device("cpu"), dtype=torch.float32)
    out_focal, out_pp = intr.get(scale)

    expected_focal = torch.tensor([fx * scale_x, fy * scale_y], dtype=torch.float32)
    expected_pp = torch.tensor(
        [(cx + 0.5) * scale_x - 0.5, (cy + 0.5) * scale_y - 0.5],
        dtype=torch.float32,
    )
    assert_close(out_focal, expected_focal, atol=1.0e-6, rtol=1.0e-6)
    assert_close(out_pp, expected_pp, atol=1.0e-6, rtol=1.0e-6)


@DEFAULT_SETTINGS
@given(num_views=st.integers(min_value=1, max_value=6))
def test_camera_bundle_keeps_first_view_identity_and_regularizer_nonnegative(num_views: int) -> None:
    init_shifts_px = torch.zeros(num_views, 2, dtype=torch.float32)
    bundle = LearnableCameraBundle(
        num_views=num_views,
        focal=torch.tensor([1000.0, 1000.0]),
        init_shifts_px=init_shifts_px,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    r, t = bundle.world_to_camera()

    assert r.shape == (num_views, 3, 3)
    assert t.shape == (num_views, 3)
    assert_close(r[0], torch.eye(3, dtype=torch.float32), atol=1.0e-6, rtol=1.0e-6)
    assert_close(t[0], torch.zeros(3, dtype=torch.float32), atol=1.0e-6, rtol=1.0e-6)
    assert float(bundle.pose_regularizer().item()) >= 0.0


def test_shared_intrinsics_registers_parameters_only_when_learnable() -> None:
    focal = torch.tensor([1000.0, 900.0], dtype=torch.float32)
    principal = torch.tensor([320.0, 240.0], dtype=torch.float32)

    learnable = LearnableSharedIntrinsics(focal, principal, learn=True, device=torch.device("cpu"), dtype=torch.float32)
    frozen = LearnableSharedIntrinsics(focal, principal, learn=False, device=torch.device("cpu"), dtype=torch.float32)

    assert set(dict(learnable.named_parameters()).keys()) == {"log_focal", "principal"}
    # frozen: both are nn.Parameter with requires_grad=False, so they
    # appear in parameters() but with grad disabled.
    assert all(not p.requires_grad for p in frozen.parameters())
