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
    initial = torch.tensor([fx, fy, cx, cy], dtype=torch.float32)
    intr = LearnableSharedIntrinsics(initial=initial, learn_intrinsics=False)
    out = intr.get(scale_x=float(scale_x), scale_y=float(scale_y))

    expected = torch.tensor(
        [
            float(fx) * float(scale_x),
            float(fy) * float(scale_y),
            (float(cx) + 0.5) * float(scale_x) - 0.5,
            (float(cy) + 0.5) * float(scale_y) - 0.5,
        ],
        dtype=torch.float32,
    )
    assert_close(out, expected, atol=1.0e-6, rtol=1.0e-6)


@DEFAULT_SETTINGS
@given(num_views=st.integers(min_value=1, max_value=6))
def test_camera_bundle_keeps_first_view_identity_and_regularizer_nonnegative(num_views: int) -> None:
    init_shifts_px = torch.zeros(num_views, 2, dtype=torch.float32)
    bundle = LearnableCameraBundle(
        num_views=num_views,
        fx=1000.0,
        fy=1000.0,
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
    initial = torch.tensor([1000.0, 900.0, 320.0, 240.0], dtype=torch.float32)

    learnable = LearnableSharedIntrinsics(initial=initial, learn_intrinsics=True)
    frozen = LearnableSharedIntrinsics(initial=initial, learn_intrinsics=False)

    assert set(dict(learnable.named_parameters()).keys()) == {"log_fx", "log_fy", "cx", "cy"}
    assert set(dict(frozen.named_parameters()).keys()) == set()
    assert set(dict(frozen.named_buffers()).keys()) == {"log_fx", "log_fy", "cx", "cy"}
