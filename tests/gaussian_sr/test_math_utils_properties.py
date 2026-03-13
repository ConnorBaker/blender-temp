import torch
from hypothesis import given, strategies as st
from torch.testing import assert_close

from blender_temp.gaussian_sr.math_utils import default_intrinsics, normalize_quaternion, pose_vec_to_rt, so3_exp_map
from blender_temp.gaussian_sr.posefree_config import CameraInit

from .strategies import DEFAULT_SETTINGS, pose_vectors, quaternion_batches


@DEFAULT_SETTINGS
@given(q=quaternion_batches())
def test_normalize_quaternion_returns_unit_norm(q: torch.Tensor) -> None:
    qn = normalize_quaternion(q)
    norms = qn.norm(dim=-1)
    assert torch.isfinite(qn).all()
    assert_close(norms, torch.ones_like(norms), atol=1.0e-5, rtol=1.0e-5)


@DEFAULT_SETTINGS
@given(xi=pose_vectors())
def test_pose_vec_to_rt_preserves_translation_component(xi: torch.Tensor) -> None:
    r, t = pose_vec_to_rt(xi)
    assert r.shape == (xi.shape[0], 3, 3)
    assert t.shape == (xi.shape[0], 3)
    assert torch.isfinite(r).all()
    assert torch.isfinite(t).all()
    assert_close(t, xi[..., 3:], atol=1.0e-6, rtol=1.0e-6)


@DEFAULT_SETTINGS
@given(
    height=st.integers(min_value=1, max_value=128),
    width=st.integers(min_value=1, max_value=128),
    fov_degrees=st.floats(min_value=5.0, max_value=120.0, allow_nan=False, allow_infinity=False, width=32),
)
def test_default_intrinsics_center_matches_image_center(height: int, width: int, fov_degrees: float) -> None:
    init = CameraInit(default_fov_degrees=float(fov_degrees))
    intr = default_intrinsics(height, width, device=torch.device("cpu"), dtype=torch.float32, init=init)

    assert intr.shape == (4,)
    assert float(intr[0].item()) > 0.0
    assert float(intr[1].item()) > 0.0
    assert_close(intr[2], torch.tensor((width - 1.0) * 0.5, dtype=torch.float32), atol=1.0e-6, rtol=1.0e-6)
    assert_close(intr[3], torch.tensor((height - 1.0) * 0.5, dtype=torch.float32), atol=1.0e-6, rtol=1.0e-6)


def test_so3_exp_map_zero_returns_identity_with_finite_gradient() -> None:
    omega = torch.zeros(1, 3, dtype=torch.float32, requires_grad=True)
    r = so3_exp_map(omega)

    assert_close(r, torch.eye(3, dtype=torch.float32).unsqueeze(0), atol=1.0e-6, rtol=1.0e-6)

    loss = r.sum()
    loss.backward()

    assert omega.grad is not None
    assert torch.isfinite(omega.grad).all()


def test_so3_exp_map_compiles_fullgraph_without_graph_breaks() -> None:
    if not hasattr(torch, "compile"):
        return

    eager_input = torch.tensor([[0.0, 0.0, 0.0], [1.0e-4, -2.0e-4, 3.0e-4]], dtype=torch.float32)
    eager = so3_exp_map(eager_input)
    compiled = torch.compile(so3_exp_map, fullgraph=True, dynamic=False)
    actual = compiled(eager_input.clone())

    assert_close(actual, eager, atol=1.0e-6, rtol=1.0e-6)


@DEFAULT_SETTINGS
@given(xi=pose_vectors())
def test_so3_exp_map_small_inputs_have_finite_gradients(xi: torch.Tensor) -> None:
    omega = (xi[..., :3] * 1.0e-3).clone().detach().requires_grad_(True)
    r = so3_exp_map(omega)

    assert torch.isfinite(r).all()

    loss = r.sum()
    loss.backward()

    assert omega.grad is not None
    assert torch.isfinite(omega.grad).all()
