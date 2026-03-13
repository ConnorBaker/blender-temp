import numpy as np
import torch
from hypothesis import HealthCheck, settings, strategies as st
from hypothesis.extra import numpy as hnp

DEFAULT_SETTINGS = settings(
    deadline=None,
    max_examples=40,
    suppress_health_check=[HealthCheck.too_slow],
)

finite_float32 = st.floats(
    min_value=-8.0,
    max_value=8.0,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)


def _f32(value: float) -> float:
    return float(np.float32(value))


positive_float32 = st.floats(
    min_value=_f32(1.0e-3),
    max_value=_f32(8.0),
    allow_nan=False,
    allow_infinity=False,
    width=32,
)
unit_float32 = st.floats(
    min_value=_f32(0.0),
    max_value=_f32(1.0),
    allow_nan=False,
    allow_infinity=False,
    width=32,
)


@st.composite
def finite_vectors(
    draw,
    min_len: int = 1,
    max_len: int = 64,
    min_value: float = -8.0,
    max_value: float = 8.0,
) -> torch.Tensor:
    n = draw(st.integers(min_value=min_len, max_value=max_len))
    arr = draw(
        hnp.arrays(
            dtype=np.float32,
            shape=(n,),
            elements=st.floats(
                min_value=_f32(min_value),
                max_value=_f32(max_value),
                allow_nan=False,
                allow_infinity=False,
                width=32,
            ),
        )
    )
    return torch.from_numpy(arr)


@st.composite
def chw_images(
    draw,
    channels: int = 3,
    min_side: int = 1,
    max_side: int = 8,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> torch.Tensor:
    h = draw(st.integers(min_value=min_side, max_value=max_side))
    w = draw(st.integers(min_value=min_side, max_value=max_side))
    arr = draw(
        hnp.arrays(
            dtype=np.float32,
            shape=(channels, h, w),
            elements=st.floats(
                min_value=_f32(min_value),
                max_value=_f32(max_value),
                allow_nan=False,
                allow_infinity=False,
                width=32,
            ),
        )
    )
    return torch.from_numpy(arr)


@st.composite
def same_shape_chw_image_pairs(
    draw,
    channels: int = 3,
    min_side: int = 2,
    max_side: int = 8,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    h = draw(st.integers(min_value=min_side, max_value=max_side))
    w = draw(st.integers(min_value=min_side, max_value=max_side))
    elems = st.floats(
        min_value=_f32(min_value),
        max_value=_f32(max_value),
        allow_nan=False,
        allow_infinity=False,
        width=32,
    )
    arr_a = draw(hnp.arrays(dtype=np.float32, shape=(channels, h, w), elements=elems))
    arr_b = draw(hnp.arrays(dtype=np.float32, shape=(channels, h, w), elements=elems))
    return torch.from_numpy(arr_a), torch.from_numpy(arr_b)


@st.composite
def quaternion_batches(draw, min_batch: int = 1, max_batch: int = 8) -> torch.Tensor:
    n = draw(st.integers(min_value=min_batch, max_value=max_batch))
    arr = draw(
        hnp.arrays(
            dtype=np.float32,
            shape=(n, 4),
            elements=st.floats(
                min_value=-4.0,
                max_value=4.0,
                allow_nan=False,
                allow_infinity=False,
                width=32,
            ),
        )
    )
    q = torch.from_numpy(arr)
    tiny = q.norm(dim=-1, keepdim=True) < 1.0e-3
    q = q + tiny.to(q.dtype) * torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=q.dtype)
    return q


@st.composite
def pose_vectors(draw, min_batch: int = 1, max_batch: int = 8) -> torch.Tensor:
    n = draw(st.integers(min_value=min_batch, max_value=max_batch))
    arr = draw(
        hnp.arrays(
            dtype=np.float32,
            shape=(n, 6),
            elements=st.floats(
                min_value=-1.5,
                max_value=1.5,
                allow_nan=False,
                allow_infinity=False,
                width=32,
            ),
        )
    )
    return torch.from_numpy(arr)


@st.composite
def appearance_inputs(draw, min_batch: int = 1, max_batch: int = 8, sh_degree: int = 2):
    from blender_temp.gaussian_sr.appearance import num_sh_bases

    n = draw(st.integers(min_value=min_batch, max_value=max_batch))
    base_logits = torch.from_numpy(
        draw(
            hnp.arrays(
                dtype=np.float32,
                shape=(n, 3),
                elements=st.floats(
                    min_value=_f32(-6.0),
                    max_value=_f32(6.0),
                    allow_nan=False,
                    allow_infinity=False,
                    width=32,
                ),
            )
        )
    )
    means = torch.from_numpy(
        draw(
            hnp.arrays(
                dtype=np.float32,
                shape=(n, 3),
                elements=st.floats(
                    min_value=_f32(-2.0),
                    max_value=_f32(2.0),
                    allow_nan=False,
                    allow_infinity=False,
                    width=32,
                ),
            )
        )
    )
    means[:, 2] = means[:, 2].abs() + 0.5
    coeff_count = max(num_sh_bases(sh_degree) - 1, 0)
    sh_coeffs = torch.from_numpy(
        draw(
            hnp.arrays(
                dtype=np.float32,
                shape=(n, 3, coeff_count),
                elements=st.floats(
                    min_value=_f32(-2.0),
                    max_value=_f32(2.0),
                    allow_nan=False,
                    allow_infinity=False,
                    width=32,
                ),
            )
        )
    )
    return base_logits, sh_coeffs, means


@st.composite
def tiny_image_batches(
    draw,
    min_views: int = 1,
    max_views: int = 2,
    min_side: int = 4,
    max_side: int = 8,
    min_value: float = 0.05,
    max_value: float = 0.95,
) -> torch.Tensor:
    v = draw(st.integers(min_value=min_views, max_value=max_views))
    h = draw(st.integers(min_value=min_side, max_value=max_side))
    w = draw(st.integers(min_value=min_side, max_value=max_side))
    arr = draw(
        hnp.arrays(
            dtype=np.float32,
            shape=(v, 3, h, w),
            elements=st.floats(
                min_value=_f32(min_value),
                max_value=_f32(max_value),
                allow_nan=False,
                allow_infinity=False,
                width=32,
            ),
        )
    )
    return torch.from_numpy(arr)


__all__ = [
    "DEFAULT_SETTINGS",
    "finite_float32",
    "positive_float32",
    "unit_float32",
    "finite_vectors",
    "chw_images",
    "same_shape_chw_image_pairs",
    "quaternion_batches",
    "pose_vectors",
    "appearance_inputs",
    "tiny_image_batches",
]
