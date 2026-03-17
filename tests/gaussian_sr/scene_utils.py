"""Shared test utilities for generating visible Gaussian scenes."""

import torch
from hypothesis import strategies as st


def make_visible_scene(n: int, width: int, height: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Create a minimal scene of *n* Gaussians guaranteed to be visible.

    Returns a dict with keys ``means``, ``quat``, ``scale``, ``viewmat``, and ``K``,
    all contiguous ``float32`` tensors on *device*.
    """
    dtype = torch.float32
    means = torch.stack(
        (
            torch.linspace(-0.2, 0.2, n, device=device, dtype=dtype),
            torch.linspace(0.15, -0.15, n, device=device, dtype=dtype),
            torch.full((n,), 2.0, device=device, dtype=dtype),
        ),
        dim=-1,
    )
    quat = torch.stack(
        (
            torch.ones(n, device=device, dtype=dtype),
            torch.zeros(n, device=device, dtype=dtype),
            torch.zeros(n, device=device, dtype=dtype),
            torch.zeros(n, device=device, dtype=dtype),
        ),
        dim=-1,
    )
    scale = torch.full((n, 3), 0.05, device=device, dtype=dtype)
    viewmat = torch.eye(4, device=device, dtype=dtype)
    k = torch.tensor(
        [
            [float(width), 0.0, (width - 1.0) * 0.5],
            [0.0, float(height), (height - 1.0) * 0.5],
            [0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    )
    return {
        "means": means.contiguous(),
        "quat": quat.contiguous(),
        "scale": scale.contiguous(),
        "viewmat": viewmat.contiguous(),
        "K": k.contiguous(),
    }


@st.composite
def visible_scene_strategy(
    draw,
    min_n: int = 1,
    max_n: int = 4,
    min_side: int = 4,
    max_side: int = 8,
) -> dict[str, torch.Tensor]:
    """Hypothesis composite strategy that draws a visible scene dict.

    The scene is created on CPU; move tensors to the desired device in tests.
    """
    n = draw(st.integers(min_value=min_n, max_value=max_n))
    width = draw(st.integers(min_value=min_side, max_value=max_side))
    height = draw(st.integers(min_value=min_side, max_value=max_side))
    return make_visible_scene(n, width, height, device=torch.device("cpu"))


__all__ = [
    "make_visible_scene",
    "visible_scene_strategy",
]
