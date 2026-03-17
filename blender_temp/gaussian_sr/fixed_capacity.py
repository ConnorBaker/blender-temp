from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class CapacityAppendResult:
    appended: int
    dropped: int
    new_active_count: int
    capacity: int


def resolve_capacity(initial_count: int, configured_capacity: int) -> int:
    initial = max(int(initial_count), 0)
    configured = max(int(configured_capacity), 0)
    if configured <= 0:
        return initial
    return max(initial, configured)


def available_capacity(capacity: int, active_count: int) -> int:
    return max(int(capacity) - int(active_count), 0)


def compact_rows_in_place(
    rows: Mapping[str, Tensor],
    *,
    active_count: int,
    keep_mask: Tensor,
) -> int:
    current = int(active_count)
    if keep_mask.ndim != 1 or int(keep_mask.shape[0]) != current:
        raise ValueError("keep_mask must be 1D and match active_count")
    keep = keep_mask.to(dtype=torch.bool, device=keep_mask.device)
    keep_indices = keep.nonzero(as_tuple=False).squeeze(-1)
    kept = int(keep_indices.numel())
    for tensor in rows.values():
        if tensor.ndim == 0:
            raise ValueError("capacity rows must be at least 1D")
        if int(tensor.shape[0]) < current:
            raise ValueError("tensor capacity is smaller than active_count")
        if kept > 0:
            source = tensor[:current].index_select(0, keep_indices.to(device=tensor.device))
            tensor[:kept].copy_(source)
    return kept


def append_rows_in_place(
    rows: Mapping[str, Tensor],
    *,
    active_count: int,
    new_rows: Mapping[str, Tensor],
    overflow_policy: str = "freeze_density_and_warn",
) -> CapacityAppendResult:
    if overflow_policy not in {"freeze_density_and_warn", "abort"}:
        raise ValueError(f"unsupported overflow_policy: {overflow_policy!r}")
    if not rows:
        raise ValueError("rows must not be empty")
    if set(rows.keys()) != set(new_rows.keys()):
        raise ValueError("rows and new_rows must have identical keys")

    current = int(active_count)
    capacities = {name: int(tensor.shape[0]) for name, tensor in rows.items()}
    capacity = min(capacities.values())
    if capacity < current:
        raise ValueError("active_count exceeds storage capacity")

    requested = None
    for name, tensor in new_rows.items():
        if tensor.ndim == 0:
            raise ValueError(f"new_rows[{name!r}] must be at least 1D")
        if requested is None:
            requested = int(tensor.shape[0])
        elif int(tensor.shape[0]) != requested:
            raise ValueError("all new_rows tensors must share the same leading dimension")
        base_shape = rows[name].shape[1:]
        if tensor.shape[1:] != base_shape:
            raise ValueError(f"shape mismatch for {name!r}: expected (*, {base_shape}), got {tensor.shape}")
    request_count = int(requested or 0)
    free = available_capacity(capacity, current)
    if request_count > free and overflow_policy == "abort":
        raise RuntimeError("fixed-capacity storage overflow")

    write_count = min(request_count, free)
    start = current
    end = current + write_count
    if write_count > 0:
        for name, tensor in rows.items():
            tensor[start:end].copy_(new_rows[name][:write_count].to(device=tensor.device, dtype=tensor.dtype))
    return CapacityAppendResult(
        appended=write_count,
        dropped=max(request_count - write_count, 0),
        new_active_count=end,
        capacity=capacity,
    )


__all__ = [
    "CapacityAppendResult",
    "append_rows_in_place",
    "available_capacity",
    "compact_rows_in_place",
    "resolve_capacity",
]
