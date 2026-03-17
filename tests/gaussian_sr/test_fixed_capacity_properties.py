import pytest
import torch

from blender_temp.gaussian_sr.fixed_capacity import (
    append_rows_in_place,
    available_capacity,
    compact_rows_in_place,
    resolve_capacity,
)


def test_resolve_capacity_respects_initial_and_configured_floor() -> None:
    assert resolve_capacity(128, 0) == 128
    assert resolve_capacity(128, 64) == 128
    assert resolve_capacity(128, 256) == 256


def test_available_capacity_clamps_at_zero() -> None:
    assert available_capacity(16, 4) == 12
    assert available_capacity(16, 16) == 0
    assert available_capacity(16, 20) == 0


def test_compact_rows_in_place_moves_survivors_to_front() -> None:
    rows = {
        "value": torch.tensor([[1.0], [2.0], [3.0], [4.0], [0.0], [0.0]], dtype=torch.float32),
        "index": torch.tensor([10, 11, 12, 13, -1, -1], dtype=torch.long),
    }

    kept = compact_rows_in_place(
        rows,
        active_count=4,
        keep_mask=torch.tensor([True, False, True, False]),
    )

    assert kept == 2
    assert torch.equal(rows["value"][:2, 0], torch.tensor([1.0, 3.0]))
    assert torch.equal(rows["index"][:2], torch.tensor([10, 12]))


def test_append_rows_in_place_appends_until_capacity_then_reports_drops() -> None:
    rows = {
        "value": torch.zeros((4, 2), dtype=torch.float32),
        "index": torch.full((4,), -1, dtype=torch.long),
    }
    rows["value"][:2] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    rows["index"][:2] = torch.tensor([5, 6])

    result = append_rows_in_place(
        rows,
        active_count=2,
        new_rows={
            "value": torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], dtype=torch.float32),
            "index": torch.tensor([7, 8, 9], dtype=torch.long),
        },
    )

    assert result.appended == 2
    assert result.dropped == 1
    assert result.new_active_count == 4
    assert torch.equal(rows["value"], torch.tensor([[1.0, 2.0], [3.0, 4.0], [7.0, 8.0], [9.0, 10.0]]))
    assert torch.equal(rows["index"], torch.tensor([5, 6, 7, 8]))


def test_append_rows_in_place_can_abort_on_overflow() -> None:
    rows = {
        "value": torch.zeros((2, 1), dtype=torch.float32),
    }

    with pytest.raises(RuntimeError):
        append_rows_in_place(
            rows,
            active_count=2,
            new_rows={"value": torch.ones((1, 1), dtype=torch.float32)},
            overflow_policy="abort",
        )
