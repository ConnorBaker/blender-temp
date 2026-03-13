import json
from pathlib import Path

from blender_temp.gaussian_sr.density_logging import append_density_event_jsonl, emit_density_event
from blender_temp.gaussian_sr.progress_logging import append_progress_event_jsonl, emit_progress_event


def test_append_density_event_jsonl_serializes_nested_paths(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "density.jsonl"
    event = {
        "step": 3,
        "path": tmp_path / "artifact.bin",
        "items": (1, 2, {"inner": tmp_path / "nested.txt"}),
    }

    append_density_event_jsonl(event, path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["step"] == 3
    assert payload["path"] == str(tmp_path / "artifact.bin")
    assert payload["items"][2]["inner"] == str(tmp_path / "nested.txt")


def test_emit_density_event_writes_and_invokes_callback(tmp_path: Path) -> None:
    seen: list[dict] = []
    path = tmp_path / "density.jsonl"
    event = {"kind": "density", "values": [1, 2, 3]}

    emit_density_event(event, jsonl_path=path, callback=seen.append)

    assert len(seen) == 1
    assert seen[0] == event
    lines = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0]) == event


def test_append_progress_event_jsonl_serializes_nested_paths(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "progress.jsonl"
    event = {
        "step": 7,
        "path": tmp_path / "trace.bin",
        "items": [tmp_path / "a.txt", {"inner": tmp_path / "b.txt"}],
    }

    append_progress_event_jsonl(event, path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["step"] == 7
    assert payload["path"] == str(tmp_path / "trace.bin")
    assert payload["items"][0] == str(tmp_path / "a.txt")
    assert payload["items"][1]["inner"] == str(tmp_path / "b.txt")


def test_emit_progress_event_writes_and_invokes_callback(tmp_path: Path) -> None:
    seen: list[dict] = []
    path = tmp_path / "progress.jsonl"
    event = {"event": "step_end", "loss": 0.25}

    emit_progress_event(event, jsonl_path=path, callback=seen.append)

    assert len(seen) == 1
    assert seen[0] == event
    lines = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0]) == event
