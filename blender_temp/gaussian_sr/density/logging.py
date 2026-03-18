from collections.abc import Callable, Mapping
import json
from pathlib import Path
from typing import Any


DensityEventCallback = Callable[[dict[str, Any]], None]


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def append_density_event_jsonl(event: Mapping[str, Any], path: str | Path) -> None:
    jsonl_path = Path(path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_to_jsonable(dict(event)), sort_keys=True) + "\n")


def emit_density_event(
    event: Mapping[str, Any],
    jsonl_path: str | Path | None = None,
    callback: DensityEventCallback | None = None,
) -> None:
    payload = _to_jsonable(dict(event))
    if jsonl_path is not None:
        append_density_event_jsonl(payload, jsonl_path)
    if callback is not None:
        callback(payload)


__all__ = [
    "DensityEventCallback",
    "append_density_event_jsonl",
    "emit_density_event",
]
