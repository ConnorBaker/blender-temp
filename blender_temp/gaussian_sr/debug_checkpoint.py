from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor


def load_debug_checkpoint(path: str | Path) -> dict[str, object]:
    return torch.load(Path(path), map_location="cpu")


def restore_module_state_dict(
    module: nn.Module,
    state_dict: Mapping[str, Tensor],
) -> list[str]:
    modules = dict(module.named_modules())
    try:
        fallback_device = next(module.parameters()).device
    except StopIteration:
        fallback_device = torch.device("cpu")
    skipped: list[str] = []
    with torch.no_grad():
        for name, tensor in state_dict.items():
            module_name, _, leaf = name.rpartition(".")
            if module_name and module_name not in modules:
                skipped.append(name)
                continue
            owner = module if not module_name else modules[module_name]
            existing_param = owner._parameters.get(leaf)
            if existing_param is not None:
                restored = tensor.to(device=existing_param.device, dtype=existing_param.dtype).clone()
                owner._parameters[leaf] = nn.Parameter(
                    restored,
                    requires_grad=existing_param.requires_grad,
                )
                continue
            if leaf in owner._buffers:
                existing_buffer = owner._buffers[leaf]
                target_device = fallback_device if existing_buffer is None else existing_buffer.device
                target_dtype = tensor.dtype if existing_buffer is None else existing_buffer.dtype
                owner._buffers[leaf] = tensor.to(device=target_device, dtype=target_dtype).clone()
                continue
            skipped.append(name)
    return skipped


def restore_module_from_debug_checkpoint(
    module: nn.Module,
    path: str | Path,
) -> tuple[dict[str, object], list[str]]:
    payload = load_debug_checkpoint(path)
    skipped = restore_module_state_dict(module, payload["pipeline_state_dict"])
    return payload, skipped


__all__ = [
    "load_debug_checkpoint",
    "restore_module_from_debug_checkpoint",
    "restore_module_state_dict",
]
