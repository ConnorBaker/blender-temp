import warp as wp  # type: ignore

wp.init()
_WARP_AVAILABLE = True
_WARP_IMPORT_ERROR = None


def require_warp() -> None:
    return None


__all__ = [
    "wp",
    "_WARP_AVAILABLE",
    "_WARP_IMPORT_ERROR",
    "require_warp",
]
