from __future__ import annotations

from functools import lru_cache
from typing import Any


class VisionRuntimeUnavailable(RuntimeError):
    """Raised when the optional local computer-vision runtime is unavailable."""


@lru_cache(maxsize=1)
def require_numpy() -> Any:
    try:
        import numpy
    except (ImportError, OSError) as exc:
        raise VisionRuntimeUnavailable(
            "camera vision requires NumPy; install the optional local geometry runtime "
            "with `pip install -e '.[label]'`"
        ) from exc
    return numpy


@lru_cache(maxsize=1)
def require_opencv() -> Any:
    try:
        import cv2
    except (ImportError, OSError) as exc:
        raise VisionRuntimeUnavailable(
            "camera vision requires OpenCV; install the optional local geometry runtime "
            "with `pip install -e '.[label]'`"
        ) from exc
    return cv2


def require_vision_runtime() -> tuple[Any, Any]:
    return require_opencv(), require_numpy()
