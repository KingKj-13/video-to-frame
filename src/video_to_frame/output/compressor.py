from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np


def resize_frame(frame_bgr: np.ndarray, target_long_edge: Optional[int] = None,
                  target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    if target_size and target_size[0] and target_size[1]:
        return cv2.resize(frame_bgr, tuple(target_size), interpolation=cv2.INTER_AREA)
    if target_long_edge and max(h, w) > target_long_edge:
        scale = target_long_edge / max(h, w)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return frame_bgr


def output_extension(fmt: str) -> str:
    fmt = fmt.lower()
    return "jpg" if fmt == "jpeg" else fmt


def encode_and_save(frame_bgr: np.ndarray, path: str, fmt: str = "webp", quality: int = 90,
                     png_compression_level: int = 1) -> int:
    """`quality` (1-100) only applies to webp/jpg -- PNG is always lossless
    regardless of compression level, so `quality` never touches pixel data
    for PNG. `png_compression_level` (0-9, default 1) controls PNG's
    zlib-effort/file-size tradeoff only.

    Phase 2.5 profiling found PNG level 9 (the old default, derived from
    `quality // 10`) costs ~820ms/frame on a real 640x360 photo vs. ~27ms at
    level 1 on the same frame -- a ~30x difference in pure DEFLATE effort for
    a ~25% file-size difference, with the decoded pixels identical either
    way. There is no quality trade-off here, only a speed/size one, so this
    defaults to fast.
    """
    fmt = fmt.lower()
    if fmt == "webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, quality]
    elif fmt in ("jpg", "jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif fmt == "png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, min(9, max(0, png_compression_level))]
    else:
        raise ValueError(f"Unsupported output format: {fmt}")

    ok = cv2.imwrite(path, frame_bgr, params)
    if not ok:
        raise IOError(f"Failed to write frame to {path}")
    return os.path.getsize(path)
