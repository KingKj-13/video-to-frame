"""Shared synthetic-scene helpers for selector-level tests.

A fixed, wide "scene" canvas that a narrower frame pans across, modeling a
real capture (camera moving around a static scene): nearby pan offsets share
most of their absolute scene content (redundant/overlapping views), while
far-apart offsets share little or none of it (a genuinely new viewpoint).

Unlike pure per-pixel random noise, this canvas is built from actual
geometric shapes (rectangles/circles/triangles) so ORB's corner detector
finds real, stable keypoints -- pure noise has no spatially-coherent gradient
structure for FAST/ORB to key on and yields zero keypoints, which silently
made earlier redundancy comparisons fall back to whole-frame hash/SSIM
instead of exercising real feature matching.
"""

import cv2
import numpy as np

CANVAS_W, CANVAS_H = 1200, 200
FRAME_W, FRAME_H = 200, 200


def _build_canvas() -> np.ndarray:
    canvas = np.full((CANVAS_H, CANVAS_W, 3), 40, dtype=np.uint8)
    canvas = cv2.add(canvas, np.random.default_rng(0).integers(0, 25, size=canvas.shape, dtype=np.uint8))

    shape_rng = np.random.default_rng(42)
    for cx in range(20, CANVAS_W - 20, 28):
        cy = int(shape_rng.integers(30, CANVAS_H - 30))
        size = int(shape_rng.integers(5, 16))
        color = tuple(int(c) for c in shape_rng.integers(50, 230, size=3))
        kind = shape_rng.integers(0, 3)
        if kind == 0:
            cv2.rectangle(canvas, (cx - size, cy - size), (cx + size, cy + size), color, -1)
        elif kind == 1:
            cv2.circle(canvas, (cx, cy), size, color, -1)
        else:
            pts = np.array([[cx, cy - size], [cx - size, cy + size], [cx + size, cy + size]], dtype=np.int32)
            cv2.fillPoly(canvas, [pts], color)
    return canvas


_CANVAS = _build_canvas()


def tile(offset: int) -> np.ndarray:
    return _CANVAS[:, offset:offset + FRAME_W].copy()
