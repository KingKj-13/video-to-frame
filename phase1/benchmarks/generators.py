"""Synthetic generators for benchmark categories with no real footage
available yet (static object orbit, indoor scene, difficult lighting).

Drone and handheld already have real footage (see phase1/benchmarks/drone/
and phase1/benchmarks/handheld/, populated from the videos supplied earlier
in this project) -- these generators exist only to give the remaining
categories *something* concrete to test against until real footage is
supplied. They are deliberately built from the same primitives
(tests/synthetic.py's landmark-canvas idea) that already proved necessary
for ORB to find real keypoints, rather than plain noise.
"""

from __future__ import annotations

import os

import cv2
import numpy as np


def _landmark_canvas(width: int, height: int, seed: int, background: int = 40,
                      density: int = 28, tile: bool = False) -> np.ndarray:
    canvas = np.full((height, width, 3), background, dtype=np.uint8)
    canvas = cv2.add(canvas, np.random.default_rng(seed).integers(0, 25, size=canvas.shape, dtype=np.uint8))
    shape_rng = np.random.default_rng(seed + 1)

    if tile:
        # Repeating structure (bricks/tiles) -- stresses "large static
        # background must not cause false redundancy" (Phase 1 section 8).
        tile_w, tile_h = 60, 40
        for ty in range(0, height, tile_h):
            for tx in range(0, width, tile_w):
                shade = 70 if ((tx // tile_w) + (ty // tile_h)) % 2 == 0 else 55
                cv2.rectangle(canvas, (tx + 2, ty + 2), (tx + tile_w - 2, ty + tile_h - 2), (shade, shade, shade), 1)

    for cx in range(20, width - 20, density):
        cy = int(shape_rng.integers(30, height - 30))
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


def generate_static_object_orbit(path: str, n_frames: int = 90, w: int = 320, h: int = 240, fps: int = 30) -> None:
    """A camera orbiting a static, richly-textured object: a wide landmark
    canvas panned across cyclically (wrapping at the edges, simulating a
    360-degree orbit) -- tests viewpoint change, feature overlap, coverage."""
    canvas_w = w * 3
    canvas = _landmark_canvas(canvas_w, h, seed=1)
    wrapped = np.hstack([canvas, canvas[:, :w]])  # seam-free wraparound

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for i in range(n_frames):
        offset = int((i / n_frames) * canvas_w)
        writer.write(wrapped[:, offset:offset + w].copy())
    writer.release()


def generate_indoor_scene(path: str, n_frames: int = 90, w: int = 320, h: int = 240, fps: int = 30) -> None:
    """A pan across a repeating-structure "room" (tiled background) with a
    handful of distinct landmarks -- tests that large static/repetitive
    backgrounds don't cause genuinely different viewpoints to be
    misclassified as redundant (Phase 1 section 8)."""
    canvas_w = w * 4
    canvas = _landmark_canvas(canvas_w, h, seed=2, density=45, tile=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    max_offset = canvas_w - w
    for i in range(n_frames):
        offset = int((i / (n_frames - 1)) * max_offset)
        writer.write(canvas[:, offset:offset + w].copy())
    writer.release()


def generate_difficult_lighting(path: str, n_frames: int = 90, w: int = 320, h: int = 240, fps: int = 30) -> None:
    """The same panning scene as static_object_orbit, but with brightness
    modulated (slow sinusoidal drift plus a couple of sudden exposure jumps)
    -- tests similarity/quality-scoring robustness under lighting change."""
    canvas_w = w * 3
    canvas = _landmark_canvas(canvas_w, h, seed=3)
    max_offset = canvas_w - w

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for i in range(n_frames):
        offset = int((i / (n_frames - 1)) * max_offset)
        frame = canvas[:, offset:offset + w].copy().astype(np.float32)

        drift = 1.0 + 0.35 * np.sin(2 * np.pi * i / n_frames)
        jump = 0.5 if (n_frames // 3 <= i < n_frames // 3 + 8) else (1.6 if (2 * n_frames // 3 <= i < 2 * n_frames // 3 + 8) else 1.0)
        frame = np.clip(frame * drift * jump, 0, 255).astype(np.uint8)
        writer.write(frame)
    writer.release()


def generate_all(benchmarks_root: str) -> None:
    generate_static_object_orbit(os.path.join(benchmarks_root, "static_object", "static_object_orbit.mp4"))
    generate_indoor_scene(os.path.join(benchmarks_root, "indoor", "indoor_scene.mp4"))
    generate_difficult_lighting(os.path.join(benchmarks_root, "difficult_lighting", "difficult_lighting.mp4"))


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(__file__)
    generate_all(root)
    print(f"Generated synthetic benchmark videos under {root}")
