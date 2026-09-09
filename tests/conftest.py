import cv2
import numpy as np
import pytest

# A fixed noise texture, applied to every frame. Without it, a perfectly
# symmetric synthetic scene (plain background + a circle + a rectangle)
# produces literally duplicate ORB descriptors at symmetric points, which
# breaks cross-check matching's tie-resolution even between genuinely
# identical frames. Real camera footage has sensor noise/texture that avoids
# this; this mirrors that instead of exercising a pathological corner case.
_NOISE = np.random.default_rng(0).integers(0, 20, size=(240, 320, 3), dtype=np.uint8)


def _make_frame(t: float, w: int = 320, h: int = 240, blur: bool = False) -> np.ndarray:
    img = np.full((h, w, 3), 30, dtype=np.uint8)
    img = cv2.add(img, _NOISE[:h, :w])
    cv2.rectangle(img, (10, 10), (w - 10, h - 10), (60, 60, 60), 2)
    x = int(20 + (w - 60) * t)
    cv2.circle(img, (x, h // 2), 20, (0, 200, 255), -1)
    if blur:
        img = cv2.GaussianBlur(img, (25, 25), 15)
    return img


@pytest.fixture
def synthetic_video(tmp_path):
    """A short synthetic clip: a shape sweeps left->right (distinct viewpoints),
    holds still (redundant frames), gets blurry for a stretch (rejected), then
    sweeps back. Returns (path, total_frame_count)."""
    path = str(tmp_path / "synthetic.mp4")
    fps = 30
    w, h = 320, 240

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))

    n_moving = 60
    n_still = 30
    n_blur = 10

    for i in range(n_moving):
        t = i / (n_moving - 1)
        writer.write(_make_frame(t, w, h))
    for _ in range(n_still):
        writer.write(_make_frame(1.0, w, h))
    for _ in range(n_blur):
        writer.write(_make_frame(1.0, w, h, blur=True))
    for i in range(n_moving):
        t = 1.0 - i / (n_moving - 1)
        writer.write(_make_frame(t, w, h))

    writer.release()

    total_frames = n_moving + n_still + n_blur + n_moving
    return path, total_frames
