import cv2
import numpy as np

from video_to_frame.analysis.quality import compute_quality_metrics


def test_sharp_frame_scores_higher_than_blurry_frame():
    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
    blurry = cv2.GaussianBlur(sharp, (21, 21), 10)

    sharp_metrics = compute_quality_metrics(sharp)
    blurry_metrics = compute_quality_metrics(blurry)

    assert sharp_metrics.sharpness > blurry_metrics.sharpness


def test_flat_frame_has_near_zero_sharpness():
    flat = np.full((100, 100, 3), 128, dtype=np.uint8)
    metrics = compute_quality_metrics(flat)
    assert metrics.sharpness < 1.0
