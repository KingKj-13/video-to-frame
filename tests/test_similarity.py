import cv2
import numpy as np
import pytest

from video_to_frame.analysis.similarity import dhash, hamming_distance, ssim_precomputed, ssim_score


def test_identical_frames_are_maximally_similar():
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(img, (50, 50), 30, 255, -1)

    h1, h2 = dhash(img), dhash(img.copy())
    assert hamming_distance(h1, h2) == 0
    assert ssim_score(img, img) > 0.99


def test_different_frames_are_less_similar_than_identical():
    img_a = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(img_a, (20, 20), 15, 255, -1)

    img_b = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(img_b, (60, 60), (95, 95), 255, -1)

    h1, h2 = dhash(img_a), dhash(img_b)
    assert hamming_distance(h1, h2) > 0
    assert ssim_score(img_a, img_b) < ssim_score(img_a, img_a)


def test_ssim_matches_skimage_reference_implementation():
    """Phase 2.5 replaced skimage's structural_similarity with a faster
    cv2.boxFilter reimplementation of the same algorithm (same default
    K1/K2, 7x7 uniform window, unbiased covariance). This locks in that the
    two stay numerically equivalent -- skimage's own logic changing, or a
    future edit to ssim_precomputed, would show up here as a real deviation,
    not just a passed/failed threshold check."""
    skimage = pytest.importorskip("skimage.metrics")
    rng = np.random.default_rng(0)

    for trial in range(10):
        a = rng.integers(0, 255, (128, 128), dtype=np.uint8)
        if trial % 3 == 0:
            b = a.copy()
        elif trial % 3 == 1:
            b = cv2.GaussianBlur(a, (5, 5), 2)
        else:
            b = rng.integers(0, 255, (128, 128), dtype=np.uint8)

        reference = skimage.structural_similarity(a, b)
        actual = ssim_precomputed(a, b)
        assert actual == pytest.approx(reference, abs=1e-9)
