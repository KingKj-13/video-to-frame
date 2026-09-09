from __future__ import annotations

import cv2
import numpy as np


def dhash(gray: np.ndarray, hash_size: int = 8) -> int:
    """Perceptual difference-hash: robust to minor compression/exposure noise,
    cheap to compare via Hamming distance."""
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    h = 0
    for bit in diff.flatten():
        h = (h << 1) | int(bit)
    return h


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def ssim_precomputed(gray_a: np.ndarray, gray_b: np.ndarray, win_size: int = 7) -> float:
    """SSIM for two already same-sized grayscale images.

    Phase 2.5: this used to call skimage.metrics.structural_similarity
    directly. Profiling found it was one of the largest single costs in the
    pipeline (~32% of total time on one benchmark) despite existing only as
    a cheap-looking fallback signal -- skimage's generic scipy.ndimage
    uniform_filter backend carries real per-call overhead. This reimplements
    the exact same algorithm skimage uses by default (K1=0.01, K2=0.03,
    7x7 uniform/box window, unbiased covariance, border-cropped mean) with
    cv2.boxFilter instead, validated to match skimage's output to floating-
    point exactness (0.0 max deviation across 30 varied test pairs -- see
    tests/test_similarity.py) while running ~2x faster. Same numbers, same
    thresholds, no behavior change -- only a faster backend.
    """
    img_a = gray_a.astype(np.float64)
    img_b = gray_b.astype(np.float64)

    data_range = 255.0
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    ksize = (win_size, win_size)
    num_pixels = win_size * win_size
    cov_norm = num_pixels / (num_pixels - 1)  # skimage's unbiased covariance correction

    def box(arr: np.ndarray) -> np.ndarray:
        return cv2.boxFilter(arr, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)

    ux, uy = box(img_a), box(img_b)
    uxx, uyy, uxy = box(img_a * img_a), box(img_b * img_b), box(img_a * img_b)

    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)

    numerator = (2 * ux * uy + c1) * (2 * vxy + c2)
    denominator = (ux * ux + uy * uy + c1) * (vx + vy + c2)
    ssim_map = numerator / denominator

    pad = win_size // 2
    cropped = ssim_map[pad:-pad, pad:-pad] if pad else ssim_map
    return float(cropped.mean())


def ssim_score(gray_a: np.ndarray, gray_b: np.ndarray, size: tuple[int, int] = (128, 128)) -> float:
    ra = cv2.resize(gray_a, size, interpolation=cv2.INTER_AREA)
    rb = cv2.resize(gray_b, size, interpolation=cv2.INTER_AREA)
    return ssim_precomputed(ra, rb)
