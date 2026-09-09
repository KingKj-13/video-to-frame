from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


class FeatureExtractor:
    """Wraps ORB (default) or SIFT keypoint/descriptor extraction.

    ORB is the default: free of patent concerns, fast, and sufficient for the
    overlap/coverage heuristics used here. SIFT is available as a swap-in via
    config for users who want denser/more distinctive features at extra cost.
    """

    def __init__(self, max_features: int = 500, use_sift: bool = False):
        self.use_sift = use_sift
        if use_sift:
            if not hasattr(cv2, "SIFT_create"):
                raise RuntimeError(
                    "cv2.SIFT_create is unavailable in this OpenCV build; "
                    "install opencv-contrib-python or set use_sift: false."
                )
            self._detector = cv2.SIFT_create(nfeatures=max_features)
        else:
            self._detector = cv2.ORB_create(nfeatures=max_features)

    def extract(self, gray: np.ndarray) -> Tuple[list, Optional[np.ndarray]]:
        keypoints, descriptors = self._detector.detectAndCompute(gray, None)
        return keypoints, descriptors


def match_descriptors(desc_a: Optional[np.ndarray], desc_b: Optional[np.ndarray],
                       use_sift: bool = False, ratio: float = 0.75,
                       max_hamming_distance: int = 64) -> int:
    """Count of desc_a's descriptors that have a good match in desc_b.

    SIFT (float, L2) uses Lowe's ratio test, which relies on a well-separated
    best/second-best distance distribution that continuous descriptors give you.
    ORB (binary, Hamming) does not have that property -- with a coarse, often
    repetitive 256-bit space, the ratio test rejects many true positives (e.g.
    on a genuinely identical image, since near-duplicate local patterns give a
    close second-best distance too). Cross-check (mutual nearest neighbor) plus
    an absolute Hamming-distance cutoff is the standard, more reliable approach
    for binary descriptors.
    """
    if desc_a is None or desc_b is None or len(desc_a) == 0 or len(desc_b) == 0:
        return 0

    if use_sift:
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        k = 2 if len(desc_b) >= 2 else 1
        knn = matcher.knnMatch(desc_a, desc_b, k=k)
        good = 0
        for pair in knn:
            if len(pair) < 2:
                good += 1
                continue
            m, n = pair
            if m.distance < ratio * n.distance:
                good += 1
        return good

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc_a, desc_b)
    return sum(1 for m in matches if m.distance <= max_hamming_distance)
