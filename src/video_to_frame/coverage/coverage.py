from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class CoveragePool:
    """Accumulates descriptors from every *kept* frame to answer: does this
    candidate contain information not captured anywhere yet in the video?

    This is what distinguishes a real revisit of an already-seen viewpoint
    (should not be treated as novel just because it's temporally far from the
    last kept frame) from a genuinely new viewpoint -- the redundancy-cluster
    logic in the selector only looks at *recent* reference frames, so this
    pool is the guard against that blind spot.
    """

    def __init__(self, use_sift: bool = False, max_per_frame: int = 60,
                 max_pool_size: int = 20000, ratio: float = 0.75,
                 max_hamming_distance: int = 64, random_seed: int = 0):
        self.use_sift = use_sift
        self.max_per_frame = max_per_frame
        self.max_pool_size = max_pool_size
        self.ratio = ratio
        self.max_hamming_distance = max_hamming_distance
        self._pool: Optional[np.ndarray] = None
        # See analysis.features.match_descriptors for why ORB (Hamming) uses
        # cross-check + an absolute distance cutoff instead of a ratio test.
        self._matcher = cv2.BFMatcher(cv2.NORM_L2) if use_sift else cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Phase 3.9 (determinism): a fresh, unseeded np.random.default_rng()
        # was previously created on every add() call -- the same input video
        # could select a different random descriptor subsample each run,
        # which changed coverage_gain for later frames and could shift
        # selection decisions. One seeded generator, created once, makes
        # subsampling reproducible for the same input/config.
        self._rng = np.random.default_rng(random_seed)

    def coverage_gain(self, descriptors: Optional[np.ndarray], extra_pool: Optional[np.ndarray] = None) -> float:
        """Fraction of `descriptors` that do NOT match anything in the pool.
        1.0 means fully novel information; 0.0 means fully already covered.

        `extra_pool` lets a caller fold in descriptors that aren't permanently
        committed yet (e.g. a currently-open redundancy cluster's anchor frame)
        without calling `add()`.
        """
        if descriptors is None or len(descriptors) == 0:
            return 0.0

        pool = self._pool
        if extra_pool is not None and len(extra_pool) > 0:
            pool = extra_pool if pool is None else np.vstack([pool, extra_pool])
        if pool is None or len(pool) == 0:
            return 1.0

        if self.use_sift:
            k = 2 if len(pool) >= 2 else 1
            knn = self._matcher.knnMatch(descriptors, pool, k=k)
            unmatched = 0
            for pair in knn:
                if len(pair) < 2:
                    unmatched += 1
                    continue
                m, n = pair
                if m.distance >= self.ratio * n.distance:
                    unmatched += 1
            return unmatched / len(descriptors)

        matches = self._matcher.match(descriptors, pool)
        matched_query_idx = {m.queryIdx for m in matches if m.distance <= self.max_hamming_distance}
        return 1.0 - (len(matched_query_idx) / len(descriptors))

    def add(self, descriptors: Optional[np.ndarray]) -> None:
        if descriptors is None or len(descriptors) == 0:
            return
        sample = descriptors
        if len(sample) > self.max_per_frame:
            idx = self._rng.choice(len(sample), self.max_per_frame, replace=False)
            sample = sample[idx]
        self._pool = sample if self._pool is None else np.vstack([self._pool, sample])
        if len(self._pool) > self.max_pool_size:
            self._pool = self._pool[-self.max_pool_size:]
