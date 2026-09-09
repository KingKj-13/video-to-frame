from __future__ import annotations

import numpy as np

from ..config import ScoreWeights
from ..core.models import FrameMetrics


class RunningNormalizer:
    """Normalizes a value to [0, 1] against the max seen so far in this video.
    Avoids needing a fixed, dataset-specific reference (Laplacian variance and
    keypoint counts vary wildly with resolution/content)."""

    def __init__(self, initial_max: float = 1e-6):
        self.max_value = initial_max

    def normalize(self, value: float) -> float:
        self.max_value = max(self.max_value, value)
        if self.max_value <= 0:
            return 0.0
        return float(np.clip(value / self.max_value, 0.0, 1.0))


def exposure_score(brightness: float, contrast: float, low: float, high: float, min_contrast: float) -> float:
    if brightness < low or brightness > high:
        return 0.3
    if contrast < min_contrast:
        return 0.5
    return 1.0


class FrameScorer:
    """Combines quality/feature/viewpoint/coverage/redundancy sub-scores into
    the spec's weighted frame score. All sub-scores are normalized to [0, 1]
    before weighting."""

    def __init__(self, weights: ScoreWeights, exposure_low: float, exposure_high: float, exposure_min_contrast: float):
        self.weights = weights
        self.exposure_low = exposure_low
        self.exposure_high = exposure_high
        self.exposure_min_contrast = exposure_min_contrast
        self._sharpness_norm = RunningNormalizer()
        self._feature_norm = RunningNormalizer()

    def score(self, sharpness: float, brightness: float, contrast: float, keypoint_count: int,
              redundancy: float, viewpoint_change: float, coverage_gain: float) -> FrameMetrics:
        sharpness_n = self._sharpness_norm.normalize(sharpness)
        exposure = exposure_score(brightness, contrast, self.exposure_low, self.exposure_high, self.exposure_min_contrast)
        quality_n = sharpness_n * exposure
        feature_n = self._feature_norm.normalize(keypoint_count)

        w = self.weights
        total_score = (
            w.quality * quality_n
            + w.sharpness * sharpness_n
            + w.features * feature_n
            + w.viewpoint * viewpoint_change
            + w.coverage * coverage_gain
            - w.redundancy * redundancy
        )

        return FrameMetrics(
            sharpness=sharpness,
            brightness=brightness,
            contrast=contrast,
            quality_norm=quality_n,
            feature_richness_norm=feature_n,
            redundancy_norm=redundancy,
            viewpoint_change_norm=viewpoint_change,
            coverage_gain_norm=coverage_gain,
            score=total_score,
        )
