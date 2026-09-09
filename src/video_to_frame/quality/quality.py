from __future__ import annotations

import cv2
import numpy as np

from ..core.models import QualityMetrics


def laplacian_variance(gray: np.ndarray) -> float:
    """Standard sharpness proxy: variance of the Laplacian. Lower = blurrier."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_contrast(gray: np.ndarray) -> tuple[float, float]:
    return float(gray.mean()), float(gray.std())


def compute_quality_metrics(frame_bgr: np.ndarray) -> QualityMetrics:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    sharpness = laplacian_variance(gray)
    brightness, contrast = brightness_contrast(gray)
    return QualityMetrics(sharpness=sharpness, brightness=brightness, contrast=contrast)
