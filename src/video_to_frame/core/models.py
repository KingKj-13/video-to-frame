from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class RejectionReason(str, Enum):
    NONE = "none"
    BLUR = "blur"
    REDUNDANT = "redundant"
    OVER_CAPACITY = "over_capacity"


class KeptReason(str, Enum):
    """Why a *kept* frame was kept -- distinct from RejectionReason, which is
    only meaningful for rejected frames."""
    NORMAL = "normal"
    MINIMUM_OVERLAP_PROTECTION = "minimum_overlap_protection"


@dataclass
class QualityMetrics:
    sharpness: float
    brightness: float
    contrast: float


@dataclass
class FrameMetrics:
    sharpness: float
    brightness: float
    contrast: float
    quality_norm: float
    feature_richness_norm: float
    redundancy_norm: float
    viewpoint_change_norm: float
    coverage_gain_norm: float
    score: float


@dataclass
class FrameRecord:
    index: int
    timestamp_ms: float
    metrics: Optional[FrameMetrics]
    kept: bool
    rejection_reason: RejectionReason
    output_filename: Optional[str] = None
    kept_reason: Optional[KeptReason] = None
    uncertain: bool = False
    uncertainty_reasons: List[str] = field(default_factory=list)


def record_to_dict(record: FrameRecord) -> dict:
    metrics = record.metrics
    return {
        "index": record.index,
        "timestamp_ms": record.timestamp_ms,
        "kept": record.kept,
        "rejection_reason": record.rejection_reason.value,
        "kept_reason": record.kept_reason.value if record.kept_reason else None,
        "output_filename": record.output_filename,
        "uncertain": record.uncertain,
        "uncertainty_reasons": record.uncertainty_reasons,
        "score": metrics.score if metrics else None,
        "sharpness": metrics.sharpness if metrics else None,
        "quality_norm": metrics.quality_norm if metrics else None,
        "feature_richness_norm": metrics.feature_richness_norm if metrics else None,
        "redundancy_norm": metrics.redundancy_norm if metrics else None,
        "viewpoint_change_norm": metrics.viewpoint_change_norm if metrics else None,
        "coverage_gain_norm": metrics.coverage_gain_norm if metrics else None,
    }
