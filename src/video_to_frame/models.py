"""Backward-compatibility shim -- the real implementation moved to
video_to_frame.core.models (Phase 4.1 restructure). Import from there in
new code; this path is kept working for existing callers."""

from .core.models import (
    FrameMetrics,
    FrameRecord,
    KeptReason,
    QualityMetrics,
    RejectionReason,
    record_to_dict,
)

__all__ = [
    "FrameMetrics", "FrameRecord", "KeptReason", "QualityMetrics", "RejectionReason", "record_to_dict",
]
