"""Backward-compatibility shim -- the real implementation moved to
video_to_frame.quality (Phase 4.1 restructure). Import from there in new
code; this path is kept working for existing callers."""

from ..quality.quality import brightness_contrast, compute_quality_metrics, laplacian_variance

__all__ = ["brightness_contrast", "compute_quality_metrics", "laplacian_variance"]
