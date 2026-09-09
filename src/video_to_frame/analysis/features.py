"""Backward-compatibility shim -- the real implementation moved to
video_to_frame.features (Phase 4.1 restructure). Import from there in new
code; this path is kept working for existing callers."""

from ..features.features import FeatureExtractor, match_descriptors

__all__ = ["FeatureExtractor", "match_descriptors"]
