"""Backward-compatibility shim -- the real implementation moved to
video_to_frame.features.similarity (Phase 4.1 restructure). Import from
there in new code; this path is kept working for existing callers."""

from ..features.similarity import dhash, hamming_distance, ssim_precomputed, ssim_score

__all__ = ["dhash", "hamming_distance", "ssim_precomputed", "ssim_score"]
