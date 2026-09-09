"""Backward-compatibility shim -- the real implementation moved to
video_to_frame.scoring (Phase 4.1 restructure: scoring is conceptually
separate from selection, even though the streaming selector is the scorer's
only current caller). Import from there in new code; this path is kept
working for existing callers."""

from ..scoring.scorer import FrameScorer, RunningNormalizer, exposure_score

__all__ = ["FrameScorer", "RunningNormalizer", "exposure_score"]
