"""Backward-compatibility shim -- the real implementation moved to
video_to_frame.core.pipeline (Phase 4.1 restructure). Import from there in
new code; this path is kept working for existing callers (including
phase1's evaluation framework, which imports `from video_to_frame.pipeline
import run_pipeline` extensively)."""

from .core.pipeline import KeptPair, run_pipeline, _enforce_max_frames

__all__ = ["KeptPair", "run_pipeline"]
