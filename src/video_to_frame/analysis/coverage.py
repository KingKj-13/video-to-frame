"""Backward-compatibility shim -- the real implementation moved to
video_to_frame.coverage (Phase 4.1 restructure). Import from there in new
code; this path is kept working for existing callers."""

from ..coverage.coverage import CoveragePool

__all__ = ["CoveragePool"]
