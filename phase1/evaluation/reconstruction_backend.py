"""Backward-compatibility shim -- promoted to video_to_frame.reconstruction
(Phase 4.1/4.14: the reconstruction backend interface is now part of the
core, pluggable system, not locked into this evaluation-only package).
Import from there in new code; this path is kept working for existing
callers within phase1."""

from video_to_frame.reconstruction import (
    ExternalCommandBackend,
    NullBackend,
    PycolmapBackend,
    ReconstructionBackend,
    ReconstructionMetrics,
    build_backend,
)

__all__ = [
    "ExternalCommandBackend", "NullBackend", "PycolmapBackend",
    "ReconstructionBackend", "ReconstructionMetrics", "build_backend",
]
