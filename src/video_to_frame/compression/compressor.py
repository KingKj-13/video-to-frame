"""Backward-compatibility shim -- the real implementation moved to
video_to_frame.output (Phase 4.1 restructure). Import from there in new
code; this path is kept working for existing callers."""

from ..output.compressor import encode_and_save, output_extension, resize_frame

__all__ = ["encode_and_save", "output_extension", "resize_frame"]
