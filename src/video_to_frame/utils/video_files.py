"""Small shared helpers with no better home in a specific pipeline stage.
Kept deliberately minimal -- consolidated from a genuine duplication
(VIDEO_EXTENSIONS existed identically in both cli and batch) rather than
created speculatively."""

from __future__ import annotations

import os

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def is_video_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS
