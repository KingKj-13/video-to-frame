from __future__ import annotations

import os
import shutil
import subprocess
from typing import List


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def write_preview_video(frame_paths: List[str], output_path: str, fps: float) -> None:
    """Muxes the selected frames, in their given (temporal) order, into an mp4.

    Each frame is shown for a constant 1/fps duration -- this is a *preview*
    of what information was retained, not a real-time reconstruction of the
    original video's timing (selected frames are temporally sparse and real
    gaps between them can be very uneven).
    """
    if not frame_paths:
        return
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg not found on PATH; required to build optimized_preview.mp4")

    fps = fps if fps and fps > 0 else 5.0
    duration = 1.0 / fps

    list_file = output_path + ".concat.txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for p in frame_paths:
            abs_path = os.path.abspath(p).replace("\\", "/")
            f.write(f"file '{abs_path}'\n")
            f.write(f"duration {duration}\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", list_file,
        "-vsync", "vfr", "-pix_fmt", "yuv420p", "-c:v", "libx264",
        output_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    finally:
        os.remove(list_file)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to build preview video:\n{result.stderr[-2000:]}")
