from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator, Tuple

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int
    size_bytes: int
    duration_sec: float


class FrameExtractor:
    """Streams decoded frames from a video in original order, with timestamps.

    Uses cv2.VideoCapture (OpenCV's bundled ffmpeg backend) so mp4/mov/avi/mkv
    are all handled without shelling out. Frames are decoded one at a time --
    never materializes the whole video in memory.
    """

    def __init__(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Input video not found: {path}")
        self.path = path
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise IOError(f"Could not open video (unsupported codec/container?): {path}")

        fps = self._cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        size_bytes = os.path.getsize(path)
        duration_sec = (frame_count / fps) if fps > 0 else 0.0

        self.video_info = VideoInfo(
            path=path,
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
            size_bytes=size_bytes,
            duration_sec=duration_sec,
        )
        self._released = False

    def frames(self) -> Iterator[Tuple[np.ndarray, int, float]]:
        """Yields (frame_bgr, index, timestamp_ms) in temporal order."""
        index = 0
        try:
            while True:
                ok, frame = self._cap.read()
                if not ok:
                    break
                timestamp_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
                yield frame, index, timestamp_ms
                index += 1
        finally:
            self._cap.release()
            self._released = True
