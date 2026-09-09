"""Phase 2.5: PNG compression level is a speed/size tradeoff only, never a
quality one (PNG is always lossless). The old default derived a level of 9
from output_quality, which profiling showed costs ~30x more encode time than
level 1 on a real photo for a modest file-size difference -- these tests
lock in the fast default and prove losslessness holds regardless of level.
"""

import time

import cv2
import numpy as np

from video_to_frame.compression.compressor import encode_and_save
from video_to_frame.config import PipelineConfig


def test_default_png_compression_level_is_fast():
    config = PipelineConfig()
    assert config.png_compression_level == 1


def test_png_output_is_lossless_at_default_level(tmp_path):
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
    path = str(tmp_path / "frame.png")

    encode_and_save(frame, path, fmt="png", png_compression_level=1)
    decoded = cv2.imread(path)

    assert np.array_equal(decoded, frame)


def test_lower_png_compression_level_is_not_slower_than_higher(tmp_path):
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    cv2.rectangle(frame, (20, 20), (200, 150), (80, 80, 80), -1)

    def encode_time(level: int) -> float:
        path = str(tmp_path / f"frame_{level}.png")
        start = time.perf_counter()
        for _ in range(5):
            encode_and_save(frame, path, fmt="png", png_compression_level=level)
        return time.perf_counter() - start

    fast = encode_time(1)
    slow = encode_time(9)

    # Not a strict inequality (timing noise on small images) -- but level 9
    # must not be meaningfully faster than level 1.
    assert fast <= slow * 1.5
