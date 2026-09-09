"""Phase 3.6: individual bad inputs must fail clearly and cleanly, never
silently or by leaving misleading partial output behind.
"""

import os

import cv2
import numpy as np
import pytest

from video_to_frame.config import PipelineConfig
from video_to_frame.pipeline import run_pipeline


def test_missing_video_raises_clear_error_and_creates_no_output(tmp_path):
    missing = str(tmp_path / "does_not_exist.mp4")
    out_dir = str(tmp_path / "out")

    with pytest.raises(FileNotFoundError):
        run_pipeline(missing, out_dir, PipelineConfig(), show_progress=False)

    assert not os.path.isdir(out_dir)


def test_empty_file_raises_clear_error_and_creates_no_output(tmp_path):
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    out_dir = str(tmp_path / "out")

    with pytest.raises(OSError):
        run_pipeline(str(empty), out_dir, PipelineConfig(), show_progress=False)

    assert not os.path.isdir(out_dir)


def test_non_video_file_raises_clear_error_and_creates_no_output(tmp_path):
    fake = tmp_path / "fake.mp4"
    fake.write_text("not a video" * 100)
    out_dir = str(tmp_path / "out")

    with pytest.raises(OSError):
        run_pipeline(str(fake), out_dir, PipelineConfig(), show_progress=False)

    assert not os.path.isdir(out_dir)


def test_all_frames_rejected_does_not_crash_and_warns(tmp_path):
    video_path = str(tmp_path / "all_blurry.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, 30, (200, 200))
    rng = np.random.default_rng(0)
    for _ in range(10):
        img = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
        img = cv2.GaussianBlur(img, (41, 41), 30)
        writer.write(img)
    writer.release()

    out_dir = str(tmp_path / "out")
    report = run_pipeline(video_path, out_dir, PipelineConfig(), show_progress=False)

    assert report["summary"]["selected_frame_count"] == 0
    assert not os.path.isfile(os.path.join(out_dir, "optimized_preview.mp4"))
    assert os.path.isfile(os.path.join(out_dir, "report.json"))
    assert any("No frames were selected" in w for w in report["warnings"])
