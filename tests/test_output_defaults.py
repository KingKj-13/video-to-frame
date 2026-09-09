"""Test D: default output is quality-first (PNG, original resolution), and
explicit overrides (webp/jpg + resizing) still work."""

import os

from video_to_frame.config import PipelineConfig
from video_to_frame.pipeline import run_pipeline


def test_default_config_is_png_and_original_resolution():
    config = PipelineConfig()
    assert config.output_format == "png"
    assert config.resize_long_edge is None
    assert config.resize_width is None
    assert config.resize_height is None


def test_default_pipeline_run_produces_full_resolution_png(tmp_path, synthetic_video):
    video_path, _ = synthetic_video
    out_dir = str(tmp_path / "out")

    config = PipelineConfig()
    report = run_pipeline(video_path, out_dir, config, show_progress=False)

    assert report["summary"]["output_format"] == "png"
    frames_dir = os.path.join(out_dir, "frames")
    frame_files = [f for f in os.listdir(frames_dir) if f.endswith(".png")]
    assert frame_files, "expected PNG frames in the default configuration"

    import cv2
    sample = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    assert sample.shape[:2] == (240, 320)  # matches synthetic_video's source resolution, unresized


def test_explicit_webp_and_resize_still_work(tmp_path, synthetic_video):
    video_path, _ = synthetic_video
    out_dir = str(tmp_path / "out")

    config = PipelineConfig(output_format="webp", output_quality=80, resize_long_edge=160)
    report = run_pipeline(video_path, out_dir, config, show_progress=False)

    assert report["summary"]["output_format"] == "webp"
    frames_dir = os.path.join(out_dir, "frames")
    frame_files = [f for f in os.listdir(frames_dir) if f.endswith(".webp")]
    assert frame_files

    import cv2
    sample = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    assert max(sample.shape[:2]) <= 160
