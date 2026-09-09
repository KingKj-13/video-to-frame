"""Phase 3.1/3.4: footage-scenario robustness beyond the existing synthetic
content-variation generators (static_object/indoor/difficult_lighting in
phase1/benchmarks/generators.py, which vary *scene content* at a fixed
320x240/30fps/H.264 encoding). These tests instead vary *codec/resolution/
frame-timing* axes -- 4K resolution, HEVC encoding, and irregular
(variable-rate) frame timestamps -- none of which any existing test
exercises. Real footage for these three specific conditions wasn't
available; per explicit direction, synthetic approximations are used
instead, clearly labeled as such here and not claimed to validate real 4K/
HEVC/VFR camera output -- only that the pipeline decodes and processes these
container/timing variations without crashing or silently mishandling them.
"""

from __future__ import annotations

import shutil
import subprocess

import cv2
import pytest

from phase1.benchmarks.generators import _landmark_canvas
from video_to_frame.config import PipelineConfig
from video_to_frame.pipeline import run_pipeline

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None
requires_ffmpeg = pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not on PATH")


def _panning_frames(n_frames, w, h, canvas_scale=3, density=28):
    canvas_w = w * canvas_scale
    canvas = _landmark_canvas(canvas_w, h, seed=99, density=density)
    max_offset = canvas_w - w
    frames = []
    for i in range(n_frames):
        offset = int((i / max(1, n_frames - 1)) * max_offset)
        frames.append(canvas[:, offset:offset + w].copy())
    return frames


def _write_pngs(frames, png_dir):
    png_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(str(png_dir / f"frame_{i:04d}.png"), frame)


@requires_ffmpeg
def test_4k_resolution_does_not_crash_and_produces_correct_output_resolution(tmp_path):
    # Only 6 frames: this is testing that 4K decode/analysis runs correctly
    # end to end, not benchmarking throughput at scale (that's phase1's
    # performance_benchmark / phase2_5's profiling, which already cover
    # runtime scaling separately).
    w, h, n_frames = 3840, 2160, 6
    frames = _panning_frames(n_frames, w, h, canvas_scale=2, density=90)
    png_dir = tmp_path / "pngs"
    _write_pngs(frames, png_dir)

    video_path = str(tmp_path / "synthetic_4k.mp4")
    cmd = ["ffmpeg", "-y", "-framerate", "24", "-i", str(png_dir / "frame_%04d.png"),
           "-pix_fmt", "yuv420p", "-c:v", "libx264", video_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]

    config = PipelineConfig(hard_blur_floor=0.0, blur_threshold=0.0)
    report = run_pipeline(video_path, str(tmp_path / "out_4k"), config, show_progress=False)

    assert report["summary"]["selected_frame_count"] > 0
    assert report["summary"]["source_resolution"] == f"{w}x{h}"


@requires_ffmpeg
def test_hevc_codec_decodes_and_processes_without_error(tmp_path):
    w, h, n_frames = 640, 480, 10
    frames = _panning_frames(n_frames, w, h, canvas_scale=3, density=28)
    png_dir = tmp_path / "pngs"
    _write_pngs(frames, png_dir)

    video_path = str(tmp_path / "synthetic_hevc.mp4")
    cmd = ["ffmpeg", "-y", "-framerate", "24", "-i", str(png_dir / "frame_%04d.png"),
           "-pix_fmt", "yuv420p", "-c:v", "libx265", "-tag:v", "hvc1", video_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]

    config = PipelineConfig(hard_blur_floor=0.0, blur_threshold=0.0)
    report = run_pipeline(video_path, str(tmp_path / "out_hevc"), config, show_progress=False)

    assert report["summary"]["original_frame_count"] == n_frames
    assert report["summary"]["selected_frame_count"] > 0


@requires_ffmpeg
def test_variable_frame_timing_is_read_correctly_not_assumed_constant(tmp_path):
    w, h, n_frames = 320, 240, 12
    frames = _panning_frames(n_frames, w, h, canvas_scale=3, density=28)
    png_dir = tmp_path / "pngs"
    _write_pngs(frames, png_dir)

    # Alternate short/long per-frame display durations -- a crude but
    # concrete approximation of variable-frame-rate capture (some mobile/
    # drone cameras adapt capture rate to motion or light). Reuses the same
    # ffmpeg concat-demuxer + per-file `duration` pattern already used by
    # io/video_writer.py's write_preview_video for the same reason.
    durations = [0.033 if i % 3 != 0 else 0.25 for i in range(n_frames)]
    list_file = tmp_path / "concat.txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for i in range(n_frames):
            p = (png_dir / f"frame_{i:04d}.png").resolve().as_posix()
            f.write(f"file '{p}'\n")
            f.write(f"duration {durations[i]}\n")

    video_path = str(tmp_path / "synthetic_vfr.mp4")
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
           "-vsync", "vfr", "-pix_fmt", "yuv420p", "-c:v", "libx264", video_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]

    config = PipelineConfig(hard_blur_floor=0.0, blur_threshold=0.0)
    report = run_pipeline(video_path, str(tmp_path / "out_vfr"), config, show_progress=False)

    timestamps = [f["timestamp_ms"] for f in report["frames"]]
    deltas = [round(b - a) for a, b in zip(timestamps, timestamps[1:])]
    assert len(set(deltas)) > 1, (
        f"expected non-uniform frame timestamps from a variable-duration source, got deltas={deltas}"
    )
    assert report["summary"]["selected_frame_count"] > 0
