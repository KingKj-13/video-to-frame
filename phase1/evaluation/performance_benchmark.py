"""Large-scale performance benchmark (Phase 1.1 spec section 8): processing
time, throughput, and peak memory as frame count scales up, to confirm the
pipeline stays streaming rather than loading the whole video into memory.

Uses synthetic footage generated at each target size (see
phase1/benchmarks/generators.py's landmark-canvas approach) rather than real
footage, specifically so the largest tiers (15k/30k frames) are reachable at
all in an interactive session -- generating and decoding real 4K/1080p
footage at that length would itself take a prohibitively long time here.
This measures the pipeline's *scaling behavior*, not its output quality on
real content (that's what the other benchmark categories are for).

If a tier is too expensive to actually run here, it is extrapolated from
the measured throughput of smaller tiers and clearly labeled "estimated",
never presented as measured data.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from video_to_frame.config import PipelineConfig
from video_to_frame.pipeline import run_pipeline

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


@dataclass
class PerformanceResult:
    frame_count: int
    measured: bool
    processing_time_sec: Optional[float] = None
    frames_per_sec: Optional[float] = None
    peak_memory_mb: Optional[float] = None
    selected_frame_count: Optional[int] = None
    note: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "frame_count": self.frame_count,
            "measured": self.measured,
            "processing_time_sec": round(self.processing_time_sec, 2) if self.processing_time_sec else None,
            "frames_per_sec": round(self.frames_per_sec, 2) if self.frames_per_sec else None,
            "peak_memory_mb": round(self.peak_memory_mb, 1) if self.peak_memory_mb else None,
            "selected_frame_count": self.selected_frame_count,
            "note": self.note,
        }


def generate_perf_video(path: str, n_frames: int, w: int = 320, h: int = 240, fps: int = 30) -> None:
    """Cheap, fast-to-generate synthetic footage for scale testing: a
    landmark canvas panned across in a loop, so ORB has real keypoints
    (same rationale as tests/synthetic.py) without the cost of real video
    decode/encode at these frame counts."""
    canvas_w = w * 3
    canvas = np.full((h, canvas_w, 3), 40, dtype=np.uint8)
    canvas = cv2.add(canvas, np.random.default_rng(7).integers(0, 25, size=canvas.shape, dtype=np.uint8))
    rng = np.random.default_rng(8)
    for cx in range(20, canvas_w - 20, 30):
        cy = int(rng.integers(30, h - 30))
        size = int(rng.integers(5, 14))
        color = tuple(int(c) for c in rng.integers(50, 230, size=3))
        cv2.circle(canvas, (cx, cy), size, color, -1)
    wrapped = np.hstack([canvas, canvas[:, :w]])

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for i in range(n_frames):
        offset = int((i / n_frames) * canvas_w) % canvas_w
        writer.write(wrapped[:, offset:offset + w].copy())
    writer.release()


def _peak_memory_mb(process: "psutil.Process") -> Optional[float]:
    if not _HAS_PSUTIL:
        return None
    try:
        info = process.memory_info()
        # Windows exposes peak working set directly; fall back to current RSS elsewhere.
        peak = getattr(info, "peak_wset", None) or info.rss
        return peak / (1024 * 1024)
    except Exception:
        return None


def run_performance_tier(video_dir: str, frame_count: int, config: Optional[PipelineConfig] = None) -> PerformanceResult:
    """Generates a synthetic video with `frame_count` frames and runs the
    real pipeline (video_to_frame.pipeline.run_pipeline) over it, measuring
    wall time and peak process memory. Uses webp output (not the PNG
    default) to keep the benchmark's own disk I/O from dominating the
    measurement -- output format is a compression concern, not a scaling one.
    """
    config = config or PipelineConfig(output_format="webp", resize_long_edge=None)
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, f"perf_{frame_count}.mp4")
    generate_perf_video(video_path, frame_count)

    process = psutil.Process(os.getpid()) if _HAS_PSUTIL else None
    out_dir = os.path.join(video_dir, f"perf_{frame_count}_out")

    start = time.time()
    report = run_pipeline(video_path, out_dir, config, show_progress=False)
    elapsed = time.time() - start
    peak_mb = _peak_memory_mb(process) if process else None

    return PerformanceResult(
        frame_count=frame_count, measured=True,
        processing_time_sec=elapsed, frames_per_sec=(frame_count / elapsed if elapsed else None),
        peak_memory_mb=peak_mb, selected_frame_count=report["summary"]["selected_frame_count"],
        note=None if _HAS_PSUTIL else "psutil unavailable -- peak memory not recorded",
    )


def extrapolate_tier(frame_count: int, measured: List[PerformanceResult]) -> PerformanceResult:
    """For a tier not actually run, estimates time/throughput by linear
    extrapolation from the measured tiers' average frames/sec -- explicitly
    labeled as an estimate, never presented as measured."""
    valid = [r for r in measured if r.measured and r.frames_per_sec]
    if not valid:
        return PerformanceResult(frame_count=frame_count, measured=False,
                                  note="No measured tiers available to extrapolate from.")
    avg_fps = sum(r.frames_per_sec for r in valid) / len(valid)
    est_time = frame_count / avg_fps if avg_fps else None
    return PerformanceResult(
        frame_count=frame_count, measured=False,
        processing_time_sec=est_time, frames_per_sec=avg_fps,
        note=f"ESTIMATED by linear extrapolation from measured throughput ({avg_fps:.1f} fps avg "
             f"across {len(valid)} measured tier(s)) -- not run directly; real throughput may differ "
             "at this scale due to memory/cache effects the measured tiers don't exercise.",
    )
