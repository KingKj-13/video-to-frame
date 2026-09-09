"""Phase 4.7: process multiple videos, one output subfolder per video, plus
a batch summary. One failed video does not stop the others -- each is
independent, and failures are collected and reported rather than raised.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional

from .api import ProcessResult, process_video
from .config import PipelineConfig
from .utils import VIDEO_EXTENSIONS, is_video_file


@dataclass
class BatchItemResult:
    video_path: str
    success: bool
    result: Optional[ProcessResult] = None
    error: Optional[str] = None


def find_videos(input_dir: str) -> List[str]:
    return sorted(
        os.path.join(input_dir, f) for f in os.listdir(input_dir) if is_video_file(f)
    )


def process_batch(input_dir: str, output_root: str, config: Optional[PipelineConfig] = None,
                   show_progress: bool = False) -> List[BatchItemResult]:
    config = config or PipelineConfig()
    videos = find_videos(input_dir)
    if not videos:
        raise FileNotFoundError(f"No video files found in {input_dir}")

    os.makedirs(output_root, exist_ok=True)
    results: List[BatchItemResult] = []

    for video_path in videos:
        name = os.path.splitext(os.path.basename(video_path))[0]
        out_dir = os.path.join(output_root, name)
        try:
            result = process_video(video_path, out_dir, config, show_progress=show_progress)
            results.append(BatchItemResult(video_path=video_path, success=True, result=result))
        except Exception as exc:  # one bad video must not abort the batch
            results.append(BatchItemResult(video_path=video_path, success=False, error=str(exc)))

    write_batch_summary(results, os.path.join(output_root, "batch_summary.json"))
    return results


def write_batch_summary(results: List[BatchItemResult], path: str) -> None:
    summary = {
        "total": len(results),
        "succeeded": sum(1 for r in results if r.success),
        "failed": sum(1 for r in results if not r.success),
        "videos": [],
    }
    for r in results:
        if r.success:
            summary["videos"].append({
                "video": r.video_path, "success": True,
                "input_frames": r.result.input_frames, "output_frames": r.result.output_frames,
                "reduction_percent": round(r.result.reduction_ratio * 100, 2),
                "runtime_sec": round(r.result.runtime_sec, 2),
                "fps": round(r.result.input_frames / r.result.runtime_sec, 2) if r.result.runtime_sec else None,
                "output_dir": r.result.output_dir,
            })
        else:
            summary["videos"].append({"video": r.video_path, "success": False, "error": r.error})

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def print_batch_table(results: List[BatchItemResult]) -> None:
    print(f"{'Video':<40}{'Input':>8}{'Output':>8}{'Reduction':>11}{'Time':>10}{'FPS':>8}")
    print("-" * 85)
    for r in results:
        name = os.path.basename(r.video_path)
        if r.success:
            fps = r.result.input_frames / r.result.runtime_sec if r.result.runtime_sec else 0
            print(f"{name:<40}{r.result.input_frames:>8}{r.result.output_frames:>8}"
                  f"{r.result.reduction_ratio*100:>10.1f}%{r.result.runtime_sec:>9.1f}s{fps:>8.1f}")
        else:
            print(f"{name:<40}{'FAILED':>8}  {r.error}")
    print("-" * 85)
    succeeded = sum(1 for r in results if r.success)
    print(f"{succeeded}/{len(results)} succeeded")
