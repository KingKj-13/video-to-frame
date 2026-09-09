"""Phase 4.8: a formal, structured programmatic API.

`run_pipeline` (pipeline.py) already IS the core, CLI-independent entry
point -- phase1's evaluation framework already calls it directly. This
module adds a typed, attribute-access result on top for callers who don't
want to parse a raw report dict, without changing run_pipeline's existing
contract (report_json / phase1 / any other current caller keeps working
unmodified).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

from .config import PipelineConfig
from .pipeline import run_pipeline


@dataclass
class ProcessResult:
    input_frames: int
    output_frames: int
    reduction_ratio: float  # 0-1, fraction of input frames removed
    rejected_blur: int
    rejected_redundant: int
    rejected_over_capacity: int
    forced_keeps: int
    uncertain_frames: int
    runtime_sec: float
    output_dir: str
    report_path: str
    report_html_path: str
    preview_path: Optional[str]
    warnings: list
    raw_report: dict  # full report.json contents, for anything not surfaced above


def process_video(input_path: str, output_dir: str, config: Optional[PipelineConfig] = None,
                   show_progress: bool = False) -> ProcessResult:
    """The recommended entry point for any non-CLI caller (a batch script,
    a future GUI, a notebook). CLI/batch/GUI code should call this, not
    reach into pipeline.run_pipeline's dict directly."""
    config = config or PipelineConfig()

    start = time.time()
    report = run_pipeline(input_path, output_dir, config, show_progress=show_progress)
    runtime = time.time() - start

    summary = report["summary"]
    preview_path = os.path.join(output_dir, "optimized_preview.mp4")

    return ProcessResult(
        input_frames=summary["original_frame_count"],
        output_frames=summary["selected_frame_count"],
        reduction_ratio=summary["frame_reduction_percent"] / 100.0,
        rejected_blur=summary["frames_rejected_blur"],
        rejected_redundant=summary["frames_rejected_redundant"],
        rejected_over_capacity=summary["frames_rejected_over_capacity"],
        forced_keeps=summary["frames_forced_by_minimum_overlap"],
        uncertain_frames=summary["uncertain_kept_frames"],
        runtime_sec=runtime,
        output_dir=output_dir,
        report_path=os.path.join(output_dir, "report.json"),
        report_html_path=os.path.join(output_dir, "report.html"),
        preview_path=preview_path if os.path.isfile(preview_path) else None,
        warnings=report.get("warnings", []),
        raw_report=report,
    )
