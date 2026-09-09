"""Phase 2.5 section 1: per-stage profiling of the existing pipeline.

Uses cProfile (stdlib, zero changes to the tested pipeline code) rather than
hand-instrumenting every call site -- OpenCV/ORB/BFMatcher calls are each
distinct C-extension functions, so cProfile's function-level attribution
already separates "decode" from "grayscale conversion" from "ORB
detectAndCompute" from "descriptor matching" from "imwrite" without any
manual timer placement, and without risking a bug in already-tested code.

cProfile's call-tracing overhead inflates absolute times (typically 10-30%)
but does not change *relative* proportions between stages, which is what
this section needs: where does the time actually go.

Usage:
    python -m phase2_5.profile_pipeline <video> [output_dir]
"""

from __future__ import annotations

import cProfile
import os
import pstats
import sys
from typing import Dict, Optional, Tuple

from video_to_frame.config import PipelineConfig
from video_to_frame.pipeline import run_pipeline

STAGE_PATTERNS = {
    "Video decoding": ["extractor.py"],
    "Frame preprocessing (color convert/resize)": ["cvtColor", "resize_frame", "compressor.py"],
    "Blur/sharpness": ["quality.py"],
    "ORB feature extraction": ["extract", "features.py"],
    "Feature matching": ["match_descriptors", "BFMatcher", "knnMatch", "match'"],
    "Similarity (hash/SSIM)": ["similarity.py"],
    "Coverage analysis": ["coverage.py"],
    "Redundancy/selector logic": ["selector.py"],
    "Scoring": ["scorer.py"],
    "Image encoding": ["imwrite", "encode_and_save"],
    "Preview generation": ["video_writer.py", "subprocess"],
    "Report generation": ["report.py", "html_report.py", "json.dump", "json.dumps"],
    "Uncertainty flagging": ["uncertainty.py"],
}


def _categorize(func_key: Tuple[str, int, str]) -> str:
    filename, _lineno, funcname = func_key
    haystack = f"{filename}:{funcname}"
    for stage, patterns in STAGE_PATTERNS.items():
        if any(p in haystack for p in patterns):
            return stage
    return "Other"


def profile_run(video_path: str, output_dir: str, config: Optional[PipelineConfig] = None) -> Tuple[Dict[str, dict], dict]:
    config = config or PipelineConfig()
    profiler = cProfile.Profile()

    profiler.enable()
    report = run_pipeline(video_path, output_dir, config, show_progress=False)
    profiler.disable()

    stats = pstats.Stats(profiler)
    stage_time: Dict[str, float] = {}
    stage_calls: Dict[str, int] = {}
    for func_key, (_cc, num_calls, total_time, _cum_time, _callers) in stats.stats.items():
        stage = _categorize(func_key)
        stage_time[stage] = stage_time.get(stage, 0.0) + total_time
        stage_calls[stage] = stage_calls.get(stage, 0) + num_calls

    grand_total = sum(stage_time.values())
    result = {
        stage: {
            "total_time_sec": round(t, 3),
            "percent": round((t / grand_total * 100) if grand_total else 0.0, 2),
            "calls": stage_calls[stage],
        }
        for stage, t in stage_time.items()
    }
    result["_grand_total_sec"] = round(grand_total, 3)
    return result, report


def print_stage_report(result: Dict[str, dict]) -> None:
    grand_total = result.pop("_grand_total_sec")
    print(f"{'Stage':<42}{'Time':>10}{'% Total':>10}{'Calls':>12}")
    print("-" * 74)
    for stage, d in sorted(result.items(), key=lambda kv: -kv[1]["total_time_sec"]):
        print(f"{stage:<42}{d['total_time_sec']:>9.2f}s{d['percent']:>9.1f}%{d['calls']:>12}")
    print("-" * 74)
    print(f"{'Total (cProfile cumulative, self-time only)':<42}{grand_total:>9.2f}s")
    result["_grand_total_sec"] = grand_total


if __name__ == "__main__":
    video = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "profile_out")
    stage_result, _report = profile_run(video, out)
    print_stage_report(stage_result)
