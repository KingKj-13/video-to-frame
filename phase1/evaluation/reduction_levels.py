"""Runs the optimizer at several redundancy-clustering settings to
approximate a spread of frame-reduction levels (Phase 1 spec section 2), for
comparing DATA REDUCTION against the CONNECTIVITY PROXY (not reconstruction
quality -- see connectivity_metrics.py's module note).

This deliberately does NOT force the optimizer to hit an exact percentage --
target_levels are reference points for the sweep, not requirements. Whatever
reduction each configuration naturally achieves (which minimum-overlap
protection may pull back up from a more aggressive setting) is reported as
the real "achieved" figure.
"""

from __future__ import annotations

import dataclasses
import os
import time
from dataclasses import dataclass
from typing import List, Optional

from video_to_frame.config import PipelineConfig
from video_to_frame.pipeline import run_pipeline

from .connectivity_metrics import ConnectivityMetrics, compute_connectivity_metrics

DEFAULT_TARGET_LEVELS = [0.0, 25.0, 50.0, 75.0, 90.0, 95.0]  # percent reduction; 0 = full-frame baseline


@dataclass
class ReductionLevelResult:
    target_reduction_percent: float
    achieved_reduction_percent: float
    selected_frame_count: int
    original_frame_count: int
    output_dir: str
    dataset_size_bytes: int
    processing_time_sec: float
    config_used: dict
    connectivity: ConnectivityMetrics
    report: dict

    def to_dict(self) -> dict:
        return {
            "target_reduction_percent": self.target_reduction_percent,
            "achieved_reduction_percent": self.achieved_reduction_percent,
            "selected_frame_count": self.selected_frame_count,
            "original_frame_count": self.original_frame_count,
            "output_dir": self.output_dir,
            "dataset_size_bytes": self.dataset_size_bytes,
            "processing_time_sec": round(self.processing_time_sec, 2),
            "connectivity": self.connectivity.to_dict(),
        }


def _config_for_target(base_config: PipelineConfig, target_percent: float) -> PipelineConfig:
    if target_percent <= 0:
        # Full-frame baseline: clustering effectively disabled (still applies
        # the hard blur floor and minimum-overlap protection is moot with
        # nothing being rejected).
        return dataclasses.replace(base_config, redundancy_threshold=0.999, coverage_gain_threshold=0.0)

    # Coarse, monotonic mapping from "more reduction wanted" to "more
    # permissive redundancy clustering". Not a search for an exact
    # percentage -- min_overlap_threshold still overrides this per-candidate
    # regardless of how permissive redundancy_threshold gets, which is why
    # the achieved reduction can plateau well below the target (a real,
    # meaningful signal -- see the Phase 1 report, not a bug in this sweep).
    redundancy_threshold = max(0.5, 0.95 - (target_percent / 100.0) * 0.45)
    return dataclasses.replace(base_config, redundancy_threshold=redundancy_threshold)


def evaluate_reduction_levels(
    video_path: str,
    output_root: str,
    base_config: Optional[PipelineConfig] = None,
    target_levels: Optional[List[float]] = None,
    weak_link_threshold: int = 15,
    show_progress: bool = False,
) -> List[ReductionLevelResult]:
    base_config = base_config or PipelineConfig()
    target_levels = DEFAULT_TARGET_LEVELS if target_levels is None else target_levels

    results = []
    for target in target_levels:
        config = _config_for_target(base_config, target)
        out_dir = os.path.join(output_root, f"target_{int(round(target))}pct")

        start = time.time()
        report = run_pipeline(video_path, out_dir, config, show_progress=show_progress)
        elapsed = time.time() - start

        summary = report["summary"]
        frame_paths = sorted(
            os.path.join(out_dir, "frames", f) for f in os.listdir(os.path.join(out_dir, "frames"))
        )
        connectivity = compute_connectivity_metrics(
            frame_paths, use_sift=config.use_sift, max_features=config.max_features,
            descriptor_match_ratio=config.descriptor_match_ratio,
            orb_max_hamming_distance=config.orb_max_hamming_distance,
            weak_link_threshold=weak_link_threshold,
        )
        dataset_size = sum(os.path.getsize(p) for p in frame_paths)

        results.append(ReductionLevelResult(
            target_reduction_percent=target,
            achieved_reduction_percent=summary["frame_reduction_percent"],
            selected_frame_count=summary["selected_frame_count"],
            original_frame_count=summary["original_frame_count"],
            output_dir=out_dir,
            dataset_size_bytes=dataset_size,
            processing_time_sec=elapsed,
            config_used=config.to_dict(),
            connectivity=connectivity,
            report=report,
        ))
    return results
