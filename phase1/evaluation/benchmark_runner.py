"""Ties the Phase 1 pieces together for one video: runs the intelligent
selector plus the two naive baselines, measures feature connectivity on each
dataset, optionally runs an external reconstruction backend on each, and
classifies the result. See phase1/reports/evaluation_report.py for the
JSON/HTML this produces.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from video_to_frame.config import PipelineConfig
from video_to_frame.pipeline import run_pipeline

from .baselines import (
    BaselineDataset,
    blur_filter_only_baseline,
    every_nth_frame_baseline,
    full_dataset_connectivity,
)
from .connectivity_metrics import ConnectivityMetrics, compute_connectivity_metrics
from .failure_analysis import FailureCase, classify
from .reconstruction_backend import NullBackend, ReconstructionBackend, ReconstructionMetrics
from .weak_link_diagnostics import WeakLinkDiagnosis, diagnose_weak_links


@dataclass
class DatasetEvaluation:
    name: str
    frame_paths: List[str]
    original_frame_count: int
    selected_frame_count: int
    reduction_percent: float
    dataset_size_bytes: int
    connectivity: ConnectivityMetrics
    reconstruction: Optional[ReconstructionMetrics]
    failure_case: FailureCase
    extra: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "original_frame_count": self.original_frame_count,
            "selected_frame_count": self.selected_frame_count,
            "reduction_percent": round(self.reduction_percent, 2),
            "dataset_size_bytes": self.dataset_size_bytes,
            "connectivity": self.connectivity.to_dict(),
            "failure_analysis": self.failure_case.to_dict(),
        }
        if self.reconstruction is not None:
            d["reconstruction"] = self.reconstruction.to_dict()
        d.update(self.extra)
        return d


def _dataset_size(frame_paths: List[str]) -> int:
    return sum(os.path.getsize(p) for p in frame_paths if os.path.isfile(p))


def _evaluate_dataset(name: str, frame_paths: List[str], original_frame_count: int,
                       config: PipelineConfig, backend: ReconstructionBackend,
                       workdir: str, weak_link_threshold: int,
                       extra: Optional[Dict] = None) -> DatasetEvaluation:
    frame_paths_sorted = sorted(frame_paths)  # filenames are zero-padded by original frame index
    connectivity = compute_connectivity_metrics(
        frame_paths_sorted,
        use_sift=config.use_sift,
        max_features=config.max_features,
        descriptor_match_ratio=config.descriptor_match_ratio,
        orb_max_hamming_distance=config.orb_max_hamming_distance,
        weak_link_threshold=weak_link_threshold,
    )

    reconstruction = None
    if not isinstance(backend, NullBackend):
        image_dir = os.path.dirname(frame_paths_sorted[0]) if frame_paths_sorted else workdir
        reconstruction = backend.run(image_dir, os.path.join(workdir, "reconstruction"))

    failure_case = classify(connectivity, reconstruction)
    selected_count = len(frame_paths)
    reduction = 100.0 * (1 - selected_count / original_frame_count) if original_frame_count else 0.0

    return DatasetEvaluation(
        name=name, frame_paths=frame_paths_sorted, original_frame_count=original_frame_count,
        selected_frame_count=selected_count, reduction_percent=reduction,
        dataset_size_bytes=_dataset_size(frame_paths_sorted),
        connectivity=connectivity, reconstruction=reconstruction, failure_case=failure_case,
        extra=extra or {},
    )


def _evaluate_full_dataset(video_path: str, config: PipelineConfig, weak_link_threshold: int) -> DatasetEvaluation:
    """Baseline A: 'all usable frames' -- computed as a streaming, files-free
    connectivity pass (see full_dataset_connectivity's docstring for why no
    images are written). Serves as the reference ceiling other datasets are
    compared against; reconstruction is not run on it since there is no
    image directory (use blur_filter_only, which is the same frame set
    written to disk, for a reconstruction-backend comparison against 'all
    usable frames')."""
    connectivity = full_dataset_connectivity(
        video_path, hard_blur_floor=config.hard_blur_floor,
        use_sift=config.use_sift, max_features=config.max_features,
        descriptor_match_ratio=config.descriptor_match_ratio,
        orb_max_hamming_distance=config.orb_max_hamming_distance,
        weak_link_threshold=weak_link_threshold,
    )
    failure_case = classify(connectivity, None)
    return DatasetEvaluation(
        name="full_dataset", frame_paths=[], original_frame_count=connectivity.frame_count,
        selected_frame_count=connectivity.frame_count, reduction_percent=0.0,
        dataset_size_bytes=os.path.getsize(video_path),
        connectivity=connectivity, reconstruction=None, failure_case=failure_case,
        extra={"note": "Streaming, files-free reference (all frames above the hard blur floor); "
                        "'dataset_size_bytes' is the source VIDEO's size, not a decoded-image dataset."},
    )


def run_benchmark(
    video_path: str,
    output_dir: str,
    config: Optional[PipelineConfig] = None,
    backend: Optional[ReconstructionBackend] = None,
    weak_link_threshold: int = 15,
    include_baselines: bool = True,
) -> dict:
    """Runs the intelligent selector and (optionally) both naive baselines
    on `video_path`, matching the baselines' frame budget to the intelligent
    selector's actual output count so the comparison is apples-to-apples at
    the same data budget (Phase 1 spec section 11).
    """
    config = config or PipelineConfig()
    backend = backend or NullBackend()
    os.makedirs(output_dir, exist_ok=True)

    start = time.time()
    intelligent_dir = os.path.join(output_dir, "intelligent")
    intelligent_report = run_pipeline(video_path, intelligent_dir, config, show_progress=False)
    intelligent_summary = intelligent_report["summary"]
    intelligent_frames = sorted(
        os.path.join(intelligent_dir, "frames", f) for f in os.listdir(os.path.join(intelligent_dir, "frames"))
    )
    original_frame_count = intelligent_summary["original_frame_count"]
    target_count = max(1, intelligent_summary["selected_frame_count"])

    datasets: Dict[str, DatasetEvaluation] = {}
    datasets["intelligent_selector"] = _evaluate_dataset(
        "intelligent_selector", intelligent_frames, original_frame_count,
        config, backend, os.path.join(output_dir, "intelligent"), weak_link_threshold,
        extra={
            "blur_rejected": intelligent_summary["frames_rejected_blur"],
            "redundant_rejected": intelligent_summary["frames_rejected_redundant"],
            "frames_forced_by_minimum_overlap": intelligent_summary["frames_forced_by_minimum_overlap"],
        },
    )

    weak_links = datasets["intelligent_selector"].connectivity.weak_link_pairs
    weak_link_diagnoses: List[WeakLinkDiagnosis] = (
        diagnose_weak_links(weak_links, intelligent_report["frames"]) if weak_links else []
    )

    if include_baselines:
        datasets["full_dataset"] = _evaluate_full_dataset(video_path, config, weak_link_threshold)

        every_nth_dir = os.path.join(output_dir, "baseline_every_nth", "frames")
        every_nth: BaselineDataset = every_nth_frame_baseline(
            video_path, every_nth_dir, target_count=target_count, output_format=config.output_format,
        )
        datasets["every_nth_frame"] = _evaluate_dataset(
            "every_nth_frame", every_nth.frame_paths, every_nth.original_frame_count,
            config, backend, os.path.join(output_dir, "baseline_every_nth"), weak_link_threshold,
        )

        blur_dir = os.path.join(output_dir, "baseline_blur_only", "frames")
        blur_only: BaselineDataset = blur_filter_only_baseline(
            video_path, blur_dir, hard_blur_floor=config.hard_blur_floor, output_format=config.output_format,
        )
        datasets["blur_filter_only"] = _evaluate_dataset(
            "blur_filter_only", blur_only.frame_paths, blur_only.original_frame_count,
            config, backend, os.path.join(output_dir, "baseline_blur_only"), weak_link_threshold,
        )

    elapsed = time.time() - start

    return {
        "video": video_path,
        "original_frames": original_frame_count,
        "processing_time_sec": round(elapsed, 2),
        "reconstruction_backend": backend.name,
        "datasets": {k: v.to_dict() for k, v in datasets.items()},
        "weak_link_diagnostics": [d.to_dict() for d in weak_link_diagnoses],
    }
