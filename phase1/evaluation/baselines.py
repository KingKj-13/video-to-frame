"""Simple baseline selectors, for comparison against the intelligent
selector (Phase 1 spec section 11). Neither baseline uses redundancy,
coverage, or minimum-overlap logic -- they exist purely as a reference point
to demonstrate whether the intelligent selector actually preserves more
reconstruction-relevant information than naive sampling at the same frame
budget.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import cv2

from video_to_frame.analysis.features import FeatureExtractor
from video_to_frame.analysis.quality import compute_quality_metrics
from video_to_frame.compression.compressor import encode_and_save, output_extension
from video_to_frame.io.extractor import FrameExtractor

from .connectivity_metrics import ConnectivityMetrics, _pairwise_connectivity


@dataclass
class BaselineDataset:
    name: str
    frame_paths: List[str]
    frame_indices: List[int]
    original_frame_count: int

    @property
    def selected_frame_count(self) -> int:
        return len(self.frame_paths)

    @property
    def reduction_percent(self) -> float:
        if not self.original_frame_count:
            return 0.0
        return 100.0 * (1 - self.selected_frame_count / self.original_frame_count)


def every_nth_frame_baseline(video_path: str, output_dir: str, target_count: int,
                              output_format: str = "png") -> BaselineDataset:
    """Baseline 1: uniform sampling to approximately `target_count` frames.
    This is exactly the naive strategy the Phase 0 spec called out as
    insufficient -- kept here specifically so it can be measured, not used.

    Uses evenly-spaced target indices across the video rather than a modulo
    step (`index % step == 0`): a modulo step of `round(total / target_count)`
    collapses to 1 -- i.e. "keep every frame", 0% reduction -- whenever
    target_count exceeds half of total, which silently produced a
    no-reduction "baseline" for any video where the intelligent selector's
    own reduction was under 50% (found via a Phase 4.16 real-video
    comparison run on a video with 18.35% selector reduction).
    """
    extractor = FrameExtractor(video_path)
    total = extractor.video_info.frame_count or 1
    target_count = max(1, min(target_count, total))
    if target_count >= total:
        target_indices = set(range(total))
    elif target_count == 1:
        target_indices = {0}
    else:
        target_indices = {round(i * (total - 1) / (target_count - 1)) for i in range(target_count)}

    os.makedirs(output_dir, exist_ok=True)
    ext = output_extension(output_format)

    paths: List[str] = []
    indices: List[int] = []
    processed = 0
    for frame_bgr, index, _timestamp_ms in extractor.frames():
        processed += 1
        if index in target_indices:
            path = os.path.join(output_dir, f"frame_{index:06d}.{ext}")
            encode_and_save(frame_bgr, path, fmt=output_format, quality=90)
            paths.append(path)
            indices.append(index)

    return BaselineDataset(name="every_nth_frame", frame_paths=paths, frame_indices=indices,
                            original_frame_count=processed)


def blur_filter_only_baseline(video_path: str, output_dir: str, hard_blur_floor: float,
                               output_format: str = "png") -> BaselineDataset:
    """Baseline 2: keep every frame at or above the blur floor -- no
    redundancy, coverage, or minimum-overlap logic at all."""
    extractor = FrameExtractor(video_path)
    os.makedirs(output_dir, exist_ok=True)
    ext = output_extension(output_format)

    paths: List[str] = []
    indices: List[int] = []
    processed = 0
    for frame_bgr, index, _timestamp_ms in extractor.frames():
        processed += 1
        quality = compute_quality_metrics(frame_bgr)
        if quality.sharpness >= hard_blur_floor:
            path = os.path.join(output_dir, f"frame_{index:06d}.{ext}")
            encode_and_save(frame_bgr, path, fmt=output_format, quality=90)
            paths.append(path)
            indices.append(index)

    return BaselineDataset(name="blur_filter_only", frame_paths=paths, frame_indices=indices,
                            original_frame_count=processed)


def full_dataset_connectivity(
    video_path: str,
    hard_blur_floor: float,
    use_sift: bool = False,
    max_features: int = 500,
    descriptor_match_ratio: float = 0.75,
    orb_max_hamming_distance: int = 64,
    weak_link_threshold: int = 15,
) -> ConnectivityMetrics:
    """Baseline A: 'all usable frames' -- every frame at or above the hard
    blur floor, i.e. everything that isn't outright unusable, with no other
    filtering. This is the reference ceiling the other datasets are compared
    against.

    Computed as a single streaming pass with descriptors kept only in
    memory (never written to disk) -- for a real video, writing every usable
    frame as a full-resolution image purely to measure a reference
    connectivity number would cost far more time/disk than the measurement
    is worth, and the "original video size" figure already gives the size
    reference for baseline A.
    """
    extractor = FeatureExtractor(max_features=max_features, use_sift=use_sift)
    video_extractor = FrameExtractor(video_path)

    descriptors: List[Optional[object]] = []
    labels: List[str] = []
    for frame_bgr, index, _timestamp_ms in video_extractor.frames():
        quality = compute_quality_metrics(frame_bgr)
        if quality.sharpness < hard_blur_floor:
            continue
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        _keypoints, desc = extractor.extract(gray)
        descriptors.append(desc)
        labels.append(f"frame_{index:06d}")

    if len(descriptors) < 2:
        return ConnectivityMetrics(
            frame_count=len(descriptors), pair_count=0, average_matches=0.0,
            minimum_matches=0, weak_link_threshold=weak_link_threshold,
            weak_link_count=0, low_texture_pair_count=0,
        )

    return _pairwise_connectivity(
        descriptors, labels, use_sift, descriptor_match_ratio, orb_max_hamming_distance, weak_link_threshold,
    )
