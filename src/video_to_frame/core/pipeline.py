from __future__ import annotations

import json
import os
import shutil
from typing import List, Tuple

import numpy as np

from ..config import PipelineConfig
from ..coverage import CoveragePool
from ..features import FeatureExtractor
from ..io.extractor import FrameExtractor
from ..io.video_writer import write_preview_video
from ..output import encode_and_save, output_extension, resize_frame
from ..quality import compute_quality_metrics
from ..reporting.html_report import render_html_report
from ..reporting.report import build_report
from ..scoring import FrameScorer
from ..selection.selector import StreamingSelector
from ..selection.uncertainty import flag_uncertain_frames
from .models import FrameRecord, RejectionReason

KeptPair = Tuple[FrameRecord, np.ndarray]


def _enforce_max_frames(kept_buffer: List[KeptPair], config: PipelineConfig) -> List[KeptPair]:
    if config.max_frames is None or len(kept_buffer) <= config.max_frames:
        return kept_buffer

    by_score = sorted(kept_buffer, key=lambda pair: pair[0].metrics.score)
    to_drop = len(kept_buffer) - config.max_frames
    dropped_ids = set()
    for record, _frame in by_score[:to_drop]:
        record.kept = False
        record.rejection_reason = RejectionReason.OVER_CAPACITY
        dropped_ids.add(id(record))

    return [(r, f) for r, f in kept_buffer if id(r) not in dropped_ids]


def run_pipeline(input_path: str, output_dir: str, config: PipelineConfig, show_progress: bool = True) -> dict:
    config.validate()
    # Phase 3.6 (failure recovery): validate the input can actually be opened
    # BEFORE touching the output directory -- an invalid/missing/corrupt
    # video used to still leave behind an empty output_dir/frames/, which
    # could look like a partially-succeeded run to anyone just listing the
    # directory rather than checking the exception.
    extractor = FrameExtractor(input_path)
    video_info = extractor.video_info

    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames")
    # A stale frames/ directory from a previous run (especially in a
    # different output_format) must not linger alongside this run's
    # frames -- report.json only ever lists this run's paths, so leftover
    # files would silently misrepresent the dataset to anyone who lists the
    # directory directly instead of trusting the report.
    if os.path.isdir(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)

    feature_extractor = FeatureExtractor(max_features=config.max_features, use_sift=config.use_sift)
    coverage_pool = CoveragePool(
        use_sift=config.use_sift,
        max_per_frame=config.coverage_pool_max_per_frame,
        max_pool_size=config.coverage_pool_max_size,
        ratio=config.descriptor_match_ratio,
        max_hamming_distance=config.orb_max_hamming_distance,
    )
    scorer = FrameScorer(config.weights, config.exposure_low, config.exposure_high, config.exposure_min_contrast)

    frame_records: List[FrameRecord] = []
    kept_buffer: List[KeptPair] = []
    counts = {"kept": 0, "rejected": 0}

    def on_kept(record: FrameRecord, frame_bgr: np.ndarray) -> None:
        frame_records.append(record)
        kept_buffer.append((record, frame_bgr))
        counts["kept"] += 1

    def on_rejected(record: FrameRecord) -> None:
        frame_records.append(record)
        counts["rejected"] += 1

    selector = StreamingSelector(config, coverage_pool, feature_extractor, scorer, on_kept, on_rejected)

    iterator = extractor.frames()
    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, total=video_info.frame_count or None, desc="Analyzing frames", unit="frame")

    processed_count = 0
    for frame_bgr, index, timestamp_ms in iterator:
        processed_count += 1
        quality = compute_quality_metrics(frame_bgr)
        if quality.sharpness < config.hard_blur_floor:
            record = FrameRecord(index=index, timestamp_ms=timestamp_ms, metrics=None,
                                  kept=False, rejection_reason=RejectionReason.BLUR)
            on_rejected(record)
        else:
            selector.process(frame_bgr, index, timestamp_ms, quality)

        if show_progress:
            # Kept/rejected lag slightly behind processed_count -- a redundancy
            # cluster only resolves (and fires on_kept/on_rejected for its
            # members) when it closes, not frame-by-frame. Elapsed/ETA/FPS
            # are already in tqdm's default bar; this postfix adds the
            # selection-specific counts the spec asks for in one display.
            reduction_pct = 100.0 * (1 - counts["kept"] / processed_count) if processed_count else 0.0
            iterator.set_postfix(kept=counts["kept"], rejected=counts["rejected"],
                                  reduction=f"{reduction_pct:.1f}%", refresh=False)

    selector.finalize()

    kept_buffer.sort(key=lambda pair: pair[0].index)
    kept_buffer = _enforce_max_frames(kept_buffer, config)
    flag_uncertain_frames(kept_buffer, config)

    ext = output_extension(config.output_format)
    resize_target = (
        (config.resize_width, config.resize_height)
        if config.resize_width and config.resize_height else None
    )

    frame_paths = []
    for record, frame_bgr in kept_buffer:
        resized = resize_frame(frame_bgr, config.resize_long_edge, resize_target)
        filename = f"frame_{record.index:06d}.{ext}"
        path = os.path.join(frames_dir, filename)
        encode_and_save(resized, path, fmt=config.output_format, quality=config.output_quality,
                         png_compression_level=config.png_compression_level)
        record.output_filename = os.path.join("frames", filename).replace("\\", "/")
        frame_paths.append(path)

    preview_path = os.path.join(output_dir, "optimized_preview.mp4")
    if frame_paths:
        write_preview_video(frame_paths, preview_path, config.preview_fps)

    report = build_report(video_info, frame_records, config, frame_paths, processed_count)

    with open(os.path.join(output_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Phase 4.5: the effective config is already embedded in report.json
    # (report["config"]), but a standalone config.json lets a user or script
    # diff/reuse a run's exact settings without parsing the full report.
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)

    render_html_report(report, video_info, os.path.join(output_dir, "report.html"), preview_exists=bool(frame_paths))

    return report
