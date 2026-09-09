from __future__ import annotations

import os
from typing import List

from ..config import PipelineConfig
from ..core.models import FrameRecord, KeptReason, RejectionReason, record_to_dict
from ..io.extractor import VideoInfo


def _resolution_label(config: PipelineConfig, video_info: VideoInfo) -> str:
    if config.resize_width and config.resize_height:
        return f"{config.resize_width}x{config.resize_height}"
    if config.resize_long_edge:
        return f"long-edge {config.resize_long_edge}px"
    return f"{video_info.width}x{video_info.height} (original)"


def build_report(video_info: VideoInfo, frame_records: List[FrameRecord], config: PipelineConfig,
                  kept_frame_paths: List[str], processed_count: int) -> dict:
    kept = [r for r in frame_records if r.kept]
    blur_rejected = [r for r in frame_records if r.rejection_reason == RejectionReason.BLUR]
    redundant_rejected = [r for r in frame_records if r.rejection_reason == RejectionReason.REDUNDANT]
    capacity_rejected = [r for r in frame_records if r.rejection_reason == RejectionReason.OVER_CAPACITY]
    forced_by_min_overlap = [r for r in kept if r.kept_reason == KeptReason.MINIMUM_OVERLAP_PROTECTION]
    uncertain = [r for r in kept if r.uncertain]

    original_count = processed_count
    selected_count = len(kept)
    reduction_pct = 100.0 * (1 - selected_count / original_count) if original_count else 0.0

    output_frames_size = sum(os.path.getsize(p) for p in kept_frame_paths)
    compression_ratio = (video_info.size_bytes / output_frames_size) if output_frames_size else 0.0

    avg_quality = (
        sum(r.metrics.quality_norm for r in kept if r.metrics) / selected_count
        if selected_count else 0.0
    )

    warnings = []
    if original_count > 0 and selected_count == 0:
        warnings.append(
            "No frames were selected -- every frame in the source video was rejected "
            f"(blur: {len(blur_rejected)}, redundant: {len(redundant_rejected)}). No preview "
            "video was generated. Check hard_blur_floor and the source footage quality."
        )
    if uncertain:
        warnings.append(
            f"{len(uncertain)} kept frame(s) are flagged uncertain (weak connectivity to the "
            "previous kept frame, and/or very low local texture) -- see each frame's "
            "'uncertainty_reasons' in the frames array. These frames were not rejected or "
            "altered; this is a visibility flag for manual review, not an automatic action."
        )
    if config.min_frames is not None and selected_count < config.min_frames:
        warnings.append(
            f"Selected frame count ({selected_count}) is below configured min_frames "
            f"({config.min_frames}); the source video did not contain enough distinct "
            "or usable viewpoints to meet it -- frames cannot be invented."
        )

    summary = {
        "original_frame_count": original_count,
        "selected_frame_count": selected_count,
        "frame_reduction_percent": round(reduction_pct, 2),
        "original_video_size_bytes": video_info.size_bytes,
        "output_frames_size_bytes": output_frames_size,
        "compression_ratio": round(compression_ratio, 3),
        "average_frame_quality": round(avg_quality, 4),
        "frames_rejected_blur": len(blur_rejected),
        "frames_rejected_redundant": len(redundant_rejected),
        "frames_rejected_over_capacity": len(capacity_rejected),
        "minimum_overlap_threshold": config.min_overlap_threshold,
        "frames_forced_by_minimum_overlap": len(forced_by_min_overlap),
        "uncertain_kept_frames": len(uncertain),
        "output_resolution": _resolution_label(config, video_info),
        "output_format": config.output_format,
        "source_video_path": video_info.path,
        "source_fps": video_info.fps,
        "source_resolution": f"{video_info.width}x{video_info.height}",
    }

    frames_detail = [record_to_dict(r) for r in sorted(frame_records, key=lambda r: r.index)]

    return {
        "summary": summary,
        "warnings": warnings,
        "config": config.to_dict(),
        "frames": frames_detail,
    }
