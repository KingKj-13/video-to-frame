"""Phase 2: uncertainty flagging is a post-selection, reporting-only pass.
It must never change what was kept or rejected -- these tests verify the
flags fire (and don't fire) correctly, using the same synthetic panning
scene as the selector tests for consistent, real ORB keypoints.
"""

import numpy as np

from video_to_frame.config import PipelineConfig
from video_to_frame.models import FrameMetrics, FrameRecord, RejectionReason
from video_to_frame.selection.uncertainty import LOW_TEXTURE_KEYPOINT_FLOOR, flag_uncertain_frames

from synthetic import tile as _tile


def _record(index: int) -> FrameRecord:
    metrics = FrameMetrics(sharpness=500.0, brightness=120.0, contrast=50.0,
                            quality_norm=0.8, feature_richness_norm=0.8,
                            redundancy_norm=0.1, viewpoint_change_norm=0.9,
                            coverage_gain_norm=0.5, score=5.0)
    return FrameRecord(index=index, timestamp_ms=index * 33.0, metrics=metrics,
                        kept=True, rejection_reason=RejectionReason.NONE)


def test_well_connected_frames_are_not_flagged_uncertain():
    config = PipelineConfig()
    kept = [(_record(0), _tile(0)), (_record(1), _tile(2)), (_record(2), _tile(4))]
    flag_uncertain_frames(kept, config)
    assert all(not r.uncertain for r, _f in kept)
    assert all(r.uncertainty_reasons == [] for r, _f in kept)


def test_large_gap_after_previous_kept_frame_is_flagged_weak_connectivity():
    config = PipelineConfig()
    # offset 0 and offset 700 share no scene content on the wide synthetic canvas.
    kept = [(_record(0), _tile(0)), (_record(1), _tile(700))]
    flag_uncertain_frames(kept, config)

    first, second = kept[0][0], kept[1][0]
    assert not first.uncertain  # nothing precedes the first kept frame
    assert second.uncertain
    assert any("weak_connectivity_to_previous_kept" in r for r in second.uncertainty_reasons)


def test_low_keypoint_count_frame_is_flagged_low_texture():
    config = PipelineConfig()
    flat_frame = np.full((200, 200, 3), 60, dtype=np.uint8)  # featureless -> ~0 ORB keypoints
    kept = [(_record(0), flat_frame)]
    flag_uncertain_frames(kept, config)

    record = kept[0][0]
    assert record.uncertain
    assert any("low_texture" in r for r in record.uncertainty_reasons)


def test_poor_exposure_kept_frame_is_flagged_exposure_anomaly():
    config = PipelineConfig()
    metrics = FrameMetrics(sharpness=500.0, brightness=5.0, contrast=50.0,
                            quality_norm=0.3, feature_richness_norm=0.8,
                            redundancy_norm=0.1, viewpoint_change_norm=0.9,
                            coverage_gain_norm=0.5, score=3.0)
    record = FrameRecord(index=0, timestamp_ms=0.0, metrics=metrics,
                          kept=True, rejection_reason=RejectionReason.NONE)
    kept = [(record, _tile(0))]
    flag_uncertain_frames(kept, config)

    assert record.uncertain
    assert any("exposure_anomaly" in r for r in record.uncertainty_reasons)


def test_well_exposed_frame_is_not_flagged_exposure_anomaly():
    config = PipelineConfig()
    kept = [(_record(0), _tile(0))]  # brightness=120, contrast=50 -- within default range
    flag_uncertain_frames(kept, config)
    assert not any("exposure_anomaly" in r for r in kept[0][0].uncertainty_reasons)


def test_empty_and_single_frame_inputs_do_not_error():
    config = PipelineConfig()
    flag_uncertain_frames([], config)  # must not raise

    kept = [(_record(0), _tile(0))]
    flag_uncertain_frames(kept, config)
    assert not kept[0][0].uncertain  # no previous frame to compare against, and has real texture
