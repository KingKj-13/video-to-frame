"""Phase 2.5: analysis_scale lets ORB run on a downscaled copy for speed.
Default (None) must be a strict no-op -- these tests lock in that the
default pipeline behaves exactly as before, that an enabled scale doesn't
break the pipeline, and that config validation rejects nonsense values.
"""

import pytest

from video_to_frame.config import PipelineConfig
from video_to_frame.pipeline import run_pipeline


def test_analysis_scale_defaults_to_disabled():
    assert PipelineConfig().analysis_scale is None


def test_invalid_analysis_scale_is_rejected():
    with pytest.raises(ValueError):
        PipelineConfig(analysis_scale=0.0).validate()
    with pytest.raises(ValueError):
        PipelineConfig(analysis_scale=1.5).validate()


def test_default_disabled_scale_matches_unset_scale(tmp_path, synthetic_video):
    """analysis_scale=None and analysis_scale=1.0 must both mean 'disabled'
    and produce identical selection to each other."""
    video_path, _ = synthetic_video

    report_none = run_pipeline(video_path, str(tmp_path / "none"), PipelineConfig(analysis_scale=None), show_progress=False)
    report_one = run_pipeline(video_path, str(tmp_path / "one"), PipelineConfig(analysis_scale=1.0), show_progress=False)

    kept_none = {f["index"] for f in report_none["frames"] if f["kept"]}
    kept_one = {f["index"] for f in report_one["frames"] if f["kept"]}
    assert kept_none == kept_one


def test_enabled_analysis_scale_runs_without_error(tmp_path, synthetic_video):
    video_path, total_frames = synthetic_video
    config = PipelineConfig(analysis_scale=0.75)
    report = run_pipeline(video_path, str(tmp_path / "scaled"), config, show_progress=False)

    summary = report["summary"]
    assert summary["original_frame_count"] == total_frames
    assert 0 < summary["selected_frame_count"] < total_frames
