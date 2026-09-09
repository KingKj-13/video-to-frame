"""Phase 3.9: same input + same configuration must produce the same result.

Root cause of the one nondeterminism found: CoveragePool.add() created a
fresh, unseeded np.random.default_rng() on every call when subsampling a
frame's descriptors down to coverage_pool_max_per_frame -- fixed by using
one seeded generator per CoveragePool instance (video_to_frame.analysis.
coverage.CoveragePool.__init__).
"""

from video_to_frame.config import PipelineConfig
from video_to_frame.pipeline import run_pipeline


def test_same_input_and_config_produce_identical_selection(tmp_path, synthetic_video):
    video_path, _ = synthetic_video
    config = PipelineConfig()

    report_a = run_pipeline(video_path, str(tmp_path / "run_a"), config, show_progress=False)
    report_b = run_pipeline(video_path, str(tmp_path / "run_b"), config, show_progress=False)

    kept_a = [(f["index"], f["score"], f["rejection_reason"]) for f in report_a["frames"]]
    kept_b = [(f["index"], f["score"], f["rejection_reason"]) for f in report_b["frames"]]
    assert kept_a == kept_b

    assert report_a["summary"]["selected_frame_count"] == report_b["summary"]["selected_frame_count"]
