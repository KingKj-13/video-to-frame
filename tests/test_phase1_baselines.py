"""Regression test for a real bug found via a Phase 4.16 comparison run on
real footage: every_nth_frame_baseline used to compute step = round(total /
target_count), which collapses to 1 (i.e. "keep every frame", 0% reduction)
whenever target_count exceeds half of total. That silently produced a
no-reduction "baseline" for any video where the intelligent selector's own
reduction was under 50% -- exactly the case on a real drone-footage run
(target_count=1642 of total=2011). Fixed to use evenly-spaced target indices
instead of a modulo step.
"""

from phase1.evaluation.baselines import every_nth_frame_baseline


def test_baseline_still_reduces_when_target_exceeds_half_of_total(tmp_path, synthetic_video):
    """The exact bug condition: target_count > total / 2."""
    video_path, total_frames = synthetic_video
    target_count = int(total_frames * 0.7)  # > half of total -- the trigger case

    dataset = every_nth_frame_baseline(video_path, str(tmp_path / "baseline_out"), target_count)

    assert dataset.selected_frame_count < total_frames, (
        "baseline produced no reduction at all -- the step-rounding bug regressed"
    )
    # Evenly-spaced selection should land close to the requested budget.
    assert abs(dataset.selected_frame_count - target_count) <= 1


def test_baseline_matches_target_count_when_well_below_half_of_total(tmp_path, synthetic_video):
    video_path, total_frames = synthetic_video
    target_count = int(total_frames * 0.2)

    dataset = every_nth_frame_baseline(video_path, str(tmp_path / "baseline_out"), target_count)

    assert abs(dataset.selected_frame_count - target_count) <= 1


def test_baseline_keeps_everything_when_target_count_meets_or_exceeds_total(tmp_path, synthetic_video):
    video_path, total_frames = synthetic_video

    dataset = every_nth_frame_baseline(video_path, str(tmp_path / "baseline_out"), total_frames * 2)

    assert dataset.selected_frame_count == total_frames
