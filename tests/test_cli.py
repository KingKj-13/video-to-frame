"""Phase 4.3: CLI subcommands, with 'process' as the transparent backward-
compatible default (no subcommand named = process). Exercises real argv
parsing end-to-end, not just the underlying pipeline functions.
"""

import json
import os

from video_to_frame.cli import main


def test_bare_invocation_is_backward_compatible_with_process(tmp_path, synthetic_video):
    """`video-to-frame INPUT --output OUT` (no subcommand) must behave
    exactly like `video-to-frame process INPUT --output OUT`."""
    video_path, _ = synthetic_video

    out_bare = str(tmp_path / "bare")
    out_explicit = str(tmp_path / "explicit")

    code_bare = main([video_path, "--output", out_bare, "--no-progress", "--quiet"])
    code_explicit = main(["process", video_path, "--output", out_explicit, "--no-progress", "--quiet"])

    assert code_bare == 0
    assert code_explicit == 0

    with open(os.path.join(out_bare, "report.json"), encoding="utf-8") as f:
        report_bare = json.load(f)
    with open(os.path.join(out_explicit, "report.json"), encoding="utf-8") as f:
        report_explicit = json.load(f)

    assert report_bare["summary"]["selected_frame_count"] == report_explicit["summary"]["selected_frame_count"]


def test_preset_flag_is_applied(tmp_path, synthetic_video):
    video_path, _ = synthetic_video
    out_dir = str(tmp_path / "out")

    code = main(["process", video_path, "--output", out_dir, "--preset", "quality_first",
                 "--no-progress", "--quiet"])
    assert code == 0

    with open(os.path.join(out_dir, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    assert config["redundancy_threshold"] == 0.97  # quality_first preset value


def test_batch_subcommand_does_not_choke_on_non_config_args(tmp_path, synthetic_video):
    """Regression: _build_config used to crash with 'Unknown config
    override: input_dir' because batch's positional argument name wasn't in
    an exclusion blocklist -- now inclusion-based against PipelineConfig
    fields, so this can't recur for any future CLI-only argument either."""
    video_path, _ = synthetic_video
    input_dir = tmp_path / "videos"
    input_dir.mkdir()
    import shutil
    shutil.copy(video_path, input_dir / "a.mp4")

    out_root = str(tmp_path / "batch_out")
    code = main(["batch", str(input_dir), "--output", out_root, "--no-progress", "--quiet"])
    assert code == 0

    with open(os.path.join(out_root, "batch_summary.json"), encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["succeeded"] == 1
    assert summary["failed"] == 0


def test_inspect_subcommand_reads_existing_report(tmp_path, synthetic_video, capsys):
    video_path, _ = synthetic_video
    out_dir = str(tmp_path / "out")
    main(["process", video_path, "--output", out_dir, "--no-progress", "--quiet"])

    code = main(["inspect", out_dir])
    assert code == 0
    captured = capsys.readouterr()
    assert "selected_frame_count" in captured.out


def test_inspect_subcommand_on_missing_report_fails_cleanly(tmp_path):
    code = main(["inspect", str(tmp_path)])
    assert code == 1
