import json
import os

from video_to_frame.config import PipelineConfig
from video_to_frame.pipeline import run_pipeline


def test_end_to_end_smoke(tmp_path, synthetic_video):
    video_path, total_frames = synthetic_video
    out_dir = str(tmp_path / "out")

    config = PipelineConfig()
    report = run_pipeline(video_path, out_dir, config, show_progress=False)

    assert os.path.isdir(os.path.join(out_dir, "frames"))
    assert os.path.isfile(os.path.join(out_dir, "report.json"))
    assert os.path.isfile(os.path.join(out_dir, "report.html"))

    summary = report["summary"]
    assert summary["original_frame_count"] == total_frames
    assert 0 < summary["selected_frame_count"] < total_frames
    assert summary["frames_rejected_blur"] > 0
    assert summary["frames_rejected_redundant"] > 0

    frame_files = os.listdir(os.path.join(out_dir, "frames"))
    assert len(frame_files) == summary["selected_frame_count"]

    with open(os.path.join(out_dir, "report.json"), encoding="utf-8") as f:
        data = json.load(f)
    assert data["summary"]["selected_frame_count"] == summary["selected_frame_count"]
    assert len(data["frames"]) == total_frames
