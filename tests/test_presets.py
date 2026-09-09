"""Phase 4.10: presets map to concrete config values; explicit CLI/config
overrides must still win over a preset."""

import os

import pytest

from video_to_frame.config import PipelineConfig
from video_to_frame.pipeline import run_pipeline
from video_to_frame.presets import PRESETS, apply_preset


def test_balanced_preset_matches_default_config():
    default = PipelineConfig()
    preset = apply_preset("balanced")
    assert preset.redundancy_threshold == default.redundancy_threshold
    assert preset.coverage_gain_threshold == default.coverage_gain_threshold
    assert preset.min_overlap_threshold == default.min_overlap_threshold


def test_reconstruction_safe_is_more_conservative_than_aggressive():
    safe = apply_preset("reconstruction_safe")
    aggressive = apply_preset("aggressive")
    # Higher redundancy_threshold = stricter similarity required to drop a
    # frame = more retained = more conservative.
    assert safe.redundancy_threshold > aggressive.redundancy_threshold
    assert safe.min_overlap_threshold >= aggressive.min_overlap_threshold


def test_aggressive_preset_does_not_weaken_safety_floor_below_default():
    default = PipelineConfig()
    aggressive = apply_preset("aggressive")
    assert aggressive.min_overlap_threshold >= default.min_overlap_threshold


def test_unknown_preset_raises():
    with pytest.raises(ValueError):
        apply_preset("not_a_real_preset")


def test_every_preset_produces_a_valid_config():
    for name in PRESETS:
        apply_preset(name).validate()  # must not raise


def test_pipeline_writes_config_json(tmp_path, synthetic_video):
    video_path, _ = synthetic_video
    out_dir = str(tmp_path / "out")
    run_pipeline(video_path, out_dir, PipelineConfig(), show_progress=False)

    config_path = os.path.join(out_dir, "config.json")
    assert os.path.isfile(config_path)

    import json
    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["output_format"] == "png"
