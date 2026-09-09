"""Phase 4.10: named presets mapping to concrete config values.

Presets only touch *selection aggressiveness* (redundancy/coverage/
minimum-overlap thresholds) -- never output format or resizing, which stay
a separate, orthogonal choice (matching the existing "compression is
evaluated independently of selection" design principle: a user shouldn't
have to give up PNG output just to pick a more conservative selection
strategy, or vice versa).

min_overlap_threshold is a SAFETY floor, not a quality dial -- "aggressive"
therefore leaves it at the default rather than weakening the connectivity
guarantee; only "reconstruction_safe"/"quality_first" raise it, since a more
conservative floor is consistent with their goal.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Optional

from .config import PipelineConfig

PRESETS: Dict[str, Dict[str, float]] = {
    "reconstruction_safe": {
        "redundancy_threshold": 0.90,
        "coverage_gain_threshold": 0.10,
        "min_overlap_threshold": 0.40,
    },
    "balanced": {
        # Matches PipelineConfig()'s own defaults, spelled out explicitly so
        # --preset balanced is self-documenting rather than "does nothing".
        "redundancy_threshold": 0.82,
        "coverage_gain_threshold": 0.15,
        "min_overlap_threshold": 0.30,
    },
    "aggressive": {
        "redundancy_threshold": 0.65,
        "coverage_gain_threshold": 0.20,
        "min_overlap_threshold": 0.30,  # safety floor left unchanged, not weakened
    },
    "quality_first": {
        "redundancy_threshold": 0.97,
        "coverage_gain_threshold": 0.05,
        "min_overlap_threshold": 0.45,
    },
}


def apply_preset(name: str, base: Optional[PipelineConfig] = None) -> PipelineConfig:
    if name not in PRESETS:
        raise ValueError(f"Unknown preset '{name}'. Available: {sorted(PRESETS)}")
    base = base or PipelineConfig()
    return dataclasses.replace(base, **PRESETS[name])
