"""Post-selection uncertainty detection (Phase 2).

Evidence-based, not speculative: Phase 1's weak-link diagnostics on the
low-texture synthetic stress video found that minimum-overlap protection
(Phase 0.1) has a structural blind spot -- it only ever checks a candidate a
cluster would otherwise *reject* against the previous kept frame. The
*winner* of a newly-started cluster is never checked, so when a long run of
correctly-rejected redundant and blurry frames is crossed, the next kept
frame can end up weakly connected to the previous one without anything
noticing (every individual rejection along the way was locally correct).

This module only detects and reports that condition -- it never changes
what was kept or rejected. Real-video evidence (handheld: 0 weak links, 3 of
4 healthy-texture synthetic benchmarks: 0 weak links) showed no evidence
that a more invasive fix (e.g. retroactively rescuing a rejected frame as a
bridge) is needed for real footage, so the conservative, zero-regression-risk
choice is visibility, not automatic intervention -- consistent with Phase
2's "prefer conservative retention, do not aggressively act on uncertainty"
principle, applied here as "do not automatically restructure selection on
uncertainty; surface it."

A third signal, exposure_anomaly, flags a kept frame whose brightness/
contrast falls outside the same exposure_low/exposure_high/
exposure_min_contrast thresholds the scorer already uses to penalize a
frame's quality score (see scoring/scorer.py's exposure_score). This adds no
new heuristic or threshold: a frame can still be a cluster's best-scoring
representative (or be force-kept by minimum-overlap protection) despite poor
exposure, and this makes that already-computed fact visible instead of only
silently depressing its score. Two other candidate signals considered during
Phase 3.7 review -- "inconsistent matching" and "unusual scene behavior" --
were not added: neither has a concrete, evidence-backed operationalization
(what counts as "inconsistent," measured how, calibrated against what data),
and Phase 2's rule against adding signals without measured justification
applies here too.

Cost: O(kept_frame_count), not O(video_length) -- this re-extracts ORB
descriptors only for the already-small kept set, after selection is
complete, so it stays bounded regardless of source video length.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from ..config import PipelineConfig
from ..core.models import FrameRecord
from ..features import FeatureExtractor, match_descriptors
from ..scoring.scorer import exposure_score

LOW_TEXTURE_KEYPOINT_FLOOR = 30


def flag_uncertain_frames(kept: List[Tuple[FrameRecord, np.ndarray]], config: PipelineConfig) -> None:
    """Mutates each kept FrameRecord's `uncertain`/`uncertainty_reasons`
    in place. `kept` must be sorted in original temporal order."""
    if not kept:
        return

    extractor = FeatureExtractor(max_features=config.max_features, use_sift=config.use_sift)
    descriptors = []
    for _record, frame_bgr in kept:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        _keypoints, desc = extractor.extract(gray)
        descriptors.append(desc)

    for i, (record, _frame_bgr) in enumerate(kept):
        reasons: List[str] = []

        keypoint_count = 0 if descriptors[i] is None else len(descriptors[i])
        if keypoint_count < LOW_TEXTURE_KEYPOINT_FLOOR:
            reasons.append(f"low_texture (keypoints={keypoint_count})")

        if i > 0:
            overlap = _overlap_ratio(descriptors[i], descriptors[i - 1], config)
            if overlap < config.min_overlap_threshold:
                reasons.append(f"weak_connectivity_to_previous_kept (overlap={overlap:.2f})")

        if record.metrics is not None:
            exp_score = exposure_score(
                record.metrics.brightness, record.metrics.contrast,
                config.exposure_low, config.exposure_high, config.exposure_min_contrast,
            )
            if exp_score < 1.0:
                reasons.append(
                    f"exposure_anomaly (brightness={record.metrics.brightness:.1f}, "
                    f"contrast={record.metrics.contrast:.1f})"
                )

        record.uncertain = bool(reasons)
        record.uncertainty_reasons = reasons


def _overlap_ratio(candidate_desc, reference_desc, config: PipelineConfig) -> float:
    total = 0 if candidate_desc is None else len(candidate_desc)
    if total == 0 or reference_desc is None or len(reference_desc) == 0:
        return 0.0
    matches = match_descriptors(
        candidate_desc, reference_desc, use_sift=config.use_sift,
        ratio=config.descriptor_match_ratio, max_hamming_distance=config.orb_max_hamming_distance,
    )
    return matches / total
