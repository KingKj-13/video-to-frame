"""Tests for minimum-overlap protection (Phase 0.1).

Two safety constraints must stay independent:
  - redundancy_threshold: "are these frames too similar to keep both?"
  - min_overlap_threshold: "if we drop this one anyway, would that leave
    the previous kept frame and the next one too disconnected for a
    downstream feature matcher to link?"

Tests A/B exercise _flush_cluster/_overlap_ratio directly with hand-built
descriptor arrays -- this gives exact, deterministic control over ORB
feature-overlap ratios (0.0 / 0.5 / 1.0) without depending on incidental
properties of a rendered synthetic image. Test C exercises the full
blur-floor-then-selector gate (mirroring pipeline.py's real per-frame
decision) to confirm the hard blur floor still wins over this new safety net.
"""

import numpy as np

from video_to_frame.analysis.coverage import CoveragePool
from video_to_frame.analysis.features import FeatureExtractor
from video_to_frame.analysis.quality import compute_quality_metrics
from video_to_frame.config import PipelineConfig
from video_to_frame.models import FrameMetrics, FrameRecord, KeptReason, RejectionReason
from video_to_frame.selection.scorer import FrameScorer
from video_to_frame.selection.selector import StreamingSelector, _Candidate

from synthetic import tile as _tile

_RNG = np.random.default_rng(123)


def _descriptors(n: int = 40) -> np.ndarray:
    return _RNG.integers(0, 256, size=(n, 32), dtype=np.uint8)


def _candidate(index: int, descriptors: np.ndarray, score: float) -> _Candidate:
    metrics = FrameMetrics(
        sharpness=500.0, brightness=120.0, contrast=50.0,
        quality_norm=0.8, feature_richness_norm=0.8,
        redundancy_norm=0.9, viewpoint_change_norm=0.1,
        coverage_gain_norm=0.05, score=score,
    )
    record = FrameRecord(index=index, timestamp_ms=index * 33.0, metrics=metrics,
                          kept=False, rejection_reason=RejectionReason.NONE)
    frame_bgr = np.zeros((4, 4, 3), dtype=np.uint8)
    gray_small = np.zeros((128, 128), dtype=np.uint8)
    return _Candidate(record=record, frame_bgr=frame_bgr, descriptors=descriptors,
                       dhash=index, gray_small=gray_small)


def _make_selector(config: PipelineConfig, kept: list, rejected: list) -> StreamingSelector:
    feature_extractor = FeatureExtractor(max_features=config.max_features, use_sift=config.use_sift)
    coverage_pool = CoveragePool(
        use_sift=config.use_sift,
        max_per_frame=config.coverage_pool_max_per_frame,
        max_pool_size=config.coverage_pool_max_size,
        ratio=config.descriptor_match_ratio,
        max_hamming_distance=config.orb_max_hamming_distance,
    )
    scorer = FrameScorer(config.weights, config.exposure_low, config.exposure_high, config.exposure_min_contrast)
    return StreamingSelector(
        config, coverage_pool, feature_extractor, scorer,
        on_kept=lambda r, f: kept.append(r),
        on_rejected=lambda r: rejected.append(r),
    )


def test_low_overlap_forces_retention():
    """Test A: candidate B has ~0 feature overlap with the previous kept
    frame A. Normal selection would reject B (it loses its cluster to a
    near-duplicate C) -- minimum-overlap protection must keep it anyway."""
    config = PipelineConfig()  # min_overlap_threshold=0.30, redundancy_threshold=0.82
    kept, rejected = [], []
    selector = _make_selector(config, kept, rejected)

    desc_a = _descriptors()
    desc_b = _descriptors()  # independent random descriptors -> ~0 overlap with A
    desc_c = desc_b.copy()   # near-duplicate of B -> clusters with B, and outscores it

    a = _candidate(0, desc_a, score=5.0)
    b = _candidate(1, desc_b, score=1.0)
    c = _candidate(2, desc_c, score=9.0)  # higher score -> wins the B/C cluster

    selector._cluster = [a]
    selector._flush_cluster()  # A becomes the previous kept frame
    assert kept[-1].index == 0 and kept[-1].kept_reason == KeptReason.NORMAL

    selector._cluster = [b, c]
    selector._flush_cluster()

    kept_indices = {r.index for r in kept}
    assert kept_indices == {0, 1, 2}
    # B (index 1) must be present and marked as forced by minimum-overlap protection.
    b_record = next(r for r in kept if r.index == 1)
    assert b_record.kept_reason == KeptReason.MINIMUM_OVERLAP_PROTECTION
    c_record = next(r for r in kept if r.index == 2)
    assert c_record.kept_reason == KeptReason.NORMAL
    assert rejected == []


def test_sufficient_overlap_does_not_force_retention():
    """Test B: candidate B shares half its features with the previous kept
    frame A (well above min_overlap_threshold, well below redundancy_threshold).
    B still loses its cluster to C, and should be rejected normally -- not
    force-kept."""
    config = PipelineConfig()
    kept, rejected = [], []
    selector = _make_selector(config, kept, rejected)

    desc_a = _descriptors()
    desc_b = np.vstack([desc_a[:20], _descriptors(20)])  # 50% overlap with A
    desc_c = desc_b.copy()  # near-duplicate of B -> clusters with B, outscores it

    a = _candidate(0, desc_a, score=5.0)
    b = _candidate(1, desc_b, score=1.0)
    c = _candidate(2, desc_c, score=9.0)

    selector._cluster = [a]
    selector._flush_cluster()

    selector._cluster = [b, c]
    selector._flush_cluster()

    kept_indices = {r.index for r in kept}
    rejected_indices = {r.index for r in rejected}
    assert kept_indices == {0, 2}
    assert rejected_indices == {1}
    assert rejected[0].rejection_reason == RejectionReason.REDUNDANT


def test_blur_floor_still_wins_over_min_overlap_protection():
    """Test C: a candidate with insufficient overlap AND severe blur must
    still be rejected by the hard blur floor -- minimum-overlap protection
    must never override it. This mirrors pipeline.py's actual per-frame gate
    (hard-blur check happens before a frame ever reaches the selector)."""
    import cv2

    config = PipelineConfig()
    kept, rejected = [], []
    selector = _make_selector(config, kept, rejected)

    def feed(frame_bgr, index):
        quality = compute_quality_metrics(frame_bgr)
        if quality.sharpness < config.hard_blur_floor:
            rejected.append(FrameRecord(index=index, timestamp_ms=index * 33.0, metrics=None,
                                         kept=False, rejection_reason=RejectionReason.BLUR))
            return
        selector.process(frame_bgr, index, index * 33.0, quality)

    sharp_a = _tile(0)
    feed(sharp_a, 0)

    # Frame at a very different pan offset (low overlap with A) AND heavily
    # blurred (well below the hard blur floor).
    far_and_blurry = cv2.GaussianBlur(_tile(600), (31, 31), 20)
    feed(far_and_blurry, 1)

    selector.finalize()

    assert any(r.index == 1 and r.rejection_reason == RejectionReason.BLUR for r in rejected)
    assert not any(r.index == 1 for r in kept)
