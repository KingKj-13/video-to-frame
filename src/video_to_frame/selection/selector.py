from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Callable, List, Optional

import cv2
import numpy as np

from ..config import PipelineConfig
from ..coverage import CoveragePool
from ..features import FeatureExtractor, match_descriptors
from ..features.similarity import dhash, hamming_distance, ssim_precomputed
from ..core.models import FrameRecord, KeptReason, QualityMetrics, RejectionReason
from ..scoring import FrameScorer

logger = logging.getLogger(__name__)

OnKept = Callable[[FrameRecord, np.ndarray], None]
OnRejected = Callable[[FrameRecord], None]


@dataclass
class _Candidate:
    record: FrameRecord
    frame_bgr: np.ndarray
    descriptors: Optional[np.ndarray]
    dhash: int
    gray_small: np.ndarray


class StreamingSelector:
    """Groups consecutive, highly-overlapping frames into a "redundancy
    cluster" and keeps only the single highest-scoring representative from
    each cluster -- this is what implements the spec's "Frame 1-2-3-4-5,
    keep only one representative" behavior in O(n), without pairwise
    comparison across the whole video.

    A candidate is compared against:
      - the anchor (first frame) of the *currently open* cluster, so
        consecutive near-duplicates actually get grouped before any of them
        is flushed to the kept set;
      - a small rolling window of the most-recently-*kept* frames, so a
        cluster boundary doesn't reset comparison to nothing;
      - a global descriptor coverage pool (see CoveragePool), so a frame that
        revisits an earlier viewpoint after the camera wandered away isn't
        treated as novel just because it's not adjacent to the last kept frame.

    A cluster closes -- flushing its best-scoring frame as "kept" and
    discarding the rest as "redundant" -- as soon as a candidate is not
    redundant against all of the above.

    Minimum-overlap protection (see _flush_cluster/_overlap_ratio) is a
    separate safety layer applied to every candidate a cluster flush would
    otherwise reject: if its ORB feature-overlap with the nearest kept
    predecessor falls below config.min_overlap_threshold, it is force-kept
    instead, so a downstream SfM feature matcher never gets handed two
    consecutive kept frames with too little in common to link. This is the
    opposite concern from redundancy_threshold ("too similar to keep") and
    must not be conflated with it. It never overrides the hard blur floor,
    which is applied upstream in the pipeline before a frame ever reaches
    the selector.
    """

    def __init__(self, config: PipelineConfig, coverage_pool: CoveragePool,
                 feature_extractor: FeatureExtractor, scorer: FrameScorer,
                 on_kept: OnKept, on_rejected: OnRejected):
        self.config = config
        self.coverage_pool = coverage_pool
        self.feature_extractor = feature_extractor
        self.scorer = scorer
        self.on_kept = on_kept
        self.on_rejected = on_rejected
        self._cluster: List[_Candidate] = []
        self._reference_window: deque = deque(maxlen=config.reference_window_size)

    def process(self, frame_bgr: np.ndarray, index: int, timestamp_ms: float, quality: QualityMetrics) -> None:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        analysis_gray = gray
        scale = self.config.analysis_scale
        if scale and scale < 1.0:
            analysis_gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        _keypoints, descriptors = self.feature_extractor.extract(analysis_gray)

        frame_hash = dhash(gray)
        gray_small = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)

        candidate = _Candidate(
            record=FrameRecord(index=index, timestamp_ms=timestamp_ms, metrics=None,
                                kept=False, rejection_reason=RejectionReason.NONE),
            frame_bgr=frame_bgr,
            descriptors=descriptors,
            dhash=frame_hash,
            gray_small=gray_small,
        )

        references = list(self._reference_window)
        if self._cluster:
            references.append(self._cluster[0])

        redundancy, viewpoint_change = self._compare_to_references(candidate, references)
        # The global pool only gains a cluster's descriptors once it's flushed,
        # so a candidate being compared against its own still-open cluster would
        # otherwise see an empty/stale pool and look "fully novel" every time.
        # Treat the open cluster's anchor as already covered too.
        anchor_descriptors = self._cluster[0].descriptors if self._cluster else None
        coverage_gain = self.coverage_pool.coverage_gain(descriptors, extra_pool=anchor_descriptors)

        num_descriptors = 0 if descriptors is None else len(descriptors)
        candidate.record.metrics = self.scorer.score(
            sharpness=quality.sharpness,
            brightness=quality.brightness,
            contrast=quality.contrast,
            keypoint_count=num_descriptors,
            redundancy=redundancy,
            viewpoint_change=viewpoint_change,
            coverage_gain=coverage_gain,
        )

        is_redundant = (
            redundancy >= self.config.redundancy_threshold
            and coverage_gain < self.config.coverage_gain_threshold
        )

        if is_redundant and self._cluster:
            self._cluster.append(candidate)
        else:
            self._flush_cluster()
            self._cluster = [candidate]

    def _compare_to_references(self, candidate: _Candidate, references: List[_Candidate]) -> tuple[float, float]:
        if not references:
            return 0.0, 1.0

        best_redundancy = 0.0
        best_viewpoint_change = 1.0
        total = 0 if candidate.descriptors is None else len(candidate.descriptors)

        for ref in references:
            matches = match_descriptors(
                candidate.descriptors, ref.descriptors,
                use_sift=self.config.use_sift, ratio=self.config.descriptor_match_ratio,
                max_hamming_distance=self.config.orb_max_hamming_distance,
            )
            overlap_ratio = (matches / total) if total else 0.0

            if total > 0:
                # Feature overlap is the primary, spatially-meaningful redundancy
                # signal -- it reflects actual scene-point correspondence. Whole-
                # frame hash/SSIM only escalate redundancy for near-static
                # duplicates (e.g. a paused camera); they must not drown out a
                # real viewpoint change just because a large, unchanging
                # background dominates the frame.
                hd = hamming_distance(candidate.dhash, ref.dhash) / 64.0
                hash_similarity = 1.0 - hd
                structural_similarity = ssim_precomputed(candidate.gray_small, ref.gray_small)
                near_static = hash_similarity > 0.97 or structural_similarity > 0.97
                redundancy = max(overlap_ratio, 1.0 if near_static else 0.0)
            else:
                # No features to match (e.g. a flat/textureless frame) -- fall
                # back to whole-frame similarity as the only available signal.
                hd = hamming_distance(candidate.dhash, ref.dhash) / 64.0
                structural_similarity = ssim_precomputed(candidate.gray_small, ref.gray_small)
                redundancy = max(1.0 - hd, structural_similarity)

            viewpoint_change = (1.0 - overlap_ratio) if total else 1.0

            best_redundancy = max(best_redundancy, redundancy)
            best_viewpoint_change = min(best_viewpoint_change, viewpoint_change)

        return best_redundancy, best_viewpoint_change

    def _flush_cluster(self) -> None:
        if not self._cluster:
            return

        winner = max(self._cluster, key=lambda c: c.record.metrics.score)
        # The frame chronologically closest before this cluster -- updated as
        # we go, so a candidate is always checked against whichever candidate
        # (winner or an earlier forced-keep) is now its true nearest kept
        # predecessor, not a stale pre-cluster snapshot.
        previous_kept = self._reference_window[-1] if self._reference_window else None

        cluster_size = len(self._cluster)
        for candidate in self._cluster:
            if candidate is winner:
                # Current logic already keeps this one -- no overlap check needed.
                candidate.record.kept = True
                candidate.record.rejection_reason = RejectionReason.NONE
                candidate.record.kept_reason = KeptReason.NORMAL
                self._commit_kept(candidate)
                logger.debug("frame %d kept (cluster winner of %d, score=%.3f)",
                             candidate.record.index, cluster_size, candidate.record.metrics.score)
                previous_kept = candidate
                continue

            # Normal logic would reject this candidate (it lost its cluster).
            # Before actually rejecting it, make sure that doesn't leave the
            # previous kept frame and whatever comes next too disconnected
            # for a downstream feature matcher to link them.
            force_keep = False
            if previous_kept is not None:
                overlap = self._overlap_ratio(candidate, previous_kept)
                force_keep = overlap < self.config.min_overlap_threshold

            if force_keep:
                candidate.record.kept = True
                candidate.record.rejection_reason = RejectionReason.NONE
                candidate.record.kept_reason = KeptReason.MINIMUM_OVERLAP_PROTECTION
                self._commit_kept(candidate)
                logger.debug("frame %d force-kept (overlap=%.3f < min_overlap_threshold=%.3f vs frame %d)",
                             candidate.record.index, overlap, self.config.min_overlap_threshold,
                             previous_kept.record.index if previous_kept else -1)
                previous_kept = candidate
            else:
                candidate.record.kept = False
                candidate.record.rejection_reason = RejectionReason.REDUNDANT
                self.on_rejected(candidate.record)
                logger.debug("frame %d rejected as redundant (lost cluster of %d, score=%.3f)",
                             candidate.record.index, cluster_size, candidate.record.metrics.score)

        self._cluster = []

    def _commit_kept(self, candidate: _Candidate) -> None:
        self.coverage_pool.add(candidate.descriptors)
        self._reference_window.append(candidate)
        self.on_kept(candidate.record, candidate.frame_bgr)

    def _overlap_ratio(self, candidate: _Candidate, reference: _Candidate) -> float:
        """ORB feature-overlap ratio of `candidate` against `reference`: the
        fraction of candidate's descriptors with a good match in reference.

        Deliberately simpler than _compare_to_references' redundancy formula
        (no whole-frame hash/SSIM escalation) -- this is the minimum-overlap
        *safety* check, answering "can a feature matcher actually link these
        two frames," not "do they look similar" (that's redundancy_threshold's
        job; the two must stay separate, see module docstring). No descriptors
        on either side means no feature link is possible, so that's treated as
        zero overlap -- the conservative, force-keep-favoring outcome.
        """
        total = 0 if candidate.descriptors is None else len(candidate.descriptors)
        if total == 0 or reference.descriptors is None or len(reference.descriptors) == 0:
            return 0.0
        matches = match_descriptors(
            candidate.descriptors, reference.descriptors,
            use_sift=self.config.use_sift, ratio=self.config.descriptor_match_ratio,
            max_hamming_distance=self.config.orb_max_hamming_distance,
        )
        return matches / total

    def finalize(self) -> None:
        """Flushes any still-open cluster -- ensures the video's final
        viewpoint isn't lost just because the stream ended mid-cluster."""
        self._flush_cluster()
