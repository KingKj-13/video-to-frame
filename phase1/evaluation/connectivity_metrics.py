"""Backend-independent feature-connectivity metrics -- a CONNECTIVITY PROXY,
not a measure of reconstruction quality (see module note below).

These reuse the Phase 0 ORB infrastructure directly (video_to_frame.analysis
.features) to answer the "feature connectivity" questions from the Phase 1
spec (section 3) without needing an external reconstruction backend at all:
how many ORB feature matches exist between temporally consecutive frames,
and where are the weak links.

IMPORTANT -- CONNECTIVITY PROXY, NOT RECONSTRUCTION QUALITY: a high ORB
match count between two frames means a feature matcher *could* link them; it
says nothing about whether an actual SfM/MVS pipeline would successfully
triangulate, register, or produce accurate geometry from them. Do not treat
"average_matches is high" as "this dataset reconstructs well" -- only a real
reconstruction backend (see reconstruction_backend.py) can answer that.

Per Phase 1 spec section 7: a low absolute keypoint_count is not
automatically a "bad frame" if its match_ratio is still healthy -- weak
texture and weak connectivity are different signals, tracked separately here
rather than conflated into one.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from video_to_frame.analysis.features import FeatureExtractor, match_descriptors

LOW_TEXTURE_KEYPOINT_FLOOR = 30  # below this, a frame is considered low-texture regardless of match ratio


@dataclass
class PairConnectivity:
    label_a: str
    label_b: str
    keypoints_a: int
    keypoints_b: int
    match_count: int
    match_ratio: float  # match_count / keypoints_a
    is_weak_link: bool  # match_count below the absolute weak_link_threshold
    is_low_texture: bool  # keypoints_a and/or keypoints_b below LOW_TEXTURE_KEYPOINT_FLOOR

    def to_dict(self) -> dict:
        return {
            "frame_a": self.label_a, "frame_b": self.label_b,
            "keypoints_a": self.keypoints_a, "keypoints_b": self.keypoints_b,
            "match_count": self.match_count, "match_ratio": round(self.match_ratio, 3),
            "is_weak_link": self.is_weak_link, "is_low_texture": self.is_low_texture,
        }


@dataclass
class ConnectivityMetrics:
    frame_count: int
    pair_count: int
    average_matches: float
    minimum_matches: int
    weak_link_threshold: int
    weak_link_count: int
    low_texture_pair_count: int
    pairs: List[PairConnectivity] = field(default_factory=list)

    @property
    def weak_link_pairs(self) -> List[PairConnectivity]:
        return [p for p in self.pairs if p.is_weak_link]

    def to_dict(self) -> dict:
        return {
            "frame_count": self.frame_count,
            "pair_count": self.pair_count,
            "average_matches": round(self.average_matches, 2),
            "minimum_matches": self.minimum_matches,
            "weak_link_threshold": self.weak_link_threshold,
            "weak_link_count": self.weak_link_count,
            "low_texture_pair_count": self.low_texture_pair_count,
            "weak_link_pairs": [p.to_dict() for p in self.weak_link_pairs],
        }


def _pairwise_connectivity(
    descriptors: List[Optional[np.ndarray]],
    labels: List[str],
    use_sift: bool,
    descriptor_match_ratio: float,
    orb_max_hamming_distance: int,
    weak_link_threshold: int,
) -> ConnectivityMetrics:
    pairs: List[PairConnectivity] = []
    for i in range(len(descriptors) - 1):
        desc_a, desc_b = descriptors[i], descriptors[i + 1]
        kp_a = 0 if desc_a is None else len(desc_a)
        kp_b = 0 if desc_b is None else len(desc_b)
        m = match_descriptors(
            desc_a, desc_b, use_sift=use_sift, ratio=descriptor_match_ratio,
            max_hamming_distance=orb_max_hamming_distance,
        )
        ratio = (m / kp_a) if kp_a else 0.0
        pairs.append(PairConnectivity(
            label_a=labels[i], label_b=labels[i + 1],
            keypoints_a=kp_a, keypoints_b=kp_b, match_count=m, match_ratio=ratio,
            is_weak_link=m < weak_link_threshold,
            is_low_texture=(kp_a < LOW_TEXTURE_KEYPOINT_FLOOR or kp_b < LOW_TEXTURE_KEYPOINT_FLOOR),
        ))

    match_counts = [p.match_count for p in pairs]
    return ConnectivityMetrics(
        frame_count=len(descriptors),
        pair_count=len(pairs),
        average_matches=(sum(match_counts) / len(match_counts)) if match_counts else 0.0,
        minimum_matches=min(match_counts) if match_counts else 0,
        weak_link_threshold=weak_link_threshold,
        weak_link_count=sum(1 for p in pairs if p.is_weak_link),
        low_texture_pair_count=sum(1 for p in pairs if p.is_low_texture),
        pairs=pairs,
    )


def compute_connectivity_metrics(
    frame_paths: List[str],
    use_sift: bool = False,
    max_features: int = 500,
    descriptor_match_ratio: float = 0.75,
    orb_max_hamming_distance: int = 64,
    weak_link_threshold: int = 15,
) -> ConnectivityMetrics:
    """File-based connectivity metrics over an already-selected dataset.
    `frame_paths` is expected to already be in original temporal order (e.g.
    sorted by the frame index in each filename).

    weak_link_threshold is an absolute match-count floor, not a ratio --
    photogrammetry tooling generally wants at least a few dozen robust
    correspondences to reliably estimate a relative pose between two views,
    so this defaults conservatively low (15) as a "clearly insufficient"
    flag rather than a quality target.
    """
    if len(frame_paths) < 2:
        return ConnectivityMetrics(
            frame_count=len(frame_paths), pair_count=0, average_matches=0.0,
            minimum_matches=0, weak_link_threshold=weak_link_threshold,
            weak_link_count=0, low_texture_pair_count=0,
        )

    extractor = FeatureExtractor(max_features=max_features, use_sift=use_sift)
    descriptors = []
    for path in frame_paths:
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"Could not read frame for connectivity analysis: {path}")
        _keypoints, desc = extractor.extract(gray)
        descriptors.append(desc)

    labels = [os.path.basename(p) for p in frame_paths]
    return _pairwise_connectivity(
        descriptors, labels, use_sift, descriptor_match_ratio, orb_max_hamming_distance, weak_link_threshold,
    )
