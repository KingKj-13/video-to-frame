"""Classifies whether the optimized frame set looks safe for downstream
reconstruction, per the Phase 1 spec's Case A/B/C framing (section 5).

When a real reconstruction backend is configured, its metrics drive the
verdict directly. When none is configured, the connectivity-metrics proxy
(phase1.evaluation.connectivity_metrics) is the best signal available, and
the verdict says so explicitly rather than pretending to know reconstruction
actually succeeded or failed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from .connectivity_metrics import ConnectivityMetrics, PairConnectivity
from .reconstruction_backend import ReconstructionMetrics


class Verdict(str, Enum):
    GOOD = "good"
    WARNING = "warning"
    FAILURE = "failure"
    UNKNOWN = "unknown"  # no reconstruction backend configured to confirm directly


@dataclass
class FailureCase:
    verdict: Verdict
    reason: str
    weak_link_pairs: List[PairConnectivity] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "reason": self.reason,
            "weak_link_pairs": [p.to_dict() for p in self.weak_link_pairs],
        }


def classify(
    optimized_connectivity: ConnectivityMetrics,
    optimized_reconstruction: Optional[ReconstructionMetrics] = None,
) -> FailureCase:
    """Case A (GOOD): reconstruction backend confirms full/near-full
    registration with one connected component. Case B (WARNING): backend
    confirms reconstruction ran but with reduced registration. Case C
    (FAILURE): backend confirms a disconnected reconstruction. UNKNOWN: no
    backend configured -- falls back to the weak-link connectivity proxy.
    """
    if optimized_reconstruction is not None and optimized_reconstruction.success:
        components = optimized_reconstruction.disconnected_components
        rate = optimized_reconstruction.registration_rate

        if components is not None and components > 1:
            return FailureCase(
                Verdict.FAILURE,
                f"Reconstruction produced {components} disconnected components -- "
                "the frame set likely has a connectivity gap.",
                optimized_connectivity.weak_link_pairs,
            )
        if rate is not None and rate < 0.9:
            return FailureCase(
                Verdict.WARNING,
                f"Only {rate:.0%} of selected frames were registered by the reconstruction backend.",
                optimized_connectivity.weak_link_pairs,
            )
        return FailureCase(Verdict.GOOD, "Reconstruction backend reports full registration.", [])

    if optimized_reconstruction is not None and not optimized_reconstruction.success:
        return FailureCase(
            Verdict.UNKNOWN,
            f"Reconstruction backend did not complete successfully: {optimized_reconstruction.error}",
            optimized_connectivity.weak_link_pairs,
        )

    if optimized_connectivity.weak_link_count > 0:
        return FailureCase(
            Verdict.WARNING,
            f"{optimized_connectivity.weak_link_count} consecutive kept-frame pair(s) fall below the "
            f"weak-link feature-match threshold ({optimized_connectivity.weak_link_threshold}) -- a "
            "downstream SfM matcher may fail to link them. No reconstruction backend was configured "
            "to confirm this directly; this is the connectivity proxy only.",
            optimized_connectivity.weak_link_pairs,
        )

    return FailureCase(
        Verdict.UNKNOWN,
        "No weak links detected by the connectivity proxy, but no reconstruction backend was "
        "configured to directly confirm reconstruction success.",
        [],
    )
