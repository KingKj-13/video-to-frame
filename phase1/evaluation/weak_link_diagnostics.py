"""Per-weak-link root-cause diagnostics (Phase 1.1 spec section 6).

Cross-references a connectivity weak link (a pair of consecutive *kept*
frames with too few ORB matches) against the original pipeline run's full
per-frame report -- every frame the selector ever saw, kept or rejected --
to answer: what happened between these two kept frames, and why weren't any
of the in-between frames kept instead?

This is diagnostic only. It does not change the selector or draw a
conclusion on your behalf -- it assembles the evidence (rejection reasons,
blur scores, viewpoint-change scores, keypoint/match counts) so a human (or
a later, evidence-justified fix) can determine the actual cause: low
texture, blur, an excessive viewpoint jump, exposure change, occlusion,
feature-detector failure, or overly aggressive redundancy filtering.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .connectivity_metrics import PairConnectivity

_INDEX_RE = re.compile(r"(\d+)")


def _index_from_label(label: str) -> Optional[int]:
    m = _INDEX_RE.search(label)
    return int(m.group(1)) if m else None


@dataclass
class IntermediateFrame:
    index: int
    kept: bool
    rejection_reason: Optional[str]
    kept_reason: Optional[str]
    sharpness: Optional[float]
    redundancy_norm: Optional[float]
    viewpoint_change_norm: Optional[float]
    coverage_gain_norm: Optional[float]

    def to_dict(self) -> dict:
        return {
            "index": self.index, "kept": self.kept,
            "rejection_reason": self.rejection_reason, "kept_reason": self.kept_reason,
            "sharpness": self.sharpness, "redundancy_norm": self.redundancy_norm,
            "viewpoint_change_norm": self.viewpoint_change_norm, "coverage_gain_norm": self.coverage_gain_norm,
        }


@dataclass
class WeakLinkDiagnosis:
    frame_a_index: int
    frame_b_index: int
    keypoints_a: int
    keypoints_b: int
    match_count: int
    match_ratio: float
    is_low_texture: bool
    gap_frame_count: int  # how many original frames lie strictly between A and B
    intermediate_frames: List[IntermediateFrame] = field(default_factory=list)
    likely_causes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "frame_a_index": self.frame_a_index, "frame_b_index": self.frame_b_index,
            "keypoints_a": self.keypoints_a, "keypoints_b": self.keypoints_b,
            "match_count": self.match_count, "match_ratio": round(self.match_ratio, 3),
            "is_low_texture": self.is_low_texture,
            "gap_frame_count": self.gap_frame_count,
            "intermediate_frames": [f.to_dict() for f in self.intermediate_frames],
            "likely_causes": self.likely_causes,
        }


def _infer_causes(pair: PairConnectivity, intermediates: List[IntermediateFrame]) -> List[str]:
    """Evidence-based, descriptive labels only -- this never modifies
    selection behavior, it just names what the recorded evidence points to
    so a human can decide what (if anything) to do about it."""
    causes = []

    if pair.is_low_texture:
        causes.append(
            f"low_texture (keypoints_a={pair.keypoints_a}, keypoints_b={pair.keypoints_b} -- "
            "few features exist to match regardless of selection choices)"
        )
    if not pair.is_low_texture and pair.match_ratio < 0.15:
        causes.append(f"excessive_viewpoint_jump (match_ratio={pair.match_ratio:.2f} despite adequate keypoints)")

    blur_rejects = [f for f in intermediates if f.rejection_reason == "blur"]
    if blur_rejects:
        causes.append(f"blur ({len(blur_rejects)} intermediate frame(s) rejected as blurry)")

    redundant_rejects = [f for f in intermediates if f.rejection_reason == "redundant"]
    if redundant_rejects:
        avg_viewpoint = [f.viewpoint_change_norm for f in redundant_rejects if f.viewpoint_change_norm is not None]
        if avg_viewpoint and max(avg_viewpoint) > 0.5:
            causes.append(
                f"overly_aggressive_redundancy_filtering ({len(redundant_rejects)} intermediate frame(s) "
                f"rejected as redundant despite viewpoint_change_norm up to {max(avg_viewpoint):.2f})"
            )
        else:
            causes.append(f"redundancy_filtering ({len(redundant_rejects)} intermediate frame(s) rejected as redundant)")

    if not intermediates and not causes:
        causes.append("adjacent_kept_frames_still_weak (no intermediate frames existed to choose from instead)")

    if not causes:
        causes.append("undetermined (evidence inconclusive -- inspect intermediate_frames manually)")

    return causes


def diagnose_weak_links(
    weak_links: List[PairConnectivity],
    original_frame_records: List[dict],
) -> List[WeakLinkDiagnosis]:
    """`original_frame_records` is the `frames` array from the intelligent
    selector's own report.json (video_to_frame.models.record_to_dict output)
    -- every frame the pipeline ever saw, in index order."""
    by_index: Dict[int, dict] = {r["index"]: r for r in original_frame_records}

    diagnoses = []
    for pair in weak_links:
        idx_a = _index_from_label(pair.label_a)
        idx_b = _index_from_label(pair.label_b)
        if idx_a is None or idx_b is None:
            continue

        intermediates = []
        for i in range(idx_a + 1, idx_b):
            record = by_index.get(i)
            if record is None:
                continue
            intermediates.append(IntermediateFrame(
                index=i, kept=record.get("kept", False),
                rejection_reason=record.get("rejection_reason"), kept_reason=record.get("kept_reason"),
                sharpness=record.get("sharpness"), redundancy_norm=record.get("redundancy_norm"),
                viewpoint_change_norm=record.get("viewpoint_change_norm"),
                coverage_gain_norm=record.get("coverage_gain_norm"),
            ))

        diagnosis = WeakLinkDiagnosis(
            frame_a_index=idx_a, frame_b_index=idx_b,
            keypoints_a=pair.keypoints_a, keypoints_b=pair.keypoints_b,
            match_count=pair.match_count, match_ratio=pair.match_ratio,
            is_low_texture=pair.is_low_texture,
            gap_frame_count=max(0, idx_b - idx_a - 1),
            intermediate_frames=intermediates,
        )
        diagnosis.likely_causes = _infer_causes(pair, intermediates)
        diagnoses.append(diagnosis)

    return diagnoses


def render_text_diagram(diagnosis: WeakLinkDiagnosis) -> str:
    """Simple ASCII visual: previous kept frame -> rejected frames -> next kept frame."""
    lines = [f"Frame {diagnosis.frame_a_index} (kept)", "   |"]
    if not diagnosis.intermediate_frames:
        lines.append("   | (no intermediate frames -- adjacent in the original video)")
    for f in diagnosis.intermediate_frames:
        tag = f.kept_reason if f.kept else f.rejection_reason
        lines.append(f"   +-- frame {f.index}: {tag} (sharpness={f.sharpness}, viewpoint_change={f.viewpoint_change_norm})")
    lines.append("   |")
    lines.append(f"Frame {diagnosis.frame_b_index} (kept)  <- match_count={diagnosis.match_count}, "
                  f"ratio={diagnosis.match_ratio:.2f}, keypoints=({diagnosis.keypoints_a},{diagnosis.keypoints_b})")
    return "\n".join(lines)
