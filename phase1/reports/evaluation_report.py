"""Builds evaluation.json / evaluation.html from a benchmark_runner result
(and, optionally, a reduction-level sweep) -- the Phase 1 spec section 4
deliverable.

Labeling discipline (Phase 1.1 spec section 2): ORB feature-connectivity
numbers are always reported under "connectivity proxy" headings, never
"reconstruction quality." Reconstruction fields are only ever populated from
an actual ReconstructionMetrics result and are explicitly marked UNAVAILABLE
when no backend was configured -- never backfilled from connectivity data.
"""

from __future__ import annotations

import html
import json
import os
from typing import List, Optional


def build_evaluation_json(benchmark_result: dict, reduction_levels: Optional[List[dict]] = None) -> dict:
    datasets = benchmark_result["datasets"]
    intelligent = datasets.get("intelligent_selector", {})
    conn = intelligent.get("connectivity", {})

    evaluation = {
        "video": benchmark_result["video"],
        "original_frames": benchmark_result["original_frames"],
        "processing_time_sec": benchmark_result["processing_time_sec"],
        "selected_frames": intelligent.get("selected_frame_count"),
        "reduction_percentage": intelligent.get("reduction_percent"),
        "dataset_size_mb": round(intelligent.get("dataset_size_bytes", 0) / (1024 * 1024), 2),
        "frames_forced_by_minimum_overlap": intelligent.get("frames_forced_by_minimum_overlap"),
        "blur_rejected": intelligent.get("blur_rejected"),
        "redundant_rejected": intelligent.get("redundant_rejected"),
        "connectivity_proxy": {
            "average_feature_matches": conn.get("average_matches"),
            "minimum_feature_matches": conn.get("minimum_matches"),
            "weak_link_count": conn.get("weak_link_count"),
            "low_texture_pair_count": conn.get("low_texture_pair_count"),
            "note": "ORB feature-match counts between consecutive kept frames -- a connectivity "
                    "PROXY, not a measure of actual reconstruction quality.",
        },
        "failure_verdict": intelligent.get("failure_analysis", {}).get("verdict"),
        "datasets": datasets,
    }

    recon = intelligent.get("reconstruction")
    if recon is None:
        evaluation["reconstruction"] = {
            "status": "UNAVAILABLE",
            "reason": "No reconstruction backend was configured for this run.",
        }
    elif not recon.get("success"):
        evaluation["reconstruction"] = {
            "status": "FAILED",
            "backend": recon.get("backend_name"),
            "error": recon.get("error"),
        }
    else:
        evaluation["reconstruction"] = {
            "status": "COMPLETE",
            "backend": recon.get("backend_name"),
            "registered_frame_count": recon.get("registered_frame_count"),
            "registration_rate": recon.get("registration_rate"),
            "disconnected_components": recon.get("disconnected_components"),
            "point_count": recon.get("point_count"),
            "mean_reprojection_error": recon.get("mean_reprojection_error"),
        }

    if reduction_levels is not None:
        evaluation["reduction_level_sweep"] = reduction_levels

    return evaluation


def write_evaluation_json(evaluation: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2)


_CSS = """
:root { color-scheme: light dark; }
body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; padding: 24px; background: #f7f7f8; color: #1a1a1a; }
h1, h2, h3 { font-weight: 600; }
table { border-collapse: collapse; margin-bottom: 16px; width: 100%; }
td, th { padding: 6px 12px; border-bottom: 1px solid #ddd; text-align: left; font-size: 13px; }
th { background: #eee; }
.verdict-good { color: #1b5e20; font-weight: 600; }
.verdict-warning { color: #8a5a00; font-weight: 600; }
.verdict-failure { color: #b71c1c; font-weight: 600; }
.verdict-unknown { color: #555; font-weight: 600; }
.status-complete { color: #1b5e20; font-weight: 600; }
.status-unavailable, .status-failed { color: #8a5a00; font-weight: 600; }
.proxy-banner { background: #e3f2fd; border: 1px solid #90caf9; color: #0d47a1; padding: 8px 12px;
    border-radius: 6px; margin-bottom: 12px; font-size: 13px; }
section { margin-bottom: 32px; }
.empty { color: #888; font-style: italic; }
@media (prefers-color-scheme: dark) {
  body { background: #17181a; color: #e6e6e6; }
  td, th { border-color: #333; }
  th { background: #222; }
  .proxy-banner { background: #0d2a42; border-color: #1565c0; color: #90caf9; }
}
"""


def _verdict_class(verdict: Optional[str]) -> str:
    return f"verdict-{verdict}" if verdict else "verdict-unknown"


def _tradeoff_svg(sweep: List[dict], width: int = 700, height: int = 240) -> str:
    points = [r for r in sweep if r.get("connectivity", {}).get("average_matches") is not None]
    if not points:
        return "<p class='empty'>No connectivity data in the sweep to plot.</p>"

    frame_counts = [r["selected_frame_count"] for r in points]
    matches = [r["connectivity"]["average_matches"] for r in points]
    min_f, max_f = min(frame_counts), max(frame_counts)
    min_m, max_m = min(matches), max(matches)
    f_span = (max_f - min_f) or 1
    m_span = (max_m - min_m) or 1

    def x_of(f):
        return 40 + (width - 60) * ((f - min_f) / f_span)

    def y_of(m):
        return height - 30 - (height - 50) * ((m - min_m) / m_span)

    pts = [(x_of(f), y_of(m), r) for f, m, r in zip(frame_counts, matches, points)]
    pts.sort(key=lambda p: p[0])
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in pts)
    circles = "".join(
        f"<circle cx='{x:.1f}' cy='{y:.1f}' r='4' fill='#1976d2'>"
        f"<title>{r['selected_frame_count']} frames ({r['achieved_reduction_percent']}% reduction) "
        f"-> avg {r['connectivity']['average_matches']} matches</title></circle>"
        for x, y, r in pts
    )
    return (
        f"<svg viewBox='0 0 {width} {height}' width='100%' height='{height}' role='img' "
        f"aria-label='Frame count vs connectivity proxy'>"
        f"<line x1='40' y1='{height-30}' x2='{width-20}' y2='{height-30}' stroke='#999'/>"
        f"<line x1='40' y1='20' x2='40' y2='{height-30}' stroke='#999'/>"
        f"<text x='{width/2}' y='{height-5}' font-size='11' text-anchor='middle' fill='currentColor'>Selected frame count &#8594;</text>"
        f"<text x='12' y='{height/2}' font-size='11' text-anchor='middle' fill='currentColor' "
        f"transform='rotate(-90 12 {height/2})'>Avg connectivity (ORB matches)</text>"
        f"<polyline fill='none' stroke='#1976d2' stroke-width='2' points='{polyline}' />{circles}</svg>"
    )


def render_evaluation_html(evaluation: dict, path: str) -> None:
    datasets = evaluation.get("datasets", {})
    rows = []
    for name, d in datasets.items():
        conn = d.get("connectivity", {})
        fa = d.get("failure_analysis", {})
        rows.append(
            f"<tr><td>{html.escape(name)}</td>"
            f"<td>{d.get('selected_frame_count')}</td>"
            f"<td>{d.get('reduction_percent')}%</td>"
            f"<td>{round(d.get('dataset_size_bytes', 0) / (1024*1024), 2)} MB</td>"
            f"<td>{conn.get('average_matches')}</td>"
            f"<td>{conn.get('minimum_matches')}</td>"
            f"<td>{conn.get('weak_link_count')}</td>"
            f"<td>{conn.get('low_texture_pair_count')}</td>"
            f"<td class='{_verdict_class(fa.get('verdict'))}'>{html.escape(fa.get('verdict', 'unknown'))}</td></tr>"
        )
    dataset_rows = "".join(rows) if rows else "<tr><td colspan='9' class='empty'>No datasets.</td></tr>"

    sweep_html = "<p class='empty'>No reduction-level sweep included.</p>"
    tradeoff_svg = "<p class='empty'>No reduction-level sweep included.</p>"
    sweep = evaluation.get("reduction_level_sweep")
    if sweep:
        sweep_rows = []
        for r in sweep:
            c = r.get("connectivity", {})
            sweep_rows.append(
                f"<tr><td>{r['target_reduction_percent']}%</td><td>{r['achieved_reduction_percent']}%</td>"
                f"<td>{r['selected_frame_count']}</td><td>{r['original_frame_count']}</td>"
                f"<td>{c.get('average_matches', '-')}</td><td>{c.get('minimum_matches', '-')}</td>"
                f"<td>{c.get('weak_link_count', '-')}</td></tr>"
            )
        sweep_html = (
            "<table><tr><th>Target reduction</th><th>Achieved reduction</th>"
            "<th>Selected frames</th><th>Original frames</th><th>Avg matches</th>"
            f"<th>Min matches</th><th>Weak links</th></tr>{''.join(sweep_rows)}</table>"
        )
        tradeoff_svg = _tradeoff_svg(sweep)

    recon = evaluation.get("reconstruction", {})
    recon_status = recon.get("status", "UNAVAILABLE")
    if recon_status == "COMPLETE":
        recon_html = (
            f"<table>"
            f"<tr><td>Backend</td><td>{html.escape(str(recon.get('backend')))}</td></tr>"
            f"<tr><td>Registered frames</td><td>{recon.get('registered_frame_count')}</td></tr>"
            f"<tr><td>Registration rate</td><td>{recon.get('registration_rate')}</td></tr>"
            f"<tr><td>Disconnected components</td><td>{recon.get('disconnected_components')}</td></tr>"
            f"<tr><td>3D point count</td><td>{recon.get('point_count')}</td></tr>"
            f"<tr><td>Mean reprojection error</td><td>{recon.get('mean_reprojection_error')}</td></tr>"
            f"</table>"
        )
    else:
        recon_html = (
            f"<p class='status-{recon_status.lower()}'>{html.escape(recon_status)}</p>"
            f"<p class='empty'>{html.escape(str(recon.get('reason') or recon.get('error') or ''))}</p>"
        )

    cp = evaluation.get("connectivity_proxy", {})

    doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Phase 1 Evaluation Report</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Phase 1 Evaluation Report</h1>
<div class="proxy-banner">ORB feature-connectivity numbers on this page are a <b>connectivity proxy</b> --
they indicate whether a feature matcher could plausibly link two frames, not whether an actual
3D reconstruction would succeed or be accurate. See the "Reconstruction" section below for real
reconstruction-backend results, or its explicit UNAVAILABLE/FAILED status when none were produced.</div>
<section>
<h2>Summary</h2>
<table>
<tr><td>Video</td><td>{html.escape(str(evaluation.get('video')))}</td></tr>
<tr><td>Original frames</td><td>{evaluation.get('original_frames')}</td></tr>
<tr><td>Selected frames</td><td>{evaluation.get('selected_frames')}</td></tr>
<tr><td>Reduction</td><td>{evaluation.get('reduction_percentage')}%</td></tr>
<tr><td>Blur rejected / Redundant rejected</td><td>{evaluation.get('blur_rejected')} / {evaluation.get('redundant_rejected')}</td></tr>
<tr><td>Frames forced by minimum-overlap protection</td><td>{evaluation.get('frames_forced_by_minimum_overlap')}</td></tr>
<tr><td>Processing time</td><td>{evaluation.get('processing_time_sec')} s</td></tr>
<tr><td>Failure verdict (intelligent selector)</td>
    <td class="{_verdict_class(evaluation.get('failure_verdict'))}">{html.escape(str(evaluation.get('failure_verdict')))}</td></tr>
</table>
</section>
<section>
<h2>Connectivity Proxy (intelligent selector)</h2>
<table>
<tr><td>Average ORB matches/pair</td><td>{cp.get('average_feature_matches')}</td></tr>
<tr><td>Minimum ORB matches</td><td>{cp.get('minimum_feature_matches')}</td></tr>
<tr><td>Weak-link pairs</td><td>{cp.get('weak_link_count')}</td></tr>
<tr><td>Low-texture pairs (&lt;{30} keypoints on either side)</td><td>{cp.get('low_texture_pair_count')}</td></tr>
</table>
</section>
<section>
<h2>Reconstruction (real backend)</h2>
{recon_html}
</section>
<section>
<h2>Dataset comparison (connectivity proxy, not reconstruction quality)</h2>
<table>
<tr><th>Dataset</th><th>Frames</th><th>Reduction</th><th>Size</th>
    <th>Avg matches/pair</th><th>Min matches</th><th>Weak links</th><th>Low-texture pairs</th><th>Verdict</th></tr>
{dataset_rows}
</table>
</section>
<section>
<h2>Reduction-level sweep</h2>
{sweep_html}
</section>
<section>
<h2>Tradeoff curve: frame count vs. connectivity proxy</h2>
{tradeoff_svg}
<p class="empty">This is NOT a reconstruction-quality curve unless a real reconstruction backend
was used to produce it (see Reconstruction section above). It only shows how the ORB
feature-connectivity proxy changes as more frames are removed.</p>
</section>
</body>
</html>
"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
