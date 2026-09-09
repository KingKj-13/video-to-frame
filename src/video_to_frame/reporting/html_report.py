from __future__ import annotations

import html
import os
from typing import List

from ..io.extractor import VideoInfo

_CSS = """
:root { color-scheme: light dark; }
body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; padding: 24px; background: #f7f7f8; color: #1a1a1a; }
h1, h2, h3 { font-weight: 600; }
.summary-table { border-collapse: collapse; margin-bottom: 8px; }
.summary-table td, .summary-table th { padding: 4px 12px; border-bottom: 1px solid #ddd; text-align: left; }
.warning { color: #8a5a00; background: #fff6e0; border: 1px solid #f0d68a; padding: 8px 12px; border-radius: 6px; margin-bottom: 12px; }
.videos { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 8px; }
.videos video { max-width: 480px; width: 100%; background: #000; }
.grid { display: flex; flex-wrap: wrap; gap: 8px; }
.thumb { margin: 0; width: 140px; font-size: 11px; text-align: center; }
.thumb img { width: 100%; height: 100px; object-fit: cover; border-radius: 4px; border: 1px solid #ccc; }
.thumb .placeholder { width: 100%; height: 100px; background: #eee; border: 1px dashed #bbb; border-radius: 4px; display: flex; align-items: center; justify-content: center; color: #888; }
.thumb.uncertain img { border: 2px solid #f0a500; }
.thumb { cursor: default; }
.thumb.clickable { cursor: pointer; }
figcaption { margin-top: 2px; color: #555; }
section { margin-bottom: 32px; }
.empty { color: #888; font-style: italic; }
.hist-row { display: flex; flex-wrap: wrap; gap: 16px; }
.hist-card { background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 8px 12px; }
.hist-card h4 { margin: 0 0 4px 0; font-size: 13px; }
.compare-panel { background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 12px; }
.compare-pair { display: flex; gap: 16px; flex-wrap: wrap; }
.compare-slot { flex: 1; min-width: 240px; }
.compare-slot img { width: 100%; border-radius: 4px; border: 1px solid #ccc; background: #000; }
.compare-slot .meta { font-size: 12px; color: #555; margin-top: 4px; white-space: pre-line; }
.compare-controls { margin-top: 10px; display: flex; gap: 8px; align-items: center; }
.compare-controls button { cursor: pointer; padding: 4px 10px; }
.compare-controls span { font-size: 12px; color: #555; }
@media (prefers-color-scheme: dark) {
  body { background: #17181a; color: #e6e6e6; }
  .summary-table td, .summary-table th { border-color: #333; }
  .thumb img { border-color: #444; }
  .thumb .placeholder { background: #222; border-color: #444; color: #999; }
  .warning { background: #332a10; border-color: #6b551c; color: #f0d68a; }
  .hist-card, .compare-panel { background: #1f2023; border-color: #333; }
  .compare-slot img { border-color: #444; }
}
"""


def _thumb_grid(frames: List[dict], clickable: bool = False) -> str:
    if not frames:
        return "<p class='empty'>None.</p>"
    items = []
    for i, f in enumerate(frames):
        caption = f"#{f['index']} t={f['timestamp_ms']:.0f}ms"
        if f.get("score") is not None:
            caption += f" score={f['score']:.2f}"
        if f.get("kept_reason") == "minimum_overlap_protection":
            caption += " [forced: min overlap]"
        if f.get("uncertain"):
            caption += " [uncertain: " + "; ".join(f.get("uncertainty_reasons", [])) + "]"
        src = f.get("output_filename")
        if src:
            body = f"<img src='{html.escape(src)}' loading='lazy' alt=''>"
        else:
            body = f"<div class='placeholder'>{html.escape(f.get('rejection_reason', 'rejected'))}</div>"
        thumb_class = "thumb uncertain" if f.get("uncertain") else "thumb"
        onclick = ""
        if clickable and src:
            thumb_class += " clickable"
            onclick = f" onclick='cmpShow({i})'"
        items.append(
            f"<figure class='{thumb_class}'{onclick}>{body}<figcaption>{html.escape(caption)}</figcaption></figure>"
        )
    return f"<div class='grid'>{''.join(items)}</div>"


def _score_timeline_svg(frames: List[dict], width: int = 900, height: int = 180) -> str:
    scored = [f for f in frames if f.get("score") is not None]
    if not scored:
        return "<p class='empty'>No score data.</p>"

    max_index = max(f["index"] for f in scored) or 1
    scores = [f["score"] for f in scored]
    min_s, max_s = min(scores), max(scores)
    span = (max_s - min_s) or 1.0

    def x_of(i: int) -> float:
        return 10 + (width - 20) * (i / max_index)

    def y_of(s: float) -> float:
        return height - 10 - (height - 20) * ((s - min_s) / span)

    points = []
    markers = []
    for f in scored:
        x, y = x_of(f["index"]), y_of(f["score"])
        points.append(f"{x:.1f},{y:.1f}")
        color = "#2e7d32" if f["kept"] else "#c62828"
        title = f"#{f['index']} score={f['score']:.2f} {'kept' if f['kept'] else f.get('rejection_reason')}"
        markers.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='2.5' fill='{color}' opacity='0.85'><title>{html.escape(title)}</title></circle>")

    polyline = f"<polyline fill='none' stroke='#90a4ae' stroke-width='1' points='{' '.join(points)}' />"
    return (
        f"<svg viewBox='0 0 {width} {height}' width='100%' height='{height}' role='img' aria-label='Frame score timeline'>"
        f"{polyline}{''.join(markers)}</svg>"
    )


def _histogram_svg(values: List[float], color: str, bins: int = 16, width: int = 260, height: int = 110) -> str:
    """Simple binned-count histogram over already-computed per-frame metrics
    -- no new signal, just a different view of values already in the report
    (score/sharpness/feature-richness/redundancy)."""
    if not values:
        return "<p class='empty'>No data.</p>"

    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    counts = [0] * bins
    for v in values:
        idx = min(bins - 1, int((v - lo) / span * bins))
        counts[idx] += 1
    max_count = max(counts) or 1

    pad = 4
    bar_w = (width - 2 * pad) / bins
    bars = []
    for i, c in enumerate(counts):
        bar_h = (height - 2 * pad) * (c / max_count)
        x = pad + i * bar_w
        y = height - pad - bar_h
        title = f"[{lo + i * span / bins:.3f}, {lo + (i + 1) * span / bins:.3f}): {c} frame(s)"
        bars.append(
            f"<rect x='{x:.1f}' y='{y:.1f}' width='{max(bar_w - 1, 0.5):.1f}' height='{bar_h:.1f}' "
            f"fill='{color}' opacity='0.85'><title>{html.escape(title)}</title></rect>"
        )
    return (
        f"<svg viewBox='0 0 {width} {height}' width='100%' height='{height}' role='img' aria-label='Histogram'>"
        f"{''.join(bars)}</svg>"
        f"<div style='display:flex;justify-content:space-between;font-size:10px;color:#888;'>"
        f"<span>{lo:.3f}</span><span>{hi:.3f}</span></div>"
    )


def _distributions_section(frames: List[dict]) -> str:
    """Phase 4.6: distribution views over metrics the pipeline already computes
    for every scored frame (kept and rejected) -- gives a sense of the whole
    video's quality/redundancy/connectivity spread, not just the kept set."""
    scored = [f for f in frames if f.get("score") is not None]
    if not scored:
        return "<p class='empty'>No score data (all frames rejected before scoring, e.g. by the hard blur floor).</p>"

    cards = [
        ("Score", "#5c6bc0", [f["score"] for f in scored]),
        ("Sharpness", "#26a69a", [f["sharpness"] for f in scored]),
        ("Feature richness (normalized)", "#8d6e63", [f["feature_richness_norm"] for f in scored]),
        ("Match strength to reference (redundancy, normalized)", "#ef6c00", [f["redundancy_norm"] for f in scored]),
    ]
    html_cards = []
    for title, color, values in cards:
        html_cards.append(
            f"<div class='hist-card'><h4>{html.escape(title)}</h4>{_histogram_svg(values, color)}</div>"
        )
    return f"<div class='hist-row'>{''.join(html_cards)}</div>"


def _compare_panel(kept_frames: List[dict]) -> str:
    """Phase 4.6: click any kept-frame thumbnail (or Prev/Next) to inspect it
    side by side with the previous kept frame -- purely a client-side viewer
    over data already embedded in the report, no server required."""
    if len(kept_frames) < 1:
        return "<p class='empty'>No kept frames to compare.</p>"

    import json
    payload = [
        {
            "index": f["index"],
            "timestamp_ms": f["timestamp_ms"],
            "src": f.get("output_filename"),
            "score": f.get("score"),
            "sharpness": f.get("sharpness"),
            "redundancy_norm": f.get("redundancy_norm"),
            "kept_reason": f.get("kept_reason"),
            "uncertain": f.get("uncertain"),
        }
        for f in kept_frames
    ]
    data_json = json.dumps(payload)

    return f"""
<div class="compare-panel">
  <div class="compare-pair">
    <div class="compare-slot">
      <div><strong>Previous kept frame</strong></div>
      <img id="cmp-prev-img" src="" alt="">
      <div class="meta" id="cmp-prev-meta"></div>
    </div>
    <div class="compare-slot">
      <div><strong>Selected frame</strong></div>
      <img id="cmp-cur-img" src="" alt="">
      <div class="meta" id="cmp-cur-meta"></div>
    </div>
  </div>
  <div class="compare-controls">
    <button type="button" onclick="cmpStep(-1)">&larr; Prev</button>
    <button type="button" onclick="cmpStep(1)">Next &rarr;</button>
    <span id="cmp-position"></span>
  </div>
</div>
<script id="kept-frames-data" type="application/json">{data_json}</script>
<script>
(function() {{
  var KEPT = JSON.parse(document.getElementById('kept-frames-data').textContent);
  var cur = 0;

  function fmtMeta(f) {{
    if (!f) return '(none -- first kept frame)';
    var lines = ['#' + f.index + '  t=' + Math.round(f.timestamp_ms) + 'ms'];
    if (f.score !== null && f.score !== undefined) lines.push('score=' + f.score.toFixed(3));
    if (f.sharpness !== null && f.sharpness !== undefined) lines.push('sharpness=' + f.sharpness.toFixed(1));
    if (f.redundancy_norm !== null && f.redundancy_norm !== undefined) lines.push('redundancy=' + f.redundancy_norm.toFixed(3));
    if (f.kept_reason === 'minimum_overlap_protection') lines.push('forced: minimum overlap protection');
    if (f.uncertain) lines.push('flagged uncertain');
    return lines.join('\\n');
  }}

  window.cmpShow = function(i) {{
    if (KEPT.length === 0) return;
    cur = Math.max(0, Math.min(KEPT.length - 1, i));
    var f = KEPT[cur];
    var prev = cur > 0 ? KEPT[cur - 1] : null;
    document.getElementById('cmp-cur-img').src = f.src || '';
    document.getElementById('cmp-cur-meta').textContent = fmtMeta(f);
    document.getElementById('cmp-prev-img').src = prev ? (prev.src || '') : '';
    document.getElementById('cmp-prev-meta').textContent = fmtMeta(prev);
    document.getElementById('cmp-position').textContent = (cur + 1) + ' / ' + KEPT.length + ' kept frames';
  }};

  window.cmpStep = function(delta) {{ window.cmpShow(cur + delta); }};

  window.cmpShow(0);
}})();
</script>
"""


def render_html_report(report: dict, video_info: VideoInfo, html_path: str, preview_exists: bool) -> None:
    summary = report["summary"]
    warnings = report.get("warnings", [])
    frames = report["frames"]
    kept_frames = [f for f in frames if f["kept"]]
    rejected_frames = [f for f in frames if not f["kept"]]
    uncertain_frames = [f for f in kept_frames if f.get("uncertain")]

    rows = "".join(f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>" for k, v in summary.items())
    warning_html = "".join(f"<div class='warning'>{html.escape(w)}</div>" for w in warnings)

    original_video_uri = "file:///" + os.path.abspath(video_info.path).replace("\\", "/")
    preview_block = (
        "<video controls src='optimized_preview.mp4'></video>" if preview_exists
        else "<p class='empty'>No preview generated (no frames selected).</p>"
    )

    html_doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Video-to-Frame Report</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Video-to-Frame Optimization Report</h1>
{warning_html}
<section>
<h2>Summary</h2>
<table class="summary-table">{rows}</table>
</section>
<section>
<h2>Original vs. Optimized Preview</h2>
<div class="videos">
  <div><h3>Original</h3><video controls src="{html.escape(original_video_uri)}"></video></div>
  <div><h3>Optimized Preview</h3>{preview_block}</div>
</div>
<p class="empty">If a video does not play, some browsers block local file:// playback across
directories &mdash; try serving this folder with <code>python -m http.server</code> and opening
report.html via http://localhost instead.</p>
</section>
<section>
<h2>Score Timeline</h2>
{_score_timeline_svg(frames)}
<p class="empty">Green = kept, red = rejected. Hover a point for details.</p>
</section>
<section>
<h2>Metric Distributions</h2>
<p class="empty">Over every scored frame (kept and rejected) -- shows the whole video's spread, not
just the kept set. "Match strength" is the redundancy signal used to build clusters (see the
selector's docstring); it is a connectivity proxy, not a measure of reconstruction quality.</p>
{_distributions_section(frames)}
</section>
<section>
<h2>Frame Inspector</h2>
<p class="empty">Click a kept-frame thumbnail below (or use Prev/Next) to compare it against the
previous kept frame.</p>
{_compare_panel(kept_frames)}
</section>
<section>
<h2>Uncertain Kept Frames ({len(uncertain_frames)})</h2>
<p class="empty">Flagged for manual review (weak connectivity to the previous kept frame, and/or
very low local texture) -- not rejected or altered, purely a visibility flag.</p>
{_thumb_grid(uncertain_frames)}
</section>
<section>
<h2>Kept Frames ({len(kept_frames)})</h2>
{_thumb_grid(kept_frames, clickable=True)}
</section>
<section>
<h2>Rejected Frames ({len(rejected_frames)})</h2>
{_thumb_grid(rejected_frames)}
</section>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_doc)
