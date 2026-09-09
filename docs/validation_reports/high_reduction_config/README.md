# Validated high-reduction configuration

**Not a change to any default.** The selector never targets a fixed
reduction percentage by design (see the main README's "How selection
works") -- the numbers below are a validated *recipe* for footage where
more reduction is wanted, achieved entirely through levers that already
existed (the `aggressive` preset plus a footage-appropriate blur floor),
verified with the same real-ORB connectivity check used everywhere else in
this project. Nothing in the selector's code changed.

```bash
video-to-frame --preset aggressive --hard-blur-floor 50 INPUT.mp4 --output OUT_DIR
```

## Why these two levers, and not others

- `--preset aggressive` lowers `redundancy_threshold` (0.82 -> 0.65) and
  raises `coverage_gain_threshold` (0.15 -> 0.2) -- classifies more frames
  as genuinely redundant, without touching `min_overlap_threshold` (the
  safety floor stays at 0.3, same as `balanced`).
- `--hard-blur-floor 50` raises the *unconditional* blur rejection floor
  from the default 30. This was chosen empirically for this footage: the
  default-run drone video's kept frames had a median sharpness of only 78.4
  (inherent FPV motion blur, not a defect), so the floor was raised only
  as far as testing showed was safe, not to an arbitrary round number.

Raising the blur floor turned out to help speed as well as reduction:
blur-rejected frames skip the entire ORB pipeline (rejected before ever
reaching the selector), while the aggressive preset's extra reduction
actually costs *some* time back (larger redundancy clusters need more
minimum-overlap safety checks per flush before dropping non-winners) -- the
combination nets out favorably because more frames now exit cheaply via the
blur floor than the extra safety-check overhead adds back.

## Results (real footage, full videos, current code)

| Video | Default (`balanced`) | This config | Real connectivity (this config) |
|---|---|---|---|
| Drone (2011 frames) | 1642 kept, 18.35% reduction, ~8.8 min | **718 kept, 64.3% reduction, 6.2 min** (5.36 fps) | 0 weak links, 0 low-texture pairs, min 47 / avg 304 matches per consecutive pair |
| Handheld (240 frames) | 158 kept, 34.17% reduction, ~1 min | **91 kept, 62.1% reduction, 60.3s** (4.0 fps) | 0 weak links, 0 low-texture pairs, min 94 / avg 326 matches per consecutive pair |

Connectivity was measured directly with real ORB matching between every
consecutive pair of *actually kept* frames (`phase1.evaluation.connectivity_metrics`),
not inferred from the minimum-overlap safety counter (which was also 0 on
both videos, but doesn't by itself prove connectivity -- see the note
below).

Full reports: [drone/report.json](drone/report.json), [drone/report.html](drone/report.html), [drone/config.json](drone/config.json);
[handheld/report.json](handheld/report.json), [handheld/report.html](handheld/report.html), [handheld/config.json](handheld/config.json).

## A structural note worth knowing before raising the blur floor further

The minimum-overlap safety net only ever checks candidates the *redundancy
clustering* would otherwise reject -- frames rejected by the hard blur
floor are removed upstream, before they ever reach the selector, so the
safety net never sees them and can't catch a connectivity gap the blur
floor itself creates. On both real videos here, the actual (not just
safety-net-inferred) connectivity came back clean at floor=50, but this
was verified empirically on these two videos specifically, not derived
from a guarantee in the code. Raising the floor further, or applying this
to footage with longer continuous blurry stretches (e.g. rapid banking
turns), should be re-verified the same way -- run the video, then check
`phase1.evaluation.connectivity_metrics.compute_connectivity_metrics` (or
`video-to-frame evaluate`) against the actual kept-frame set rather than
assuming it still holds.

## As a preset, if this becomes a common need

If more footage validates the same recipe, it would be worth adding a
named preset (e.g. `high_reduction`) rather than asking every caller to
remember `--preset aggressive --hard-blur-floor 50`. Not done yet since two
videos isn't enough evidence to promote it to a first-class, permanently
supported preset -- see `src/video_to_frame/presets.py` for where it would
go.
