# Validation run report

Full pipeline run against both real videos in `input/`, on the current
(fully restructured/productized) code. Deterministic -- re-running produces
identical frame counts every time; only wall-clock runtime varies with
system load.

## FPV Drone Flight (`FPV Drone Flight through Beautiful Iceland Canyon_0_67.mp4`)

| Metric | Value |
|---|---|
| Input -> Output frames | 2011 -> **1642** |
| Reduction | **18.35%** |
| Runtime | 527.3s (~8.8 min) on a quiet system, 3.81 fps |
| Rejected -- blur | 70 |
| Rejected -- redundant | 299 |
| Forced by minimum-overlap safety net | 0 |
| Flagged uncertain (review-only, not auto-acted-on) | 6 (weak connectivity to previous kept frame) |
| Output resolution | 640x360 (original), lossless PNG |

Full detail: [drone/report.json](drone/report.json), [drone/report.html](drone/report.html), [drone/config.json](drone/config.json).

## Handheld video (`create_a_video_liek_taken_from.mp4`)

| Metric | Value |
|---|---|
| Input -> Output frames | 240 -> **158** |
| Reduction | **34.17%** |
| Runtime | 60.1s (~1 min), 4.0 fps |
| Rejected -- blur | 41 |
| Rejected -- redundant | 41 |
| Forced by minimum-overlap safety net | 0 |
| Flagged uncertain (review-only, not auto-acted-on) | 1 (weak connectivity to previous kept frame) |
| Output resolution | 1280x720 (original), lossless PNG |

Full detail: [handheld/report.json](handheld/report.json), [handheld/report.html](handheld/report.html), [handheld/config.json](handheld/config.json).

## Reduction percentage is a result, not a target

Both videos were processed with default (`balanced`) settings. The reduction
percentage differs sharply between them (18.35% vs 34.17%) because it
reflects how much genuine redundancy exists in each video's footage, not a
fixed target -- the drone flyover has continuously changing viewpoints
(little true redundancy), while the handheld clip holds on some viewpoints
longer (more true redundancy). This is a deliberate design choice: the
selector is never tuned toward a specific reduction number, since doing so
would mean discarding real information on low-redundancy footage just to
hit a target.

The `aggressive` preset exists for cases where more reduction is wanted
without weakening the minimum-overlap safety floor; see the README's
Presets section.

## Higher-reduction configuration

A validated recipe (`--preset aggressive --hard-blur-floor 50`) achieves
64.3% reduction on the drone video and 62.1% on the handheld video, in
6.2 min and 60s respectively, with 0 weak links / 0 low-texture pairs on
both (real ORB connectivity, not just the safety-net counter). See
[high_reduction_config/README.md](high_reduction_config/README.md) for the
full evidence, and why this is a documented recipe rather than a new
default.

## Note on this specific run's frames/preview output

The full-resolution `frames/` directories and `optimized_preview.mp4` files
these reports reference are NOT committed to this repo (regenerable,
several hundred MB each -- see `.gitignore`). Re-run
`video-to-frame batch input --output output` to regenerate them locally;
the frame counts and reduction percentages above will match exactly
(deterministic).
