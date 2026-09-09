# Video-to-Frame Optimization System

Turns a video into a minimal, information-preserving set of frames for a
**downstream 3D reconstruction model**. This system does not do any 3D
reconstruction itself -- it only decides which frames are worth keeping.

## Install

```bash
pip install -e ".[dev]"
```

Requires `ffmpeg` on PATH (used only to mux the final preview video).

## Usage

Drop a single video into `input/` and run with no arguments -- results land in `output/`:

```bash
python -m video_to_frame
# or, once the console script is on PATH:
video-to-frame
```

You can also pass paths explicitly:

```bash
video-to-frame INPUT.mp4 --output OUT_DIR
```

Useful overrides (see `config/default.yaml` for the full list):

```bash
video-to-frame --config my_config.yaml \
  --blur-threshold 80 \
  --max-frames 800 \
  --min-overlap-threshold 0.35
```

By default, output is full-resolution lossless PNG (quality-first, for
reconstruction). Pass `--format webp` or `--format jpg` (optionally with
`--resize N` and `--quality N`) for a smaller, lossy "compact mode":

```bash
video-to-frame --format webp --resize 1920 --quality 85
```

### Presets

Named selection-aggressiveness presets (`--preset <name>`, applied before
individual threshold flags so those still win):

| Preset | Behavior |
|---|---|
| `reconstruction_safe` | Most conservative -- prioritizes information preservation |
| `balanced` | Same as the built-in defaults |
| `aggressive` | More reduction; the minimum-overlap safety floor is never weakened |
| `quality_first` | Minimal selection, maximum preservation |

```bash
video-to-frame --preset quality_first
```

For footage where more reduction is wanted, `--preset aggressive --hard-blur-floor <N>`
(with `<N>` chosen for your footage's actual sharpness distribution, not
guessed) is a validated combination -- see
[docs/validation_reports/high_reduction_config/](docs/validation_reports/high_reduction_config/)
for real-footage evidence (64.3%/62.1% reduction, 0 weak links) and how to
verify it holds on your own footage before relying on it.

### Subcommands

`process` (the default when no subcommand is named -- `video-to-frame INPUT.mp4 --output OUT`
is shorthand for `video-to-frame process INPUT.mp4 --output OUT`), plus:

```bash
video-to-frame batch videos/ --output results/     # process every video in a folder;
                                                     # one failure doesn't stop the rest
video-to-frame inspect results/some_video/         # print an existing run's report.json summary
video-to-frame evaluate INPUT.mp4 --output eval/   # Phase 1 evaluation framework (baselines,
                                                     # connectivity proxy, optional reconstruction backend)
```

### Programmatic API

```python
from video_to_frame.api import process_video
from video_to_frame.presets import apply_preset

config = apply_preset("quality_first")
result = process_video("video.mp4", "output/", config)
result.input_frames, result.output_frames, result.reduction_ratio
result.rejected_blur, result.rejected_redundant, result.forced_keeps
result.runtime_sec, result.report_path
```

## Output

```
OUT_DIR/
├── frames/frame_000123.png    # selected frames, original order preserved in filenames
├── optimized_preview.mp4      # selected frames only, in temporal order
├── report.json                # full stats + per-frame decisions (see below)
├── report.html                # self-contained visual comparison report
└── config.json                # the exact effective config used for this run
```

(`batch` additionally writes `batch_summary.json` at the root of its output directory.)

Open `report.html` in a browser for a summary table, a kept/rejected score
timeline, per-metric distribution histograms (score, sharpness, feature
richness, redundancy/match-strength), a click-to-compare frame inspector
(any kept-frame thumbnail vs. the previous kept frame, with Prev/Next),
thumbnail grids, and side-by-side original vs. optimized-preview video
players.

> Some browsers restrict local `file://` video playback across directories.
> If videos don't play in `report.html`, serve the folder instead:
> `python -m http.server` from inside `OUT_DIR`, then open
> `http://localhost:8000/report.html`.

## How selection works

1. **Extraction** -- frames are decoded in order with timestamps (`cv2.VideoCapture`).
2. **Quality** -- Laplacian-variance sharpness (+ brightness/contrast). A hard
   floor rejects unusable frames outright; a softer threshold only penalizes
   score, so a blurry frame can still win if it's the only frame covering its
   viewpoint.
3. **Redundancy clustering** -- consecutive frames that are highly similar
   (perceptual hash + SSIM + ORB feature overlap, compared against the open
   cluster's anchor and a rolling window of recently-kept frames) are grouped;
   only the single best-scoring frame per cluster is kept.
4. **Coverage** -- a growing pool of ORB descriptors from every kept frame
   detects when a frame reintroduces information not captured anywhere yet
   (including revisits of an earlier viewpoint after the camera moved away),
   overriding a redundancy classification.
5. **Scoring** -- `score = w_q*quality + w_s*sharpness + w_f*features + w_v*viewpoint_change + w_c*coverage_gain - w_r*redundancy`,
   weights configurable in `config/default.yaml`.
6. **Minimum-overlap protection** -- a separate safety net from redundancy
   clustering. A frame that clustering would reject is instead force-kept if
   its ORB feature-overlap with the *previous kept frame* falls below
   `min_overlap_threshold` (default 0.30) -- this guards against two kept
   frames being too disconnected for a downstream SfM feature matcher to link,
   which redundancy/coverage logic alone doesn't check for. It never overrides
   the hard blur floor. Forced keeps are recorded per-frame as
   `"kept_reason": "minimum_overlap_protection"` and aggregated in
   `report.json` as `frames_forced_by_minimum_overlap`.
7. **Compression** -- applied only to the final selected frames, independent
   of selection, so you can tell whether quality loss comes from frame
   removal or from image compression. Defaults to full-resolution lossless
   PNG; pass `--format webp`/`--format jpg` and/or `--resize` for a smaller,
   lossy output.
8. **Uncertainty flagging** (Phase 2/3.7) -- a post-selection, reporting-only
   pass that flags a kept frame for any of three conditions: weak ORB
   connectivity to the *previous kept frame*, very low own keypoint count
   ("low texture"), or brightness/contrast outside the same exposure
   thresholds the scorer already uses ("exposure anomaly" -- surfaces an
   already-computed penalty rather than adding a new signal). This never
   changes what was kept or rejected -- the connectivity check exists because
   minimum-overlap protection only checks candidates a cluster would
   otherwise *reject*; the winner of a newly-started cluster is never
   checked, so a long run of correctly-rejected redundant/blurry frames can
   still leave the next kept frame weakly connected to the previous one
   without anything noticing. Flagged frames are marked `"uncertain": true`
   with `"uncertainty_reasons"` in `report.json` and outlined in
   `report.html`; `frames_forced_by_minimum_overlap` has a sibling summary
   field `uncertain_kept_frames`. Two other candidate signals ("inconsistent
   matching", "unusual scene behavior") were evaluated and deliberately not
   added -- neither has a concrete, evidence-backed definition yet.

## Known limitations

- `min_overlap_threshold` (default 0.30) is a conservative starting point, not
  a validated-optimal value -- it hasn't been tuned against actual downstream
  reconstruction quality, only against the logic it's meant to guard.
- `max_frames` is enforced by dropping the lowest-scoring kept frames after
  selection; `min_frames` is informational only -- frames that don't exist
  can't be invented, and the report emits a warning if the target isn't met.
- Redundancy/coverage detection is classical CV (ORB/SIFT + perceptual hash +
  SSIM), not a learned embedding -- the `features/`, `coverage/`, and
  `selection/` packages are structured so a future phase can swap in
  something like SuperPoint, NetVLAD, or real SfM-based coverage estimation
  without touching the rest of the pipeline.
- `CAP_PROP_FRAME_COUNT` (used only for the progress bar) can be an estimate
  for some containers; the report's `original_frame_count` is the actual
  number of frames decoded, not the container's metadata.

## Package layout

For anyone integrating this into another codebase, import from the canonical
locations below (old top-level paths like `video_to_frame.models` and
`video_to_frame.pipeline` still work -- they're kept as thin backward-compatible
shims -- but new code should use these):

| Package | Contents |
|---|---|
| `video_to_frame.core` | `FrameRecord`/`FrameMetrics`/`QualityMetrics`/`KeptReason`/`RejectionReason` (`.models`), `run_pipeline` (`.pipeline`) |
| `video_to_frame.quality` | Laplacian-variance sharpness, brightness/contrast |
| `video_to_frame.features` | ORB/SIFT extraction + matching, dHash/SSIM similarity |
| `video_to_frame.coverage` | The global descriptor `CoveragePool` |
| `video_to_frame.scoring` | `FrameScorer`, the weighted-score formula |
| `video_to_frame.selection` | `StreamingSelector` (clustering + minimum-overlap protection), uncertainty flagging |
| `video_to_frame.output` | Resize + encode kept frames |
| `video_to_frame.reporting` | `report.json` and `report.html` generation |
| `video_to_frame.reconstruction` | Pluggable reconstruction backends (`NullBackend`, `ExternalCommandBackend`, `PycolmapBackend`) -- evaluation-only, not used by `run_pipeline` |
| `video_to_frame.api` / `.batch` / `.cli` / `.presets` / `.config` | Public entry points |

The `phase1/` package (evaluation framework -- baselines, connectivity-proxy
metrics, reduction sweeps) is intentionally kept as a separate top-level
package, not a `video_to_frame` subpackage: it pulls in heavier,
research-only dependencies (e.g. `pycolmap`) that production callers of
`video_to_frame` should never be forced to install.

## Tests

```bash
pytest tests/
```

Includes unit tests for quality/similarity/selector logic and an end-to-end
smoke test that generates a synthetic video (moving shape, a still/redundant
stretch, and an intentionally blurred stretch) and runs the full pipeline
on it.
