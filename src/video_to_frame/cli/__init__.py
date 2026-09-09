from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

from ..api import process_video
from ..batch import print_batch_table, process_batch
from ..config import PipelineConfig
from ..presets import PRESETS, apply_preset
from ..utils import VIDEO_EXTENSIONS

DEFAULT_INPUT_DIR = "input"


def _configure_logging(verbose: bool, quiet: bool) -> None:
    """Phase 4.11: ERROR/WARNING/INFO/DEBUG via the stdlib logging module,
    scoped to this package so it doesn't affect a caller's own logging
    config when video_to_frame is used as a library. --verbose exposes
    per-frame selection reasoning (see selection/selector.py's debug logs);
    default (INFO) stays quiet during normal processing per the spec."""
    level = logging.DEBUG if verbose else (logging.WARNING if quiet else logging.INFO)
    pkg_logger = logging.getLogger("video_to_frame")
    pkg_logger.setLevel(level)
    if not pkg_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        pkg_logger.addHandler(handler)
DEFAULT_OUTPUT_DIR = "output"
SUBCOMMANDS = {"process", "batch", "inspect", "evaluate"}


def _add_pipeline_arguments(parser: argparse.ArgumentParser) -> None:
    """Config/threshold flags shared by any subcommand that runs the pipeline."""
    parser.add_argument("--config", help="Path to a YAML config file (overrides built-in defaults)")
    parser.add_argument(
        "--preset", choices=sorted(PRESETS), default=None,
        help="Named selection-aggressiveness preset (applied after --config, before individual "
             "threshold flags, so explicit flags always win): reconstruction_safe (most "
             "conservative), balanced (= built-in defaults), aggressive (more reduction, safety "
             "floor unchanged), quality_first (minimal selection, maximum preservation)",
    )
    parser.add_argument("--hard-blur-floor", type=float, dest="hard_blur_floor",
                         help="Laplacian-variance floor below which a frame is unconditionally rejected")
    parser.add_argument("--blur-threshold", type=float, dest="blur_threshold",
                         help="Laplacian-variance threshold below which a frame's score is penalized")
    parser.add_argument("--redundancy-threshold", type=float, dest="redundancy_threshold")
    parser.add_argument("--coverage-gain-threshold", type=float, dest="coverage_gain_threshold")
    parser.add_argument(
        "--min-overlap-threshold", type=float, dest="min_overlap_threshold",
        help="Minimum ORB feature-overlap a candidate must have with the previous kept frame "
             "to be safely dropped; below this it is force-kept to protect SfM connectivity "
             "(default: 0.30 -- separate from, and must stay below, --redundancy-threshold)",
    )
    parser.add_argument(
        "--analysis-scale", type=float, dest="analysis_scale",
        help="Run ORB feature extraction/matching on a scaled-down copy for speed, e.g. 0.75 or "
             "0.5 (default: disabled, full resolution). Never affects blur detection or output "
             "frames. 0.25 measured too aggressive -- prefer 0.5-0.75.",
    )
    parser.add_argument("--use-sift", action="store_true", default=None, dest="use_sift",
                         help="Use SIFT instead of ORB for feature extraction")
    parser.add_argument("--min-frames", type=int, dest="min_frames")
    parser.add_argument("--max-frames", type=int, dest="max_frames")
    parser.add_argument("--resize", type=int, dest="resize_long_edge",
                         help="Resize the long edge of kept frames to N pixels "
                              "(default: disabled -- frames are kept at original resolution)")
    parser.add_argument("--quality", type=int, dest="output_quality",
                         help="Encode quality 1-100 (only applies to webp/jpg, not png)")
    parser.add_argument("--format", dest="output_format", choices=["webp", "jpg", "jpeg", "png"],
                         help="Output image format (default: png, lossless -- pass webp/jpg for a "
                              "smaller, lossy 'compact mode')")
    parser.add_argument("--png-compression-level", type=int, dest="png_compression_level",
                         help="PNG zlib effort 0-9 (default: 1, fast -- PNG is lossless at any "
                              "level, this only trades encode speed for file size)")
    parser.add_argument("--preview-fps", type=float, dest="preview_fps")
    parser.add_argument("--no-progress", action="store_true", help="Disable the progress bar")
    parser.add_argument("--verbose", action="store_true", help="Print per-frame diagnostic detail")
    parser.add_argument("--quiet", action="store_true", help="Suppress the summary printout")


def _build_config(args: argparse.Namespace) -> PipelineConfig:
    config = PipelineConfig.from_yaml(args.config)
    if args.preset:
        config = apply_preset(args.preset, base=config)
    # Inclusion-based, not an exclusion blocklist: only pass through args
    # that are actually PipelineConfig fields. A blocklist silently breaks
    # every time a new CLI-only argument (a positional, --output, --verbose,
    # a future subcommand's own flag) is added elsewhere and someone forgets
    # to list it here -- this can't go stale the same way.
    config_fields = {f.name for f in dataclasses.fields(PipelineConfig)}
    overrides = {k: v for k, v in vars(args).items() if k in config_fields and v is not None}
    return config.apply_overrides(**overrides)


def resolve_input_video(path_arg: Optional[str]) -> str:
    """Resolves the CLI's input argument to a concrete video file path.

    If omitted, or if given a directory, looks for exactly one video file in
    it -- this is what lets a user just drop a video into ./input/ and run
    the tool with no arguments.
    """
    candidate = Path(path_arg) if path_arg else Path(DEFAULT_INPUT_DIR)

    if candidate.is_dir():
        videos = sorted(
            p for p in candidate.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        )
        if not videos:
            raise FileNotFoundError(
                f"No video found in '{candidate}' (expected one of {sorted(VIDEO_EXTENSIONS)})."
            )
        if len(videos) > 1:
            names = ", ".join(v.name for v in videos)
            raise ValueError(
                f"Multiple videos found in '{candidate}': {names}. "
                "Pass the one you want explicitly as the input path."
            )
        return str(videos[0])

    if candidate.is_file():
        return str(candidate)

    raise FileNotFoundError(f"Input video not found: {candidate}")


def _run_process(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="video-to-frame process",
        description="Reduce a video to a minimal, information-preserving frame set "
                    "for a downstream 3D reconstruction model. Does not do 3D reconstruction itself.",
    )
    parser.add_argument(
        "input", nargs="?", default=None,
        help=f"Path to the input video, or a directory to auto-detect one in "
             f"(default: ./{DEFAULT_INPUT_DIR}/ -- drop a single video there and omit this argument)",
    )
    parser.add_argument("--output", "-o", default=None, help=f"Output directory (default: ./{DEFAULT_OUTPUT_DIR}/)")
    _add_pipeline_arguments(parser)
    args = parser.parse_args(argv)
    _configure_logging(args.verbose, args.quiet)

    config = _build_config(args)

    try:
        input_path = resolve_input_video(args.input)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    output_dir = args.output or DEFAULT_OUTPUT_DIR

    try:
        result = process_video(input_path, output_dir, config, show_progress=not args.no_progress)
    except Exception as exc:  # surfaced as a clean CLI error, not a traceback
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        print("\n--- Video-to-Frame Report ---")
        for key, value in result.raw_report["summary"].items():
            print(f"{key}: {value}")
        for warning in result.warnings:
            print(f"\nWarning: {warning}")
        if args.verbose:
            print(f"\nPer-frame detail: {len(result.raw_report['frames'])} records in {result.report_path}")
        print(f"\nOutput written to: {output_dir}")
    return 0


def _run_batch(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="video-to-frame batch",
        description="Process every video in a directory; one failed video does not stop the rest.",
    )
    parser.add_argument("input_dir", help="Directory containing videos to process")
    parser.add_argument("--output", "-o", required=True, help="Root output directory (one subfolder per video)")
    _add_pipeline_arguments(parser)
    args = parser.parse_args(argv)
    _configure_logging(args.verbose, args.quiet)

    config = _build_config(args)

    try:
        results = process_batch(args.input_dir, args.output, config, show_progress=not args.no_progress)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        print()
        print_batch_table(results)
        print(f"\nBatch summary written to: {os.path.join(args.output, 'batch_summary.json')}")
    return 0 if all(r.success for r in results) else 1


def _run_inspect(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="video-to-frame inspect",
        description="Print a summary of an existing output directory's report.json.",
    )
    parser.add_argument("output_dir", help="An output directory previously produced by 'process'")
    args = parser.parse_args(argv)

    report_path = os.path.join(args.output_dir, "report.json")
    if not os.path.isfile(report_path):
        print(f"Error: no report.json found in {args.output_dir}", file=sys.stderr)
        return 1

    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)

    print(f"--- {args.output_dir} ---")
    for key, value in report["summary"].items():
        print(f"{key}: {value}")
    for warning in report.get("warnings", []):
        print(f"\nWarning: {warning}")
    return 0


def _run_evaluate(argv: Sequence[str]) -> int:
    """Thin delegator to the Phase 1 evaluation framework (phase1.evaluate) --
    kept separate from production `process`/`batch` since it pulls in
    heavier, research-oriented tooling (connectivity proxies, optional
    reconstruction backends), not duplicated here."""
    try:
        from phase1.evaluate import main as evaluate_main
    except ImportError as exc:
        print(f"Error: the 'evaluate' command requires the phase1 evaluation package: {exc}", file=sys.stderr)
        return 1
    return evaluate_main(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(argv) if argv is not None else sys.argv[1:]

    # Backward compatible: `video-to-frame INPUT.mp4 --output OUT` (no
    # subcommand named) is treated as `process INPUT.mp4 --output OUT`.
    if argv and argv[0] in SUBCOMMANDS:
        command, rest = argv[0], argv[1:]
    else:
        command, rest = "process", argv

    if command == "process":
        return _run_process(rest)
    if command == "batch":
        return _run_batch(rest)
    if command == "inspect":
        return _run_inspect(rest)
    if command == "evaluate":
        return _run_evaluate(rest)
    raise AssertionError(f"unreachable: unknown command {command!r}")  # SUBCOMMANDS kept in sync above


if __name__ == "__main__":
    raise SystemExit(main())
