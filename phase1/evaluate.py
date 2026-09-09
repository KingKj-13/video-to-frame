"""Phase 1 CLI: run the evaluation framework against one video.

    python -m phase1.evaluate INPUT.mp4 --output phase1_output/
    python -m phase1.evaluate INPUT.mp4 --output phase1_output/ --sweep
    python -m phase1.evaluate INPUT.mp4 --output phase1_output/ \\
        --reconstruction-command "colmap_wrapper.py --images {image_dir} --out {output_json}"

With no --reconstruction-command, reconstruction metrics are unavailable and
the report relies on the ORB feature-connectivity proxy -- see
phase1/evaluation/reconstruction_backend.py for the plug-in contract.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence

from video_to_frame.config import PipelineConfig

from .evaluation.benchmark_runner import run_benchmark
from .evaluation.reconstruction_backend import build_backend
from .evaluation.reduction_levels import DEFAULT_TARGET_LEVELS, evaluate_reduction_levels
from .reports.evaluation_report import build_evaluation_json, render_evaluation_html, write_evaluation_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phase1-evaluate",
        description="Evaluate whether the video-to-frame optimizer preserves reconstruction-relevant "
                    "information, against naive baselines and (optionally) an external reconstruction backend.",
    )
    parser.add_argument("input", help="Path to the input video")
    parser.add_argument("--output", "-o", required=True, help="Output directory for the evaluation run")
    parser.add_argument("--config", help="Path to a YAML PipelineConfig file for the intelligent selector")
    parser.add_argument("--no-baselines", action="store_true",
                         help="Skip the every-Nth-frame and blur-only baseline comparisons")
    parser.add_argument("--weak-link-threshold", type=int, default=15,
                         help="ORB match count below which a consecutive-frame pair is flagged as a weak link")
    parser.add_argument("--reconstruction-backend", choices=["none", "pycolmap"], default="none",
                         help="'pycolmap' runs a real local COLMAP build (pip install pycolmap); "
                              "'none' (default) leaves reconstruction metrics UNAVAILABLE")
    parser.add_argument("--reconstruction-command",
                         help="Shell command template for an external reconstruction backend "
                              "(placeholders: {image_dir}, {workdir}, {output_json}); ignored if "
                              "--reconstruction-backend pycolmap is set")
    parser.add_argument("--reconstruction-timeout", type=float, default=None,
                         help="Timeout in seconds for the reconstruction command")
    parser.add_argument("--sweep", action="store_true",
                         help="Also run the reduction-level sweep (default targets: "
                              f"{DEFAULT_TARGET_LEVELS}) for the tradeoff-curve analysis")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    config = PipelineConfig.from_yaml(args.config)
    backend = build_backend(
        args.reconstruction_command, timeout_sec=args.reconstruction_timeout,
        use_pycolmap=(args.reconstruction_backend == "pycolmap"),
    )

    try:
        benchmark_result = run_benchmark(
            args.input, args.output, config=config, backend=backend,
            weak_link_threshold=args.weak_link_threshold,
            include_baselines=not args.no_baselines,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    reduction_levels = None
    if args.sweep:
        sweep_dir = os.path.join(args.output, "reduction_sweep")
        results = evaluate_reduction_levels(args.input, sweep_dir, base_config=config,
                                             weak_link_threshold=args.weak_link_threshold)
        reduction_levels = [r.to_dict() for r in results]

    evaluation = build_evaluation_json(benchmark_result, reduction_levels=reduction_levels)
    write_evaluation_json(evaluation, os.path.join(args.output, "evaluation.json"))
    render_evaluation_html(evaluation, os.path.join(args.output, "evaluation.html"))

    cp = evaluation.get("connectivity_proxy", {})
    recon = evaluation.get("reconstruction", {})
    print("\n--- Phase 1 Evaluation ---")
    for key in ("video", "original_frames", "selected_frames", "reduction_percentage",
                "dataset_size_mb", "blur_rejected", "redundant_rejected",
                "frames_forced_by_minimum_overlap", "failure_verdict"):
        print(f"{key}: {evaluation.get(key)}")
    print(f"average_feature_matches (connectivity proxy): {cp.get('average_feature_matches')}")
    print(f"minimum_feature_matches (connectivity proxy): {cp.get('minimum_feature_matches')}")
    print(f"weak_link_count (connectivity proxy): {cp.get('weak_link_count')}")
    print(f"low_texture_pair_count: {cp.get('low_texture_pair_count')}")
    print(f"reconstruction: {recon.get('status')}"
          + (f" ({recon.get('backend')})" if recon.get("status") == "COMPLETE" else ""))
    print(f"\nFull report: {os.path.join(args.output, 'evaluation.json')}")
    print(f"HTML report: {os.path.join(args.output, 'evaluation.html')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
