"""Pluggable interface to an external 3D reconstruction system.

This project must never perform 3D reconstruction itself -- every backend
here treats the reconstruction tool as a black box: hand it a directory of
images, get back whatever metrics it's able to report. No metric is invented
for a backend that doesn't provide it (every ReconstructionMetrics field is
Optional and left None when unavailable).

PycolmapBackend wraps a real, locally-installed COLMAP build (via the
`pycolmap` PyPI wheel, which ships a full COLMAP binary -- no separate
install needed) and has been smoke-tested end-to-end: feature extraction,
matching, and incremental mapping all run and produce real registration/
point-count/reprojection-error numbers. NullBackend remains the default
when no backend is requested; ExternalCommandBackend is available for any
other CLI reconstruction tool but is untested against a real one.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ReconstructionMetrics:
    """Metrics reported by a downstream reconstruction backend for one run.

    Only fields the backend actually reports should be populated -- see
    Phase 1 spec section 3/6: "Do not invent metrics that the reconstruction
    backend cannot actually provide."
    """
    backend_name: str
    success: bool
    input_frame_count: Optional[int] = None
    registered_frame_count: Optional[int] = None
    registration_rate: Optional[float] = None
    disconnected_components: Optional[int] = None
    point_count: Optional[int] = None
    mean_reprojection_error: Optional[float] = None
    reconstructed_area: Optional[float] = None
    processing_time_sec: Optional[float] = None
    raw_output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class ReconstructionBackend(ABC):
    """An external 3D reconstruction system, configured by the caller and
    invoked as a black box."""

    name: str = "unknown"

    @abstractmethod
    def run(self, image_dir: str, workdir: str) -> ReconstructionMetrics:
        """Runs reconstruction over every image in `image_dir`, using
        `workdir` for the backend's own scratch/output files, and returns
        whatever metrics it can report."""
        raise NotImplementedError


class NullBackend(ReconstructionBackend):
    """No reconstruction backend configured.

    Lets every other piece of the evaluation framework (baseline comparison,
    connectivity metrics, reduction-level sweeps, report generation) run and
    be tested even when no external reconstruction tool is installed --
    registration/point-cloud metrics simply come back unavailable instead of
    the framework failing outright.
    """

    name = "none"

    def run(self, image_dir: str, workdir: str) -> ReconstructionMetrics:
        return ReconstructionMetrics(
            backend_name=self.name,
            success=False,
            error="No reconstruction backend configured -- pass --reconstruction-command "
                  "to plug one in, or use connectivity metrics as a proxy.",
        )


class ExternalCommandBackend(ReconstructionBackend):
    """Generic adapter for any command-line reconstruction tool.

    Runs a user-supplied shell command against the image directory, then
    reads a JSON metrics file the command is expected to have written. This
    is the intended integration point for COLMAP, Meshroom, RealityCapture's
    CLI, or an in-house pipeline -- this package never needs to know
    anything backend-specific beyond the invocation and the output schema.

    `command_template` is a string with `{image_dir}`, `{workdir}`, and
    `{output_json}` placeholders, e.g. for a hypothetical COLMAP wrapper
    script that writes the expected JSON itself:

        "python colmap_wrapper.py --images {image_dir} --workspace {workdir} "
        "--out {output_json}"

    The command must write a JSON object to `{output_json}` with any subset
    of these keys: input_frame_count, registered_frame_count,
    registration_rate, disconnected_components, point_count,
    mean_reprojection_error, reconstructed_area. Unrecognized keys are kept
    under `raw_output` rather than discarded.
    """

    name = "external_command"

    _KNOWN_KEYS = (
        "input_frame_count", "registered_frame_count", "registration_rate",
        "disconnected_components", "point_count", "mean_reprojection_error",
        "reconstructed_area",
    )

    def __init__(self, command_template: str, timeout_sec: Optional[float] = None, name: Optional[str] = None):
        self.command_template = command_template
        self.timeout_sec = timeout_sec
        if name:
            self.name = name

    def run(self, image_dir: str, workdir: str) -> ReconstructionMetrics:
        os.makedirs(workdir, exist_ok=True)
        output_json = os.path.join(workdir, "reconstruction_metrics.json")
        command = self.command_template.format(
            image_dir=image_dir, workdir=workdir, output_json=output_json,
        )

        start = time.time()
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=self.timeout_sec)
        except subprocess.TimeoutExpired:
            return ReconstructionMetrics(
                backend_name=self.name, success=False,
                processing_time_sec=self.timeout_sec,
                error=f"Reconstruction command timed out after {self.timeout_sec}s",
            )
        elapsed = time.time() - start

        if result.returncode != 0:
            return ReconstructionMetrics(
                backend_name=self.name, success=False, processing_time_sec=elapsed,
                error=f"Command exited {result.returncode}: {result.stderr[-2000:]}",
            )

        if not os.path.isfile(output_json):
            return ReconstructionMetrics(
                backend_name=self.name, success=False, processing_time_sec=elapsed,
                error=f"Command completed but did not write the expected output file: {output_json}",
            )

        with open(output_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        known = {k: data.get(k) for k in self._KNOWN_KEYS if k in data}
        return ReconstructionMetrics(
            backend_name=self.name, success=True, processing_time_sec=elapsed,
            raw_output=data, **known,
        )


class PycolmapBackend(ReconstructionBackend):
    """Real local COLMAP, via the `pycolmap` wheel (bundles a full COLMAP
    build -- `pip install pycolmap`, no separate install). Runs SIFT feature
    extraction, sequential or exhaustive matching, then incremental SfM
    mapping, and reports whatever the resulting Reconstruction(s) expose.

    COLMAP can return *multiple* Reconstruction objects from one mapping
    call when the image set doesn't fully connect -- each one is a separate
    connected component. That count is reported directly as
    disconnected_components (1 = fully connected), and registered_frame_count
    sums registrations across all of them.
    """

    name = "pycolmap"

    def __init__(self, matching: str = "sequential", camera_model: str = "SIMPLE_RADIAL", verbose: bool = False):
        if matching not in ("sequential", "exhaustive"):
            raise ValueError("matching must be 'sequential' or 'exhaustive'")
        self.matching = matching
        self.camera_model = camera_model
        self.verbose = verbose

    def run(self, image_dir: str, workdir: str) -> ReconstructionMetrics:
        import time

        if not self.verbose:
            os.environ.setdefault("GLOG_minloglevel", "2")

        import pycolmap

        os.makedirs(workdir, exist_ok=True)
        db_path = os.path.join(workdir, "database.db")
        if os.path.isfile(db_path):
            os.remove(db_path)

        image_extensions = (".png", ".jpg", ".jpeg", ".webp")
        input_count = len([f for f in os.listdir(image_dir) if f.lower().endswith(image_extensions)])
        if input_count == 0:
            return ReconstructionMetrics(backend_name=self.name, success=False,
                                          error=f"No images found in {image_dir}")

        start = time.time()
        try:
            reader_options = pycolmap.ImageReaderOptions()
            reader_options.camera_model = self.camera_model
            pycolmap.extract_features(db_path, image_dir, reader_options=reader_options)

            if self.matching == "sequential":
                pycolmap.match_sequential(db_path)
            else:
                pycolmap.match_exhaustive(db_path)

            reconstructions = pycolmap.incremental_mapping(db_path, image_dir, workdir)
        except Exception as exc:
            return ReconstructionMetrics(
                backend_name=self.name, success=False, processing_time_sec=time.time() - start,
                input_frame_count=input_count, error=f"{type(exc).__name__}: {exc}",
            )
        elapsed = time.time() - start

        if not reconstructions:
            return ReconstructionMetrics(
                backend_name=self.name, success=False, processing_time_sec=elapsed,
                input_frame_count=input_count,
                error="COLMAP could not initialize any reconstruction (no valid initial image pair found)",
            )

        per_recon = {str(i): {"registered_images": r.num_reg_images(), "points3D": r.num_points3D()}
                     for i, r in reconstructions.items()}

        # COLMAP's incremental_mapping can register the SAME image across
        # more than one of its returned reconstructions (e.g. a looping
        # orbit revisiting a similar viewpoint) -- summing num_reg_images()
        # naively can therefore exceed input_count. Dedup by image name so
        # registration_rate never reports over 100%.
        registered_names = set()
        for r in reconstructions.values():
            for image_id in r.reg_image_ids():
                registered_names.add(r.image(image_id).name)
        total_registered = len(registered_names)

        best = max(reconstructions.values(), key=lambda r: r.num_reg_images())

        return ReconstructionMetrics(
            backend_name=self.name, success=True, processing_time_sec=elapsed,
            input_frame_count=input_count,
            registered_frame_count=total_registered,
            registration_rate=(total_registered / input_count) if input_count else None,
            disconnected_components=len(reconstructions),
            point_count=best.num_points3D(),
            mean_reprojection_error=best.compute_mean_reprojection_error(),
            raw_output={"num_reconstructions": len(reconstructions), "per_reconstruction": per_recon},
        )


def build_backend(command_template: Optional[str], timeout_sec: Optional[float] = None,
                   use_pycolmap: bool = False) -> ReconstructionBackend:
    """Factory used by the CLI: --reconstruction-backend pycolmap -> real
    local COLMAP; a --reconstruction-command -> ExternalCommandBackend;
    neither -> NullBackend."""
    if use_pycolmap:
        return PycolmapBackend()
    if not command_template:
        return NullBackend()
    return ExternalCommandBackend(command_template, timeout_sec=timeout_sec)
