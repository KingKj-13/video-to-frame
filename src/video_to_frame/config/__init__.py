from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field
from typing import Optional

import yaml


@dataclass
class ScoreWeights:
    quality: float = 1.0
    sharpness: float = 1.0
    features: float = 0.5
    viewpoint: float = 1.5
    coverage: float = 1.5
    redundancy: float = 2.0


@dataclass
class PipelineConfig:
    # Blur handling
    hard_blur_floor: float = 30.0
    blur_threshold: float = 100.0

    # Redundancy / viewpoint change
    redundancy_threshold: float = 0.82
    coverage_gain_threshold: float = 0.15
    reference_window_size: int = 6

    # Minimum-overlap protection: a *safety* floor, separate from and opposite
    # to redundancy_threshold. redundancy_threshold asks "are these frames too
    # similar to keep both?"; min_overlap_threshold asks "if we're about to
    # drop this frame anyway, would that leave the previous kept frame and the
    # next one too disconnected for a downstream SfM feature matcher to link
    # them?" ORB feature-overlap ratio (matches / candidate descriptor count)
    # below this floor forces the candidate to be kept even though normal
    # selection logic would have rejected it as redundant. 0.30 is a
    # conservative "bare minimum connectivity" floor, not a quality target --
    # most photogrammetry guidance wants 60%+ overlap for *good* reconstructions;
    # this only guards against a broken link. Tune per dataset.
    min_overlap_threshold: float = 0.30

    # Feature extraction
    use_sift: bool = False
    max_features: int = 500
    descriptor_match_ratio: float = 0.75  # SIFT (L2) matching: Lowe's ratio test
    orb_max_hamming_distance: int = 64  # ORB (Hamming) matching: cross-check + absolute distance

    # Phase 2.5: ORB feature extraction (and its downstream matching) is one
    # of the largest real costs on high-resolution real photos, and measured
    # experiments (6 consecutive real drone frames) showed overlap ratios
    # stay within ~0.04 of full-resolution at 0.75x scale and ~0.07 at 0.5x
    # (well inside the margin between redundancy_threshold and
    # min_overlap_threshold), while extraction+matching run 6-12x faster.
    # 0.25x was measurably too aggressive (keypoint counts approach the
    # low-texture floor, ratios become noisy). None/1.0 = disabled (default) --
    # this only affects what ORB is computed FROM; blur detection and output
    # frames are never touched by this, so it changes speed, not blur
    # semantics or output quality.
    analysis_scale: Optional[float] = None

    # Global coverage pool
    coverage_pool_max_per_frame: int = 60
    coverage_pool_max_size: int = 20000

    # Scoring
    weights: ScoreWeights = field(default_factory=ScoreWeights)

    # Frame-count bounds
    min_frames: Optional[int] = None
    max_frames: Optional[int] = None

    # Compression -- defaults are quality-first (full resolution, lossless
    # PNG) since output feeds a downstream 3D reconstruction model; pass
    # --format webp/jpg and/or --resize for a smaller, lossy "compact mode".
    resize_long_edge: Optional[int] = None
    resize_width: Optional[int] = None
    resize_height: Optional[int] = None
    output_format: str = "png"
    output_quality: int = 90
    # PNG is always lossless regardless of this value -- it only trades
    # encode speed for file size (Phase 2.5 profiling: level 9 costs ~30x
    # level 1's time on a real photo for ~25% smaller output). Default fast;
    # raise it only if you specifically need smaller PNGs and can afford the
    # time.
    png_compression_level: int = 1

    # Preview video
    preview_fps: float = 5.0

    # Exposure gating
    exposure_low: int = 15
    exposure_high: int = 240
    exposure_min_contrast: float = 10.0

    @classmethod
    def from_yaml(cls, path: Optional[str]) -> "PipelineConfig":
        if not path:
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineConfig":
        data = dict(data)
        weights_data = data.pop("weights", None)
        known_fields = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known_fields
        if unknown:
            raise ValueError(f"Unknown config field(s): {sorted(unknown)}")
        cfg = cls(**data)
        if weights_data:
            cfg.weights = ScoreWeights(**weights_data)
        return cfg

    def apply_overrides(self, **overrides) -> "PipelineConfig":
        cfg = dataclasses.replace(self)
        for key, value in overrides.items():
            if value is None:
                continue
            if not hasattr(cfg, key):
                raise ValueError(f"Unknown config override: {key}")
            setattr(cfg, key, value)
        return cfg

    def validate(self) -> None:
        if self.hard_blur_floor > self.blur_threshold:
            raise ValueError("hard_blur_floor must be <= blur_threshold")
        if self.output_format.lower() not in ("webp", "jpg", "jpeg", "png"):
            raise ValueError(f"Unsupported output_format: {self.output_format}")
        if not 0 <= self.png_compression_level <= 9:
            raise ValueError("png_compression_level must be within [0, 9]")
        if self.analysis_scale is not None and not 0.0 < self.analysis_scale <= 1.0:
            raise ValueError("analysis_scale must be within (0, 1] (or None to disable)")
        if self.min_frames is not None and self.max_frames is not None and self.min_frames > self.max_frames:
            raise ValueError("min_frames must be <= max_frames")
        if not 0.0 <= self.redundancy_threshold <= 1.0:
            raise ValueError("redundancy_threshold must be within [0, 1]")
        if not 0.0 <= self.coverage_gain_threshold <= 1.0:
            raise ValueError("coverage_gain_threshold must be within [0, 1]")
        if not 0.0 <= self.min_overlap_threshold <= 1.0:
            raise ValueError("min_overlap_threshold must be within [0, 1]")
        if self.min_overlap_threshold >= self.redundancy_threshold:
            raise ValueError(
                "min_overlap_threshold must be < redundancy_threshold "
                "(they are opposite safety constraints: a candidate can't be "
                "both 'too similar to keep' and 'too disconnected to drop')"
            )

    def to_dict(self) -> dict:
        return asdict(self)
