from video_to_frame.analysis.coverage import CoveragePool
from video_to_frame.analysis.features import FeatureExtractor
from video_to_frame.analysis.quality import compute_quality_metrics
from video_to_frame.config import PipelineConfig
from video_to_frame.selection.scorer import FrameScorer
from video_to_frame.selection.selector import StreamingSelector

from synthetic import tile as _tile


def _make_selector(config: PipelineConfig, kept: list, rejected: list) -> StreamingSelector:
    feature_extractor = FeatureExtractor(max_features=config.max_features, use_sift=config.use_sift)
    coverage_pool = CoveragePool(
        use_sift=config.use_sift,
        max_per_frame=config.coverage_pool_max_per_frame,
        max_pool_size=config.coverage_pool_max_size,
        ratio=config.descriptor_match_ratio,
        max_hamming_distance=config.orb_max_hamming_distance,
    )
    scorer = FrameScorer(config.weights, config.exposure_low, config.exposure_high, config.exposure_min_contrast)
    return StreamingSelector(
        config, coverage_pool, feature_extractor, scorer,
        on_kept=lambda r, f: kept.append(r),
        on_rejected=lambda r: rejected.append(r),
    )


def test_consecutive_near_duplicates_collapse_to_one_representative():
    config = PipelineConfig()
    kept, rejected = [], []
    selector = _make_selector(config, kept, rejected)

    # First 4 pan offsets are nearly identical (a near-static camera); the 5th
    # is a big pan with zero scene overlap -- a genuinely new viewpoint.
    offsets = [0, 2, 1, 3, 500]
    for i, off in enumerate(offsets):
        frame = _tile(off)
        quality = compute_quality_metrics(frame)
        selector.process(frame, i, float(i) * 33.3, quality)
    selector.finalize()

    assert len(kept) == 2
    assert kept[0].index in (0, 1, 2, 3)
    assert kept[1].index == 4
    assert len(rejected) == 3


def test_every_frame_kept_when_viewpoint_keeps_changing():
    config = PipelineConfig()
    kept, rejected = [], []
    selector = _make_selector(config, kept, rejected)

    # Non-overlapping pan offsets (each tile is 200px wide, spaced 200px apart
    # on a 1000px canvas) -- no two frames share any scene content.
    offsets = [0, 200, 400, 600]
    for i, off in enumerate(offsets):
        frame = _tile(off)
        quality = compute_quality_metrics(frame)
        selector.process(frame, i, float(i) * 33.3, quality)
    selector.finalize()

    assert len(kept) == len(offsets)
    assert rejected == []
