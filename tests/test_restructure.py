"""Phase 4.1: the restructure into quality/features/coverage/scoring/output/
core/reconstruction/utils packages must not break any existing import path
-- old locations are kept as backward-compatible shims. This locks in both
directions so a future edit can't silently break either one.
"""


def test_new_canonical_imports_work():
    from video_to_frame.quality import compute_quality_metrics
    from video_to_frame.features import FeatureExtractor, match_descriptors
    from video_to_frame.features.similarity import dhash, ssim_precomputed
    from video_to_frame.coverage import CoveragePool
    from video_to_frame.scoring import FrameScorer
    from video_to_frame.output import encode_and_save, resize_frame
    from video_to_frame.core import FrameRecord, run_pipeline
    from video_to_frame.core.models import RejectionReason
    from video_to_frame.core.pipeline import run_pipeline as run_pipeline2
    from video_to_frame.reconstruction import NullBackend, PycolmapBackend
    from video_to_frame.utils import VIDEO_EXTENSIONS, is_video_file
    assert run_pipeline is run_pipeline2


def test_old_shim_paths_still_work():
    from video_to_frame.analysis.quality import compute_quality_metrics
    from video_to_frame.analysis.features import FeatureExtractor, match_descriptors
    from video_to_frame.analysis.similarity import dhash, ssim_precomputed
    from video_to_frame.analysis.coverage import CoveragePool
    from video_to_frame.selection.scorer import FrameScorer
    from video_to_frame.compression.compressor import encode_and_save, resize_frame
    from video_to_frame.pipeline import run_pipeline
    from video_to_frame.models import FrameRecord, RejectionReason
    from video_to_frame.cli import main


def test_shim_and_canonical_resolve_to_the_same_object():
    from video_to_frame.pipeline import run_pipeline as via_shim
    from video_to_frame.core.pipeline import run_pipeline as via_core
    assert via_shim is via_core

    from video_to_frame.analysis.coverage import CoveragePool as via_shim_cp
    from video_to_frame.coverage import CoveragePool as via_core_cp
    assert via_shim_cp is via_core_cp


def test_video_extensions_is_not_duplicated_across_modules():
    from video_to_frame.batch import VIDEO_EXTENSIONS as batch_ext
    from video_to_frame.cli import VIDEO_EXTENSIONS as cli_ext
    from video_to_frame.utils import VIDEO_EXTENSIONS as utils_ext
    assert batch_ext is utils_ext
    assert cli_ext is utils_ext
