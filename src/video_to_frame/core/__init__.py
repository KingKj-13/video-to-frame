from .models import (
    FrameMetrics,
    FrameRecord,
    KeptReason,
    QualityMetrics,
    RejectionReason,
    record_to_dict,
)

__all__ = [
    "FrameMetrics", "FrameRecord", "KeptReason", "QualityMetrics", "RejectionReason", "record_to_dict",
    "KeptPair", "run_pipeline",
]


def __getattr__(name: str):
    # KeptPair/run_pipeline are re-exported lazily (PEP 562), not via an
    # eager `from .pipeline import ...` above: core.pipeline transitively
    # imports quality/scoring/selection/reporting, several of which import
    # from the root `video_to_frame.models` shim, and that shim imports
    # from `core.models` -- forcing this __init__ to run. An eager import
    # here would re-enter this same module before it finished executing,
    # a real circular import. Deferring until first access breaks the cycle
    # without changing the public surface: video_to_frame.core.run_pipeline
    # still works exactly as before.
    if name in ("KeptPair", "run_pipeline"):
        from . import pipeline as _pipeline
        return getattr(_pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
