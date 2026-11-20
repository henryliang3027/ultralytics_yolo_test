"""Real-time person detection, tracking and re-identification pipeline.

Names are imported lazily (on first use), so light submodules such as track_update_contract can be imported
without loading torch, ultralytics or onnxruntime (the shared gallery side does this).
"""

from importlib import import_module

_EXPORTS = {
    "CameraConfig": ".config",
    "CameraStream": ".camera",
    "CupolaEventSink": ".cupola",
    "FrameSource": ".capture",
    "GStreamerSource": ".capture",
    "IdentityGallery": ".gallery",
    "OpenCVSource": ".capture",
    "PersonDetector": ".detector",
    "PersonPipeline": ".pipeline",
    "PersonTracker": ".tracker",
    "PipelineConfig": ".config",
    "ReIDExtractor": ".reid",
    "ResultSink": ".sinks",
    "Visualizer": ".visualizer",
    "create_frame_source": ".capture",
    "load_config": ".config",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        value = getattr(import_module(_EXPORTS[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
