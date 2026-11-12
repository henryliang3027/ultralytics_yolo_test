"""Real-time person detection, tracking and re-identification pipeline."""

from .capture import FrameSource, GStreamerSource, OpenCVSource, create_frame_source
from .config import PipelineConfig, load_config
from .cupola import CupolaEventSink
from .sinks import ResultSink
from .detector import PersonDetector
from .gallery import IdentityGallery
from .pipeline import PersonPipeline
from .reid import ReIDExtractor
from .tracker import PersonTracker
from .visualizer import Visualizer

__all__ = [
    "CupolaEventSink",
    "FrameSource",
    "ResultSink",
    "GStreamerSource",
    "IdentityGallery",
    "OpenCVSource",
    "PersonDetector",
    "PersonPipeline",
    "PersonTracker",
    "PipelineConfig",
    "ReIDExtractor",
    "Visualizer",
    "create_frame_source",
    "load_config",
]
