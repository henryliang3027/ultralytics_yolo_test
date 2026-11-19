"""Typed configuration loaded from config.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from ultralytics.utils import YAML  # bundled YAML loader, avoids an extra PyYAML dependency


@dataclass
class CameraConfig:
    name: str  # used in window titles, log lines and per-camera recording file names
    url: str  # RTSP URL or local video file
    cupola_camera_url: str = ""  # Cupola camera uri / relayUri / path to draw on; empty = no Cupola push
    camera_id: str = ""  # tracklet contract camera id (e.g. "51_203"); empty = use `name`
    recording_dir: str = ""  # Cupola recording folder of this camera (record mode); relative to record.recordings_root


@dataclass
class SourceConfig:
    """Decoding options shared by every camera; `url` is filled in per camera."""

    url: str = ""
    backend: str = "opencv"
    transport: str = "tcp"
    reconnect_delay_s: float = 2.0
    gst_decode: str = "rtph264depay ! h264parse ! avdec_h264"


@dataclass
class DetectorConfig:
    weights: str = "yolo26s.pt"
    imgsz: int = 1920
    conf: float = 0.1
    device: Any = 0
    half: bool = True


@dataclass
class TrackerConfig:
    track_high_thresh: float = 0.4
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.5
    match_thresh: float = 0.8
    track_buffer: int = 30
    gmc_method: str = "none"


@dataclass
class ReIDConfig:
    model: str = ""
    providers: list = field(default_factory=lambda: ["CPUExecutionProvider"])
    input_size: list = field(default_factory=lambda: [128, 256])
    mean: list = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: list = field(default_factory=lambda: [0.226, 0.226, 0.226])
    min_box_height: int = 48
    min_conf: float = 0.5
    max_overlap_iou: float = 0.3
    update_interval: int = 5


@dataclass
class GalleryConfig:
    max_embeddings_per_person: int = 20
    match_threshold: float = 0.65
    match_metric: str = "max"
    track_timeout_s: float = 5.0
    identity_ttl_s: float = 600.0


@dataclass
class DisplayConfig:
    show: bool = True
    window_size: list = field(default_factory=lambda: [1440, 720])
    draw_detections: bool = True
    save_path: str | None = None


@dataclass
class CupolaConfig:
    enabled: bool = False
    server: str = ""
    resolution: str = "auto"
    send_interval_s: float = 0.2
    box_ttl_ms: int = 1000


@dataclass
class RecordConfig:
    """Record mode (record.py): closed tracklets from Cupola recordings -> tracklets.jsonl."""

    recordings_root: str = ""  # folder holding one sub-folder of .ts segments per camera
    out_dir: str = "tracklets"  # tracklets.jsonl, summary.json and the per-day track id registry go here
    timezone: str = "Asia/Taipei"  # segment file names are local start times in this zone
    max_embeddings: int = 20  # embeddings per tracklet, picked evenly over the track's life
    reid_model_version: str = ""  # empty = Re-ID ONNX file name without extension
    vector_decimals: int = 6  # rounding of embedding values in the JSON (keeps the file small)


@dataclass
class LoggingConfig:
    stats_interval_s: float = 5.0


@dataclass
class PipelineConfig:
    cameras: list[CameraConfig]
    source: SourceConfig
    detector: DetectorConfig
    tracker: TrackerConfig
    reid: ReIDConfig
    gallery: GalleryConfig
    display: DisplayConfig
    cupola: CupolaConfig
    logging: LoggingConfig
    record: RecordConfig


def _build(cls, data: dict | None):
    """Create a dataclass from a dict, rejecting unknown keys so typos in config.yaml fail loudly."""
    data = data or {}
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys in config: {sorted(unknown)}")
    return cls(**data)


def _resolve(path: str | None, base: Path) -> str | None:
    """Resolve a path relative to the config file's folder (absolute paths and URLs are kept)."""
    if not path or "://" in path:
        return path
    p = Path(path).expanduser()
    return str(p if p.is_absolute() else (base / p).resolve())


def load_config(path: str | Path) -> PipelineConfig:
    path = Path(path).resolve()
    raw = YAML.load(path) or {}
    base = path.parent
    cameras = [_build(CameraConfig, c) for c in raw.get("cameras") or []]
    if not cameras:
        raise ValueError("config needs at least one entry under `cameras`")
    names = [c.name for c in cameras]
    if len(set(names)) != len(names):
        raise ValueError(f"camera names must be unique, got {names}")
    for cam in cameras:
        cam.url = _resolve(cam.url, base)  # URLs are kept as is, file paths become absolute
    cfg = PipelineConfig(
        cameras=cameras,
        source=_build(SourceConfig, raw.get("source")),
        detector=_build(DetectorConfig, raw.get("detector")),
        tracker=_build(TrackerConfig, raw.get("tracker")),
        reid=_build(ReIDConfig, raw.get("reid")),
        gallery=_build(GalleryConfig, raw.get("gallery")),
        display=_build(DisplayConfig, raw.get("display")),
        cupola=_build(CupolaConfig, raw.get("cupola")),
        logging=_build(LoggingConfig, raw.get("logging")),
        record=_build(RecordConfig, raw.get("record")),
    )
    cfg.detector.weights = _resolve(cfg.detector.weights, base)
    cfg.reid.model = _resolve(cfg.reid.model, base)
    cfg.record.recordings_root = _resolve(cfg.record.recordings_root, base)
    cfg.record.out_dir = _resolve(cfg.record.out_dir, base)
    if cfg.display.save_path:
        cfg.display.save_path = _resolve(cfg.display.save_path, base)
    return cfg
