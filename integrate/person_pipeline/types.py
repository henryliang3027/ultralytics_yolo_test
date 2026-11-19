"""Data passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FramePacket:
    """One decoded frame from the source."""

    image: np.ndarray  # BGR, HxWx3
    frame_id: int  # running count of frames decoded by the source (gaps = frames dropped to stay real-time)
    timestamp: float  # time.monotonic() when the frame was decoded


@dataclass
class Detections:
    """Person detections of one frame, all arrays aligned by row."""

    xyxy: np.ndarray  # (N, 4) float32 pixel boxes
    conf: np.ndarray  # (N,) float32
    cls: np.ndarray  # (N,) float32

    def __len__(self) -> int:
        return len(self.conf)

    @classmethod
    def empty(cls) -> Detections:
        return cls(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.float32))


@dataclass
class FrameResult:
    """Everything the pipeline produced for one frame, aligned with `detections` rows."""

    packet: FramePacket
    detections: Detections
    track_ids: np.ndarray  # (N,) int, -1 = not tracked
    person_ids: list  # (N,) int | None: gallery identity, None = tracked but not identified yet
    timings_ms: dict  # stage name -> milliseconds
    embeddings: dict = field(default_factory=dict)  # track id -> (D,) Re-ID embedding computed this frame
