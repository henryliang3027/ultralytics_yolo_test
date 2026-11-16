"""Person detection module: Ultralytics YOLO restricted to the COCO 'person' class."""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from .config import DetectorConfig
from .types import Detections

PERSON_CLASS_ID = 0


class PersonDetector:
    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self.model = YOLO(cfg.weights)
        self.half = bool(cfg.half) and str(cfg.device) != "cpu"

    def warmup(self, batch: int = 1, shape=(1920, 3840, 3)) -> None:
        """First inference builds GPU kernels and is slow; do it before the stream starts, at the real batch size."""
        self.detect([np.zeros(shape, np.uint8)] * batch)

    def detect(self, images: list[np.ndarray]) -> list[Detections]:
        """Run one batched inference over several frames (one per camera). Returns one Detections per image."""
        results = self.model.predict(
            images,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf,
            classes=[PERSON_CLASS_ID],
            device=self.cfg.device,
            quantize=16 if self.half else 32,  # replaces the deprecated half=True
            verbose=False,
        )
        return [self._to_detections(r) for r in results]

    @staticmethod
    def _to_detections(result) -> Detections:
        boxes = result.boxes.cpu().numpy()
        if len(boxes) == 0:
            return Detections.empty()
        return Detections(
            xyxy=boxes.xyxy.astype(np.float32),
            conf=boxes.conf.astype(np.float32),
            cls=boxes.cls.astype(np.float32),
        )
