"""Pipeline orchestrator (Facade): sources -> batched detect -> per-camera track / Re-ID / gallery -> outputs.

The YOLO and Re-ID models are loaded once and shared by every camera. Each loop takes the newest frame from every
camera that has one, runs YOLO once on that batch, then hands each camera its own detections (see camera.py).
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from .camera import CameraStream
from .config import PipelineConfig
from .detector import PersonDetector
from .reid import ReIDExtractor

IDLE_SLEEP_S = 0.002  # no camera has a new frame yet: wait this long before polling again


def _save_path(base: str | None, name: str, n_cameras: int) -> str | None:
    """output.mp4 -> output.mp4 for one camera, output_cam1.mp4 / output_cam2.mp4 for several."""
    if not base or n_cameras == 1:
        return base
    p = Path(base)
    return str(p.with_name(f"{p.stem}_{name}{p.suffix}"))


class PersonPipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        n = len(cfg.cameras)
        print(f"[pipeline] loading models for {n} camera(s): {', '.join(c.name for c in cfg.cameras)}")
        self.detector = PersonDetector(cfg.detector)
        self.detector.warmup(batch=n)
        self.reid = ReIDExtractor(cfg.reid)
        self.cameras = [
            CameraStream(cam, cfg, self.reid, _save_path(cfg.display.save_path, cam.name, n)) for cam in cfg.cameras
        ]
        self._running = False

    def run(self) -> None:
        for cam in self.cameras:
            cam.start()
        last_report = time.monotonic()
        self._running = True
        try:
            while self._running:
                batch = [(cam, p) for cam in self.cameras if (p := cam.poll()) is not None]
                if not batch:
                    if all(cam.finished for cam in self.cameras):
                        break  # every video file has ended
                    time.sleep(IDLE_SLEEP_S)
                    continue

                t = time.perf_counter()
                dets_list = self.detector.detect([packet.image for _, packet in batch])
                detect_ms = (time.perf_counter() - t) * 1000  # one shared call for the whole batch

                for (cam, packet), dets in zip(batch, dets_list):
                    cam.output(cam.process(packet, dets, detect_ms))

                if self.cfg.display.show and cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                now = time.monotonic()
                if now - last_report >= self.cfg.logging.stats_interval_s:
                    for cam in self.cameras:
                        if (line := cam.stats_line()) is not None:
                            print(line)
                    last_report = now
        finally:
            for cam in self.cameras:
                cam.stop()
            if self.cfg.display.show:
                cv2.destroyAllWindows()

    def stop(self) -> None:
        self._running = False
