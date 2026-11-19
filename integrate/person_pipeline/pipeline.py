"""Pipeline orchestrator (Facade): sources -> batched detect -> per-camera track / Re-ID / gallery -> outputs.

The YOLO and Re-ID models are loaded once and shared by every camera. Each loop takes the newest frame from every
camera that has one, runs YOLO once on that batch, then hands each camera its own detections (see camera.py).
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from .camera import CameraStream
from .config import CameraConfig, PipelineConfig
from .detector import PersonDetector
from .reid import ReIDExtractor

IDLE_SLEEP_S = 0.002  # no camera has a new frame yet: wait this long before polling again


def _save_path(base: str | None, cam: CameraConfig, n_cameras: int) -> str | None:
    """output.mp4 -> output.mp4 for one camera, output_cam1.mp4 / output_cam2.mp4 for several.
    A path without a file extension is a folder: each camera is saved as <source video name>.mp4 in it
    (live streams, which have no file name, use the camera name)."""
    if not base:
        return base
    p = Path(base)
    if not p.suffix:
        src = Path(cam.url)
        stem = src.stem if "://" not in cam.url and src.is_file() else cam.name
        out = p / f"{stem}.mp4"
        if src.is_file() and out.resolve() == src.resolve():  # never overwrite the source video
            out = p / f"{stem}_annotated.mp4"
        return str(out)
    if n_cameras == 1:
        return base
    return str(p.with_name(f"{p.stem}_{cam.name}{p.suffix}"))


class PersonPipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        n = len(cfg.cameras)
        print(f"[pipeline] loading models for {n} camera(s): {', '.join(c.name for c in cfg.cameras)}")
        self.detector = PersonDetector(cfg.detector)
        self.detector.warmup(batch=n)
        self.reid = ReIDExtractor(cfg.reid)
        self.cameras = [
            CameraStream(cam, cfg, self.reid, _save_path(cfg.display.save_path, cam, n)) for cam in cfg.cameras
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
