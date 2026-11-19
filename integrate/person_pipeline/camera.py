"""Per-camera state: frame source, tracker, gallery, Cupola sink, stats, window and recorder.

Everything that remembers something between frames lives here, one instance per camera, so two cameras never mix
their tracks or person IDs. The stateless models (YOLO, Re-ID) are owned by the pipeline and shared.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import replace

import cv2
import numpy as np

from .capture import FrameSource, create_frame_source
from .config import CameraConfig, PipelineConfig
from .cupola import CupolaEventSink
from .gallery import IdentityGallery
from .recorder import VideoRecorder
from .reid import ReIDExtractor
from .sinks import ResultSink
from .tracker import PersonTracker
from .types import Detections, FramePacket, FrameResult
from .visualizer import Visualizer


def _pairwise_iou(xyxy: np.ndarray) -> np.ndarray:
    x1 = np.maximum(xyxy[:, None, 0], xyxy[None, :, 0])
    y1 = np.maximum(xyxy[:, None, 1], xyxy[None, :, 1])
    x2 = np.minimum(xyxy[:, None, 2], xyxy[None, :, 2])
    y2 = np.minimum(xyxy[:, None, 3], xyxy[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    iou = inter / (area[:, None] + area[None, :] - inter + 1e-9)
    np.fill_diagonal(iou, 0.0)
    return iou


class CameraStream:
    def __init__(self, cam: CameraConfig, cfg: PipelineConfig, reid: ReIDExtractor, save_path: str | None):
        self.name = cam.name
        self.cam = cam
        self.cfg = cfg
        self.reid = reid  # shared model
        self.tracker = PersonTracker(cfg.tracker)
        self.gallery = IdentityGallery(cfg.gallery)
        self.visualizer = Visualizer(cfg.display.draw_detections)
        self.sinks: list[ResultSink] = []
        if cfg.cupola.enabled and cam.cupola_camera_url:
            self.sinks.append(CupolaEventSink(cfg.cupola, cam.cupola_camera_url, cam.name))
        self.source: FrameSource | None = None
        self.window = f"pipeline {cam.name}"
        self.save_path = save_path
        self._writer: VideoRecorder | None = None
        self._last_embedded: dict[int, int] = {}  # track id -> processed-frame index of its last embedding
        self._frame_idx = 0
        # stats
        self._done_times = deque(maxlen=31)  # completion times of the last 30 frames, for FPS
        self._stats = defaultdict(float)
        self._n_stats = 0
        self._last_frame_id = None
        self._dropped = 0
        self.fps = 0.0

    # ---- lifecycle ----
    def start(self) -> None:
        self.sinks = [s for s in self.sinks if s.start()]
        self.source = create_frame_source(replace(self.cfg.source, url=self.cam.url)).start()
        if self.cfg.display.show:
            cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window, *self.cfg.display.window_size)

    def stop(self) -> None:
        if self.source is not None:
            self.source.stop()
        for sink in self.sinks:
            sink.stop()
        if self._writer is not None:
            self._writer.close()  # waits for the queued frames, then finalizes the file

    @property
    def finished(self) -> bool:
        return self.source is not None and self.source.finished

    def poll(self) -> FramePacket | None:
        """Newest frame not processed yet, or None right away if there is none."""
        packet = self.source.read(timeout=0)
        if packet is not None:
            if self._last_frame_id is not None:
                self._dropped += packet.frame_id - self._last_frame_id - 1
            self._last_frame_id = packet.frame_id
        return packet

    # ---- one frame (after the shared YOLO call) ----
    def process(self, packet: FramePacket, dets: Detections, detect_ms: float, now: float | None = None) -> FrameResult:
        """Track, embed and identify one frame. `now` (seconds) drives the gallery timeouts; defaults to the wall
        clock, record mode passes the video time so timeouts follow the recording, not the processing speed."""
        timings = {"detect": detect_ms}
        t = time.perf_counter()
        track_ids = self.tracker.update(dets, packet.image)
        timings["track"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        rows = self._select_for_reid(dets, track_ids)
        feats = self.reid.extract(packet.image, dets.xyxy[rows])
        embeddings = {int(track_ids[r]): f for r, f in zip(rows, feats)}
        for tid in embeddings:
            self._last_embedded[tid] = self._frame_idx
        timings["reid"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        visible = [int(t_) for t_ in track_ids if t_ >= 0]
        boxes = {int(t_): dets.xyxy[i] for i, t_ in enumerate(track_ids) if t_ >= 0}
        mapping = self.gallery.update(visible, embeddings, time.monotonic() if now is None else now, boxes)
        person_ids = [mapping.get(int(t_)) if t_ >= 0 else None for t_ in track_ids]
        timings["gallery"] = (time.perf_counter() - t) * 1000

        self._frame_idx += 1
        if self._frame_idx % 300 == 0:  # drop bookkeeping for tracks the gallery has forgotten
            self._last_embedded = {k: v for k, v in self._last_embedded.items() if self.gallery.person_of(k) is not None}
        return FrameResult(packet, dets, track_ids, person_ids, timings, embeddings)

    def _select_for_reid(self, dets: Detections, track_ids: np.ndarray) -> list[int]:
        """Pick the detection rows worth a Re-ID embedding this frame.

        - Unidentified track: always (it needs one embedding to get a person ID).
        - Identified track: every `update_interval` frames, only for clean crops (big enough, confident,
          not overlapping another person) so the feature bank is not polluted.
        """
        rc = self.cfg.reid
        iou = _pairwise_iou(dets.xyxy) if len(dets) > 1 else np.zeros((len(dets), len(dets)))
        heights = dets.xyxy[:, 3] - dets.xyxy[:, 1]
        widths = dets.xyxy[:, 2] - dets.xyxy[:, 0]
        rows = []
        for i, tid in enumerate(track_ids):
            if tid < 0 or heights[i] < 8 or widths[i] < 4:
                continue
            if self.gallery.person_of(int(tid)) is None:
                rows.append(i)
                continue
            due = self._frame_idx - self._last_embedded.get(int(tid), -(10**9)) >= rc.update_interval
            clean = (
                heights[i] >= rc.min_box_height
                and dets.conf[i] >= rc.min_conf
                and (len(dets) < 2 or iou[i].max() <= rc.max_overlap_iou)
            )
            if due and clean:
                rows.append(i)
        return rows

    # ---- output ----
    def output(self, result: FrameResult) -> None:
        """Hand the result to the sinks, update stats, and draw / record."""
        for sink in self.sinks:
            sink.publish(result)

        now = time.monotonic()
        latency_ms = (now - result.packet.timestamp) * 1000  # decode -> result ready
        self._done_times.append(now)
        span = self._done_times[-1] - self._done_times[0]
        self.fps = (len(self._done_times) - 1) / span if span > 0 else 0.0
        for k, v in result.timings_ms.items():
            self._stats[k] += v
        self._stats["latency"] += latency_ms
        self._n_stats += 1

        disp = self.cfg.display
        if disp.show or self.save_path:
            status = (
                f"{self.name}  frame {result.packet.frame_id}  {self.fps:.1f} fps  {latency_ms:.0f} ms  "
                f"people {len(self.gallery)}"
            )
            vis = self.visualizer.draw(result, status)
            if self.save_path:
                if self._writer is None:
                    h, w = vis.shape[:2]
                    self._writer = VideoRecorder(self.save_path, self.source.fps or 15.0, (w, h), self.name)
                self._writer.write(vis)  # encoded on the recorder's thread
            if disp.show:
                cv2.imshow(self.window, vis)

    def stats_line(self) -> str | None:
        """Averages since the last call, then reset. None if no frame was processed."""
        n = self._n_stats
        if not n:
            return None
        avg = " ".join(f"{k} {self._stats[k] / n:.1f}ms" for k in ("detect", "track", "reid", "gallery"))
        line = (
            f"[stats {self.name}] {self.fps:.1f} fps | {avg} | latency {self._stats['latency'] / n:.0f}ms | "
            f"dropped {self._dropped} frames | people {len(self.gallery)}"
        )
        self._stats.clear()
        self._n_stats = self._dropped = 0
        return line
