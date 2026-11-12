"""Pipeline orchestrator (Facade): source -> detect -> track -> Re-ID -> gallery -> display/record."""

from __future__ import annotations

import time
from collections import defaultdict, deque

import cv2
import numpy as np

from .capture import FrameSource, create_frame_source
from .config import PipelineConfig
from .detector import PersonDetector
from .gallery import IdentityGallery
from .cupola import CupolaEventSink
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


class PersonPipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        print("[pipeline] loading models...")
        self.detector = PersonDetector(cfg.detector)
        self.detector.warmup()
        self.tracker = PersonTracker(cfg.tracker)
        self.reid = ReIDExtractor(cfg.reid)
        self.gallery = IdentityGallery(cfg.gallery)
        self.visualizer = Visualizer(cfg.display.draw_detections)
        self.sinks: list[ResultSink] = []
        if cfg.cupola.enabled:
            self.sinks.append(CupolaEventSink(cfg.cupola))
        self._last_embedded: dict[int, int] = {}  # track id -> processed-frame index of its last embedding
        self._frame_idx = 0
        self._running = False

    # ---- one frame ----
    def process(self, packet: FramePacket) -> FrameResult:
        timings = {}
        t = time.perf_counter()
        dets = self.detector.detect(packet.image)
        timings["detect"] = (time.perf_counter() - t) * 1000

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
        mapping = self.gallery.update(visible, embeddings, time.monotonic(), boxes)
        person_ids = [mapping.get(int(t_)) if t_ >= 0 else None for t_ in track_ids]
        timings["gallery"] = (time.perf_counter() - t) * 1000

        self._frame_idx += 1
        if self._frame_idx % 300 == 0:  # drop bookkeeping for tracks the gallery has forgotten
            self._last_embedded = {k: v for k, v in self._last_embedded.items() if self.gallery.person_of(k) is not None}
        return FrameResult(packet, dets, track_ids, person_ids, timings)

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

    # ---- main loop ----
    def run(self) -> None:
        sinks = [s for s in self.sinks if s.start()]
        source: FrameSource = create_frame_source(self.cfg.source).start()
        disp = self.cfg.display
        writer = None
        if disp.show:
            cv2.namedWindow("pipeline", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("pipeline", *disp.window_size)

        stats = defaultdict(float)
        n_stats, last_report = 0, time.monotonic()
        last_frame_id, dropped = None, 0
        done_times = deque(maxlen=31)  # completion times of the last 30 frames, for FPS
        self._running = True
        try:
            while self._running:
                packet = source.read(timeout=1.0)
                if packet is None:
                    if source.finished:
                        break
                    continue
                if last_frame_id is not None:
                    dropped += packet.frame_id - last_frame_id - 1
                last_frame_id = packet.frame_id

                result = self.process(packet)
                for sink in sinks:
                    sink.publish(result)

                now = time.monotonic()
                latency_ms = (now - packet.timestamp) * 1000  # decode -> result ready
                done_times.append(now)
                span = done_times[-1] - done_times[0]
                fps = (len(done_times) - 1) / span if span > 0 else 0.0
                for k, v in result.timings_ms.items():
                    stats[k] += v
                stats["latency"] += latency_ms
                n_stats += 1

                if disp.show or disp.save_path:
                    status = f"frame {packet.frame_id}  {fps:.1f} fps  {latency_ms:.0f} ms  people {len(self.gallery)}"
                    vis = self.visualizer.draw(result, status)
                    if disp.save_path:
                        if writer is None:
                            h, w = vis.shape[:2]
                            out_fps = source.fps or 15.0
                            writer = cv2.VideoWriter(disp.save_path, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))
                        writer.write(vis)
                    if disp.show:
                        cv2.imshow("pipeline", vis)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                if now - last_report >= self.cfg.logging.stats_interval_s and n_stats:
                    avg = " ".join(f"{k} {stats[k] / n_stats:.1f}ms" for k in ("detect", "track", "reid", "gallery"))
                    print(
                        f"[stats] {fps:.1f} fps | {avg} | latency {stats['latency'] / n_stats:.0f}ms | "
                        f"dropped {dropped} frames | people {len(self.gallery)}"
                    )
                    stats.clear()
                    n_stats, dropped, last_report = 0, 0, now
        finally:
            source.stop()
            for sink in sinks:
                sink.stop()
            if writer is not None:
                writer.release()
            if disp.show:
                cv2.destroyAllWindows()

    def stop(self) -> None:
        self._running = False
