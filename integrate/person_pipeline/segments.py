"""Read Cupola recording segments in order, every frame, each with its contract UTC timestamp.

Cupola writes ~32 s MPEG-TS segments named by their LOCAL start time (`YYYY-MM-DD_HH-MM-SS-ffffff.ts`). The tracklet
contract defines a frame's time as: segment start (converted to UTC) + frame_index / fps, where frame_index counts
frames inside that segment. Unlike the live sources in capture.py nothing is dropped: a background thread decodes
into a small bounded queue and waits when the pipeline is slower.
"""

from __future__ import annotations

import bisect
import queue
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from .tracklet_contract import frame_utc, segment_start_utc

SEGMENT_MAX = timedelta(minutes=2)  # generous upper bound on one segment's length
_END = object()


@dataclass
class SegmentFrame:
    image: np.ndarray  # BGR
    ts: datetime  # UTC, contract rule (segment start + index / fps)
    index: int  # running count of frames yielded by this reader
    pts_drift_s: float  # contract time minus the frame's own PTS time; non-zero when the camera skipped frames


def list_segments(folder: str | Path, tz: ZoneInfo) -> list[tuple[datetime, Path]]:
    """(UTC start, path) of every segment in the folder, sorted."""
    segs = []
    for p in Path(folder).glob("*.ts"):
        try:
            segs.append((segment_start_utc(p, tz), p))
        except ValueError:
            continue
    return sorted(segs)


def segments_for_range(folder: str | Path, start: datetime, end: datetime, tz: ZoneInfo) -> list[Path]:
    """Segments that may hold frames in [start, end] (UTC): the last one starting before `start`, then the rest."""
    segs = list_segments(folder, tz)
    starts = [s for s, _ in segs]
    i = max(bisect.bisect_right(starts, start) - 1, 0)
    return [p for s, p in segs[i:] if s <= end and s + SEGMENT_MAX >= start]


class SegmentReader:
    """Background decoder over the segments covering [start, end]; `read()` returns frames in order, None at the end."""

    def __init__(self, folder: str | Path, start: datetime, end: datetime, tz: ZoneInfo, max_queue: int = 8):
        self.folder = Path(folder)
        self.start, self.end, self.tz = start, end, tz
        self.segments = segments_for_range(self.folder, start, end, tz)
        if not self.segments:
            raise FileNotFoundError(f"no segments in {self.folder} for {start} .. {end}")
        self.fps = 0.0
        self.max_abs_drift_s = 0.0
        self.backwards = 0  # frames whose contract time is not after the previous frame's (segments overlap)
        self.max_backwards_s = 0.0
        self.max_forward_gap_s = 0.0  # largest jump between consecutive frames (recording gaps)
        self._q: queue.Queue = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"seg-{self.folder.name[:8]}", daemon=True)

    def start_reading(self) -> SegmentReader:
        self._thread.start()
        return self

    def read(self) -> SegmentFrame | None:
        item = self._q.get()
        return None if item is _END else item

    def stop(self) -> None:
        self._stop.set()
        while not self._q.empty():  # unblock the decoder if it waits on a full queue
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self._thread.join(timeout=5)

    def _put(self, item) -> bool:
        while not self._stop.is_set():
            try:
                self._q.put(item, timeout=0.5)
                return True
            except queue.Full:
                continue
        return False

    def _run(self) -> None:
        n = 0
        prev = None
        try:
            for path in self.segments:
                cap = cv2.VideoCapture(str(path))
                if not cap.isOpened():
                    print(f"[segments] warning: cannot open {path}")
                    continue
                fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
                self.fps = self.fps or fps
                seg_start = segment_start_utc(path, self.tz)
                k = 0
                while not self._stop.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        break
                    ts = frame_utc(path, k, fps, self.tz)
                    pts = seg_start + timedelta(milliseconds=cap.get(cv2.CAP_PROP_POS_MSEC))
                    k += 1
                    if ts < self.start:
                        continue
                    if ts > self.end:
                        break
                    if prev is not None:
                        step = (ts - prev).total_seconds()
                        if step <= 0:
                            self.backwards += 1
                            self.max_backwards_s = max(self.max_backwards_s, -step)
                        self.max_forward_gap_s = max(self.max_forward_gap_s, step)
                    prev = ts
                    drift = (ts - pts).total_seconds()
                    self.max_abs_drift_s = max(self.max_abs_drift_s, abs(drift))
                    if not self._put(SegmentFrame(frame, ts, n, drift)):
                        break
                    n += 1
                cap.release()
                if self._stop.is_set() or (k and ts > self.end):
                    break
        finally:
            self._put(_END)
