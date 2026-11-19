"""Record mode output: turn per-frame results into closed tracklets (tracklet_schema v0.2) and write tracklets.jsonl.

A tracklet is one raw tracker track of one camera. It closes when the tracker would delete it: unseen for more than
`track_buffer` processed frames (30 frames at the recordings' 15 fps = 2 s). Until then TrackTrack can still re-find
it under the same ID, so closing earlier would split one track in two. Nothing is stitched or merged here.

local_track_id = per-run offset + tracker ID. TrackTrack numbers tracks from 1 in every new process, so the offset
comes from a small registry file (per camera, per local day) to keep (camera_id, local_track_id) unique within a day.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from .tracklet_contract import Embedding, Frame, Tracklet
from .types import FrameResult

EXPECTED_DIM = 256


@dataclass
class _OpenTrack:
    frames: list = field(default_factory=list)  # Frame
    embeddings: list = field(default_factory=list)  # (ts, vector, det_conf, box_h)
    person_ids: Counter = field(default_factory=Counter)
    last_seen: int = 0  # processed-frame index


def pick_evenly(items: list, k: int) -> list:
    """Up to k items spread evenly over the list (always keeps the first and the last)."""
    if len(items) <= k:
        return list(items)
    idx = np.unique(np.round(np.linspace(0, len(items) - 1, k)).astype(int))
    return [items[i] for i in idx]


class TrackIdRegistry:
    """Next free local_track_id per camera per local day, persisted as JSON so new processes do not reuse IDs."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data: dict = json.loads(self.path.read_text()) if self.path.exists() else {}

    def offset(self, camera_id: str, days: list[date]) -> int:
        cam = self.data.get(camera_id, {})
        return max((cam.get(d.isoformat(), 0) for d in days), default=0)

    def commit(self, camera_id: str, days: list[date], next_free: int) -> None:
        cam = self.data.setdefault(camera_id, {})
        for d in days:
            cam[d.isoformat()] = max(cam.get(d.isoformat(), 0), next_free)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, sort_keys=True))
        os.replace(tmp, self.path)


class TrackletWriter:
    """Shared JSONL writer: validates every line against the contract before writing it."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w")
        self.written = 0

    def write(self, tracklet: Tracklet) -> None:
        line = tracklet.model_dump_json()
        Tracklet.model_validate_json(line)  # round trip: exactly what multi_cam will parse
        self._f.write(line + "\n")
        self.written += 1

    def close(self) -> None:
        self._f.close()


class TrackletBuilder:
    """Collects one camera's per-frame results into tracklets and hands closed ones to the writer."""

    def __init__(
        self,
        camera_id: str,
        writer: TrackletWriter,
        reid_model_version: str,
        id_offset: int,
        close_after_frames: int,
        max_embeddings: int = 20,
        vector_decimals: int = 6,
    ):
        self.camera_id = camera_id
        self.writer = writer
        self.reid_model_version = reid_model_version
        self.id_offset = id_offset
        self.close_after = close_after_frames
        self.max_embeddings = max_embeddings
        self.decimals = vector_decimals
        self._open: dict[int, _OpenTrack] = {}
        self._frame_idx = 0
        self.max_tracker_id = 0
        self.stats = Counter()  # written, dropped_no_embedding, invalid, ...
        self.errors: list[str] = []

    def update(self, result: FrameResult, ts: datetime) -> None:
        dets = result.detections
        row_of = {}
        for i, tid in enumerate(result.track_ids):
            if tid < 0:
                continue
            tid = int(tid)
            row_of[tid] = i
            x1, y1, x2, y2 = (float(v) for v in dets.xyxy[i])
            tr = self._open.get(tid)
            if tr is None:
                tr = self._open[tid] = _OpenTrack()
                self.max_tracker_id = max(self.max_tracker_id, tid)
            tr.frames.append(
                Frame(ts=ts, bbox=(round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)), foot_px=(round((x1 + x2) / 2, 2), round(y2, 2)))
            )
            tr.last_seen = self._frame_idx
            pid = result.person_ids[i]
            if pid is not None:
                tr.person_ids[pid] += 1
        for tid, vec in result.embeddings.items():
            i = row_of.get(int(tid))
            if i is None:
                continue
            x1, y1, x2, y2 = dets.xyxy[i]
            self._open[int(tid)].embeddings.append((ts, vec, float(dets.conf[i]), float(y2 - y1)))

        for tid in [t for t, tr in self._open.items() if self._frame_idx - tr.last_seen > self.close_after]:
            self._close(tid)
        self._frame_idx += 1

    def flush(self) -> None:
        """End of the recording window: close every open track (they end at the window edge, not by being lost)."""
        for tid in list(self._open):
            self._close(tid)

    @property
    def next_free_id(self) -> int:
        return self.id_offset + self.max_tracker_id + 1

    def _close(self, tid: int) -> None:
        tr = self._open.pop(tid)
        if not tr.embeddings:
            self.stats["dropped_no_embedding"] += 1  # contract needs >= 1; crop was too small every frame
            return
        embs = []
        for ts, vec, conf, h in pick_evenly(tr.embeddings, self.max_embeddings):
            v = np.asarray(vec, np.float64)
            v = v / (np.linalg.norm(v) + 1e-12)
            embs.append(Embedding(ts=ts, vector=[round(float(x), self.decimals) for x in v], det_conf=min(max(conf, 0.0), 1.0), box_h_px=h))
        try:
            t = Tracklet(
                camera_id=self.camera_id,
                local_track_id=self.id_offset + tid,
                start_ts=min(f.ts for f in tr.frames),  # min/max: segment overlap can step time back slightly
                end_ts=max(f.ts for f in tr.frames),
                reid_model_version=self.reid_model_version,
                camera_person_id=tr.person_ids.most_common(1)[0][0] if tr.person_ids else None,
                frames=tr.frames,
                embeddings=embs,
            )
            self._check_extra(t)
            self.writer.write(t)
        except Exception as e:  # validation failure: count it, keep going, report it
            self.stats["invalid"] += 1
            if len(self.errors) < 20:
                self.errors.append(f"{self.camera_id}/{self.id_offset + tid}: {e}")
            return
        self.stats["written"] += 1
        self.stats["frames"] += len(t.frames)
        self.stats["embeddings"] += len(t.embeddings)

    def _check_extra(self, t: Tracklet) -> None:
        """Rules stated in the contract text but not enforced by the pydantic models."""
        for e in t.embeddings:
            if len(e.vector) != EXPECTED_DIM:
                raise ValueError(f"embedding dim {len(e.vector)} != {EXPECTED_DIM}")
            if not math.isclose(math.fsum(x * x for x in e.vector), 1.0, abs_tol=1e-3):
                raise ValueError("embedding not L2-normalised")
        for ts in [t.start_ts, t.end_ts] + [f.ts for f in t.frames] + [e.ts for e in t.embeddings]:
            if ts.utcoffset() is None or ts.utcoffset().total_seconds() != 0:
                raise ValueError(f"timestamp {ts} is not UTC")


def local_days(start: datetime, end: datetime, tz: ZoneInfo) -> list[date]:
    """Local calendar days touched by [start, end] (UTC datetimes)."""
    d0, d1 = start.astimezone(tz).date(), end.astimezone(tz).date()
    return [date.fromordinal(o) for o in range(d0.toordinal(), d1.toordinal() + 1)]
