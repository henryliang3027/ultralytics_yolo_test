"""Track updates: every identified person on every camera, sent together at fixed clock ticks.

Interface agreed with the cross-camera side (2026-11-20):
  - Each camera keeps its own (separate) gallery; person_id is that camera's ID and restarts at 1 with the program.
  - A batch is sent every `interval` frames of the 15 fps cameras: 5 frames = every 1/3 s, at .000 / .333 / .667 of
    every second (UTC and Taipei differ by whole hours: same boundaries). The tick is decided by time, not by counting frames, so both cameras send in the same batch
    even when they start at different times or drop frames (live).
  - On a tick frame every identified track is embedded (one schedule for all tracks, see CameraStream.process).
  - One record per person that has a person_id; tracks without one yet are not sent.
  - embedding is null when the crop is not clean (box under reid.min_box_height, detection confidence under
    reid.min_conf, or IoU with another person over reid.max_overlap_iou).

One JSON line per tick:
  {"tick_ts": "...+08:00", "records": [{"timestamp", "camera_id", "person_id", "bbox", "embedding"}, ...]}
All times are ISO 8601 in the site's local time with its offset (record.timezone, Asia/Taipei = +08:00), so
they read like the video file names: frame 5 of 51_203_20261111_153342_... is 2026-11-11T15:33:42.333333+08:00.
(The closed-tracklet contract, tracklets.jsonl, stays UTC.) bbox is [x1, y1, x2, y2] in pixels of that camera's frame. embedding is 256 floats,
L2-normalised (resnet50_market1501_aicity156), or null.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone, tzinfo
from fractions import Fraction
from pathlib import Path

from .types import FrameResult

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
US = timedelta(microseconds=1)
EXPECTED_DIM = 256


def iso_time(ts: datetime, tz: tzinfo) -> str:
    """ISO 8601 in `tz` with its offset, e.g. 2026-11-11T15:33:42.333333+08:00."""
    return ts.astimezone(tz).isoformat(timespec="microseconds")


class TickClock:
    """Send times every `period` seconds, anchored at whole UTC seconds (so every camera and run agree).

    Frame times are rounded to microseconds (k / 15 s is not exact), so a frame up to `tolerance_us` before a tick
    boundary counts as reaching it; integers only, no float comparisons.
    """

    def __init__(self, period: Fraction, tolerance_us: int = 1000):
        if period <= 0:
            raise ValueError("tick period must be positive")
        self.period = period
        self.tol = tolerance_us
        self._last: int | None = None

    def index(self, ts: datetime) -> int:
        us = (ts - EPOCH) // US + self.tol
        return (us * self.period.denominator) // (self.period.numerator * 1_000_000)

    def time_of(self, n: int) -> datetime:
        return EPOCH + timedelta(microseconds=round(n * self.period * 1_000_000))

    def due(self, ts: datetime) -> int | None:
        """Tick index if `ts` reached a new tick since the last call, else None."""
        n = self.index(ts)
        if self._last is None or n > self._last:
            self._last = n
            return n
        return None


def records_of(result: FrameResult, camera_id: str, ts: datetime, tz: tzinfo, decimals: int = 6) -> list[dict]:
    """One record per person_id in this frame. The gallery may bind two visible tracks to one person (a box inside
    another, e.g. upper body behind a partition next to a full-body box); then the larger box is kept."""
    best: dict[int, dict] = {}
    for i, tid in enumerate(result.track_ids):
        pid = result.person_ids[i]
        if tid < 0 or pid is None:
            continue
        tid = int(tid)
        emb = result.embeddings.get(tid) if tid in result.clean_tracks else None
        x1, y1, x2, y2 = (round(float(v), 2) for v in result.detections.xyxy[i])
        rec = {
            "timestamp": iso_time(ts, tz),
            "camera_id": camera_id,
            "person_id": int(pid),
            "bbox": [x1, y1, x2, y2],
            "embedding": None if emb is None else [round(float(v), decimals) for v in emb],
        }
        old = best.get(int(pid))
        if old is None or _area(rec["bbox"]) > _area(old["bbox"]):
            best[int(pid)] = rec
    return [best[p] for p in sorted(best)]


def _area(b: list[float]) -> float:
    return (b[2] - b[0]) * (b[3] - b[1])


class TrackUpdateWriter:
    def __init__(self, path: str | Path, tz: tzinfo):
        self.path = Path(path)
        self.tz = tz
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w")
        self.batches = 0

    def write(self, tick_ts: datetime, records: list[dict]) -> None:
        line = json.dumps({"tick_ts": iso_time(tick_ts, self.tz), "records": records}, separators=(",", ":"))
        problems = validate_batch(json.loads(line), self.tz)
        if problems:
            raise ValueError(f"invalid batch at {iso_time(tick_ts, self.tz)}: {problems[:3]}")
        self._f.write(line + "\n")
        self.batches += 1

    def close(self) -> None:
        self._f.close()


def _time(s, what: str, tz: tzinfo, problems: list) -> datetime | None:
    try:
        ts = datetime.fromisoformat(s)
    except (TypeError, ValueError):
        problems.append(f"{what}: not ISO 8601")
        return None
    if ts.utcoffset() is None:
        problems.append(f"{what}: no UTC offset")
    elif ts.utcoffset() != ts.astimezone(tz).utcoffset():
        problems.append(f"{what}: offset {ts.utcoffset()} is not {tz}")
    return ts


def validate_batch(batch: dict, tz: tzinfo) -> list[str]:
    """Problems with one batch line (empty list = valid)."""
    p: list[str] = []
    if set(batch) != {"tick_ts", "records"}:
        p.append(f"batch keys {sorted(batch)}")
        return p
    _time(batch["tick_ts"], "tick_ts", tz, p)
    if not isinstance(batch["records"], list):
        return p + ["records is not a list"]
    seen = set()
    for r in batch["records"]:
        if set(r) != {"timestamp", "camera_id", "person_id", "bbox", "embedding"}:
            p.append(f"record keys {sorted(r)}")
            continue
        _time(r["timestamp"], "timestamp", tz, p)
        if not isinstance(r["camera_id"], str) or not r["camera_id"]:
            p.append("camera_id")
        if not isinstance(r["person_id"], int) or r["person_id"] < 1:
            p.append(f"person_id {r['person_id']!r}")
        if (r["camera_id"], r["person_id"]) in seen:
            p.append(f"person {r['camera_id']}/{r['person_id']} twice in one batch")
        seen.add((r["camera_id"], r["person_id"]))
        b = r["bbox"]
        if not (isinstance(b, list) and len(b) == 4 and b[2] > b[0] and b[3] > b[1]):
            p.append(f"bbox {b}")
        e = r["embedding"]
        if e is not None:
            if not isinstance(e, list) or len(e) != EXPECTED_DIM:
                p.append(f"embedding dim {len(e) if isinstance(e, list) else type(e).__name__}")
            elif not math.isclose(math.fsum(x * x for x in e), 1.0, abs_tol=1e-3):
                p.append("embedding not L2-normalised")
    return p
