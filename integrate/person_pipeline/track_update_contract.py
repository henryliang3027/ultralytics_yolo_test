"""Track-update interface between the per-camera pipeline (integrate/) and the shared gallery (shared_gallery/).

Shared by both sides; standard library only, so the shared gallery can import it without torch, ultralytics or the
Re-ID model. Changing anything here changes the interface: agree on it first.

Agreed 2026-11-20:
  - Each camera keeps its own (separate) gallery; person_id is that camera's ID and restarts at 1 with the program.
  - A batch is sent every 5 frames of the 15 fps cameras = every 1/3 s, at .000 / .333 / .667 of every second.
    The tick is decided by time, not by counting frames, so both cameras send in the same batch even when they start
    at different times or drop frames (live).
  - On a tick frame every identified track is embedded (one schedule for all tracks and cameras).
  - One record per person that has a person_id; tracks without one yet are not sent.
  - embedding is null when the crop is not clean: box under 48 px high, detection confidence under 0.5, or IoU with
    another person over 0.3 (integrate/config.yaml reid section).

One JSON line per tick:
  {"tick_ts": "2026-11-11T15:33:42.333333+08:00",
   "records": [{"timestamp": "...+08:00", "camera_id": "51_203", "person_id": 2,
                "bbox": [x1, y1, x2, y2], "embedding": [256 floats] | null}, ...]}
  - Times: ISO 8601 in the site's local time with its offset (Asia/Taipei = +08:00), so they read like the video
    file names: frame 5 of 51_203_20261111_153342_... is 2026-11-11T15:33:42.333333+08:00.
    (The closed-tracklet contract, tracklets.jsonl, stays UTC.)
  - bbox: pixels of that camera's 3840x1920 frame.
  - embedding: 256 floats, L2-normalised, model resnet50_market1501_aicity156; or null.
  - A person appears at most once per camera per batch.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone, tzinfo
from fractions import Fraction
from pathlib import Path

EMBEDDING_DIM = 256
SITE_TIMEZONE = "Asia/Taipei"
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_US = timedelta(microseconds=1)

# Test clips: <camera_id>_<YYYYMMDD>_<HHMMSS start>_<HHMMSS end>.mp4, times local (SITE_TIMEZONE).
VIDEO_NAME = re.compile(r"^(?P<cam>[^_]+_[^_]+)_(?P<date>\d{8})_(?P<start>\d{6})_(?P<end>\d{6})$")


def parse_video_name(path: str | Path, tz: tzinfo) -> tuple[str, datetime] | None:
    """(camera_id, local start time) from a test clip's file name, or None if the name does not match."""
    m = VIDEO_NAME.match(Path(path).stem)
    if not m:
        return None
    return m["cam"], datetime.strptime(m["date"] + m["start"], "%Y%m%d%H%M%S").replace(tzinfo=tz)


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
        us = (ts - EPOCH) // _US + self.tol
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
    """Problems with one batch (one JSON line); empty list = valid."""
    p: list[str] = []
    if not isinstance(batch, dict) or set(batch) != {"tick_ts", "records"}:
        return [f"batch keys {sorted(batch) if isinstance(batch, dict) else type(batch).__name__}"]
    _time(batch["tick_ts"], "tick_ts", tz, p)
    if not isinstance(batch["records"], list):
        return p + ["records is not a list"]
    seen = set()
    for r in batch["records"]:
        if not isinstance(r, dict) or set(r) != {"timestamp", "camera_id", "person_id", "bbox", "embedding"}:
            p.append(f"record keys {sorted(r) if isinstance(r, dict) else type(r).__name__}")
            continue
        _time(r["timestamp"], "timestamp", tz, p)
        if not isinstance(r["camera_id"], str) or not r["camera_id"]:
            p.append("camera_id")
        if not isinstance(r["person_id"], int) or isinstance(r["person_id"], bool) or r["person_id"] < 1:
            p.append(f"person_id {r['person_id']!r}")
        if (r["camera_id"], r["person_id"]) in seen:
            p.append(f"person {r['camera_id']}/{r['person_id']} twice in one batch")
        seen.add((r["camera_id"], r["person_id"]))
        b = r["bbox"]
        if not (isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b)
                and b[2] > b[0] and b[3] > b[1]):
            p.append(f"bbox {b}")
        e = r["embedding"]
        if e is not None:
            if not isinstance(e, list) or len(e) != EMBEDDING_DIM:
                p.append(f"embedding dim {len(e) if isinstance(e, list) else type(e).__name__}")
            elif not math.isclose(math.fsum(x * x for x in e), 1.0, abs_tol=1e-3):
                p.append("embedding not L2-normalised")
    return p
