"""Track updates, pipeline side: turn one frame's results into interface records and write track_updates.jsonl.

The interface itself (format, tick clock, validation) is in track_update_contract.py, shared with shared_gallery/.
"""

from __future__ import annotations

import json
from datetime import datetime, tzinfo
from pathlib import Path

from .track_update_contract import EMBEDDING_DIM, TickClock, iso_time, validate_batch  # noqa: F401  (re-exported)
from .types import FrameResult


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
