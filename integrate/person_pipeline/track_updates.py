"""Track updates, pipeline side: turn one frame's results into interface records and write tracklets.jsonl.

The interface itself (format, tick clock, validation) is in track_update_contract.py, shared with shared_gallery/.

id_type "pid" (default) writes that interface. id_type "tid" is for building tracklet data, not for the shared gallery:
one record per tracker track, with "track_id" in place of "person_id" (see records_of).
"""

from __future__ import annotations

import json
from datetime import datetime, tzinfo
from pathlib import Path

from .track_update_contract import EMBEDDING_DIM, TickClock, iso_time, validate_batch  # noqa: F401  (re-exported)
from .types import FrameResult


ID_KEY = {"pid": "person_id", "tid": "track_id"}


def records_of(result: FrameResult, camera_id: str, ts: datetime, tz: tzinfo, decimals: int = 6,
               id_type: str = "pid") -> list[dict]:
    """id_type "pid": one record per person_id in this frame. The gallery may bind two visible tracks to one person (a
    box inside another, e.g. upper body behind a partition next to a full-body box); then the larger box is kept.
    Tracks without a person_id are not sent.

    id_type "tid": one record per tracker track, keyed "track_id" (the tracker's ID, unique per camera within one
    run, restarting at 1 every run). Every track is sent, with or without a person_id, and nothing is merged."""
    if id_type not in ID_KEY:
        raise ValueError(f"id_type must be one of {sorted(ID_KEY)}, got {id_type!r}")
    key = ID_KEY[id_type]
    best: dict[int, dict] = {}
    for i, tid in enumerate(result.track_ids):
        pid = result.person_ids[i]
        if tid < 0 or (id_type == "pid" and pid is None):
            continue
        tid = int(tid)
        emb = result.embeddings.get(tid) if tid in result.clean_tracks else None
        x1, y1, x2, y2 = (round(float(v), 2) for v in result.detections.xyxy[i])
        oid = int(pid) if id_type == "pid" else tid
        rec = {
            "timestamp": iso_time(ts, tz),
            "camera_id": camera_id,
            key: oid,
            "bbox": [x1, y1, x2, y2],
            "embedding": None if emb is None else [round(float(v), decimals) for v in emb],
        }
        old = best.get(oid)
        if old is None or _area(rec["bbox"]) > _area(old["bbox"]):
            best[oid] = rec
    return [best[k] for k in sorted(best)]


def check_batch(batch: dict, tz: tzinfo) -> list[str]:
    """validate_batch for either id_type: a "track_id" record is checked like a "person_id" one (positive int,
    at most once per camera per batch)."""
    if isinstance(batch, dict) and isinstance(batch.get("records"), list):
        batch = {**batch, "records": [
            {("person_id" if k == "track_id" else k): v for k, v in r.items()} if isinstance(r, dict) else r
            for r in batch["records"]]}
    return validate_batch(batch, tz)


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
        problems = check_batch(json.loads(line), self.tz)
        if problems:
            raise ValueError(f"invalid batch at {iso_time(tick_ts, self.tz)}: {problems[:3]}")
        self._f.write(line + "\n")
        self.batches += 1

    def close(self) -> None:
        self._f.close()
