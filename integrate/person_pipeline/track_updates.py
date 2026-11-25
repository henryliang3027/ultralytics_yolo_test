"""Track updates, pipeline side: turn one frame's results into interface records and write tracklets.jsonl.

The interface itself (format, tick clock, validation) is in track_update_contract.py, shared with shared_gallery/.

id_type "pid" (default) writes that interface. id_type "tid" is for building tracklet data, not for the shared gallery:
one record per tracker track, with "track_id" in place of "person_id". id_type "both" is "tid" plus the track's
"person_id" (null while it has none) in every record (see records_of).
"""

from __future__ import annotations

import json
from datetime import datetime, tzinfo
from pathlib import Path

from .track_update_contract import EMBEDDING_DIM, TickClock, iso_time, validate_batch  # noqa: F401  (re-exported)
from .types import FrameResult


ID_KEY = {"pid": "person_id", "tid": "track_id", "both": "track_id"}  # the key records are unique by, per camera


def records_of(result: FrameResult, camera_id: str, ts: datetime, tz: tzinfo, decimals: int = 6,
               id_type: str = "pid") -> list[dict]:
    """id_type "pid": one record per person_id in this frame. The gallery may bind two visible tracks to one person (a
    box inside another, e.g. upper body behind a partition next to a full-body box); then the larger box is kept.
    Tracks without a person_id are not sent.

    id_type "tid": one record per tracker track, keyed "track_id" (the tracker's ID, unique per camera within one
    run, restarting at 1 every run). Every track is sent, with or without a person_id, and nothing is merged.

    id_type "both": as "tid", and each record also carries "person_id" (the gallery's ID for that track, None while it
    has none). Two tracks bound to one person give two records with the same person_id."""
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
        rec = {"timestamp": iso_time(ts, tz), "camera_id": camera_id, key: oid}
        if id_type == "both":
            rec["person_id"] = None if pid is None else int(pid)
        rec["bbox"] = [x1, y1, x2, y2]
        rec["embedding"] = None if emb is None else [round(float(v), decimals) for v in emb]
        old = best.get(oid)
        if old is None or _area(rec["bbox"]) > _area(old["bbox"]):
            best[oid] = rec
    return [best[k] for k in sorted(best)]


def check_batch(batch: dict, tz: tzinfo) -> list[str]:
    """validate_batch for any id_type: a "track_id" record is checked like a "person_id" one (positive int, at most
    once per camera per batch). A "both" record (track_id and person_id) also needs person_id None or a positive int;
    it may repeat within a batch."""
    extra: list[str] = []
    if isinstance(batch, dict) and isinstance(batch.get("records"), list):
        records = []
        for r in batch["records"]:
            if isinstance(r, dict) and "track_id" in r:
                r = dict(r)
                if "person_id" in r:
                    pid = r.pop("person_id")
                    if pid is not None and (not isinstance(pid, int) or isinstance(pid, bool) or pid < 1):
                        extra.append(f"person_id {pid!r}")
                r["person_id"] = r.pop("track_id")
            records.append(r)
        batch = {**batch, "records": records}
    return validate_batch(batch, tz) + extra


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
