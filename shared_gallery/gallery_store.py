"""Shared gallery on PostgreSQL: track-update batches in, global person IDs out.

Every camera keeps its own gallery and sends (camera_id, person_id, bbox, embedding | null) every tick (1/3 s).
This store maps each (camera_id, person_id) to a global_person_id with provisional IDs that are merged later:

  1. A (camera_id, person_id) seen for the first time gets a NEW global person at once, so everyone on screen has
     a global ID from the first tick, also people whose embedding is always null (too small to embed).
  2. Non-null embeddings go into that global person's bank (newest `bank_size` per camera).
  3. After every batch, two global persons are MERGED when
       - both banks hold >= `min_embeddings` embeddings,
       - the cosine of the banks' mean embeddings is >= `merge_threshold`, and
       - they were never visible on the same camera at the same time (one camera cannot see one person twice).
     Best pair first; the older (lower) ID survives, the other is recorded in global_alias.

Why not decide on the first embedding (match a new person against the banks once and keep it): on the
2026-11-11 15:33:42 clip a single cross-camera embedding of the blue-jacket walker scored 0.78 against himself and
up to 0.88 against a seated man in a similar navy shirt. The mean of the accumulated embeddings separates them:
0.92-0.96 for the true pair, <= 0.79 for every other cross-camera pair.

Matching runs in Python over the banks read from Postgres (no pgvector in the stock image); banks are small.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import psycopg

SCHEMA = Path(__file__).with_name("schema.sql")
DEFAULT_DSN = "postgresql://gallery:gallery@127.0.0.1:5433/shared_gallery"


@dataclass
class GalleryParams:
    merge_threshold: float = 0.85
    min_embeddings: int = 5
    bank_size: int = 20  # per global person per camera
    timezone: str = "Asia/Taipei"  # times returned by the store


class SharedGalleryStore:
    def __init__(self, dsn: str = DEFAULT_DSN, params: GalleryParams | None = None):
        self.p = params or GalleryParams()
        self.conn = psycopg.connect(dsn, autocommit=True)
        self.conn.execute(f"SET TIME ZONE '{self.p.timezone}'")
        self.conn.execute(SCHEMA.read_text())

    def reset(self) -> None:
        self.conn.execute(
            "TRUNCATE batch_log, global_alias, embedding_bank, person_map, global_person RESTART IDENTITY CASCADE"
        )

    # ---- one batch ----
    def submit(self, batch: dict) -> dict:
        tick = datetime.fromisoformat(batch["tick_ts"])
        recs = batch["records"]
        keys = [(r["camera_id"], r["person_id"]) for r in recs]
        status: dict[tuple, str] = {}
        merges: list[dict] = []
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO batch_log (tick_ts, records) VALUES (%s, %s) ON CONFLICT DO NOTHING", (tick, len(recs))
            )
            duplicate = cur.rowcount == 0
            if not duplicate:
                mapped = self._lookup(cur, keys)
                for r, k in zip(recs, keys):
                    g = mapped.get(k)
                    if g is None:
                        cur.execute(
                            "INSERT INTO global_person (created_ts, last_seen_ts) VALUES (%s, %s) "
                            "RETURNING global_person_id", (tick, tick))
                        g = cur.fetchone()[0]
                        cur.execute(
                            "INSERT INTO person_map (camera_id, person_id, global_person_id, first_seen_ts, "
                            "last_seen_ts) VALUES (%s, %s, %s, %s, %s)", (*k, g, tick, tick))
                        status[k] = "new"
                    else:
                        cur.execute("UPDATE person_map SET last_seen_ts = %s WHERE camera_id = %s AND person_id = %s",
                                    (tick, *k))
                        cur.execute("UPDATE global_person SET last_seen_ts = %s WHERE global_person_id = %s", (tick, g))
                        status[k] = "known"
                    if r["embedding"] is not None:
                        cur.execute(
                            "INSERT INTO embedding_bank (global_person_id, camera_id, person_id, ts, vector) "
                            "VALUES (%s, %s, %s, %s, %s)",
                            (g, *k, datetime.fromisoformat(r["timestamp"]), r["embedding"]))
                        self._trim(cur, g, k[0])
                merges = self._merge(cur, tick)
                cur.execute("UPDATE batch_log SET new_global = %s, merges = %s WHERE tick_ts = %s",
                            (sum(s == "new" for s in status.values()), len(merges), tick))
            current = self._lookup(cur, keys)
        assignments = [
            {"camera_id": k[0], "person_id": k[1], "global_person_id": current.get(k),
             "status": status.get(k, "duplicate_batch")}
            for k in keys
        ]
        return {"tick_ts": batch["tick_ts"], "duplicate": duplicate, "assignments": assignments, "merges": merges}

    def _merge(self, cur, tick) -> list[dict]:
        cur.execute("SELECT global_person_id, vector FROM embedding_bank")
        vecs = defaultdict(list)
        for g, v in cur.fetchall():
            vecs[g].append(v)
        means = {}
        for g, v in vecs.items():
            if len(v) >= self.p.min_embeddings:
                m = np.asarray(v, np.float32).mean(axis=0)
                means[g] = m / (np.linalg.norm(m) + 1e-12)
        if len(means) < 2:
            return []
        cur.execute("SELECT global_person_id, camera_id, first_seen_ts, last_seen_ts FROM person_map")
        spans = defaultdict(list)
        for g, cam, t0, t1 in cur.fetchall():
            spans[g].append((cam, t0, t1))

        def seen_together(a: int, b: int) -> bool:
            return any(ca == cb and not (a1 < b0 or b1 < a0) for ca, a0, a1 in spans[a] for cb, b0, b1 in spans[b])

        ids = sorted(means)
        pairs = []
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                s = float(means[a] @ means[b])
                if s >= self.p.merge_threshold and not seen_together(a, b):
                    pairs.append((s, a, b))
        done, out = set(), []
        for s, a, b in sorted(pairs, reverse=True):
            if a in done or b in done:
                continue
            keep, gone = min(a, b), max(a, b)
            cams_gone = sorted({c for c, _, _ in spans[gone]})
            cams_keep = sorted({c for c, _, _ in spans[keep]})
            cur.execute("UPDATE person_map SET global_person_id = %s WHERE global_person_id = %s", (keep, gone))
            cur.execute("UPDATE embedding_bank SET global_person_id = %s WHERE global_person_id = %s", (keep, gone))
            for cam in cams_gone:
                self._trim(cur, keep, cam)
            cur.execute(
                "UPDATE global_person k SET last_seen_ts = greatest(k.last_seen_ts, g.last_seen_ts), "
                "created_ts = least(k.created_ts, g.created_ts) FROM global_person g "
                "WHERE k.global_person_id = %s AND g.global_person_id = %s", (keep, gone))
            cur.execute("DELETE FROM global_person WHERE global_person_id = %s", (gone,))
            cur.execute("UPDATE global_alias SET to_global_id = %s WHERE to_global_id = %s", (keep, gone))
            cur.execute("INSERT INTO global_alias (from_global_id, to_global_id, merged_ts, similarity) "
                        "VALUES (%s, %s, %s, %s)", (gone, keep, tick, s))
            out.append({"from": gone, "into": keep, "similarity": round(s, 4),
                        "cross_camera": bool(set(cams_gone) - set(cams_keep) or set(cams_keep) - set(cams_gone))})
            done |= {a, b}
        return out

    # ---- helpers ----
    @staticmethod
    def _lookup(cur, keys: list[tuple]) -> dict[tuple, int]:
        if not keys:
            return {}
        cur.execute(
            "SELECT m.camera_id, m.person_id, m.global_person_id FROM person_map m "
            "JOIN unnest(%s::text[], %s::int[]) AS k(camera_id, person_id) USING (camera_id, person_id)",
            ([k[0] for k in keys], [k[1] for k in keys]),
        )
        return {(c, p): g for c, p, g in cur.fetchall()}

    def _trim(self, cur, g: int, camera_id: str) -> None:
        cur.execute(
            "DELETE FROM embedding_bank WHERE global_person_id = %s AND camera_id = %s AND id NOT IN "
            "(SELECT id FROM embedding_bank WHERE global_person_id = %s AND camera_id = %s ORDER BY id DESC LIMIT %s)",
            (g, camera_id, g, camera_id, self.p.bank_size),
        )

    # ---- read side ----
    def mapping(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT m.global_person_id, m.camera_id, m.person_id, m.first_seen_ts, m.last_seen_ts, "
            "(SELECT count(*) FROM embedding_bank e WHERE e.global_person_id = m.global_person_id) "
            "FROM person_map m ORDER BY 1, 2, 3"
        ).fetchall()
        cols = ["global_person_id", "camera_id", "person_id", "first_seen_ts", "last_seen_ts", "bank_size"]
        return [{c: (v.isoformat() if isinstance(v, datetime) else v) for c, v in zip(cols, row)} for row in rows]

    def aliases(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT from_global_id, to_global_id, merged_ts, similarity FROM global_alias ORDER BY merged_ts"
        ).fetchall()
        return [{"from": f, "into": t, "merged_ts": ts.isoformat(), "similarity": round(s, 4)} for f, t, ts, s in rows]

    def stats(self) -> dict:
        q = lambda sql: self.conn.execute(sql).fetchone()[0]  # noqa: E731
        return {
            "batches": q("SELECT count(*) FROM batch_log"),
            "global_persons": q("SELECT count(*) FROM global_person"),
            "local_persons_mapped": q("SELECT count(*) FROM person_map"),
            "cross_camera_global_persons": q(
                "SELECT count(*) FROM (SELECT global_person_id FROM person_map GROUP BY 1 "
                "HAVING count(DISTINCT camera_id) > 1) x"),
            "merges": q("SELECT count(*) FROM global_alias"),
            "embeddings": q("SELECT count(*) FROM embedding_bank"),
        }
