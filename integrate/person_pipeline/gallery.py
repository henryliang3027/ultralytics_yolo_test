"""Gallery module: long-term person identities built from Re-ID embeddings.

The tracker gives short-term track IDs that change when a person is lost (occlusion, leaving and coming back).
The gallery keeps a bank of up to `max_embeddings_per_person` embeddings per person and binds each track ID to a
person ID:

- A track that is already bound keeps its person ID, and its new embeddings are added to that person's bank.
- A new track is compared with every person not currently visible under another track. If the best cosine
  similarity reaches `match_threshold`, the track takes over that person ID; otherwise a new person is created.
  Exception: when the other track's box mostly lies inside the new track's box (or the reverse), both are the same
  person seen twice (e.g. upper-body box while a partition hides the legs, then a new full-body box), so that
  person stays a candidate.
- A track with no embedding yet stays unidentified (None) until one arrives.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .config import GalleryConfig


@dataclass
class Identity:
    person_id: int
    bank: deque  # of (D,) L2-normalized embeddings, oldest dropped first
    last_seen: float
    track_ids: set = field(default_factory=set)  # every track ID ever bound to this person

    def similarity(self, emb: np.ndarray, metric: str) -> float:
        feats = np.stack(self.bank)
        if metric == "mean":
            centroid = feats.mean(axis=0)
            return float(centroid @ emb / (np.linalg.norm(centroid) + 1e-9))
        return float((feats @ emb).max())


class IdentityGallery:
    def __init__(self, cfg: GalleryConfig):
        if cfg.match_metric not in ("max", "mean"):
            raise ValueError(f"gallery.match_metric must be 'max' or 'mean', got {cfg.match_metric!r}")
        self.cfg = cfg
        self.identities: dict[int, Identity] = {}
        self._track_to_person: dict[int, int] = {}
        self._track_last_seen: dict[int, float] = {}
        self._next_person_id = 1

    # ---- queries ----
    def person_of(self, track_id: int) -> int | None:
        return self._track_to_person.get(track_id)

    def __len__(self) -> int:
        return len(self.identities)

    # ---- update ----
    def update(
        self,
        track_ids: list[int],
        embeddings: dict[int, np.ndarray],
        now: float,
        boxes: dict[int, np.ndarray] | None = None,
    ) -> dict[int, int | None]:
        """Process one frame.

        Args:
            track_ids: track IDs visible in this frame.
            embeddings: track ID -> (D,) embedding for the tracks that were embedded this frame.
            now: current time (seconds, monotonic).
            boxes: track ID -> xyxy box, used to spot two tracks on the same person.

        Returns:
            track ID -> person ID (None when the track has not been identified yet).
        """
        visible = set(track_ids)
        for tid in visible:
            self._track_last_seen[tid] = now

        # 1) Tracks already bound: refresh and grow the bank.
        unbound = []
        for tid in visible:
            pid = self._track_to_person.get(tid)
            if pid is None or pid not in self.identities:
                self._track_to_person.pop(tid, None)
                if tid in embeddings:
                    unbound.append(tid)
                continue
            ident = self.identities[pid]
            ident.last_seen = now
            if tid in embeddings:
                ident.bank.append(embeddings[tid])

        # 2) New tracks: greedy best-first match against people not visible under another track right now.
        if unbound:
            bound_visible = [t for t in visible if t in self._track_to_person]
            pairs = []
            for tid in unbound:
                occupied = {
                    self._track_to_person[t] for t in bound_visible if not self._same_person_boxes(boxes, tid, t)
                }
                for ident in self.identities.values():
                    if ident.person_id not in occupied:
                        pairs.append((ident.similarity(embeddings[tid], self.cfg.match_metric), tid, ident.person_id))
            pairs.sort(reverse=True)
            taken_tracks, taken_people = set(), set()
            for sim, tid, pid in pairs:
                if sim < self.cfg.match_threshold:
                    break
                if tid in taken_tracks or pid in taken_people:
                    continue
                self._bind(tid, pid, embeddings[tid], now)
                taken_tracks.add(tid)
                taken_people.add(pid)
            for tid in unbound:
                if tid not in taken_tracks:
                    self._bind(tid, self._new_identity(now), embeddings[tid], now)

        self._expire(now)
        return {tid: self._track_to_person.get(tid) for tid in track_ids}

    # ---- internals ----
    @staticmethod
    def _same_person_boxes(boxes: dict | None, a: int, b: int, min_containment: float = 0.7) -> bool:
        """True when the smaller of the two boxes lies mostly inside the larger one."""
        if not boxes or a not in boxes or b not in boxes:
            return False
        ax1, ay1, ax2, ay2 = boxes[a]
        bx1, by1, bx2, by2 = boxes[b]
        inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
        smaller = min((ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1))
        return smaller > 0 and inter / smaller >= min_containment

    def _new_identity(self, now: float) -> int:
        pid = self._next_person_id
        self._next_person_id += 1
        self.identities[pid] = Identity(pid, deque(maxlen=self.cfg.max_embeddings_per_person), now)
        return pid

    def _bind(self, tid: int, pid: int, emb: np.ndarray, now: float) -> None:
        ident = self.identities[pid]
        ident.bank.append(emb)
        ident.last_seen = now
        ident.track_ids.add(tid)
        self._track_to_person[tid] = pid

    def _expire(self, now: float) -> None:
        for tid in [t for t, seen in self._track_last_seen.items() if now - seen > self.cfg.track_timeout_s]:
            self._track_last_seen.pop(tid)
            self._track_to_person.pop(tid, None)
        for pid in [p for p, ident in self.identities.items() if now - ident.last_seen > self.cfg.identity_ttl_s]:
            self.identities.pop(pid)
