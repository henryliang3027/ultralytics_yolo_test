"""Track module: TrackTrack (CVPR 2025) called directly, without model.track().

TRACKTRACK.update() takes detections exposing .xywh / .conf / .cls plus the BGR frame, and returns an (M, 8) array
[x1, y1, x2, y2, track_id, score, cls, idx], where idx is the input detection row the track matched this frame.
"""

from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
from ultralytics.trackers.track_tracker import TRACKTRACK
from ultralytics.utils import YAML
from ultralytics.utils.checks import check_yaml

from .config import TrackerConfig
from .types import Detections


class PersonTracker:
    def __init__(self, cfg: TrackerConfig):
        args = YAML.load(check_yaml("tracktrack.yaml"))  # library defaults for every knob not in our config
        args.update(asdict(cfg))
        self._tracker = TRACKTRACK(SimpleNamespace(**args))

    def update(self, dets: Detections, image: np.ndarray) -> np.ndarray:
        """Advance one frame. Returns (N,) track id per detection row, -1 if that detection is not tracked."""
        xyxy = dets.xyxy
        xywh = np.empty_like(xyxy)
        xywh[:, :2] = (xyxy[:, :2] + xyxy[:, 2:]) / 2
        xywh[:, 2:] = xyxy[:, 2:] - xyxy[:, :2]

        out = self._tracker.update(SimpleNamespace(xywh=xywh, conf=dets.conf, cls=dets.cls), img=image)

        ids = np.full(len(dets), -1, dtype=int)
        for row in out:
            idx = int(row[7])
            if 0 <= idx < len(ids):
                ids[idx] = int(row[4])
        return ids
