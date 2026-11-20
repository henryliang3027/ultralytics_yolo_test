"""Drawing module: boxes, IDs and a status line on the frame."""

from __future__ import annotations

import cv2
import numpy as np

from .types import FrameResult

GREEN = (0, 255, 0)  # YOLO detection
ORANGE = (0, 165, 255)  # identified person
GRAY = (170, 170, 170)  # tracked, not identified yet
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _label(img, text, x, y, color, scale=1.0, above=True):
    (tw, th), base = cv2.getTextSize(text, FONT, scale, 2)
    ty = y - base - 4 if above and y - th - base - 4 > 0 else y + th + 4
    cv2.rectangle(img, (x, ty - th - 4), (x + tw, ty + base), color, -1)
    cv2.putText(img, text, (x, ty), FONT, scale, (0, 0, 0), 2, cv2.LINE_AA)


def person_color(pid: int) -> tuple:
    rng = np.random.default_rng(pid)
    return tuple(int(c) for c in rng.integers(80, 256, 3))


class Visualizer:
    def __init__(self, draw_detections: bool = True, global_ids: bool = False):
        self.draw_detections = draw_detections
        self.global_ids = global_ids  # shared gallery: label people G001, G002, ... (same on every camera)

    def draw(self, result: FrameResult, status: str = "") -> np.ndarray:
        img = result.packet.image.copy()
        dets = result.detections
        for i in np.argsort(dets.conf):  # low score first, so the best label ends on top
            x1, y1, x2, y2 = map(int, dets.xyxy[i])
            if self.draw_detections:
                cv2.rectangle(img, (x1, y1), (x2, y2), GREEN, 1)
                _label(img, f"{dets.conf[i]:.2f}", x1, y1, GREEN, 0.8, above=True)
            tid, pid = int(result.track_ids[i]), result.person_ids[i]
            if tid < 0:
                continue
            color = person_color(pid) if pid is not None else GRAY
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            if pid is None:
                text = f"T{tid} ?"
            else:
                text = f"G{pid:03d}" if self.global_ids else f"ID {pid}"
            _label(img, text, x1, y2, color, 1.2, above=False)

        if status:
            (tw, th), base = cv2.getTextSize(status, FONT, 1.5, 3)
            x = img.shape[1] - tw - 20
            cv2.rectangle(img, (x - 10, 10), (x + tw + 10, 30 + th + base), (0, 0, 0), -1)
            cv2.putText(img, status, (x, 20 + th), FONT, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
        return img
