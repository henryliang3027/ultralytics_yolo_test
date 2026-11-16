"""Cupola360 EVENT API v1.2 sink: push each identified person's box and ID as ROIs.

Ported from py/MediaMTX/cupola_event.py. Transport: Socket.IO client on the Cupola server
(socketio_path='/socket'), emit 'registerMeasuringPoint' once per connection, then one JSON string per update on the
'serverData' event.

Each person becomes two ROIs (the spec's "box + name" sample):
  * box   (id = 2*pid):   Sphere bounds, 4 corners normalized 0..1 on the equirectangular frame.
  * label (id = 2*pid+1): child of the box (parentId must be < id), "ID <pid>" text above the box,
                          displayTime -1 = lives as long as its parent.
The box's displayTime is a short TTL refreshed on every update, so boxes fade out by themselves if this process
dies; people who leave between updates are also closed explicitly with displayTime 0.

publish() runs on the pipeline thread and only swaps a reference; all network I/O is on a daemon thread throttled to
`send_interval_s`, so a slow or unreachable server never stalls the pipeline.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone

from .config import CupolaConfig
from .sinks import ResultSink
from .types import FrameResult

API_VERSION = "1.2"
EVENT_NAME = "serverData"
SOCKETIO_PATH = "/socket"

BOX_APPEARANCE = {
    "borderColor": "rgba(0, 255, 51, 255)",
    "backgroundColor": "rgba(0, 0, 0, 0)",
    "borderSize": 3,
    "visibilityThreshold": 0,
}
LABEL_APPEARANCE = {
    "borderColor": "rgba(0, 0, 0, 0)",
    "backgroundColor": "rgba(0, 0, 0, 153)",
    "borderSize": 0,
    "visibilityThreshold": 0,
}
LABEL_FONT_COLOR = "rgba(0, 255, 51, 255)"
LABEL_FONT_SIZE = 14
LABEL_W_PX = 64  # >1 means pixels per the spec; AdjustBounds grows it to fit the text
LABEL_H_PX = 24


def _clamp01(v) -> float:
    return min(max(float(v), 0.0), 1.0)


def box_roi_id(pid: int) -> int:
    return 2 * pid


def label_roi_id(pid: int) -> int:
    return 2 * pid + 1


def build_rois(people, box_ttl_ms: int) -> list[dict]:
    """people: iterable of (pid, nx1, ny1, nx2, ny2), normalized 0..1. Returns roiList entries."""
    rois = []
    for pid, x1, y1, x2, y2 in people:
        x1, y1, x2, y2 = _clamp01(x1), _clamp01(y1), _clamp01(x2), _clamp01(y2)
        rois.append(
            {
                "id": box_roi_id(pid),
                "displayTime": box_ttl_ms,
                "appearance": BOX_APPEARANCE,
                "bounds": {
                    "type": "Sphere",
                    "points": [  # same corner order as the spec samples: LT, RT, LB, RB
                        {"x": x1, "y": y1},
                        {"x": x2, "y": y1},
                        {"x": x1, "y": y2},
                        {"x": x2, "y": y2},
                    ],
                },
            }
        )
        rois.append(
            {
                "id": label_roi_id(pid),
                "displayTime": -1,
                "parentId": box_roi_id(pid),
                "appearance": LABEL_APPEARANCE,
                "bounds": {
                    "type": "ViewportOffset",
                    "anchor": {"parent": "LeftTop", "self": "LeftBottom"},
                    "x": 0,
                    "y": 0,
                    "width": LABEL_W_PX,
                    "height": LABEL_H_PX,
                },
                "content": {
                    "type": "Info",
                    "text": f"ID {pid}",
                    "fontColor": LABEL_FONT_COLOR,
                    "fontSize": LABEL_FONT_SIZE,
                    "isBold": True,
                    "autoSizeMode": "AdjustBounds",
                },
            }
        )
    return rois


def build_close_rois(pids) -> list[dict]:
    """Closing the parent box also hides its label (child displayTime is -1)."""
    return [{"id": box_roi_id(pid), "displayTime": 0} for pid in pids]


def build_event(camera_url: str, resolution: str, roi_list: list, force_update: str | None = None) -> dict:
    now = datetime.now(timezone.utc)
    msg = {
        "version": API_VERSION,
        "url": camera_url,
        "resolution": resolution,
        "timestamp": now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z",
        "roiList": roi_list,
    }
    if force_update:
        msg["forceUpdate"] = force_update
    return msg


def people_from_result(result: FrameResult) -> list[tuple]:
    """Identified people of one frame as (pid, nx1, ny1, nx2, ny2), one entry per person ID.

    Two tracks can briefly share a person ID (old upper-body box + new full-body box); keep the larger box so the
    ROI id is unique in the message.
    """
    h, w = result.packet.image.shape[:2]
    best: dict[int, tuple] = {}
    for (x1, y1, x2, y2), pid in zip(result.detections.xyxy.tolist(), result.person_ids):
        if pid is None:
            continue
        area = (x2 - x1) * (y2 - y1)
        if pid not in best or area > best[pid][0]:
            best[pid] = (area, (pid, x1 / w, y1 / h, x2 / w, y2 / h))
    return [entry for _, entry in best.values()]


class CupolaEventSink(ResultSink):
    def __init__(self, cfg: CupolaConfig, camera_url: str, name: str = ""):
        self.cfg = cfg
        self.camera_url = camera_url  # Cupola camera uri / relayUri / path the boxes are drawn on
        self._tag = f"[cupola {name}]" if name else "[cupola]"
        self._latest = None  # list of (pid, nx1, ny1, nx2, ny2), swapped by reference
        self._resolution = cfg.resolution  # "auto" is replaced by the first frame's size
        self._shown: set[int] = set()  # person IDs currently displayed on the server
        self._need_clear = True  # clear stale ROIs from a previous run on (re)connect
        self._stop = threading.Event()
        self._sio = None
        self._thread = None

    def start(self) -> bool:
        if not self.camera_url:
            print(f"{self._tag} cupola_camera_url is empty, Cupola push disabled")
            return False
        try:
            import socketio
        except ImportError:
            print(f"{self._tag} python-socketio not installed, Cupola push disabled (see README)")
            return False

        self._sio = socketio.Client(reconnection=True, ssl_verify=False, handle_sigint=False)

        @self._sio.event
        def connect():
            print(f"{self._tag} connected to {self.cfg.server}, drawing on {self.camera_url}")
            self._sio.emit("registerMeasuringPoint")
            self._need_clear = True

        @self._sio.event
        def disconnect(*_):
            print(f"{self._tag} disconnected")

        self._thread = threading.Thread(target=self._run, name="cupola-sender", daemon=True)
        self._thread.start()
        return True

    def publish(self, result: FrameResult) -> None:
        if self._resolution == "auto":
            h, w = result.packet.image.shape[:2]
            self._resolution = f"{w}x{h}"
        self._latest = people_from_result(result)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._sio is not None and self._sio.connected:
            try:
                self._emit(build_event(self.camera_url, self._resolution, [], "SINGLE"))
                self._sio.sleep(0.3)  # emit only queues the packet; let it flush
            except Exception:
                pass
            self._sio.disconnect()

    # ---- network thread ----
    def _emit(self, msg: dict) -> None:
        self._sio.emit(EVENT_NAME, json.dumps(msg, separators=(",", ":")))

    def _connect_loop(self) -> bool:
        while not self._stop.is_set():
            try:
                self._sio.connect(self.cfg.server, socketio_path=SOCKETIO_PATH, wait_timeout=5)
                return True
            except Exception as e:
                print(f"{self._tag} connect to {self.cfg.server} failed: {e}; retrying in 3s")
                self._stop.wait(3)
        return False

    def _run(self) -> None:
        if not self._connect_loop():
            return
        while not self._stop.wait(self.cfg.send_interval_s):
            if not self._sio.connected or self._resolution == "auto":
                continue  # socketio.Client reconnects on its own; resolution known after the first frame
            try:
                if self._need_clear:
                    self._emit(build_event(self.camera_url, self._resolution, [], "SINGLE"))
                    self._shown = set()
                    self._need_clear = False

                people = self._latest
                if people is None:
                    continue
                self._latest = None

                current = {p[0] for p in people}
                roi_list = build_close_rois(sorted(self._shown - current)) + build_rois(people, self.cfg.box_ttl_ms)
                if roi_list:
                    self._emit(build_event(self.camera_url, self._resolution, roi_list))
                self._shown = current
            except Exception as e:
                print(f"{self._tag} send error: {e}")
