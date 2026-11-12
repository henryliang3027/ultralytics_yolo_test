"""Frame sources (Strategy pattern): every backend decodes in a background thread and keeps only the newest frame.

Keeping one frame instead of a queue is what keeps latency flat: when inference is slower than the stream, old frames
are dropped instead of piling up, so the pipeline always works on the most recent image.
"""

from __future__ import annotations

import os
import threading
import time
from abc import ABC, abstractmethod

import cv2

from .config import SourceConfig
from .types import FramePacket


class FrameSource(ABC):
    """Background decoder that exposes the latest frame."""

    def __init__(self, cfg: SourceConfig):
        self.cfg = cfg
        self._cond = threading.Condition()
        self._latest: FramePacket | None = None
        self._last_returned_id = -1
        self._frame_count = 0
        self._running = False
        self._finished = False  # file sources: reached end of file
        self._thread: threading.Thread | None = None
        self.fps: float = 0.0  # stream fps reported by the source (0 if unknown)

    # ---- public API ----
    def start(self) -> FrameSource:
        self._running = True
        self._thread = threading.Thread(target=self._run, name=type(self).__name__, daemon=True)
        self._thread.start()
        return self

    def read(self, timeout: float = 1.0) -> FramePacket | None:
        """Block until a frame newer than the last one returned arrives; None on timeout or end of stream."""
        with self._cond:
            ok = self._cond.wait_for(
                lambda: (self._latest is not None and self._latest.frame_id != self._last_returned_id)
                or self._finished
                or not self._running,
                timeout,
            )
            if not ok or self._latest is None or self._latest.frame_id == self._last_returned_id:
                return None
            self._last_returned_id = self._latest.frame_id
            return self._latest

    @property
    def finished(self) -> bool:
        return self._finished

    def stop(self) -> None:
        self._running = False
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=5)

    # ---- for subclasses ----
    def _publish(self, image) -> None:
        packet = FramePacket(image=image, frame_id=self._frame_count, timestamp=time.monotonic())
        self._frame_count += 1
        with self._cond:
            self._latest = packet  # replace, never queue
            self._cond.notify_all()

    def _finish(self) -> None:
        with self._cond:
            self._finished = True
            self._cond.notify_all()

    @abstractmethod
    def _run(self) -> None:
        """Decode loop; call self._publish(bgr_image) for every frame until self._running is False."""


class OpenCVSource(FrameSource):
    """FFmpeg through cv2.VideoCapture. Works with the pip OpenCV build; reconnects when the stream drops.

    A local file is played at its own fps (not as fast as possible) so offline tests behave like a live stream.
    """

    def _open(self) -> cv2.VideoCapture:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            f"rtsp_transport;{self.cfg.transport}|fflags;nobuffer|flags;low_delay"
        )
        cap = cv2.VideoCapture(self.cfg.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # ignored by some FFmpeg builds; the drop-old-frames design covers it
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        return cap

    def _run(self) -> None:
        is_file = os.path.isfile(self.cfg.url)
        while self._running:
            cap = self._open()
            if not cap.isOpened():
                print(f"[capture] cannot open {self.cfg.url}, retrying in {self.cfg.reconnect_delay_s}s")
                time.sleep(self.cfg.reconnect_delay_s)
                continue
            print(f"[capture] opened {self.cfg.url} ({self.fps:.1f} fps, opencv/ffmpeg)")
            period = 1.0 / self.fps if is_file and self.fps > 0 else 0.0
            next_t = time.monotonic()
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    break
                self._publish(frame)
                if period:  # pace file playback to real time
                    next_t += period
                    time.sleep(max(0.0, next_t - time.monotonic()))
            cap.release()
            if is_file:
                print("[capture] end of file")
                self._finish()
                return
            if self._running:
                print(f"[capture] stream lost, reconnecting in {self.cfg.reconnect_delay_s}s")
                time.sleep(self.cfg.reconnect_delay_s)


class GStreamerSource(FrameSource):
    """GStreamer through PyGObject: rtspsrc latency=0 + appsink drop=true max-buffers=1.

    Requires PyGObject in this Python environment and the decoder plugins named in `gst_decode` (see README).
    """

    def _pipeline_str(self) -> str:
        return (
            f"rtspsrc location={self.cfg.url} latency=0 protocols={self.cfg.transport} drop-on-latency=true ! "
            f"{self.cfg.gst_decode} ! videoconvert ! video/x-raw,format=BGR ! "
            "appsink name=sink emit-signals=false sync=false max-buffers=1 drop=true"
        )

    def _run(self) -> None:
        import gi
        import numpy as np

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        Gst.init(None)
        while self._running:
            pipeline = Gst.parse_launch(self._pipeline_str())
            sink = pipeline.get_by_name("sink")
            pipeline.set_state(Gst.State.PLAYING)
            bus = pipeline.get_bus()
            print(f"[capture] gstreamer pipeline: {self._pipeline_str()}")
            while self._running:
                msg = bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
                if msg is not None:
                    if msg.type == Gst.MessageType.ERROR:
                        err, dbg = msg.parse_error()
                        print(f"[capture] gstreamer error: {err} ({dbg})")
                    break
                sample = sink.emit("try-pull-sample", 500 * Gst.MSECOND)
                if sample is None:
                    continue
                buf = sample.get_buffer()
                s = sample.get_caps().get_structure(0)
                w, h = s.get_value("width"), s.get_value("height")
                ok, info = buf.map(Gst.MapFlags.READ)
                if not ok:
                    continue
                try:
                    stride = info.size // h  # rows can be padded
                    rows = np.frombuffer(info.data, np.uint8).reshape(h, stride)
                    self._publish(rows[:, : w * 3].reshape(h, w, 3).copy())
                finally:
                    buf.unmap(info)
            pipeline.set_state(Gst.State.NULL)
            if self._running:
                print(f"[capture] stream lost, reconnecting in {self.cfg.reconnect_delay_s}s")
                time.sleep(self.cfg.reconnect_delay_s)


_BACKENDS = {"opencv": OpenCVSource, "gstreamer": GStreamerSource}


def create_frame_source(cfg: SourceConfig) -> FrameSource:
    """Factory: pick the backend named in config."""
    try:
        return _BACKENDS[cfg.backend](cfg)
    except KeyError:
        raise ValueError(f"unknown source.backend {cfg.backend!r}, choose from {list(_BACKENDS)}") from None
