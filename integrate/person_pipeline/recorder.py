"""Video recording on a background thread, so encoding never blocks the detection loop.

cv2.VideoWriter.write() compresses each frame on the CPU and only returns when it is done; called inline it adds its
whole cost to every frame of the pipeline. Here the loop only puts the frame on a queue and a worker thread encodes it
(OpenCV releases the GIL while encoding, so the two really run in parallel).
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path

import cv2
import numpy as np

QUEUE_SIZE = 8  # frames waiting to be encoded; full -> the loop waits (no frame is ever dropped from the video)


class VideoRecorder:
    def __init__(self, path: str, fps: float, size: tuple[int, int], name: str = ""):
        """`size` is (width, height). Raises right away if the file cannot be opened, not later on the thread."""
        self.path = path
        self.name = name
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not self._writer.isOpened():  # OpenCV fails silently otherwise
            raise RuntimeError(f"[{name}] cannot open video writer for {path}")
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=QUEUE_SIZE)
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name=f"recorder-{name}", daemon=True)
        self._thread.start()
        print(f"[{name}] recording to {path}")

    def write(self, frame: np.ndarray) -> None:
        """Queue one frame. The caller must not draw on `frame` afterwards (the thread reads it later)."""
        if self._error is not None:
            raise RuntimeError(f"[{self.name}] video writer failed: {self._error}") from self._error
        self._queue.put(frame)  # blocks only if the encoder is QUEUE_SIZE frames behind

    def close(self) -> None:
        """Encode every frame still queued, then finalize the file (an unfinalized .mp4 does not play)."""
        if self._thread.is_alive():
            self._queue.put(None)
            self._thread.join()

    def _run(self) -> None:
        try:
            while (frame := self._queue.get()) is not None:
                self._writer.write(frame)
        except BaseException as e:  # surfaced to the loop on its next write()
            self._error = e
            while self._queue.get() is not None:  # keep draining so write()/close() never block forever
                pass
        finally:
            self._writer.release()
