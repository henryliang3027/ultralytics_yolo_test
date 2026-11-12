"""Result sinks (Observer pattern): the pipeline hands every FrameResult to each registered sink.

A sink must never block the pipeline: do network or disk I/O on its own thread and only hand data over in publish().
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .types import FrameResult


class ResultSink(ABC):
    def start(self) -> bool:
        """Prepare the sink; return False to have the pipeline skip it."""
        return True

    @abstractmethod
    def publish(self, result: FrameResult) -> None:
        """Receive one frame's result. Called on the pipeline thread; must return quickly."""

    def stop(self) -> None:
        """Flush and release resources."""
