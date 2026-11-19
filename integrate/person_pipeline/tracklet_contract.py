"""Shared tracklet contract (tracklet_schema v0.2) used by record mode.

The source of truth is `packages/tracklet_schema` in github.com/entienlin/reid-heatmap. When that package is
installed (`pip install -e <reid-heatmap>/packages/tracklet_schema`) it is used directly, so every line is validated
against the real models. Otherwise the local copy below is used; it is a verbatim copy of the repo's v0.2 models and
of `cupola_time`. Do NOT change it here: contract changes are agreed with the multi_cam owner, made in their repo,
and then copied over. `SOURCE` says which one is active.
"""

from __future__ import annotations

try:
    from tracklet_schema import SCHEMA_VERSION, Embedding, Frame, Tracklet
    from tracklet_schema.cupola_time import frame_utc, segment_start_utc

    SOURCE = "tracklet_schema package"
except ImportError:  # ---- local copy of reid-heatmap packages/tracklet_schema (v0.2) ----
    import re
    from datetime import datetime, timedelta, timezone
    from pathlib import Path
    from zoneinfo import ZoneInfo

    from pydantic import BaseModel, Field, model_validator

    SOURCE = "local copy (tracklet_schema not installed)"
    SCHEMA_VERSION = "0.2"

    class Frame(BaseModel):
        ts: datetime  # UTC
        bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 (equirectangular px)
        foot_px: tuple[float, float]  # u, v (equirectangular px)
        occluded_bottom: bool = False  # foot point unreliable when True

    class Embedding(BaseModel):
        ts: datetime
        vector: list[float]  # 256-d, L2-normalised (current baseline model)
        det_conf: float = Field(ge=0.0, le=1.0)  # detector confidence for this crop
        box_h_px: float = Field(gt=0)  # crop height; quality is computed by multi_cam

    class Tracklet(BaseModel):
        """One closed track from one camera."""

        schema_version: str = SCHEMA_VERSION
        camera_id: str
        local_track_id: int
        start_ts: datetime
        end_ts: datetime
        reid_model_version: str
        camera_person_id: int | None = None  # optional hint: single-cam gallery ID (not authoritative)
        frames: list[Frame]
        embeddings: list[Embedding] = Field(min_length=1)  # several samples, not one average

        @model_validator(mode="after")
        def _check(self) -> Tracklet:
            if self.end_ts < self.start_ts:
                raise ValueError("end_ts before start_ts")
            if self.schema_version != SCHEMA_VERSION:
                raise ValueError(f"schema {self.schema_version} != expected {SCHEMA_VERSION}")
            return self

    SITE_TZ = ZoneInfo("Asia/Taipei")
    _NAME = re.compile(r"(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})-(\d{6})")

    def segment_start_utc(path: str | Path, tz: ZoneInfo = SITE_TZ) -> datetime:
        m = _NAME.search(Path(path).stem)
        if not m:
            raise ValueError(f"not a Cupola segment name: {path}")
        y, mo, d, h, mi, s, us = map(int, m.groups())
        return datetime(y, mo, d, h, mi, s, us, tzinfo=tz).astimezone(timezone.utc)

    def frame_utc(path: str | Path, frame_index: int, fps: float, tz: ZoneInfo = SITE_TZ) -> datetime:
        return segment_start_utc(path, tz) + timedelta(seconds=frame_index / fps)


__all__ = ["SCHEMA_VERSION", "SOURCE", "Embedding", "Frame", "Tracklet", "frame_utc", "segment_start_utc"]
