"""Turn one camera of record.py's tracklets.jsonl (track updates) into MOTChallenge files: a CVAT pre-annotation zip
and a TrackEval prediction.

    python tracklets_to_mot.py ../results/track_updates/20261111_153342_153431/tracklets.jsonl --camera 51_203

The video and its start time come from summary.json next to the jsonl (written by record.py); pass --video for a
file without one (the start time is then read from the video name, <camera_id>_<YYYYMMDD>_<HHMMSS>_<HHMMSS>.mp4).

Writes, under --out (default: ./<video name>/):
  <seq>_cvat_mot.zip     import into CVAT as "MOT 1.1" (gt/gt.txt + gt/labels.txt), then fix it by hand
  pred/<seq>.txt         the same boxes as a TrackEval tracker result (frame,id,x,y,w,h,conf,-1,-1,-1)

ID = person_id of each record (the ID the pipeline showed on that frame).
Frame = (record timestamp - video start) * fps + 1, the same frame-time rule record.py used; a timestamp that is not
exactly a frame of the video stops the script.

record.py sends a batch only every 5 frames, and CVAT ends a track at any frame gap. So the frames between two
consecutive batches are filled by linear interpolation when the person is in both (--no-fill: tick frames only).
Tracks that never got a person_id are not in the file, so they are missing from the pre-annotation.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

VIDEO_NAME = re.compile(r"^(?P<cam>[^_]+_[^_]+)_(?P<date>\d{8})_(?P<start>\d{6})_(?P<end>\d{6})$")
SITE_TIMEZONE = "Asia/Taipei"


def video_info(path: Path) -> tuple[int, int, int, float]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    cap.release()
    return n, w, h, fps


def source(jsonl: Path, camera: str, video: str | None) -> tuple[Path, datetime]:
    """(video, start time) of `camera`: from --video and its name, else from summary.json next to the jsonl."""
    meta_path = jsonl.with_name("summary.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if video:
        m = VIDEO_NAME.match(Path(video).stem)
        if not m:
            raise SystemExit(f"cannot read the start time from {Path(video).name}")
        tz = ZoneInfo(meta.get("timezone", SITE_TIMEZONE))
        return Path(video), datetime.strptime(m["date"] + m["start"], "%Y%m%d%H%M%S").replace(tzinfo=tz)
    if camera not in meta.get("camera_ids", []):
        raise SystemExit(f"no summary.json listing camera {camera} next to {jsonl}; pass --video")
    i = meta["camera_ids"].index(camera)
    return Path(meta["videos"][i]), datetime.fromisoformat(meta["start_local"][i])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tracklets", help="tracklets.jsonl from integrate/record.py")
    p.add_argument("--camera", required=True, help="camera_id in the jsonl, e.g. 51_203")
    p.add_argument("--video", help="the mp4 of that camera (default: from summary.json next to the jsonl)")
    p.add_argument("--out", help="output folder (default: ./<video name>)")
    p.add_argument("--no-fill", action="store_true", help="write only the frames a batch was sent on")
    args = p.parse_args()

    jsonl = Path(args.tracklets)
    video, start = source(jsonl, args.camera, args.video)
    n_video, width, height, fps = video_info(video)
    start = start.astimezone(timezone.utc)

    def frame_of(ts: str) -> int:
        k = (datetime.fromisoformat(ts) - start) / timedelta(seconds=1) * fps
        if abs(k - round(k)) > 0.01 or not 0 <= round(k) < n_video:
            raise SystemExit(f"{ts} is not a frame of {video.name} ({n_video} frames from {start.isoformat()})")
        return round(k) + 1

    # one entry per batch, in order: {person_id: (frame, bbox)} of this camera (empty when it sent nobody)
    ticks: list[dict[int, tuple[int, list[float]]]] = []
    for line in open(jsonl):
        if not line.strip():
            continue
        batch = json.loads(line)
        if set(batch) != {"tick_ts", "records"}:
            raise SystemExit(f"{jsonl} is not record.py's track-update file (keys {sorted(batch)})")
        ticks.append({r["person_id"]: (frame_of(r["timestamp"]), r["bbox"])
                      for r in batch["records"] if r["camera_id"] == args.camera})
    if not any(ticks):
        raise SystemExit(f"no record with camera_id {args.camera}")

    boxes: dict[tuple[int, int], list[float]] = {}  # (frame, person_id) -> x1, y1, x2, y2
    filled = 0
    for j, tick in enumerate(ticks):
        nxt = ticks[j + 1] if j + 1 < len(ticks) else {}
        for pid, (fr, b) in tick.items():
            boxes[fr, pid] = b
            if args.no_fill or pid not in nxt:
                continue
            fr2, b2 = nxt[pid]
            for f in range(fr + 1, fr2):
                a = (f - fr) / (fr2 - fr)
                boxes[f, pid] = [u + (v - u) * a for u, v in zip(b, b2)]
                filled += 1

    rows = []
    for (fr, pid), (x1, y1, x2, y2) in sorted(boxes.items()):
        x1, y1, x2, y2 = max(x1, 0.0), max(y1, 0.0), min(x2, width), min(y2, height)
        if x2 > x1 and y2 > y1:
            rows.append((fr, pid, x1, y1, x2 - x1, y2 - y1))

    seq = video.stem
    out = Path(args.out or Path(__file__).with_name(seq))
    (out / "pred").mkdir(parents=True, exist_ok=True)

    gt = "".join(f"{fr},{pid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,1,1.0\n" for fr, pid, x, y, w, h in rows)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("gt/gt.txt", gt)
        z.writestr("gt/labels.txt", "person\n")
    zip_path = out / f"{seq}_cvat_mot.zip"
    zip_path.write_bytes(buf.getvalue())

    pred = "".join(f"{fr},{pid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n" for fr, pid, x, y, w, h in rows)
    (out / "pred" / f"{seq}.txt").write_text(pred)

    ids = sorted({pid for _, pid, *_ in rows})
    print(f"{args.camera}: {len(ticks)} batches -> {len(rows)} boxes ({filled} interpolated), "
          f"{n_video} frames {width}x{height} @ {fps:g} fps, ids {ids}")
    print(f"  CVAT import (MOT 1.1): {zip_path}")
    print(f"  TrackEval prediction:  {out / 'pred' / f'{seq}.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
