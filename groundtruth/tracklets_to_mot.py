"""Turn one camera of a tracklets.jsonl into MOTChallenge files: a CVAT pre-annotation zip and a TrackEval prediction.

    python tracklets_to_mot.py ../results/tracklets/20261111_153342_153431/tracklets.jsonl --camera 51_203 \
        --video ../videos/51_203_20261111_153342_153431.mp4

Writes, under --out (default: ./<video name>/):
  <seq>_cvat_mot.zip     import into CVAT as "MOT 1.1" (gt/gt.txt + gt/labels.txt), then fix it by hand
  pred/<seq>.txt         the same boxes as a TrackEval tracker result (frame,id,x,y,w,h,conf,-1,-1,-1)

ID = camera_person_id (the identity the pipeline shows); tracklets without one get 100000 + local_track_id.
Frame number = position of the tracklet time among all times of that camera, plus 1 (MOT frames start at 1). This only
equals the video frame when some track covers every frame (record mode: every decoded frame is processed); the script
checks the count against the video and stops if they differ.

Limit: camera_person_id is one value per tracklet (majority vote), so frames where the pipeline showed another ID, or
none yet, are written with the majority ID. Good enough to pre-annotate; for the final score use per-frame IDs.
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

NO_PERSON_OFFSET = 100000


def video_info(path: str) -> tuple[int, int, int]:
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return n, w, h


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tracklets")
    p.add_argument("--camera", required=True, help="camera_id in the jsonl, e.g. 51_203")
    p.add_argument("--video", required=True, help="the mp4 these tracklets were made from (frame count check)")
    p.add_argument("--out", help="output folder (default: ./<video name>)")
    args = p.parse_args()

    tracks = [t for line in open(args.tracklets) if line.strip() and (t := json.loads(line))["camera_id"] == args.camera]
    if not tracks:
        raise SystemExit(f"no tracklet with camera_id {args.camera}")
    times = sorted({f["ts"] for t in tracks for f in t["frames"]})
    n_video, width, height = video_info(args.video)
    if len(times) != n_video:
        raise SystemExit(f"{len(times)} distinct tracklet times but {n_video} video frames: cannot map times to frames")
    frame_of = {ts: i + 1 for i, ts in enumerate(times)}

    # (frame, id) -> (x, y, w, h, track length); two tracklets with the same person in one frame keep the longer one
    boxes: dict[tuple[int, int], tuple] = {}
    duplicates = 0
    for t in tracks:
        pid = t["camera_person_id"]
        oid = pid if pid is not None else NO_PERSON_OFFSET + t["local_track_id"]
        for f in t["frames"]:
            x1, y1, x2, y2 = f["bbox"]
            key = (frame_of[f["ts"]], oid)
            box = (max(x1, 0.0), max(y1, 0.0), min(x2, width) - max(x1, 0.0), min(y2, height) - max(y1, 0.0), len(t["frames"]))
            if key in boxes:
                duplicates += 1
                if boxes[key][4] >= box[4]:
                    continue
            boxes[key] = box

    rows = sorted(boxes.items())
    seq = Path(args.video).stem
    out = Path(args.out or Path(__file__).with_name(seq))
    (out / "pred").mkdir(parents=True, exist_ok=True)

    gt = "".join(f"{fr},{oid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,1,1.0\n" for (fr, oid), (x, y, w, h, _) in rows)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("gt/gt.txt", gt)
        z.writestr("gt/labels.txt", "person\n")
    zip_path = out / f"{seq}_cvat_mot.zip"
    zip_path.write_bytes(buf.getvalue())

    pred = "".join(f"{fr},{oid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n" for (fr, oid), (x, y, w, h, _) in rows)
    (out / "pred" / f"{seq}.txt").write_text(pred)

    ids = sorted({oid for _, oid in boxes})
    print(f"{args.camera}: {len(tracks)} tracklets -> {len(rows)} boxes, {n_video} frames {width}x{height}, ids {ids}")
    if duplicates:
        print(f"  {duplicates} frame(s) had the same ID twice; kept the box of the longer tracklet")
    print(f"  CVAT import (MOT 1.1): {zip_path}")
    print(f"  TrackEval prediction:  {out / 'pred' / f'{seq}.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
