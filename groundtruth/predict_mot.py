"""Run the pipeline over every frame of an mp4 and write its per-frame result as MOTChallenge files for eval_idf1.py.

    python predict_mot.py ../videos/51_203_20261111_153342_153431.mp4

Writes <out>/pred/<video name>_person.txt (the person ID shown on each frame) and _track.txt (local_track_id, the
raw tracker ID). Default out: ./<video name>/. Same models and config as record.py (../integrate/config.yaml); every
frame is processed in order, none is dropped, and the gallery timeouts follow video time (frame / fps), so frame N of
these files is frame N of the video, as in the CVAT task.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2

INTEGRATE = Path(__file__).resolve().parent.parent / "integrate"
sys.path.insert(0, str(INTEGRATE))

from person_pipeline import CameraStream, PersonDetector, ReIDExtractor, load_config  # noqa: E402
from person_pipeline.types import FramePacket  # noqa: E402


def write_mot(files, frame: int, result, id_offset: int) -> None:
    """One MOTChallenge line per tracked box (frame from 1). A box whose track has no person ID yet gets
    100000 + local_track_id in the person file, so it can never count as the right person."""
    person_f, track_f = files
    for i, tid in enumerate(result.track_ids):
        if tid < 0:
            continue
        x1, y1, x2, y2 = (float(v) for v in result.detections.xyxy[i])
        box = f"{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{float(result.detections.conf[i]):.3f},-1,-1,-1"
        local = id_offset + int(tid)
        pid = result.person_ids[i]
        person_f.write(f"{frame},{pid if pid is not None else 100000 + local},{box}\n")
        track_f.write(f"{frame},{local},{box}\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("--config", default=str(INTEGRATE / "config.yaml"))
    p.add_argument("--out", help="output folder (default: ./<video name>)")
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg.display = replace(cfg.display, show=False, save_path=None)
    cfg.cupola = replace(cfg.cupola, enabled=False)
    cam = replace(cfg.cameras[0], url=args.video)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    seq = Path(args.video).stem
    out = Path(args.out or Path(__file__).with_name(seq)) / "pred"
    out.mkdir(parents=True, exist_ok=True)

    detector = PersonDetector(cfg.detector)
    detector.warmup(batch=1)
    stream = CameraStream(cam, cfg, ReIDExtractor(cfg.reid), None)
    files = (open(out / f"{seq}_person.txt", "w"), open(out / f"{seq}_track.txt", "w"))
    n, t0 = 0, time.monotonic()
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            t = time.perf_counter()
            dets = detector.detect([image])[0]
            result = stream.process(FramePacket(image, n, time.monotonic()), dets, (time.perf_counter() - t) * 1000, now=n / fps)
            write_mot(files, n + 1, result, 0)
            n += 1
            if n % 150 == 0:
                print(f"[predict] {n} frames, {n / (time.monotonic() - t0):.1f} frames/s")
    finally:
        cap.release()
        for f in files:
            f.close()
    print(f"[predict] {n} frames -> {files[0].name}\n                      {files[1].name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
