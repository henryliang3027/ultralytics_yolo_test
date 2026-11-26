"""Detection only: run the integrate pipeline's YOLO26 person detector on camera videos and save the boxes.

No Re-ID, gallery or Cupola. The detector and its settings (weights, imgsz, conf, device, half) come from
integrate/config.yaml, the same as main.py. With --use_tracker each camera also gets its own TrackTrack tracker
(settings from the tracker section of the same config, as in main.py) and boxes are labelled with the track id. Every frame is processed (none dropped), so the output video has the same
length and fps as the input.

    python detect.py --source ../../videos/a.mp4 ../../videos/b.mp4 --output ../../results/detection
    python detect.py --source a.mp4,b.mp4 --output out.mp4          # -> out_cam1.mp4, out_cam2.mp4
    python detect.py --source a.mp4 b.mp4 --output out_dir --show   # also show a window per camera (q quits)
    python detect.py --source a.mp4 --output out_dir --use_tracker  # add TrackTrack ids (no Re-ID model)
    python detect.py --source a.mp4 --output out_dir --show --step  # space = next frame, q = quit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INTEGRATE = ROOT / "integrate"
sys.path.insert(0, str(INTEGRATE))

from person_pipeline.config import load_config  # noqa: E402
from person_pipeline.detector import PersonDetector  # noqa: E402
from person_pipeline.recorder import VideoRecorder  # noqa: E402
from person_pipeline.tracker import PersonTracker  # noqa: E402
from person_pipeline.types import Detections  # noqa: E402

DET_COLOR = (0, 255, 0)  # green: YOLO detection
TRACK_COLOR = (255, 0, 255)  # magenta: TrackTrack track
FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(INTEGRATE / "config.yaml"), help="config file (detector section is used)")
    p.add_argument("--source", nargs="+", required=True, help="video files, one per camera (spaces or commas)")
    p.add_argument(
        "--output",
        required=True,
        help="folder (each camera saved as <video name>_detection.mp4) or a .mp4 path (several cameras -> "
        "<name>_cam1.mp4, <name>_cam2.mp4)",
    )
    p.add_argument("--show", action="store_true", help="show a window per camera while processing")
    p.add_argument("--step", action="store_true", help="with --show: wait for a key each frame (space = next, q = quit)")
    p.add_argument("--use_tracker", action="store_true", help="track persons with TrackTrack (config tracker section)")
    p.add_argument("--conf", type=float, help="override detector.conf")
    p.add_argument("--imgsz", type=int, help="override detector.imgsz")
    return p.parse_args()


def output_paths(output: str, sources: list[Path]) -> list[Path]:
    out = Path(output).expanduser().resolve()
    if not out.suffix:
        return [out / f"{s.stem}_detection.mp4" for s in sources]
    if len(sources) == 1:
        return [out]
    return [out.with_name(f"{out.stem}_cam{i + 1}{out.suffix}") for i in range(len(sources))]


def _label(img: np.ndarray, text: str, x: int, y: int, color: tuple, above: bool) -> None:
    """Filled label at (x, y): above=True sits on top of y, False hangs below it."""
    (tw, th), base = cv2.getTextSize(text, FONT, 1.0, 2)
    ty = y - base - 4 if above and y - th - base - 4 > 0 else y + th + 4
    cv2.rectangle(img, (x, ty - th - 4), (x + tw, ty + base), color, -1)
    cv2.putText(img, text, (x, ty), FONT, 1.0, (0, 0, 0), 2, cv2.LINE_AA)


def draw(img: np.ndarray, dets: Detections, status: str, tracks: np.ndarray | None = None) -> np.ndarray:
    """Detections: green box, conf label under the box. Tracks (TRACKTRACK rows [x1,y1,x2,y2,id,score,cls,idx]):
    magenta box from the tracker's own (Kalman) state, 'T<id>' label on top. Only tracks updated this frame
    are in the output, so a person YOLO missed has no magenta box that frame."""
    for (x1, y1, x2, y2), conf in zip(dets.xyxy.astype(int), dets.conf):
        cv2.rectangle(img, (x1, y1), (x2, y2), DET_COLOR, 5)
        _label(img, f"{conf:.2f}", x1, y2, DET_COLOR, above=False)
    if tracks is not None:
        for row in tracks:
            x1, y1, x2, y2 = row[:4].astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), TRACK_COLOR, 2)
            _label(img, f"T{int(row[4])}", x1, y1, TRACK_COLOR, above=True)
    (tw, th), base = cv2.getTextSize(status, FONT, 1.5, 3)
    x = img.shape[1] - tw - 20
    cv2.rectangle(img, (x - 10, 10), (x + tw + 10, 30 + th + base), (0, 0, 0), -1)
    cv2.putText(img, status, (x, 20 + th), FONT, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
    return img


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.conf is not None:
        cfg.detector.conf = args.conf
    if args.imgsz is not None:
        cfg.detector.imgsz = args.imgsz

    sources = [Path(u.strip()).expanduser().resolve() for a in args.source for u in a.split(",") if u.strip()]
    for s in sources:
        if not s.is_file():
            sys.exit(f"video not found: {s}")
    names = [f"cam{i + 1}" for i in range(len(sources))]
    outs = output_paths(args.output, sources)
    for s, o in zip(sources, outs):
        if o == s:
            sys.exit(f"output would overwrite the source video: {o}")

    caps = [cv2.VideoCapture(str(s), cv2.CAP_FFMPEG) for s in sources]
    for s, cap in zip(sources, caps):
        if not cap.isOpened():
            sys.exit(f"cannot open {s}")
    fps = [cap.get(cv2.CAP_PROP_FPS) or 15.0 for cap in caps]
    totals = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps]

    print(f"[detect] {cfg.detector.weights} imgsz={cfg.detector.imgsz} conf={cfg.detector.conf} "
          f"device={cfg.detector.device} half={cfg.detector.half}")
    detector = PersonDetector(cfg.detector)
    detector.warmup(batch=len(sources))
    trackers = [PersonTracker(cfg.tracker) for _ in sources] if args.use_tracker else None
    if trackers:
        print(f"[detect] TrackTrack {cfg.tracker}")

    writers: list[VideoRecorder | None] = [None] * len(sources)
    frame_idx = [0] * len(sources)
    det_count = [0] * len(sources)
    active = [True] * len(sources)
    if args.show:
        for n in names:
            cv2.namedWindow(n, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(n, *cfg.display.window_size)

    t0 = last_report = time.monotonic()
    try:
        while any(active):
            batch = []  # (camera index, frame)
            for i, cap in enumerate(caps):
                if not active[i]:
                    continue
                ok, frame = cap.read()
                if not ok:
                    active[i] = False
                    print(f"[{names[i]}] end of file after {frame_idx[i]} frames")
                    continue
                batch.append((i, frame))
            if not batch:
                break

            t = time.perf_counter()
            dets_list = detector.detect([f for _, f in batch])
            detect_ms = (time.perf_counter() - t) * 1000

            for (i, frame), dets in zip(batch, dets_list):
                det_count[i] += len(dets)
                status = f"{names[i]}  frame {frame_idx[i]}  persons {len(dets)}  detect {detect_ms:.0f} ms"
                tracks = None
                if trackers:
                    t = time.perf_counter()
                    trackers[i].update(dets, frame)  # before draw: TrackTrack reads the clean frame
                    tracks = trackers[i].last_tracks
                    status += f"  tracks {len(tracks)}  track {(time.perf_counter() - t) * 1000:.0f} ms"
                vis = draw(frame, dets, status, tracks)
                if writers[i] is None:
                    h, w = vis.shape[:2]
                    writers[i] = VideoRecorder(str(outs[i]), fps[i], (w, h), names[i])
                writers[i].write(vis)
                if args.show:
                    cv2.imshow(names[i], vis)
                frame_idx[i] += 1

            if args.show:
                key = cv2.waitKey(0 if args.step else 1) & 0xFF
                while args.step and key not in (ord(" "), ord("q")):  # step mode: only space advances
                    key = cv2.waitKey(0) & 0xFF
                if key == ord("q"):
                    print("[detect] q pressed, stopping")
                    break

            now = time.monotonic()
            if now - last_report >= cfg.logging.stats_interval_s:
                progress = "  ".join(f"{n} {f}/{t}" for n, f, t in zip(names, frame_idx, totals))
                print(f"[detect] {progress}  detect {detect_ms:.1f} ms/batch")
                last_report = now
    except KeyboardInterrupt:
        print("[detect] interrupted, finalizing videos")
    finally:
        for cap in caps:
            cap.release()
        for w in writers:
            if w is not None:
                w.close()
        if args.show:
            cv2.destroyAllWindows()

    elapsed = time.monotonic() - t0
    for n, f, d, o in zip(names, frame_idx, det_count, outs):
        print(f"[{n}] {f} frames, {d / max(f, 1):.1f} persons/frame -> {o}")
    print(f"[detect] done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
