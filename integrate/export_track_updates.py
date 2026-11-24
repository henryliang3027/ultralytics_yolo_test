"""Offline track-update export: run detection -> tracking -> Re-ID -> separate (per-camera) gallery over video files,
every frame of every camera in lockstep, and write one batch of track updates per clock tick (see
person_pipeline/track_updates.py for the interface).

Frame time = video start + frame_index / fps. The start is read from the file name (..._YYYYMMDD_HHMMSS_HHMMSS.mp4,
local record.timezone) unless --start is given; camera_id is the first two parts of the name (51_203).

    python export_track_updates.py ../videos/51_203_20261111_153342_153431.mp4 ../videos/51_205_20261111_153342_153431.mp4
    python export_track_updates.py --validate ../results/track_updates/20261111_153342_153431/tracklets.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from person_pipeline import CameraStream, PersonDetector, ReIDExtractor, load_config
from person_pipeline.gallery import make_galleries
from person_pipeline.track_update_contract import VIDEO_NAME as NAME
from person_pipeline.track_updates import TickClock, TrackUpdateWriter, iso_time, records_of, validate_batch
from person_pipeline.types import FramePacket


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("videos", nargs="*", help="one video file per camera, all covering the same time window")
    p.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    p.add_argument("--camera-ids", nargs="+", help="camera_id per video (default: from the file names)")
    p.add_argument("--start", help="local start time of every video, YYYY-MM-DD HH:MM:SS (default: from the names)")
    p.add_argument("--interval", type=int, default=5, help="frames between batches (5 at 15 fps = every 1/3 s)")
    p.add_argument("--out", help="output folder (default: ../results/track_updates/<date>_<start>_<end>)")
    p.add_argument("--overwrite", action="store_true", help="replace an existing tracklets.jsonl")
    p.add_argument("--validate", metavar="JSONL", help="only validate an existing tracklets.jsonl and exit")
    p.add_argument("--show", action="store_true",
                   help="show the cameras stacked while exporting (q stops early, space pauses)")
    p.add_argument("--show-scale", type=float, default=0.25, help="window size of each camera relative to 3840x1920")
    return p.parse_args()


FONT = cv2.FONT_HERSHEY_SIMPLEX


def _color(pid: int) -> tuple:
    rng = np.random.default_rng(pid * 7919 + 13)
    return tuple(int(c) for c in rng.integers(60, 256, 3))


def draw_panel(img, r, cid: str, ts: datetime, k: int, recs: list | None, tz, scale: float):
    """One camera's frame: every tracked box, labelled P<person_id> (T<track id> ? before it has one). On a tick
    frame (recs = the records just sent) the sent boxes are thick and say E (embedding sent) or null."""
    vis = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sent = {rec["person_id"]: rec for rec in recs or []}
    for i, tid in enumerate(r.track_ids):
        if tid < 0:
            continue
        pid = r.person_ids[i]
        x1, y1, x2, y2 = (int(round(float(v) * scale)) for v in r.detections.xyxy[i])
        col = (128, 128, 128) if pid is None else _color(int(pid))
        rec = sent.get(pid)
        thick = rec is not None and rec["bbox"] == [round(float(v), 2) for v in r.detections.xyxy[i]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2 if thick else 1)
        text = f"T{int(tid)} ?" if pid is None else f"P{pid}"
        if thick:
            text += " E" if rec["embedding"] is not None else " null"
        (tw, th), base = cv2.getTextSize(text, FONT, 0.4, 1)
        y = max(y1 - 2, th + base + 2)
        cv2.rectangle(vis, (x1, y - th - base - 2), (x1 + tw + 4, y), col, -1)
        cv2.putText(vis, text, (x1 + 2, y - base), FONT, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    iso = iso_time(ts, tz)
    head = f"{cid}  {iso[:-9]}{iso[-6:]}  frame {k}" + (f"  TICK sent {len(recs)}" if recs is not None else "")
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(vis, head, (6, 16), FONT, 0.5, (0, 255, 255) if recs is not None else (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def validate_file(path: str | Path, tz: ZoneInfo, period: Fraction | None = None) -> int:
    n, bad, prev = 0, 0, None
    gaps = Counter()
    with open(path) as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            n += 1
            batch = json.loads(line)
            problems = validate_batch(batch, tz)
            t = datetime.fromisoformat(batch["tick_ts"])
            if prev is not None:
                gaps[round((t - prev).total_seconds(), 6)] += 1
            prev = t
            if problems:
                bad += 1
                print(f"line {i}: {problems[:5]}")
    if period is not None and any(abs(g - float(period)) > 2e-6 for g in gaps):  # 1/3 s rounds to whole microseconds
        print(f"[validate] tick spacing not always {float(period):.6f}s: {dict(gaps)}")
        bad += 1
    print(f"[validate] {path}: {n} batches, {bad} problems; tick spacing {dict(gaps)}")
    return 1 if bad else 0


def main() -> int:
    args = parse_args()
    if args.validate:
        return validate_file(args.validate, ZoneInfo(load_config(args.config).record.timezone))
    if not args.videos:
        raise SystemExit("give one video per camera")

    cfg = load_config(args.config)
    tz = ZoneInfo(cfg.record.timezone)
    if cfg.gallery.mode != "per_camera":
        print(f"[export] gallery.mode {cfg.gallery.mode} -> per_camera (track updates carry per-camera person IDs)")
        cfg.gallery.mode = "per_camera"
    cfg.cupola.enabled = False
    cfg.display.show = False

    paths = [Path(v).resolve() for v in args.videos]
    names = [NAME.match(p.stem) for p in paths]
    cam_ids = args.camera_ids or [m["cam"] if m else None for m in names]
    if None in cam_ids or len(cam_ids) != len(paths):
        raise SystemExit("cannot take camera_id from the file names; pass --camera-ids")
    if args.start:
        starts = [datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S").replace(tzinfo=tz)] * len(paths)
    elif all(names):
        starts = [datetime.strptime(m["date"] + m["start"], "%Y%m%d%H%M%S").replace(tzinfo=tz) for m in names]
    else:
        raise SystemExit("cannot take the start time from the file names; pass --start")
    starts = [s.astimezone(timezone.utc) for s in starts]

    caps = [cv2.VideoCapture(str(p)) for p in paths]
    fps_list = [c.get(cv2.CAP_PROP_FPS) or 15.0 for c in caps]
    fps = fps_list[0]
    if any(abs(f - fps) > 1e-3 for f in fps_list):
        raise SystemExit(f"cameras have different fps: {fps_list}")
    period = Fraction(args.interval) / Fraction(fps).limit_denominator(1000)
    clock = TickClock(period)

    tag = f"{names[0]['date']}_{names[0]['start']}_{names[0]['end']}" if names[0] else paths[0].stem
    out_dir = Path(args.out or Path(__file__).resolve().parent.parent / "results" / "track_updates" / tag).resolve()
    out_path = out_dir / "tracklets.jsonl"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"{out_path} exists; pick another --out or pass --overwrite")
    print(f"[export] cameras {cam_ids}, start {[s.astimezone(tz).isoformat() for s in starts]}, {fps:g} fps, "
          f"batch every {args.interval} frames ({float(period):.3f}s)")

    detector = PersonDetector(cfg.detector)
    detector.warmup(batch=len(paths))
    reid = ReIDExtractor(cfg.reid)
    cams = [type(cfg.cameras[0])(name=c, url=str(p), camera_id=c) for c, p in zip(cam_ids, paths)]
    streams = [CameraStream(c, cfg, reid, None, g) for c, g in zip(cams, make_galleries(cfg.gallery, cam_ids))]
    writer = TrackUpdateWriter(out_path, tz)

    stats = {c: Counter() for c in cam_ids}
    persons = {c: set() for c in cam_ids}
    t0 = time.monotonic()
    k = 0
    stopped_early = paused = False
    if args.show:
        cv2.namedWindow("export track updates", cv2.WINDOW_NORMAL)
    try:
        while True:
            frames = [c.read() for c in caps]
            if not all(ok for ok, _ in frames):
                break
            imgs = [f for _, f in frames]
            ts = [s + timedelta(seconds=k / fps) for s in starts]
            n = clock.due(max(ts))  # one decision for all cameras: they send in the same batch
            dets_list = detector.detect(imgs)
            batch = []
            panels = []
            for cid, s, img, dets, t in zip(cam_ids, streams, imgs, dets_list, ts):
                r = s.process(FramePacket(img, k, time.monotonic()), dets, 0.0, now=t.timestamp(), tick=n is not None)
                recs = records_of(r, cid, t, tz) if n is not None else None
                if args.show:
                    panels.append(draw_panel(img, r, cid, t, k, recs, tz, args.show_scale))
                if n is None:
                    continue
                batch += recs
                st = stats[cid]
                st["records"] += len(recs)
                for rec in recs:
                    persons[cid].add(rec["person_id"])
                    if rec["embedding"] is None:
                        h = rec["bbox"][3] - rec["bbox"][1]
                        st["null_small_box" if h < cfg.reid.min_box_height else "null_overlap_or_conf"] += 1
                unsent = sum(1 for i, tid in enumerate(r.track_ids) if tid >= 0 and r.person_ids[i] is None)
                st["tracked_without_person_id"] += unsent
            if n is not None:
                writer.write(clock.time_of(n), batch)
            k += 1
            if args.show:
                cv2.imshow("export track updates", np.vstack(panels))
                while True:  # runs as fast as processing allows; space pauses, q stops
                    key = cv2.waitKey(0 if paused else 1) & 0xFF
                    if key == ord(" "):
                        paused = not paused
                    if key == ord("q"):
                        stopped_early = True
                        break
                    if not paused:
                        break
                if stopped_early:
                    print(f"[export] stopped early at frame {k}: tracklets.jsonl covers only what was processed")
                    break
    finally:
        if args.show:
            cv2.destroyAllWindows()
        writer.close()
        for c in caps:
            c.release()

    summary = {
        "videos": [str(p) for p in paths],
        "camera_ids": cam_ids,
        "start_local": [s.astimezone(tz).isoformat() for s in starts],
        "timezone": cfg.record.timezone,
        "fps": fps,
        "interval_frames": args.interval,
        "tick_period_s": float(period),
        "frames_per_camera": k,
        "stopped_early": stopped_early,
        "batches": writer.batches,
        "gallery_mode": cfg.gallery.mode,
        "reid_model": Path(cfg.reid.model).stem,
        "processing_s": round(time.monotonic() - t0, 1),
        "cameras": {},
    }
    for c in cam_ids:
        st = stats[c]
        nulls = st["null_small_box"] + st["null_overlap_or_conf"]
        summary["cameras"][c] = {
            "records": st["records"],
            "records_per_batch": round(st["records"] / max(writer.batches, 1), 2),
            "embedding_null": nulls,
            "embedding_null_ratio": round(nulls / max(st["records"], 1), 3),
            "null_small_box": st["null_small_box"],
            "null_overlap_or_conf": st["null_overlap_or_conf"],
            "tracked_but_not_sent_no_person_id": st["tracked_without_person_id"],
            "distinct_person_ids": sorted(persons[c]),
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[export] wrote {writer.batches} batches to {out_path}")
    return validate_file(out_path, tz, period)


if __name__ == "__main__":
    raise SystemExit(main())
