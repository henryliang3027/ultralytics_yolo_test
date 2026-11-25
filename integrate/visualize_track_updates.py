"""Check a tracklets.jsonl against the videos it came from, and draw it onto them.

The videos are read again with the same frame-time rule record.py used (start time + frame index / fps),
so every record is matched to its source frame by exact time (microseconds), not by guessing a frame index.
The cameras are stacked top to bottom in one output video (and window with --show).

Drawing:
  - tick frame (a batch was sent): thick boxes, label "P<person_id>" plus "E" (embedding sent) or "null"
  - frames between ticks: the last batch's boxes, thin, i.e. what the receiving side currently knows

Checks:
  - every batch passes the interface validator (person_pipeline/track_update_contract.py)
  - batches are one tick period apart; each record's timestamp falls in its batch's tick
  - each record's timestamp is a real frame of that camera's video
  - re-embedding the stored bbox crop with the Re-ID model gives back the stored vector (cosine ~ 1)

    python visualize_track_updates.py ../results/track_updates/20261111_153342_153431/tracklets.jsonl
    python visualize_track_updates.py <tracklets.jsonl> --show            # also open a window (q quit, space pause)
    python visualize_track_updates.py <tracklets.jsonl> --videos a.mp4 b.mp4 --no-recheck

Use a file from `record.py --id-type pid` (the default). A `--id-type tid` or `both` file (records with "track_id") is refused.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from person_pipeline import load_config
from person_pipeline.track_update_contract import VIDEO_NAME, TickClock, iso_time, validate_batch

FONT = cv2.FONT_HERSHEY_SIMPLEX


def color(pid: int) -> tuple:
    rng = np.random.default_rng(pid * 7919 + 13)
    return tuple(int(c) for c in rng.integers(60, 256, 3))


def key(ts: datetime) -> int:
    return (ts - datetime(1970, 1, 1, tzinfo=timezone.utc)) // timedelta(microseconds=1)


def label(img, text, x, y, col, scale):
    (tw, th), base = cv2.getTextSize(text, FONT, scale, 2)
    y = max(y, th + base + 4)
    cv2.rectangle(img, (x, y - th - base - 4), (x + tw + 4, y), col, -1)
    cv2.putText(img, text, (x + 2, y - base - 2), FONT, scale, (0, 0, 0), 2, cv2.LINE_AA)


def sources(path: Path, videos: list[str] | None, camera_ids: list[str] | None, tz) -> list[tuple[str, Path, datetime]]:
    """(camera_id, video, start UTC) per camera: from the arguments, else from summary.json next to the jsonl."""
    meta = json.loads(path.with_name("summary.json").read_text()) if path.with_name("summary.json").exists() else {}
    vids = [Path(v).resolve() for v in (videos or meta.get("videos", []))]
    if not vids:
        raise SystemExit("no summary.json next to the file: pass --videos")
    names = [VIDEO_NAME.match(v.stem) for v in vids]
    cids = camera_ids or (meta.get("camera_ids") if not videos else None) or [m["cam"] if m else None for m in names]
    if None in cids or len(cids) != len(vids):
        raise SystemExit("cannot take camera_id from the video names; pass --camera-ids")
    if not videos and "start_local" in meta:
        starts = [datetime.fromisoformat(s) for s in meta["start_local"]]
    elif all(names):
        starts = [datetime.strptime(m["date"] + m["start"], "%Y%m%d%H%M%S").replace(tzinfo=tz) for m in names]
    else:
        raise SystemExit("cannot take the start time from the video names")
    return [(c, v, s.astimezone(timezone.utc)) for c, v, s in zip(cids, vids, starts)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("updates", help="tracklets.jsonl from record.py")
    ap.add_argument("--videos", nargs="+", help="one video per camera (default: summary.json next to the file)")
    ap.add_argument("--camera-ids", nargs="+", help="camera_id per video (default: summary.json / the video names)")
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "results" / "visualize_track_updates"))
    ap.add_argument("--scale", type=float, default=0.25, help="size of each camera relative to 3840x1920")
    ap.add_argument("--show", action="store_true", help="also show a window (q quit, space pause)")
    ap.add_argument("--no-save", action="store_true", help="do not write the output video")
    ap.add_argument("--no-recheck", action="store_true", help="skip re-embedding the crops (no Re-ID model needed)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tz = ZoneInfo(cfg.record.timezone)
    path = Path(args.updates).resolve()
    srcs = sources(path, args.videos, args.camera_ids, tz)
    cids = [c for c, _, _ in srcs]

    # Parse and check the batches on their own.
    batches, issues = [], defaultdict(list)
    for i, line in enumerate(open(path), 1):
        if not line.strip():
            continue
        b = json.loads(line)
        if any("person_id" not in r or "track_id" in r for r in b.get("records") or [] if isinstance(r, dict)):
            raise SystemExit(f"{path} has track_id records (record.py --id-type tid or both?), not person_id ones; "
                             "run record.py with --id-type pid (the default) and check that file")
        for p in validate_batch(b, tz):
            issues["invalid_batch"].append(f"line {i}: {p}")
        batches.append(b)
    print(f"[check] {len(batches)} batches, {sum(len(b['records']) for b in batches)} records, cameras {cids}")

    caps = [cv2.VideoCapture(str(v)) for _, v, _ in srcs]
    for (c, v, _), cap in zip(srcs, caps):
        if not cap.isOpened():
            raise SystemExit(f"cannot open {v}")
    fps = caps[0].get(cv2.CAP_PROP_FPS) or 15.0
    meta_path = path.with_name("summary.json")
    interval = json.loads(meta_path.read_text()).get("interval_frames", 5) if meta_path.exists() else 5
    period = Fraction(interval) / Fraction(fps).limit_denominator(1000)  # 5 frames at 15 fps = 1/3 s
    clock = TickClock(period)

    ticks = [datetime.fromisoformat(b["tick_ts"]) for b in batches]
    per_us = float(period) * 1e6
    for a, b in zip(ticks, ticks[1:]):
        if abs((b - a) / timedelta(microseconds=1) - per_us) > 2:
            issues["tick_spacing"].append(f"{iso_time(a, tz)} -> {iso_time(b, tz)}")

    # Index the records by (camera, frame time).
    rec_at = defaultdict(list)  # (camera_id, time key) -> [record]
    for b, t in zip(batches, ticks):
        n = clock.index(t)
        for r in b["records"]:
            ts = datetime.fromisoformat(r["timestamp"])
            if clock.index(ts) != n:
                issues["record_not_in_its_tick"].append(f"{r['camera_id']}/P{r['person_id']} {r['timestamp']}")
            rec_at[(r["camera_id"], key(ts))].append(r)
    unknown = {c for c, _ in rec_at} - set(cids)
    if unknown:
        issues["camera_without_video"] = sorted(unknown)

    reid = None
    if not args.no_recheck:
        from person_pipeline import ReIDExtractor
        reid = ReIDExtractor(cfg.reid)

    out_dir = Path(args.out).resolve()
    tag = path.parent.name
    writer = None
    matched, cosines = set(), []
    held = {c: [] for c in cids}  # last batch's records per camera
    k = 0
    paused = False
    if args.show:
        cv2.namedWindow("track updates", cv2.WINDOW_NORMAL)
    try:
        while True:
            frames = [cap.read() for cap in caps]
            if not all(ok for ok, _ in frames):
                break
            panels = []
            for (cid, _, start), (_, img) in zip(srcs, frames):
                ts = start + timedelta(seconds=k / fps)  # same rule as record.py
                recs = rec_at.get((cid, key(ts)))
                tick = recs is not None
                if tick:
                    matched.add((cid, key(ts)))
                    held[cid] = recs
                    with_emb = [r for r in recs if r["embedding"] is not None]
                    if reid is not None and with_emb:
                        feats = reid.extract(img, np.array([r["bbox"] for r in with_emb], np.float32))
                        for r, v in zip(with_emb, feats):
                            cosines.append((float(np.dot(v, np.asarray(r["embedding"], np.float32))),
                                            f"{cid}/P{r['person_id']} {r['timestamp']}"))
                s = args.scale
                vis = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                for r in held[cid]:
                    x1, y1, x2, y2 = (int(round(v * s)) for v in r["bbox"])
                    col = color(r["person_id"])
                    cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2 if tick else 1)
                    if tick:
                        txt = f"P{r['person_id']} " + ("E" if r["embedding"] is not None else "null")
                        label(vis, txt, x1, y1 - 2, col, 0.4)
                    else:
                        label(vis, f"P{r['person_id']}", x1, y1 - 2, col, 0.35)
                head = (f"{cid}  {iso_time(ts, tz)[:-9]}{iso_time(ts, tz)[-6:]}  frame {k}  "
                        + (f"TICK {len(held[cid])} records" if tick else f"(last batch: {len(held[cid])})"))
                cv2.rectangle(vis, (0, 0), (vis.shape[1], 22), (0, 0, 0), -1)
                cv2.putText(vis, head, (6, 16), FONT, 0.5, (0, 255, 255) if tick else (255, 255, 255), 1,
                            cv2.LINE_AA)
                panels.append(vis)
            canvas = np.vstack(panels)
            if not args.no_save:
                if writer is None:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(str(out_dir / f"{tag}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                             canvas.shape[1::-1])
                writer.write(canvas)
            if args.show:
                cv2.imshow("track updates", canvas)
                while True:
                    key_ = cv2.waitKey(0 if paused else max(1, int(1000 / fps))) & 0xFF
                    if key_ == ord(" "):
                        paused = not paused
                    if key_ == ord("q"):
                        raise KeyboardInterrupt
                    if not paused:
                        break
            k += 1
    except KeyboardInterrupt:
        print("[check] stopped early: the frame checks below cover only what was played")
    finally:
        for cap in caps:
            cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    expected = set(rec_at)
    unmatched = sorted(expected - matched)
    if unmatched:
        issues["record_time_not_a_video_frame"] = [f"{c} {iso_time(datetime.fromtimestamp(t / 1e6, timezone.utc), tz)}"
                                                   for c, t in unmatched[:20]]
    cos = np.array([c for c, _ in cosines]) if cosines else np.zeros(0)
    low = [n for c, n in cosines if c < 0.99]
    if low:
        issues["embedding_cosine_below_0.99"] = low[:20]

    report = {
        "track_updates_file": str(path),
        "videos": {c: str(v) for c, v, _ in srcs},
        "video": None if args.no_save else str(out_dir / f"{tag}.mp4"),
        "frames_per_camera": k,
        "batches": len(batches),
        "records": sum(len(b["records"]) for b in batches),
        "record_frames_matched": f"{len(expected & matched)}/{len(expected)}",
        "embeddings_rechecked": int(len(cos)),
        "embedding_cosine_min": round(float(cos.min()), 5) if len(cos) else None,
        "embedding_cosine_median": round(float(np.median(cos)), 5) if len(cos) else None,
        "issues": {k_: v[:20] for k_, v in issues.items()},
    }
    ok = not issues
    report["all_checks_passed"] = ok
    out_dir.mkdir(parents=True, exist_ok=True)
    rp = out_dir / f"check_{tag}.json"
    rp.write_text(json.dumps(report, indent=2))
    print(f"[check] {json.dumps({k_: v for k_, v in report.items() if k_ != 'issues'})}")
    print(f"[check] issues: {report['issues'] or 'none'}")
    print(f"[check] {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'} -> {rp}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
