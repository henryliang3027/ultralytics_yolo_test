"""Check a tracklets.jsonl against the recordings it came from, and draw it onto video.

The frames are read again from the Cupola recording segments with the same contract timestamps record.py used, so
every tracklet frame is matched to its source frame by exact UTC time (not by guessing a frame index in an mp4).

Checks (besides the pydantic contract):
  - every tracklet frame / embedding time is a real decoded frame inside the window
  - foot_px is the bottom-centre of bbox; start_ts / end_ts are the first / last frame; no repeated frame time
  - embedding times are frame times, box_h_px equals that frame's bbox height
  - re-embedding the stored bbox crop with the Re-ID model gives back the stored vector (cosine ~ 1)
  - (camera_id, local_track_id) unique; gaps inside a tracklet not longer than the tracker's lost time

    python visualize_tracklets.py ../results/tracklets/20261111_153342_153431/tracklets.jsonl \
        --out ../results/visualize_tracklets
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from person_pipeline import ReIDExtractor, load_config
from person_pipeline.segments import SegmentReader
from person_pipeline.tracklet_contract import Tracklet

FONT = cv2.FONT_HERSHEY_SIMPLEX


def color(tid: int) -> tuple:
    rng = np.random.default_rng(tid * 7919 + 13)
    return tuple(int(c) for c in rng.integers(60, 256, 3))


def key(ts: datetime) -> int:
    return round(ts.timestamp() * 1e6)  # microseconds: exact match between decoded frames and tracklet frames


def label(img, text, x, y, col, scale):
    (tw, th), base = cv2.getTextSize(text, FONT, scale, 2)
    y = max(y, th + base + 4)
    cv2.rectangle(img, (x, y - th - base - 4), (x + tw + 4, y), col, -1)
    cv2.putText(img, text, (x + 2, y - base - 2), FONT, scale, (0, 0, 0), 2, cv2.LINE_AA)


def static_checks(trs: list[Tracklet], gap_limit_s: float) -> dict:
    issues = defaultdict(list)
    seen = set()
    for t in trs:
        name = f"{t.camera_id}/{t.local_track_id}"
        if (t.camera_id, t.local_track_id) in seen:
            issues["duplicate_id"].append(name)
        seen.add((t.camera_id, t.local_track_id))
        ts = [f.ts for f in t.frames]
        if len(set(ts)) != len(ts):
            issues["repeated_frame_time"].append(name)
        if t.start_ts != min(ts) or t.end_ts != max(ts):
            issues["start_end_mismatch"].append(name)
        if t.camera_person_id is None:
            issues["no_camera_person_id"].append(name)
        for f in t.frames:
            x1, y1, x2, y2 = f.bbox
            if not (x2 > x1 and y2 > y1):
                issues["bad_bbox"].append(name)
                break
            if abs(f.foot_px[0] - (x1 + x2) / 2) > 0.02 or abs(f.foot_px[1] - y2) > 0.02:
                issues["foot_not_bottom_centre"].append(name)
                break
        st = sorted(ts)
        gaps = [(b - a).total_seconds() for a, b in zip(st, st[1:])]
        if gaps and max(gaps) > gap_limit_s:
            issues["gap_longer_than_lost_time"].append(f"{name} ({max(gaps):.2f}s)")
        by_ts = {f.ts: f for f in t.frames}
        for e in t.embeddings:
            f = by_ts.get(e.ts)
            if f is None:
                issues["embedding_time_not_a_frame"].append(name)
                break
            if abs((f.bbox[3] - f.bbox[1]) - e.box_h_px) > 0.02:
                issues["box_h_mismatch"].append(f"{name} ({e.box_h_px:.2f} vs {f.bbox[3] - f.bbox[1]:.2f})")
                break
    return {k: v for k, v in issues.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tracklets", help="tracklets.jsonl from record.py")
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "results" / "visualize_tracklets"))
    ap.add_argument("--scale", type=float, default=0.5, help="output video size relative to 3840x1920")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rc = cfg.record
    tz = ZoneInfo(rc.timezone)
    path = Path(args.tracklets).resolve()
    trs = [Tracklet.model_validate_json(l) for l in open(path) if l.strip()]
    print(f"[check] {len(trs)} tracklets parse with the contract models")

    # Window: the one record.py used (summary.json next to the file), else the tracklets' own span.
    summary = path.with_name("summary.json")
    shared = False
    if summary.exists():
        meta = json.loads(summary.read_text())
        shared = meta.get("gallery_mode") == "shared"  # camera_person_id is then a global ID: label it G001
        w = meta["window_utc"]
        start, end = datetime.fromisoformat(w[0]), datetime.fromisoformat(w[1])
    else:
        start, end = min(t.start_ts for t in trs), max(t.end_ts for t in trs)

    gap_limit = (cfg.tracker.track_buffer + 1) / 15.0 + 1e-3  # lost time at the recordings' 15 fps, plus one frame
    report = {"tracklets_file": str(path), "window_utc": [start.isoformat(), end.isoformat()],
              "tracklets": len(trs), "static_issues": static_checks(trs, gap_limit), "cameras": {}}

    reid = ReIDExtractor(cfg.reid)
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cam_by_id = {(c.camera_id or c.name): c for c in cfg.cameras}

    for cid in sorted({t.camera_id for t in trs}):
        cam = cam_by_id[cid]
        mine = [t for t in trs if t.camera_id == cid]
        frames_at = defaultdict(list)  # time key -> [(tracklet, frame, is_first, is_last)]
        emb_at = defaultdict(list)  # time key -> [(tracklet, embedding)]
        for t in mine:
            first, last = key(t.start_ts), key(t.end_ts)
            for f in t.frames:
                k = key(f.ts)
                frames_at[k].append((t, f, k == first, k == last))
            for e in t.embeddings:
                emb_at[key(e.ts)].append((t, e))

        reader = SegmentReader(Path(rc.recordings_root) / cam.recording_dir, start, end, tz).start_reading()
        fps = None
        writer = None
        out_path = out_dir / f"{cid}_{path.parent.name}.mp4"
        matched_frames, decoded, cosines, n_boxes = set(), 0, [], 0
        try:
            while (fr := reader.read()) is not None:
                decoded += 1
                fps = fps or reader.fps or 15.0
                k = key(fr.ts)
                img = fr.image
                # Re-embed the stored crops and compare with the stored vectors (on the full-res frame).
                if k in emb_at:
                    boxes = np.array([next(f.bbox for f in t.frames if key(f.ts) == k) for t, _ in emb_at[k]])
                    feats = reid.extract(img, boxes)
                    for (t, e), v in zip(emb_at[k], feats):
                        cosines.append((float(np.dot(v, np.asarray(e.vector))), f"{cid}/{t.local_track_id}"))
                vis = cv2.resize(img, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
                s = args.scale
                embedded_here = {(t.local_track_id) for t, _ in emb_at.get(k, [])}
                for t, f, is_first, is_last in frames_at.get(k, []):
                    matched_frames.add((t.local_track_id, k))
                    n_boxes += 1
                    x1, y1, x2, y2 = (int(round(v * s)) for v in f.bbox)
                    col = color(t.local_track_id)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), col, 3 if t.local_track_id in embedded_here else 2)
                    fx, fy = (int(round(v * s)) for v in f.foot_px)
                    cv2.circle(vis, (fx, fy), 5, col, -1)
                    pid = t.camera_person_id
                    tag = f"T{t.local_track_id} " + (f"G{pid:03d}" if shared and pid is not None else f"P{pid}")
                    if t.local_track_id in embedded_here:
                        tag += " E"
                    if is_first:
                        tag += " START"
                    if is_last:
                        tag += " END"
                    label(vis, tag, x1, y1 - 2, col, 0.6)
                local = fr.ts.astimezone(tz)
                head = (f"{cid}  {local:%Y-%m-%d %H:%M:%S.%f}"[:-3] + f" {rc.timezone}  |  {fr.ts:%H:%M:%S.%f}"[:-3]
                        + f" UTC  |  frame {fr.index}  |  tracklets {len(frames_at.get(k, []))}")
                cv2.rectangle(vis, (0, 0), (vis.shape[1], 34), (0, 0, 0), -1)
                cv2.putText(vis, head, (10, 24), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                if writer is None:
                    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, vis.shape[1::-1])
                writer.write(vis)
        finally:
            reader.stop()
            if writer is not None:
                writer.release()

        expected = {(t.local_track_id, key(f.ts)) for t in mine for f in t.frames}
        unmatched = expected - matched_frames
        cos = np.array([c for c, _ in cosines]) if cosines else np.zeros(0)
        report["cameras"][cid] = {
            "video": str(out_path),
            "frames_decoded": decoded,
            "tracklets": len(mine),
            "boxes_drawn": n_boxes,
            "tracklet_frames": len(expected),
            "tracklet_frames_without_source_frame": len(unmatched),
            "embeddings_rechecked": int(len(cos)),
            "embedding_cosine_min": round(float(cos.min()), 5) if len(cos) else None,
            "embedding_cosine_median": round(float(np.median(cos)), 5) if len(cos) else None,
            "embeddings_below_0.99": [n for c, n in cosines if c < 0.99],
        }
        print(f"[check] {cid}: {json.dumps(report['cameras'][cid])}")

    ok = (not report["static_issues"]
          and all(c["tracklet_frames_without_source_frame"] == 0 and not c["embeddings_below_0.99"]
                  for c in report["cameras"].values()))
    report["all_checks_passed"] = ok
    rp = out_dir / f"check_{path.parent.name}.json"
    rp.write_text(json.dumps(report, indent=2))
    print(f"[check] static issues: {report['static_issues'] or 'none'}")
    print(f"[check] {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'} -> {rp}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
