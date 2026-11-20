"""Record mode: run detection -> tracking -> Re-ID -> gallery over Cupola recordings and export closed tracklets.

Every frame of every camera in the window is processed (no real-time dropping), each with its contract UTC time.
Closed tracklets go to <out_dir>/tracklets.jsonl, one Tracklet.model_dump_json() per line, validated before writing.

    python record.py --start "2026-11-11 15:32:00" --end "2026-11-11 15:52:00"   # local Asia/Taipei time
    python record.py ... --cameras cam1 --out ../results/tracklets_test   # one folder per run
    python record.py --validate ../results/tracklets/tracklets.jsonl              # check an existing file
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from person_pipeline import CameraStream, PersonDetector, ReIDExtractor, load_config
from person_pipeline.gallery import make_galleries
from person_pipeline.segments import SegmentReader
from person_pipeline.tracklet_contract import SCHEMA_VERSION, SOURCE, Tracklet
from person_pipeline.tracklets import TrackIdRegistry, TrackletBuilder, TrackletWriter, local_days
from person_pipeline.types import FramePacket

TIME_FMT = "%Y-%m-%d %H:%M:%S"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    p.add_argument("--start", help="local time, YYYY-MM-DD HH:MM:SS, in record.timezone")
    p.add_argument("--end", help="local time, YYYY-MM-DD HH:MM:SS, in record.timezone")
    p.add_argument("--cameras", nargs="+", help="camera names from config (default: all)")
    p.add_argument("--out", help="output folder (default: record.out_dir)")
    p.add_argument("--overwrite", action="store_true", help="replace an existing tracklets.jsonl in the output folder")
    p.add_argument("--no-registry", action="store_true",
                   help="IDs start at 1 and the per-day registry is neither read nor updated (one-off samples; "
                   "IDs may then repeat other runs of the same day)")
    p.add_argument("--gallery", choices=["per_camera", "shared"], help="override gallery.mode; shared makes "
                   "camera_person_id one ID space over all cameras")
    p.add_argument("--validate", metavar="JSONL", help="only validate an existing tracklets.jsonl and exit")
    p.add_argument("--mot", action="store_true",
                   help="also write every frame's boxes as MOTChallenge files for IDF1 (<out>/mot/<camera_id>_person.txt "
                   "with the person ID shown on that frame, <camera_id>_track.txt with local_track_id)")
    return p.parse_args()


def validate_file(path: str) -> int:
    ok, bad = 0, 0
    with open(path) as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                Tracklet.model_validate_json(line)
                ok += 1
            except Exception as e:
                bad += 1
                print(f"line {n}: {e}")
    print(f"[validate] {path}: {ok} valid, {bad} invalid (schema {SCHEMA_VERSION}, {SOURCE})")
    return 1 if bad else 0


def write_mot(files, frame: int, result, id_offset: int) -> None:
    """One MOTChallenge line per tracked box (frame from 1 = n-th processed frame of the window). A box whose track has
    no person ID yet gets 100000 + local_track_id in the person file, so it can never count as the right person."""
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


def describe(values: list[float]) -> dict:
    if not values:
        return {}
    a = np.asarray(values, float)
    return {"min": float(a.min()), "median": float(np.median(a)), "mean": round(float(a.mean()), 1), "max": float(a.max())}


def main() -> int:
    args = parse_args()
    if args.validate:
        return validate_file(args.validate)
    if not (args.start and args.end):
        raise SystemExit("--start and --end are required")

    cfg = load_config(args.config)
    if args.gallery:
        cfg.gallery.mode = args.gallery
    rc = cfg.record
    tz = ZoneInfo(rc.timezone)
    start = datetime.strptime(args.start, TIME_FMT).replace(tzinfo=tz).astimezone(timezone.utc)
    end = datetime.strptime(args.end, TIME_FMT).replace(tzinfo=tz).astimezone(timezone.utc)
    if end <= start:
        raise SystemExit("--end must be after --start")
    cams = [c for c in cfg.cameras if not args.cameras or c.name in args.cameras]
    if not cams:
        raise SystemExit(f"no camera matches {args.cameras}")
    out_dir = Path(args.out or rc.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "tracklets.jsonl").exists() and not args.overwrite:
        raise SystemExit(f"{out_dir / 'tracklets.jsonl'} exists; pick another --out or pass --overwrite")

    cfg.display = replace(cfg.display, show=False, save_path=None)
    cfg.cupola = replace(cfg.cupola, enabled=False)
    model_version = rc.reid_model_version or Path(cfg.reid.model).stem
    print(f"[record] {start.isoformat()} .. {end.isoformat()} UTC, cameras {[c.name for c in cams]}")
    print(f"[record] gallery mode {cfg.gallery.mode}")
    print(f"[record] contract schema {SCHEMA_VERSION} from {SOURCE}; reid_model_version={model_version}")

    detector = PersonDetector(cfg.detector)
    detector.warmup(batch=len(cams))
    reid = ReIDExtractor(cfg.reid)
    # One registry for every run, whatever --out is, so local_track_id never repeats within a camera and day.
    Path(rc.out_dir).mkdir(parents=True, exist_ok=True)
    registry = None if args.no_registry else TrackIdRegistry(Path(rc.out_dir) / "track_id_registry.json")
    days = local_days(start, end, tz)
    writer = TrackletWriter(out_dir / "tracklets.jsonl")

    streams, readers, builders, mot_files = [], [], [], []
    for cam in cams:
        cid = cam.camera_id or cam.name
        folder = Path(rc.recordings_root) / cam.recording_dir
        readers.append(SegmentReader(folder, start, end, tz).start_reading())
        streams.append(CameraStream(cam, cfg, reid, None, gal))
        builders.append(
            TrackletBuilder(
                cid, writer, model_version, registry.offset(cid, days) if registry else 0, cfg.tracker.track_buffer,
                rc.max_embeddings, rc.vector_decimals,
            )
        )
        if args.mot:
            (out_dir / "mot").mkdir(exist_ok=True)
            mot_files.append((open(out_dir / "mot" / f"{cid}_person.txt", "w"), open(out_dir / "mot" / f"{cid}_track.txt", "w")))
        print(f"[record] {cam.name} -> camera_id {cid}: {len(readers[-1].segments)} segments, "
              f"local_track_id offset {builders[-1].id_offset}")

    t0 = time.monotonic()
    last_log = t0
    n_frames = Counter()
    active = list(range(len(cams)))
    try:
        while active:
            batch = []
            for k in list(active):
                fr = readers[k].read()
                if fr is None:
                    active.remove(k)
                    continue
                batch.append((k, fr))
            if not batch:
                break
            tdet = time.perf_counter()
            dets_list = detector.detect([fr.image for _, fr in batch])
            detect_ms = (time.perf_counter() - tdet) * 1000
            for (k, fr), dets in zip(batch, dets_list):
                packet = FramePacket(fr.image, fr.index, time.monotonic())
                result = streams[k].process(packet, dets, detect_ms, now=fr.ts.timestamp())
                builders[k].update(result, fr.ts)
                if mot_files:
                    write_mot(mot_files[k], n_frames[k] + 1, result, builders[k].id_offset)
                n_frames[k] += 1
            now = time.monotonic()
            if now - last_log >= 30:
                done = max((fr.ts - start).total_seconds() for _, fr in batch)
                speed = sum(n_frames.values()) / (now - t0)
                print(f"[record] {done / 60:.1f}/{(end - start).total_seconds() / 60:.0f} min of video, "
                      f"{speed:.1f} frames/s, written {writer.written}")
                last_log = now
    finally:
        for r in readers:
            r.stop()
        for files in mot_files:
            for f in files:
                f.close()
    for b in builders:
        b.flush()
    writer.close()
    if registry:
        for b in builders:
            registry.commit(b.camera_id, days, b.next_free_id)

    # ---- summary ----
    lines = [json.loads(l) for l in open(writer.path)]
    summary = {
        "window_local": [args.start, args.end, rc.timezone],
        "window_utc": [start.isoformat(), end.isoformat()],
        "schema_version": SCHEMA_VERSION,
        "schema_source": SOURCE,
        "reid_model_version": model_version,
        "gallery_mode": cfg.gallery.mode,
        "processing_s": round(time.monotonic() - t0, 1),
        "cameras": {},
    }
    for k, (cam, b, r) in enumerate(zip(cams, builders, readers)):
        mine = [t for t in lines if t["camera_id"] == b.camera_id]
        durs = [(datetime.fromisoformat(t["end_ts"]) - datetime.fromisoformat(t["start_ts"])).total_seconds() for t in mine]
        summary["cameras"][b.camera_id] = {
            "name": cam.name,
            "frames_processed": n_frames[k],
            "fps": r.fps,
            "segments": len(r.segments),
            "tracklets_written": b.stats["written"],
            "dropped_no_embedding": b.stats["dropped_no_embedding"],
            "validation_failures": b.stats["invalid"],
            "validation_errors": b.errors,
            "frames_per_tracklet": describe([len(t["frames"]) for t in mine]),
            "embeddings_per_tracklet": describe([len(t["embeddings"]) for t in mine]),
            "duration_s_per_tracklet": describe(durs),
            "with_camera_person_id": sum(t["camera_person_id"] is not None for t in mine),
            "distinct_camera_person_ids": len({t["camera_person_id"] for t in mine} - {None}),
            "first_ts": min((t["start_ts"] for t in mine), default=None),
            "last_ts": max((t["end_ts"] for t in mine), default=None),
            "local_track_id_range": [min((t["local_track_id"] for t in mine), default=None),
                                     max((t["local_track_id"] for t in mine), default=None)],
            "timestamps": {
                "max_abs_contract_vs_pts_s": round(r.max_abs_drift_s, 3),
                "frames_stepping_back": r.backwards,
                "max_step_back_s": round(r.max_backwards_s, 3),
                "max_gap_between_frames_s": round(r.max_forward_gap_s, 3),
            },
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[record] wrote {writer.written} tracklets to {writer.path}")
    return validate_file(str(writer.path))


if __name__ == "__main__":
    raise SystemExit(main())
