"""Shared-gallery simulation GUI: play two cameras' videos, send a track-update batch to the shared gallery service at
every tick, and show the (camera_id, person_id) -> global_person_id mapping it keeps in PostgreSQL.

    left top    : camera 1 video (boxes labelled  P<person_id> > G<global_person_id>)
    left bottom : camera 2 video
    right       : the shared gallery's mapping table + the latest assignments

Sources of the batches:
  --source jsonl     replay results/track_updates/.../track_updates.jsonl (already exported); boxes on screen are
                     the last batch's (they update 3 times a second)
  --source pipeline  run the real pipeline (YOLO -> TrackTrack -> Re-ID -> separate per-camera galleries) on the
                     videos and build each batch on the fly, exactly as record.py does: the
                     integration test. --dump writes what was sent, to compare with the exported file.

    python service.py &                                   # first: docker compose up -d
    python simulate.py                                    # jsonl replay, real time
    python simulate.py --source pipeline --dump sent.jsonl
    python simulate.py --no-show --speed 0 --save ../results/shared_gallery_sim/replay.mp4   # headless, fast
Keys: q quit, space pause.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "integrate"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import requests  # noqa: E402

from person_pipeline.track_update_contract import SITE_TIMEZONE, parse_video_name  # noqa: E402  (stdlib only)

VIDEOS = ROOT / "videos"
DEFAULT_TOP = VIDEOS / "51_203_20261111_153342_153431.mp4"
DEFAULT_BOTTOM = VIDEOS / "51_205_20261111_153342_153431.mp4"
DEFAULT_UPDATES = ROOT / "results" / "track_updates" / "20261111_153342_153431" / "track_updates.jsonl"

W, H = 1920, 1080
LEFT_W = 960
VID_H = 480
HEAD_H = 40
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, GRAY, DIM, BG, PANEL = (255, 255, 255), (150, 150, 150), (110, 110, 110), (24, 24, 24), (38, 38, 38)
CROSS_BG = (40, 70, 40)
EARLY_MS = 1  # frame times are rounded to microseconds: a tick reached within 1 ms counts


def gcolor(g: int | None) -> tuple:
    if g is None:
        return GRAY
    rng = np.random.default_rng(g * 7919 + 3)
    return tuple(int(c) for c in rng.integers(90, 256, 3))


def text(img, s, x, y, scale=0.55, col=WHITE, thick=1):
    cv2.putText(img, s, (x, y), FONT, scale, col, thick, cv2.LINE_AA)


def hms(iso: str | None, tz: ZoneInfo) -> str:
    if not iso:
        return "-"
    t = datetime.fromisoformat(iso).astimezone(tz)
    return f"{t:%H:%M:%S}.{t.microsecond // 100000}"


class GalleryClient:
    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self.s = requests.Session()

    def health(self) -> None:
        try:
            self.s.get(self.url + "/health", timeout=3).raise_for_status()
        except requests.RequestException as e:
            raise SystemExit(f"shared gallery service not reachable at {self.url} ({e}); start service.py first")

    def reset(self) -> None:
        self.s.post(self.url + "/reset", timeout=10).raise_for_status()

    def send(self, line: str) -> dict:
        r = self.s.post(self.url + "/track_updates", data=line.encode(), headers={"Content-Type": "application/json"},
                        timeout=10)
        if r.status_code != 200:
            raise SystemExit(f"service rejected a batch ({r.status_code}): {r.text[:500]}")
        return r.json()

    def mapping(self) -> dict:
        r = self.s.get(self.url + "/mapping", timeout=10)
        r.raise_for_status()
        return r.json()


class JsonlSource:
    """Batches from an exported track_updates.jsonl, released when the video reaches their tick_ts."""

    def __init__(self, path: Path):
        self.lines = [l.strip() for l in open(path) if l.strip()]
        self.batches = [json.loads(l) for l in self.lines]
        self.ticks = [datetime.fromisoformat(b["tick_ts"]) for b in self.batches]
        self.i = 0
        self.live = None  # boxes come from the batches

    def frame(self, imgs, ts):
        out = []
        now = max(ts) + timedelta(milliseconds=EARLY_MS)
        while self.i < len(self.batches) and self.ticks[self.i] <= now:
            out.append((self.lines[self.i], self.batches[self.i]))
            self.i += 1
        return out


class PipelineSource:
    """The real per-camera pipeline; builds each batch like record.py."""

    def __init__(self, cfg, cam_ids, fps, interval, tz):
        from fractions import Fraction

        from person_pipeline import CameraStream, PersonDetector, ReIDExtractor
        from person_pipeline.gallery import make_galleries
        from person_pipeline.track_updates import TickClock, iso_time, records_of
        from person_pipeline.types import FramePacket

        cfg.gallery.mode = "per_camera"
        cfg.cupola.enabled = False
        cfg.display.show = False
        self.cam_ids, self.tz = cam_ids, tz
        self.iso_time, self.records_of, self.FramePacket = iso_time, records_of, FramePacket
        self.detector = PersonDetector(cfg.detector)
        self.detector.warmup(batch=len(cam_ids))
        reid = ReIDExtractor(cfg.reid)
        cams = [type(cfg.cameras[0])(name=c, url=c, camera_id=c) for c in cam_ids]
        self.streams = [CameraStream(c, cfg, reid, None, g) for c, g in zip(cams, make_galleries(cfg.gallery, cam_ids))]
        self.clock = TickClock(Fraction(interval) / Fraction(fps).limit_denominator(1000))
        self.k = 0
        self.live = {}  # camera -> [(person_id | None, bbox)] of the current frame

    def frame(self, imgs, ts):
        n = self.clock.due(max(ts))
        dets_list = self.detector.detect(imgs)
        records = []
        for cid, s, img, dets, t in zip(self.cam_ids, self.streams, imgs, dets_list, ts):
            r = s.process(self.FramePacket(img, self.k, time.monotonic()), dets, 0.0, now=t.timestamp(),
                          tick=n is not None)
            self.live[cid] = [(r.person_ids[i], [float(v) for v in dets.xyxy[i]])
                              for i, tid in enumerate(r.track_ids) if tid >= 0]
            if n is not None:
                records += self.records_of(r, cid, t, self.tz)
        self.k += 1
        if n is None:
            return []
        line = json.dumps({"tick_ts": self.iso_time(self.clock.time_of(n), self.tz), "records": records},
                          separators=(",", ":"))
        return [(line, json.loads(line))]


def render(frames, cam_ids, boxes, gid_of, mapping, stats, info, events, tz, ts):
    canvas = np.full((H, W, 3), BG, np.uint8)
    s = LEFT_W / frames[0].shape[1]
    for idx, (cid, img) in enumerate(zip(cam_ids, frames)):
        y0 = idx * (HEAD_H + VID_H + 20)
        vis = cv2.resize(img, (LEFT_W, VID_H), interpolation=cv2.INTER_AREA)
        for pid, (x1, y1, x2, y2) in boxes.get(cid, []):
            g = gid_of.get((cid, pid)) if pid is not None else None
            col = gcolor(g)
            p1, p2 = (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s))
            cv2.rectangle(vis, p1, p2, col, 2)
            lab = f"P{pid}>" + (f"G{g:03d}" if g is not None else "?") if pid is not None else "?"
            (tw, th), _ = cv2.getTextSize(lab, FONT, 0.45, 1)
            ly = p1[1] - 4 if p1[1] - th - 6 > 0 else p2[1] + th + 4
            cv2.rectangle(vis, (p1[0], ly - th - 3), (p1[0] + tw + 4, ly + 3), col, -1)
            text(vis, lab, p1[0] + 2, ly, 0.45, (0, 0, 0))
        canvas[y0 + HEAD_H:y0 + HEAD_H + VID_H, :LEFT_W] = vis
        t = ts[idx].astimezone(tz)
        text(canvas, f"camera {cid}", 12, y0 + 28, 0.75, WHITE, 2)
        text(canvas, f"{t:%Y-%m-%d %H:%M:%S}.{t.microsecond // 1000:03d}  frame {info['frame']}", 300, y0 + 28, 0.6, GRAY)

    # ---- right panel ----
    x0 = LEFT_W
    cv2.rectangle(canvas, (x0, 0), (W, H), PANEL, -1)
    text(canvas, "Shared gallery (PostgreSQL)", x0 + 20, 40, 0.9, WHITE, 2)
    text(canvas, f"source: {info['source']}   last tick {hms(info['tick'], tz)}   batches sent {info['sent']}   "
                 f"records {info['records']}   POST {info['post_ms']:.0f} ms", x0 + 20, 72, 0.5, GRAY)
    text(canvas, f"global persons {stats.get('global_persons', 0)}   local persons mapped "
                 f"{stats.get('local_persons_mapped', 0)}   cross-camera {stats.get('cross_camera_global_persons', 0)}"
                 f"   embeddings stored {stats.get('embeddings', 0)}", x0 + 20, 98, 0.5, GRAY)

    cols = [("global_person_id", 20), (f"{cam_ids[0]} person_id", 230), (f"{cam_ids[1]} person_id", 430),
            ("bank", 630), ("last seen", 710)]
    y = 140
    for name, dx in cols:
        text(canvas, name, x0 + dx, y, 0.55, WHITE, 1)
    cv2.line(canvas, (x0 + 15, y + 10), (W - 15, y + 10), DIM, 1)
    by_g = defaultdict(list)
    for m in mapping:
        by_g[m["global_person_id"]].append(m)
    y += 40
    row_h = 30
    max_rows = (820 - y) // row_h
    for g in sorted(by_g, key=lambda g: max(m["last_seen_ts"] for m in by_g[g]), reverse=True)[:max_rows]:
        rows = by_g[g]
        per_cam = {c: ", ".join(f"P{m['person_id']}" for m in rows if m["camera_id"] == c) for c in cam_ids}
        cross = sum(1 for c in cam_ids if per_cam[c]) > 1
        if cross:
            cv2.rectangle(canvas, (x0 + 12, y - 21), (W - 12, y + 8), CROSS_BG, -1)
        cv2.rectangle(canvas, (x0 + 20, y - 15), (x0 + 36, y + 1), gcolor(g), -1)
        text(canvas, f"G{g:03d}", x0 + 46, y, 0.6, WHITE, 2 if cross else 1)
        for c, (_, dx) in zip(cam_ids, cols[1:3]):
            text(canvas, per_cam[c] or "-", x0 + dx, y, 0.6, WHITE if per_cam[c] else DIM)
        text(canvas, str(rows[0]["bank_size"]), x0 + 630, y, 0.55, GRAY)
        text(canvas, hms(max(m["last_seen_ts"] for m in rows), tz), x0 + 710, y, 0.55, GRAY)
        y += row_h
    text(canvas, "green row = same person on both cameras;  sorted by last seen", x0 + 20, 830, 0.45, DIM)

    text(canvas, "Assignments (latest first)", x0 + 20, 870, 0.6, WHITE, 1)
    for i, (line, col) in enumerate(events):
        text(canvas, line, x0 + 20, 900 + i * 24, 0.5, col)
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", default=str(DEFAULT_TOP), help="camera 1 video (left top)")
    ap.add_argument("--bottom", default=str(DEFAULT_BOTTOM), help="camera 2 video (left bottom)")
    ap.add_argument("--source", choices=["jsonl", "pipeline"], default="jsonl")
    ap.add_argument("--updates", default=str(DEFAULT_UPDATES), help="track_updates.jsonl for --source jsonl")
    ap.add_argument("--service", default="http://127.0.0.1:8765")
    ap.add_argument("--config", default=str(ROOT / "integrate" / "config.yaml"), help="--source pipeline only")
    ap.add_argument("--timezone", default=SITE_TIMEZONE, help="time zone of the video file names and batches")
    ap.add_argument("--interval", type=int, default=5, help="--source pipeline: frames between batches")
    ap.add_argument("--speed", type=float, default=1.0, help="1 = real time, 0 = as fast as possible")
    ap.add_argument("--keep", action="store_true", help="do not reset the shared gallery first")
    ap.add_argument("--save", help="also write the GUI to this mp4")
    ap.add_argument("--dump", help="write every batch sent to this jsonl")
    ap.add_argument("--no-show", action="store_true")
    a = ap.parse_args()

    tz = ZoneInfo(a.timezone)
    paths = [Path(a.top).resolve(), Path(a.bottom).resolve()]
    names = [parse_video_name(p, tz) for p in paths]
    if not all(names):
        raise SystemExit("video names must look like <camera>_<YYYYMMDD>_<HHMMSS>_<HHMMSS>.mp4")
    cam_ids = [n[0] for n in names]
    starts = [n[1] for n in names]
    if len(set(cam_ids)) != 2:
        raise SystemExit(f"need two different cameras, got {cam_ids}")
    caps = [cv2.VideoCapture(str(p)) for p in paths]
    fps = caps[0].get(cv2.CAP_PROP_FPS) or 15.0
    for where, cid, st, path in zip(("top", "bottom"), cam_ids, starts, paths):
        print(f"[sim] {where:6s} camera {cid}  starts {st:%Y-%m-%d %H:%M:%S}  {path}")

    if a.source == "jsonl":
        src = JsonlSource(Path(a.updates))
        file_cams = {r["camera_id"] for b in src.batches for r in b["records"]}
        first, last = src.ticks[0], src.ticks[-1]
        for cid, st in zip(cam_ids, starts):
            if abs((first - st).total_seconds()) > 1:
                raise SystemExit(f"{a.updates} starts at {first.astimezone(tz)} but the {cid} video starts at {st}: "
                                 "the batches do not belong to these videos")
        if not file_cams <= set(cam_ids):
            raise SystemExit(f"{a.updates} has cameras {sorted(file_cams)}, the videos are {cam_ids}")
        print(f"[sim] replaying {len(src.batches)} batches {first.astimezone(tz)} .. {last.astimezone(tz)}")
    else:
        from person_pipeline import load_config  # the full pipeline environment (integrate/), only for this source

        src = PipelineSource(load_config(a.config), cam_ids, fps, a.interval, tz)

    client = GalleryClient(a.service)
    client.health()
    if not a.keep:
        client.reset()
    dump = open(a.dump, "w") if a.dump else None
    writer = None
    if a.save:
        Path(a.save).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not a.no_show:
        cv2.namedWindow("shared gallery simulation", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("shared gallery simulation", W * 3 // 4, H * 3 // 4)

    info = {"source": a.source, "tick": None, "sent": 0, "records": 0, "post_ms": 0.0, "frame": 0}
    boxes, gid_of, mapping, stats = {}, {}, [], {}
    events = deque(maxlen=7)
    totals = defaultdict(int)
    t_wall0 = time.monotonic()
    k = 0
    try:
        while True:
            frames = [c.read() for c in caps]
            if not all(ok for ok, _ in frames):
                break
            imgs = [f for _, f in frames]
            ts = [st + timedelta(seconds=k / fps) for st in starts]
            for line, batch in src.frame(imgs, ts):
                t = time.perf_counter()
                res = client.send(line)
                info["post_ms"] = (time.perf_counter() - t) * 1000
                info.update(tick=batch["tick_ts"], sent=info["sent"] + 1, records=len(batch["records"]))
                if dump:
                    dump.write(line + "\n")
                for asg in res["assignments"]:
                    totals[asg["status"]] += 1
                    if asg["status"] == "new":
                        g = asg["global_person_id"]
                        events.appendleft((f"{hms(batch['tick_ts'], tz)}  {asg['camera_id']} P{asg['person_id']} "
                                           f"-> G{g:03d}  new (provisional)", gcolor(g)))
                for mg in res["merges"]:
                    totals["merges"] += 1
                    how = "cross-camera merge" if mg["cross_camera"] else "same-camera merge"
                    events.appendleft((f"{hms(batch['tick_ts'], tz)}  G{mg['from']:03d} -> G{mg['into']:03d}  {how} "
                                       f"(mean cos {mg['similarity']:.2f})", (120, 255, 120)))
                m = client.mapping()
                mapping, stats = m["mapping"], m["stats"]
                gid_of = {(r["camera_id"], r["person_id"]): r["global_person_id"] for r in mapping}
                if src.live is None:  # jsonl: boxes of the last batch
                    boxes = defaultdict(list)
                    for r in batch["records"]:
                        boxes[r["camera_id"]].append((r["person_id"], r["bbox"]))
            if src.live is not None:
                boxes = src.live
            info["frame"] = k
            canvas = render(imgs, cam_ids, boxes, gid_of, mapping, stats, info, events, tz, ts)
            if writer:
                writer.write(canvas)
            if not a.no_show:
                cv2.imshow("shared gallery simulation", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    paused = time.monotonic()
                    while (cv2.waitKey(50) & 0xFF) not in (ord(" "), ord("q")):
                        pass
                    t_wall0 += time.monotonic() - paused
            k += 1
            if a.speed > 0:
                delay = t_wall0 + k / fps / a.speed - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
    finally:
        for c in caps:
            c.release()
        if writer:
            writer.release()
        if dump:
            dump.close()
        if not a.no_show:
            cv2.destroyAllWindows()

    print(f"[sim] {k} frames, {info['sent']} batches sent; assignment statuses {dict(totals)}")
    print(f"[sim] gallery: {stats}")
    by_g = defaultdict(list)
    for m in mapping:
        by_g[m["global_person_id"]].append(f"{m['camera_id']}/P{m['person_id']}")
    for g, members in sorted(by_g.items()):
        cams = {x.split('/')[0] for x in members}
        print(f"  G{g:03d}: {', '.join(members)}" + ("   <- both cameras" if len(cams) > 1 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
