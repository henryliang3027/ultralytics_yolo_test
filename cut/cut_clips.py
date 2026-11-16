"""Cut time-synchronised MP4 clips from Cupola recording folders.

Each recording folder holds ~32 s MPEG-TS segments named by their local start
time, e.g. ``2026-11-11_15-33-20-141607.ts``. For the requested wall-clock
range this script builds one common output timeline (start + i / fps) and, for
every camera, writes the source frame closest to each timeline tick. All output
clips therefore have exactly the same number of frames and frame i of every
clip shows the same moment.

Usage:
    python cut_clips.py --start "2026-11-11 15:33:42" --end "2026-11-11 15:34:31"
"""

import argparse
import bisect
from datetime import datetime, timedelta
from pathlib import Path

import cv2

RECORDINGS = Path(
    "/home/ubuntu/Documents/Cupola_Patrol/20261006/amd64/"
    "cupola360-factory-web-amd64_2.2609.3/cupola360-factory-web-amd64/recordings"
)
CAMERAS = {
    "51_203": RECORDINGS / "fa57a71f-d37d-4fbd-a268-6a04e9ff852d",
    "51_205": RECORDINGS / "76bb2c48-21cb-429a-999e-5afa2e4d9672",
}
TIME_FMT = "%Y-%m-%d %H:%M:%S"
NAME_FMT = "%Y-%m-%d_%H-%M-%S-%f"
SEGMENT_MAX = timedelta(minutes=2)  # generous upper bound on one segment's length


def list_segments(folder: Path) -> list[tuple[datetime, Path]]:
    segs = []
    for p in folder.glob("*.ts"):
        try:
            segs.append((datetime.strptime(p.stem, NAME_FMT), p))
        except ValueError:
            continue
    return sorted(segs)


def segments_for_range(folder: Path, start: datetime, end: datetime) -> list[tuple[datetime, Path]]:
    """Segments that may contain frames in [start, end]."""
    segs = list_segments(folder)
    starts = [s for s, _ in segs]
    # Last segment starting at or before `start` (it covers the beginning), plus all after it up to `end`.
    i = max(bisect.bisect_right(starts, start) - 1, 0)
    picked = [seg for seg in segs[i:] if seg[0] <= end]
    if not picked or picked[0][0] > start or picked[-1][0] + SEGMENT_MAX < end:
        print(f"  warning: {folder.name} may not fully cover {start} ~ {end}")
    return picked


def iter_frames(segments):
    """Yield (wall_clock_datetime, frame) for every frame, in time order."""
    for seg_start, path in segments:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            print(f"  warning: cannot open {path}")
            continue
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = seg_start + timedelta(milliseconds=cap.get(cv2.CAP_PROP_POS_MSEC))
            yield t, frame
        cap.release()


def probe(segments) -> tuple[float, int, int]:
    cap = cv2.VideoCapture(str(segments[0][1]))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, w, h


def cut_camera(name, folder, start, end, fps, n_frames, out_dir) -> int:
    segments = segments_for_range(folder, start - timedelta(seconds=1), end)
    if not segments:
        raise SystemExit(f"{name}: no segments found for {start} ~ {end}")
    _, w, h = probe(segments)
    out_path = out_dir / f"{name}_{start:%Y%m%d_%H%M%S}_{end:%H%M%S}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise SystemExit(f"cannot open writer for {out_path}")

    step = timedelta(seconds=1 / fps)
    targets = [start + i * step for i in range(n_frames)]
    frames = iter_frames(segments)
    prev_t, prev_f = None, None
    cur_t, cur_f = next(frames, (None, None))
    written = 0
    first_src = last_src = None

    for target in targets:
        # Advance until the current frame is at/after the target time.
        while cur_t is not None and cur_t < target:
            prev_t, prev_f = cur_t, cur_f
            cur_t, cur_f = next(frames, (None, None))
        # Choose whichever neighbour is closer to the target.
        if cur_t is None and prev_t is None:
            raise SystemExit(f"{name}: no frames decoded")
        if cur_t is None or (prev_t is not None and target - prev_t <= cur_t - target):
            src_t, frame = prev_t, prev_f
        else:
            src_t, frame = cur_t, cur_f
        if abs((src_t - target).total_seconds()) > 1.0:
            print(f"  warning: {name} gap at {target:%H:%M:%S.%f}, nearest frame {src_t:%H:%M:%S.%f}")
        writer.write(frame)
        written += 1
        first_src = first_src or src_t
        last_src = src_t

    writer.release()
    print(f"  {name}: {written} frames -> {out_path}")
    print(f"         source frames {first_src:%H:%M:%S.%f} .. {last_src:%H:%M:%S.%f}")
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2026-11-11 15:33:42", help="local time, YYYY-MM-DD HH:MM:SS")
    ap.add_argument("--end", default="2026-11-11 15:34:31", help="local time, YYYY-MM-DD HH:MM:SS")
    ap.add_argument("--fps", type=float, default=None, help="output fps (default: source fps)")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "output")
    args = ap.parse_args()

    start = datetime.strptime(args.start, TIME_FMT)
    end = datetime.strptime(args.end, TIME_FMT)
    if end <= start:
        raise SystemExit("--end must be after --start")
    args.out.mkdir(parents=True, exist_ok=True)

    first_cam = next(iter(CAMERAS.values()))
    fps = args.fps or probe(segments_for_range(first_cam, start, end))[0]
    n_frames = round((end - start).total_seconds() * fps)
    print(f"range {start} ~ {end}, {fps:g} fps -> {n_frames} frames per clip")

    counts = {name: cut_camera(name, folder, start, end, fps, n_frames, args.out) for name, folder in CAMERAS.items()}
    assert len(set(counts.values())) == 1, counts
    print("done, all clips have", n_frames, "frames")


if __name__ == "__main__":
    main()
