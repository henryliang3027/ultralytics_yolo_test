"""Sample frames from camera videos and pre-label persons, so a human only corrects boxes before measuring mAP.

Labelling rule: a person hidden by a partition gets a box around the VISIBLE part only (no guessed full body).

Output (Ultralytics dataset layout, usable by `yolo val` once the labels are corrected):
    <output>/images/val/<video>_f<frame>.jpg
    <output>/labels/val/<video>_f<frame>.txt     class 0 = person, normalized cx cy w h
    <output>/data.yaml
    <output>/cvat_yolo1.1.zip                     upload to CVAT: Actions > Upload annotations > YOLO 1.1

    python prelabel.py --source ../../cut/output/51_203_20261126_104149_104223.mp4 \\
        ../../cut/output/51_205_20261126_104149_104223.mp4 --output ../../datasets/person_eval
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
INTEGRATE = ROOT / "integrate"
sys.path.insert(0, str(INTEGRATE))

from person_pipeline.config import load_config  # noqa: E402
from person_pipeline.detector import PersonDetector  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(INTEGRATE / "config.yaml"), help="config file (detector section is used)")
    p.add_argument("--source", nargs="+", required=True, help="video files (spaces or commas)")
    p.add_argument("--output", required=True, help="dataset folder to create")
    p.add_argument("--every", type=float, default=1.0, help="seconds between sampled frames (default 1.0; 0 = every frame)")
    p.add_argument("--weights", help="pre-label model (default: detector.weights from the config)")
    p.add_argument("--conf", type=float, default=0.25, help="pre-label confidence (default 0.25, lower than deploy)")
    p.add_argument("--imgsz", type=int, help="override detector.imgsz")
    return p.parse_args()


def sample_frames(path: Path, every_s: float):
    """Yield (frame index, BGR frame) every `every_s` seconds; frames in between are only grabbed, not decoded."""
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        sys.exit(f"cannot open {path}")
    step = max(1, round((cap.get(cv2.CAP_PROP_FPS) or 30.0) * every_s))
    idx = 0
    try:
        while True:
            if idx % step == 0:
                ok, frame = cap.read()
                if not ok:
                    return
                yield idx, frame
            elif not cap.grab():
                return
            idx += 1
    finally:
        cap.release()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg.detector.conf = args.conf
    if args.weights:
        cfg.detector.weights = args.weights
    if args.imgsz is not None:
        cfg.detector.imgsz = args.imgsz

    sources = [Path(u.strip()).expanduser().resolve() for a in args.source for u in a.split(",") if u.strip()]
    for s in sources:
        if not s.is_file():
            sys.exit(f"video not found: {s}")
    out = Path(args.output).expanduser().resolve()
    img_dir, lbl_dir = out / "images" / "val", out / "labels" / "val"
    if img_dir.exists() and any(img_dir.iterdir()):
        sys.exit(f"{img_dir} is not empty; pick a new --output so corrected labels are never overwritten")
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    print(f"[prelabel] {cfg.detector.weights} imgsz={cfg.detector.imgsz} conf={cfg.detector.conf}")
    detector = PersonDetector(cfg.detector)
    names, n_boxes = [], 0
    for src in sources:
        for idx, frame in sample_frames(src, args.every):
            name = f"{src.stem}_f{idx:05d}"
            cv2.imwrite(str(img_dir / f"{name}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            dets = detector.detect([frame])[0]
            h, w = frame.shape[:2]
            lines = []
            for x1, y1, x2, y2 in dets.xyxy:
                lines.append(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            (lbl_dir / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            names.append(name)
            n_boxes += len(lines)
        print(f"[prelabel] {src.name}: {sum(n.startswith(src.stem) for n in names)} frames")

    (out / "data.yaml").write_text(f"path: {out}\ntrain: images/val\nval: images/val\nnames:\n  0: person\n")

    with zipfile.ZipFile(out / "cvat_yolo1.1.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("obj.names", "person\n")
        z.writestr("obj.data", "classes = 1\ntrain = data/train.txt\nnames = data/obj.names\nbackup = backup/\n")
        z.writestr("train.txt", "".join(f"data/obj_train_data/{n}.jpg\n" for n in names))
        for n in names:
            z.write(lbl_dir / f"{n}.txt", f"obj_train_data/{n}.txt")

    print(f"[prelabel] {len(names)} frames, {n_boxes} boxes -> {out}")


if __name__ == "__main__":
    main()
