"""Picture of who each GT id is: one crop per GT id (first, middle, last frame), labelled with the prediction id
eval_idf1.py matched it to. Answers "GT 3 is which person?" without opening CVAT.

    python show_ids.py ../videos/51_203_20261111_153342_153431.mp4 \
        --gt 51_203_20261111_153342_153431/gt/gt.txt --pred 51_203_20261111_153342_153431/pred/51_203_20261111_153342_153431_person.txt

Writes <gt folder>/ids.jpg.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from eval_idf1 import load

CELL = 220  # crop height in the sheet


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("--gt", required=True)
    p.add_argument("--pred", help="prediction file; if given, each GT id is labelled with the prediction id covering it most")
    p.add_argument("--iou", type=float, default=0.5)
    args = p.parse_args()

    tracks = defaultdict(list)  # gt id -> [(frame, x, y, w, h)]
    for f, i, x, y, w, h, c in load(args.gt):
        if c != 0:
            tracks[i].append((f, x, y, w, h))
    best = {}
    if args.pred:
        from eval_idf1 import iou

        pred = defaultdict(list)
        for f, i, x, y, w, h, _ in load(args.pred):
            pred[f].append((i, x, y, w, h))
        for g, boxes in tracks.items():
            votes = defaultdict(int)
            for f, *b in boxes:
                if pred[f]:
                    q = np.array(pred[f], float)
                    o = iou(np.array([b], float), q[:, 1:])[0]
                    if o.max() >= args.iou:
                        votes[int(q[o.argmax(), 0])] += 1
            best[g] = max(votes, key=votes.get) if votes else None

    wanted = defaultdict(list)  # frame -> [(gt id, slot, box)]
    for g, boxes in tracks.items():
        boxes.sort()
        for slot, k in enumerate((0, len(boxes) // 2, len(boxes) - 1)):
            f, *b = boxes[k]
            wanted[f].append((g, slot, b, f))
    crops = {}
    cap = cv2.VideoCapture(args.video)
    n = 0
    while wanted:
        ok, img = cap.read()
        if not ok:
            break
        n += 1  # MOT frames start at 1
        for g, slot, (x, y, w, h), f in wanted.pop(n, []):
            x1, y1 = max(int(x - 0.2 * w), 0), max(int(y - 0.1 * h), 0)
            x2, y2 = min(int(x + 1.2 * w), img.shape[1]), min(int(y + 1.1 * h), img.shape[0])
            c = img[y1:y2, x1:x2].copy()
            cv2.rectangle(c, (int(x) - x1, int(y) - y1), (int(x + w) - x1, int(y + h) - y1), (0, 255, 0), 2)
            c = cv2.resize(c, (max(int(c.shape[1] * CELL / max(c.shape[0], 1)), 1), CELL))
            crops[g, slot] = (c, f)

    rows = []
    for g in sorted(tracks):
        label = f"GT {g}: {len(tracks[g])} frames" + (f" -> pred {best.get(g)}" if args.pred else "")
        head = np.full((40, 1, 3), 255, np.uint8)
        cells = []
        for slot in range(3):
            if (g, slot) in crops:
                c, f = crops[g, slot]
                c = np.vstack([np.full((24, c.shape[1], 3), 40, np.uint8), c])
                cv2.putText(c, f"frame {f}", (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cells += [c, np.full((CELL + 24, 8, 3), 255, np.uint8)]
        row = np.hstack(cells) if cells else np.full((CELL + 24, 400, 3), 255, np.uint8)
        head = np.full((40, max(row.shape[1], 520), 3), 255, np.uint8)
        cv2.putText(head, label, (6, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        row = np.hstack([row, np.full((row.shape[0], head.shape[1] - row.shape[1], 3), 255, np.uint8)])
        rows.append(np.vstack([head, row]))
    width = max(r.shape[1] for r in rows)
    sheet = np.vstack([np.hstack([r, np.full((r.shape[0], width - r.shape[1], 3), 255, np.uint8)]) for r in rows])
    out = Path(args.gt).with_name("ids.jpg")
    cv2.imwrite(str(out), sheet)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
