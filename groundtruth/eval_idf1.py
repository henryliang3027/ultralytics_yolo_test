"""IDF1 of a tracker result against a MOTChallenge ground truth, overall and per GT person.

    python eval_idf1.py --gt 51_203_20261111_153342_153431/gt/gt.txt \
        --pred 51_203_20261111_153342_153431/pred/51_203_20261111_153342_153431.txt

Same definition as TrackEval's Identity metric (Ristani et al. 2016):
  - a (GT id, predicted id) pair "overlaps" in a frame when their boxes have IoU >= --iou
  - GT ids and predicted ids are matched one-to-one over the whole video, maximising the overlapping frames
    (so the ID numbers themselves do not matter: GT 7 can match prediction 3)
  - IDTP = overlapping frames of the matched pairs, IDFN = GT boxes - IDTP, IDFP = predicted boxes - IDTP
  - IDF1 = 2 IDTP / (2 IDTP + IDFP + IDFN)
GT rows with conf 0 are ignore regions: dropped, together with predicted boxes that overlap them.
Input files: frame,id,x,y,w,h,conf,... (MOT: x,y = top-left corner, frames from 1).
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment


def load(path: str) -> list[tuple]:
    rows = []
    for line in open(path):
        v = line.strip().split(",")
        if len(v) < 6:
            continue
        conf = float(v[6]) if len(v) > 6 else 1.0
        rows.append((int(float(v[0])), int(float(v[1])), *map(float, v[2:6]), conf))
    return rows


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (n, 4), b: (m, 4) as x, y, w, h -> (n, m)."""
    ax2, ay2, bx2, by2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    return inter / (a[:, 2:].prod(1)[:, None] + b[:, 2:].prod(1)[None] - inter + 1e-9)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt", required=True)
    p.add_argument("--pred", required=True)
    p.add_argument("--iou", type=float, default=0.5)
    args = p.parse_args()

    gt_by_frame, ign_by_frame, pred_by_frame = defaultdict(list), defaultdict(list), defaultdict(list)
    for f, i, x, y, w, h, c in load(args.gt):
        (gt_by_frame if c != 0 else ign_by_frame)[f].append((i, x, y, w, h))
    for f, i, x, y, w, h, _ in load(args.pred):
        pred_by_frame[f].append((i, x, y, w, h))

    overlap = defaultdict(int)  # (gt id, pred id) -> frames with IoU >= thr
    gt_count, pred_count, dropped = defaultdict(int), defaultdict(int), 0
    for f in sorted(set(gt_by_frame) | set(pred_by_frame)):
        g = np.array(gt_by_frame[f], float).reshape(-1, 5)
        q = np.array(pred_by_frame[f], float).reshape(-1, 5)
        if f in ign_by_frame and len(q):  # predictions on an ignore region count neither way
            ig = np.array(ign_by_frame[f], float).reshape(-1, 5)
            keep = iou(q[:, 1:], ig[:, 1:]).max(1) < args.iou
            dropped += int((~keep).sum())
            q = q[keep]
        for i in g[:, 0]:
            gt_count[int(i)] += 1
        for i in q[:, 0]:
            pred_count[int(i)] += 1
        if len(g) and len(q):
            gi, qi = np.nonzero(iou(g[:, 1:], q[:, 1:]) >= args.iou)
            for a, b in zip(gi, qi):
                overlap[int(g[a, 0]), int(q[b, 0])] += 1

    gids, pids = sorted(gt_count), sorted(pred_count)
    if not gids or not pids:
        raise SystemExit("ground truth or prediction is empty")
    m = np.zeros((len(gids), len(pids)))
    for (g, q), n in overlap.items():
        m[gids.index(g), pids.index(q)] = n
    rows, cols = linear_sum_assignment(-m)
    match = {gids[r]: (pids[c], int(m[r, c])) for r, c in zip(rows, cols) if m[r, c] > 0}

    idtp = sum(n for _, n in match.values())
    n_gt, n_pred = sum(gt_count.values()), sum(pred_count.values())
    idfn, idfp = n_gt - idtp, n_pred - idtp
    print(f"GT {len(gids)} ids / {n_gt} boxes, prediction {len(pids)} ids / {n_pred} boxes, IoU >= {args.iou}"
          + (f", {dropped} predicted boxes on ignore regions dropped" if dropped else ""))
    print(f"IDF1 {2 * idtp / (2 * idtp + idfp + idfn):.4f}   IDP {idtp / n_pred:.4f}   IDR {idtp / n_gt:.4f}   "
          f"IDTP {idtp}  IDFP {idfp}  IDFN {idfn}\n")

    print(f"{'GT id':>6} {'frames':>7} {'pred id':>8} {'IDTP':>6} {'IDFN':>6} {'IDFP':>6} {'IDF1':>7}   other pred ids on this person (frames)")
    for g in gids:
        others = sorted(((n, q) for (gg, q), n in overlap.items() if gg == g and (g not in match or q != match[g][0])), reverse=True)
        other_txt = ", ".join(f"{q} ({n})" for n, q in others[:6])
        if g in match:
            q, tp = match[g]
            fn, fp = gt_count[g] - tp, pred_count[q] - tp
            print(f"{g:>6} {gt_count[g]:>7} {q:>8} {tp:>6} {fn:>6} {fp:>6} {2 * tp / (2 * tp + fp + fn):>7.4f}   {other_txt}")
        else:
            print(f"{g:>6} {gt_count[g]:>7} {'-':>8} {0:>6} {gt_count[g]:>6} {'-':>6} {0:>7.4f}   {other_txt}")
    unmatched = [q for q in pids if q not in {v[0] for v in match.values()}]
    if unmatched:
        print("\nprediction ids matched to no GT person (all their boxes are IDFP): "
              + ", ".join(f"{q} ({pred_count[q]})" for q in unmatched))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
