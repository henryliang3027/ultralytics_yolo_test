import os
from types import SimpleNamespace

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.trackers.track_tracker import TRACKTRACK
from ultralytics.utils import YAML
from ultralytics.utils.checks import check_yaml

source1 = "videos/solutions_ci_demo.mp4"
source2 = "videos/test1.mp4"
source3 = "videos/51_203_output20261105_142123.mp4"
source4 = "videos/51_203_output20261106_095222.mp4"
source5 = "videos/51_205_output20261106_095222.mp4"
source6 = "videos/51_203_output20261111_153342.mp4"
source7 = "videos/51_205_output20261111_153342.mp4"

model = YOLO("yolo26s.pt")


# ---------------------------------------------------------------------------
# TrackTrack (CVPR 2025, "Focusing on Tracks for Online Multi-Object Tracking")
#
# TRACKTRACK.update() needs, per frame:
#   - detections exposing .xywh (N,4 center x/y, w, h in pixels), .conf (N,), .cls (N,)
#   - img: the current BGR frame (used for camera-motion compensation)
# and returns an (M, 8) array: [x1, y1, x2, y2, track_id, score, cls, idx],
# where idx is the row of the input detection that track matched in this frame.
# track() below wraps this: it turns idx into a per-detection `ids` array, and with
# return_tracks=True also returns only the first 5 columns (box + track_id) as `tracks`.
# ---------------------------------------------------------------------------

# Overrides on top of ultralytics/cfg/trackers/tracktrack.yaml.
# The yaml defaults (high 0.6 / new track 0.7) are strict; people in this
# 3840x1920 video are small and mostly score 0.25-0.6, so lower them here.
TRACKTRACK_OVERRIDES = {
    "track_high_thresh": 0.4,  # dets >= this are matched normally and may start new tracks
    "track_low_thresh": 0.1,  # dets between low and high are matched with a cost penalty, never start tracks
    "new_track_thresh": 0.5,  # min score to start a new track
    "track_buffer": 30,  # frames a lost track is kept (≈1 s at 30 fps)
    # Max match cost (default 0.7). When a partition hides the legs, YOLO's box jumps from full body to upper body
    # (height ~310 -> ~185 px); HMIoU then drops to ~0.2 (cost ~0.78) and the person got a new ID. 0.8 accepts it.
    "match_thresh": 0.7,
}


def create_tracker(**overrides):
    """Build a TRACKTRACK instance from the default yaml plus overrides."""
    cfg = YAML.load(check_yaml("tracktrack.yaml"))
    cfg.update(TRACKTRACK_OVERRIDES)
    cfg.update(overrides)
    return TRACKTRACK(SimpleNamespace(**cfg))


def track(tracker, xyxy, conf, cls, frame, return_tracks=False):
    """Run one TrackTrack step on YOLO detections.

    Args:
        tracker: instance from create_tracker(); keep one per video, call once per frame.
        xyxy (np.ndarray): (N, 4) boxes [x1, y1, x2, y2] in pixels.
        conf (np.ndarray): (N,) confidences.
        cls (np.ndarray): (N,) class ids.
        frame (np.ndarray): current BGR frame.
        return_tracks (bool): also return the tracker's own boxes.

    Returns:
        np.ndarray: (N,) int track id for each input detection, -1 if not (yet) tracked.
        If return_tracks: (ids, tracks) where tracks is (M, 5) [x1, y1, x2, y2, track_id],
        the Kalman-smoothed box of every track updated in this frame
        (the first 5 columns of TRACKTRACK.update()'s (M, 8) output).
    """
    xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
    conf = np.asarray(conf, dtype=np.float32).reshape(-1)
    cls = np.asarray(cls, dtype=np.float32).reshape(-1)

    xywh = np.empty_like(xyxy)
    xywh[:, :2] = (xyxy[:, :2] + xyxy[:, 2:]) / 2  # center
    xywh[:, 2:] = xyxy[:, 2:] - xyxy[:, :2]  # width, height

    dets = SimpleNamespace(xywh=xywh, conf=conf, cls=cls)
    out = tracker.update(dets, img=frame)  # (M, 8): x1, y1, x2, y2, id, score, cls, idx

    ids = np.full(len(xyxy), -1, dtype=int)
    for row in out:
        idx = int(row[7])
        if 0 <= idx < len(ids):
            ids[idx] = int(row[4])
    if return_tracks:
        tracks = out[:, :5] if len(out) else np.zeros((0, 5), dtype=np.float32)
        return ids, tracks
    return ids


GREEN = (0, 255, 0)  # BGR: YOLO detections
ORANGE = (0, 165, 255)  # BGR: TrackTrack tracks


def draw_box(frame, box, label, color, thickness=3, label_at="top"):
    """Draw one box with a filled label above it ("top") or below it ("bottom")."""
    h = frame.shape[0]
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    font, scale, tk = cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
    (tw, th), baseline = cv2.getTextSize(label, font, scale, tk)
    if label_at == "top":  # above the box, or inside it at the top edge of the frame
        ty = y1 - baseline - 4 if y1 - th - baseline - 4 > 0 else y1 + th + 4
    else:  # below the box, or inside it at the bottom edge of the frame
        ty = y2 + th + 4 if y2 + th + baseline + 4 < h else y2 - baseline - 4
    cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw, ty + baseline), color, -1)
    cv2.putText(frame, label, (x1, ty), font, scale, (0, 0, 0), tk, cv2.LINE_AA)


def draw_detections(frame, xyxy, conf, cls, names):
    """YOLO detections: green box, '<class> <conf>' label on top.

    Drawn lowest confidence first, so when two boxes on one person overlap, the higher score's label stays visible.
    """
    for i in np.argsort(conf):
        draw_box(frame, xyxy[i], f"{names[int(cls[i])]} {conf[i]:.2f}", GREEN, label_at="top")
    return frame


def draw_tracks(frame, tracks):
    """TrackTrack tracks: orange box, 'ID <n>' label at the bottom so it doesn't cover the detection label."""
    for x1, y1, x2, y2, tid in tracks.tolist():
        draw_box(frame, (x1, y1, x2, y2), f"ID {int(tid)}", ORANGE, thickness=2, label_at="bottom")
    return frame


def draw_frame_count(frame, frame_idx, total=None, scale=2.0, margin=20):
    """Write 'frame <idx> / <total>' in the top-right corner (white text on a black background)."""
    text = f"frame {frame_idx}" + (f" / {total}" if total else "")
    font, tk = cv2.FONT_HERSHEY_SIMPLEX, 4
    (tw, th), baseline = cv2.getTextSize(text, font, scale, tk)
    x, y = frame.shape[1] - tw - margin, margin + th
    cv2.rectangle(frame, (x - 10, y - th - 10), (x + tw + 10, y + baseline + 10), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), tk, cv2.LINE_AA)
    return frame


if __name__ == "__main__":
    source = source7  # change only this line to switch videos
    output_path = os.path.basename(source)  # "videos/x.mp4" -> "x.mp4", saved in the current folder

    tracker = create_tracker()
    # conf=0.1 so TrackTrack also receives low-score boxes for its second matching stage
    # imgsz=1920: the 3840x1920 frame shrunk to 640 makes people too small to score high
    results = model.predict(source=source, classes=[0], conf=0.6, imgsz=1920, stream=True, verbose=False)  # 0 = person
    cv2.namedWindow("detections", cv2.WINDOW_NORMAL)  # resizable: source is 3840x1920
    cv2.resizeWindow("detections", 1440, 720)  # width, height

    # Output video: same fps as the source, frame size taken from the first frame
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    writer = None

    for frame_idx, r in enumerate(results):  # frame_idx starts at 0
        frame = r.orig_img.copy()  # raw BGR frame, no Ultralytics annotations
        boxes = r.boxes.cpu().numpy()
        ids, tracks = track(tracker, boxes.xyxy, boxes.conf, boxes.cls, frame, return_tracks=True)

        draw_detections(frame, boxes.xyxy, boxes.conf, boxes.cls, r.names)  # every YOLO person box, green
        draw_tracks(frame, tracks)  # tracked people, orange, drawn on top
        draw_frame_count(frame, frame_idx, total_frames)
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(frame)

        cv2.imshow("detections", frame)
        if cv2.waitKey(0) & 0xFF == ord("q"):
            break

    if writer is not None:
        writer.release()  # finalize the mp4; without this the file may be unplayable
    cv2.destroyAllWindows()
