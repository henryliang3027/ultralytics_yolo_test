---
title: "When a TrackTrack track splits, check whether YOLO missed the person or the box changed shape before changing parameters"
author: HenryLiang
date: 2026-11-26
branch: main
cost: { minutes: 1, tokens: 1599 }
id: l-f4893c05
scope: repo
status: active
src: agent
touches:
  - integrate/person_pipeline/tracker.py
---

## Decided
- Before tuning tracker parameters, check whether the detection disappeared or the box changed shape. Print the per-term cost of the track against the nearest detection inside TRACKTRACK._cost_matrix.
- detect.py --use_tracker draws YOLO detections in green and TrackTrack output in magenta (T<id>); --show --step advances one frame per space. PersonTracker.last_tracks keeps the raw TRACKTRACK rows so the tracker's own box can be drawn.

## Watch out
- Case: cam1 of cut/output/51_203_20261126_104149_104223.mp4, frames 47 to 50. The person in the blue jacket went from T1 to T5. YOLO did NOT miss: detections at frames 48 to 50 had conf 0.77, 0.86, 0.88. He stepped out from behind a partition and the box height jumped from 126 to 324 px. The T1 Kalman prediction stayed about 126 px tall. HIoU was 0.386 and IoU 0.263, so 1 - HMIoU = 0.899 and the total cost was 0.943, above match_thresh 0.8. This confirms the 2026-11-15 learning on ID switches after occlusion with an exact example: a 2.6x jump still breaks the track.
- The cost is (1 - HIoU*IoU) + 0.1*|projected conf - det conf| + 0.05*angle*det conf. A pair with IoU <= 0.1 costs 1. A pair matches only if cost < match_thresh. HIoU is vertical overlap divided by vertical union, computed on float boxes. Rounded heights mislead: the tops differ by 0.48 px, which gives 125.32/324.48, not 126/324.
- A new ID is NOT caused by the old track failing N times. An unmatched detection with conf >= track_high_thresh, conf > new_track_thresh, and IoU <= tai_thr 0.55 with every Tracked track gets its ID number in the same frame (state New, not output). It is shown only after min_track_len 3 consecutive matches. T5 was created at frame 48 and first output at frame 50. A New track that misses one frame is removed and its number is never reused. The old track stays Lost for track_buffer 30 frames, but cannot win back a detection that a confirmed track matches more cheaply.
- TRACKTRACK.update returns only tracks matched this frame, so Lost and New tracks are invisible on screen.

In the code: ultralytics/trackers/track_tracker.py (_cost_matrix, _hmiou_distance, _track_aware_nms, TTSTrack.activate/update); integrate/config.yaml tracker section.
