---
title: Record mode now exports track updates from video files; the old closed-tracklet export was removed
author: HenryLiang
date: 2026-11-24
branch: main
cost: { minutes: 1, tokens: 1267 }
id: l-1b44325d
scope: repo
status: active
src: agent
touches:
  - integrate/record.py
  - groundtruth/tracklets_to_mot.py
  - groundtruth/predict_mot.py
---

## Decided
- Record mode (integrate/record.py) runs the per-camera pipeline over one video file per camera and writes one batch of track updates every 1/3 s (tick_ts + records of camera_id, person_id, bbox, embedding; +08:00 times). This is the format the shared gallery consumes, so it is the only export kept.
- The old record mode (read Cupola .ts recording segments, write closed tracklets in the tracklet_schema v0.2 contract in UTC, keep local_track_id unique per day with a registry file) was deleted together with its modules and config fields. Nothing used it after the shared gallery switched to track updates. It is in git history if the cross-camera side ever needs v0.2 tracklets again.
- The CVAT pre-annotation is built from record.py's output. IDs are the per-frame person_id. Frames between batches (every 5 frames) are filled by linear interpolation, because CVAT ends an imported track at any frame gap.

## Watch out
- Tracks that never got a person_id are not in the track-update file, so they are missing from the pre-annotation and must be added by hand in CVAT. Score IDF1 with predict_mot.py output, not with the converter's pred file.
- Detector conf 0.55 is above every tracker threshold (high 0.4, low 0.1, new 0.5), so those thresholds do nothing.
- People crossing the panorama's left/right edge get split into two tracks (no box spans the seam).
- config.yaml must match the config dataclasses exactly: load_config rejects unknown keys.

In the code: integrate/record.py (renamed from export_track_updates.py), person_pipeline/track_updates.py, person_pipeline/track_update_contract.py, groundtruth/tracklets_to_mot.py (--no-fill), groundtruth/predict_mot.py (write_mot moved here). Removed: person_pipeline/segments.py, tracklets.py, tracklet_contract.py; RecordConfig keeps only timezone; CameraConfig.recording_dir removed.
