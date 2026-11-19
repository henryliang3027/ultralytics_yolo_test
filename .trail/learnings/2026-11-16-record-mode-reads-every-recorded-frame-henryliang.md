---
title: Record mode reads every recorded frame and keeps track IDs unique per camera per day through a registry file
author: HenryLiang
date: 2026-11-16
branch: main
cost: { minutes: 1, tokens: 1732 }
id: l-3b74457e
scope: repo
status: active
src: agent
---

## Decided
- Record mode exports closed tracks for the cross-camera side in the shared tracklet_schema v0.2 contract. It reads Cupola recording segments directly and processes every frame (no real-time dropping), because the contract defines frame time as segment file-name time + frame_index / fps.
- The tracker numbers tracks from 1 in every new process, so (camera_id, local_track_id) would repeat within a day. A per-camera, per-local-day registry file stores the next free ID, and each run adds that offset.
- A tracklet closes after track_buffer (30) processed frames unseen = 2 s at the recordings' real 15 fps (not 30 fps as the colleague's notes say). Closing earlier would split tracks the tracker can still re-find.
- The contract is never edited here; questions go to the colleague. The installed tracklet_schema package is used when present, otherwise a verbatim local copy.

## Watch out
- frame_index / fps differs from the frame PTS by up to ~0.23 s, and a few frames step back up to 60 ms at segment boundaries.
- Detector conf 0.55 is above every tracker threshold (high 0.4, low 0.1, new 0.5), so those thresholds do nothing.
- People crossing the panorama's left/right edge get split into two tracks (no box spans the seam).

In the code: integrate/record.py, person_pipeline/tracklets.py (TrackIdRegistry, TrackletBuilder), person_pipeline/segments.py, person_pipeline/tracklet_contract.py; registry at results/tracklets/track_id_registry.json (record.out_dir).
