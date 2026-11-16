---
title: "With several cameras, share the models but keep a separate tracker and gallery per camera"
author: HenryLiang
date: 2026-11-15
standing: true
id: l-69507b71
scope: repo
status: active
src: init-sessions
touches:
  - person_pipeline/camera.py
---

## Decided
- Detection and ReID models are stateless, but the tracker and gallery remember previous frames of one camera. Mixing cameras would corrupt track IDs, so each camera got its own gallery first, with a shared cross-camera gallery planned later.
- In the code: integrate/main.py --source takes multiple videos; one source means one camera. Per-camera state lives in person_pipeline/camera.py (CameraStream); the detector is batched. Cross-camera ID plan is in integrate/SHARED_GALLERY_PLAN.md.

_From past Claude sessions here, Nov 10._
