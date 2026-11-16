---
title: "ID switches after occlusion are caused by box size jumps, not long disappearance"
author: HenryLiang
date: 2026-11-15
standing: true
id: l-31377157
scope: repo
status: active
src: init-sessions
---

## Decided
- When an office partition hides or reveals legs, the YOLO box height jumps (e.g. 186 to 310), IoU falls, and the tracker issues a new ID. Raising match_thresh to 0.8 fixed 3 of 5 switches.
- In the code: TRACKTRACK_OVERRIDES in tracker.py has match_thresh 0.8. Remaining switches are handled by the gallery's Re-ID matching, which reassigns the person ID.

_From past Claude sessions here, Nov 10._
