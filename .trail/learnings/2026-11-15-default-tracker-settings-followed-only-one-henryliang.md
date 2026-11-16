---
title: Default tracker settings followed only one person on this wide panorama video
author: HenryLiang
date: 2026-11-15
standing: true
id: l-2c7002ee
scope: repo
status: active
src: init-sessions
---

## Watch out
- The 3840x1920 frame is shrunk to 640 px by default, so people are tiny and score low. Default tracker thresholds then dropped most of them.
- In the code: In tracker.py use imgsz=1920 and the lowered TRACKTRACK_OVERRIDES. Keep predict conf at or below track_high_thresh (0.4), since conf=0.6 starves the tracker of low-score boxes it needs to keep tracks.

_From past Claude sessions here, Nov 10._
