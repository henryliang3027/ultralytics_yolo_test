---
title: Cupola boxes lag the display because they are sent only five times per second
author: HenryLiang
date: 2026-11-15
standing: true
id: l-da2ceec6
scope: repo
status: active
src: init-sessions
---

## Watch out
- The Cupola push runs in its own thread on a fixed interval, while the window updates every frame. Boxes and video also travel separate paths to the browser.
- In the code: Reduce cupola.send_interval_s in integrate/config.yaml (currently 0.2) to make boxes track people more smoothly.

_From past Claude sessions here, Nov 10._
