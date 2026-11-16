---
title: "Use GStreamer for low-latency RTSP, but it is untested and the default is OpenCV"
author: HenryLiang
date: 2026-11-15
standing: true
id: l-8c63ed7d
scope: repo
status: active
src: init-sessions
---

## Watch out
- Frame queue buildup is the biggest latency risk. The GStreamer backend exists, but this machine lacks PyGObject and the H.264 plugins, so it never ran.
- In the code: integrate/config.yaml has backend: opencv. The OpenCV path needs a thread that keeps only the newest frame. Switching to --backend gstreamer fails until gi and h264parse/avdec_h264 are installed.

_From past Claude sessions here, Nov 10._
