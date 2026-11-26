---
title: Low-latency FFmpeg options are only for RTSP; on video files they cause gray blocks at the start
author: HenryLiang
date: 2026-11-26
branch: main
cost: { minutes: 12, tokens: 45460 }
id: l-ac26db4d
scope: repo
status: active
src: agent
touches:
  - integrate/person_pipeline/capture.py
---

## Decided
The FFmpeg options nobuffer and low_delay are applied only to network streams. Local video files are opened without them.

## Tried and ruled out
Testing each option separately showed that only nobuffer causes the problem. low_delay and rtsp_transport do not, and plain cv2.VideoCapture is fine. nobuffer drops the packets FFmpeg reads while probing the stream, and those include the first keyframe. Decoding then starts on a P-frame, FFmpeg prints "first frame is no keyframe" and uses a gray dummy reference frame, so the picture is gray noise until the next I-frame. The test clips are mp4v with a GOP of 12, so this lasts about 0.8 s.

## Watch out
RTSP keeps nobuffer for latency. If a camera connects in the middle of a GOP, a short gray burst at connect time is expected and clears on its own.

In the code: person_pipeline/capture.py OpenCVSource._open checks os.path.isfile(url) and pops OPENCV_FFMPEG_CAPTURE_OPTIONS for files.
