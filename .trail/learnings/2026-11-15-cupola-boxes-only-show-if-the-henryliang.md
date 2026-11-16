---
title: "Cupola boxes only show if the event URL equals the camera's registered stream name"
author: HenryLiang
date: 2026-11-15
standing: true
id: l-d8a48c45
scope: repo
status: active
src: init-sessions
---

## Watch out
- Boxes did not appear on Cupola even though connection and sends looked fine. The viewer matches the event url exactly against the camera's uri, relayUri or path, and silently drops it otherwise.
- In the code: In integrate/config.yaml set cupola.camera_url to the MediaMTX stream name (e.g. fa57a71f-d37d-4fbd-a268-6a04e9ff852d), not the full rtsp:// URL. Server returns no error on mismatch, so verify visually with a static test box.

_From past Claude sessions here, Nov 10._
