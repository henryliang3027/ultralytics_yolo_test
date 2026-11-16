---
title: "Gallery remembers people for 10 minutes in memory only, so IDs reset on restart"
author: HenryLiang
date: 2026-11-15
standing: true
id: l-6f00bf86
scope: repo
status: active
src: init-sessions
---

## Decided
- Each person keeps only their latest 20 appearance features, and is dropped after 600 s unseen. Nothing is saved to disk, and deleted IDs are never reused.
- In the code: integrate/config.yaml: max_embeddings_per_person 20, identity_ttl_s 600, match threshold 0.65. Cross-day or cross-camera recognition needs persistence or the shared gallery plan.

_From past Claude sessions here, Nov 13._
