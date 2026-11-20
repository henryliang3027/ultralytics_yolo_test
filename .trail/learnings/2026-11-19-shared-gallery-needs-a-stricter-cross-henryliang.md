---
title: "Shared gallery needs a stricter cross-camera threshold, and its iteration order must not come from a set"
author: HenryLiang
date: 2026-11-19
branch: main
cost: { minutes: 104, tokens: 526783 }
id: l-48712af7
scope: repo
status: active
src: agent
touches:
  - groundtruth/README.md
  - groundtruth/.gitignore
  - groundtruth/51_203_20261111_153342_153431/51_203_20261111_153342_153431_cvat_mot.zip
  - groundtruth/51_203_20261111_153342_153431/gt/gt.txt
  - groundtruth/51_203_20261111_153342_153431/gt/ids.jpg
  - groundtruth/51_203_20261111_153342_153431/pred/51_203_20261111_153342_153431.txt
  - groundtruth/51_203_20261111_153342_153431/pred/51_203_20261111_153342_153431_person.txt
  - groundtruth/51_203_20261111_153342_153431/pred/51_203_20261111_153342_153431_track.txt
  - groundtruth/eval_idf1.py
  - groundtruth/predict_mot.py
---

## Decided
- Shared gallery (one gallery for all cameras, keys (camera, track_id)) uses match_threshold 0.65 only for a person already seen on the same camera; a person never seen on this camera needs cross_camera_threshold 0.80.
- Why: on the 2026-11-11 15:33:42-15:34:31 clip (51_203 + 51_205), 0.65 merged two different seated people across cameras (a navy-shirt man with a woman in black); 0.75 still merged a back-of-head crop with a woman. 0.80 kept only the true link (blue jacket walker, 529/529 and 415/415 frames). Single-embedding-vs-bank similarity: true pair median 0.84 / p90 0.89; wrong pairs mostly 0.63-0.70; a similar-clothes man 0.79.
- "Cannot be in two places at once" is checked within one camera only, because the two cameras see the same person at the same moment.

## Tried and ruled out
- One threshold for all matches (0.65): false cross-camera merges, see above.

## Watch out
- Iterating a set of (camera, id) tuples gives a different order each process (string hash randomisation), so new person IDs got different numbers run to run. Iterate the tracker's order instead.
- 0.80 is tuned on one 49 s clip with ~12 people; recheck on a longer labelled clip.

In the code: integrate/person_pipeline/gallery.py (CameraGalleryView, make_galleries, _camera_of), gallery.mode / gallery.cross_camera_threshold in integrate/config.yaml, --gallery flag on main.py and record.py.
