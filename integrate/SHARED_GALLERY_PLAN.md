# Shared Gallery: How to Split the Work

Goal: the same person gets the same person ID on both cameras. Today each camera has its own gallery, so person IDs are numbered per camera.

## How to split the work

The cleanest split is: **you own the pipeline, your colleague owns the gallery algorithm, and the two of you agree on the interface between them first.**

### 1. Agree on the interface first

The gallery is already isolated ([gallery.py](person_pipeline/gallery.py)). It only sees this per frame:

- **Input:** visible track IDs, new embeddings, boxes, timestamp
- **Output:** track ID → person ID

For a shared gallery, add the camera name to the input, for example a track key of `(camera, track_id)`.

Write this down as a small interface, with a dummy implementation. Once you both agree on it, neither of you needs to touch the other's code.

### 2. You prepare the pipeline side

- Add a config switch, something like `gallery.mode: per_camera | shared`. In shared mode, all `CameraStream`s get the same gallery object.
- Add a **record mode**. It saves per frame: camera, timestamp, track_id, box, embedding, and optionally a small crop image.

### 3. Give your colleague data, not cameras

This is the most important part.

Record a session where a few people walk between cam1 and cam2, and dump it to a file. Then label the ground truth once: "track 5 on cam1 and track 3 on cam2 are the same person".

With that file, your colleague can:

- Develop the algorithm in plain Python and numpy, with no GPU, no RTSP, no YOLO and no Cupola.
- Replay a whole session in seconds instead of watching video.
- Measure results with numbers instead of eyeballing. For example: how many people were wrongly merged, how many were wrongly split, and IDF1.

### 4. What your colleague actually has to solve

- **Matching across cameras:** lighting and angle differ a lot between cameras, so a single 0.65 threshold may not work. They may need a separate threshold for cam1-vs-cam2 matches.
- **The "one person can't be in two places at once" rule:** it holds within one camera. Across cameras it depends on whether the two views overlap.
- **Travel time:** if walking between the two areas takes at least, say, 5 seconds, a match 1 second after someone left the other camera is suspicious.
- **When two cameras' tracks claim the same person at the same moment:** decide which one wins.

### 5. Git workflow

- Make the first commit (`integrate/` hasn't been committed yet) and push it to a shared remote.
- Your colleague works on a branch such as `shared-gallery` and merges through a pull request.
- Final check: run it on the live cameras together.

## Is Docker the better choice?

**Docker solves a different problem.** It makes everyone's environment identical. It doesn't help split the task; the interface and the recorded data do that.

For this task specifically, **Docker isn't worth it now**:

| | Docker | No Docker |
|---|---|---|
| What your colleague needs | Recorded data + numpy, no GPU | Same |
| Your GPU setup | AMD ROCm. The container needs `/dev/kfd` and `/dev/dri` passed in, the image is 10+ GB, and it won't work on an NVIDIA machine. | A `requirements.txt` with pinned versions is enough |
| Effort | Writing and maintaining a Dockerfile | Little |

If your colleague develops against recorded data, they only need Python, numpy and the gallery code. A pinned `requirements.txt`, or `pip freeze` from `pvenv`, covers that.

**Docker becomes worth it later, for deployment.** When this pipeline runs permanently on a server next to MediaMTX and Cupola, a container gives you:

- automatic restart
- the same versions every time
- easy moves to another machine

That's a separate task from the gallery work.

## Summary

1. Agree on the gallery interface with the camera added.
2. Record a two-camera session with labels.
3. Your colleague develops and scores the algorithm offline on that recording.
4. Merge it through a pull request and test live together.
