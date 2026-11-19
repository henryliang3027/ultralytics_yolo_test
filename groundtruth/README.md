# Ground truth for tracking metrics (IDF1 / HOTA)

The pipeline's output is used as a pre-annotation. It is fixed by hand in CVAT, exported in MOTChallenge format, and
then scored with TrackEval.

## CVAT (local, docker)

`cvat/` is CVAT v2.83.0. It is not tracked in this repo. To set it up the first time:

```bash
cd groundtruth
git clone --branch v2.83.0 https://github.com/cvat-ai/cvat.git cvat
```

Start, stop and check it:

```bash
cd groundtruth/cvat
CVAT_VERSION=v2.83.0 docker compose up -d       # start; UI at http://localhost:8080
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'   # first time only: make a login
docker compose ps                               # all cvat_* containers should be "running"
docker compose logs -f cvat_server              # see server logs if the UI does not load
docker compose stop                             # stop (data is kept in docker volumes)
```

- `CVAT_VERSION` already defaults to `v2.83.0` in this checkout's `docker-compose.yml`. Passing it just makes the
  version explicit.
- `docker compose down` also keeps the data. `docker compose down -v` deletes the volumes, which removes every task
  and annotation.
- The first start takes a few minutes. Open the UI after `cvat_server` has finished its migrations.

## 1. Pre-annotation from tracklets

```bash
cd groundtruth
../pvenv/bin/python tracklets_to_mot.py ../results/tracklets/20261111_153342_153431/tracklets.jsonl \
    --camera 51_203 --video ../videos/51_203_20261111_153342_153431.mp4
```

This writes `51_203_20261111_153342_153431/` with two files:

- `*_cvat_mot.zip`: the pre-annotation to import into CVAT.
- `pred/*.txt`: the same boxes as a TrackEval tracker result.

## 2. Fix in CVAT

1. Create a task with the label `person` and upload the mp4 as the data.
2. Use **Actions → Upload annotations → MOT 1.1** to load the zip. Each ID becomes one track.
3. Fix the annotation:
   - Add boxes for missed frames. Mark keyframes and CVAT interpolates the frames between them.
   - Merge tracks of the same person.
   - Change wrong IDs.
   - Delete false boxes.
   - Make every box cover the visible person.
4. Use **Export annotations → MOT 1.1** and save the result as `<video name>/gt/gt.txt`.

## Annotation rules (keep the same for every video)

| Case | Rule |
|---|---|
| Person split at the 0/3840 seam | Box only the larger part. |
| Occlusion | Box only the visible part. If nothing is visible, add no box. |
| Leaves and comes back | Keep the same ID. |
| Reflections, tiny far people | Set `conf=0` (ignored), or do not annotate. |
| Static targets for the whole video | Annotate them if they are real people. |

## 3. Prediction (per frame)

```bash
../pvenv/bin/python predict_mot.py ../videos/51_203_20261111_153342_153431.mp4
```

This runs the pipeline (same config as record mode) over every frame of the mp4 and writes two files to `<video name>/pred/`:

- `*_person.txt`: the person ID shown on each frame. Use it for the system's IDF1.
- `*_track.txt`: the raw tracker ID (`local_track_id`). Use it for the tracker's IDF1.

For new recordings, `record.py --mot` writes the same files to `<out>/mot/`.

Do not use `pred/<video name>.txt` from `tracklets_to_mot.py` for scoring. Its IDs are one majority vote per tracklet,
so it hides ID errors and the score comes out too high.

## 4. IDF1

Run from `groundtruth/`. You need `gt/gt.txt` from step 2 and the `pred/` files from step 3.

```bash
cd groundtruth
D=51_203_20261111_153342_153431
../pvenv/bin/python eval_idf1.py --gt $D/gt/gt.txt --pred $D/pred/${D}_person.txt   # whole system
../pvenv/bin/python eval_idf1.py --gt $D/gt/gt.txt --pred $D/pred/${D}_track.txt    # tracker only
```

| Option | Default | Meaning |
|---|---|---|
| `--gt` | (required) | MOT ground-truth file (`frame,id,x,y,w,h,conf,...`, frames start at 1). |
| `--pred` | (required) | MOT prediction file in the same format. |
| `--iou` | `0.5` | A GT box and a predicted box match in a frame when their IoU is at least this value. |

GT rows with `conf=0` are ignore regions. They are dropped, along with any predicted boxes that overlap them.

The script prints the overall IDF1, IDP and IDR, then one row per GT person with these columns:

- the prediction ID it matched;
- IDTP, IDFN, IDFP and IDF1;
- the other prediction IDs that covered this person.

It uses the same definition as TrackEval's Identity metric. The ID numbers in the GT and the prediction do not need to be the same.
