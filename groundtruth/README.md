# Ground truth for tracking metrics (IDF1 / HOTA)

The pipeline's output is used as a pre-annotation. It is fixed by hand in CVAT, exported in MOTChallenge format, and
then scored with TrackEval.

## CVAT (local, docker)

`cvat/` is CVAT v2.83.0 (`git clone --branch v2.83.0 https://github.com/cvat-ai/cvat.git`), not tracked in this repo.

```bash
cd groundtruth/cvat
CVAT_VERSION=v2.83.0 docker compose up -d       # start; UI at http://localhost:8080
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'   # first time only
docker compose stop                             # stop (data is kept in docker volumes)
```

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

```bash
D=51_203_20261111_153342_153431
../pvenv/bin/python eval_idf1.py --gt $D/gt/gt.txt --pred $D/pred/${D}_person.txt   # whole system
../pvenv/bin/python eval_idf1.py --gt $D/gt/gt.txt --pred $D/pred/${D}_track.txt    # tracker only
```

The script prints the overall IDF1, IDP and IDR, then one row per GT person with these columns:

- the prediction ID it matched;
- IDTP, IDFN, IDFP and IDF1;
- the other prediction IDs that covered this person.

It uses the same definition as TrackEval's Identity metric. The ID numbers in the GT and the prediction do not need to be the same.
