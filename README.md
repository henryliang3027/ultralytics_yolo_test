# ultralytics_yolo_test

## 兩台影片 → tracklets → shared gallery global ID → 重播影片

### 1. 製作 tracklets

不加 `--id-type` 時預設是 `pid`，shared gallery 只吃 pid。

```bash
cd /home/ubuntu/Documents/Github/ultralytics_yolo_test/integrate
../pvenv/bin/python record.py \
    ../videos/51_203_20261111_153342_153431.mp4 \
    ../videos/51_205_20261111_153342_153431.mp4 \
    --overwrite
```

### 2. 啟動 shared gallery service（保持開著）

```bash
cd /home/ubuntu/Documents/Github/ultralytics_yolo_test/shared_gallery
docker compose up -d
../pvenv/bin/python service.py --verbose
```

### 3. 重播 jsonl 取得 global ID，存成影片（另開 terminal）

```bash
cd /home/ubuntu/Documents/Github/ultralytics_yolo_test/shared_gallery
../pvenv/bin/python simulate.py \
    --updates ../results/track_updates/20261111_153342_153431/tracklets.jsonl \
    --save ../results/shared_gallery_sim/replay_jsonl.mp4 \
    --no-show --speed 0
```

<br>
<br>

## 整個資料夾批次製作 tracklets（`record_folder.sh`）

資料夾裡的 `51_203_<時間>.mp4` 和 `51_205_<時間>.mp4` 自動配對，輸出到 `<資料夾>/<id-type>/<時間>/`。

```bash
cd /home/ubuntu/Documents/Github/ultralytics_yolo_test/integrate
./record_folder.sh <資料夾> [pid|tid|both，預設 both] [其他 record.py 參數]

# 例
./record_folder.sh /run/media/ubuntu/TRANS32_2/At_Least_1_Minute/Easy both
./record_folder.sh /run/media/ubuntu/TRANS32_2/At_Least_1_Minute/Hard both --overwrite
```

已有 `tracklets.jsonl` 的組會跳過，加 `--overwrite` 重跑。
