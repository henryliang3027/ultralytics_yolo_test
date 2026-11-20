# Shared gallery（PostgreSQL）與模擬 GUI

兩台攝影機（51_203、51_205）各自用 separate gallery 給 `person_id`，每 1/3 秒把畫面上的人送出來（track update）。
這個資料夾是**接收端**：把每台的 `(camera_id, person_id)` 對應到跨攝影機共用的 `global_person_id`，存在 PostgreSQL。

```
pipeline（integrate/，送出端）──HTTP POST，每 1/3 秒一批──▶ service.py ──▶ PostgreSQL（Docker，port 5433）
   或 track_updates.jsonl 重播                                  │
        ▲                                                       │
        └──────────── GET /mapping（GUI 右半邊的表）◀─────────────┘
```

**不需要** GPU、torch、YOLO 或 Re-ID 模型，只要 Python、Docker 和 `requirements.txt`。

---

## 快速開始

### 1. 取得程式

```bash
git clone https://github.com/henryliang3027/ultralytics_yolo_test.git   # 已經有就 git pull
cd ultralytics_yolo_test
```

### 2. 放測試資料

`videos/` 和 `results/` 不在 git 裡，跟 Henry 拿這三樣，**放在 repo 裡下面這些路徑**：

```
ultralytics_yolo_test/
├── videos/
│   ├── 51_203_20261111_153342_153431.mp4                           (115 MB)
│   └── 51_205_20261111_153342_153431.mp4                           ( 99 MB)
└── results/track_updates/20261111_153342_153431/
    ├── track_updates.jsonl                                         (2.2 MB，147 批)
    └── summary.json
```

確認檔案都在：

```bash
ls -la videos/51_20?_20261111_153342_153431.mp4 results/track_updates/20261111_153342_153431/
```

### 3. 建立環境（第一次）

```bash
cd shared_gallery
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt       # numpy、opencv-python、psycopg、requests
```

測試過 Python 3.14。`.venv/` 已在 `.gitignore`。

### 4. 啟動資料庫

```bash
docker compose up -d                            # container 名稱 shared-gallery-db
docker compose ps                               # STATUS 要是 healthy
```

只聽本機 `127.0.0.1:5433`，資料存在 Docker volume `shared_gallery_pgdata`。

### 5. 啟動 service（開一個 terminal，保持開著）

```bash
cd shared_gallery
.venv/bin/python service.py --verbose
```

看到這行就是好了：

```
[service] http://127.0.0.1:8765  db 127.0.0.1:5433/shared_gallery  GalleryParams(merge_threshold=0.85, ...)
```

資料表在第一次啟動時自動建立。`--verbose` 會在每一批進來時印一行。

### 6. 開 GUI（再開一個 terminal）

```bash
cd shared_gallery
.venv/bin/python simulate.py
```

- 會照真實速度播 49 秒的影片，每 1/3 秒送一批給 service。
- 左上是 51_203、左下是 51_205，框上標 `P2>G002`，代表 person_id → global_person_id。
- 右邊是資料庫裡的對應表，兩台都有的人整列標綠色；下方是最新事件（新增、合併）。
- 按 `q` 結束，空白鍵暫停。
- 每次執行會先清空 gallery 再開始。

### 7. 確認結果正確

結束時 terminal 會印出：

```
[sim] 735 frames, 147 batches sent; assignment statuses {'new': 11, 'known': 1001, 'merges': 1}
  G001: 51_203/P1
  G002: 51_203/P2, 51_205/P6   <- both cameras
  ...
```

**目前的正確答案**：
- 藍色外套走動的人是 **51_203 P2 和 51_205 P6**，要是同一個 G。
- 他在 51_205 一開始會拿到暫定的 G009，大約 **15:33:53.7** 併進 G002。
- **沒有其他合併**：其他 9 個人各自一個 G。

不想看畫面、只想快速驗證時：

```bash
.venv/bin/python simulate.py --no-show --speed 0           # 幾秒跑完，只印結果
```

---

## 修改與測試（日常開發）

你主要會改的是 **`gallery_store.py`**（給 ID、合併的規則）。流程：

1. 改 `gallery_store.py`，或用參數調門檻（見下表）。
2. **重啟 service**：在 service 的 terminal 按 Ctrl+C，再執行一次 `service.py`。程式碼是啟動時載入的，不重啟不會生效。
3. 跑 `.venv/bin/python simulate.py --no-show --speed 0`，看印出的對應結果。
4. 跟上面的「正確答案」比對，再開 GUI 看細節。

調整用的參數（`service.py`）：

| 參數 | 預設 | 作用 |
|---|---|---|
| `--merge-threshold` | 0.85 | 兩個 global person 平均 embedding 的 cosine 達到多少才合併 |
| `--min-embeddings` | 5 | 兩邊各要至少幾個 embedding 才考慮合併 |
| `--bank-size` | 20 | 每人每台攝影機保留最新幾個 embedding |
| `--port` / `--dsn` | 8765 / 本機 5433 | 服務 port、資料庫連線 |

例：`.venv/bin/python service.py --verbose --merge-threshold 0.80`

直接查資料庫：

```bash
docker exec -it shared-gallery-db psql -U gallery -d shared_gallery
# SELECT * FROM person_map ORDER BY global_person_id;
# SELECT * FROM global_alias;
```

`schema.sql` 的表結構如果改了，舊的表不會自動更新：先清掉再重啟 service。

```bash
docker compose down -v && docker compose up -d     # -v 會刪掉整個資料庫
```

---

## 分工：哪些可以自己改

| 檔案 | 誰維護 |
|---|---|
| `shared_gallery/gallery_store.py`、`service.py`、`schema.sql`、`simulate.py`（jsonl 模式） | **你** |
| `integrate/`（偵測、追蹤、Re-ID、separate gallery、產生 track update） | Henry |
| **`integrate/person_pipeline/track_update_contract.py`**（介面：欄位、時區、送出時機、驗證） | **共同，改之前要先講好** |

`service.py` 只從 `integrate/` 匯入 `track_update_contract.py` 這一個檔案，它只用 Python 標準函式庫。收到的每一批都用它驗證，不符合就回 400 並列出原因。

## 送進來的資料長什麼樣

`track_updates.jsonl` 一行就是一批，等於 service 收到的一次 POST：

```json
{"tick_ts": "2026-11-11T15:33:42.333333+08:00", "records": [
  {"timestamp": "2026-11-11T15:33:42.333333+08:00", "camera_id": "51_203", "person_id": 2,
   "bbox": [x1, y1, x2, y2], "embedding": [256 個數]},
  {"timestamp": "...", "camera_id": "51_205", "person_id": 6, "bbox": [...], "embedding": null}
]}
```

| 欄位 | 說明 |
|---|---|
| `tick_ts` / `timestamp` | ISO 8601 台灣時間（+08:00），每秒的 .000 / .333 / .667 |
| `camera_id` | `51_203` 或 `51_205` |
| `person_id` | 該攝影機自己的 ID。**pipeline 重啟後從 1 重新編**，所以重啟時要呼叫 `POST /reset` |
| `bbox` | `[x1, y1, x2, y2]`，3840×1920 畫面的像素座標 |
| `embedding` | 256 維、L2 正規化（模型 resnet50_market1501_aicity156）。畫面不乾淨時是 `null`：框高 < 48 px，或跟別人重疊 IoU > 0.3 |

- 每批包含兩台所有已有 person_id 的人，同一台同一個人只出現一次。
- 檢查檔案格式：在 Henry 的完整環境裡執行 `python integrate/export_track_updates.py --validate <檔案>`。

## API

| | |
|---|---|
| `POST /track_updates` | body 是一批（jsonl 的一行）。回傳每筆的 `global_person_id`、`status`（`new` / `known`）和這次的 `merges`。格式不對回 400 並列出原因 |
| `GET /mapping` | `(camera_id, person_id) → global_person_id` 全表、合併紀錄、統計 |
| `POST /reset` | 清空 gallery |
| `GET /health` | 確認服務活著 |

同一個 `tick_ts` 送兩次只會處理一次，回傳 `duplicate: true`。

## GUI 參數（`simulate.py`）

| 參數 | 作用 |
|---|---|
| `--speed 1` / `--speed 0` | 照真實速度（預設）/ 盡量快 |
| `--no-show` | 不開視窗 |
| `--save out.mp4` | 把 GUI 畫面存成影片 |
| `--top` / `--bottom` / `--updates` | 換影片或 jsonl。影片檔名要是 `<camera>_<YYYYMMDD>_<HHMMSS>_<HHMMSS>.mp4`；jsonl 跟影片時間對不上會直接報錯 |
| `--keep` | 不清空 gallery，接著上次的資料 |
| `--dump sent.jsonl` | 把送出的批次存下來 |
| `--source pipeline` | 直接跑 pipeline、邊處理邊送。**要用 integrate/ 的完整環境**（GPU、模型），Henry 用來做串接測試 |

---

## 目前怎麼給 global ID

1. **新出現的 `(camera_id, person_id)` 立刻給一個新的 global ID（暫定）。** 一直沒有 embedding 的人也有 ID。
2. 有 embedding 就存進這個人的外觀庫（每台攝影機保留最新 20 個）。
3. **每批之後檢查合併**。兩個 global person 同時符合以下條件就合併，較小的 ID 保留，另一個記到 `global_alias`：
   - 兩邊都至少 5 個 embedding
   - 兩邊**平均 embedding** 的 cosine ≥ 0.85
   - 從來沒有在同一台攝影機同時出現過

**為什麼不在第一個 embedding 就決定**：在這段影片裡，藍衣人在 51_205 的第一個 embedding 跟他自己在 51_203 的外觀庫只有 0.78。前幾秒他甚至更像 51_203 那位穿深藍上衣坐著的男生（最高 0.88）。改用累積後的平均 embedding，同一人是 0.92～0.96，其他所有跨攝影機配對都 ≤ 0.79。

## 資料表

| 表 | 一列代表 |
|---|---|
| `global_person` | 一個還存在的 global person |
| `person_map` | `(camera_id, person_id) → global_person_id`，以及這個人可見的時間範圍 |
| `embedding_bank` | 一個 embedding（256 維 `REAL[]`），屬於哪個 global person、哪台攝影機 |
| `global_alias` | 一次合併：`from_global_id` 併進 `to_global_id`，時間和相似度 |
| `batch_log` | 收到的一批：筆數、新增幾個 global、合併幾次 |

## 已知限制（可以改進的方向）

- **門檻只在這一段 49 秒、11 個人的影片上調過。** 錯誤配對最高 0.79，門檻 0.85，餘裕不大。
- **目前只確認了藍衣人這一組正確答案**，其他人沒有正式標註，所以還不能算錯誤合併率或錯誤拆分率。
- 藍衣人從出現到合併大約 **9 秒**，這段時間兩台顯示不同 ID。
- 只靠外觀。還沒用位置或移動時間，那需要兩台攝影機的安裝位置和校正。
- 合併後不會再拆開，人也不會過期刪除。長時間執行需要清理機制。
- service 是單一執行緒，一次處理一批。

## 常見問題

| 狀況 | 處理 |
|---|---|
| `shared gallery service not reachable at http://127.0.0.1:8765` | service 沒開，先做第 5 步 |
| service 啟動時連不到資料庫 | `docker compose ps` 確認 container 是 healthy；沒有就 `docker compose up -d` |
| `... starts at ... but the 51_205 video starts at ...: the batches do not belong to these videos` | jsonl 跟影片不是同一段時間，檢查第 2 步的檔名 |
| `video names must look like ...` | 影片檔名被改過，要維持 `<camera>_<YYYYMMDD>_<HHMMSS>_<HHMMSS>.mp4` |
| port 5433 或 8765 被占用 | 改 `docker-compose.yml` 的 port 並在 service 加 `--dsn`；service 用 `--port`，GUI 用 `--service` |
| 改了 `gallery_store.py` 沒效果 | service 要重啟 |
| 想從頭來 | `docker compose down -v && docker compose up -d`，再重啟 service |
