# Shared gallery（PostgreSQL）與模擬 GUI

驗證「每台攝影機用自己的 separate gallery、每個 tick 送出 track update → 共用的 shared gallery 給 global_person_id」這條串接。

```
pipeline (integrate/) ──HTTP POST，每個 tick 一行 batch──▶ service.py ──▶ PostgreSQL（Docker，port 5433）
        ▲                                                     │
        └──────────── GET /mapping（GUI 右半邊）◀───────────────┘
```

| 檔案 | 內容 |
|---|---|
| `docker-compose.yml` | PostgreSQL 15 container `shared-gallery-db`，只聽 `127.0.0.1:5433`，跟 CVAT 的 `cvat_db` 分開 |
| `schema.sql` | 資料表（service 啟動時自動建立） |
| `gallery_store.py` | shared gallery 的邏輯：收 batch、給 global ID、合併 |
| `service.py` | HTTP 服務，收 batch 前先用 `integrate/person_pipeline/track_updates.py` 的同一套規則驗證 |
| `simulate.py` | GUI：左上 / 左下是兩台攝影機的影片，右邊是 PostgreSQL 裡的對應表和最新事件 |

## 環境

這個資料夾**不需要** GPU、torch、YOLO 或 Re-ID 模型。只需要：
- Python（測試過 3.14）
- `requirements.txt`：numpy、opencv-python、psycopg、requests
- Docker：跑 PostgreSQL

```bash
cd shared_gallery
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

**介面定義在 `integrate/person_pipeline/track_update_contract.py`**：欄位、時區、tick 規則、驗證。這個檔案只用 Python 標準函式庫，兩邊共用同一份。`service.py` 只從 `integrate/` 匯入這一個檔案。改它等於改介面，要兩邊先講好。

需要的資料（`videos/` 和 `results/` 不在 git 裡，要另外拿）：
- `videos/51_203_20261111_153342_153431.mp4`、`videos/51_205_20261111_153342_153431.mp4`
- `results/track_updates/20261111_153342_153431/track_updates.jsonl`（和 `summary.json`）

放在 repo 裡同樣的路徑，`simulate.py` 不加參數就找得到。

## 執行

```bash
cd shared_gallery
docker compose up -d                       # 1. 資料庫
.venv/bin/python service.py --verbose      # 2. 服務，http://127.0.0.1:8765（另開一個 terminal）
.venv/bin/python simulate.py               # 3. GUI：重播 track_updates.jsonl（再開一個 terminal）

# 直接跑 pipeline、邊處理邊送（真正的串接測試）：要用 integrate/ 的完整環境（GPU、模型）
../pvenv/bin/python simulate.py --source pipeline
```

- 按鍵：`q` 結束，空白鍵暫停。
- `--speed 1` 照真實速度（預設），`--speed 0` 盡量快。
- `--no-show --save out.mp4` 不開視窗，只存成影片。
- `--dump sent.jsonl` 把送出去的 batch 存下來（用來跟 export 的檔案比對）。
- `--top` / `--bottom` 換影片。影片檔名要是 `<camera>_<YYYYMMDD>_<HHMMSS>_<HHMMSS>.mp4`，攝影機 ID 和開始時間從檔名讀。用 jsonl 重播時，若 batch 的時間跟影片對不上會直接報錯。
- 每次執行會先清空 gallery（`POST /reset`），`--keep` 則保留。

## API

| | |
|---|---|
| `POST /track_updates` | body = 一個 batch（`track_updates.jsonl` 的一行）。回傳每筆的 `global_person_id`、`status`（`new` / `known`）和這次的 `merges`。格式不對回 400 並列出原因 |
| `GET /mapping` | `(camera_id, person_id) → global_person_id` 全表、合併紀錄、統計 |
| `POST /reset` | 清空。某台攝影機的 pipeline 重啟時要呼叫，因為 person_id 會從 1 重新編 |

同一個 `tick_ts` 送兩次只會處理一次（`batch_log` 以 `tick_ts` 為 key），回傳 `duplicate: true`。

## 怎麼給 global ID

1. **新出現的 `(camera_id, person_id)` 立刻給一個新的 global ID（暫定）。** 一直沒有 embedding 的人（框太小）也有 ID。
2. 有 embedding 就存進這個 global person 的外觀庫（每台攝影機保留最新 20 個）。
3. **每個 batch 之後檢查合併**，兩個 global person 同時符合以下條件就合併，較小的 ID 保留，另一個記到 `global_alias`：
   - 兩邊外觀庫都至少 5 個 embedding（`--min-embeddings`）
   - 兩邊**平均 embedding** 的 cosine ≥ 0.85（`--merge-threshold`）
   - 從來沒有在同一台攝影機同時出現過

**為什麼不在第一個 embedding 就決定**：在 2026-11-11 15:33:42 這段影片裡，藍衣人在 51_205 的第一個 embedding 跟他自己在 51_203 的外觀庫只有 0.78。前幾秒甚至更像 51_203 那位穿深藍上衣、坐著的男生（最高 0.88）。改用累積後的平均 embedding 比較，同一人是 0.92～0.96，其他所有跨攝影機配對都 ≤ 0.79。

## 資料表

| 表 | 一列代表 |
|---|---|
| `global_person` | 一個還存在的 global person |
| `person_map` | `(camera_id, person_id) → global_person_id`，加上這個人可見的時間範圍 |
| `embedding_bank` | 一個 embedding（256 維 `REAL[]`），屬於哪個 global person、哪台攝影機 |
| `global_alias` | 一次合併：`from_global_id` 併進 `to_global_id`，時間和相似度 |
| `batch_log` | 收到的一個 batch：筆數、新增幾個 global、合併幾次 |

直接查資料庫：`docker exec -it shared-gallery-db psql -U gallery -d shared_gallery`

## 驗證結果（2026-11-11 15:33:42–15:34:31，51_203 + 51_205）

- **pipeline 送出的 147 個 batch 和 `export_track_updates.py` 產生的檔案逐位元組相同**，service 全部接受。
- 結果：11 個 local person → 10 個 global person。唯一的合併是藍衣人：51_205 P6 先拿到 G009，15:33:53.667 併進 G002（51_203 P2，平均 cos 0.851），之後兩台都顯示 G002。沒有其他合併。
- service 每個 batch 約 5～10 ms。
- 格式錯誤的 batch（時區不是 +08:00、embedding 不是 256 維）會被擋下，重送的 batch 不會重複處理。

## 限制

- **門檻只在這一段 49 秒、11 個人的影片上調過。** 錯誤配對最高 0.79，門檻 0.85，餘裕不大，要用更長的影片確認。
- 藍衣人從出現到合併大約花了 **9 秒**。這段時間內兩台顯示的是不同 ID。
- 只靠外觀。還沒用位置或移動時間，因為那需要兩台攝影機的安裝位置和校正。
- 合併後不會再拆開，人也不會過期刪除。長時間執行需要加上清理機制。
- service 是單一執行緒，一次處理一個 batch。目前只有一個送出端，這樣就夠了。
