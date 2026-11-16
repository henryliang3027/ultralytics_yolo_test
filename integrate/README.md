# 即時人員偵測、追蹤與 Re-ID Pipeline

從 RTSP 串流讀取 3840×1920 全景畫面，依序做人員偵測（YOLO）、多目標追蹤（TrackTrack）、外觀特徵擷取（Re-ID），再由 Gallery 維護每個人的長期 ID。人被隔板擋住、走出畫面後再回來時，仍能拿回原本的 ID。

可以同時處理多台攝影機（`config.yaml` 的 `cameras` 清單）。YOLO 和 Re-ID 模型只載入一份，所有攝影機共用；追蹤器和 Gallery 則每台攝影機各一份，所以 person ID 是**各攝影機分開編號**的（同一個人在 cam1 是 ID 2，在 cam2 可能是 ID 5）。

參考：`py/MediaMTX/yolov8s_reidnet_neuron_rtsp_client_webcam.py`（MediaTek APU 版本）。本版本改在 AMD GPU 上以 Ultralytics 執行偵測，並把每個功能拆成獨立模組。

## 處理流程

```
cam1 RTSP ──▶ FrameSource（背景執行緒，只留最新一張）──┐
cam2 RTSP ──▶ FrameSource（背景執行緒，只留最新一張）──┤
                                                       │ 每台有新畫格的攝影機各取一張
                                                       ▼
                        ┌── 主執行緒（PersonPipeline）─────────────────────┐
                        │ 1. PersonDetector  一次 batch 推論所有畫格（共用）│
                        └───────────────┬──────────────────┬──────────────┘
                                        ▼                  ▼
                        ┌─ CameraStream cam1 ─┐   ┌─ CameraStream cam2 ─┐
                        │ 2. PersonTracker    │   │ 2. PersonTracker    │  每台各自一份
                        │ 3. ReIDExtractor ◀──┼───┼── 共用同一個模型    │
                        │ 4. IdentityGallery  │   │ 4. IdentityGallery  │  每台各自一份
                        │ 5. Visualizer       │   │ 5. Visualizer       │
                        └──────────┬──────────┘   └──────────┬──────────┘
                                   ▼                         ▼
                  視窗「pipeline cam1」/ 錄影       視窗「pipeline cam2」/ 錄影
                  CupolaEventSink → cam1 的畫面     CupolaEventSink → cam2 的畫面
```

只有一台攝影機時，流程相同，只是 batch 裡只有一張畫格。

**為什麼這樣分：** 模型不記得前一張畫格（輸入影像、輸出結果），所以可以共用，GPU 記憶體只用一份，而且一次 batch 推論兩張比分開推論兩次快。追蹤器和 Gallery 會記住「上一張畫格的人在哪、長什麼樣」，兩台攝影機共用會把兩個房間的人混在一起，所以每台各一份，放在 `camera.py` 的 `CameraStream` 裡。

### 1. 讀取串流：只保留最新畫格

解碼在背景執行緒進行，只保留**最新一張**畫格，不排隊。推論比串流慢時，舊畫格會被直接丟掉，處理的永遠是最新畫面，延遲不會越跑越大。統計輸出中的 `dropped` 就是被跳過的畫格數。

### 2. 偵測：`PersonDetector`

Ultralytics YOLO（預設 `yolo26s.pt`），只保留 COCO `person` 類別。全景畫面中的人很小，所以用 `imgsz: 1920`。`conf: 0.1` 刻意保留低分框，交給追蹤器在人被部分遮擋時延續軌跡。

### 3. 追蹤：`PersonTracker`

直接呼叫 Ultralytics 內建的 `TRACKTRACK` 類別（CVPR 2025），不使用 `model.track()`。輸入偵測框和畫面，輸出每個框的短期 track ID（`-1` 表示尚未追蹤）。

`match_thresh: 0.8`（預設值 0.7）：人走到隔板後時，偵測框會在一兩幀內從全身變成上半身，預設門檻會把它當成新的人。

### 4. Re-ID：`ReIDExtractor`

把人裁切出來，縮放成 128×256，經 NVIDIA TAO ReIdentificationNet（ResNet50，Market-1501）得到 256 維特徵並做 L2 正規化。這個 ONNX 模型的 batch 固定為 1，所以逐張推論；如果換成 batch 可變動的模型，程式會自動一次送整批。

不是每個框每幀都算特徵，以節省時間：

| 情況 | 是否算特徵 |
|---|---|
| 新的 track，還沒有 person ID | 一定算，因為要靠它比對身分 |
| 已有 person ID | 每 `update_interval` 幀算一次，而且必須是「乾淨」的框：高度 ≥ `min_box_height`、分數 ≥ `min_conf`、沒有和別人重疊（IoU ≤ `max_overlap_iou`） |

### 5. Gallery：`IdentityGallery`

追蹤器的 track ID 只能維持短期：人被完全擋住或離開再回來，就會換新的 track ID。Gallery 為每個人保存最多 `max_embeddings_per_person`（預設 20）筆特徵，超過時丟掉最舊的一筆，並把 track ID 對應到長期的 person ID：

1. **已綁定的 track**：沿用 person ID，新特徵加入這個人的特徵庫。
2. **新的 track**：和「目前沒有出現在畫面中」的每個人比對 cosine 相似度。最高分 ≥ `match_threshold` 就沿用那個人的 ID，否則建立新的人。多個新 track 同時出現時，從相似度最高的組合開始分配，一個人只會被分給一個 track。
3. **例外**：如果某人目前的框大部分落在新 track 的框裡（或反過來），兩者視為同一個人。這是為了處理「隔板後的上半身框還殘留一兩幀，同時已出現新的全身框」的情況。
4. track 超過 `track_timeout_s` 沒出現就解除綁定；人超過 `identity_ttl_s` 沒出現就從 Gallery 移除。

畫面上，有 person ID 的框顯示 `ID n`，各自有固定顏色；還沒拿到 ID 的 track 顯示灰色的 `T<track id> ?`。

## 檔案結構

```
integrate/
├── config.yaml              # 所有參數
├── main.py                  # 進入點（命令列參數）
└── person_pipeline/
    ├── config.py            # 讀 config.yaml → 有型別的 dataclass，打錯 key 會直接報錯
    ├── types.py             # 模組之間傳遞的資料：FramePacket、Detections、FrameResult
    ├── capture.py           # 串流讀取：FrameSource 介面 + OpenCV / GStreamer 實作 + factory
    ├── detector.py          # 人員偵測模組
    ├── tracker.py           # 追蹤模組（TrackTrack）
    ├── reid.py              # Re-ID 模組（ONNX Runtime）
    ├── gallery.py           # Gallery 模組（長期身分）
    ├── camera.py            # CameraStream：一台攝影機自己的狀態（來源、追蹤器、Gallery、輸出、統計）
    ├── visualizer.py        # 繪圖
    ├── sinks.py             # 輸出介面 ResultSink：每張畫格的結果交給所有已註冊的輸出
    ├── cupola.py            # Cupola360 EVENT API 推送（ResultSink 實作）
    └── pipeline.py          # 主流程：共用模型、batch 偵測，再交給各攝影機
```

### 設計模式

| 模式 | 位置 | 用途 |
|---|---|---|
| **Strategy** | `capture.py` 的 `FrameSource` | OpenCV 和 GStreamer 實作同一個介面，換後端不影響其他模組 |
| **Factory** | `create_frame_source()` | 依 `source.backend` 建立對應的讀取器 |
| **Facade** | `pipeline.py` 的 `PersonPipeline` | 對外只有 `run()` / `stop()`，內部協調各模組和各攝影機 |
| **Observer** | `sinks.py` 的 `ResultSink` | 每張畫格處理完後通知所有輸出（目前是 Cupola）；新增輸出不用改主流程 |
| **單一職責** | 每個模組一個類別 | 偵測、追蹤、Re-ID、Gallery 互不依賴，透過 `types.py` 的資料結構溝通，可以單獨替換或測試 |

例如要換成別的 Re-ID 模型，只需改 `reid.py` 或 `config.yaml` 的 `reid.model`；要加新的讀取方式，就新增一個 `FrameSource` 子類別並註冊到 `_BACKENDS`。

## 安裝

所有套件都裝在專案的虛擬環境 `pvenv`：

```bash
cd /home/ubuntu/Documents/Github/ultralytics_yolo_test
pvenv/bin/python -m pip install ultralytics onnxruntime python-socketio requests websocket-client
```

需要的模型檔（路徑可在 `config.yaml` 修改）：

| 檔案 | 預設位置 |
|---|---|
| YOLO | `../yolo26s.pt`（專案根目錄） |
| Re-ID | `../../py/MediaMTX/resnet50_market1501_aicity156.onnx` |

`config.yaml` 裡的相對路徑都以 `config.yaml` 所在的資料夾為基準。

## 執行

```bash
cd /home/ubuntu/Documents/Github/ultralytics_yolo_test/integrate

# 讀 config.yaml 中所有攝影機的 RTSP 串流，每台一個視窗（按 q 結束）
../pvenv/bin/python main.py

# 不開視窗，把結果錄成影片（Ctrl+C 結束，影片仍會正常存好）
# 多台攝影機時每台一個檔案：output_cam1.mp4、output_cam2.mp4
../pvenv/bin/python main.py --no-show --save output.mp4

# 用本機影片離線測試（會照影片原本的 fps 播放，模擬即時串流）
# 給幾個來源就跑幾台：依序取代 cameras 清單的 url，空白或逗號分隔都可以
../pvenv/bin/python main.py --source ../videos/51_203_output20261106_095222.mp4
../pvenv/bin/python main.py --source ../videos/51_203_output20261106_095222.mp4 ../videos/51_205_output20261106_095222.mp4

# 用影片測試時不要把框推到 Cupola 的真實攝影機畫面上
../pvenv/bin/python main.py --source a.mp4 b.mp4 --no-cupola

# 指定其他設定檔、改用 GStreamer 讀取
../pvenv/bin/python main.py --config my_config.yaml --backend gstreamer
```

每隔 `logging.stats_interval_s` 秒，每台攝影機各印一行統計：

```
[stats cam1] 15.6 fps | detect 19.1ms track 0.8ms reid 14.8ms gallery 0.0ms | latency 49ms | dropped 7 frames | people 8
```

| 欄位 | 意義 |
|---|---|
| `fps` | 每秒實際處理的畫格數 |
| `detect` / `track` / `reid` / `gallery` | 各階段平均耗時。`detect` 是整個 batch（所有攝影機）一次推論的時間 |
| `latency` | 從畫格解碼完成到結果產生的時間（不含網路傳輸和攝影機編碼） |
| `dropped` | 為了維持即時而跳過的畫格數 |
| `people` | Gallery 中目前記得的人數 |

## 實測（AMD Radeon RX 9070 XT，Re-ID 用 CPU）

| 來源 | 處理速度 | 延遲 | 各階段 |
|---|---|---|---|
| RTSP 3840×1920 @ 15 fps，畫面中 8–10 人 | 約 15 fps（跟得上串流） | 約 50 ms | detect 19 ms、Re-ID 12–17 ms、track < 1 ms |
| 本機影片 @ 29 fps，同時錄影 | 約 21 fps | 約 40 ms | detect 18.5 ms、Re-ID 約 5 ms |
| 兩支本機影片 @ 29 fps，不開視窗 | 每台約 20–23 fps | 50–70 ms | detect（batch 2）29–32 ms、Re-ID 每台 4–12 ms |
| 兩支本機影片 @ 29 fps，開兩個視窗 | 每台約 13 fps | 60–80 ms | 兩張 4K 畫面的繪圖和顯示也要花時間 |
| 兩支本機影片 @ 29 fps，同時錄兩支影片 | 每台約 9 fps | 60–100 ms | 4K 影片編碼最花時間 |

兩台攝影機 batch 推論約 30 ms，比分開推論兩次（2 × 19 ms）快。實際的攝影機是 15 fps；不錄影時，兩台都跟得上。

在本機影片 `51_203_output20261106_095222.mp4` 上，一名穿黑衣、多次被隔板擋住的人（包含被完全擋住約 4 秒），追蹤器從第 22 幀起給了 5 個不同的 track ID；經過 Gallery 後，全部對應到同一個 person ID。他在影片開頭穿過全景左右接縫的那一段仍被當成另一個人（見「已知限制」）。

## 讀取後端：OpenCV 與 GStreamer

| | `opencv`（預設） | `gstreamer` |
|---|---|---|
| 狀態 | 已在本機測試 | **尚未在本機測試**：本機缺少下列套件 |
| 安裝 | pip 版 OpenCV 即可 | 需要 PyGObject 和解碼外掛 |
| 緩衝控制 | 背景執行緒只保留最新畫格；FFmpeg 端設定 `nobuffer`、`low_delay` | `rtspsrc latency=0` + `appsink drop=true max-buffers=1` |
| 硬體解碼 | 無（CPU 解碼） | 可以（AMD 用 VA-API） |

啟用 GStreamer 後端：

```bash
# 系統套件：H.264 parser / 解碼器、VA-API 硬體解碼、編譯 PyGObject 需要的標頭檔
sudo apt install gstreamer1.0-plugins-bad gstreamer1.0-libav libgirepository-2.0-dev libcairo2-dev pkg-config
# PyGObject 裝進 pvenv
pvenv/bin/python -m pip install PyGObject
# 確認外掛
gst-inspect-1.0 h264parse avdec_h264 vah264dec
```

要用 GPU 硬體解碼，把 `config.yaml` 的 `gst_decode` 改成 `rtph264depay ! h264parse ! vah264dec`；如果攝影機是 H.265，改用 `rtph265depay ! h265parse ! vah265dec`（或軟體解碼的 `avdec_h265`）。

## 常調整的參數

| 現象 | 調整 |
|---|---|
| 同一個人回來後拿到新 ID | 降低 `gallery.match_threshold`（例如 0.6） |
| 不同人被當成同一個 ID | 提高 `gallery.match_threshold`（例如 0.7），或改 `match_metric: mean` |
| 有人一直沒有框 / 沒有 track | 降低 `tracker.new_track_thresh`、`tracker.track_high_thresh` |
| 被隔板擋住時 track ID 跳號 | 提高 `tracker.match_thresh`（Gallery 通常會把 ID 接回來） |
| Re-ID 太慢 | 提高 `reid.update_interval` |
| 偵測太慢 | 降低 `detector.imgsz`（小的人會比較容易漏掉） |

## Cupola360 事件推送

把 Gallery 辨識出的每個人（框 + `ID n`）以 EVENT API v1.2 的 ROI 推送到 Cupola360 server，畫在對應攝影機的畫面上。實作移植自 `py/MediaMTX/cupola_event.py`，訊息格式與 JSON 範例見 `py/MediaMTX/README.md`。

`config.yaml`：

```yaml
cameras:
  - name: cam1
    url: rtsp://192.168.51.135:8554/fa57a71f-d37d-4fbd-a268-6a04e9ff852d
    cupola_camera_url: fa57a71f-d37d-4fbd-a268-6a04e9ff852d   # MediaMTX 串流名稱，不是完整的 RTSP URL
  - name: cam2
    url: rtsp://192.168.51.135:8554/76bb2c48-21cb-429a-999e-5afa2e4d9672
    cupola_camera_url: 76bb2c48-21cb-429a-999e-5afa2e4d9672

cupola:                    # 所有攝影機共用的推送設定
  enabled: true
  server: http://192.168.51.135:3000/
  resolution: auto         # 從第一張畫格取得，例如 3840x1920
  send_interval_s: 0.067   # 約 15 Hz
  box_ttl_ms: 1000
```

每台攝影機有自己的 `CupolaEventSink`（自己的連線），推到自己的 `cupola_camera_url`。Cupola 前端只顯示「目前正在看的那台攝影機」的事件，所以兩台的 person ID 都從 1 開始也不會互相干擾。

- `cupola_camera_url` 決定框要畫在哪台攝影機的畫面上。Cupola 前端只有在它**完全等於**攝影機的 `uri`、`relayUri` 或 `path` 其中一個欄位時，才會顯示（前端程式的 `getCameraUris`）。
  - 這裡的攝影機要用 `path`，也就是 MediaMTX 的串流名稱（`url` 最後一段的 UUID）。
  - 完整的 relay URL `rtsp://192.168.51.135:8554/fa57a71f-...` **對不上**：訊息會被 server 收下，但畫面上不會出現框，也不會有任何錯誤訊息。
  - 新增或更換攝影機時，把 `cupola_camera_url` 設成新攝影機的串流名稱（可以從 MediaMTX API `http://<host>:9997/v3/paths/list` 查到）。
- 某台的 `cupola_camera_url` 留空，那台就不推送；`enabled: false` 或命令列 `--no-cupola` 會全部停用。
- 用 `--source` 換成本機影片時，第 i 支影片仍推到第 i 台攝影機的 Cupola 畫面。影片不是那台攝影機的即時畫面，框會對不上，所以測試時建議加 `--no-cupola`。
- 框座標以全景畫面正規化成 0~1，所以來源必須是同一台攝影機、未裁切的 2:1 全景畫面。
- 推送的是 Gallery 的 **person ID**（不是追蹤器的 track ID），所以人被擋住再出現時，Cupola 上的 ID 也不會變。還沒拿到 person ID 的 track 不推送。
- 兩個 track 短暫對應到同一個 person ID 時（例如隔板後的上半身框和新的全身框），只送面積較大的框，避免同一個 ROI id 在一則訊息裡出現兩次。

流程：

1. 連到 `server`（Socket.IO，`socketio_path='/socket'`），連線後 emit 一次 `registerMeasuringPoint`。
2. 先送一則空的 `roiList` 加 `"forceUpdate": "SINGLE"`，清掉上次執行殘留的 ROI。
3. 每 `send_interval_s` 秒送一次最新畫格的人物框；離開畫面的人送 `displayTime: 0` 關掉。
4. 程式結束時再清空一次，然後斷線。

網路 I/O 在獨立的背景執行緒；主流程只把最新結果交過去，不會等待。Cupola server 慢或連不上，都不會拖慢偵測。斷線時會自動重連，每次重連都會先清空一次。

## 已知限制

- **跨攝影機不共用身分**：每台攝影機各自一個 Gallery，同一個人在不同攝影機會拿到不同的 person ID。要跨攝影機認人，需要改成共用 Gallery（track ID 要加上攝影機名稱區分，「同一時間只能在一個地方」的規則也要調整，因為兩台的視野可能重疊）。
- **全景接縫**：人穿過畫面左右邊界時，追蹤器會斷開；Re-ID 有時接得回來，但邊緣影像變形，不一定成功。
- **第一次比對只用一張特徵**：新 track 一出現就用第一張裁切決定身分。如果那一張被遮住大半，可能配錯人或建立新的人。
- **Re-ID 在 CPU 上執行**：pip 版 `onnxruntime` 只有 CPU。畫面中人多時，`reid` 時間會隨人數增加。
- **錄影的播放速度**：錄影以串流的 fps 寫入，處理過程中跳過的畫格不會寫入，所以錄出來的影片播放起來會比實際時間稍快。
