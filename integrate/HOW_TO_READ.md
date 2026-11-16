# 怎麼閱讀這個專案的程式（初學者導讀）

這份文件不重複 README 講過的「這個系統做什麼」，而是教你**怎麼讀程式碼**：從哪個檔案開始、每個檔案要看哪幾行、看到不熟的 Python 語法時它在做什麼，以及怎麼自己動手驗證。

建議一邊開著程式碼一邊讀。文中的 `檔名:行號` 都是連結：在 VS Code 編輯器裡 Ctrl+點擊，或按 Ctrl+Shift+V 開預覽後直接點，會跳到該檔案的那一行。

---

## 0. 先建立整體地圖

整個程式只做一件事：**一張一張拿影像，每張影像依序經過幾個關卡，最後把結果畫出來、推到 Cupola。**

```
攝影機 (RTSP)
   │
   ▼
capture.py      拿到一張影像  → FramePacket
   │
   ▼
detector.py     YOLO 找出人   → Detections (框 + 分數)
   │
   ▼
tracker.py      TrackTrack    → 每個框一個 track id（短期編號，被遮住可能會換）
   │
   ▼
reid.py         ReID 模型     → 每個人一個 256 維的「外觀特徵」
   │
   ▼
gallery.py      比對特徵       → 每個 track 對應到 person id（長期編號，換了 track 也能認回來）
   │
   ├──► visualizer.py  畫在畫面上
   └──► cupola.py      推送到 Cupola 網頁
```

`pipeline.py` 是「指揮官」，`camera.py` 是「每台攝影機自己的工作台」，兩個一起負責按照上面的順序呼叫每個模組。**其他檔案彼此不認識對方。** 這就是為什麼你可以單獨讀任何一個模組，不用擔心它偷偷呼叫別的地方。

有兩台攝影機時，上面這條流程就有兩條。YOLO 和 ReID 模型兩條共用（只載入一份），追蹤器和 gallery 每台各一份，因為它們要記住「這台攝影機」上一張畫面的人。

---

## 1. 建議閱讀順序

不要照檔名字母順序讀。照「資料流動的順序」讀：

| 順序 | 檔案 | 為什麼這時候讀 | 大約行數 |
|---|---|---|---|
| 1 | `main.py` | 程式的入口，看它怎麼啟動 | 73 |
| 2 | `config.yaml` + `person_pipeline/config.py` | 所有參數從哪裡來 | 74 + 151 |
| 3 | `person_pipeline/types.py` | 各關卡之間傳遞的「資料長什麼樣」 | 43 |
| 4 | `person_pipeline/pipeline.py` → `person_pipeline/camera.py` | 指揮官 + 每台攝影機的工作台，看完就懂整體流程 | 80 + 195 |
| 5 | `detector.py` → `tracker.py` → `reid.py` → `gallery.py` | 依資料流向，一個一個關卡細讀 | 各 40–150 |
| 6 | `capture.py` | 拿影像的細節（有多執行緒，較難，放後面） | 191 |
| 7 | `visualizer.py`、`sinks.py`、`cupola.py` | 輸出端 | 53 + 23 + 238 |

**第 3 步很重要。** 先知道資料的形狀，後面讀每個函式時就知道「輸入是什麼、輸出是什麼」，不會迷路。

---

## 2. 一步一步讀

### 2.1 `main.py`：入口

從 [main.py:72](main.py#L72) 開始讀：

```python
if __name__ == "__main__":
    main()
```

意思是：「直接執行這個檔案時，呼叫 `main()`」（被別的檔案 import 的時候不會執行）。

接著看 `main()`（[main.py:46](main.py#L46)），它只做四件事：

1. `parse_args()`：讀命令列參數（`--source`、`--no-show` 等）
2. `load_config(...)`：讀 `config.yaml`
3. 用命令列參數**覆蓋** config 的值。例如 `--source a.mp4 b.mp4` 會依序蓋掉 `cameras` 清單裡前兩台的 `url`，而且只跑這兩台（[main.py:36](main.py#L36) `override_sources()`）
4. `PersonPipeline(cfg)` 建立 pipeline，然後 `pipeline.run()` 開始跑

`signal.signal(...)` 那段（[main.py:62-67](main.py#L62-L67)）是讓你按 Ctrl+C 時能正常收尾（關閉攝影機、清掉 Cupola 上的框），而不是直接斷掉。

> **讀完你應該知道：** 程式的主體在 `PersonPipeline.run()`。

### 2.2 `config.py`：參數怎麼變成物件

`config.yaml` 是文字檔，程式需要把它變成 Python 物件。看 [config.py:13](person_pipeline/config.py#L13)：

```python
@dataclass
class CameraConfig:
    name: str
    url: str
    cupola_camera_url: str = ""
```

**`@dataclass` 是什麼？** 它幫你自動產生 `__init__`，等於你寫了：

```python
class SourceConfig:
    def __init__(self, name, url, cupola_camera_url=""):
        self.name = name
        self.url = url
        self.cupola_camera_url = cupola_camera_url
```

所以之後程式裡寫 `cfg.detector.imgsz`，就是在拿 yaml 裡 `detector:` 底下的 `imgsz:`。**yaml 的層級 = 點號的層級。** `cameras:` 是清單（每項開頭有 `-`），所以 `cfg.cameras` 是一個 list，`cfg.cameras[0].url` 是第一台攝影機的網址。

`_build()`（[config.py:106](person_pipeline/config.py#L106)）多做了一件事：如果你在 yaml 打錯字（例如 `imgszz`），它會直接報錯，而不是默默忽略。

> **讀完你應該知道：** 看到程式裡的 `self.cfg.xxx`，就去 `config.yaml` 對應的段落找它的值和註解。

### 2.3 `types.py`：資料長什麼樣

三個 dataclass，全部是「裝資料的盒子」，沒有邏輯：

- **`FramePacket`**（[types.py:10](person_pipeline/types.py#L10)）：一張影像
  - `image`：numpy 陣列，形狀 `(高, 寬, 3)`，顏色順序是 BGR（OpenCV 的習慣，不是 RGB）
  - `frame_id`：第幾張
  - `timestamp`：解碼完成的時間，用來算延遲

- **`Detections`**（[types.py:19](person_pipeline/types.py#L19)）：YOLO 的偵測結果
  - `xyxy`：形狀 `(N, 4)`，N 個框，每個框是 `[左上x, 左上y, 右下x, 右下y]`
  - `conf`：形狀 `(N,)`，每個框的信心分數
  - **重點：三個陣列的第 i 列都是同一個人。** 這叫「列對齊」（row-aligned），整個專案都靠這個約定。

- **`FrameResult`**（[types.py:35](person_pipeline/types.py#L35)）：一張影像的所有結果
  - `track_ids[i]`、`person_ids[i]` 都對應 `detections` 的第 i 個框
  - `track_id = -1` 表示這個框沒被追蹤；`person_id = None` 表示還沒認出是誰

> **numpy 形狀小提醒：** `(N, 4)` 讀作「N 列、每列 4 個數字」。`xyxy[i]` 拿第 i 個框，`xyxy[:, 3]` 拿所有框的右下 y，`xyxy[rows]` 拿好幾個指定的框。

### 2.4 `pipeline.py` + `camera.py`：指揮官和工作台（最重要的兩個檔案）

先記住分工：

- **`PersonPipeline`（pipeline.py）**：只有一個。擁有共用的 YOLO 和 ReID 模型，跑主迴圈。
- **`CameraStream`（camera.py）**：每台攝影機一個。擁有這台自己的影像來源、追蹤器、gallery、Cupola 推送、視窗。

**(a) `run()`：主迴圈**（[pipeline.py:43](person_pipeline/pipeline.py#L43)）

把雜事（統計、結束條件）先跳過，骨架其實只有這樣：

```python
for cam in self.cameras: cam.start()                 # 每台開始拿影像
while self._running:
    batch = [(cam, p) for cam in self.cameras if (p := cam.poll()) is not None]
    if not batch: sleep; continue                    # 還沒有新影像
    dets_list = self.detector.detect([p.image for _, p in batch])   # 一次 YOLO 推論所有影像
    for (cam, packet), dets in zip(batch, dets_list):
        cam.output(cam.process(packet, dets, ...))   # 各台各自追蹤、ReID、畫圖、推送
```

第一行的 `batch = [...]` 用到兩個語法：

- **list comprehension**：`[x for cam in ... if ...]` 等於「建一個空 list，跑迴圈，符合條件的就 append」。
- **`:=`（海象運算子）**：`(p := cam.poll())` 是「呼叫 `cam.poll()`，結果存進 `p`，同時拿這個結果來判斷 `is not None`」。不用它的話要寫成：

```python
batch = []
for cam in self.cameras:
    p = cam.poll()              # 這台有沒有新影像？沒有就馬上回傳 None，不會等
    if p is not None:
        batch.append((cam, p))
```

**讀長函式的技巧：先找出骨架，把統計、例外處理、if 分支暫時當成沒看到。** 骨架懂了再回頭看細節。

`try: ... finally:`（[pipeline.py:48](person_pipeline/pipeline.py#L48) 和 [pipeline.py:73](person_pipeline/pipeline.py#L73)）的意思是：不管迴圈怎麼結束（影片播完、按 q、Ctrl+C、出錯），`finally` 裡的收尾一定會執行。

**(b) `CameraStream.process()`：一張影像經過剩下的關卡**（[camera.py:95](person_pipeline/camera.py#L95)）

YOLO 已經在主迴圈做完了，這裡接著做第 0 節那張圖的後半段，每段都用 `t = time.perf_counter()` 計時：

```python
track_ids = self.tracker.update(dets, packet.image)       # 追蹤（這台自己的追蹤器）
rows      = self._select_for_reid(dets, track_ids)        # 挑哪些框要算特徵
feats     = self.reid.extract(packet.image, dets.xyxy[rows])   # 共用的 ReID 模型
mapping   = self.gallery.update(visible, embeddings, ...)  # track id → person id（這台自己的 gallery）
```

[camera.py:104](person_pipeline/camera.py#L104) 值得停下來看：

```python
embeddings = {int(track_ids[r]): f for r, f in zip(rows, feats)}
```

這是 **dict comprehension**，等於：

```python
embeddings = {}
for r, f in zip(rows, feats):       # zip 把兩個串列一對一配對
    embeddings[int(track_ids[r])] = f
```

結果是 `{track id: 特徵向量}`。

`_select_for_reid()`（[camera.py:121](person_pipeline/camera.py#L121)）決定「哪些框值得算 ReID」。ReID 在 CPU 上一個框約 9 ms，不能每個框每張都算，所以：
- 還沒認出身分的 track：一定算（不算就永遠認不出來）
- 已經認出的 track：每 5 張才算一次，而且框要夠大、分數夠高、沒跟別人重疊（避免特徵混到兩個人的外觀）

**(c) `CameraStream.output()`：結果交出去**（[camera.py:150](person_pipeline/camera.py#L150)）

交給 Cupola（`sink.publish`）、更新統計、畫圖後顯示在這台自己的視窗（`pipeline cam1`），需要的話也錄影。

> **讀完你應該知道：** 想改「每台攝影機的處理流程」（例如加一個關卡），改 `CameraStream.process()`；想改「多台怎麼排程」，改 `PersonPipeline.run()`。

### 2.5 各個關卡

每個關卡都是一個 class，結構都一樣：`__init__` 載入模型，另一個方法處理影像。讀的時候**只要抓住輸入和輸出**。

| 檔案 | 方法 | 輸入 | 輸出 |
|---|---|---|---|
| [detector.py:24](person_pipeline/detector.py#L24) | `detect(images)` | 影像的 list（每台攝影機一張） | `Detections` 的 list |
| [tracker.py:27](person_pipeline/tracker.py#L27) | `update(dets, image)` | `Detections` + 影像 | `(N,)` track id 陣列 |
| [reid.py:42](person_pipeline/reid.py#L42) | `extract(image, boxes)` | 影像 + `(K, 4)` 框 | `(K, 256)` 特徵 |
| [gallery.py:59](person_pipeline/gallery.py#L59) | `update(track_ids, embeddings, now, boxes)` | 看得到的 track + 新特徵 | `{track id: person id}` |

**`detector.py`**：最簡單的一個。呼叫 Ultralytics 的 `model.predict()`，`classes=[0]` 只要「人」這個類別，然後把結果轉成我們自己的 `Detections`。一次傳多張影像進去，GPU 會一起算（batch），比一張一張算快。

**`tracker.py`**：重點在 [tracker.py:36-41](person_pipeline/tracker.py#L36-L41)。TrackTrack 回傳的是 `(M, 8)` 陣列，第 8 欄 `idx` 告訴你「這個 track 對應到輸入的第幾個框」。這段迴圈把它轉回「每個框一個 track id」，維持列對齊：

```python
ids = np.full(len(dets), -1)      # 先全部填 -1（沒被追蹤）
for row in out:
    ids[int(row[7])] = int(row[4])  # 第 idx 個框的 track id = row[4]
```

**`reid.py`**：`_preprocess()`（[reid.py:32](person_pipeline/reid.py#L32)）把框切下來，然後做模型要求的前處理：BGR→RGB、縮放成 128×256、正規化、`HWC→CHW`（把顏色通道從最後一維移到最前面，因為模型要這種排列）。`extract()` 最後一行把每個特徵除以自己的長度（L2 正規化），這樣之後兩個特徵做內積就直接是 **cosine 相似度**（1 = 完全一樣，0 = 無關）。

**`gallery.py`**：最有「演算法」味道的檔案，建議先讀檔案最上面的說明（[gallery.py:1-14](person_pipeline/gallery.py#L1-L14)），再讀 `update()`。`update()` 分兩步：

1. [gallery.py:81](person_pipeline/gallery.py#L81)：已經有身分的 track，把新特徵加進那個人的特徵庫（`deque(maxlen=20)`：滿了自動丟掉最舊的）
2. [gallery.py:95](person_pipeline/gallery.py#L95)：新 track 拿去跟所有人比相似度，由高到低配對；最高的相似度 ≥ 0.65 就沿用舊的 person id，否則開一個新人

「為什麼不能配給正在畫面上的人」和「那個例外」：一個人不可能同時出現在兩個地方，所以正被別的 track 佔用的人不能配。例外是兩個框幾乎疊在一起時（例如隔板擋住腳時的半身框，和新出現的全身框），它們其實是同一個人。

### 2.6 `capture.py`：拿影像（多執行緒）

這個檔案用到 **執行緒（thread）**，第一次讀會覺得難，可以最後再看。

核心想法只有一句（[capture.py:1-5](person_pipeline/capture.py#L1-L5)）：**背景執行緒一直解碼，但只保留最新的一張。** 主迴圈來拿的時候，永遠拿到最新的，舊的直接丟掉。這樣 YOLO 慢的時候，延遲不會越積越多。

看兩個方法就好：

- `_publish()`（[capture.py:67](person_pipeline/capture.py#L67)）：背景執行緒呼叫。`self._latest = packet` 直接**覆蓋**舊的，然後 `notify_all()` 叫醒等待的人。
- `read()`（[capture.py:41](person_pipeline/capture.py#L41)）：主迴圈呼叫。`wait_for(...)` 會一直等，直到「有一張沒拿過的新影像」才回傳。

`threading.Condition` 你可以把它想成「一個鎖 + 一個門鈴」：`with self._cond:` 是上鎖（避免兩個執行緒同時改 `_latest`），`notify_all()` 是按門鈴，`wait_for()` 是等門鈴響。

`FrameSource(ABC)` 和 `@abstractmethod`（[capture.py:20](person_pipeline/capture.py#L20)、[capture.py:79](person_pipeline/capture.py#L79)）的意思是：「`FrameSource` 只是一個樣板，子類別**必須**自己寫 `_run()`。」`OpenCVSource` 和 `GStreamerSource` 各自寫了一份。`create_frame_source()`（[capture.py:186](person_pipeline/capture.py#L186)）依照 config 的 `backend` 決定建立哪一個。

### 2.7 輸出端

- **[visualizer.py:32](person_pipeline/visualizer.py#L32) `draw()`**：用 `cv2.rectangle` 和 `cv2.putText` 畫框。注意第一行 `img = result.packet.image.copy()`：先複製再畫，不會弄髒原始影像。
- **`sinks.py`**：只定義一個樣板 `ResultSink`，規定每個輸出端要有 `start / publish / stop`。
- **`cupola.py`**：也是背景執行緒。`publish()`（[cupola.py:185](person_pipeline/cupola.py#L185)）只是把結果存起來，馬上返回；真正的網路傳送在 `_run()`（[cupola.py:217](person_pipeline/cupola.py#L217)）每 `send_interval_s` 秒做一次。每台攝影機有自己的一個 `CupolaEventSink`，推到自己的 `cupola_camera_url`。這樣網路慢的時候不會拖慢 YOLO。

---

## 3. 讀程式的通用技巧

1. **先看檔案最上面的 docstring（三個引號包起來的說明）。** 這個專案每個檔案開頭都寫了「這個模組做什麼、為什麼這樣設計」。
2. **先看函式簽名（輸入、輸出），再看內容。** 例如 `def extract(self, image, boxes) -> np.ndarray`，`->` 後面就是回傳型別。
3. **長函式先找骨架。** 跳過 log、計時、錯誤處理，找出真正在處理資料的那幾行。
4. **用 VS Code 的「跳到定義」（F12）。** 看到 `self.gallery.update(...)` 不知道做什麼，游標放上去按 F12 直接跳過去。
5. **看不懂一行的時候，印出來。** 見下一節。

---

## 4. 動手驗證：最有效的學習方式

光讀不夠，**改一下、跑一下、看輸出**最快懂。以下都用本機影片測，不用連攝影機：

```bash
cd /home/ubuntu/Documents/Github/ultralytics_yolo_test/integrate
../pvenv/bin/python main.py --source ../videos/test1.mp4 --no-cupola
```

`--no-cupola` 讓測試影片的框不要推到 Cupola 的真實攝影機畫面上。

**練習 1：看資料的形狀。** 在 `camera.py` 的 `process()` 一開頭（`timings = ...` 下一行）加：

```python
print(self.name, len(dets), dets.xyxy.shape, dets.conf[:3])
```

你會看到每張影像有幾個人、框的陣列形狀、前三個分數。

**練習 2：看 track id 和 person id 的關係。** 在 `camera.py` 的 `process()` 最後 `return` 之前加：

```python
print(self._frame_idx, list(zip(track_ids.tolist(), person_ids)))
```

觀察有人被隔板擋住時，track id 換了，但 person id 有沒有維持一樣。

**練習 3：改參數看效果。** 把 `config.yaml` 的 `gallery.match_threshold` 從 0.65 改成 0.9，再跑一次。門檻變嚴格，同一個人會比較常被當成新人（person id 一直增加）。

**練習 4：兩台攝影機。** 給兩支影片，會開兩個視窗，統計也是每台一行：

```bash
../pvenv/bin/python main.py --source ../videos/51_203_output20261106_095222.mp4 ../videos/51_205_output20261106_095222.mp4 --no-cupola
```

兩個視窗的 person ID 都從 1 開始，因為每台攝影機有自己的 gallery。

**練習 5：用除錯器一步一步走。** 在 VS Code 的 [camera.py:98](person_pipeline/camera.py#L98) 行號左邊點一下設中斷點，按 F5 用 debugger 執行 `main.py`。停下來後，滑鼠移到變數上就能看到它的值，按 F10 執行下一行。

做完記得把加的 `print` 刪掉，`print` 每張影像都印會拖慢速度。

---

## 5. 常見名詞對照

| 名詞 | 意思 |
|---|---|
| bbox / 框 | 圍住一個人的長方形，`xyxy` 格式 |
| conf / score | YOLO 對這個框有多確定是人（0–1） |
| track id | Tracker 給的短期編號，人被擋住太久或離開再回來會換新的 |
| person id | Gallery 給的長期編號，靠外觀特徵認回同一個人 |
| embedding / 特徵 | ReID 模型把一張人的圖變成的 256 個數字，外觀越像的人，數字越接近 |
| cosine 相似度 | 兩個特徵有多像，1 = 一樣，0 = 無關 |
| gallery / 特徵庫 | 每個人最多存 20 個特徵，用來跟新出現的 track 比對 |
| latency / 延遲 | 從影像解碼完成，到結果算出來的時間 |
| sink | 「結果的出口」，例如 Cupola 推送 |
| batch | 一次把多張影像一起送進模型推論 |
