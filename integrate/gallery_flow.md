# Gallery 流程圖（單一攝影機）

以「相簿」來比喻：gallery 是一本相簿，每個人（person）一頁，每頁最多 20 張照片（Re-ID 特徵）。

程式碼位置：[gallery.py](person_pipeline/gallery.py)、[camera.py](person_pipeline/camera.py) 的 `_select_for_reid()`。參數在 [config.yaml](config.yaml) 的 `reid` 和 `gallery` 段落。

> 圖是用 Mermaid 畫的。GitHub 會直接顯示；在 VS Code 預覽（Ctrl+Shift+V）要先安裝擴充套件「Markdown Preview Mermaid Support」。

---

## 摘要：一張圖看懂

```mermaid
flowchart TD
    A(["有人出現在畫面<br/>（或被遮擋後重新出現）"]) --> B["擷取外觀特徵"]
    B --> C{"跟已知人員的照片<br/>夠像嗎？"}
    C -- 是 --> D["沿用原本的 ID"]
    C -- 否 --> E["給新的 ID"]
    D --> F["持續追蹤<br/>定期補拍清楚的照片<br/>每人保留最新 20 張"]
    E --> F
    F -- "被遮擋 / 離開畫面" --> G{"10 分鐘內<br/>回來了嗎？"}
    G -- 是 --> A
    G -- 否 --> H(["刪除這個人的資料"])
```

**一句話：** 每個人出現時比對一次外觀，認得就沿用舊 ID，不認得就給新 ID；之後持續更新他的照片，所以被遮擋或離開再回來，仍能認回同一個 ID。

以下是各步驟的細節。

---

## 1. 總覽：每張畫面、每個 track 怎麼處理

```mermaid
flowchart TD
    A["YOLO + 追蹤器<br/>得到一個 track"] --> B{"這個 track<br/>有 person ID 了嗎？"}

    B -- 有 --> C["查表<br/>track → person<br/>直接用那個 ID"]
    C --> D{"距離上次補照片 ≥ 5 張<br/>而且框夠清楚？"}
    D -- 是 --> E["算特徵<br/>貼進那個人的頁面<br/>（見第 2 節）"]
    D -- 否 --> F["這張不補照片"]

    B -- 沒有 --> G{"這張有算到特徵嗎？<br/>（框太小就沒有）"}
    G -- 沒有 --> H["顯示灰色 T&lt;track id&gt; ?<br/>下一張再試"]
    G -- 有 --> I["跟相簿比對<br/>（見第 3 節）"]
    I --> J{"最高相似度 ≥ 0.65？"}
    J -- 是 --> K["沿用舊的 person ID"]
    J -- 否 --> L["開新的一頁<br/>新的 person ID"]
```

重點：**只有還沒有 person ID 的 track 才會比對。** 認出身分之後只會查表和補照片，不會再比對。

---

## 2. person 1 怎麼存到 20 張

### 2.1 什麼時候貼照片

```mermaid
flowchart TD
    S(["person 1 的 track 出現在這張畫面"]) --> A{"第一次出現？<br/>（剛建立 person 1）"}
    A -- 是 --> P1["貼第 1 張照片<br/>（比對時算的那張特徵）"]

    A -- 否 --> B{"距離上次補照片<br/>≥ 5 張畫面？<br/>update_interval"}
    B -- 否 --> X["不補照片"]
    B -- 是 --> C{"框高 ≥ 48 px？<br/>min_box_height"}
    C -- 否 --> X
    C -- 是 --> D{"YOLO 分數 ≥ 0.5？<br/>min_conf"}
    D -- 否 --> X
    D -- 是 --> E{"跟別人重疊 IoU ≤ 0.3？<br/>max_overlap_iou"}
    E -- 否 --> X
    E -- 是 --> F["算特徵"]
    F --> G["append 到 person 1 的照片隊伍"]
    P1 --> G
    G --> H{"隊伍已經有 20 張？<br/>max_embeddings_per_person"}
    H -- 否 --> I["照片數 +1"]
    H -- 是 --> J["自動丟掉最舊的 1 張<br/>（deque maxlen=20）<br/>照片數維持 20"]
```

三個「清楚」條件的原因：框太小特徵不準；分數低的框可能只框到一部分；跟別人重疊的照片會混到兩個人的外觀。

### 2.2 照片隊伍的變化

```mermaid
flowchart LR
    subgraph T1["剛建立"]
        direction LR
        a1["照1"]
    end
    subgraph T2["存了 20 張（滿了）"]
        direction LR
        b1["照1"] --- b2["照2"] --- b3["..."] --- b20["照20"]
    end
    subgraph T3["貼第 21 張"]
        direction LR
        c2["照2"] --- c3["..."] --- c20["照20"] --- c21["照21"]
    end
    T1 -- "每 5 張畫面<br/>補一張清楚的" --> T2
    T2 -- "照1 被丟掉" --> T3
```

- 每頁永遠保留**最近的 20 張**，丟掉的一定是最舊的，不管照片好不好。
- 攝影機 15 fps、每 5 張補一張的話，20 張大約是最近 7 秒左右的外觀。常被擋住或跟別人重疊時，補得比較慢，涵蓋的時間比較長。
- 人不在畫面上時不會補照片，所以離開時的 20 張會原封不動留著。

---

## 3. track ID 換掉時怎麼比對

### 3.1 例子：被隔板擋住，track 5 變成 track 9

```mermaid
sequenceDiagram
    participant T as 追蹤器
    participant P as pipeline<br/>(_select_for_reid)
    participant R as Re-ID 模型
    participant G as Gallery

    Note over T,G: 第 1–50 張：他一直是 track 5 → person 1，照片慢慢存到最多 20 張
    T->>P: 第 51 張：track 5 被隔板擋住，消失
    Note over G: 什麼都不做，person 1 的照片還在
    T->>P: 第 120 張：他出來了，追蹤器接不上，給新的 track 9
    P->>P: track 9 還沒有 person ID → 一定要算特徵
    P->>R: 切下 track 9 的框
    R-->>P: 256 維特徵
    P->>G: update(track 9 + 特徵)
    G->>G: 候選人：沒被其他 track 佔用的人（person 1 符合）
    G->>G: track 9 的特徵 vs person 1 的 20 張照片，取最高分 = 0.82
    G->>G: 0.82 ≥ 0.65 → 綁定 track 9 → person 1，這張特徵也貼進 person 1
    G-->>P: track 9 = person 1
    Note over T,G: 畫面上一直顯示 ID 1
    Note over G: 約 5 秒後：track 5 一直沒出現 → 解除 track 5 的綁定（person 1 還在）
```

### 3.2 比對的詳細步驟

```mermaid
flowchart TD
    A(["新的 track 9，有特徵"]) --> B["列出相簿裡的每一個人"]
    B --> C{"這個人正被畫面上<br/>其他 track 佔用嗎？"}
    C -- 沒有 --> OK["列為候選人"]
    C -- 有 --> D{"佔用的那個框<br/>大部分在 track 9 的框裡<br/>（或反過來）？"}
    D -- "是（同一個人被框兩次，<br/>例如隔板後殘留的半身框）" --> OK
    D -- 否 --> NO["排除<br/>（一個人不能同時在兩個地方）"]

    OK --> E["track 9 的特徵<br/>跟這個人的每張照片算 cosine 相似度<br/>取最高的一張"]
    E --> F["所有 (track, 人) 組合<br/>依分數由高到低排序"]
    F --> G{"最高分 ≥ 0.65？<br/>match_threshold"}
    G -- 是 --> H["綁定 track 9 → 那個人<br/>這張特徵貼進他的頁面"]
    G -- 否 --> I["開新的一頁<br/>track 9 → 新的 person ID"]
```

同一張畫面有好幾個新 track 時，從分數最高的組合開始配對，每個 track 和每個人都只能配一次，所以兩個新 track 不會搶到同一個人。

---

## 4. 什麼時候忘記

```mermaid
flowchart TD
    A(["每處理一張畫面<br/>都會檢查一次（_expire）"]) --> B{"某個 track<br/>超過 5 秒沒出現？<br/>track_timeout_s"}
    B -- 是 --> C["解除 track → person 的綁定<br/>那個人的頁面還在"]
    A --> D{"某個人<br/>超過 600 秒沒出現？<br/>identity_ttl_s"}
    D -- 是 --> E["整頁刪掉（20 張照片和 ID）<br/>他之後再出現會拿到新的 ID<br/>刪掉的 ID 不會重複使用"]
    F(["程式停止"]) --> G["整本相簿消失<br/>只存在記憶體，沒有寫到硬碟<br/>重開後 ID 從 1 重新編號"]
```

---

## 5. 參數對照

| 參數 | 預設值 | 在流程中的作用 |
|---|---|---|
| `reid.update_interval` | 5 | 已認出的人，每幾張畫面補一張照片 |
| `reid.min_box_height` | 48 | 補照片時，框至少要多高（px） |
| `reid.min_conf` | 0.5 | 補照片時，YOLO 分數至少多少 |
| `reid.max_overlap_iou` | 0.3 | 補照片時，跟別人重疊不能超過多少 |
| `gallery.max_embeddings_per_person` | 20 | 每頁最多幾張照片 |
| `gallery.match_threshold` | 0.65 | 新 track 要多像才沿用舊的 ID |
| `gallery.match_metric` | max | 跟每頁的照片比時，取最像的一張（`mean` = 跟平均值比） |
| `gallery.track_timeout_s` | 5 | track 消失多久後解除綁定 |
| `gallery.identity_ttl_s` | 600 | 人消失多久後整頁刪掉 |
