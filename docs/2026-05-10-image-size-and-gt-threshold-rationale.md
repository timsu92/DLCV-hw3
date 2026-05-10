# 設計決策說明：影像大小、Anchor、訓練過濾閾值

## 1. 訓練影像解析度：`min_size=(640,768,896,1024)`、`max_size=1024`

### 為什麼要提高解析度？

訓練集影像幾乎都是方形（約 1771×1760 px），但測試集有極端長寬比（最大約 9.4×）及更廣的尺寸分布。提高訓練解析度有兩個目的：

1. **保留小物件特徵**：class2 的實例中有 **39.6%** 的 sqrt(area) < 16 px（即面積小於 256 像素²）。在低解析度下縮放後，這些細胞可能只剩 1–2 個像素，難以學習到有效特徵。
2. **模擬測試集多尺度分布**：Multi-scale training（短邊在 640–1024 之間隨機取樣）讓模型見過不同縮放比例，提升對未知尺寸的泛化。

### 為什麼 `max_size=1024`（transforms 用 `1025`）？

torchvision v2 的 `v2.Resize` 有嚴格限制：`max_size` 必須**嚴格大於** `size`，否則丟出 `ValueError`。因此 transforms 設為 `max_size=1025`，而 MaskRCNN 模型的 `GeneralizedRCNNTransform` 沒有此限制，設為 `max_size=1024`。

### VRAM 評估

在 1024×1024 下，訓練時 peak memory 約 **13.87 GiB**（單 GPU smoke test 實測），距離 15 GiB 上限還有 ~1.1 GiB 安全餘量（配合 `expandable_segments:True` 和 gradient checkpointing）。

---

## 2. Anchor 大小：`((4,8), (16,32), (32,64), (64,128), (128,256))`

### 資料集 instance 尺寸分布

以下是全資料集（train + val）31,407 個 instance 的 sqrt(area) 分布：

| sqrt(area) 範圍 | 實例數 | 佔比   |
|---------------|-------|--------|
| [0, 8)        |    11 |  0.04% |
| [8, 16)       | 6,438 | 20.50% |
| [16, 32)      |21,857 | 69.59% |
| [32, 64)      | 2,985 |  9.50% |
| [64, 128)     |   113 |  0.36% |
| [128, 256)    |     3 |  0.01% |

**超過 20% 的 instance 落在 [8, 16) 區間**，主要來自 class2（小型細胞，median sqrt-area = 16.6 px）。

### 為什麼要加入 size=4 的 anchor？

torchvision 預設的最小 anchor 是 8 px（對應 FPN P2 level，stride=4）。加入 `size=4` 可讓 RPN 直接在 P2 上對最小的細胞產生 proposal，不需完全依賴框回歸來「拉大」8 px anchor 去匹配 5–7 px 的目標。

### 為什麼維持每個 FPN level 6 個 anchor？

RPNHead 的 `num_anchors_per_location()` 必須在所有 level 相同（單一 conv 層共用）。原始設計是每 level 9 個（3 sizes × 3 ratios）；改用 2 sizes × 3 ratios = **6 個**，保持 uniform。

代價：捨棄了原本的 256 px 和 512 px anchor，但資料集中 sqrt(area) > 128 的 instance 僅佔 **0.37%**，影響極小，且這些大目標由框回歸從 128 px anchor 補足即可。

### 各 class 的 tiny instance 比例

| class | 中位數 sqrt(area) | tiny (<16 px) 佔比 |
|-------|-----------------|-------------------|
| class1 | 26.3 px | 1.6% |
| class2 | 16.6 px | **39.6%** |
| class3 | 23.7 px | 1.1% |
| class4 | 45.2 px | 0.0% |

class2 是加入小 anchor 最主要的受益者。

---

## 3. Dense Image 過濾閾值：`skip_above_instances=400`

### 為什麼要過濾？

Mask R-CNN 的 RPN 在計算 anchor-GT IoU 矩陣時，記憶體需求為：

```
n_gt × n_anchors × 4 bytes
```

在 1024×1024 解析度下（FPN stride 4/8/16/32/64），五個 level 合計 **523,776 個 anchor**：

| GT 數量 | IoU 矩陣大小 |
|--------|------------|
| 100    | 0.20 GiB   |
| 200    | 0.39 GiB   |
| 400    | 0.78 GiB   |
| 600    | 1.17 GiB   |
| 772    | 1.51 GiB   |

在正向傳播的 activation（約 10–12 GiB）基礎上，GT=600–772 的圖像在反向傳播時容易 OOM。

### 為什麼不是「每張圖隨機取樣 GT」？

隨機刪去 GT 標注會讓模型把**有細胞的位置**視為背景，造成 false-negative anchor supervision——模型學到錯誤的負樣本分布，比完全跳過這張圖更有害。

### 為什麼選 400 而不是其他值？

核心考量：保護**稀有類別（class3、class4）**的訓練樣本，同時控制 OOM 風險。

#### 訓練集（178 張）各閾值比較

| 閾值 | 跳過圖數 | class3 保留率 | class4 保留率 |
|-----|---------|-------------|-------------|
| 300 | 36 張   | 93%         | 98%         |
| 350 | 25 張   | 96%         | 99%         |
| **400** | **15 張** | **97%** | **100%**  |
| 450 | 11 張   | 97%         | 100%        |
| 500 | 10 張   | 97%         | 100%        |
| 600 |  5 張   | 100%        | 100%        |

閾值從 400 提高到 500 只少跳過 5 張圖，但 class3/4 的保留率完全不變（97%/100%）。閾值從 400 降到 300 則需多跳過 21 張圖，class3 保留率從 97% 降至 93%。

**400 是 class3/class4 保留率曲線的拐點**：再往下降，稀有類別損失明顯加速；往上調，OOM 風險增加但幾乎沒有稀有類別增益。

#### 被跳過的 15 張圖，各 class 的 annotation 損失

| class | 訓練集總量 | 保留   | 損失   | 損失率 |
|-------|---------|--------|--------|------|
| class1 | 11,574 | 8,448  | 3,126  | 27.0% |
| class2 | 14,637 | 9,305  | 5,332  | 36.4% |
| class3 |    538 |   522  |    16  |  3.0% |
| class4 |    550 |   549  |     1  |  0.2% |

class1 和 class2 損失較多，但這 15 張都是**高密度圖**（423–772 個 instance），每張保留的密度資訊遠超一般圖像——跳過後，class1 仍有 8,448 個樣本、class2 仍有 9,305 個，訓練資料充足。

被跳過的 15 張圖的 instance 分布：

| 範圍      | 張數 |
|----------|-----|
| 400–499  |  5  |
| 500–599  |  5  |
| 600–699  |  1  |
| 700–800  |  4  |

---

## 4. 推論時 `pre_resize_image` 的對齊

推論與訓練 val eval 使用**相同**的前處理：

- 短邊縮至 1024 px，長邊上限 1025 px（v2 constraint）
- 轉為 FP32 且除以 255（scale=True）
- 使用 torchvision tensor path + `antialias=True`（非 PIL resize）

之所以特別對齊，是因為之前的版本推論用 PIL + autocast，導致 val AP50=0.6997 但測試集只有 0.4958，排除管線差異後確認 gap 為真實 distribution shift（測試集有更極端的長寬比與尺寸分布）。
