# Cross-class NMS & COCOeval Fix

> **變更摘要**：修正 Mask R-CNN inference 的跨類別重複偵測問題，並修正訓練期間 COCOeval 低估 AP50 的兩個根本原因。

---

## 問題一：跨類別重複偵測

### 根本原因

Mask R-CNN 在 `postprocess_detections`（`roi_heads.py`）做的是 **per-class NMS**——每個類別各自對自己做，不同類別之間的重疊完全不管。
結果：同一個細胞可以同時出現在 predictions 裡以不同類別（例如 class1 score=0.8、class2 score=0.6）。

### 影響

- **class1 AP**：高分的 TP 配對成功，沒問題
- **class2 AP**：那筆 class2 預測找不到 GT 配對 → 計入 FP → precision 下降 → AP 降低

### 修正：`cross_class_nms()` in `src/utils.py`

```python
def cross_class_nms(pred: dict, iou_threshold: float = 0.5) -> dict:
    boxes = pred["boxes"]
    if len(boxes) == 0:
        return pred
    keep = _box_nms(boxes, pred["scores"], iou_threshold)
    return {k: v[keep] for k, v in pred.items()}
```

`torchvision.ops.nms` 是 class-agnostic 的——用同一組 box 再做一次，忽略類別標籤，保留 score 最高的。IoU threshold 設 0.5 對應 AP50 的評估標準。

套用位置：`src/inference.py`（提交前）與 `src/train.py evaluate()`（訓練期間驗證）。

---

## 問題二：COCOeval `maxDets=100` 壓制 recall

### 根本原因

COCOeval 預設 `params.maxDets = [1, 10, 100]`，AP 使用 maxDets=100。
本資料集單張圖最多 772 個 instance，top-100 的 recall 上限只有 ~13%，
PR 曲線被強制截斷，AP 大幅低估。

### 修正

```python
evaluator.params.maxDets = [1, 10, 1500]
```

設 1500 覆蓋所有可能的 instance 數量（訓練集最大值 ~772，加上 margin）。

---

## 問題三：`score_thresh` 在 COCOeval 前過濾

### 根本原因

原本 `evaluate()` 用 `score_thresh=0.05` 把低信心預測丟掉再傳給 COCOeval。
COCOeval 內部本來就會掃描所有 score threshold 來畫 PR 曲線——提前過濾等於主動砍掉曲線的高 recall 部份，直接降低 AP。

### 修正

移除 `evaluate()` 的 `score_thresh` 參數，讓所有預測進入 COCOeval。

---

## AP50 提升（epoch 1 對比）

| 版本 | Epoch 1 AP50 |
|------|-------------|
| 修正前（`score_thresh=0.05`，`maxDets=100`，無 NMS） | 0.1748 |
| 修正後（無過濾，`maxDets=1500`，cross-class NMS）     | 0.1941 |
