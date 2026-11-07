# Deep Learning Project #2 — CNN for MNIST and CIFAR-100

> 專案更新日期：2026-11-07  
> 目前狀態：**MNIST CNN 已完成訓練與測試；CIFAR-100 Baseline 與 V2 均已完成訓練、驗證與比較；V3 Residual CNN 尚未實作。**

## 1. 專案目標

本作業使用 **PyTorch** 自行設計與訓練卷積神經網路（CNN），不使用預訓練權重或直接套用現成的完整模型實作。

1. **MNIST**：訓練 CNN，與 Project #1 自行實作的 BPNN 比較辨識表現及架構差異。
2. **CIFAR-100**：設計 CNN Baseline，根據實驗結果改良架構與訓練策略。
3. **助教測試流程（待完成）**：支援載入新訓練／測試資料、重新訓練、批次推論及輸出指定格式的 TXT。
4. **實驗分析與簡報（待完成）**：整理模型架構、超參數、Learning Curves、比較結果與限制。

## 2. 開發環境

- 作業系統：Windows（VS Code / PowerShell）
- Conda 環境：`dl_project1`
- Python：3.11.17
- PyTorch：2.14.1+cu130
- CUDA：13.0
- GPU：NVIDIA GeForce RTX 5060 Laptop GPU
- 主要套件：`torch`、`torchvision`、`numpy`、`Pillow`、`tqdm`

已驗證 PyTorch 可以使用 CUDA GPU 執行運算。

## 3. 專案目錄

```text
project2/
├── README.md
├── data/
│   ├── cifar-100-python/
│   │   ├── train
│   │   ├── test
│   │   └── meta
│   └── MNIST/                    # torchvision 下載的 MNIST 資料
├── src/
│   ├── data_loader.py           # MNIST / CIFAR-100 資料載入與分割
│   ├── model.py                 # MNISTCNN / CIFAR100CNN / CIFAR100CNNV2
│   ├── train.py                 # MNIST 訓練
│   ├── evaluate.py              # MNIST 測試
│   ├── train_cifar100.py        # CIFAR-100 Baseline 訓練
│   ├── train_cifar100_v2.py     # CIFAR-100 V2 訓練
│   └── evaluate_cifar100_v2.py  # V2 Train / Val Evaluation Mode 評估
└── outputs/
    ├── models/
    │   ├── best_mnist_cnn.pth
    │   ├── best_cifar100_baseline.pth
│   └── best_cifar100_v2.pth
    └── results/
        ├── mnist_training_history.csv
        ├── cifar100_baseline_history.csv
        └── cifar100_v2_history.csv
```

> 上述為已使用／預期保留的檔案結構；實際存在的檔案以本機為準。Baseline 與 V2 使用不同的訓練腳本、權重檔名及 CSV，避免互相覆蓋。

## 4. 已完成工作

### 4.1 MNIST — CNN 與 BPNN 比較

**資料與前處理**

- MNIST：28 × 28 灰階影像，10 類別。
- Training / Validation / Test：55,000 / 5,000 / 10,000。
- 使用 `transforms.ToTensor()` 將像素轉成 `[0, 1]`。
- Training/Validation 分割使用固定隨機種子 `42`。

**CNN 架構**

```text
Input: 1 × 28 × 28
  → Conv(1→32, 3×3) → ReLU → MaxPool(2)
  → Conv(32→64, 3×3) → ReLU → MaxPool(2)
  → Flatten (64×7×7)
  → Linear(3136→128) → ReLU
  → Linear(128→10)
```

- Trainable Parameters：**421,642**。
- Optimizer：Adam；Learning Rate：0.001。
- Batch Size：64；Epochs：10。
- Loss：CrossEntropyLoss。
- 使用最佳 Validation Accuracy 儲存模型權重。

**實驗結果**

| 指標 | 結果 |
|---|---:|
| Best Validation Accuracy | **98.84%** |
| Best Epoch（首次達到） | 6 |
| Test Accuracy | **98.79%** |
| Test Loss | 0.0360 |
| 正確分類 | 9,879 / 10,000 |
| Project #1 BPNN Test Accuracy | 97.77% |
| CNN 相對 BPNN 的準確率差異 | +1.02 個百分點 |

**解讀**：本次實驗中 CNN 的 Test Accuracy 高於 BPNN；但兩次實驗的 Optimizer 等條件不完全相同，不能把全部改善歸因於 CNN 架構。

### 4.2 CIFAR-100 — Baseline CNN

**資料與前處理**

- 使用本機 `data/cifar-100-python/train`、`test`、`meta`（Python Pickle 格式）。
- 自訂 `LocalCIFAR100` 讀取資料，使用 `fine_labels`（100 類別）。
- 影像：32 × 32 RGB；Tensor Shape：`[batch, 3, 32, 32]`。
- Training / Validation / Test：45,000 / 5,000 / 10,000。
- Training/Validation 分割使用固定種子 `42`。
- `ToTensor()` + CIFAR-100 RGB Mean/Std Normalization。
- **Baseline 未使用 Data Augmentation。**

**Baseline CNN 架構**

```text
Input: 3 × 32 × 32
  → [Conv(3→32)   → ReLU → Conv(32→32)   → ReLU → MaxPool]
  → [Conv(32→64)  → ReLU → Conv(64→64)   → ReLU → MaxPool]
  → [Conv(64→128) → ReLU → Conv(128→128) → ReLU → MaxPool]
  → Flatten (128×4×4 = 2048)
  → Linear(2048→256) → ReLU
  → Linear(256→100)
```

- 所有 Conv：3×3、padding=1；MaxPool：2×2。
- Trainable Parameters：**837,252**。
- Optimizer：Adam；Learning Rate：0.001。
- Batch Size：128；Epochs：30。
- Loss：CrossEntropyLoss。
- Best Checkpoint：依最高 Validation Accuracy 儲存。
- 已加入 `tqdm` 進度條與每個 Epoch 的耗時紀錄。

**Baseline 實驗結果**（根據 `cifar100_baseline_history.csv`）

| 指標 | 結果 |
|---|---:|
| Best Validation Accuracy | **38.52%** |
| Best Epoch | **8** |
| 最低 Validation Loss | **2.5009**（Epoch 7） |
| Epoch 30 Training Accuracy | **92.68%** |
| Epoch 30 Validation Accuracy | **33.36%** |
| Epoch 30 Validation Loss | **6.9923** |
| 30 Epoch 紀錄耗時合計 | 約 **203.1 秒** |
| CIFAR-100 Test Accuracy | **尚未測試** |

**主要發現**

- Training Accuracy 持續上升，Epoch 30 已達 92.68%。
- Validation Accuracy 在 Epoch 8 達到 38.52% 後未再突破，最後下降至 33.36%。
- Validation Loss 在後期明顯上升，顯示**嚴重過度擬合（Overfitting）**。
- 因此下一步優先改善泛化能力，而非單純增加訓練 Epochs。
- 目前的 38.52% 是 **Validation Accuracy，不是 Test Accuracy**。

### 4.3 CIFAR-100 — V2 CNN（已完成）

**改良目的**：改善 Baseline 在訓練後期 Training Accuracy 持續上升、Validation Accuracy 卻下降的嚴重 Overfitting。

**資料與前處理**

- 使用與 Baseline **相同的 45,000 / 5,000 / 10,000 分割**，Seed = 42。
- Training：`RandomCrop(32, padding=4)`、`RandomHorizontalFlip(p=0.5)`、`ToTensor()`、`Normalize(mean, std)`。
- Validation / Test：只有 `ToTensor()`、`Normalize(mean, std)`，**不做隨機增強**。
- `get_cifar100_loaders(batch_size=128, augment=True)`；Training 與 Validation 分別使用不同 transform 的 Dataset，再套用不重疊的索引。

**V2 模型架構**（自行實作）

```text
Input: 3 × 32 × 32
  → Conv(3→64)   → BN → ReLU → Conv(64→64)   → BN → ReLU → MaxPool → Dropout2d(0.1)
  → Conv(64→128) → BN → ReLU → Conv(128→128) → BN → ReLU → MaxPool → Dropout2d(0.2)
  → Conv(128→256)→ BN → ReLU → Conv(256→256) → BN → ReLU → MaxPool → Dropout2d(0.3)
  → AdaptiveAvgPool2d(1) → Flatten → Dropout(0.3) → Linear(256→100)
```

- 所有 Conv：3×3、padding=1；MaxPool：2×2。
- **Trainable Parameters：1,172,900**（已用 `model.py` Forward Test 驗證）。
- Forward Test：`[128, 3, 32, 32] → [128, 100]`。
- Optimizer：AdamW，Learning Rate = 0.001，Weight Decay = 0.0001。
- Scheduler：`CosineAnnealingLR(T_max=50, eta_min=1e-5)`，每個 Epoch 更新一次。
- Loss：CrossEntropyLoss；Batch Size：128；Epochs：50；Seed：42。
- 每個 Epoch 更新 `cifar100_v2_history.csv`，並依最高 Validation Accuracy 儲存 `best_cifar100_v2.pth`。

**V2 訓練結果**（根據 `cifar100_v2_history.csv`）

| 指標 | 結果 |
|---|---:|
| Best Validation Accuracy | **63.62%** |
| Best Epoch | **46 / 50** |
| Epoch 50 Training Accuracy（訓練模式，含增強） | **61.82%** |
| Epoch 50 Validation Accuracy | **63.56%** |
| Epoch 50 Validation Loss | **1.2845** |
| CIFAR-100 Test Accuracy | **尚未測試** |

**Best V2 Checkpoint：Evaluation Mode 補充測試**（執行 `evaluate_cifar100_v2.py`）

| 指標 | Training Set | Validation Set |
|---|---:|---:|
| Loss | 0.8503 | 1.2873 |
| Accuracy | **74.83%** | **63.62%** |

- Train–Validation Accuracy Gap：**11.21 percentage points (pp)**。
- 此次測試的 Training 與 Validation 都**不使用 Data Augmentation**，且模型設定 `eval()`（關閉 Dropout，BatchNorm 使用 Running Statistics）。
- 訓練日誌中的 61.82% 與 Evaluation Mode 的 74.83% **不是相同評估條件**，不能直接比較並斷言 Underfitting。
- V2 仍有泛化差距，但相較 Baseline，Validation Accuracy 明顯改善，且訓練後期沒有同樣程度的驗證準確率崩落。

### 4.4 Baseline 與 V2 比較

| 指標 | Baseline | V2 |
|---|---:|---:|
| Trainable Parameters | 837,252 | 1,172,900 |
| Epochs | 30 | 50 |
| Best Validation Accuracy | **38.52%** | **63.62%** |
| Best Epoch | 8 | 46 |
| Final Training Accuracy（訓練模式） | 92.68% | 61.82% |
| Final Validation Accuracy | 33.36% | 63.56% |
| Final Validation Loss | 6.9923 | 1.2845 |

- **最佳 Validation Accuracy 提升 25.10 pp**（38.52% → 63.62%）。
- 兩版訓練設定、正則化與模型架構都不同，因此目前只能將提升歸因於 **V2 整體改良方案**，不能單獨歸因於 BatchNorm、Dropout 或 Data Augmentation。
- 兩版最終 Training Accuracy 的測量條件不同；若要公平比較泛化差距，應在兩版最佳 Checkpoint 上使用相同的 Evaluation Mode、無增強資料進行評估。

## 5. 執行方式

先啟用 Conda 環境，進入 `src/`：

```powershell
conda activate dl_project1
cd C:\Data\code\hw\deep_learning_HW\project2\src
```

**資料載入／模型形狀檢查**

```powershell
python .\data_loader.py
python .\model.py
```

**MNIST 訓練與測試**

```powershell
python .\train.py
python .\evaluate.py
```

**CIFAR-100 Baseline 訓練**

```powershell
python .\train_cifar100.py
```

**CIFAR-100 V2 訓練與補充評估**

```powershell
python .\train_cifar100_v2.py
python .\evaluate_cifar100_v2.py
```

> 重新執行同一版本的訓練程式會覆寫該版本同名權重與 CSV；要保留本次實驗，請先備份。`evaluate_cifar100_v2.py` 不會重新訓練，只會載入已儲存的最佳 V2 權重進行評估。

## 6. 下一階段：CIFAR-100 V3（規劃中，尚未實作）

**優先方向：自行實作 Residual CNN**，不使用現成完整 ResNet 模型或預訓練權重。

預計流程：

1. 在 `model.py` 自行定義 `ResidualBlock`（含 Skip Connection），處理通道數或空間尺寸變化時的維度對齊。
2. 組合 `CIFAR100CNNV3`，規劃 64 / 128 / 256 Channels 的 Residual Stages。
3. 先驗證 Forward Output Shape、Trainable Parameters，再建立獨立的 V3 訓練程式及輸出檔名。
4. 使用相同 Training / Validation Split 比較 V2 與 V3，觀察泛化表現與訓練成本。
5. 視實驗結果再測試正則化強度或 Ablation Study，避免一次修改太多因素。

**研究判斷**：V2 Evaluation Mode 的 Training Accuracy 為 74.83%，Validation Accuracy 為 63.62%，Gap 為 11.21 pp。顯示仍有一定泛化差距；尚不能確定是模型容量、最佳化或正則化強度造成瓶頸。V3 的改良效果需由實驗驗證，不能預設一定提升。

**Test Set 原則**：目前 CIFAR-100 結果均為 Validation 評估；先以 Validation 選定模型，最後才使用 Test Set 做獨立評估，避免反覆用 Test Accuracy 調整模型。

## 7. 實驗檔案保留策略

| 檔案 | 建議 | 原因 |
|---|---|---|
| `mnist_training_history.csv` | 保留 | MNIST Learning Curves 與比較依據 |
| `best_mnist_cnn.pth` | 保留 | 已完成測試的 MNIST 最佳模型 |
| `cifar100_baseline_history.csv` | **務必保留** | CIFAR-100 Baseline 完整訓練紀錄 |
| `best_cifar100_baseline.pth` | 建議暫留 | 可用於 Baseline 推論、錯誤分析及重現比較 |
| `cifar100_v2_history.csv` | **務必保留** | V2 50 Epochs 的 Loss、Accuracy、LR 與時間紀錄 |
| `best_cifar100_v2.pth` | **保留** | 已驗證最佳模型，供 Evaluation Mode、錯誤分析與後續比較 |

**重要**：CSV 只有每個 Epoch 的整體指標，**無法還原模型權重**。如果刪除 `.pth`，將無法直接對該次訓練所得的 Baseline 模型做推論，除非重新訓練。若已確定只需要 Learning Curves、已記錄所有比較指標，且不需要再做 Baseline 推論，可以刪除 Baseline `.pth` 節省空間。

## 8. 待辦事項

- [x] MNIST Data Loader
- [x] MNIST CNN Forward Test
- [x] MNIST CNN Training / Validation
- [x] MNIST Test Evaluation
- [x] MNIST CNN vs. BPNN 初步比較
- [x] CIFAR-100 Local Data Loader
- [x] CIFAR-100 Baseline CNN Forward Test
- [x] CIFAR-100 Baseline Training / Validation
- [x] CIFAR-100 Baseline Overfitting 分析
- [x] CIFAR-100 V2 Data Augmentation / Model / Training
- [x] CIFAR-100 Baseline vs. V2 數值比較
- [x] V2 最佳 Checkpoint：Training / Validation Evaluation Mode 測試
- [ ] Baseline / V2 Learning Curves 圖表整理
- [ ] CIFAR-100 V3 ResidualBlock / Model / Training
- [ ] CIFAR-100 最終 Test Evaluation
- [ ] 助教新資料集訓練與自動批次預測
- [ ] 指定格式 TXT 輸出
- [ ] 英文簡報與實驗報告

---

**目前階段結論**：MNIST CNN 已完成；CIFAR-100 V2 將最佳 Validation Accuracy 從 38.52% 提升至 63.62%（+25.10 pp），顯著改善 Baseline 的嚴重 Overfitting。V2 的無增強 Evaluation Mode 顯示 11.21 pp 的 Train–Validation Gap。下一步將在保留 Baseline、V2 完整實驗紀錄的前提下，自行設計並驗證 V3 Residual CNN。
