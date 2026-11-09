# Deep Learning Project #2 — CNN for MNIST and CIFAR-100

> 專案更新日期：2026-11-08  
> 目前狀態：**MNIST CNN 已完成訓練與測試；CIFAR-100 已完成 Baseline、V2、V3 系列與 V4 整合優化。最佳 Validation Accuracy 為 74.86%（V4 EMA）。CIFAR-100 獨立 Test Set 與助教測試流程尚未完成。**

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
│   └── MNIST/
├── src/
│   ├── data_loader.py             # MNIST / CIFAR-100 載入、分割、增強
│   ├── model.py                   # MNIST CNN、CIFAR-100 CNN 與 Residual CNN
│   ├── train.py                   # MNIST 訓練
│   ├── evaluate.py                # MNIST 測試
│   ├── train_cifar100.py          # CIFAR-100 Baseline
│   ├── train_cifar100_v2.py       # CIFAR-100 V2
│   ├── train_cifar100_v3_3.py     # V3.3：Mixup（其他 V3 腳本依本機為準）
│   ├── train_cifar100_v3_5.py     # V3.5：100 Epochs
│   ├── train_cifar100_v4.py       # V4：Label Smoothing + EMA + Warmup
│   └── evaluate_cifar100_v2.py   # 可調整 checkpoint 路徑評估不同版本
└── outputs/
    ├── models/
    │   ├── best_mnist_cnn.pth
    │   ├── best_cifar100_baseline.pth
    │   ├── best_cifar100_v2.pth
    │   ├── best_cifar100_v3_5.pth
    │   ├── best_cifar100_v4.pth
    │   └── best_cifar100_v4_ema.pth
    └── results/
        ├── mnist_training_history.csv
        ├── cifar100_baseline_history.csv
        ├── cifar100_v2_history.csv
        ├── cifar100_v3_5_history.csv
        └── cifar100_v4_history.csv
```

> 以上為實驗中使用或預期保留的代表性檔案；實際檔案與其他 V3 版本腳本以本機為準。各版本採用獨立權重與 CSV 檔名，以避免覆寫。

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

### 4.4 CIFAR-100 — V3 Residual CNN 系列（已完成）

**架構改良**：自行實作 `ResidualBlock`（兩層 3×3 Conv、BatchNorm、ReLU 與 Skip Connection），在輸入／輸出維度不同時使用 1×1 Conv Shortcut，不使用預訓練模型或直接載入現成完整 ResNet 實作。

```text
Input: 3 × 32 × 32
  → Conv(3→64) → BN → ReLU
  → Residual Stage 1: 64 channels, 2 blocks
  → Residual Stage 2: 128 channels, 2 blocks（第一個 block stride=2）
  → Residual Stage 3: 256 channels, 2 blocks（第一個 block stride=2）
  → AdaptiveAvgPool2d(1) → Flatten
  → Dropout(0.3)（V3.1 起）→ Linear(256→100)
```

- V3 架構約 **2,800,804 個參數**；初始 V3 無分類器 Dropout，V3.1 起加入 `Dropout(0.3)`。
- 共同設定：Batch Size 128、AdamW（LR 0.001、Weight Decay 1e-4）、Cosine Learning Rate、固定 Seed 42。
- V3.2 起加入 `RandomErasing(p=0.25)`；V3.3 使用 Mixup（α=0.2）；V3.4 改用 CutMix（α=1.0）；V3.5 恢復 Mixup 並將訓練延長至 100 Epochs。

| 版本 | 主要差異 | Epochs | Best Validation Accuracy |
|---|---|---:|---:|
| V3 | Residual CNN | 50 | 69.14% |
| V3.1 | + Dropout 0.3 | 50 | 68.92% |
| V3.2 | + Random Erasing | 50 | 69.12% |
| V3.3 | + Mixup α=0.2 | 50 | 70.64% |
| V3.4 | 改用 CutMix α=1.0 | 50 | 69.06% |
| V3.5 | 恢復 Mixup、訓練 100 Epochs | 100 | **72.66%**（Epoch 85） |

**V3.5 最佳 Checkpoint：Evaluation Mode（無隨機增強）**

| 指標 | Training Set | Validation Set |
|---|---:|---:|
| Loss | 0.0365 | 1.0640 |
| Accuracy | 99.82% | **72.66%** |

- Train–Validation Accuracy Gap：**27.16 pp**，仍有明顯 Overfitting。
- Mixup 訓練日誌的加權 Accuracy 與未增強的 Evaluation Mode Accuracy **不能直接相減作為泛化差距**。
- V3.4 的 CutMix 使 Gap 下降，但 Validation Accuracy 也下降；因此沒有選為後續最佳方案。

### 4.5 CIFAR-100 — V4 整合優化（已完成）

以 V3.5 為基礎，**保留相同 Residual CNN、Mixup α=0.2、RandomCrop、HorizontalFlip、Random Erasing**，並整合三項訓練策略：

1. **Label Smoothing（ε=0.05）**：訓練 Loss 使用 `CrossEntropyLoss(label_smoothing=0.05)`；Validation 使用一般 CrossEntropyLoss。
2. **EMA（decay 上限 0.999）**：每次 Optimizer Step 後更新 EMA 模型，並同步複製 BatchNorm Buffers；分別評估與儲存 Original / EMA 最佳權重。EMA 的衰減係數於訓練初期逐步增加。
3. **5-Epoch LR Warmup + Cosine Decay**：Warmup 由 0.0002 逐步增加至 0.001，再衰減至 1e-5。

**V4 訓練設定**

| 項目 | 設定 |
|---|---|
| Epochs / Batch Size | 100 / 128 |
| Optimizer | AdamW |
| Base Learning Rate / Weight Decay | 0.001 / 0.0001 |
| Scheduler | 5-Epoch Warmup + Cosine Decay（Min LR = 1e-5） |
| Data Augmentation | RandomCrop、HorizontalFlip、Random Erasing、Mixup α=0.2 |
| Training Loss | CrossEntropyLoss + Label Smoothing 0.05 |
| Validation Loss | CrossEntropyLoss（不使用 Label Smoothing） |
| EMA | 最大 decay 0.999，BatchNorm Buffer 複製 |
| Random Seed | 42 |

**最佳 Checkpoint：Evaluation Mode（未增強的 Training / Validation 資料）**

| 指標 | V3.5 | V4 Original | V4 EMA |
|---|---:|---:|---:|
| Train Loss | 0.0365 | 0.0725 | 0.0842 |
| Train Accuracy | 99.82% | 99.88% | 99.85% |
| Validation Loss | 1.0640 | **1.0218** | 1.0319 |
| **Validation Accuracy** | 72.66% | 74.78% | **74.86%** |
| Train–Validation Gap | 27.16 pp | 25.10 pp | **24.99 pp** |

- V4 EMA 的最佳 Validation Accuracy 為 **74.86%（Epoch 88）**；V4 Original 為 **74.78%（Epoch 89）**。
- 相對 V3.5，V4 EMA 的 Validation Accuracy 提升 **2.20 pp**，Gap 縮小 **2.17 pp**。
- EMA 與 Original 的 Accuracy 僅差 **0.08 pp**，不宜據此認定 EMA 有顯著優勢；Original 的 Validation Loss 反而略低。
- 因同時加入三項策略，目前**無法將 V4 的整體提升單獨歸因於 Label Smoothing、EMA 或 Warmup**。
- 即使最佳 Accuracy 提升，Training Accuracy 仍接近 100%，顯示 Overfitting 尚未消失。
- **CIFAR-100 獨立 Test Set 尚未測試；以上數字均非 Test Accuracy。**

### 4.6 CIFAR-100 各主要版本比較

| 模型 | Parameters（約） | Best Validation Accuracy | 相對 Baseline 提升 |
|---|---:|---:|---:|
| Baseline CNN | 837,252 | 38.52% | — |
| V2 CNN | 1,172,900 | 63.62% | +25.10 pp |
| V3 Residual CNN | 2,800,804 | 69.14% | +30.62 pp |
| V3.3 Mixup | 2,800,804 | 70.64% | +32.12 pp |
| V3.5 Mixup / 100 Epochs | 2,800,804 | 72.66% | +34.14 pp |
| **V4 EMA** | **2,800,804** | **74.86%** | **+36.34 pp** |

> 不同版本可能同時改變架構、增強與訓練設定，這是逐版實驗結果比較，不是嚴格控制所有變因的消融實驗。

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

**CIFAR-100 各主要版本訓練**

```powershell
python .\train_cifar100.py
python .\train_cifar100_v2.py
python .\train_cifar100_v3_5.py
python .\train_cifar100_v4.py
```

**CIFAR-100 最佳權重補充評估**

```powershell
python .\evaluate_cifar100_v2.py
```

> `evaluate_cifar100_v2.py` 是沿用的評估腳本名稱；評估 V4 時，需確認腳本中的模型類別為 `CIFAR100CNNV3`，且 checkpoint 路徑已設為 `outputs/models/best_cifar100_v4_ema.pth`。它會在 Evaluation Mode 下評估未增強的 Training / Validation 資料，不會重新訓練。重新執行訓練腳本可能覆寫同名 checkpoint 與 CSV，請先備份重要結果。

## 6. 下一階段規劃

1. **V5 資料增強實驗（規劃中，尚未實作）**：考慮以較保守的 RandAugment 取代 Random Erasing，保留 Mixup 與 V4 訓練策略；以 Validation Accuracy 判斷是否有效。
2. **CIFAR-100 獨立 Test Evaluation**：先完成模型選擇，再對獨立 Test Set 評估，避免使用 Test Accuracy 反覆調參。
3. **助教新資料流程**：確認新資料集格式，支援必要的重新訓練、讀取全部測試影像、自動預測與指定格式 TXT 輸出。
4. **英文簡報與實驗分析**：整理 Learning Curves、版本比較、Overfitting、EMA 與各策略的限制；若時間允許，補充 Ablation Study 或重複實驗。

## 7. 實驗檔案保留策略

| 檔案 | 建議 | 原因 |
|---|---|---|
| `mnist_training_history.csv` / `best_mnist_cnn.pth` | 保留 | MNIST 比較與測試結果 |
| `cifar100_baseline_history.csv` / `best_cifar100_baseline.pth` | 保留 | Baseline 與 Overfitting 比較 |
| `cifar100_v2_history.csv` / `best_cifar100_v2.pth` | 保留 | V2 改良效果與補充評估 |
| `cifar100_v3_5_history.csv` / `best_cifar100_v3_5.pth` | **保留** | V3 系列最佳對照組 |
| `cifar100_v4_history.csv` | **務必保留** | Original / EMA 每 Epoch 驗證與訓練紀錄 |
| `best_cifar100_v4.pth` | **保留** | V4 Original 最佳模型（74.78%） |
| `best_cifar100_v4_ema.pth` | **務必保留** | 目前最佳模型（74.86%） |

**注意**：CSV 僅儲存訓練指標，不能還原模型權重；`.pth` 才是實際推論所需的權重。重要實驗完成後應備份兩者。

## 8. 待辦事項

- [x] MNIST Data Loader / CNN Forward Test / Training / Validation / Test
- [x] MNIST CNN vs. Project #1 BPNN 比較
- [x] CIFAR-100 Local Data Loader、Baseline CNN 與 Overfitting 分析
- [x] CIFAR-100 V2：BatchNorm / Dropout / Data Augmentation
- [x] CIFAR-100 V3：自行實作 ResidualBlock 與 Residual CNN
- [x] V3.1–V3.5：Dropout、Random Erasing、Mixup / CutMix、100 Epochs 實驗
- [x] V4：Label Smoothing + EMA + Warmup 整合優化
- [x] V4 Original / EMA 最佳 Checkpoint：Evaluation Mode 比較
- [ ] 彙整各版本 Learning Curves 圖表
- [ ] V5 RandAugment 實驗（視需求）
- [ ] CIFAR-100 最終 Test Evaluation
- [ ] 助教新資料集重新訓練與自動批次預測
- [ ] 指定格式 TXT 輸出
- [ ] 英文簡報與實驗報告

---

**目前階段結論**：MNIST CNN Test Accuracy 為 **98.79%**，高於 Project #1 BPNN 的 **97.77%**。CIFAR-100 自 Baseline 的 **38.52%**，經 V2、V3 與 V4 逐步改善至 **74.86% Validation Accuracy（V4 EMA，+36.34 pp）**。V4 的 Train–Validation Gap 仍達 **24.99 pp**，後續將持續檢驗泛化能力，並完成獨立測試與助教指定的推論輸出流程。
