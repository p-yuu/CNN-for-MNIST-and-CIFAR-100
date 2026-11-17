# CNN for MNIST and CIFAR-100

**Deep Learning Project #2 | PyTorch | From-scratch CNN and Residual CNN**

本專案使用 PyTorch 自行設計與訓練卷積神經網路（CNN），探討影像分類模型的架構設計與訓練策略。實驗分為兩部分：在 **MNIST** 上比較 CNN 與 Project #1 的 BPNN；在 **CIFAR-100** 上從 Baseline CNN 開始，逐步測試 Batch Normalization、Residual Connections、Data Augmentation、Mixup/CutMix、Learning Rate Scheduling 與不同 Optimizers，最終得到 V8 模型。

> **Project status (2026-11-17):** MNIST 與 CIFAR-100 主要實驗及最終測試已完成。以下準確率為既有實驗紀錄；若重新訓練，結果可能因環境與隨機性而略有差異。

## Highlights

| Dataset / Model | Evaluation | Accuracy |
|---|---|---:|
| MNIST — BPNN (Project #1) | Test | 97.77% |
| **MNIST — CNN** | **Test** | **98.79%** |
| CIFAR-100 — Baseline CNN | Best validation | 38.52% |
| **CIFAR-100 — V8 Residual CNN** | **Best validation** | **78.86%** |
| **CIFAR-100 — V8 Residual CNN** | **Test** | **78.79%** |
| **CIFAR-100 — V8 + horizontal-flip TTA** | **Test** | **79.29%** |

- CIFAR-100 的 Best Validation Accuracy 從 **38.52% 提升至 78.86%（+40.34 percentage points）**。
- 最終 V8 採用自行實作的 **6-block Residual CNN**，約 **2.80M trainable parameters**。
- **不使用預訓練權重**，也不直接套用現成完整 ResNet 模型。
- 所有版本以 Validation Set 進行模型選擇；最終 Test Set 用於獨立評估。

## 1. Project Objectives

1. **MNIST:** 建立 CNN，與先前自行實作的 Backpropagation Neural Network（BPNN）比較影像分類表現與架構差異。
2. **CIFAR-100:** 建立 Baseline CNN，分析過度擬合，逐步改善模型架構與訓練策略。
3. **Experimental Analysis:** 記錄不同版本的 Validation Accuracy、訓練曲線、參數量與設計取捨。
4. **Inference:** 使用選定的最佳模型進行獨立測試，並評估 Horizontal-Flip Test-Time Augmentation（TTA）。

## 2. Environment

原始開發紀錄如下（以實際執行環境為準）：

| Item | Environment |
|---|---|
| OS | Windows / PowerShell / VS Code |
| Python | 3.11.17 |
| Conda environment | `dl_project1` |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| CUDA | 13.0（原始環境紀錄） |
| Libraries | `torch`, `torchvision`, `numpy`, `Pillow`, `tqdm` |

> **版本提醒：** 舊版 README 記錄的 PyTorch 版本字串為 `2.14.1+cu130`，尚未與目前環境重新核對，因此此處不將其列為已驗證版本。可使用下列指令查詢實際安裝版本。

```powershell
python --version
python -c "import torch, torchvision; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### Installation

建議建立獨立環境，並依照 [PyTorch 官方安裝說明](https://pytorch.org/get-started/locally/) 安裝與 GPU 相容的 `torch` / `torchvision`。

```powershell
conda create -n dl_project2 python=3.11 -y
conda activate dl_project2
# Install torch and torchvision using the command appropriate for your platform.
pip install numpy pillow tqdm
```

> 若儲存庫提供 `requirements.txt` 或環境鎖定檔，請優先使用該檔案；目前上傳的 README 未提供可核對的完整套件版本清單。

## 3. Dataset and Splits

| Dataset | Input | Classes | Train | Validation | Test |
|---|---|---:|---:|---:|---:|
| MNIST | 1 × 28 × 28 | 10 | 55,000 | 5,000 | 10,000 |
| CIFAR-100 | 3 × 32 × 32 | 100 | 45,000 | 5,000 | 10,000 |

- **MNIST:** 透過 `torchvision.datasets.MNIST` 載入，使用 `ToTensor()`，像素值縮放至 `[0, 1]`。
- **CIFAR-100:** 原始實驗使用 `data/cifar-100-python/` 下的 Python Pickle 檔（`train`、`test`、`meta`），透過自訂 `LocalCIFAR100` 讀取 `fine_labels`，並使用 CIFAR-100 Mean/Std Normalization。
- **Data split:** 官方 Training Set 再切分 Training / Validation，使用固定 Seed **42**；官方 Test Set 保留供最終評估。
- **Validation/Test transforms:** 不使用隨機資料增強；訓練階段的增強策略依版本而異。

> **Data leakage prevention:** Training 與 Validation 使用互不重疊的索引；同一組資料切分用於各版本比較。不同版本可能同時更改多個設定，因此版本間的差異不等同於嚴格控制變因的 Ablation Study。

## 4. Project Structure

以下目錄依目前 VS Code 專案樹整理。`data/`、`outputs/` 與 `test_in_class/` 內部未展開的內容，以實際儲存庫為準；`__pycache__/` 為 Python 自動產生的快取，不建議提交 Git。

```text
project2/
├── README.md
├── .gitignore
├── data/                           # MNIST / CIFAR-100 datasets
├── outputs/                        # Checkpoints, training histories, figures
├── src/
│   ├── data_loader.py              # Dataset loading and preprocessing
│   ├── model.py                    # CNN / Residual CNN architectures
│   ├── train.py                    # MNIST training
│   ├── evaluate.py                 # MNIST evaluation
│   ├── train_cifar100.py           # CIFAR-100 baseline
│   ├── train_cifar100_v2.py        # CIFAR-100 V2
│   ├── train_cifar100_v3.py        # CIFAR-100 V3 series entry
│   ├── train_cifar100_v4.py        # CIFAR-100 V4
│   ├── train_cifar100_v5.py        # CIFAR-100 V5
│   ├── train_cifar100_v6.py        # CIFAR-100 V6
│   ├── train_cifar100_v7.py        # CIFAR-100 V7 series
│   ├── train_cifar100_v8.py        # Final V8 training
│   ├── train_cifar100_v9.py        # Additional V9 experiment (results not documented here)
│   ├── evaluate_cifar100_v2.py     # CIFAR-100 checkpoint evaluation
│   ├── evaluate_cifar100_tta.py    # CIFAR-100 test-time augmentation evaluation
│   └── plot_history.py             # Training-history visualization
└── test_in_class/                  # In-class / TA dataset workflow
    ├── data/
    ├── outputs/
    └── src/
        ├── data_loader.py
        ├── model.py
        ├── train.py
        ├── evaluate.py
        ├── predict.py
        ├── export_cifar100.py
        └── plot_history.py
```

> **註：** `train_cifar100_v3.py` 和 `train_cifar100_v7.py` 是目前看到的檔名；V3.1～V3.5、V7.1 是否由腳本內設定切換，需依程式確認。`train_cifar100_v9.py` 確實存在，但此 README 不將 V9 視為已確認的最終最佳模型。`test_in_class/` 的資料格式、CLI 參數及輸出格式也應以其實際程式為準。

大型資料集、訓練輸出與模型權重可透過 `.gitignore` 排除；如需分享權重，建議使用 GitHub Releases 或提供下載連結。

## 5. MNIST — CNN vs. BPNN

### CNN Architecture

```text
Input: 1 × 28 × 28
  → Conv(1→32, 3×3) → ReLU → MaxPool(2)
  → Conv(32→64, 3×3) → ReLU → MaxPool(2)
  → Flatten (64×7×7)
  → Linear(3136→128) → ReLU
  → Linear(128→10)
```

| Hyperparameter | Setting |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 64 |
| Epochs | 10 |
| Loss | CrossEntropyLoss |
| Trainable parameters | 421,642 |
| Checkpoint selection | Best validation accuracy |

### Results

| Metric | Result |
|---|---:|
| Best validation accuracy | 98.84% (Epoch 6) |
| CNN test accuracy | **98.79%** |
| CNN test loss | 0.0360 |
| Correct predictions | 9,879 / 10,000 |
| BPNN test accuracy (Project #1) | 97.77% |
| CNN improvement | **+1.02 pp** |

CNN 保留影像的二維空間結構，利用 Convolution 與 Weight Sharing 學習局部特徵；BPNN 則將影像攤平成向量並透過 Fully Connected Layers 分類。CNN 在本次 MNIST 實驗中表現較好，但兩者的訓練設定不完全相同，**不能將全部提升單獨歸因於 CNN 架構**。

> **BPNN parameters:** 若最終 BPNN 確實採用 `784 → 512 → 10` 且兩個 Linear Layers 均有 Bias，參數量為 **407,050**；此數字為架構推算，應以 Project #1 最終程式確認後再用於正式比較。

## 6. CIFAR-100 — Architecture and Experiments

### 6.1 Baseline CNN

```text
Input: 3 × 32 × 32
  → [Conv(3→32) → ReLU → Conv(32→32) → ReLU → MaxPool]
  → [Conv(32→64) → ReLU → Conv(64→64) → ReLU → MaxPool]
  → [Conv(64→128) → ReLU → Conv(128→128) → ReLU → MaxPool]
  → Flatten (128×4×4)
  → Linear(2048→256) → ReLU
  → Linear(256→100)
```

- Conv Kernels：3×3、Padding 1；Pooling：2×2。
- **837,252 parameters**；Adam（LR 0.001）、Batch Size 128、30 Epochs。
- 未使用 Data Augmentation。
- Best Validation Accuracy **38.52%（Epoch 8）**；Final Validation Accuracy **33.36%**。
- 最後一個 Epoch 的 Training Accuracy 達 **92.68%**，Validation Loss 升至 **6.9923**，顯示明顯 Overfitting。

### 6.2 V2 — Batch Normalization and Regularization

- 引入 BatchNorm、Dropout、RandomCrop、RandomHorizontalFlip。
- 使用 AdamW（LR 0.001、Weight Decay 1e-4）、CosineAnnealingLR、50 Epochs、Batch Size 128。
- **1,172,900 parameters**。
- Best Validation Accuracy **63.62%（Epoch 46）**，比 Baseline 提升 **25.10 pp**。
- Best Checkpoint 在無增強的 Evaluation Mode 下：Train **74.83%**、Validation **63.62%**，差距 **11.21 pp**。

### 6.3 V3 — Custom Residual CNN

自行實作的 `ResidualBlock` 包含兩層 3×3 Convolution、BatchNorm、ReLU 和 Skip Connection。當特徵圖尺寸或 Channel 數不同時，使用 1×1 Convolution 進行 Shortcut Projection。

```text
Input: 3 × 32 × 32
  → Stem: Conv(3→64, 3×3) + BN + ReLU
  → Stage 1: 2 Residual Blocks, 64 channels  → 64 × 32 × 32
  → Stage 2: 2 Residual Blocks, 128 channels → 128 × 16 × 16
  → Stage 3: 2 Residual Blocks, 256 channels → 256 × 8 × 8
  → Adaptive Average Pooling → 256 features
  → Dropout(p=0.3) [from V3.1 onward]
  → Linear(256→100)
```

**Trainable parameters: 2,800,804**。初始 V3 不使用分類器 Dropout；V3.1 起加入 `Dropout(0.3)`。後續 V8 保留六個 Residual Blocks 的主要架構。

### 6.4 Version-by-Version Results

下表的 Accuracy 均為 **Best Validation Accuracy**，不是 Test Accuracy。

| Version | Main change / experiment | Epochs | Best val. acc. |
|---|---|---:|---:|
| Baseline | Basic CNN, no augmentation | 30 | 38.52% |
| V2 | BatchNorm + Dropout + augmentation | 50 | 63.62% |
| V3 | Custom Residual CNN (6 blocks) | 50 | 69.14% |
| V3.1 | Add Dropout (0.3) | 50 | 68.92% |
| V3.2 | Add Random Erasing | 50 | 69.12% |
| V3.3 | Mixup (α=0.2) | 50 | 70.64% |
| V3.4 | CutMix (α=1.0) instead of Mixup | 50 | 69.06% |
| V3.5 | Mixup + longer training | 100 | 72.66% |
| V4 (EMA) | Label Smoothing + Warmup + EMA | 100 | 74.86% |
| V5 | RandAugment experiment | 100 | 75.44% |
| V6 | Deeper Residual CNN (9 blocks) | 100 | 76.04% |
| V7 | Return to 6 blocks; SGD + Nesterov | 50 | 77.00% |
| V7.1 | Extended training | 100 | 78.04% |
| **V8** | **Final 6-block model and training configuration** | **100** | **78.86%** |

> **Interpretation:** 這是逐步探索的版本紀錄，並非每一列都只改變一個因素。例如 V6 → V7 同時改變網路深度、Optimizer 和 Epochs，因此不能將 Accuracy 差異完全歸因於 SGD。

### 6.5 Key Experimental Observations

- **Baseline overfitting:** Baseline 的 Training Accuracy 持續上升，但 Validation Accuracy 在早期達到最高值後下降。
- **Residual connections:** V3 比 V2 提高 **5.52 pp**，但同時模型規模與其他設定也有差異。
- **Regularization experiments:** V3.1（Dropout）與 V3.2（Random Erasing）在當次設定下未優於 V3。
- **Mixup vs. CutMix:** 50-Epoch 實驗中 V3.3 Mixup（70.64%）高於 V3.4 CutMix（69.06%）；這不代表 Mixup 在所有設定下都優於 CutMix。
- **Longer training:** V3.5 和 V7.1 的結果支持進一步探索較長的訓練排程。
- **Deeper is not always better:** V6 使用九個 Blocks，但後續採用六個 Blocks 的 V7/V8 取得更高 Accuracy；比較仍受到其他訓練設定差異影響。
- **V4 EMA:** Original 最佳 Validation Accuracy **74.78%**，EMA **74.86%**，差異僅 **0.08 pp**，不足以單獨證明 EMA 有顯著優勢。

## 7. Final Model — V8

### 7.1 Architecture

| Layer | Output shape | Configuration |
|---|---|---|
| Input | 3 × 32 × 32 | RGB image |
| Stem | 64 × 32 × 32 | 3×3 Conv + BN + ReLU |
| Residual Stage 1 | 64 × 32 × 32 | 2 blocks |
| Residual Stage 2 | 128 × 16 × 16 | 2 blocks; downsampling |
| Residual Stage 3 | 256 × 8 × 8 | 2 blocks; downsampling |
| Global Average Pooling | 256 | AdaptiveAvgPool2d(1) |
| Dropout | 256 | p=0.3 |
| Classification Head | 100 | Linear(256, 100) |

**Total trainable parameters: 2,800,804 (~2.80M).**

### 7.2 Final Training Configuration

| Setting | V8 configuration |
|---|---|
| Epochs | 100 |
| Optimizer | SGD with Nesterov momentum |
| Initial learning rate | 0.1 |
| Momentum | 0.9 |
| LR schedule | Warmup + Cosine Decay |
| Label smoothing | 0.05 |
| Dropout | 0.3 |
| Image augmentation | RandomCrop, HorizontalFlip, RandAugment |
| Sample mixing | Mixup + CutMix |
| Dataset split seed | 42 |

**待由 V8 原始程式核對：** Batch Size、Weight Decay、Warmup Epochs、Minimum LR、Mixup/CutMix 的 α 與選用機率，以及是否保留 Random Erasing。這些設定不應直接從 V4 推定為 V8 的設定。

### 7.3 Final CIFAR-100 Results

| Evaluation | Accuracy |
|---|---:|
| Best validation (Epoch 99) | **78.86%** |
| Independent test | **78.79%** |
| Test + horizontal-flip TTA | **79.29%** |

- **Validation vs. Test:** 差異僅 **0.07 pp**。
- **Horizontal-flip TTA:** 相對標準 Test Inference 提升 **0.50 pp**。
- **Baseline → V8:** Best Validation Accuracy 提升 **40.34 pp**。

### 7.4 Training Curves and Metric Interpretation

V8 訓練共 100 Epochs，最佳 Validation Accuracy 出現在 **Epoch 99**。既有曲線顯示 Validation Accuracy 整體上升、Validation Loss 整體下降。

在訓練曲線中，**Training Accuracy 低於 Validation Accuracy**，且 **Training Loss 高於 Validation Loss**。由於 Training 階段可能包含 RandAugment、Mixup、CutMix 與 Label Smoothing，而 Validation 階段使用未隨機增強的影像，兩者的計算條件可能不同。**不可直接以這兩條曲線的差值判斷 Underfitting / Overfitting**；若要比較泛化差距，應使用同一模型在無隨機增強的 Training / Validation Sets 上以 `eval()` 重新評估。

> 若已將 Accuracy / Loss 圖提交至儲存庫，可在此加入圖片，例如 `![V8 accuracy](docs/figures/v8_accuracy.png)`。請先確認實際路徑，避免 GitHub 顯示破圖。

## 8. How to Run

以下指令以 **PowerShell、專案根目錄 `project2/`** 為起點。環境名稱請替換為你實際建立的 Conda 環境；原始開發紀錄使用 `dl_project1`。

```powershell
conda activate dl_project1
cd src
```

### Data Loader and Model Checks

```powershell
python data_loader.py
python model.py
```

### MNIST

```powershell
python train.py
python evaluate.py
```

### CIFAR-100 — Training Scripts

```powershell
python train_cifar100.py
python train_cifar100_v2.py
python train_cifar100_v3.py
python train_cifar100_v4.py
python train_cifar100_v5.py
python train_cifar100_v6.py
python train_cifar100_v7.py
python train_cifar100_v8.py
```

以上是**實際存在的腳本入口**，並非要求依序重新執行全部實驗。不同版本可能需要不同設定或較長訓練時間；執行前請檢查程式內的 Dataset、Checkpoint 與輸出路徑。`train_cifar100_v9.py` 另有保留，但 V9 的實驗設定與結果尚未納入本文。

### CIFAR-100 — Evaluation and Visualization

```powershell
python evaluate_cifar100_v2.py
python evaluate_cifar100_tta.py
python plot_history.py
```

`evaluate_cifar100_v2.py` 為沿用的評估腳本名稱；請先確認其載入的模型類別與權重版本。`evaluate_cifar100_tta.py` 用於 TTA 評估。上述指令是否需額外 CLI 參數，仍應以程式內的 `argparse` 或設定為準。

### In-Class / TA Workflow

回到專案根目錄後：

```powershell
cd ..
cd test_in_class\src
```

此資料夾包含 `train.py`、`evaluate.py`、`predict.py`、`export_cifar100.py` 與 `plot_history.py`，用於課堂測試資料的訓練、推論、匯出及結果視覺化。**實際執行順序、輸入資料格式與輸出檔案規格，需依各腳本內容確認**，不應直接假設與主要 CIFAR-100 流程相同。

> **Warning:** 重新執行訓練可能覆寫同名 `.pth` Checkpoints 與 `.csv` Training Histories；請先備份重要結果。

## 9. Outputs and Reproducibility

| Artifact | Purpose |
|---|---|
| `outputs/models/*.pth` | Model checkpoints / weights |
| `outputs/results/*.csv` | Epoch-level training histories |
| Accuracy / Loss figures | Training dynamics and report figures |
| Final report (if committed) | Architecture, experimental comparison, and limitations |

為提升可重現性，建議在 GitHub 同時提供：

1. `requirements.txt` 或 Conda `environment.yml`（實際測試過的版本）。
2. V8 完整 Training / Evaluation Scripts，以及所有重要 CLI Arguments 或 Configurations。
3. Dataset 下載或準備方式與固定切分 Seed。
4. 最終 Checkpoint 的取得方式、載入方法與推論輸出格式。
5. 最終 Training Curves 與實驗紀錄；若權重太大，使用 GitHub Releases 或外部下載連結。

## 10. Limitations and Future Work

- **Controlled ablations:** 不同版本可能同時改變多個設定，無法從目前結果隔離各項技術的獨立貢獻。
- **Single split / seed:** 主要實驗使用固定資料切分與 Seed 42，尚缺乏多次獨立訓練的變異分析。
- **Per-class performance:** 整體 Accuracy 無法顯示哪些 CIFAR-100 類別最容易混淆，可進一步加入 Confusion Matrix 與 Per-Class Accuracy。
- **Computational efficiency:** 未對所有版本進行一致的訓練時間、推論延遲與計算量比較。
- **Reproducibility:** 最終 V8 的部分超參數與完整命令仍需對照實際原始程式補齊。

## 11. Notes

- **Validation Accuracy ≠ Test Accuracy:** 各版本表格為最佳 Validation 結果；只有最終獨立測試的數字才標記為 Test。
- **Percentage points (pp):** 例如 38.52% → 78.86% 是 **+40.34 pp**，不是「相對增加 40.34%」。
- **Training vs. evaluation metrics:** 受 Data Augmentation、Sample Mixing、Dropout 與 Loss 設定影響，Training Log 與 Evaluation Mode 的 Accuracy / Loss 不一定可直接比較。
- **No pretrained weights:** 所有 CNN / Residual CNN 實驗從頭訓練。

---

**Summary:** MNIST CNN achieved **98.79% test accuracy**, compared with **97.77%** for the BPNN. On CIFAR-100, the best validation accuracy improved from **38.52% (Baseline)** to **78.86% (V8)**. The final V8 model achieved **78.79% test accuracy**, increasing to **79.29% with horizontal-flip TTA**.
