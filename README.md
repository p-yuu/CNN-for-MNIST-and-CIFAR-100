# Deep Learning Project #2 — CNN for MNIST and CIFAR-100

> 專案紀錄日期：2026-11-06  
> 目前狀態：**MNIST CNN 已完成訓練與測試；CIFAR-100 Baseline 已完成訓練與驗證，準備進行 V2 改良。**

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
│   ├── CIFAR-100/
│   │   ├── train
│   │   ├── test
│   │   └── meta
│   └── MNIST/                    # torchvision 下載的 MNIST 資料
├── src/
│   ├── data_loader.py           # MNIST / CIFAR-100 資料載入與分割
│   ├── model.py                 # MNISTCNN / CIFAR100CNN
│   ├── train.py                 # MNIST 訓練
│   ├── evaluate.py              # MNIST 測試
│   └── train_cifar100.py        # CIFAR-100 Baseline 訓練
└── outputs/
    ├── models/
    │   ├── best_mnist_cnn.pth
    │   └── best_cifar100_baseline.pth
    └── results/
        ├── mnist_training_history.csv
        └── cifar100_baseline_history.csv
```

> 上述為目前規劃／使用的檔案結構；檔案是否存在請以本機資料夾為準。未來 V2 將另建程式與輸出檔案，避免覆蓋 Baseline。

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

- 使用本機 `data/CIFAR-100/train`、`test`、`meta`（Python Pickle 格式）。
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

## 5. 執行方式

先啟用 Conda 環境：

```powershell
conda activate dl_project1
cd C:\Data\code\hw\deep_learning_HW\project2\src
```

**資料載入／模型形狀檢查**

```powershell
python .\data_loader.py
python .\model.py
```

> 這兩個檔案的 `__main__` 測試區塊目前設定為 CIFAR-100；需要測試 MNIST 時可切換測試區塊。

**MNIST 訓練與測試**

```powershell
python .\train.py
python .\evaluate.py
```

**CIFAR-100 Baseline 訓練**

```powershell
python .\train_cifar100.py
```

> 重新執行訓練程式可能覆寫同名的最佳權重與 CSV。進行新實驗前，請先備份原始結果或使用新檔名。

## 6. 下一階段：CIFAR-100 V2（尚未實作）

預計建立獨立的 `CIFAR100CNNV2` 與 `train_cifar100_v2.py`，並評估以下改良：

1. **Data Augmentation**：RandomCrop、RandomHorizontalFlip；Validation/Test 不增強。
2. **Batch Normalization**：提高訓練穩定性。
3. **Dropout / Weight Decay**：減少 Overfitting。
4. **Global Average Pooling**：改變分類器設計。
5. **Learning Rate Scheduler**：後期降低 Learning Rate。
6. 視 V2 結果，再考慮 Residual Connections 與 Ablation Study。

建議保留 Baseline 與 V2 各自的模型程式、超參數、CSV 與最佳驗證結果，以便公平比較。若多項方法同時修改，只能歸因於**整體改良方案**，不能直接判定單一方法的效果。

## 7. 實驗檔案保留策略

| 檔案 | 建議 | 原因 |
|---|---|---|
| `mnist_training_history.csv` | 保留 | MNIST Learning Curves 與比較依據 |
| `best_mnist_cnn.pth` | 保留 | 已完成測試的 MNIST 最佳模型 |
| `cifar100_baseline_history.csv` | **務必保留** | CIFAR-100 Baseline 完整訓練紀錄 |
| `best_cifar100_baseline.pth` | 建議暫留 | 可用於 Baseline 推論、錯誤分析及重現比較 |
| `best_cifar100_v2.pth` | V2 完成後保留 | 後續評估及部署候選模型 |

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
- [ ] CIFAR-100 V2 Data Augmentation / Model / Training
- [ ] CIFAR-100 Baseline vs. V2 比較與圖表
- [ ] CIFAR-100 最終 Test Evaluation
- [ ] 助教新資料集訓練與自動批次預測
- [ ] 指定格式 TXT 輸出
- [ ] 英文簡報與實驗報告

---

**目前階段結論**：MNIST CNN 已成功完成分類任務；CIFAR-100 Baseline 已建立可重現的訓練與驗證流程，但存在明顯 Overfitting。下一步應在保留 Baseline 紀錄的前提下建立 V2，改善模型的泛化能力。
