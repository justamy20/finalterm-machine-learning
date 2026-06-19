# 🔍 Fraud Detection — End-to-End Deep Learning Pipeline

> Final Term Individual Task — Machine Learning Class
> *"Hands-On End-to-End Models Deep Learning"*

End-to-end deep learning pipeline to predict the probability of an online transaction being fraudulent (`isFraud`), built from raw transaction data through preprocessing, model training, hyperparameter tuning, and experiment tracking.

---

## 📌 Purpose of This Repository

This repository contains the complete deliverable for the Machine Learning final term individual task. It demonstrates a full **end-to-end deep learning workflow** — from raw tabular transaction data to a tuned, tracked, and compared set of deep learning models — applied to a real-world **binary fraud classification problem**.

---

## 📖 Project Overview

The goal is to predict whether an online transaction is fraudulent (`isFraud = 1`) or legitimate (`isFraud = 0`) using transaction-level features (amount, product code, card information, address, engineered `C`/`D`/`V` features, etc.).

The workflow covers:

1. **Data loading** — `train_transaction.csv` (with optional auto-merge of `train_identity.csv` if present)
2. **Exploratory Data Analysis (EDA)** — class balance, missing values, transaction amount distribution
3. **Data cleaning & preprocessing** — dropping high-missing columns, median/constant imputation, ordinal encoding, stratified train/val/test split, feature scaling
4. **Feature engineering & class imbalance handling** — `WeightedRandomSampler` to address the ~4–5% fraud rate
5. **Three deep learning architectures**, trained and compared:
   - **Shallow MLP** — baseline deep learning model
   - **Deep MLP** — deeper network with BatchNorm + Dropout
   - **1D-CNN (Tabular)** — convolutional model treating the feature vector as a 1D signal
6. **Hyperparameter tuning** with **Optuna** (independent search per architecture)
7. **Experiment tracking** with **MLflow** (parameters, per-epoch metrics, final metrics, model artifacts)
8. **Evaluation** using metrics suited for imbalanced classification: ROC-AUC, PR-AUC, F1, Precision, Recall, Confusion Matrix
9. **Bonus inference** on `test_transaction.csv` (auto-detects whether true labels are available)

---

## 🧠 Models & Results

All three architectures were trained on the same preprocessed data, tuned independently with Optuna, and evaluated on the same held-out test set.

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| **Deep MLP** | **0.9334** | **0.6911** | **0.4169** | **0.2821** | **0.7980** |
| Shallow MLP | 0.9188 | 0.6336 | 0.3688 | 0.2406 | 0.7893 |
| 1D-CNN Tabular | 0.8833 | 0.4805 | 0.2703 | 0.1653 | 0.7409 |

> 📊 **PR-AUC** (Average Precision) is treated as the primary metric for model selection, since it is more informative than ROC-AUC under severe class imbalance (fraud is the rare class).

### 🏆 Best Model: Deep MLP

The **Deep MLP** (multiple hidden layers with Batch Normalization + Dropout) achieved the best performance across every metric, most notably:
- **Highest PR-AUC (0.6911)** — best at ranking true fraud cases above legitimate ones under imbalance
- **Highest Recall (0.7980)** — catches ~80% of actual fraud cases, which matters most in a fraud-detection context where missing fraud (false negatives) is costlier than a false alarm

The **1D-CNN** performed worst, which is expected: convolution assumes locally meaningful order between adjacent features, but tabular columns have no such spatial/sequential structure — so the convolutional inductive bias doesn't transfer well here.

### Key Discussion Points

- **Why Deep MLP wins:** the additional depth combined with BatchNorm + Dropout lets the model capture non-linear feature interactions in the data without overfitting, outperforming the shallower baseline.
- **Class imbalance handling:** `WeightedRandomSampler` was essential — it pushed Recall up substantially (to ~0.80 for Deep MLP) at the cost of Precision (~0.28), a reasonable trade-off given that catching fraud matters more than the rate of false alarms in this domain.
- **Business interpretation:** with Deep MLP, roughly 1 in 4 flagged transactions is truly fraudulent (Precision ≈ 0.28) while ~80% of all real fraud is caught (Recall ≈ 0.80) — a strong, actionable balance for a fraud investigation team. The classification threshold (currently 0.5) can be tuned further using the Precision-Recall curve depending on business risk tolerance.
- **Limitations:** preprocessing is intentionally standard (median imputation, ordinal encoding). Further gains are likely with richer feature engineering — e.g., per-user transaction aggregations, frequency encoding, or time-window features.

**Recommendation:** Deep MLP is the recommended architecture for production use, offering the best balance between fraud detection capability (Recall) and alert precision.

---

## 🗂️ How to Navigate This Repository

```
.
├── README.md                                       <- you are here
└── notebooks/
    └── Fraud_Detection_Deep_Learning_FinalTerm.ipynb  <- main end-to-end notebook
```

**To run the notebook:**

1. Open `notebooks/Fraud_Detection_Deep_Learning_FinalTerm.ipynb` in **Google Colab**
2. Upload `train_transaction.csv` (and `train_identity.csv` if available) to a Google Drive folder
3. Update the `DATA_DIR` path in **Section 0** to match your Drive folder
4. Run all cells top to bottom — the notebook is organized into 14 clearly labeled sections:

| Section | Content |
|---|---|
| 0 | Environment setup, Drive mount, `FAST_MODE` toggle |
| 1 | Data loading (auto-detects identity table) |
| 2 | Exploratory Data Analysis |
| 3 | Data cleaning & preprocessing |
| 4 | Feature engineering & class imbalance handling |
| 5 | Model architecture definitions (3 models) |
| 6 | Training & evaluation utilities |
| 7 | MLflow tracking setup |
| 8 | Optuna hyperparameter tuning setup |
| 9 | Run tuning + final training (logged to MLflow) |
| 10 | Model comparison & evaluation (curves, confusion matrices) |
| 11 | Inspecting MLflow runs |
| 12 | Saving the best model |
| 13 | Bonus inference on `test_transaction.csv` |
| 14 | Conclusions & analysis |

> 💡 Tip: keep `FAST_MODE = True` for a quick first run to confirm everything works end-to-end, then switch to `FAST_MODE = False` (ideally on a GPU runtime) for the final full-budget run.

---

## 🛠️ Tech Stack

- **Python 3** / **PyTorch** — model implementation & training
- **Optuna** — automated hyperparameter tuning
- **MLflow** — experiment tracking (params, metrics, model artifacts)
- **scikit-learn** — preprocessing & evaluation metrics
- **pandas / numpy** — data manipulation
- **matplotlib / seaborn** — visualization

---

## 👤 Identification

- **Name:** `Syeh Khatami`
- **Class:** `TK-47-04`
- **NIM:** `101032300157`
