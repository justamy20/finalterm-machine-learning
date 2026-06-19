#  Final Term — Machine Learning Individual Task

> **UAS Individual Task | Machine Learning Class**
> *"Hands-On End-to-End Models Machine Learning and Deep Learning"*

Repository ini berisi **dua proyek end-to-end** yang dikerjakan sebagai tugas akhir individu mata kuliah Machine Learning. Setiap proyek mencakup pipeline lengkap mulai dari preprocessing, training model, hyperparameter tuning, hingga experiment tracking.

---

##  Struktur Repository

```
finalterm-machine-learning/
├── Fraud_Detection_Deep_Learning_FinalTerm(1).ipynb
├── Fraud_Detection_Deep_Learning_FinalTerm.ipynb
├── README.md
├── regression_song_year(1).ipynb 
└── regression_song_year.ipynb
---

## 📋 Daftar Tugas

| # | Tugas | Topik | Tipe | Notebook |
|---|---|---|---|---|
| 1 | Fraud Detection | Klasifikasi transaksi | Binary Classification | `Fraud_Detection_Deep_Learning_FinalTerm.ipynb` |
| 2 | Song Year Prediction | Prediksi tahun rilis lagu | Regression | `regression_song_year.ipynb` |

---

---

# 📂 Tugas 1 — Fraud Detection: End-to-End Deep Learning Pipeline

> *Predict the probability of an online transaction being fraudulent (`isFraud`)*

## 📌 Deskripsi

End-to-end deep learning pipeline untuk memprediksi apakah sebuah transaksi online bersifat **fraud (`isFraud = 1`)** atau **legitimate (`isFraud = 0`)** berdasarkan fitur transaksi mentah.

## 🔄 Pipeline

1. **Data Loading** — `train_transaction.csv` (+ auto-merge `train_identity.csv` jika tersedia)
2. **EDA** — class balance, missing values, distribusi jumlah transaksi
3. **Preprocessing** — drop kolom high-missing, imputasi median/constant, ordinal encoding, stratified split, feature scaling
4. **Feature Engineering** — `WeightedRandomSampler` untuk menangani class imbalance (~4–5% fraud)
5. **3 Arsitektur Deep Learning:**
   - **Shallow MLP** — baseline deep learning
   - **Deep MLP** — jaringan lebih dalam dengan BatchNorm + Dropout
   - **1D-CNN (Tabular)** — convolutional model pada vektor fitur 1D
6. **Hyperparameter Tuning** — Optuna (pencarian independen per arsitektur)
7. **Experiment Tracking** — MLflow (params, per-epoch metrics, artifacts)
8. **Evaluasi** — ROC-AUC, PR-AUC, F1, Precision, Recall, Confusion Matrix
9. **Bonus Inference** pada `test_transaction.csv`

## 🧠 Hasil & Perbandingan Model

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| **Deep MLP** | **0.9334** | **0.6911** | **0.4169** | **0.2821** | **0.7980** |
| Shallow MLP | 0.9188 | 0.6336 | 0.3688 | 0.2406 | 0.7893 |
| 1D-CNN Tabular | 0.8833 | 0.4805 | 0.2703 | 0.1653 | 0.7409 |

> 📊 **PR-AUC** digunakan sebagai metrik utama karena lebih informatif daripada ROC-AUC pada kondisi class imbalance yang ekstrem.

### 🏆 Model Terbaik: Deep MLP

- **PR-AUC tertinggi (0.6911)** — terbaik dalam meranking kasus fraud di atas transaksi legitimate
- **Recall tertinggi (0.7980)** — mendeteksi ~80% fraud nyata, penting karena *false negative* lebih merugikan
- **1D-CNN underperform** karena kolom tabular tidak memiliki struktur spasial/sekuensial yang dibutuhkan konvolusi

### Poin Diskusi Utama

- **Deep MLP unggul** karena depth tambahan + BatchNorm + Dropout memungkinkan model menangkap interaksi fitur non-linear tanpa overfitting
- **WeightedRandomSampler** sangat krusial — mendorong Recall hingga ~0.80, trade-off yang wajar karena mendeteksi fraud lebih penting dari false alarm
- **Interpretasi bisnis:** ~1 dari 4 transaksi yang diflag benar-benar fraud (Precision ≈ 0.28) sementara ~80% fraud nyata tertangkap (Recall ≈ 0.80)
- **Limitasi:** preprocessing standar; gain lebih lanjut memungkinkan dengan feature engineering richer (aggregasi per-user, frequency encoding, time-window features)

## 🗂️ Cara Menjalankan (Tugas 1)

1. Buka `notebooks/Fraud_Detection_Deep_Learning_FinalTerm.ipynb` di **Google Colab**
2. Upload `train_transaction.csv` (dan `train_identity.csv` jika ada) ke Google Drive
3. Update path `DATA_DIR` di **Section 0** sesuai folder Drive kamu
4. Jalankan semua cell dari atas ke bawah

| Section | Konten |
|---|---|
| 0 | Setup environment, mount Drive, `FAST_MODE` toggle |
| 1 | Data loading |
| 2 | EDA |
| 3 | Cleaning & preprocessing |
| 4 | Feature engineering & class imbalance handling |
| 5 | Definisi arsitektur model (3 model) |
| 6 | Training & evaluation utilities |
| 7 | MLflow tracking setup |
| 8 | Optuna tuning setup |
| 9 | Tuning + final training |
| 10 | Perbandingan model & evaluasi |
| 11 | Inspeksi MLflow runs |
| 12 | Simpan model terbaik |
| 13 | Bonus inference |
| 14 | Kesimpulan & analisis |

> 💡 Gunakan `FAST_MODE = True` untuk uji coba cepat, lalu `FAST_MODE = False` di GPU runtime untuk full run.

---

---

# 📂 Tugas 2 — Song Release Year Prediction: End-to-End Regression Pipeline

> *Predict the release year of a song based on audio features*

## 📌 Deskripsi

End-to-end regression pipeline untuk memprediksi **tahun rilis lagu** berdasarkan fitur audio numerik menggunakan Deep Learning (PyTorch MLP).

Dataset: `midterm-regresi-dataset.csv`
- **Kolom pertama** → target label (tahun rilis, contoh: `2001`)
- **Kolom lainnya** → fitur audio numerik (timbre, karakteristik sinyal, `feature_1` ... `feature_N`)

## 🔄 Pipeline

1. **Data Loading & EDA** — statistik deskriptif, missing values, distribusi target
2. **Preprocessing** — outlier removal (IQR), StandardScaler, train-test split 80/20
3. **Baseline Model** — Linear Regression (Scikit-learn)
4. **Deep Learning Model** — MLP PyTorch (BatchNorm + ReLU + Dropout)
5. **Hyperparameter Tuning** — Optuna (15 trials, minimize RMSE)
6. **Evaluasi & Perbandingan** — MSE, RMSE, MAE, R² untuk semua model
7. **Interpretability** — LIME (penjelasan prediksi per sampel)
8. **Experiment Tracking** — MLflow (log params, metrics, artifacts, model)

## 🧠 Arsitektur Model (MLP PyTorch)

```
Input Layer  (N fitur audio)
      │
  Linear → BatchNorm → ReLU → Dropout
      │
  Linear → BatchNorm → ReLU → Dropout
      │
    ...  (jumlah layer ditentukan Optuna: 2–4)
      │
  Output Layer (1 neuron → prediksi tahun)
```

**Hyperparameter yang di-tuning Optuna:**

| Parameter | Search Space |
|---|---|
| Jumlah hidden layer | 2 – 4 |
| Ukuran hidden layer | 64 / 128 / 256 / 512 |
| Dropout rate | 0.1 – 0.5 |
| Learning rate | 1e-4 – 1e-2 |

## 📊 Metrik Evaluasi

| Model | MSE | RMSE | MAE | R² |
|---|---|---|---|---|
| Linear Regression | — | — | — | — |
| MLP Initial | — | — | — | — |
| MLP Optuna Tuned | — | — | — | — |

> *Nilai metrik akan terisi setelah notebook dijalankan dengan dataset asli.*

## 📈 Output Visualisasi

- `eda_plots.png` — Distribusi tahun rilis & boxplot fitur
- `training_loss.png` — Kurva training loss MLP
- `optuna_history.png` — History optimasi Optuna (15 trials)
- `model_comparison.png` — Perbandingan RMSE, R², Actual vs Predicted
- `lime_explanation.png` — Penjelasan prediksi individual (LIME)

## 🗂️ Cara Menjalankan (Tugas 2)

1. Buka `notebooks/regression_song_year.ipynb` di **Google Colab**
2. Jalankan cell install dependencies
3. Upload `midterm-regresi-dataset.csv` saat cell upload muncul
4. Jalankan semua cell (`Runtime → Run All`)

> 💡 Aktifkan GPU: `Runtime → Change runtime type → T4 GPU`

---

---

## 🛠️ Tech Stack (Keseluruhan)

| Komponen | Library/Framework |
|---|---|
| Deep Learning | PyTorch |
| ML Baseline | Scikit-learn |
| Hyperparameter Tuning | Optuna |
| Experiment Tracking | MLflow |
| Interpretability | LIME |
| Visualisasi | Matplotlib, Seaborn |
| Environment | Google Colab |

---

## 👤 Identitas

| | |
|---|---|
| **Nama** | *Syeh Khatami* |
| **NIM** | *101032300157* |
| **Kelas** | *TK-47-04* |
| **Mata Kuliah** | Machine Learning |
| **Semester** | *6* |

---
