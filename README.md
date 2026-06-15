# Klasifikasi Sampah: Kardus, Kertas, Plastik

Proyek UAS Machine Learning: klasifikasi citra sampah ke dalam 3 kelas (**Sampah Kardus**, **Sampah Kertas**, **Sampah Plastik**) menggunakan transfer learning, dengan alur lengkap dari preprocessing data, pelatihan beberapa model, hingga aplikasi demo interaktif (Streamlit) dengan interpretasi model (Grad-CAM / CAM).

## Latar Belakang

Pengelolaan dan pemilahan sampah adalah isu lingkungan yang relevan di masyarakat. Sistem klasifikasi citra otomatis dapat membantu memilah sampah daur ulang. Proyek ini membandingkan beberapa arsitektur deep learning dan pendekatan hybrid (CNN + Random Forest) untuk tugas tersebut, sekaligus menyediakan demo prediksi untuk gambar baru.

## Dataset

256 gambar yang melalui tahap pembersihan dan standarisasi (224×224), terbagi atas 3 kelas:

| Kelas | Jumlah |
|---|---|
| Sampah Kardus | 97 |
| Sampah Plastik | 81 |
| Sampah Kertas | 78 |

Split 80% train / 20% test (stratified). Augmentasi (albumentations) dipakai untuk memperbanyak sekaligus **menyeimbangkan** data train menjadi 1000 gambar per kelas.

**Link dataset:** https://drive.google.com/file/d/1tfbhtb1EdhTI1ZWPdyW5YWASzZjqSvIZ/view?usp=sharing

## Alur Program

```
1. PREPROCESSING        2. PELATIHAN MODEL          3. DEMO
   raw_dataset             notebook tiap model         Streamlit_Demo/app.py
   → check_quality         (ShuffleNet, ResNet18,      upload gambar →
   → deduplicates          MobileNetV3, EfficientNet,  prediksi + Grad-CAM/CAM
   → standardization       ResNet50+RF)
   → standardized_dataset  → simpan model (.pt/.keras/.joblib)
```

## Struktur Repository

```
UAS ML/
├── README.md
├── Prepocessing_local/          # Tahap 1: script preprocessing
│   ├── 1_check_quality.py
│   ├── 2_deduplicates.py
│   └── 3_standardization.py
├── dataset/                     # raw → cleaned → deduplicated → standardized
├── Shuffle Net/                 # Tahap 2: notebook + model tiap arsitektur
├── Resnet18/
├── MobilenetV3Small/
├── EfficientNetB0/
├── Resnet 50 + Random Forest/
└── Streamlit_Demo/              # Tahap 3: aplikasi demo (mandiri)
    ├── app.py
    ├── requirements.txt
    ├── models/
    └── dataset/
```

## Perbandingan Model

| Model | Framework | Akurasi Test | Macro F1 | Parameter | Ukuran (MB) | Inferensi (ms/gambar) |
|---|---|---|---|---|---|---|
| **ShuffleNet V2** | PyTorch | **92.31%** | 0.92 | 1.26 jt | 4.96 | 2.49 |
| MobileNetV3-Small | TensorFlow | 88.46% | 0.89 | 0.94 jt | 4.12 | 6.99 |
| EfficientNetB0 | TensorFlow | 86.54% | 0.86 | 4.05 jt | 16.26 | 11.73 |
| ResNet18 | PyTorch | 84.62% | 0.85 | 11.18 jt | 42.72 | 1.18 |
| ResNet50 + RF (baseline) | TF + sklearn | 86.54% | 0.87 | 23.6 jt | 97.83 | 5.90 |
| ResNet50 + RF + Optuna | TF + sklearn | 86.54% | 0.87 | 23.6 jt | 99.72 | 8.19 |

**Temuan:** ShuffleNet V2 memberi akurasi tertinggi sekaligus paling ringan. Model lebih besar tidak meningkatkan akurasi pada dataset kecil ini. Tuning Optuna tidak menaikkan akurasi karena performa dibatasi oleh kualitas fitur ResNet (backbone beku), bukan hyperparameter classifier.

---

# 11. Demo Program

## 1. Bahasa Pemrograman

**Python 3** (dikembangkan & diuji di Python 3.10–3.12, dilatih di Google Colab dengan GPU).

## 2. Library yang Digunakan

| Kategori | Library |
|---|---|
| Deep learning | TensorFlow/Keras, PyTorch, torchvision |
| Machine learning klasik | scikit-learn (Random Forest), Optuna (tuning) |
| Pengolahan citra & augmentasi | OpenCV, albumentations, Pillow |
| Data & visualisasi | NumPy, pandas, matplotlib, seaborn |
| Aplikasi demo | Streamlit |
| Interpretasi model | Grad-CAM (implementasi manual), feature-importance CAM |

## 3. Cara Menjalankan Program

### Tahap 0: Clone repository
```bash
git clone https://github.com/Kylorts/UAS-ML1.git
cd UAS-ML1
```
Model terlatih (`.pt` / `.keras` / `.joblib`) sudah ikut dalam repository, sehingga **aplikasi demo (Tahap 3) bisa langsung dijalankan**. Dataset tidak disertakan di repository; unduh dari [link dataset](https://drive.google.com/file/d/1tfbhtb1EdhTI1ZWPdyW5YWASzZjqSvIZ/view?usp=sharing) bila ingin menjalankan preprocessing (Tahap 1) atau melatih ulang model (Tahap 2).

### Tahap 1: Preprocessing data
```bash
cd Prepocessing_local
python 1_check_quality.py     # cek & buang gambar rusak/berkualitas rendah
python 2_deduplicates.py      # hapus gambar duplikat
python 3_standardization.py   # resize & standarisasi ke 224x224
```
Menghasilkan `dataset/standardized_dataset/` yang siap dilatih.

### Tahap 2: Pelatihan model
Buka notebook pada folder tiap model (mis. `Shuffle Net/ShuffleNet_UAS.ipynb`) di Google Colab / Jupyter, lalu jalankan seluruh sel (**Run all**). Tiap notebook melatih model, menampilkan grafik akurasi/loss, confusion matrix, Grad-CAM, ukuran model, dan waktu inferensi, lalu menyimpan model terlatih (`.pt` / `.keras` / `.joblib`).

### Tahap 3: Menjalankan aplikasi demo
```bash
cd Streamlit_Demo
pip install -r requirements.txt
streamlit run app.py
```
Aplikasi terbuka di `http://localhost:8501`. Karena model sudah tersedia di repository, tahap ini dapat langsung dijalankan setelah clone tanpa perlu mengunduh dataset.

## 4. Tampilan Hasil Program

Aplikasi menampilkan: dropdown pemilih model + akurasi test model, area upload gambar, hasil prediksi & probabilitas tiap kelas, serta visualisasi Grad-CAM/CAM bersanding dengan gambar asli.

## 5. Contoh Input dan Output

**Input:** gambar sampah (mis. kardus kemasan).

**Output:**
- Prediksi kelas: `Sampah Kardus`
- Keyakinan: `xx.x%`
- Probabilitas per kelas (bar)
- Heatmap Grad-CAM/CAM yang menyorot area dasar keputusan model

## 6. Link GitHub

> https://github.com/Kylorts/UAS-ML1
