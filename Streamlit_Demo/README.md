# Demo Klasifikasi Sampah (Streamlit)

Aplikasi demo untuk klasifikasi citra sampah (**Kardus · Kertas · Plastik**) dengan dua model dan visualisasi interpretasi (Grad-CAM / CAM). Folder ini bersifat **mandiri**, sudah berisi semua model dan dataset yang dibutuhkan.

## Cara Menjalankan

```bash
cd Streamlit_Demo
pip install -r requirements.txt
streamlit run app.py
```

Aplikasi terbuka di `http://localhost:8501`. Pilih model di panel kiri, upload gambar sampah, lalu lihat prediksi + confidence + heatmap.

## Fitur

- **Pilih model**: ShuffleNet V2 (akurasi test 92.31%) atau ResNet50 + Random Forest (86.54%).
- **Prediksi + confidence**: kelas hasil prediksi dan probabilitas tiap kelas.
- **Interpretasi model**: Grad-CAM (ShuffleNet, gradien) atau feature-importance CAM (ResNet+RF), menyorot area dasar keputusan.

## Struktur Folder

```
Streamlit_Demo/
├── app.py
├── requirements.txt
├── README.md
├── models/
│   ├── shufflenet_final.pt      # ShuffleNet V2 (PyTorch)
│   ├── resnet_fe.keras          # ResNet50 feature extractor (TF)
│   └── rf_baseline.joblib       # Random Forest (sklearn)
├── dataset/standardized_dataset/  # dipakai untuk menghitung statistik CAM
└── cam_stats.npz                # dibuat otomatis saat pertama dijalankan
```

## Catatan

- Membutuhkan **TensorFlow + PyTorch + scikit-learn** sekaligus (ResNet50 di TF, ShuffleNet di PyTorch).
- Saat pertama dijalankan dengan model ResNet+RF, statistik CAM dihitung sekali dari `dataset/` lalu disimpan ke `cam_stats.npz` (agar berikutnya instan).
- Random Forest yang dipakai adalah **baseline** (200 pohon), akurasi sama dengan versi Optuna tetapi lebih cepat dan lebih kecil.
