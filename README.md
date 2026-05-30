# 🌽 Corn Leaf Disease Detector

Website lokal untuk deteksi penyakit daun jagung menggunakan MobileNetV3-Small + Grad-CAM.

## Struktur Folder
```
corn-disease-app/
├── app.py                  ← Flask backend
├── best_corn_model.pt      ← Model PyTorch
├── requirements.txt        ← Daftar library
├── templates/
│   └── index.html          ← Frontend website
└── static/
    └── uploads/            ← Folder upload sementara
```

## Cara Menjalankan

### 1. Pastikan Python sudah terinstall
```
python --version   # minimal Python 3.8
```

### 2. Buat virtual environment (opsional tapi disarankan)
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

### 3. Install library
```bash
pip install -r requirements.txt
```

### 4. Jalankan server
```bash
python app.py
```

### 5. Buka browser
```
http://localhost:5000
```

## Fitur
- Upload foto daun jagung (PNG/JPG/JPEG/WEBP)
- Prediksi 4 kelas: Bercak Daun, Daun Sehat, Hawar Daun, Karat Daun
- Confidence score + distribusi probabilitas semua kelas
- Visualisasi Grad-CAM: Original | Heatmap | Overlay
- Tampilan modern dark mode
- Drag & drop support

## Kelas yang Dideteksi
| Kelas       | Bahasa Inggris     | Keterangan |
|-------------|-------------------|------------|
| Bercak Daun | Gray Leaf Spot    | Jamur Cercospora zeae-maydis |
| Daun Sehat  | Healthy           | Tidak terinfeksi |
| Hawar Daun  | Northern Leaf Blight | Jamur Exserohilum turcicum |
| Karat Daun  | Common Rust       | Jamur Puccinia sorghi |
