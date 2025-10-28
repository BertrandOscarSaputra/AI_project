# 🧠 Breast Cancer Classification Web App

## 📘 Deskripsi Singkat  
Proyek ini adalah **aplikasi web berbasis AI** yang digunakan untuk **mendeteksi kanker payudara** berdasarkan citra histopatologi.  
Model machine learning yang digunakan dapat mengklasifikasikan gambar menjadi dua kategori:
- 🟥 **Malignant (Ganas)**
- 🟩 **Benign (Jinak)**

Aplikasi ini memudahkan pengguna untuk mengunggah gambar jaringan payudara dan langsung memperoleh hasil prediksi secara cepat melalui antarmuka web interaktif.

---

## 🧩 Fitur Utama  
- 📤 **Upload Gambar**: Unggah citra jaringan payudara untuk analisis.  
- 🤖 **Prediksi Otomatis**: Model AI memberikan hasil klasifikasi secara instan.  
- 📊 **Tampilan Hasil yang Jelas**: Hasil prediksi disertai label kategori dan tingkat keyakinan model.  
- 🌐 **Antarmuka Web Sederhana**: Dapat dijalankan secara lokal di browser.

---

## 🗂️ Struktur Proyek  

| File / Folder | Deskripsi |
|----------------|------------|
| `app.py` | Aplikasi utama untuk menjalankan web server dan menangani prediksi gambar. |
| `evaluate.py` | Skrip untuk menguji performa model (akurasi, loss, dll). |
| `gpt_helper.py` | Modul bantu untuk tugas-tugas terkait AI. |
| `templates/` | Folder berisi file HTML (tampilan web). |
| `static/` | Folder berisi file CSS, JavaScript, dan aset pendukung tampilan. |
| `requirements.txt` | Daftar pustaka Python yang dibutuhkan proyek. |

---

## ⚙️ Teknologi yang Digunakan  
- **Python 3.8+**  
- **Flask / FastAPI** – framework web  
- **TensorFlow / PyTorch** – model deep learning  
- **NumPy, Pandas, scikit-learn** – data preprocessing dan evaluasi  
- **HTML, CSS, JS** – antarmuka pengguna  

---

## 🧠 Alur Kerja Sistem  
1. Pengguna mengunggah citra jaringan payudara.  
2. Aplikasi memproses gambar (resize, normalisasi, dsb).  
3. Model AI melakukan prediksi untuk menentukan kategori:  
   - **Benign (Jinak)**  
   - **Malignant (Ganas)**  
4. Hasil klasifikasi ditampilkan di browser.

---

## 🚀 Cara Menjalankan Proyek  

1. **Clone repositori**
   ```bash
   git clone https://github.com/BertrandOscarSaputra/AI_project.git
   cd AI_project
   ```

2. **Install dependensi**
   ```bash
   pip install -r requirements.txt
   ```

3. **Jalankan aplikasi web**
   ```bash
   python app.py
   ```

4. **Akses di browser**
   ```
   http://localhost:5000
   ```

---

## 🧪 Evaluasi Model  
Untuk menguji performa model, jalankan:
```bash
python evaluate.py
```
Script ini akan menampilkan hasil akurasi, loss, serta metrik evaluasi lainnya.

---

## 📸 Contoh Output  
| Input | Prediksi | Confidence |
|-------|-----------|-------------|
| `sample_01.png` | 🟩 **Benign** | 97.2% |
| `sample_02.png` | 🟥 **Malignant** | 93.8% |

---
