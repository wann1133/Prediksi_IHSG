# 📈 IHSG Forecasting App — Machine Learning Prediction Dashboard

Aplikasi interaktif berbasis **Streamlit** untuk memprediksi **Indeks Harga Saham Gabungan (IHSG)** menggunakan **Machine Learning (XGBoost)** dan indikator teknikal seperti **RSI**, **MACD**, dan **Bollinger Bands**.

---

## 🚀 Fitur Utama

✅ Prediksi multi-horizon:
- 1 Hari  
- 1 Minggu (5 Hari Bursa)  
- 1 Bulan (~22 Hari)  
- 3 Bulan (~66 Hari)  
- 6 Bulan (~126 Hari)  
- 1 Tahun (~252 Hari)  

✅ Analisis Teknis Otomatis  
Menggunakan indikator RSI, MACD, dan Bollinger Bands yang dihitung langsung dari data historis IHSG.

✅ Grafik Interaktif  
Visualisasi prediksi vs harga aktual menggunakan **Plotly**, lengkap dengan **confidence interval**.

✅ Fitur Download  
Hasil prediksi bisa langsung diunduh dalam format **CSV** maupun **Excel**.

✅ Mode Gelap / Terang Otomatis  
Menyesuaikan dengan tema pengguna Streamlit (dark/light mode).

---

## ⚙️ Arsitektur Model

Model menggunakan **XGBoost Regressor** yang dilatih untuk memprediksi **return harian IHSG**.  
Fitur yang digunakan mencakup:
- Return & Log Return
- Lagged Returns (1–20 hari)
- Rolling Means / Std
- Indikator RSI, MACD, Bollinger Bands
- Fitur kalender (day of week, month)

Model dilatih dan disimpan sebagai:
artifacts/model_xgb.joblib
artifacts/metadata.json


---

## 💻 Cara Menjalankan di Lokal

### 1️⃣ Clone repository
```bash
git clone https://github.com/<username>/<repo-name>.git
cd <repo-name>

### 2️⃣ Instalasi dependensi
```bash
pip install -r requirements.txt

### 3️⃣ Jalankan aplikasi
streamlit run app_streamlit.py