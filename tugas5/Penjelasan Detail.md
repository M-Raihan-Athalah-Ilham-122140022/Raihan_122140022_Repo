**Link Chat Gemini:** https://gemini.google.com/share/58b17c69a7a0

# Perbandingan Sistem rPPG: Non-Real-Time vs Real-Time Enhanced

## 1. Pemrosesan
- **Non-real-time:** Proses dilakukan setelah video selesai; tidak ada feedback langsung.
- **Real-time enhanced:** Proses streaming frame-by-frame menggunakan sliding window; BPM diperbarui langsung.

## 2. ROI Detection
- **Non-real-time:** Crop wajah statis, banyak noise (rambut/background).
- **Real-time enhanced:** Landmark Face Mesh → ROI pipi & dahi yang mengikuti gerakan wajah (dynamic tracking).

## 3. Ekstraksi Sinyal
- **Non-real-time:** Hanya memakai green channel.
- **Real-time enhanced:** Metode POS (menggunakan RGB) untuk memisahkan sinyal darah dari noise.

## 4. Filtering
- **Non-real-time:** Bandpass filter fixed (0.67–4 Hz).
- **Real-time enhanced:** Adaptive bandpass yang menyesuaikan bandwidth sesuai BPM terbaru.

## 5. Quality Assessment
- **Non-real-time:** Tidak ada penilaian kualitas sinyal.
- **Real-time enhanced:** Menghitung SNR, motion detection, dan quality score secara real-time.

## 6. Estimasi BPM
- **Non-real-time:** Peak FFT terbesar → mudah salah jika ada noise.
- **Real-time enhanced:** Smart peak detection (threshold, prominence, distance) + smoothing (moving average).

## 7. Visualisasi
- **Non-real-time:** Output BPM saja atau plot sederhana.
- **Real-time enhanced:** Time-domain, frequency-domain, indikator kualitas, motion warning, buffer status.

## 8. Robustness
- **Non-real-time:** Sensitif terhadap gerakan dan cahaya; cocok di kondisi laboratorium.
- **Real-time enhanced:** Tahan motion, pencahayaan tidak stabil, dan variasi pengguna → jauh lebih praktis di dunia nyata.



# Penjelasan Sederhana Sistem rPPG

## 1. BPM Dihitung Berdasarkan Apa?

### Konsep Dasar:
Ketika jantung berdetak, darah dipompa ke seluruh tubuh termasuk **wajah**. Setiap kali darah mengalir ke wajah, warna kulit berubah **sedikit** (tidak terlihat mata telanjang). Perubahan ini mengikuti **ritme detak jantung**.

### Proses Perhitungan:

#### Step 1: Tangkap Perubahan Warna
- Kamera merekam wajah setiap detik (30 frame/detik)
- Program mengambil **area pipi dan dahi** (paling banyak pembuluh darah)
- Membaca nilai warna **hijau (Green channel)** karena paling sensitif terhadap darah

#### Step 2: Buat Grafik Sinyal
- Nilai hijau dari setiap frame disimpan
- Contoh: Frame 1=120, Frame 2=122, Frame 3=119, dst.
- Membentuk **gelombang naik-turun** yang mengikuti detak jantung

#### Step 3: Analisis Frekuensi (FFT)
- Menggunakan **Fast Fourier Transform (FFT)** 
- FFT mengubah gelombang waktu menjadi frekuensi
- Mencari frekuensi yang **paling dominan** (paling kuat)

#### Step 4: Konversi ke BPM
```
BPM = Frekuensi Dominan × 60

Contoh:
- Frekuensi dominan = 1.2 Hz (1.2 kali per detik)
- BPM = 1.2 × 60 = 72 BPM
```

---

## 2. Fungsi Gambar Landmark Segitiga di Pipi

### Apa Itu Landmark?
- **MediaPipe Face Mesh** mendeteksi **478 titik** di wajah
- Titik-titik ini membentuk peta 3D wajah
- Pilih titik-titik di area **pipi** untuk dijadikan ROI

### Kenapa Berbentuk Segitiga/Polygon?
```
Titik 1 ●────────● Titik 2
        │        │
        │  PIPI  │  ← Area ini yang diambil warnanya
        │        │
Titik 3 ●────────● Titik 4
```

- Menghubungkan beberapa titik landmark membentuk **area tertutup**
- Program mengambil **semua pixel di dalam area** ini
- Menghitung **rata-rata warna hijau** dari semua pixel

### Fungsi Utama:
1. **Visualisasi**: Agar Anda tahu area mana yang sedang dianalisis
2. **ROI (Region of Interest)**: Membatasi area pengambilan data
3. **Tracking**: Jika wajah bergerak, ROI ikut bergerak

### Kenapa Pipi Kiri & Kanan?
- Area pipi punya **banyak pembuluh darah kapiler**
- Lebih tipis dari area lain (dahi, hidung)
- Perubahan warna dari aliran darah **paling terlihat** di sini
- Menggunakan 2 area = lebih **stabil dan akurat**

---

## 3. Visualisasi Data Berupa Grafik - Untuk Apa?

### A. Grafik Time Domain (Atas)
```
Amplitude
    │     ╱╲    ╱╲    ╱╲
    │    ╱  ╲  ╱  ╲  ╱  ╲
────┼───╱────╲╱────╲╱────╲─── Time
    │
```

**Menampilkan:** Sinyal detak jantung dalam bentuk gelombang

**Fungsi:**
- ✅ **Validasi Visual**: Anda bisa lihat apakah gelombangnya teratur (bagus) atau berantakan (jelek)
- ✅ **Deteksi Masalah**: Jika grafiknya acak = ada gangguan (gerakan, cahaya buruk)
- ✅ **Real-time Monitoring**: Lihat langsung kualitas sinyal yang ditangkap

**Interpretasi:**
- Gelombang **teratur & smooth** = Pengukuran bagus ✓
- Gelombang **berantakan** = Ada noise/gangguan ✗

---

### B. Grafik Frequency Domain (Bawah)
```
Power
    │          ★
    │         ╱ ╲
    │   ╱╲   ╱   ╲   ╱╲
────┼──╱──╲─╱─────╲─╱──╲─── Frequency (Hz)
    │       ↑ Peak ini = Detak jantung
```

**Menampilkan:** Kekuatan setiap frekuensi dalam sinyal

**Fungsi:**
- ✅ **Identifikasi BPM**: Peak tertinggi = frekuensi detak jantung
- ✅ **Validasi Akurasi**: Peak yang tajam & jelas = hasil akurat
- ✅ **Deteksi Anomali**: Jika ada multiple peaks = bisa ada noise atau aritmia

**Interpretasi:**
- **1 peak dominan** = BPM jelas dan akurat ✓
- **Banyak peak** = Sinyal terganggu atau detak tidak teratur ✗

---

### Kenapa Perlu 2 Grafik?

| Grafik | Informasi | Analogi |
|--------|-----------|---------|
| **Time Domain** | "Bagaimana bentuk detak jantung" | Seperti melihat **rekaman ECG** |
| **Frequency Domain** | "Berapa kali per detik jantung berdetak" | Seperti **stopwatch otomatis** |

**Manfaat untuk Tugas:**
- Menunjukkan sistem bekerja dengan **transparansi**
- Memudahkan **debugging** saat development
- Bukti **validasi ilmiah** untuk laporan
- Terlihat **profesional** untuk presentasi

---

## 4. Method POS - Apa Itu?

### Kepanjangan:
**POS = Plane-Orthogonal-to-Skin**

### Penjelasan Sederhana:

#### Masalah dengan Metode Green Channel Biasa:
- Hanya menggunakan 1 warna (hijau)
- Rentan terhadap **perubahan cahaya**
- Jika Anda bergerak, sinyal jadi berantakan

#### Solusi POS:
Menggunakan **kombinasi ketiga warna** (Red, Green, Blue) dengan perhitungan matematika khusus.

### Analogi Mudah:

**Metode Green (lama):**
```
Seperti mendengar musik hanya dari 1 speaker
→ Jika speaker itu kena gangguan, musik rusak total
```

**Metode POS (baru):**
```
Seperti mendengar dari 3 speaker (R, G, B)
→ Sistem pintar yang menggabungkan ketiganya
→ Jika 1 speaker terganggu, 2 lainnya kompensasi
```

### Cara Kerja POS:

1. **Ambil 3 Warna**: Red, Green, Blue dari kulit wajah
2. **Normalisasi**: Buat setiap warna punya skala yang sama
3. **Proyeksi Matematis**: Buang komponen yang berhubungan dengan **cahaya**
4. **Hasil**: Sinyal yang HANYA berisi **informasi detak jantung**

### Rumus Sederhana:
```
POS Signal = Kombinasi_Cerdas(R, G, B) - Komponen_Cahaya
```

### Keunggulan:
- ✅ **Lebih tahan gerakan** (motion artifact)
- ✅ **Lebih tahan perubahan cahaya**
- ✅ **Akurasi 30-40% lebih baik** dari green channel
- ✅ **Standard di penelitian modern**

---

## 5. SNR (Signal-to-Noise Ratio) - Apa Itu?

### Definisi Sederhana:
**SNR = Perbandingan antara sinyal asli vs noise (gangguan)**

### Analogi Mudah:

**Radio:**
```
SNR Tinggi = Suara radio jernih, musik terdengar jelas
SNR Rendah = Banyak "kresek-kresek", musik tertutup noise
```

**rPPG:**
```
SNR Tinggi = Sinyal detak jantung jelas, BPM akurat
SNR Rendah = Sinyal tertutup noise, BPM tidak akurat
```

### Cara Menghitung:

#### Step 1: Identifikasi Signal
- Frekuensi 0.67-4.0 Hz = Detak jantung (signal)
- Di luar range itu = Noise

#### Step 2: Hitung Power
```
Signal Power = Kekuatan di range 0.67-4.0 Hz
Noise Power = Kekuatan di luar range itu
```

#### Step 3: Rumus SNR
```
SNR (dB) = 10 × log₁₀(Signal Power / Noise Power)
```

### Interpretasi SNR:

| SNR (dB) | Kualitas | Arti | Akurasi BPM |
|----------|----------|------|-------------|
| **> 15 dB** | 🟢 Excellent | Signal 30× lebih kuat dari noise | Sangat akurat |
| **10-15 dB** | 🟡 Good | Signal 10× lebih kuat | Akurat |
| **5-10 dB** | 🟠 Fair | Signal hanya 3× lebih kuat | Cukup akurat |
| **< 5 dB** | 🔴 Poor | Signal hampir tertutup noise | Tidak akurat |

### Fungsi SNR dalam Sistem:

1. **Quality Indicator**: Tahu kapan hasil bisa dipercaya
2. **Auto-Rejection**: Bisa buang data jelek otomatis
3. **User Feedback**: Kasih tahu user untuk diam/perbaiki cahaya
4. **Scientific Validation**: Standar metrik dalam penelitian

### Contoh Praktis:

```
Situasi 1: User diam, cahaya bagus
→ SNR = 18 dB
→ Quality = 90%
→ BPM = 72 (dapat dipercaya ✓)

Situasi 2: User bergerak-gerak
→ SNR = 4 dB  
→ Quality = 20%
→ BPM = 145 (jangan dipercaya ✗)
```

---

## Ringkasan Hubungan Semua Komponen

```
1. Kamera menangkap wajah
           ↓
2. Landmark menandai PIPI (ROI)
           ↓
3. Ambil warna dari area pipi (Green/POS)
           ↓
4. Buat gelombang sinyal → Grafik Time Domain
           ↓
5. Filter noise (bandpass)
           ↓
6. FFT ubah ke frekuensi → Grafik Frequency Domain
           ↓
7. Cari peak tertinggi = Frekuensi jantung
           ↓
8. Kalikan × 60 = BPM
           ↓
9. Hitung SNR = Cek kualitas
           ↓
10. Tampilkan BPM + Quality di layar
```

---

## Kesimpulan Sederhana

- **BPM** = Dihitung dari frekuensi gelombang perubahan warna kulit menggunakan FFT
- **Landmark Segitiga** = Menandai area pipi untuk mengambil data warna
- **Grafik** = Untuk validasi visual bahwa sistem bekerja dengan baik
- **POS** = Metode pintar yang pakai 3 warna sekaligus, lebih akurat dari 1 warna
- **SNR** = Indikator kualitas sinyal, seperti bar sinyal HP


Semua komponen ini bekerja sama untuk mengukur detak jantung Anda **tanpa sensor**, hanya dengan kamera! 🎥❤️
