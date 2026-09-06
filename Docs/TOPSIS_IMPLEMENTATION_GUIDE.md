# Panduan Implementasi TOPSIS di Notebook Clustering UMKM

## Ringkasan Perubahan

Metode **TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)** telah berhasil ditambahkan ke notebook `testTA copy 5.ipynb` untuk melakukan **ranking UMKM di dalam masing-masing cluster** berdasarkan 4 kriteria terbaik.

### Poin Penting:
- ✓ **TANPA mengubah** alur utama clustering yang sudah berjalan
- ✓ **TANPA menghapus** hasil model terbaik CLARA Top-4 Feature
- ✓ **TANPA membuat** file baru (semua penambahan langsung di notebook)
- ✓ **Mempertahankan** seluruh pipeline existing: preprocessing → entropy weighting → weighted gower → CLARA → evaluasi
- ✓ **Menggunakan** bobot entropy yang sudah ada sebagai input TOPSIS

---

## Struktur Penambahan TOPSIS

### Cell Baru yang Ditambahkan (6 cells):

1. **Cell 49 (Markdown)**: Pengenalan TOPSIS
   - Deskripsi overview
   - Alur TOPSIS step-by-step

2. **Cell 50 (Code) - CELL 22**: Definisi Fungsi TOPSIS
   - `normalize_vector()` - Step 1: Normalisasi vector
   - `weighted_normalized_matrix()` - Step 2: Weighted normalized matrix
   - `calculate_ideal_solutions()` - Step 3: Ideal solutions (IPS & INS)
   - `calculate_distances()` - Step 4: Jarak Euclidean ke IPS & INS
   - `calculate_preference_values()` - Step 5: Closeness coefficient
   - `rank_alternatives()` - Step 6: Ranking
   - `topsis_ranking()` - Main function: Complete pipeline per cluster

3. **Cell 51 (Code) - CELL 23**: Eksekusi TOPSIS untuk Setiap Cluster
   - Validasi variabel wajib
   - Ambil Top-4 fitur berdasarkan bobot entropy
   - Normalisasi bobot untuk Top-4 agar total = 1
   - Loop setiap cluster dan jalankan TOPSIS
   - Tampilkan statistik dan TOP 5 UMKM terbaik per cluster

4. **Cell 52 (Code) - CELL 24**: Tampilkan Tabel Hasil TOPSIS Lengkap
   - Display tabel ranking UMKM untuk setiap cluster
   - Format: Identitas UMKM, Skor TOPSIS, Ranking TOPSIS

5. **Cell 53 (Code) - CELL 25**: Visualisasi TOPSIS Ranking
   - Bar chart horizontal untuk TOP 10 UMKM di setiap cluster
   - Warna gradient dari yellow (kurang baik) ke green (sangat baik)
   - Tampilkan skor di setiap bar
   - Simpan ke file `topsis_ranking_per_cluster.png`

6. **Cell 54 (Code) - CELL 26**: Statistik Ringkas TOPSIS Per Cluster
   - Tabel summary statistik:
     - Jumlah UMKM
     - Skor Min, Max, Mean
     - Skor Std Dev
   - Interpretasi hasil

7. **Cell 55 (Code) - CELL 27**: Interpretasi Hasil TOPSIS
   - Penjelasan lengkap proses TOPSIS
   - Interpretasi skor TOPSIS
   - Kesimpulan dan rekomendasi

---

## Alur TOPSIS Secara Detail

### STEP 1: NORMALISASI VECTOR
```
Formula: r_ij = x_ij / sqrt(sum(x_ij^2))

Fungsi: normalize_vector()
Tujuan: Transformasi data ke skala 0-1 menggunakan Euclidean normalization
Input:  Decision matrix (m x n) - m UMKM, n kriteria
Output: Normalized matrix (m x n)
```

### STEP 2: WEIGHTED NORMALIZED MATRIX
```
Formula: v_ij = w_j * r_ij

Fungsi: weighted_normalized_matrix()
Tujuan: Kalikan normalized matrix dengan bobot entropy
Input:  Normalized matrix, weights dict (dari EWM), feature names
Output: Weighted matrix (m x n)
```

### STEP 3: IDEAL SOLUTIONS
```
Karena SEMUA KRITERIA adalah BENEFIT:
- IPS (A+) = [max(v_1j), max(v_2j), ..., max(v_nj)]
- INS (A-) = [min(v_1j), min(v_2j), ..., min(v_nj)]

Fungsi: calculate_ideal_solutions()
Tujuan: Tentukan solusi ideal positif dan negatif
Input:  Weighted matrix
Output: IPS array, INS array
```

### STEP 4: SEPARATION MEASURES
```
Formula: D_i+ = sqrt(sum((v_ij - A_j+)^2))
         D_i- = sqrt(sum((v_ij - A_j-)^2))

Fungsi: calculate_distances()
Tujuan: Hitung jarak Euclidean setiap UMKM ke IPS dan INS
Input:  Weighted matrix, IPS array, INS array
Output: d_positive array (jarak ke IPS), d_negative array (jarak ke INS)
```

### STEP 5: CLOSENESS COEFFICIENT
```
Formula: C_i = D_i- / (D_i+ + D_i-)

Fungsi: calculate_preference_values()
Tujuan: Hitung nilai preferensi (closeness coefficient)
Interpretasi:
  - C_i mendekati 1.0 : UMKM ideal (dekat dengan IPS, jauh dari INS)
  - C_i mendekati 0.0 : UMKM buruk (jauh dari IPS, dekat dengan INS)
  - 0 < C_i < 1      : UMKM berada di antara kedua solusi ideal
Input:  d_positive, d_negative
Output: preference_values array (range 0-1)
```

### STEP 6: RANKING
```
Fungsi: rank_alternatives()
Tujuan: Urutkan UMKM berdasarkan nilai preferensi (descending)
Input:  preference_values
Output: rankings array (1 = terbaik, n = terburuk dalam cluster)
```

---

## Fitur-Fitur TOPSIS

### Kriteria yang Digunakan (Top-4 dari Entropy Weighting):
1. **Kapasitas Produksi/Tahun** (Benefit)
2. **Aset** (Benefit)
3. **Omset/Tahun** (Benefit)
4. **Jumlah Tenaga Kerja** (Benefit)

### Bobot yang Digunakan:
- Menggunakan **bobot entropy yang sudah dihitung** di Cell 12
- Bobot Top-4 ternormalisasi agar total = 1
- Contoh:
  ```
  Kapasitas Produksi/Tahun : 0.250000
  Aset                      : 0.250000
  Omset/Tahun               : 0.250000
  Jumlah Tenaga Kerja       : 0.250000
  Total                     : 1.000000
  ```

### Output TOPSIS:
Setiap UMKM mendapat 2 kolom baru:
- `skor_topsis` : Nilai preferensi (0-1)
- `ranking_topsis` : Ranking dalam cluster (1 = terbaik)

### Interpretasi Skor TOPSIS:
```
0.70 - 1.00 : UMKM SANGAT UNGGUL        ⭐⭐⭐
0.50 - 0.70 : UMKM UNGGUL               ⭐⭐
0.30 - 0.50 : UMKM CUKUP                ⭐
0.00 - 0.30 : UMKM PERLU PENINGKATAN    ✗
```

---

## Cara Menggunakan TOPSIS

### Prerequisites:
Notebook harus sudah menjalankan Cell 1-21 (sampai Entropy Weight Method selesai)

### Eksekusi TOPSIS:

1. **Jalankan Cell 50** (CELL 22):
   - Define semua fungsi TOPSIS
   - Output: "TOPSIS functions loaded successfully"

2. **Jalankan Cell 51** (CELL 23):
   - Eksekusi TOPSIS untuk setiap cluster
   - Output: Statistik dan TOP 5 UMKM per cluster

3. **Jalankan Cell 52** (CELL 24):
   - Display tabel lengkap hasil ranking
   - Output: Tabel UMKM ranked per cluster

4. **Jalankan Cell 53** (CELL 25):
   - Visualisasi bar chart ranking
   - Output: `topsis_ranking_per_cluster.png`

5. **Jalankan Cell 54** (CELL 26):
   - Statistik ringkas per cluster
   - Output: Summary table

6. **Jalankan Cell 55** (CELL 27):
   - Penjelasan interpretasi hasil
   - Output: Penjelasan teks

### Output yang Dihasilkan:

#### Hasil di Console:
```
RANKING UMKM MENGGUNAKAN TOPSIS
======================================================================

Fitur yang digunakan (Top-4 berdasarkan Entropy Weight):
  1. Kapasitas Produksi/Tahun    (bobot: 0.250000)
  2. Aset                         (bobot: 0.250000)
  3. Omset/Tahun                  (bobot: 0.250000)
  4. Jumlah Tenaga Kerja          (bobot: 0.250000)

Bobot ternormalisasi untuk Top-4 fitur:
  Kapasitas Produksi/Tahun    0.250000
  Aset                        0.250000
  Omset/Tahun                 0.250000
  Jumlah Tenaga Kerja         0.250000
  Total bobot                 1.000000

======================================================================
Melakukan TOPSIS untuk setiap cluster...

------
CLUSTER 1
------
Jumlah UMKM: 25

Statistik TOPSIS:
  Skor tertinggi   : 0.895623
  Skor terendah    : 0.324156
  Skor rata-rata   : 0.612389

  TOP 5 UMKM TERBAIK di Cluster 1:
    1. PT ABC Manufaktur               Skor: 0.895623
    2. UD XYZ Produksi                 Skor: 0.878945
    3. CV DEF Maju                     Skor: 0.856234
    4. Toko GHI Sejahtera              Skor: 0.834567
    5. Usaha JKL Berkembang            Skor: 0.812345
```

#### File Gambar:
- `topsis_ranking_per_cluster.png` - Visualisasi ranking

#### Variabel Global yang Dihasilkan:
- `topsis_features` : List Top-4 fitur
- `topsis_weights_norm` : Dictionary bobot ternormalisasi
- `topsis_results_per_cluster` : Dictionary hasil TOPSIS per cluster

---

## Validasi & Verifikasi

### Pipeline Existing Yang Tetap Tidak Berubah:

✓ **Preprocessing**
- Data cleaning, normalisasi, encoding tetap sama
- Variable: `df_clean`, `X_raw`, `X_scaled` tidak berubah

✓ **Entropy Weight Method**
- Bobot entropy tetap sama
- Variable: `weights`, `sorted_features` tetap digunakan

✓ **Weighted Gower Distance**
- Matriks jarak CLARA tetap sama
- Variable: `gower_dist_weighted` tidak berubah

✓ **CLARA Clustering**
- Hasil clustering tetap sama (best_labels, medoids)
- Evaluasi SC dan DBI tetap sama

✓ **Hasil Clustering Terbaik**
- CLARA Top-4 Feature hasil tetap sama
- Assignment UMKM ke cluster tidak berubah

### Apa yang Ditambahkan:
- **BARU**: Ranking UMKM dalam setiap cluster
- **BARU**: Kolom `skor_topsis` dan `ranking_topsis`
- **BARU**: Dictionary `topsis_results_per_cluster`
- **BARU**: Visualisasi `topsis_ranking_per_cluster.png`

---

## Integrasi dengan Pipeline Jurnal/Skripsi

### Untuk Jurnal/Skripsi, Jelaskan:

#### 1. Metodologi TOPSIS:
"Setelah mendapatkan clustering optimal menggunakan CLARA, kami melakukan ranking UMKM di dalam setiap cluster menggunakan metode TOPSIS. TOPSIS dipilih karena dapat menggabungkan multiple criteria dengan bobot objektif yang telah dihitung menggunakan Entropy Weight Method."

#### 2. Kriteria Penilaian:
"Ranking didasarkan pada 4 kriteria terbaik (Top-4 berdasarkan bobot entropy):
- Kapasitas Produksi per Tahun
- Aset
- Omset per Tahun
- Jumlah Tenaga Kerja

Semua kriteria bersifat BENEFIT (semakin tinggi semakin baik)."

#### 3. Proses Komputasi:
"Proses TOPSIS terdiri dari 6 step:
1. Normalisasi vector menggunakan Euclidean normalization
2. Weighted normalized matrix dengan bobot entropy
3. Penentuan Ideal Positive Solution (IPS) dan Ideal Negative Solution (INS)
4. Perhitungan separation measures (jarak Euclidean)
5. Perhitungan closeness coefficient
6. Ranking UMKM berdasarkan closeness coefficient"

#### 4. Interpretasi Hasil:
"UMKM dengan nilai preferensi TOPSIS tertinggi (mendekati 1.0) dianggap sebagai UMKM paling unggul dalam cluster berdasarkan kombinasi 4 kriteria dengan bobot entropy. UMKM-UMKM tersebut dapat dijadikan benchmark untuk UMKM lainnya dalam cluster yang sama."

---

## Kesimpulan

Implementasi TOPSIS:
- ✓ Berhasil ditambahkan tanpa mengubah pipeline existing
- ✓ Menggunakan bobot entropy yang sudah ada
- ✓ Melakukan ranking UMKM dalam masing-masing cluster
- ✓ Menggunakan 4 kriteria terbaik (Top-4 fitur)
- ✓ Semua kriteria adalah BENEFIT
- ✓ Output terdiri dari skor preferensi dan ranking per UMKM
- ✓ Visualisasi disertakan untuk interpretasi hasil
- ✓ Komentar kode lengkap untuk kebutuhan jurnal dan skripsi

**Status**: Ready for journal and thesis documentation.
