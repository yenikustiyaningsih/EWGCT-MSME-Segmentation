SAMPLE OUTPUT - TOPSIS Ranking untuk UMKM Clustering
=====================================================

## Contoh Output yang Akan Dihasilkan

Ketika menjalankan Cell 51 (CELL 23) - Eksekusi TOPSIS:

---

### CONSOLE OUTPUT EXAMPLE:

```
RANKING UMKM MENGGUNAKAN TOPSIS
======================================================================

Fitur yang digunakan (Top-4 berdasarkan Entropy Weight):
  1. Kapasitas Produksi/Tahun    (bobot: 0.250000)
  2. Aset                        (bobot: 0.250000)
  3. Omset/Tahun                 (bobot: 0.250000)
  4. Jumlah Tenaga Kerja         (bobot: 0.250000)

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
    1. PT ABC Manufaktur Indonesia         Skor: 0.895623
    2. UD XYZ Produksi Maju               Skor: 0.878945
    3. CV DEF Usaha Sejahtera             Skor: 0.856234
    4. Toko GHI Retail                    Skor: 0.834567
    5. Usaha JKL Berkembang               Skor: 0.812345

------
CLUSTER 2
------
Jumlah UMKM: 18

Statistik TOPSIS:
  Skor tertinggi   : 0.912456
  Skor terendah    : 0.412389
  Skor rata-rata   : 0.687234

  TOP 5 UMKM TERBAIK di Cluster 2:
    1. PT MNO Manufaktur Premium          Skor: 0.912456
    2. CV PQR Produksi Modern             Skor: 0.891234
    3. UD STU Usaha Maju                  Skor: 0.867890
    4. Toko VWX Retail Sejahtera          Skor: 0.845612
    5. Usaha YZ Berkembang Pesat           Skor: 0.823456

======================================================================
TOPSIS ranking selesai!
Total cluster yang diproses: 2
```

---

### TABEL OUTPUT - CELL 52 (CELL 24):

```
TABEL RANKING UMKM PER CLUSTER
======================================================================

------
CLUSTER 1 (25 UMKM)
------

|    | Identitas UMKM               | Skor TOPSIS | Ranking |
|----|------------------------------|-------------|---------|
|  0 | PT ABC Manufaktur Indonesia  |   0.895623  |    1    |
|  1 | UD XYZ Produksi Maju         |   0.878945  |    2    |
|  2 | CV DEF Usaha Sejahtera       |   0.856234  |    3    |
|  3 | Toko GHI Retail              |   0.834567  |    4    |
|  4 | Usaha JKL Berkembang         |   0.812345  |    5    |
|  5 | PT LMN Perdagangan           |   0.789123  |    6    |
|  6 | UD OPQ Produksi Sedang       |   0.765432  |    7    |
|  7 | CV RST Usaha Sedang          |   0.743210  |    8    |
|  8 | Toko UVW Retail Kecil        |   0.621098  |    9    |
|  9 | Usaha XYZ Pemula             |   0.524156  |   10    |
| 10 | PT ABC Kecil                 |   0.512345  |   11    |
...
| 24 | Usaha ZZZ Mikro              |   0.324156  |   25    |

------
CLUSTER 2 (18 UMKM)
------

|    | Identitas UMKM               | Skor TOPSIS | Ranking |
|----|------------------------------|-------------|---------|
|  0 | PT MNO Manufaktur Premium    |   0.912456  |    1    |
|  1 | CV PQR Produksi Modern       |   0.891234  |    2    |
|  2 | UD STU Usaha Maju            |   0.867890  |    3    |
|  3 | Toko VWX Retail Sejahtera    |   0.845612  |    4    |
|  4 | Usaha YZ Berkembang Pesat    |   0.823456  |    5    |
...
| 17 | Usaha KL Pemula              |   0.412389  |   18    |
```

---

### STATISTIK OUTPUT - CELL 54 (CELL 26):

```
STATISTIK TOPSIS SCORE PER CLUSTER
================================================================================

|  Cluster | Jumlah_UMKM | Skor_Min | Skor_Max | Skor_Mean | Skor_Std |
|----------|-------------|----------|----------|-----------|----------|
|    1     |     25      |  0.324   |  0.896   |   0.612   |  0.138   |
|    2     |     18      |  0.412   |  0.912   |   0.687   |  0.125   |
|    3     |     22      |  0.389   |  0.884   |   0.634   |  0.142   |
|    4     |     20      |  0.398   |  0.923   |   0.675   |  0.135   |

Interpretasi:
  - Skor tinggi (0.7-1.0) : UMKM sangat unggul dalam cluster
  - Skor sedang (0.5-0.7) : UMKM unggul dalam cluster
  - Skor rendah (0.0-0.5) : UMKM perlu peningkatan
  
  - Std Dev tinggi       : Cluster heterogen (kualitas UMKM sangat bervariasi)
  - Std Dev rendah       : Cluster homogen (kualitas UMKM serupa)
```

---

### VISUALISASI OUTPUT - CELL 53 (CELL 25):

File: `topsis_ranking_per_cluster.png`

```
Keterangan:
- 4 subplots (untuk 4 cluster)
- Setiap subplot menampilkan TOP 10 UMKM
- Bar horizontal dengan gradien warna:
  * Kuning (0.3-0.5) : performance kurang
  * Oranye (0.5-0.7) : performance sedang
  * Hijau (0.7-1.0) : performance baik
- Value label di setiap bar (skor TOPSIS)
- Ranking dan nama UMKM di y-axis
```

Visual representation (simplified):
```
TOP 10 UMKM Cluster 1
|#1 - PT ABC Manufaktur     |▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 0.8956
|#2 - UD XYZ Produksi       |▓▓▓▓▓▓▓▓▓▓▓▓▓   0.8789
|#3 - CV DEF Usaha          |▓▓▓▓▓▓▓▓▓▓▓▓    0.8562
|#4 - Toko GHI Retail       |▓▓▓▓▓▓▓▓▓▓▓     0.8346
|#5 - Usaha JKL Berkembang  |▓▓▓▓▓▓▓▓▓▓      0.8123
...
```

---

## INTERPRETASI HASIL

### Untuk UMKM dengan Skor Tinggi (0.7-1.0):
"UMKM ini termasuk kelompok paling unggul dalam cluster. Memiliki kombinasi optimal antara kapasitas produksi, aset, omset, dan tenaga kerja. Dapat dijadikan benchmark untuk UMKM lainnya dalam cluster."

### Untuk UMKM dengan Skor Sedang (0.5-0.7):
"UMKM ini memiliki performa yang baik namun masih ada ruang untuk peningkatan. Dapat mempelajari best practice dari UMKM dengan skor lebih tinggi dalam cluster yang sama."

### Untuk UMKM dengan Skor Rendah (0.3-0.5):
"UMKM ini perlu meningkatkan performa terutama dalam hal kapasitas produksi, aset, dan omset. Rekomendasi untuk fokus pada peningkatan efisiensi operasional dan expansion."

### Untuk UMKM dengan Skor Sangat Rendah (0.0-0.3):
"UMKM ini memerlukan perhatian khusus dan program pembinaan. Nilai preferensi rendah menunjukkan multiple challenges yang perlu ditangani secara menyeluruh."

---

## ANALISIS CLUSTER-LEVEL

### Dari Statistik yang Dihasilkan:

**Jika Mean Score Tinggi (>0.65):**
- Cluster ini terdiri dari UMKM-UMKM yang relatif berkualitas tinggi
- Performa rata-rata baik
- Fokus pada continuous improvement

**Jika Std Dev Tinggi (>0.13):**
- Keragaman kualitas tinggi dalam cluster
- Ada UMKM top performer dan yang masih struggling
- Strategi: mentoring dari top UMKM ke yang lainnya

**Jika Std Dev Rendah (<0.10):**
- Homogenitas cluster tinggi
- Semua UMKM memiliki kualitas serupa
- Dapat diberikan program pengembangan yang seragam

---

## INTEGRASI DENGAN CLUSTERING RESULT

### Hubungan dengan CLARA Clustering:
1. CLARA membagi UMKM menjadi clusters berdasarkan similarity
2. TOPSIS melakukan ranking UMKM DALAM setiap cluster
3. Hasil gabungan: cluster assignment + internal ranking

### Manfaat Kombinasi:
✓ Segmentasi + Ranking = Complete Classification
✓ Identifikasi best practice per segment
✓ Targeted intervention programs
✓ Resource allocation optimization

### Contoh Use Case:
"Cluster 1 terdiri dari UMKM menengah-besar. Ranking TOPSIS menunjukkan PT ABC Manufaktur (ranking 1) adalah model yang bagus. Kami dapat menggunakan PT ABC sebagai:
- Pilot project untuk innovation
- Mentor untuk UMKM lain di cluster
- Benchmark untuk KPI setting"

---

## CATATAN TEKNIS

### Input Requirements:
✓ best_labels (dari CLARA clustering)
✓ weights (dari Entropy Weight Method)
✓ df_scaled (data UMKM yang sudah ternormalisasi)
✓ sorted_features (urutan fitur berdasarkan bobot entropy)
✓ k_best (jumlah cluster)

### Processing:
- Per cluster loop
- TOPSIS calculation per UMKM
- Ranking assignment
- Result compilation

### Output Variables:
- `topsis_features` : List fitur yang digunakan (Top-4)
- `topsis_weights_norm` : Dict bobot ternormalisasi
- `topsis_results_per_cluster` : Dict hasil ranking per cluster

### File Generated:
- `topsis_ranking_per_cluster.png` : Visualisasi

---

## READY TO USE

Output ini akan dihasilkan ketika Anda menjalankan Cell 50-55 (CELL 22-27) dalam notebook setelah semua clustering selesai.

Gunakan output ini untuk:
1. Analisis UMKM terbaik per cluster
2. Identification of best practices
3. Benchmarking
4. Program targeting
5. Resource allocation
6. Journal/thesis documentation
