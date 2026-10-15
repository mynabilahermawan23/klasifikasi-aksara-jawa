# Klasifikasi Aksara Jawa Tulisan Tangan: ANN vs CNN vs Transfer Learning

Proyek ini melatih komputer untuk **mengenali 20 huruf aksara Jawa dari gambar tulisan tangan**. Kami membandingkan tiga jenis model:

- **ANN**: jaringan saraf biasa
- **CNN**: jaringan konvolusi
- **Transfer learning**: model MobileNetV2 yang sudah dilatih dengan jutaan foto ImageNet

Proyek ini dibuat untuk Mini Project mata kuliah Neural Network dan Deep Learning.

## Hasil singkat

Model terbaik adalah **CNN improved**, dengan akurasi test rata-rata **95,9% ± 1,7%** (3 kali training dengan seed berbeda) dan **93,5% ± 1,5%** pada Group K-Fold.

| Model | Jumlah parameter | Akurasi validasi | Akurasi test |
| --- | --- | --- | --- |
| ANN | 3.181.460 | 39,5% | 40,6% |
| ANN + BatchNorm | 3.182.612 | 55,2% | 59,8% |
| CNN materi kuliah | 1.186.100 | 57,6% | 58,0% |
| **CNN improved** | **425.332** | **98,6%** | **95,8%** |
| MobileNetV2 (dibekukan) | 2.283.604 | 87,7% | 88,1% |
| MobileNetV2 (fine-tune) | 2.283.604 | 92,0% | 93,5% |

*Angka di tabel adalah hasil Fase 1 (satu kali training per model).*

Temuan utama:

1. **CNN jauh lebih baik daripada ANN untuk gambar.** ANN hanya melihat posisi piksel, jadi huruf yang sedikit bergeser sudah dianggap berbeda. CNN mencari pola seperti garis dan lengkungan di mana pun letaknya.
2. **Model besar tidak selalu menang.** CNN improved mengalahkan MobileNetV2, padahal parameternya 5 kali lebih sedikit.
3. **Kesalahan model hanya berasal dari 2 tulisan tangan.** Dari 41 tulisan asli di data test, 39 dikenali benar. Kesalahannya terjadi pada huruf yang bentuknya mirip: ca → sa dan la → ha.

## Dataset

Kami memakai dataset **[Javanese Script (Aksara Jawa) Augmented](https://www.kaggle.com/datasets/hannanhunafa/javanese-script-aksara-jawa-augmented)** dari Kaggle (oleh hannanhunafa). Dataset tidak disertakan di repositori ini, jadi silakan unduh sendiri dari link tersebut.

- 20 kelas: ba, ca, da, dha, ga, ha, ja, ka, la, ma, na, nga, nya, pa, ra, sa, ta, tha, wa, ya
- Setiap gambar adalah varian (diputar, digeser, atau di-zoom) dari tulisan tangan asli. Satu tulisan asli punya sekitar 30 varian.

| Bagian | Jumlah gambar | Jumlah tulisan asli | Kegunaan |
| --- | --- | --- | --- |
| Train | 8.400 | 271 | Untuk belajar |
| Val | 2.400 → 2.239 setelah dibersihkan | 80 → 72 | Untuk memilih model |
| Test | 1.200 | 41 | Untuk ujian akhir |

**Kenapa ada gambar val yang dibuang?** Beberapa tulisan asli muncul di dua bagian data: 6 tulisan di train dan val, serta 2 tulisan di val dan test. Ini disebut *kebocoran data*, karena model jadi seperti "sudah melihat soal ujian". Sebanyak 161 gambar val yang bocor dibuang. Data train dan test tidak punya tulisan yang sama.

## Isi repositori

```
├── aksara_jawa_lengkap_fix.ipynb   # notebook utama (semua kode ada di sini)
├── README.md
└── output_final/                   # hasil yang dibuat otomatis oleh notebook
    ├── results.csv                 # hasil semua eksperimen
    ├── state.json                  # pengaturan dan riwayat training tiap model
    ├── seed_summary.csv            # hasil uji 3 seed
    ├── kfold_results.csv           # hasil Group K-Fold
    ├── test_per_source.csv         # akurasi per tulisan asli di data test
    ├── final_loss_accuracy.csv     # loss dan akurasi model final (train/val/test)
    ├── figures/                    # semua grafik (PNG)
    ├── tabel/                      # semua tabel (CSV)
    ├── history_per_epoch/          # loss dan akurasi per epoch untuk setiap model
    └── models/                     # bobot model yang sudah dilatih
```

Catatan: folder `models/` bisa berukuran besar. Kalau terlalu besar untuk GitHub, folder ini boleh tidak diunggah.

## Cara menjalankan

### 1. Siapkan Python

Kami memakai **Python 3.10.6** dan **TensorFlow 2.21.0** di Windows (tanpa GPU). Buka terminal di folder proyek, lalu jalankan:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install tensorflow pandas matplotlib scikit-learn ipykernel jupyter
```

Kalau muncul pesan *"running scripts is disabled"*, jalankan sekali perintah berikut, lalu ulangi perintah `Activate.ps1`:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 2. Siapkan dataset

Unduh dataset dari [Kaggle](https://www.kaggle.com/datasets/hannanhunafa/javanese-script-aksara-jawa-augmented), lalu taruh folder `train`, `val`, dan `test` (atau file `.zip`-nya) di satu folder. Setelah itu, ubah baris ini di notebook supaya sesuai dengan lokasi folder tersebut:

```python
SRC = r"D:\aksara_jawa"
```

### 3. Jalankan notebook

Buka `aksara_jawa_lengkap_fix.ipynb` di VS Code atau Jupyter, pilih kernel `.venv`, lalu klik **Restart → Run All**.

- Di komputer tanpa GPU, notebook otomatis memakai gambar 96×96 piksel dan membatasi jumlah epoch.
- Total waktu training di CPU sekitar **19,5 jam**: 16 jam untuk 20 eksperimen dan 3,5 jam untuk K-Fold. Sebaiknya dijalankan semalaman.
- Kalau proses terputus, cukup **Restart → Run All** lagi. Eksperimen yang sudah selesai tersimpan di `results.csv` dan akan dilewati.

## Alur notebook

| Bagian | Isi |
| --- | --- |
| 1. Setup | Import library dan mencari lokasi dataset |
| 2. Data dan eksplorasi | Menghitung jumlah gambar, menampilkan contoh, membersihkan kebocoran data, menyiapkan pipeline data |
| 3. Arsitektur | Membangun 5 jenis model |
| 4. Eksperimen | Training dalam 3 fase (lihat di bawah) |
| 5. Hasil dan analisis | Tabel hasil, kurva belajar, confusion matrix, contoh prediksi salah, evaluasi per tulisan asli |
| 6. Group K-Fold | Menguji model final dengan 3 pembagian data berbeda |
| 7. Ringkasan | Tabel dan draf kesimpulan untuk slide |
| 8. Simpan | Menyimpan model final, semua tabel, dan semua grafik |

## Arsitektur model

| Model | Susunan layer |
| --- | --- |
| ANN | Gambar dikecilkan ke 64×64 → Flatten → Dense 256 → Dense 128 → Softmax 20 |
| ANN + BatchNorm | Sama dengan ANN, ditambah BatchNormalization setelah tiap Dense |
| CNN materi kuliah | Conv 16 → MaxPool → Conv 32 → MaxPool → Flatten → Dense 64 → Softmax 20 |
| **CNN improved** | 4 blok (Conv → BatchNorm → ReLU → MaxPool) dengan 32/64/128/256 filter → GlobalAveragePooling → Dense 128 → Dropout 0,5 → Softmax 20 |
| MobileNetV2 | MobileNetV2 pretrained ImageNet → GlobalAveragePooling → Dropout → Softmax 20. Versi fine-tune membuka 30 layer terakhir. |

Semua model memakai aktivasi ReLU di hidden layer dan Softmax di output.

## Strategi training

- **Loss:** sparse categorical cross-entropy, dengan metrik accuracy.
- **Optimizer dasar:** Adam dengan learning rate 0,001.
- **EarlyStopping:** training berhenti kalau loss validasi tidak membaik selama 5 epoch, lalu bobot terbaik dipakai kembali.
- **ReduceLROnPlateau:** learning rate dikalikan 0,3 kalau loss validasi macet selama 2 epoch.
- **Fine-tuning MobileNetV2:** learning rate dibagi 10. Kalau hasilnya malah lebih buruk, bobot sebelum fine-tuning yang dipakai.

Eksperimen dilakukan dalam 3 fase:

1. **Fase 1 (screening):** 6 model dilatih dengan pengaturan dasar.
2. **Fase 2 (random search):** 2 arsitektur terbaik (CNN improved dan MobileNetV2 fine-tune) dicoba dengan 4 kombinasi pengaturan acak. Yang diubah adalah optimizer (Adam/SGD), learning rate, dropout, augmentasi, dan jumlah layer yang di-fine-tune. Hasilnya, tidak ada kombinasi yang mengalahkan pengaturan Fase 1.
3. **Fase 3 (uji kestabilan):** pengaturan terbaik dari tiap arsitektur dilatih ulang dengan 3 seed berbeda (1, 2, 3).

**Aturan penting:** model selalu dipilih berdasarkan **data validasi**. Data test hanya dipakai sekali di akhir supaya nilainya jujur.

## Hasil uji kestabilan (Fase 3)

| Model | Validasi | Test | F1-macro |
| --- | --- | --- | --- |
| **CNN improved** | **97,9% ± 0,1%** | **95,9% ± 1,7%** | **0,959** |
| MobileNetV2 fine-tune | 93,8% ± 0,0% | 93,0% ± 2,5% | 0,929 |

## Evaluasi model final

Model final adalah CNN improved, run seed 1 (validasi tertinggi).

| Data | Loss | Akurasi |
| --- | --- | --- |
| Train | 0,011 | 99,9% |
| Val | 0,062 | 98,0% |
| Test | 0,124 | 95,1% |

- **F1-macro test:** 0,949. Sebanyak 16 dari 20 huruf dikenali sempurna.
- **Kesalahan terbanyak:** ca → sa (28 gambar), la → ha (25 gambar), ta → ha (5 gambar).
- **Per tulisan asli:** 39 dari 41 tulisan dikenali benar, dengan selang kepercayaan 95% sebesar 87,9–99,9%.
- **Group K-Fold (3 fold):** 93,5% ± 1,5%. Pada K-Fold, varian dari tulisan yang sama selalu berada di fold yang sama.

## Keterbatasan

- Training dilakukan di CPU, sehingga gambar dikecilkan ke 96×96 dan jumlah epoch dibatasi.
- Data test hanya berisi 41 tulisan asli, jadi salah mengenali satu tulisan bisa mengubah akurasi sekitar 2,5%.
- Random search hanya 4 percobaan per arsitektur, dan ResNet50 serta EfficientNet belum dicoba.

## Anggota kelompok

| Nama | NIM | 
| --- | --- | 
| Aditya Wahyu Wijanarko | 25/574566/PPA/07251 | 
| Nabila Hermawan | 25/572261/PPA/07190 | 
| Balma Bahira Adzkia | 25/572255/PPA/07189 | 
| Siti Syakirah | 25/573284/PPA/07217 | 

## Referensi

1. hannanhunafa. *Javanese Script (Aksara Jawa) Augmented* [dataset]. Kaggle. https://www.kaggle.com/datasets/hannanhunafa/javanese-script-aksara-jawa-augmented
2. Afiahayati. *Introduction to Deep Learning* (slide kuliah). Universitas Gadjah Mada.
3. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521, 436–444.
4. Sandler, M., et al. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. *CVPR*.
5. Ioffe, S., & Szegedy, C. (2015). Batch normalization. *ICML*.
6. Srivastava, N., et al. (2014). Dropout: A simple way to prevent neural networks from overfitting. *JMLR*, 15, 1929–1958.
7. Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR*.
8. Abadi, M., et al. (2016). TensorFlow: A system for large-scale machine learning. *OSDI*.
