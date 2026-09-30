# AGENTS.md — Overview Service

Laporan kondisi repo ini untuk agen yang baru masuk. Isinya: apa service ini, bentuk
arsitekturnya, kondisi nyata yang terukur saat laporan dibuat, dan batasan yang **bukan** bug
tapi keputusan desain. Dokumentasi pengguna yang lengkap ada di `README.md` (896 baris, bahasa
Indonesia) — file ini tidak menggantikannya, hanya memberi peta supaya tahu ke mana membaca.

Tanggal audit: 2026-09-30. Commit: `f1f002f` (satu commit, branch `main`, working tree bersih).

---

## 1. Service ini apa

**Face Similarity API** — service HTTP yang mengukur kemiripan wajah, ditulis dengan FastAPI.
Satu kalimat: kirim 5 foto referensi + satu video (atau satu foto), dapat skor kemiripan
beserta putusan `match` / `no_match` / `inconclusive`, plus opsional pemeriksaan liveness.

Sifat yang menentukan hampir semua keputusan desain di dalamnya:

- **Stateless total.** Tidak ada database, antrean, cache persisten, atau file yang bertahan.
  Upload video ditulis ke `tempfile` lalu dihapus di `finally`. Yang perlu di-backup dari
  deployment hanya `.env`.
- **Tertutup secara default.** Setiap endpoint kecuali `/health` menuntut header `X-API-Key`.
  Tanpa `FSA_API_KEYS` terisi, service menjawab **503** untuk semuanya (bukan 401 — karena itu
  kesalahan operator, bukan pemanggil). Pengecualiannya hanya kalau `FSA_DEBUG=true`, dan itu
  ditulis sebagai warning saat boot.
- **CPU-bound.** Semua inference lewat `anyio.to_thread.run_sync` dan dibatasi
  `anyio.Semaphore`, jadi event loop tetap responsif. Biaya nyata ±154ms per frame di atas
  overhead tetap ±5,6s (CPU, `buffalo_l`) — angka ini terdokumentasi di komentar
  `app/core/config.py` dekat `liveness_max_sampled_frames`.

Peran dalam sistem yang lebih besar: **dipanggil oleh aplikasi Laravel**, bukan oleh browser.
Integrasinya ada di `../tenancy-app` sebagai `App\Support\Hrms\FaceApiClient`, dipakai untuk
verifikasi wajah pada absensi (`attendance_face_verifications`). Lihat `README.md:477`.

Model: **InsightFace ArcFace** (`buffalo_l`) via ONNX Runtime, embedding 512-d ternormalisasi,
dibandingkan dengan cosine similarity. Bobot diunduh saat pertama kali dipakai ke
`~/.insightface` (tidak ada di repo; `models/` masuk `.gitignore`).

---

## 2. Endpoint

Semua di bawah prefix `FSA_API_PREFIX` (default `/api/v1`).

| Endpoint | Auth | Fungsi |
|---|---|---|
| `POST /face/verify` | ✅ | 5 foto + video → skor. Kalau disertai `challenge_token`, klip yang sama juga diperiksa liveness-nya dan `passed` menuntut keduanya. |
| `POST /face/verify-image` | ✅ | 5 foto + 1 foto capture → skor. **Tidak ada blok `liveness` sama sekali**, disengaja. |
| `POST /liveness/challenge` | ✅ | Terbitkan aksi acak bertanda tangan HMAC. |
| `POST /liveness/verify` | ✅ | Token + video → apakah aksinya dilakukan, berurutan. |
| `GET /health` | ❌ | Status + apakah model sudah termuat. Sengaja terbuka: yang memanggilnya load balancer/orchestrator, yang tidak punya API key. |
| `GET /demo`, `GET /` | ❌ | Halaman demo perekam webcam (`static/index.html`, 319 baris). **Harus diblokir di nginx untuk produksi** — checklist go-live di README menuntut endpoint ini menjawab 404. |

Bentuk respons ada di `app/schemas/`. Field yang dimaksudkan untuk di-gate: **`passed`**.
`match` hanya bernilai true saat `decision is MATCH`; `passed` menambahkan syarat liveness.

---

## 3. Struktur kode

```
app/
  main.py                factory FastAPI, CORS, lifespan (validasi config + warm-up), route /demo
  api/
    deps.py              singleton engine/sampler/limiter via @lru_cache; GuardDep
    security.py          API key — satu-satunya kendali akses
    throttle.py          rate limit fixed-window, per proses
    routes/{verify,liveness,health}.py
  core/
    config.py            SEMUA setting (pydantic-settings, prefix FSA_)
    errors.py            AppError → JSON {error: {code, message, details}}
    logging.py
  schemas/{verification,liveness}.py
  services/
    face_engine.py       Protocol FaceEngine + InsightFaceEngine (import lazy)
    video.py             Protocol FrameSampler + OpenCVFrameSampler
    media.py             baca upload berbatas ukuran, decode, temp file, encode JPEG
    challenge.py         token HMAC + ACTION_CATALOGUE + ACTION_AXES
    liveness.py          metrik yaw/pitch/mata/mulut + LivenessAnalyzer
    verification.py      pipeline & keputusan (689 baris — file terbesar)
static/index.html
tests/                   engine & sampler palsu + test HTTP
```

Total ±5.000 baris Python termasuk test. Tidak ada Dockerfile, tidak ada config CI.

### Titik masuk yang perlu diketahui

- **`GuardDep`** (`app/api/deps.py:88`) adalah keseluruhan kendali akses. Route baru tanpa
  dependency ini = pintu terbuka baru. Urutannya penting dan dibungkus satu dependency
  (`guard()`): key diperiksa dulu, baru rate limit — supaya pemanggil tak dikenal tidak bisa
  menghabiskan kuota pemanggil yang dikenal.
- **`Settings`** (`app/core/config.py`) adalah satu-satunya sumber angka. ±46 field, semuanya
  punya padanan di `.env.example`. Komentar di file ini berisi alasan tiap default, termasuk
  hasil pengukuran — baca sebelum mengubah angka apa pun.
- **Dua Protocol** (`FaceEngine`, `FrameSampler`) ada supaya test bisa mengganti model. Test
  suite **tidak butuh bobot ONNX**: `tests/conftest.py` membangun landmark 68 titik yang benar
  secara geometris per pose, jadi matematika produksi tetap yang diuji.

---

## 4. Logika keputusan

### Video (`/face/verify`)

```
skor  = mean dari top-K frame terbaik,  K = min(len, max(3, ceil(0.3 × jumlah_frame)))
match = skor >= 0.38  AND  match_ratio >= 0.6
```

- Per frame, kalau ada beberapa wajah, yang dipilih adalah **yang paling mirip referensi**,
  bukan yang terbesar — supaya orang lewat di belakang tidak menjatuhkan skor. Liveness diukur
  pada wajah yang sama itu, supaya orang lain tidak bisa melakukan aksinya mewakili.
- `inconclusive` bila: frame berwajah < 3, atau |skor − threshold| ≤ 0.03, atau skor lolos tapi
  rasio frame kurang.

### Foto (`/face/verify-image`)

Satu still tidak punya frame untuk dirata-rata, jadi ketahanannya diambil dari sisi lain:

```
skor  = mean dari 3 similarity referensi terbaik (FSA_IMAGE_TOP_K_REFERENCES)
match = skor >= 0.38  AND  matched_references >= 3 (FSA_MIN_REFERENCE_MATCHES)
```

Syarat kedua bukan hiasan. Komentar di `config.py` menyimpan kasus yang memaksanya: pada
ambang 2, similarity `[0.62, 0.60, 0.05, 0.05, 0.05]` menghasilkan skor 0.42 dengan dua match
— dua foto orang yang salah di satu set enrolment cukup untuk lolos sebagai pemiliknya.

### Liveness

Semua diukur **relatif terhadap pose netral orang itu sendiri**, yaitu median seluruh klip —
bukan sudut absolut. Konsekuensinya diringkas di `ACTION_AXES` (`app/services/challenge.py`):
satu challenge **tidak boleh** menarik dua aksi pada sumbu yang sama. `look_up` + `look_down`
membuat median jatuh di tengah kedua ekstrem, kedua ekskursi terbelah dua, dan rekaman yang
benar ditolak. Karena itu `action_count` di-clamp ke jumlah sumbu berbeda dalam pool, bukan ke
ukuran pool.

Urutan aksi ditegakkan dengan cursor yang hanya maju (`LivenessAnalyzer.analyze`).

### Error yang punya kode terpisah dengan alasan

`ReferenceFacesUnusableError` vs `NoFaceDetectedError` — keduanya 422, tapi kodenya berbeda
karena pemanggil harus bisa membedakan "foto enrolment Anda tidak terbaca" dari "wajah Anda
tidak terlihat sekarang". Dulu keduanya `no_face_detected`, dan karyawan dengan satu foto
enrolment rusak disuruh mendekat ke kamera — instruksi yang mustahil dipenuhi.

---

## 5. Kondisi terukur

| Yang diperiksa | Hasil |
|---|---|
| `pytest` | **116 passed**, 1 warning, 2,43s |
| `ruff check .` | **All checks passed** |
| Python venv | 3.12.13 (rentang didukung: 3.10–3.12, batasan wheel onnxruntime+insightface) |
| Git | 1 commit, `main`, working tree bersih |
| `.env` vs `.env.example` | Set key **identik**, tidak ada yang tertinggal |
| `.env` terisi | `FSA_API_KEYS` dan `FSA_CHALLENGE_SECRET` keduanya terisi |

Versi terpasang, semuanya di dalam rentang `requirements.txt`: fastapi 0.141.1, uvicorn 0.52.4,
pydantic 2.13.5, numpy 2.2.6, opencv-python-headless 4.14.0.94, onnxruntime 1.29.0,
insightface 0.7.3.

### Selisih kecil yang ditemukan (tidak ada yang merusak)

1. **`README.md:536` menyebut "90 test", nyatanya 116.** Dokumentasi tertinggal dari suite.
2. **`.env` lokal memakai `FSA_LIVENESS_MAX_SAMPLED_FRAMES=36`, `.env.example` 60.** Ini
   penyetelan sengaja — 36 frame ≈ 11,2s vs 60 frame ≈ 14,9s, dan aman karena pool aksi default
   tidak memuat `blink`. Alasannya terdokumentasi di `config.py`.
3. **`ApiKeyDep` (`app/api/security.py:89`) tidak dipakai di mana pun.** Route memakai
   `GuardDep`, yang benar karena itu menyertakan rate limit. Alias yang tersisa.
4. **Tidak ada CI.** `requirements-test.txt` jelas ditulis untuk CI ("Light dependency set for
   CI"), tapi tidak ada workflow yang memakainya.

---

## 6. Batasan desain — jangan diperlakukan sebagai bug

Ini semua sudah diketahui, terdokumentasi di `README.md:871`, dan dipilih sadar:

- **Rate limit dihitung per proses, di memori.** Jalan dua worker uvicorn → limit efektif
  berlipat dua. Dinyatakan terbuka di docstring `app/api/throttle.py`. Untuk pembatasan yang
  sungguhan, tempatnya di reverse proxy — contoh nginx ada di README.
- **Token challenge bisa dipakai ulang selama masih berlaku.** Tanda tangan HMAC membuktikan
  token itu asli, bukan bahwa belum pernah dipakai, dan service stateless tidak bisa tahu.
  Sekali-pakai adalah tanggung jawab pemanggil: klaim `challenge_id` **dari payload token**
  (bukan dari field terpisah kiriman klien) pada indeks unik, **sebelum** request dikirim.
- **`/face/verify-image` tidak melaporkan liveness apa pun.** Still tidak bisa membawa sinyal
  liveness — pose diukur antar-frame, foto punya satu. Ini penolakan eksplisit, bukan fitur yang
  belum ada. Siapa pun yang memanggilnya memiliki pertanyaan itu sendiri.
- **Liveness ini bukan anti-spoofing lengkap.** Menghadang foto diangkat dan video lama diputar
  ulang; tidak menghadang deepfake real-time.
- **`blink` paling tidak andal** (kedipan ±0,2s vs sampling ±10 fps) dan sengaja tidak masuk
  pool default.
- **`FSA_CORS_ORIGINS` bukan kendali akses.** CORS adalah aturan yang diterapkan browser pada
  dirinya sendiri; curl, klien mobile, dan skrip mengabaikannya. Pemanggil service ini adalah
  server.
- **Lisensi `buffalo_l` non-komersial** (InsightFace). Perlu diperiksa sebelum dipakai di
  produk komersial.
- **Threshold per-request** (`match_threshold`, `min_match_ratio`) bisa dikirim pemanggil karena
  satu deployment melayani beberapa workspace dengan kamera dan pencahayaan berbeda. Hanya dua
  itu yang bisa ditimpa — frame threshold, jendela top-K, dan inconclusive band adalah properti
  cara skor dihitung, dan membiarkan pemanggil menggesernya sama dengan membiarkannya menggeser
  arti skor yang ia terima.

---

## 7. Perintah

```bash
make install       # venv + runtime penuh (model ONNX diunduh saat request pertama)
make install-test  # cukup untuk menjalankan suite, tanpa ~300MB bobot ONNX
make run           # uvicorn --reload di 127.0.0.1:8001
make serve         # 2 worker, gaya produksi
make test          # pytest
make lint          # ruff check
make fmt           # ruff check --fix + format
```

Port default **8001**, bukan 8000: 8000 dipakai Laravel pada pemasangan gabungan.

`Makefile` mengandung logika pencarian interpreter dan penyehatan venv yang tidak sepele —
mendeteksi `.venv` yang dipindah antar-direktori, membuang venv proyek lain dari PATH sebelum
mencari interpreter, dan menambal bug template `activate` pada build CPython tertentu. Setiap
blok punya komentar yang menjelaskan kegagalan apa yang dicegahnya. Jangan disederhanakan tanpa
membaca komentarnya.

---

## 8. Konvensi

- Bahasa: **komentar & docstring dalam bahasa Inggris**, dokumentasi pengguna (`README.md`,
  `Makefile`, instruksi liveness yang dilihat user) **dalam bahasa Indonesia**.
- Komentar di repo ini menjelaskan **mengapa**, sering dengan kasus kegagalan konkret yang
  memaksa keputusan itu. Perlakukan sebagai catatan keputusan, bukan noise. Kalau sebuah angka
  atau cabang terlihat aneh, komentar di atasnya biasanya menjelaskan alasannya.
- Ruff: `line-length = 100`, target `py310`, rules `E,F,I,UP,B,SIM`, `B008` di-ignore (FastAPI
  memakai callable sebagai default argumen dengan sengaja).
- `from __future__ import annotations` di setiap modul.
- Dependency berat (`cv2`, `insightface`, `onnxruntime`) **diimpor lazy di dalam fungsi**,
  supaya aplikasi, skema OpenAPI, dan test bisa jalan tanpa model tersedia. Pertahankan pola
  ini.
- Error domain selalu turunan `AppError` dengan `code` + `status_code`, supaya bentuk JSON-nya
  konsisten.
- Log menyebut **label** kunci API, bukan kuncinya. Perbandingan kunci memakai
  `hmac.compare_digest` atas **bytes** tanpa early exit — keduanya disengaja (timing leak, dan
  byte >127 pada header yang pernah membuat 500 alih-alih 401).
