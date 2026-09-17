# Face Similarity API

FastAPI starter kit untuk **mengukur kemiripan wajah** antara 5 foto referensi dan sebuah
video rekaman, plus **liveness challenge** untuk memastikan yang direkam adalah orang hidup
di depan kamera. Stateless sepenuhnya: tidak ada database, tidak ada penyimpanan file, tidak
ada antrean. Satu request masuk → skor keluar, semua file sementara dihapus.

- Deteksi + embedding wajah: **InsightFace ArcFace (`buffalo_l`)** via ONNX Runtime.
- Video di-sample beberapa frame, tiap wajah dibandingkan ke seluruh foto referensi.
- Liveness: server memberi aksi acak (menoleh, mendongak, buka mulut, berkedip) yang
  ditandatangani HMAC, lalu memeriksa aksi itu benar dilakukan **dan berurutan**.
- Halaman demo dengan perekam webcam + pemandu aksi di `/demo`.
- **Tertutup secara default.** Setiap endpoint kecuali `/health` menuntut header
  `X-API-Key`, dan tanpa `FSA_API_KEYS` terkonfigurasi servis menolak semuanya — lihat
  [Autentikasi](#autentikasi).

**Jalan cepat**

| Mau apa | Ke mana |
|---|---|
| Menjalankan di laptop | [Instalasi](#instalasi) |
| Memanggilnya dari kode | [Autentikasi](#autentikasi) lalu [Endpoint](#endpoint) |
| Memasangnya dari Laravel | [Integrasi dari Laravel](#integrasi-dari-laravel) |
| Menaikkan ke server | [Deployment di server](#deployment-di-server) |
| Menyetel ketat/longgarnya | [Menyetel threshold](#menyetel-threshold) |

---

## Cara kerja

```
5 foto  ──► decode ──► deteksi wajah ──► embedding (512-d, ternormalisasi)
                                              │
                                    cohesion check (5 foto ini orang yang sama?)
                                              │
video   ──► sampling frame ──► deteksi wajah ──► embedding ──► cosine similarity
                                    │                              │
                                landmark 68 titik          skor = mean top-K frame
                                    │                       + rasio frame cocok
                            yaw / pitch / mata / mulut             │
                                    │                     match / no_match / inconclusive
                          aksi tantangan terpenuhi?                │
                                    └──────────────┬───────────────┘
                                              passed
```

Detail keputusan kemiripan:

| Langkah | Aturan |
|---|---|
| Wajah per foto referensi | Wajah terbesar dipakai; foto tanpa wajah ditolak |
| Cohesion referensi | Rata-rata similarity antar foto; di bawah `0.35` → warning "bukan orang yang sama" |
| Wajah per frame video | Kalau ada beberapa wajah, dipilih yang **paling mirip** referensi (orang lewat tidak merusak skor) |
| Skor akhir | Rata-rata `top-K` frame terbaik (`K = max(3, 30% frame)`) — tahan terhadap frame blur/miring |
| `match` | `score >= 0.38` **dan** `match_ratio >= 0.6` |
| `inconclusive` | Frame berwajah < 3, atau skor terlalu dekat threshold (±0.03), atau skor lolos tapi rasio frame kurang |
| `no_match` | Selain di atas |

Semua angka di atas bisa diubah lewat `.env` (lihat `.env.example`).

---

## Instalasi

Butuh **Python 3.10, 3.11, atau 3.12** — batasan wheel `onnxruntime` + `insightface`,
dan 3.13+ belum termasuk. Versi mana pun dari ketiganya boleh; tidak ada yang wajib 3.11.

```bash
cd supports
cp .env.example .env

make install    # cari interpreter yang ada, buat .venv, pasang dependensi
make run        # http://127.0.0.1:8001
```

`make install` memilih sendiri `python3.12`, `python3.11`, atau `python3.10` — yang mana pun
lebih dulu ditemukan di PATH. Kalau interpreternya ada di tempat lain, sebutkan:
`make install PY=/usr/local/bin/python3.11`.

Belum punya satu pun dari ketiganya:

```bash
brew install python@3.12                                        # macOS
sudo apt install python3.12 python3.12-venv python3.12-dev      # Ubuntu 24.04
sudo apt install python3.11 python3.11-venv python3.11-dev      # Ubuntu 22.04
```

Tanpa `make` (isinya persis sama):

```bash
PY=$(command -v python3.12 || command -v python3.11 || command -v python3.10)
"$PY" -m venv .venv
source .venv/bin/activate
pip install --upgrade pip "cython<3.1" "numpy<2.3"   # insightface dibuild dari sdist
pip install -r requirements-dev.txt

uvicorn app.main:app --reload --port 8001
```

> **Jangan menyalin `.venv` dari folder atau mesin lain.** Sebuah venv menyimpan path
> absolutnya di dalam shebang setiap skrip konsol dan di dalam `activate`. Kalau ikut
> tersalin, `uvicorn` diam-diam menjalankan interpreter proyek lama dan
> `source .venv/bin/activate` memasang direktori yang tidak ada ke PATH — `python` lalu
> jatuh ke Python bawaan sistem (3.9 di macOS) dan paketnya "hilang". `make install`
> mendeteksi venv asing dan membuatnya ulang; kalau ragu, `make clean && make install`.

> **Venv proyek lain yang sedang aktif juga dilewati.** Sebuah venv menaruh `python3.x`
> miliknya paling depan di PATH, jadi `make install` yang dijalankan sementara venv proyek
> lain aktif akan membangun `.venv` di sini dari interpreter proyek itu. Direktorinya
> dibuang dari PATH sebelum pencarian, dan `make` memberi catatan bahwa venv itu aktif —
> tapi `deactivate` dulu tetap yang paling bersih.
>
> Kalau `PY` disebutkan sendiri, kutip path-nya bila mengandung spasi:
> `make install PY="/Users/nama/Application Support/python3.12"`. Path yang tidak bisa
> dijalankan ditolak dengan pesan yang menyebutkan path-nya, bukan `Error 126`.

Sebagian build CPython — Homebrew 3.12.13 salah satunya — menulis `activate` dengan
prompt terbungkus dua kali, sehingga terminal menampilkan `((.venv) )`. `--prompt` tidak
bisa menghindarinya, dan membangun ulang venv tidak mengubah apa pun karena sumbernya ada
di template Python itu sendiri. `make install` merapikannya sendiri lewat target
`normalise-prompt`, yang tidak melakukan apa-apa pada venv yang templatenya sudah benar.

Saat pertama kali dipakai, InsightFace mengunduh model pack `buffalo_l` ke
`~/.insightface` — ±325 MB setelah diekstrak. Set `FSA_MODEL_ROOT` untuk memindahkannya. Dengan
`FSA_WARM_UP_ON_STARTUP=true` (default) model dimuat saat startup, jadi request pertama
tidak kena cold start ±10–30 detik.

Buka:

- `http://127.0.0.1:8001/demo` — playground: pilih 5 foto, ambil tantangan, rekam sambil
  dipandu, lihat hasilnya. Isi kolom **Kunci API** di bagian atas halaman dengan salah satu
  nilai `FSA_API_KEYS`; tanpa itu servis menolak permintaannya.
- `http://127.0.0.1:8001/docs` — Swagger UI. Tombol *Try it out* juga perlu header
  `X-API-Key`.

---

## Autentikasi

Setiap endpoint kecuali `/health` menuntut header `X-API-Key` berisi salah satu nilai
dari `FSA_API_KEYS`.

```bash
curl -H "X-API-Key: $FACE_API_KEY" …
```

`FSA_API_KEYS` berformat `label:kunci`, dipisah koma — labelnya muncul di log agar sebuah
permintaan bisa ditelusuri tanpa kuncinya sendiri pernah tertulis:

```dotenv
FSA_API_KEYS=tenancy-app:K7f…,kiosk:9Qa…
```

**Kosong berarti tertutup, bukan terbuka.** Tanpa kunci terkonfigurasi servis menjawab
`503 api_keys_not_configured` untuk semua permintaan, kecuali `FSA_DEBUG=true` yang
memang jalur pengembangan lokal dan diperingatkan saat boot. Servis ini menjalankan
pengenalan wajah atas apa pun yang dikirim padanya: port terbuka bukan sekadar GPU
gratis, melainkan cara siapa pun mencocokkan foto orang.

`FSA_CORS_ORIGINS` bukan penggantinya. CORS adalah aturan yang diterapkan *browser*
pada permintaannya sendiri; curl, klien mobile, dan skrip mengabaikannya sepenuhnya.

**Kosong juga berarti kunci yang tidak terpakai.** Entri tanpa kunci — `label:`
tanpa apa pun sesudahnya, koma nyasar — dibuang, bukan diterima sebagai kata sandi
kosong. Kalau setelah dibuang tidak ada kunci yang tersisa, servis menjawab
`503 api_keys_not_configured` seperti halnya kalau `FSA_API_KEYS` memang kosong,
dan menyebutkan label entri yang rusak di log saat boot.

Setiap kunci juga dibatasi `FSA_RATE_LIMIT_PER_MINUTE` permintaan per menit, dihitung
**di dalam satu proses**. Dua worker berarti dua hitungan dan batas efektif dua kali
lipat; untuk batas yang benar-benar global, pakai `limit_req` nginx — lihat bagian
deployment.

Hitungannya per kunci **dan** per header `X-Tenant`, kalau pemanggil mengirimnya.
Satu pemasangan aplikasi multi-tenant adalah satu kunci, jadi tanpa itu perusahaan
pertama yang absen pukul 08:00 menghabiskan jatah semua perusahaan lain dan mereka
dijawab "layanan verifikasi wajah tidak tersedia". Header ini bukan kendali akses
dan tidak diperlakukan sebagai kendali akses: nilai yang tidak dikenal hanya
mendapat ember hitungannya sendiri di dalam batas kuncinya.

---

## Endpoint

### `POST /api/v1/face/verify`

`multipart/form-data`:

| Field | Tipe | Keterangan |
|---|---|---|
| `images` | file × 5 | Foto referensi satu orang. JPEG/PNG/WebP/BMP, maks 8 MB/file. |
| `video` | file | Video rekaman orang yang dicek. mp4/webm/mov/mkv/avi, maks 64 MB dan 60 detik. |
| `challenge_token` | string (opsional) | Token dari `/liveness/challenge`. Kalau diisi, video yang sama sekaligus dicek liveness-nya. |
| `match_threshold` | float (opsional) | Ambang kemiripan untuk permintaan ini saja, `-1`..`1`. Ambang per-frame ikut bergeser bersamanya. |
| `min_match_ratio` | float (opsional) | Porsi frame berwajah yang harus cocok, `0`..`1`. |
| `include_best_frame` | bool (opsional) | Kembalikan frame yang skornya diukur, sebagai JPEG base64 di `video.best_frame.image_base64`. |

`include_best_frame` ada untuk pemanggil yang menyimpan sebuah foto di samping
catatan kehadiran. Foto yang layak disimpan adalah frame yang **benar-benar
dinilai**, bukan potret yang dipilih sendiri oleh klien sebelum merekam — yang
tidak pernah dibandingkan dengan apa pun, lalu diarsipkan di sebelah sebuah
perbandingan. Frame itu dikecilkan lebih dulu (`FSA_BEST_FRAME_MAX_SIDE`) karena
ia ikut di dalam respons JSON.

Dua field terakhir ada karena satu pemasangan servis ini melayani beberapa workspace, dan
kamera gudang pada malam hari tidak menuntut angka yang sama dengan lobi kantor. Yang tahu
permintaan ini milik workspace mana adalah pemanggilnya, bukan servis ini — jadi angkanya
ikut di permintaan, bukan dikonfigurasi per tenant di sini. `thresholds` pada respons selalu
melaporkan angka yang benar-benar dipakai.

Hanya dua ambang penentu putusan itu yang bisa ditimpa. Ambang per-frame, jendela top-K, dan
pita `inconclusive` adalah properti *cara* skor dihitung; membiarkan pemanggil menggesernya
sama dengan membiarkannya menggeser arti skor yang ia terima.

```bash
curl -s -X POST http://127.0.0.1:8001/api/v1/face/verify \
  -H "X-API-Key: $FSA_KEY" \
  -F "images=@ref1.jpg" -F "images=@ref2.jpg" -F "images=@ref3.jpg" \
  -F "images=@ref4.jpg" -F "images=@ref5.jpg" \
  -F "video=@rekaman.webm" | jq
```

Respons (dipangkas):

```json
{
  "passed": true,
  "match": true,
  "decision": "match",
  "score": 0.6412,
  "score_percent": 64.12,
  "confidence": 0.9601,
  "thresholds": {
    "match_threshold": 0.38,
    "frame_match_threshold": 0.38,
    "min_match_ratio": 0.6,
    "inconclusive_band": 0.03
  },
  "reference": {
    "submitted": 5,
    "accepted": 5,
    "rejected": 0,
    "cohesion": 0.7218,
    "consistent": true,
    "images": [
      {
        "index": 0,
        "filename": "ref1.jpg",
        "face_detected": true,
        "detection_score": 0.892,
        "bbox": { "x": 214, "y": 96, "width": 180, "height": 232 },
        "similarity_to_group": 0.7411,
        "best_video_similarity": 0.6688,
        "mean_video_similarity": 0.5312,
        "skip_reason": null
      }
    ]
  },
  "video": {
    "duration_seconds": 5.03,
    "fps": 30.0,
    "total_frames": 151,
    "frames_sampled": 24,
    "frames_with_face": 22,
    "frames_matched": 20,
    "match_ratio": 0.9091,
    "best_frame": {
      "frame_index": 60,
      "timestamp_seconds": 2.0,
      "face_detected": true,
      "detection_score": 0.901,
      "bbox": { "x": 402, "y": 128, "width": 214, "height": 268 },
      "similarity": 0.6688,
      "matched": true,
      "best_reference_index": 0
    },
    "frames": ["… satu entri per frame yang di-sample …"]
  },
  "liveness": null,
  "warnings": [],
  "model": "buffalo_l",
  "processing_ms": 1874
}
```

Field penting:

- `passed` — **satu-satunya field yang perlu dijadikan gerbang**: wajah cocok DAN (kalau
  token dikirim) aksi liveness terpenuhi.
- `score` — cosine similarity (−1..1). Ini angka mentah yang sebaiknya kamu simpan/audit.
- `score_percent` — `score` dipetakan ke 0–100 untuk ditampilkan ke user. **Bukan** probabilitas.
- `confidence` — 0..1, seberapa yakin terhadap `decision` (jarak skor dari threshold).
- `warnings` — hal yang perlu ditindak: foto referensi tidak konsisten, wajah jarang terlihat,
  ada orang lain di frame, liveness gagal, dsb.

### `POST /api/v1/liveness/challenge`

Body JSON opsional: `{"action_count": 2}`.

```json
{
  "challenge_id": "V1v8kQ2r9m1s7pQ0dA",
  "token": "eyJhY3QiOlsidHVybl9sZWZ0Iiwib3Blbl9tb3V0aCJdLCJj….q9C2m…",
  "actions": [
    { "action": "turn_left", "instruction": "Tolehkan kepala ke kiri Anda, tahan sebentar, lalu kembali menghadap kamera." },
    { "action": "open_mouth", "instruction": "Buka mulut lebar-lebar sebentar, lalu tutup kembali." }
  ],
  "issued_at": 1786432011,
  "expires_at": 1786432131,
  "ttl_seconds": 120
}
```

Tidak ada apa pun yang disimpan: daftar aksi, `challenge_id`, dan waktu kedaluwarsa
ditandatangani HMAC-SHA256 ke dalam `token`. Server cukup memverifikasi tanda tangannya
ketika token itu kembali.

Aksi yang tersedia: `turn_left`, `turn_right`, `look_up`, `look_down`, `open_mouth`, `blink`.

Satu tantangan **tidak pernah** memuat dua aksi yang menggerakkan poros yang sama —
`turn_left`/`turn_right` berbagi poros yaw, `look_up`/`look_down` berbagi pitch. Pose
istirahat yang dijadikan pembanding disimpulkan dari median seluruh klip, bukan
diterima sebagai masukan terpisah; klip yang berisi kedua ujung satu poros membuat
median itu jatuh di antara keduanya, sehingga setiap aksi diukur terhadap aksi lain
alih-alih terhadap titik diam orangnya. Keduanya lalu gagal mencapai ambang dan
rekamannya ditolak dengan menyebut aksi yang justru diperagakan dengan benar.

Karena itu `action_count` dibatasi pada **jumlah poros** dalam pool, bukan jumlah
aksinya: pool bawaan berisi lima aksi pada tiga poros, sehingga permintaan empat aksi
dijawab tiga.

### `POST /api/v1/liveness/verify`

`multipart/form-data` dengan `token` (dari endpoint di atas) dan `video`.

```bash
curl -s -X POST http://127.0.0.1:8001/api/v1/liveness/verify \
  -H "X-API-Key: $FSA_KEY" \
  -F "token=$TOKEN" -F "video=@rekaman.webm" | jq
```

```json
{
  "live": true,
  "challenge_id": "V1v8kQ2r9m1s7pQ0dA",
  "actions": [
    {
      "action": "turn_left",
      "instruction": "Tolehkan kepala ke kiri Anda, …",
      "performed": true,
      "detected_at_seconds": 1.83,
      "frame_index": 55,
      "measured": 0.4123,
      "baseline": 0.0187,
      "reason": null
    },
    {
      "action": "open_mouth",
      "instruction": "Buka mulut lebar-lebar sebentar, …",
      "performed": true,
      "detected_at_seconds": 4.4,
      "frame_index": 132,
      "measured": 0.4815,
      "baseline": 0.0932,
      "reason": null
    }
  ],
  "frames_sampled": 60,
  "frames_with_face": 58,
  "movement_score": 0.2841,
  "warnings": [],
  "processing_ms": 3120
}
```

Cara pengukurannya — semua **relatif terhadap pose netral orang itu sendiri** (median
sepanjang klip), bukan sudut absolut, karena bentuk wajah dan posisi kamera terlalu bervariasi:

| Aksi | Yang diukur |
|---|---|
| `turn_left` / `turn_right` | Pergeseran horizontal ujung hidung terhadap garis mata, dibagi jarak antar mata |
| `look_up` / `look_down` | Pergeseran vertikal ujung hidung terhadap garis mata |
| `open_mouth` | Bukaan bibir dalam dibagi lebar mulut (MAR) |
| `blink` | Eye aspect ratio (EAR) turun ke `0.7×` baseline |

`reason` saat gagal: `not_detected` (aksi tidak terlihat), `not_enough_frames` (wajah terlalu
jarang terdeteksi), `landmarks_unavailable` (`FSA_ENABLE_LANDMARKS=false`).

`movement_score` mendekati 0 berarti wajah nyaris tidak bergerak sepanjang klip — persis
seperti foto yang diangkat ke depan kamera.

**Urutan diperiksa.** Pencarian aksi ke-2 dimulai dari frame setelah aksi ke-1 ditemukan,
jadi merekam aksi yang benar dengan urutan terbalik tetap gagal.

**Liveness diukur pada wajah yang dicocokkan.** Kalau `/face/verify` dipakai dengan token,
metrik pose diambil dari wajah yang paling mirip foto referensi — bukan wajah terbesar —
sehingga orang lain di frame tidak bisa "mengerjakan" tantangan untuk orang yang diverifikasi.

### `GET /api/v1/health`

Satu-satunya endpoint tanpa kunci. Probe kesiapan dipanggil oleh load balancer dan
orkestrator yang tidak membawa kredensial, dan yang diungkapnya — hidup, dan apakah
modelnya selesai dimuat — sudah diketahui pemanggilnya dari kenyataan bahwa servisnya
menjawab sama sekali.

```bash
curl -s http://127.0.0.1:8001/api/v1/health | jq
```

```json
{ "status": "ok", "version": "0.1.0", "model": "buffalo_l", "model_loaded": true }
```

`model_loaded: false` berarti worker masih memuat model atau gagal memuatnya — bedanya ada
di log.

### Format error

Semua error domain memakai bentuk yang sama:

```json
{ "error": { "code": "no_face_detected", "message": "…", "details": {} } }
```

| Status | `code` | Penyebab |
|---|---|---|
| 401 | `unauthorized` | `X-API-Key` tidak ada atau tidak dikenal |
| 429 | `rate_limited` | Kunci ini melewati `FSA_RATE_LIMIT_PER_MINUTE` |
| 503 | `api_keys_not_configured` | `FSA_API_KEYS` kosong — atau tidak ada entri yang berisi kunci — dan `FSA_DEBUG` mati |
| 400 | `invalid_challenge` | Token liveness rusak / tanda tangan tidak cocok |
| 400 | `challenge_expired` | Token liveness kedaluwarsa |
| 422 | `invalid_upload` | Jumlah foto salah, file kosong, bukan gambar |
| 422 | `no_face_detected` | Foto referensi yang berwajah kurang dari minimum |
| 422 | `video_decode_failed` | Video tidak bisa dibuka / terlalu panjang (diperiksa juga setelah decode, karena WebM dari MediaRecorder sering tidak mencantumkan durasi) |
| 413 | `payload_too_large` | Foto atau video melewati batas ukuran |
| 415 | `unsupported_media_type` | Content-type tidak diizinkan |
| 503 | `engine_unavailable` | Model gagal dimuat |

---

## Konfigurasi

Semua env var berprefix `FSA_`, daftar lengkap ada di `.env.example`. Yang paling sering diubah:

| Var | Default | Fungsi |
|---|---|---|
| `FSA_API_KEYS` | — | **Wajib di produksi.** Kosong berarti servis menolak semua permintaan dengan 503 |
| `FSA_RATE_LIMIT_PER_MINUTE` | `60` | Permintaan per kunci per menit, dihitung per proses; `0` mematikannya |
| `FSA_MATCH_THRESHOLD` | `0.38` | Ambang keputusan match |
| `FSA_MIN_MATCH_RATIO` | `0.6` | Porsi frame yang harus cocok |
| `FSA_MAX_SAMPLED_FRAMES` | `24` | Frame yang dianalisis — naikkan untuk akurasi, turunkan untuk kecepatan |
| `FSA_BEST_FRAME_MAX_SIDE` | `640` | Sisi terpanjang frame yang dikembalikan lewat `include_best_frame` |
| `FSA_MODEL_PACK` | `buffalo_l` | Ganti `buffalo_s` untuk CPU lemah (lebih cepat, sedikit kurang akurat) |
| `FSA_CTX_ID` | `-1` | `-1` CPU, `0` GPU pertama (butuh `onnxruntime-gpu`) |
| `FSA_MAX_CONCURRENT_INFERENCES` | `2` | Batas request berat yang jalan bersamaan |
| `FSA_CHALLENGE_SECRET` | — | **Wajib diisi di produksi**, sama di semua worker |
| `FSA_CHALLENGE_TTL_SECONDS` | `120` | Umur token; makin pendek makin sempit peluang replay |
| `FSA_CHALLENGE_ACTIONS` | 5 aksi | Kolam aksi yang diundi |
| `FSA_LIVENESS_MAX_SAMPLED_FRAMES` | `60` | Frame untuk liveness; `blink` butuh angka tinggi |
| `FSA_LIVENESS_YAW_DELTA` | `0.30` | Seberapa jauh kepala harus menoleh |

### Menyetel threshold

`0.38` adalah titik aman umum untuk ArcFace `buffalo_l`, tapi ambang yang benar bergantung
pada populasi dan kualitas kamera kamu. Cara menyetelnya: kumpulkan pasangan yang kamu tahu
jawabannya, panggil endpoint, catat `score`, lalu pilih ambang sesuai toleransi
false-accept vs false-reject. Endpoint ini mengembalikan skor mentah justru supaya kalibrasi
seperti ini bisa dilakukan tanpa mengubah kode.

Panduan kasar `score`: `> 0.6` sangat mirip · `0.4–0.6` mirip · `0.3–0.4` abu-abu · `< 0.3` beda orang.

Untuk liveness, kalau banyak orang gagal padahal menoleh dengan benar, turunkan
`FSA_LIVENESS_YAW_DELTA` ke `0.22`; kalau video foto-diangkat malah lolos, naikkan.

---

## Integrasi dari Laravel

Sudah terpasang di `../tenancy-app`, dan kodenya ada di satu tempat:
`App\Support\Hrms\FaceApiClient`. Aturannya, tiga hal ini milik pemanggil dan tidak bisa
diambil alih servis yang stateless:

1. **Foto referensi.** Servis ini tidak menyimpan siapa pun; fotonya ada di disk tenant.
2. **Sekali pakainya tantangan.** Tanda tangan HMAC membuktikan sebuah token asli, bukan
   bahwa ia belum pernah dipakai. Di tenancy-app, `challenge_id` adalah kolom unik pada
   `attendance_face_verifications` dan barisnya ditulis **sebelum** servis dipanggil —
   itulah yang membuat dua permintaan berbalapan pada satu token tidak bisa lolos berdua.
3. **Apa yang dilakukan pada `inconclusive`.** Antrekan ke manusia; jangan tolak otomatis.

```php
use Illuminate\Support\Facades\Http;

$base = config('services.face_api.url');
$key = config('services.face_api.key');

// 1. Ambil tantangan, kirim `actions` ke aplikasi untuk dipandu ke user.
$challenge = Http::withHeader('X-API-Key', $key)
    ->post("{$base}/api/v1/liveness/challenge", ['action_count' => 2])
    ->throw()
    ->json();

// 2. Klaim challenge_id-nya lebih dulu — indeks unik, bukan pengecekan.
$verification = FaceVerification::create(['challenge_id' => $challenge['challenge_id'], ...]);

// 3. Setelah aplikasi mengirim rekamannya:
$request = Http::withHeader('X-API-Key', $key)->timeout(120)->asMultipart();

foreach ($referencePaths as $path) {
    $request = $request->attach('images', fopen($path, 'r'), basename($path));
}

$result = $request
    ->attach('video', fopen($videoPath, 'r'), 'recording.mp4')
    ->post("{$base}/api/v1/face/verify", array_filter([
        'challenge_token' => $challenge['token'],
        // Ambang milik workspace ini, kalau memang berbeda.
        'match_threshold' => config('services.face_api.match_threshold'),
        'min_match_ratio' => config('services.face_api.min_match_ratio'),
    ]))
    ->throw()
    ->json();

if ($result['passed']) {
    // wajah cocok dan aksi liveness terpenuhi
}
```

---

## Testing

Suite memakai engine palsu, jadi tidak perlu mengunduh model:

```bash
make install-test    # atau: pip install -r requirements-test.txt
make test            # 90 test
make lint
```

Yang dicakup: keputusan match/no-match, frame tanpa wajah, orang lain di frame, foto
referensi campuran, validasi upload (jumlah, ukuran, content-type, file rusak), matematika
skor, sampling video sungguhan lewat OpenCV, penandatanganan & kedaluwarsa token,
pendeteksian tiap aksi liveness, urutan aksi, klip diam, bukti bahwa liveness mengikuti
wajah yang dicocokkan (bukan wajah terbesar), serta kendali akses: kunci hilang/salah/benar,
servis tanpa kunci terkonfigurasi yang menutup diri, `/health` yang tetap terbuka, batas laju
per kunci, dan ambang per-permintaan yang mengubah putusan.

Engine palsu membangun landmark 68 titik yang **benar secara geometris** untuk tiap pose,
jadi yang diuji tetap matematika produksi — hanya jaringan sarafnya yang diganti.

---

## Struktur

```
app/
  main.py                  factory FastAPI, CORS, lifespan, route /demo
  api/
    deps.py                singleton engine/sampler/limiter, semua bisa di-override saat test
    security.py            kunci API: satu-satunya kendali akses servis ini
    throttle.py            batas laju per kunci, per proses
    routes/verify.py       validasi upload, orkestrasi request
    routes/liveness.py     terbitkan tantangan, verifikasi rekaman
    routes/health.py
  core/
    config.py              seluruh setting (pydantic-settings)
    errors.py              error domain → JSON konsisten
  schemas/                 model respons
  services/
    face_engine.py         wrapper InsightFace + cosine similarity
    challenge.py           token HMAC + katalog aksi
    liveness.py            metrik pose/mata/mulut + analisis aksi
    media.py               baca upload dengan batas ukuran, decode gambar, file sementara
    video.py               sampling frame OpenCV (memori terbatas)
    verification.py        logika perbandingan & keputusan
static/index.html          halaman demo perekam webcam
tests/                     engine & sampler palsu + test HTTP
```

Semua pekerjaan CPU dijalankan lewat `anyio.to_thread` dan dibatasi semaphore, sehingga event
loop tetap responsif saat inference berjalan. Kalau `/face/verify` dipanggil dengan token,
video hanya di-decode dan dideteksi **sekali** untuk kedua pemeriksaan.

---

## Deployment di server

Tidak ada Docker di repo ini — jalankan langsung sebagai service Python di belakang nginx.
Contoh berikut untuk Ubuntu 22.04/24.04, dengan servis mendengarkan di `127.0.0.1:8001`
dan nginx sebagai satu-satunya yang menghadap keluar.

Sebelum mulai, dua hal yang menentukan bentuk pemasangan:

- **Servis ini tidak menyimpan apa pun.** Tidak ada database, tidak ada antrean, tidak ada
  berkas yang bertahan. Yang perlu di-backup hanyalah `.env`.
- **Ia melakukan pengenalan wajah atas apa pun yang dikirim padanya.** Karena itu ia
  tertutup secara default dan menolak semua permintaan sampai `FSA_API_KEYS` diisi. Port
  terbuka tanpa kunci bukan sekadar GPU gratis — itu cara siapa pun mencocokkan foto orang.

### 1. Siapkan server

```bash
sudo apt update
# 3.12 di Ubuntu 24.04, 3.11 di 22.04 — keduanya sama-sama didukung.
sudo apt install -y python3.12 python3.12-venv python3.12-dev \
    build-essential libglib2.0-0 libgomp1 nginx
# opsional, untuk codec video yang lebih luas:
sudo apt install -y ffmpeg
```

`build-essential` diperlukan karena `insightface` dibuild dari sdist. `libglib2.0-0` dan
`libgomp1` adalah dependensi runtime OpenCV dan ONNX Runtime.

**Ukuran mesin.** Tiap worker memuat salinan modelnya sendiri: ±600 MB RAM dengan landmark
aktif, di atas ±325 MB berkas model di disk. Dua worker pada 4 vCPU / 4 GB RAM adalah titik
awal yang masuk akal.

### 2. Deploy kodenya

```bash
sudo useradd -r -m -d /opt/face-api -s /usr/sbin/nologin faceapi
sudo -u faceapi git clone <repo> /opt/face-api/app     # atau rsync
cd /opt/face-api/app

sudo -u faceapi python3.12 -m venv /opt/face-api/venv
sudo -u faceapi /opt/face-api/venv/bin/pip install --upgrade pip "cython<3.1" "numpy<2.3"
sudo -u faceapi /opt/face-api/venv/bin/pip install -r requirements.txt
```

`requirements.txt`, bukan `requirements-dev.txt`: yang pertama adalah runtime saja, yang
kedua menambahkan pytest dan ruff yang tidak ada gunanya di server.

`cython` dan `numpy` dipasang lebih dulu dan dipatok karena `insightface` dibuild dari
sdist saat pip menemuinya — tanpa keduanya build-nya gagal di tengah dengan pesan yang
tidak menyebut sebabnya.

### 3. Konfigurasi

```bash
sudo -u faceapi cp .env.example /opt/face-api/app/.env
sudo chmod 600 /opt/face-api/app/.env      # berisi kunci API dan kunci HMAC

# Dua rahasia, dibuat terpisah:
python3 -c "import secrets; print('FSA_CHALLENGE_SECRET=' + secrets.token_urlsafe(32))"
python3 -c "import secrets; print('FSA_API_KEYS=tenancy-app:' + secrets.token_urlsafe(32))"
```

Isi `/opt/face-api/app/.env`:

```dotenv
FSA_DEBUG=false

# Siapa yang boleh memanggil. Format `label:kunci`, dipisah koma — labelnya muncul di log
# sehingga sebuah permintaan bisa ditelusuri tanpa kuncinya sendiri pernah tertulis.
FSA_API_KEYS=tenancy-app:K7f…
FSA_RATE_LIMIT_PER_MINUTE=60

# Wajib, dan harus identik di semua worker. Kalau kosong, aplikasi membuat kunci acak dan
# menulis warning — token liveness lalu gagal setiap kali proses restart atau ketika
# permintaan mendarat di worker lain.
FSA_CHALLENGE_SECRET=Xq2…

# Supaya model tidak ikut home direktori sementara milik systemd.
FSA_MODEL_ROOT=/opt/face-api/models

# Hanya relevan kalau ada browser yang memanggil langsung. Kalau pemanggilnya cuma Laravel,
# biarkan sempit — dan ingat CORS bukan kendali akses: curl mengabaikannya.
FSA_CORS_ORIGINS=https://app.example.com
```

**Kunci yang sama dipasang di sisi Laravel** sebagai `FACE_API_KEY` di `.env`
`../tenancy-app`, bersama `FACE_API_URL=https://face-api.example.com`. Keduanya harus
cocok; kalau tidak, setiap absensi wajah berakhir dengan `503 face_service_unavailable`.

> `EnvironmentFile` milik systemd membaca berkas ini apa adanya — tidak ada ekspansi shell,
> dan tanda `#` di tengah baris ikut terbaca sebagai nilai. Taruh komentar di barisnya
> sendiri.

Unduh model lebih dulu supaya permintaan pertama tidak menunggu ±30 detik:

```bash
sudo -u faceapi FSA_MODEL_ROOT=/opt/face-api/models /opt/face-api/venv/bin/python -c \
"from insightface.app import FaceAnalysis; \
FaceAnalysis(name='buffalo_l', root='/opt/face-api/models', \
allowed_modules=['detection','recognition','landmark_3d_68']).prepare(ctx_id=-1)"
```

`landmark_3d_68` tidak opsional kalau liveness dipakai: metrik yaw, pitch, mata, dan mulut
semuanya dibaca dari 68 titik itu.

### 4. systemd

`/etc/systemd/system/face-api.service`:

```ini
[Unit]
Description=Face Similarity API
After=network.target

[Service]
Type=exec
User=faceapi
Group=faceapi
WorkingDirectory=/opt/face-api/app
EnvironmentFile=/opt/face-api/app/.env
ExecStart=/opt/face-api/venv/bin/uvicorn app.main:app \
    --host 127.0.0.1 --port 8001 --workers 2 --timeout-keep-alive 65
Restart=always
RestartSec=5
# Model butuh waktu dimuat saat start.
TimeoutStartSec=120

# Pengetatan dasar
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/face-api/models

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now face-api
sudo journalctl -u face-api -f
```

`PrivateTmp=true` penting dan bukan sekadar pengetatan: setiap upload ditulis ke berkas
sementara sebelum di-decode, lalu dihapus. Dengan `/tmp` privat, potongan video orang tidak
pernah berbagi direktori dengan proses lain di mesin itu.

`--host 127.0.0.1` juga bukan sekadar kebiasaan. Servis ini tidak bicara TLS sendiri; nginx
yang melakukannya, dan mengikat ke `0.0.0.0` berarti port polos itu tetap terjangkau
seandainya firewall salah setel.

**Jumlah worker.** Tiap worker memuat salinan modelnya sendiri (±600 MB RAM dengan landmark
aktif) dan ONNX Runtime sendiri memakai beberapa thread per inference. Mulai dari
`--workers 2` pada mesin 4 vCPU, dan biarkan `FSA_MAX_CONCURRENT_INFERENCES=2`. Menaikkan
worker melebihi jumlah core hanya membuat request saling berebut CPU dan memperlambat semua.

Satu akibat yang perlu diingat: `FSA_RATE_LIMIT_PER_MINUTE` dihitung **di dalam satu
proses**. Dua worker berarti dua hitungan dan batas efektif dua kali lipat. Untuk batas yang
benar-benar global, pakai `limit_req` nginx di bawah.

**Octane/Swoole tidak relevan di sini** — ini proses Python terpisah; Laravel cukup
memanggilnya lewat HTTP.

### 5. nginx

Zona pembatas didefinisikan di konteks `http` (biasanya `/etc/nginx/nginx.conf`), bukan di
dalam `server`:

```nginx
# Dikunci pada kunci API, bukan pada alamat IP: seluruh trafik datang dari satu server
# Laravel, sehingga membatasi per-IP akan membatasi semua penyewa sebagai satu.
limit_req_zone $http_x_api_key zone=faceapi:10m rate=60r/m;
```

```nginx
server {
    # Bentuk lama, karena nginx di Ubuntu 22.04 (1.18) dan 24.04 (1.24) belum
    # mengenal direktif `http2 on;` yang baru ada sejak 1.25.1. Pada nginx yang
    # lebih baru, pisahkan menjadi `listen 443 ssl;` dan `http2 on;`.
    listen 443 ssl http2;
    server_name face-api.example.com;

    ssl_certificate     /etc/letsencrypt/live/face-api.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/face-api.example.com/privkey.pem;

    # 5 foto × 8 MB + video 64 MB = 104 MB, ditambah overhead multipart. Angka yang lebih
    # kecil dari ini menolak unggahan yang justru diizinkan servisnya, dengan 413 dari
    # nginx yang tidak pernah sampai ke log aplikasi.
    client_max_body_size 128m;
    client_body_timeout  120s;

    # Probe kesiapan: tanpa kunci, tanpa pembatas laju. Load balancer tidak membawa
    # kredensial, dan yang diungkap endpoint ini — hidup, dan apakah modelnya selesai
    # dimuat — sudah diketahui pemanggilnya dari kenyataan bahwa servisnya menjawab.
    location = /api/v1/health {
        proxy_pass http://127.0.0.1:8001;
        proxy_set_header Host $host;
        access_log off;
    }

    # Halaman demo dan Swagger tidak punya alasan untuk ada di server produksi.
    location ~ ^/(demo|docs|redoc|openapi\.json)$ {
        return 404;
    }

    location / {
        limit_req zone=faceapi burst=20 nodelay;
        limit_req_status 429;

        proxy_pass http://127.0.0.1:8001;
        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Inference bisa memakan waktu pada klip panjang.
        proxy_read_timeout 180s;
        proxy_request_buffering on;
    }
}
```

Kalau API ini hanya dipanggil dari backend Laravel dan bukan dari browser, yang paling aman
adalah tidak mengeksposnya ke publik sama sekali: biarkan mendengarkan di `127.0.0.1` bila
keduanya satu mesin, atau di jaringan privat bila terpisah. Kunci API tetap dipasang —
jaringan privat bukan alasan untuk melewatkannya.

### 6. Verifikasi setelah deploy

Empat perintah, dan masing-masing memeriksa hal yang berbeda:

```bash
# 1. Hidup, dan modelnya benar-benar termuat.
curl -s https://face-api.example.com/api/v1/health | jq
#    -> {"status":"ok", ..., "model_loaded":true}

# 2. Tanpa kunci harus ditolak. Kalau ini menjawab 200, FSA_API_KEYS tidak terbaca.
curl -s -o /dev/null -w '%{http_code}\n' \
  -X POST https://face-api.example.com/api/v1/liveness/challenge
#    -> 401

# 3. Dengan kunci harus memberi aksi.
curl -s -X POST https://face-api.example.com/api/v1/liveness/challenge \
  -H "X-API-Key: $FSA_KEY" | jq '.actions[].action'

# 4. Halaman demo tidak boleh terbuka.
curl -s -o /dev/null -w '%{http_code}\n' https://face-api.example.com/demo
#    -> 404
```

Pemeriksaan nomor 2 adalah yang paling mudah terlewat. Servis yang `FSA_API_KEYS`-nya gagal
terbaca **tidak** menjadi terbuka — ia menjawab `503 api_keys_not_configured` untuk
semuanya — jadi kegagalannya terlihat sebagai "servis mati", bukan sebagai lubang. Yang
justru berbahaya adalah `FSA_DEBUG=true` tertinggal: itulah satu-satunya keadaan yang
membuat permintaan tanpa kunci dilayani.

### 7. Daftar periksa sebelum go-live

- [ ] `FSA_API_KEYS` terisi, dan kunci yang sama ada di `FACE_API_KEY` milik tenancy-app
- [ ] `FSA_CHALLENGE_SECRET` terisi, identik di semua worker
- [ ] `FSA_DEBUG=false`
- [ ] `.env` ber-mode `600` dan dimiliki `faceapi`
- [ ] Model sudah diunduh ke `FSA_MODEL_ROOT`, `model_loaded: true`
- [ ] `client_max_body_size` minimal `128m`
- [ ] `/demo` dan `/docs` menjawab 404
- [ ] TLS aktif — yang lewat di sini adalah foto wajah orang
- [ ] Lisensi `buffalo_l` sudah diputuskan (lihat **Batasan**) — bobotnya non-komersial

### 8. Operasi harian

- **Log.** Semuanya ke stdout, jadi `journalctl -u face-api` sudah cukup. Tiap permintaan
  menulis satu baris ringkasan: label kunci pemanggil, decision, skor, jumlah frame, durasi.
  Tidak ada foto, tidak ada kunci, dan tidak ada embedding yang ikut tertulis.
- **Memutar kunci.** Tambahkan kunci baru di samping yang lama
  (`FSA_API_KEYS=lama:…,baru:…`), restart, pindahkan Laravel ke kunci baru, lalu hapus yang
  lama dan restart lagi. Dua langkah, tanpa jeda layanan.
- **Update.** `git pull && pip install -r requirements.txt && systemctl restart face-api`.
  Restart memutus permintaan yang sedang berjalan dan menghabiskan ±30 detik memuat model;
  lakukan saat sepi, atau jalankan dua instance bergantian di belakang load balancer.
- **Yang perlu di-backup: hanya `.env`.** Servis ini stateless; sisanya bisa dibangun ulang
  dari repo dan unduhan model.

---

## Batasan yang perlu diketahui

- **Liveness ini bukan anti-spoofing lengkap.** Challenge-response menghadang serangan paling
  umum (foto diangkat, video lama diputar ulang), tapi tidak menghadang deepfake real-time
  atau orang yang benar-benar mengikuti instruksi sambil menyamar. Untuk risiko tinggi,
  tambahkan model anti-spoof (texture/depth) atau verifikasi manual.
- **Token bisa dipakai ulang selama masih berlaku.** Servisnya stateless, jadi ia tidak tahu
  sebuah token sudah dipakai. Simpan `challenge_id` di sisi pemanggil dan tolak yang kedua.
  tenancy-app melakukannya dengan indeks unik pada `attendance_face_verifications`, ditulis
  sebelum permintaan ini dikirim — lihat bagian integrasi. Yang harus diklaim adalah
  `challenge_id` **milik token itu sendiri**, dibaca dari payload token, bukan dari
  field terpisah yang dikirim klien: id pilihan pemanggil membuat satu token sah
  bisa dibelanjakan dengan berapa pun id, yang artinya sekali pakai hanya pada
  namanya. Respons `/face/verify` menyebutkan `liveness.challenge_id` supaya
  klaim itu bisa diperiksa ulang. Pendekkan juga `FSA_CHALLENGE_TTL_SECONDS`.
- **Bukan keputusan final.** Untuk kasus berisiko tinggi, perlakukan `inconclusive` sebagai
  antrean review manusia, bukan penolakan otomatis.
- `blink` paling tidak andal dari semua aksi: kedipan berlangsung ~0,2 detik sementara video
  hanya di-sample sekitar 10 fps. Ia tidak masuk kolam aksi default; aktifkan hanya kalau
  `FSA_LIVENESS_MAX_SAMPLED_FRAMES` dinaikkan dan klipnya pendek.
- Akurasi turun pada wajah kecil (< 40 px), backlight kuat, masker, atau sudut ekstrem.
  `warnings` di respons memberi tahu kondisi ini.
- Model `buffalo_l` dan bobotnya memakai lisensi non-komersial dari InsightFace — cek
  lisensinya sebelum dipakai di produk komersial.
- Data biometrik: meski tidak ada yang disimpan di sini, log akses dan aturan retensi tetap
  jadi tanggung jawab aplikasi pemanggil.
