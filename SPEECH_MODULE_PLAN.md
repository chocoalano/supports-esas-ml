# Implementation Plan — Modul Speech (STT + TTS)

Status (2026-09-30): **Revisi 3 — ADR terkunci, implementasi langkah 1–12 selesai.**
Produksi **belum** disetujui; lihat §R.5 untuk gerbang yang tersisa. Bagian §A–§M di bawah
adalah Revisi 2 dan dibiarkan sebagai catatan; bila bertentangan dengan §R, **§R yang berlaku.**

---

## R. Revisi 3 — Architecture Decision Record & status implementasi

### R.1 Keputusan terkunci (jangan dibuka ulang tanpa blocker teknis baru yang terbukti)

| # | Keputusan | Wujud di kode |
|---|---|---|
| 1 | Caller hanya memakai **alias voice**; id provider bukan kontrak, tidak ada escape hatch | `FSA_TTS_VOICE_ALIASES`, `TextToSpeechService._resolve_voice`; `X-Speech-Voice` = alias |
| 2 | `/speech/voices` **statis dari config**, nol network | route membaca `Settings`; `VOICE_CACHE_TTL`/`VOICE_FALLBACK` dihapus |
| 3 | **Dua decode**; Protocol tetap `transcribe(path, options)` | `inspect_audio` (validasi) + faster-whisper (decode sendiri) |
| 4 | `FSA_RUNTIME_ROLE` = `face` \| `speech` \| `all`, default `face`, fail-fast | `RuntimeRole`, mounting di `create_app()` |
| 5 | `small` + int8 = **baseline benchmark**, bukan keputusan produksi | default `FSA_STT_MODEL=small` |
| 6 | Timeout berlapis: nginx 120 s, Laravel 130 s. Lambat → turunkan cap durasi dulu | `FSA_STT_MAX_AUDIO_SECONDS`; skrip benchmark memprediksi biaya pada cap |
| + | Request strict: field tak dikenal → 422 (JSON **dan** multipart) | `extra="forbid"` pada `SynthesizeRequest` dan `TranscribeForm` |
| + | `rate`/`volume`: sintaks **dan** rentang −100..+100 | `parse_prosody_percent` |
| + | `Content-Disposition: inline; filename="speech.mp3"` | route synthesize |
| + | Dependency: `faster-whisper~=1.2.1`, `edge-tts~=7.2.8`, `av>=17.1,<18` | `requirements.txt`; `av` juga di `requirements-test.txt` |

### R.2 Detail yang diputuskan saat implementasi (tidak mengubah keputusan di atas)

1. **Format `FSA_TTS_VOICE_ALIASES` = JSON satu baris**, bukan CSV — tiap alias punya lima
   field (`id`, `provider_voice`, `label`, `language`, `gender`), `label` teks bebas.
2. **Validasi semantik Speech hanya berjalan bila role melayani Speech.** Salah ketik setting
   Speech tidak pernah bisa menghentikan boot runtime Face. Galatnya `SpeechConfigError`, bukan
   `ValueError`: pydantic membungkus `ValueError` dengan menyalin seluruh input settings
   (termasuk `FSA_API_KEYS`) ke pesan galat, dan pesan itu masuk log boot.
3. **Content-Type default diperluas** dengan `video/webm`, `video/mp4`, `video/3gpp`,
   `audio/x-aac`, `application/octet-stream`. Guzzle (di bawah `Http::attach()` Laravel) mengirim
   `.webm` sebagai `video/webm`. Lapis ini bukan otoritas; tipe yang jelas salah (`image/png`,
   `text/plain`) tetap 415.
4. **Hardening decoder:** upload ditulis ke temp file **tanpa ekstensi** (nama file caller tidak
   pernah memengaruhi probe FFmpeg — di validator maupun di faster-whisper), `av.open` dengan
   `protocol_whitelist=file`, dan **allow-list demuxer audio** (`wav`, `mp3`, `aac`, `ogg`,
   `flac`, `matroska,webm`, `mov,mp4,…`). Dicoba: payload HLS/polyglot tidak memicu fetch
   network pada build av 17.1.0, jadi ini defense in depth, bukan perbaikan kebocoran nyata.
5. **Decode validasi ikut antre di semaphore STT** bersama inferensi — CPU Speech tetap satu job
   pada satu waktu. Harga: upload invalid menunggu di belakang transkripsi yang berjalan.
6. **Allow-list bahasa juga diterapkan pada bahasa hasil deteksi** (bila caller tidak meminta
   bahasa). Whisper sering mendengar bahasa Indonesia sebagai Melayu (`ms`).
7. **Edge: timeout soket library = setengah `FSA_TTS_TIMEOUT_SECONDS`.** Provider yang macet
   ditangkap library-nya sendiri, yang lalu menutup websocket dengan benar. Timeout wall-clock
   service tetap batas luar, tetapi bekerja dengan cancel, dan stream yang di-cancel tidak bisa
   menutup soketnya dengan rapi (terverifikasi nyata). Timeout dari provider → 504 yang sama.
8. **`/` dan `/demo` hanya ada di role `face`/`all`** (halaman itu memanggil `/face/*`).
9. **`/health` tidak berubah** di semua role. Di role `speech` ia tetap membangun objek
   `InsightFaceEngine` (ringan, tanpa import) untuk mengisi field `model` — dibuktikan dengan
   test subprocess bahwa `cv2`/`insightface` tetap tidak pernah ter-import.
10. 422 dari validasi pydantic (field tak dikenal, tipe salah) memakai bentuk bawaan FastAPI
    `{"detail": [...]}`, bukan envelope `{"error": ...}` — sama seperti `/liveness/challenge`
    hari ini; handler baru tidak ditambahkan (§I.3). Galat semantik (`invalid_request`,
    `voice_not_available`, dst.) memakai envelope.

### R.3 Temuan operasional (wajib masuk deployment)

- **Model diunduh saat request pertama bila belum ada** — terukur 79,7 s untuk `small`. Di
  produksi: unduh model saat provisioning ke `FSA_STT_MODEL_ROOT`, lalu set `HF_HUB_OFFLINE=1`
  di unit Speech supaya runtime tidak pernah menghubungi huggingface.co.
- `client_max_body_size` nginx untuk `/api/v1/speech/` harus ≥ `FSA_STT_MAX_AUDIO_BYTES`
  (+ overhead multipart), mis. `26m`.
- Rate limit in-process di mode `all` tetap satu bucket bersama (keterbatasan dev yang sudah
  diketahui); isolasi rate budget produksi = dua proses + dua zona nginx.

### R.4 Status langkah implementasi (urutan dari ADR)

| # | Langkah | Status |
|---|---|---|
| 1–9 | errors, Settings + `FSA_RUNTIME_ROLE`, `.env.example`, validasi audio, service, schema strict, fake, test isolasi/lazy/role, routes/deps/mounting | ✅ |
| 10 | provider faster-whisper | ✅ + verifikasi nyata (macOS): transkrip Indonesia tepat |
| 11 | provider Edge | ✅ + verifikasi nyata: ketiga alias, jalur timeout & output-cap |
| 12 | dependency | ✅ `pip check` bersih; tidak satu pun paket Face berubah versi |
| 13 | Matrix Linux py3.10/3.11/3.12 | ⏳ **belum** — tidak ada Linux/Docker di mesin ini |
| 14 | Benchmark nyata | ⏳ alat siap: `scripts/benchmark_stt.py` (§R.6) |
| 15 | Model produksi + `MemoryMax` | ⏳ menunggu 14 |
| 16 | README / AGENTS | ⏳ sengaja setelah 15 |
| 17 | systemd / nginx | ⏳ `MemoryMax=<hasil capacity planning>` |
| 18 | Smoke produksi | ⏳ |

Bukti regresi saat ini: OpenAPI runtime `role=face` **byte-identik** dengan commit `f1f002f`;
tidak satu pun berkas Face/Liveness, `security.py`, `throttle.py`, atau test lama berubah.

### R.5 Gerbang produksi yang tersisa

Linux matrix hijau (install + `pip check` + import `av`/`faster_whisper`/`edge_tts`/`cv2` +
`make test` + satu verifikasi Face nyata) · regresi penuh hijau · benchmark memori & latency
di host target · model produksi dipilih · `MemoryMax` ditetapkan dari peak RSS · unit systemd
terpisah aktif · zona nginx terpisah aktif · isolasi role terbukti di host · smoke Face hijau.
Bila hasil Linux berbeda dengan resolver/dry-run, **hasil Linux yang berlaku**; bila dependency
Face harus berubah, **STOP dan laporkan**.

### R.5a Status langkah 13–18 (2026-09-30)

Semua perangkat sudah ada dan terverifikasi di macOS; **belum ada satu pun angka Linux**,
karena setiap job GitHub Actions ditolak: *"The job was not started because your account is
locked due to a billing issue."* Yang siap dijalankan begitu Actions (atau host Linux) tersedia:

| Langkah | Perangkat | Status |
|---|---|---|
| 13 | `.github/workflows/speech-validation.yml` job `matrix` (py3.10/3.11/3.12, Ubuntu 24.04 x86_64): install penuh + bukti insightface dibuild dari sdist, perbandingan versi paket per paket dengan install Face-only `f1f002f`, `pip check`, import, ruff, test Speech/Face/penuh, smoke Face nyata base-vs-feature & dua urutan import, smoke Speech nyata offline | siap; diblokir billing |
| 14 | job `benchmark`: korpus `scripts/build_stt_corpus.py` (FLEURS + LibriVox Indonesia, manusia nyata), `base`/`small` × 2/4 thread, 3 run warm | siap; diblokir billing |
| 15C | provisioning: `scripts/provision_stt_model.py`; model hilang → 503 `model_not_provisioned` dalam ~30 ms; cek disk saat boot | ✅ (commit `eb2cd8d`) |
| 17 | `deploy/systemd/`, `deploy/nginx/`, `deploy/env/speech.env.example` | ditulis; `MemoryMax` menunggu 14 |
| 18 | `scripts/smoke_deploy.py` (25 cek HTTP + cek systemd/strace); job `staging` men-deploy ke VM Ubuntu 24.04 sesuai README lalu menjalankannya | 25/25 hijau secara lokal tanpa nginx/systemd; job diblokir billing |

**Aturan `MemoryMax` yang diusulkan** (dipakai job `staging`, angka produksi diputuskan saat
review): `MemoryMax = ceil(1,5 × puncak RSS proses terukur / 256 MiB) × 256 MiB` untuk model dan
jumlah thread produksi. Puncak proses (`ru_maxrss`, termasuk saat load) — bukan RSS setelah
load — karena itulah yang harus muat. Margin 1,5× menampung fragmentasi heap selama uptime,
empat buffer TTS bersamaan (≤ 10 MB masing-masing), spooling multipart, dan arena glibc per
thread; tetap cukup ketat untuk membunuh kebocoran sungguhan jauh sebelum menekan Face.
`MemoryHigh` sengaja tidak dipakai: working set Speech hampir seluruhnya bobot model, jadi
throttling reclaim hanya berarti latensi melonjak tanpa ada yang bisa dilepas.

### R.6 Prosedur benchmark (langkah 14)

Di host target, satu worker, `FSA_STT_MAX_CONCURRENT=1`:

```bash
.venv/bin/python scripts/benchmark_stt.py samples/ \
    --models base small medium --model-root /var/lib/fsa/models \
    --cpu-threads 2 --language id --show-text \
    --output benchmark.md --json benchmark.json
```

`samples/` berisi satu folder per kategori (nama folder = kategori), tiap audio boleh punya
`<nama>.txt` berisi ucapan sebenarnya. Kategori minimal: Indonesia jelas, percakapan, pria,
wanita, tenang, bising ringan, pendek, panjang — **rekaman nyata**, bukan hasil TTS (audio TTS
terlalu mudah; uji coba alat dengan TTS memberi WER 0,0). Tiap model dijalankan di interpreter
baru (RSS bersih), diunduh dulu lalu diukur dengan `HF_HUB_OFFLINE=1`. Laporan berisi RSS idle /
model termuat / puncak / sesudah, RTF, WER, dan **biaya terfit `tetap + per-detik`** yang
memprediksi durasi satu klip sepanjang `FSA_STT_MAX_AUDIO_SECONDS` terhadap budget 120 s.
Pilih model terkecil yang kualitas transkripnya dapat diterima bisnis (baca transkripnya, WER
menghukum "8" vs "delapan"); `MemoryMax` = peak RSS model itu + headroom operasional.

Revisi 2 menggabungkan architecture decision dari pemilik sistem (Prompt 2) dan hasil
verifikasi dependency yang benar-benar dijalankan, bukan diasumsikan. Perubahan terbesar dari
Revisi 1: **produksi memakai dua runtime terpisah dari satu codebase**, flag STT/TTS
independen, validasi audio berlapis dengan decoder sebagai otoritas, dan TTS MP3-only tanpa
field `format`.

Baseline sebelum perubahan: 116 test passed, `ruff check .` bersih, commit `f1f002f`.

> ### ⚠ Dua hal yang perlu keputusan sebelum implementasi
> 1. **Temuan baru kelas-blocker:** PyAV dan OpenCV sama-sama membundel FFmpeg sendiri dan
>    bertabrakan dalam satu proses (§L-1). Tidak menggagalkan rencana — justru memperkuat
>    keputusan dua runtime — tapi mengubah apa yang harus dites.
> 2. **6 DECISION REQUIRED** di §K, termasuk satu kontradiksi nyata antara §6 dan §7 instruksi
>    Anda (identifier voice milik provider vs kontrak yang tidak boleh bocor).

---

## A. Revised architecture

### A.1 Prinsip yang mengikat

Face/Liveness adalah production-critical (absensi). Setiap keputusan di bawah ini diturunkan
dari satu aturan: **Speech tidak boleh menurunkan availability, latency, atau reliability
Face.** Konsekuensi konkretnya: Speech tidak berbagi proses, tidak berbagi semaphore, tidak
berbagi bucket rate limit, dan tidak berbagi memori dengan Face di produksi.

### A.2 Development / test — satu proses

```
                       uvicorn :8001 (satu proses)
                                 │
      ┌──────────────────────────┼──────────────────────────┐
      ▼                          ▼                          ▼
/api/v1/face/*          /api/v1/liveness/*         /api/v1/speech/*
      │                          │                          │
      └──────────┬───────────────┘                          │
                 ▼                                          ▼
         FaceEngine (fake di test)              SttEngine / TtsEngine (fake di test)
         FrameSampler (cv2, lazy)               audio validator (av, lazy)
                 │                                          │
      get_inference_limiter()              get_stt_limiter() / get_tts_limiter()
              Semaphore(2)                    Semaphore(1)     Semaphore(4)
```

`FSA_STT_ENABLED=true`, `FSA_TTS_ENABLED=true`. Semua endpoint hidup dalam satu proses supaya
suite dan pengembangan lokal tetap sederhana. **Di sini — dan hanya di sini — cv2 dan av dimuat
dalam satu address space** (lihat §L-1), dan rate limit dihitung dalam satu bucket bersama.
Keduanya dapat diterima untuk dev, dan **tidak boleh** dipakai di produksi.

### A.3 Produksi — dua runtime, satu codebase

```
                                  NGINX  (satu-satunya yang menghadap keluar)
                                    │
           ┌────────────────────────┴────────────────────────┐
           │ location /api/v1/face/                          │ location /api/v1/speech/
           │ location /api/v1/liveness/                      │
           │ limit_req zone=face                             │ limit_req zone=speech
           ▼                                                 ▼
   ┌───────────────────────────────┐              ┌───────────────────────────────┐
   │ FACE RUNTIME   127.0.0.1:8001 │              │ SPEECH RUNTIME 127.0.0.1:8002 │
   │ uvicorn --workers 2           │              │ uvicorn --workers 1           │
   │                               │              │                               │
   │ FSA_STT_ENABLED=false         │              │ FSA_STT_ENABLED=true          │
   │ FSA_TTS_ENABLED=false         │              │ FSA_TTS_ENABLED=true          │
   │                               │              │                               │
   │ router speech TIDAK di-mount  │              │ router speech di-mount        │
   │ av / faster_whisper / edge_tts │             │ cv2 tidak pernah di-import    │
   │   TIDAK pernah di-import      │              │   (tak ada request video)     │
   │                               │              │                               │
   │ InsightFace buffalo_l         │              │ faster-whisper + edge-tts     │
   │ Semaphore(2) face             │              │ Semaphore(1) STT, (4) TTS     │
   └───────────────────────────────┘              └───────────────────────────────┘
        artifact & git commit yang sama, .env yang berbeda
```

Yang didapat dari topologi ini, dan tidak bisa didapat dengan cara lain:

| Kegagalan Speech | Akibat ke Face |
|---|---|
| OOM saat memuat model Whisper | **Tidak ada.** Proses berbeda, OOM killer membunuh Speech |
| Provider Edge TTS down / lambat | **Tidak ada.** Face tidak punya soket ke sana |
| Transkripsi 5 menit memenuhi CPU | Terbatas: Speech 1 worker × 1 concurrency, dianggarkan terhadap core |
| Tabrakan simbol FFmpeg (§L-1) | **Tidak ada.** `av` tidak pernah dimuat di Face runtime |
| Rate limit dihabiskan klien TTS | **Tidak ada.** Bucket in-process terpisah, zona nginx terpisah |
| Crash / traceback / segfault | **Tidak ada.** Batas proses |

**Isolasi didapat gratis dari batas proses, bukan dari kode.** Tidak ada kode isolasi yang perlu
ditulis, dipelihara, atau bisa salah — itulah sebabnya ini lebih kuat daripada semaphore mana
pun.

---

## B. Revised file plan

### B.1 File baru

| File | Isi |
|---|---|
| `app/api/routes/speech.py` | 4 route, semua `GuardDep`. Tidak mengandung nama provider. |
| `app/schemas/speech.py` | `TranscriptionResponse`, `TranscriptSegmentReport`, `SynthesizeRequest`, `VoicesResponse`, `VoiceReport`, `SpeechCapabilitiesResponse` |
| `app/services/speech/__init__.py` | kosong |
| `app/services/speech/audio.py` | `AUDIO_MAGIC`, `looks_like_audio()`, `inspect_audio()` (lazy `av`) — validator berlapis, lihat §E |
| `app/services/speech/stt.py` | Protocol `SpeechToTextEngine`, `TranscribeOptions`, `Transcript`, `TranscriptSegment`, `SpeechToTextService` |
| `app/services/speech/tts.py` | Protocol `TextToSpeechEngine`, `SynthesisRequest`, `Synthesis`, `Voice`, `TextToSpeechService` |
| `app/services/speech/providers/__init__.py` | kosong |
| `app/services/speech/providers/faster_whisper.py` | `FasterWhisperEngine` — lazy import, `load()` double-checked `threading.Lock` |
| `app/services/speech/providers/edge.py` | `EdgeTtsEngine` — lazy import, async, timeout, output berbatas |
| `tests/test_audio_validation.py` | Pipeline validasi berlapis, seluruh edge case §19 |
| `tests/test_speech_access.py` | Guard + rate limit 4 route |
| `tests/test_speech_transcribe.py` | |
| `tests/test_speech_synthesize.py` | |
| `tests/test_speech_voices.py` | |
| `tests/test_speech_capabilities.py` | Termasuk bukti tidak memuat model |
| `tests/test_speech_isolation.py` | Limiter terpisah, lazy import, koeksistensi cv2+av (§L-1) |

### B.2 File yang diubah — semuanya aditif

| File | Perubahan |
|---|---|
| `app/core/config.py` | Field speech + registrasi 4 field CSV baru ke `_split_csv` **dan** `NoDecode` |
| `app/core/errors.py` | Subclass `AppError` baru (§I) |
| `app/api/deps.py` | `get_stt_engine`, `get_tts_engine`, `get_stt_limiter`, `get_tts_limiter`, `get_stt_service`, `get_tts_service`, alias `Annotated` |
| `app/main.py` | Mount router speech bila `stt_enabled or tts_enabled`; validasi config speech di lifespan (warn, **tidak** raise) |
| `.env.example` | Blok Speech, kedua flag `false` |
| `requirements.txt` | `faster-whisper`, `edge-tts`, pin `av` (§D) |
| `requirements-test.txt` | `av` saja |
| `tests/conftest.py` | `FakeSttEngine`, `FakeTtsEngine`, fixture, `cache_clear()` untuk getter baru |
| `tests/test_settings.py` | `CSV_FIELDS` + 4 field baru |
| `README.md` | Endpoint, tabel error, tabel config, **bagian deployment dua runtime**, batasan |
| `AGENTS.md` | Overview + angka diperbarui |

### B.3 Tidak boleh disentuh

Perubahan pada daftar ini adalah pelanggaran requirement, bukan pilihan desain.

| Berkas / simbol | Alasan |
|---|---|
| `app/services/verification.py` | Logika keputusan Face. Nol perubahan. |
| `app/services/liveness.py`, `challenge.py`, `face_engine.py`, `video.py` | Idem |
| `app/services/media.py` | **Tidak diperluas untuk audio.** Audio dapat modulnya sendiri supaya jalur upload Face tidak tersentuh sama sekali. |
| `app/api/security.py` | Format key, `compare_digest`, label log, urutan |
| `app/api/throttle.py` | Lihat §11 instruksi — Option A dipilih, jadi nol perubahan sekarang |
| `guard()`, `scope()`, `limiter_for()`, `GuardDep`, `TENANT_HEADER` | Urutan key→rate dan bucket-nya |
| `get_inference_limiter()` | Milik Face. Semantik `FSA_MAX_CONCURRENT_INFERENCES` tidak diperluas. |
| `/health` + `HealthResponse` | Tetap satu-satunya endpoint terbuka; schema tidak berubah. Status muat model STT **tidak** masuk sini. |
| `register_exception_handlers()` | Tidak ada handler `RequestValidationError` baru (akan mengubah respons existing) |
| `app/schemas/verification.py`, `liveness.py` | Kontrak Face |
| Route `/`, `/demo`, `static/index.html` | Tidak ada UI speech di v1 |
| Nilai default field existing | Tidak satu pun digeser |

---

## C. Revised Settings

Prefix `FSA_` (rule 10). Dinamai **bernamespace** `STT_`/`TTS_` — sengaja berbeda dari gaya
Face yang tanpa namespace (`FSA_MAX_VIDEO_BYTES`), justru supaya masalah yang Anda tunjuk di §8
tidak terulang: setting Speech tidak akan pernah bisa disalahartikan sebagai memperluas
semantik setting Face.

### C.1 Feature flag

| Var | Default | Alasan default | Face runtime | Speech runtime |
|---|---|---|---|---|
| `FSA_STT_ENABLED` | `false` | Upgrade deployment existing tidak boleh mengaktifkan workload baru | `false` | `true` |
| `FSA_TTS_ENABLED` | `false` | Idem | `false` | `true` |

**Master `FSA_SPEECH_ENABLED` tidak dibuat.** Anda meminta alasannya: dua flag sudah
mengekspresikan keempat kondisi, sementara master flag menciptakan pertanyaan presedensi
(`SPEECH_ENABLED=false` + `STT_ENABLED=true` = apa?) yang tepat jenis kebingungan yang Anda
larang. Kill-switch insiden tetap ada, cukup "set keduanya `false`", dan di topologi §12
mematikan Speech sebenarnya berarti nginx berhenti merutekan ke `:8002` — satu baris, bukan env.

### C.2 STT

| Var | Default | Alasan default | Face RT | Speech RT |
|---|---|---|---|---|
| `FSA_STT_PROVIDER` | `faster_whisper` | Satu-satunya provider v1; nilai tak dikenal diteriakkan saat boot | — | `faster_whisper` |
| `FSA_STT_MODEL` | `small` | `medium`+ berisiko OOM; `small` int8 kompromi akurasi/memori. **Angka final menyusul benchmark §14** | — | `small` |
| `FSA_STT_MODEL_ROOT` | — | Cermin `FSA_MODEL_ROOT`; wajib diisi di systemd agar `PrivateTmp` tidak menelan unduhan | — | `/var/lib/fsa/models` |
| `FSA_STT_DEVICE` | `cpu` | Terpisah dari `ctx_id` supaya setting device Face tidak ditafsir ulang | — | `cpu` |
| `FSA_STT_COMPUTE_TYPE` | `int8` | Memori terkecil di CPU | — | `int8` |
| `FSA_STT_CPU_THREADS` | `0` | 0 = default library. Berkali dengan semaphore — lihat §G | — | `2` (dianggarkan) |
| `FSA_STT_BEAM_SIZE` | `1` | Greedy: tercepat, memori terkecil. Tidak diekspos ke caller (§17) | — | `1` |
| `FSA_STT_VAD_FILTER` | `true` | Melompati hening → lebih cepat. Internal, tidak diekspos | — | `true` |
| `FSA_STT_DEFAULT_LANGUAGE` | — | Kosong = autodetect | — | `id` |
| `FSA_STT_ALLOWED_LANGUAGES` | — (CSV) | Kosong = semua. **Field CSV** | — | `id,en` |
| `FSA_STT_WARM_UP_ON_STARTUP` | `false` | Rule 10: upgrade tidak boleh menambah memori. Cermin nama `FSA_WARM_UP_ON_STARTUP` | `false` | `false` |
| `FSA_STT_MAX_CONCURRENT` | `1` | Rekomendasi Anda di §8. Konservatif sampai benchmark ada | — | `1` |
| `FSA_STT_MAX_AUDIO_BYTES` | `26214400` (25 MB) | Proteksi **pertama**, berlaku saat streaming ke disk | — | `26214400` |
| `FSA_STT_MAX_AUDIO_SECONDS` | `300` | Proteksi **kedua** dan berbeda (§3). Membatasi biaya CPU, yang tidak dilakukan byte cap | — | `300` |
| `FSA_STT_ALLOWED_AUDIO_TYPES` | `audio/wav,audio/x-wav,audio/mpeg,audio/mp4,audio/m4a,audio/ogg,audio/webm,audio/flac,audio/3gpp` (CSV) | Lapis 2 saja, bukan otoritas (§E). **Field CSV** | — | idem |

`supported_containers` di `/speech/capabilities` **diturunkan** dari tabel magic byte di kode,
bukan dari config ketiga — satu sumber kebenaran, dan tidak bisa berbohong tentang apa yang
benar-benar dikenali.

### C.3 TTS

| Var | Default | Alasan default | Face RT | Speech RT |
|---|---|---|---|---|
| `FSA_TTS_PROVIDER` | `edge` | Satu-satunya provider v1 | — | `edge` |
| `FSA_TTS_DEFAULT_VOICE` | `id-ID-ArdiNeural` | Populasi pengguna Indonesia | — | idem |
| `FSA_TTS_DEFAULT_RATE` | `+0%` | Netral; dipakai bila request tidak menyebut | — | `+0%` |
| `FSA_TTS_DEFAULT_VOLUME` | `+0%` | Idem | — | `+0%` |
| `FSA_TTS_MAX_TEXT_CHARS` | `3000` | ±3 menit bicara. Membatasi output tanpa perlu menebak bitrate | — | `3000` |
| `FSA_TTS_MAX_OUTPUT_BYTES` | `10485760` (10 MB) | **Wajib.** Output provider remote tidak dibatasi cap input kita | — | `10485760` |
| `FSA_TTS_TIMEOUT_SECONDS` | `30.0` | Batas wall-clock ke pihak ketiga | — | `30.0` |
| `FSA_TTS_MAX_CONCURRENT` | `4` | Network-bound, bukan CPU — ceiling lebih tinggi wajar | — | `4` |
| `FSA_TTS_VOICE_CACHE_TTL_SECONDS` | `3600` | Katalog suara jarang berubah; cache in-process, non-persisten | — | `3600` |
| `FSA_TTS_VOICE_FALLBACK` | (CSV) | Dipakai bila katalog remote tak terjangkau — lihat DECISION #6. **Field CSV** | — | terisi |
| `FSA_TTS_VOICE_ALIASES` | (CSV `alias:voice`) | Lihat DECISION #1. **Field CSV** | — | tergantung keputusan |

### C.4 Jebakan yang wajib diingat

**4 field CSV baru** (`STT_ALLOWED_LANGUAGES`, `STT_ALLOWED_AUDIO_TYPES`, `TTS_VOICE_FALLBACK`,
`TTS_VOICE_ALIASES`) → daftar di `_split_csv` tumbuh 5 → 9, dan masing-masing **wajib**
dianotasi `NoDecode`. Kalau lupa: `SettingsError` saat **import**, service tidak bisa boot sama
sekali, traceback menyebut pydantic bukan field barunya. `tests/test_settings.py` menangkapnya
hanya kalau field baru ditambahkan ke `CSV_FIELDS`.

---

## D. Revised dependency plan — **hasil verifikasi, bukan asumsi**

### D.1 Metode

Resolver lintas-versi pip terhadap **platform produksi** (`manylinux_2_28_x86_64` +
fallback `2_17` / `manylinux2014`), bukan macOS, untuk Python 3.10 / 3.11 / 3.12. Set yang
diresolusi = seluruh `requirements.txt` existing + `faster-whisper` + `edge-tts`.

### D.2 Hasil matrix (terverifikasi)

| Paket | py3.10 | py3.11 | py3.12 |
|---|---|---|---|
| resolver | **OK** (54 paket) | **OK** (50) | **OK** (50) |
| `faster-whisper` | 1.2.1 | 1.2.1 | 1.2.1 |
| `ctranslate2` | 4.8.2 | 4.8.2 | 4.8.2 |
| `edge-tts` | 7.2.8 | 7.2.8 | 7.2.8 |
| `av` (tanpa pin) | **17.1.0** | **18.1.0** | **19.0.0** |
| `numpy` | 2.2.6 | 2.2.6 | 2.2.6 |
| `onnxruntime` | 1.23.2 | 1.30.0 | 1.30.0 |
| `opencv-python-headless` | 4.14.0.94 | 4.14.0.94 | 4.14.0.94 |

`av` tanpa pin **menyimpang per interpreter** — dugaan Revisi 1 terbukti secara empiris, dan
batas atas 3.10 adalah av 17.

### D.3 Pin yang dipilih — terverifikasi seragam

```
av>=17.1,<18
```

| | py3.10 | py3.11 | py3.12 |
|---|---|---|---|
| resolve dengan pin | OK | OK | OK |
| `av` | **17.1.0** | **17.1.0** | **17.1.0** |
| `onnxruntime` | 1.23.2 | 1.30.0 | 1.30.0 |
| `numpy` | 2.2.6 | 2.2.6 | 2.2.6 |
| `ctranslate2` | 4.8.2 | 4.8.2 | 4.8.2 |
| `huggingface-hub` | 1.33.0 | 1.33.0 | 1.33.0 |

### D.4 Instalasi nyata + `pip check` + import (Python 3.11.16)

```
pip check                → No broken requirements found.   (exit 0)
import av                → 17.1.0
import faster_whisper    → 1.2.1
import edge_tts          → 7.2.8
import numpy             → 2.2.6
import onnxruntime       → 1.30.0
import cv2               → 4.14.0
```

### D.5 Analisis konflik

| Constraint | Sumber | Status |
|---|---|---|
| `onnxruntime<2,>=1.14` | faster-whisper | **Beririsan** dengan pin repo `>=1.19,<2.0` ✓ |
| `numpy` (tanpa batas) | ctranslate2 | Pin `numpy<2.3` milik insightface tetap berlaku ✓ |
| `av>=11` (tanpa batas atas) | faster-whisper | **Masalah** — dipin di sisi kita jadi `>=17.1,<18` ✓ |
| `aiohttp<4,>=3.8` | edge-tts | Paket baru, tidak bertabrakan ✓ |
| `tokenizers<1,>=0.13` | faster-whisper | Paket baru ✓ |

**Tidak satu pun paket Face stack bergerak** karena penambahan Speech. numpy, onnxruntime,
opencv-python-headless, fastapi, pydantic identik sebelum dan sesudah.

### D.6 Batas kejujuran verifikasi ini

Harus dinyatakan, bukan disembunyikan:

1. **`insightface==0.7.3` adalah sdist-only** — tidak punya wheel sama sekali (versi ber-wheel
   di PyPI hanya 0.2.1, 1.0, 1.0.1, 2.0). Ia dibangun dari source, itulah sebabnya Makefile
   memasang `cython<3.1` dan `numpy<2.3` lebih dulu. Resolver lintas-versi **tidak bisa**
   membangun sdist, jadi insightface dikeluarkan dari matrix D.2/D.3. Yang diverifikasi adalah
   seluruh dependency binernya (numpy, onnx runtime-nya, opencv) tidak bergerak. Metadata
   insightface menuntut `numpy, onnx, tqdm, requests, matplotlib, Pillow, scipy,
   scikit-learn, scikit-image, easydict, cython, albumentations, prettytable` — **tidak satu pun
   bersinggungan dengan paket yang ditambahkan Speech.**
2. **Instalasi nyata hanya diverifikasi pada 3.11, macOS arm64.** 3.10 tidak tersedia di mesin
   ini; 3.13 juga tidak.
3. **Belum diverifikasi pada Linux nyata.** Resolver sudah menargetkan manylinux, tapi
   resolve ≠ import.

### D.7 Yang wajib dilakukan sebelum merge

| # | Verifikasi | Di mana |
|---|---|---|
| 1 | `pip install -r requirements.txt` penuh (termasuk build insightface) + `pip check` + import di **3.10** | Ubuntu, container/VM |
| 2 | Idem di **3.11** | Ubuntu |
| 3 | Idem di **3.12** | Ubuntu |
| 4 | `make test` hijau di ketiganya | Ubuntu |
| 5 | Face verification nyata (5 foto + klip) memberi skor identik sebelum/sesudah | Ubuntu |

Kalau salah satu memaksa perubahan versi dependency Face stack → **STOP dan laporkan** sesuai
instruksi §4.

### D.8 Perubahan requirements

```
# requirements.txt — tambahan
faster-whisper>=1.1,<2.0
edge-tts>=7.0,<8.0
av>=17.1,<18        # dipin: tanpa pin, av resolve 17.1/18.1/19.0 di py3.10/3.11/3.12.
                    # av 19 menuntut py>=3.12. Repo mendukung 3.10-3.12.
```

```
# requirements-test.txt — tambahan
av>=17.1,<18        # decoder-nya sendiri, bukan model. Presedennya opencv-python-headless,
                    # yang ada di sini untuk menguji frame sampler kita sendiri.
```

`faster-whisper` dan `edge-tts` **tidak** masuk requirements-test.txt (rule 15) — dan §19
menuntut test yang membuktikan aplikasi bisa di-import tanpa keduanya, yang hanya bisa
dibuktikan kalau memang tidak terpasang di test env.

### D.9 Biaya memori import saja (terukur, tanpa model)

py3.11, RSS puncak, macOS:

| | RSS | Delta |
|---|---|---|
| baseline interpreter | 14,6 MB | — |
| `+cv2` | 51,8 MB | +37,2 |
| `+av` | 38,6 MB | +23,9 |
| `+faster_whisper` | 56,8 MB | +42,1 |
| `+edge_tts` | 39,2 MB | +24,5 |
| keempatnya | 92,5 MB | +77,8 |

Face runtime dengan Speech mati tidak membayar ~41 MB itu per worker, karena tidak satu pun
di-import. **Model Whisper belum termasuk** — itu biaya besar yang harus diukur (§14).

---

## E. Exact STT validation pipeline

Lapisan-lapisannya, dan **siapa otoritasnya di tiap lapis**. Semua sudah diuji empiris dengan
PyAV 17.1.0 — hasilnya di §E.2.

```
POST /api/v1/speech/transcribe   (multipart/form-data)
 │
 ├─ 1. GuardDep ......................... API key → rate limit (urutan existing, tidak diubah)
 │       gagal → 401 / 429 / 503 api_keys_not_configured
 │
 ├─ 2. FSA_STT_ENABLED .................. mati → 503 stt_disabled
 │
 ├─ 3. BYTE SIZE LIMIT .................. temp_file(max_bytes=FSA_STT_MAX_AUDIO_BYTES)
 │       Dibatalkan SAAT streaming, bukan sesudah. File kosong → 422 invalid_upload
 │       lewat → 413 payload_too_large
 │       ► OTORITAS: mutlak. Tidak bergantung isi file.
 │
 ├─ 4. DECLARED CONTENT-TYPE ............ ensure_content_type(), HANYA bila header ada
 │       tidak cocok → 415 unsupported_media_type
 │       ► OTORITAS: TIDAK ADA. Header absen = lapis ini dilewati (temuan audit).
 │         Tidak pernah menjadi satu-satunya penjaga.
 │
 ├─ 5. MAGIC BYTE / CONTAINER SIGNATURE . looks_like_audio(head_bytes)
 │       tidak dikenali → 422 invalid_upload
 │       ► OTORITAS: TIDAK. Ini EARLY REJECTION saja — menolak yang jelas bukan audio
 │         sebelum membayar decoder. Bukti: payload `ID3` + teks LOLOS lapis ini.
 │       ► FILENAME EXTENSION TIDAK PERNAH DIPERIKSA. Bukan security boundary.
 │
 ├─ 6. DECODER / CONTAINER INSPECTION ... av.open(path)   [lazy import]
 │       InvalidDataError → 422 audio_decode_failed
 │       ► OTORITAS: ya, untuk "apakah ini benar-benar container yang bisa dibaca".
 │
 ├─ 7. AUDIO STREAM HARUS ADA ........... len(container.streams.audio) > 0
 │       kosong → 422 no_audio_stream
 │       ► OTORITAS: ya, dan INI yang menangkap kasus yang lolos lapis 5-6.
 │         Bukti: PNG di-rename .wav DIBUKA oleh av (FFmpeg punya demuxer png)
 │         dan hanya tertangkap di sini.
 │
 ├─ 8. DURATION — DUA TAHAP
 │      8a. fast path: container.duration / stream.duration bila tersedia
 │            sudah melebihi cap → 422 audio_too_long  (menolak tanpa decode sama sekali)
 │      8b. authoritative: decode inkremental, akumulasi frame.samples / frame.sample_rate
 │            ABORT SEGERA saat total > FSA_STT_MAX_AUDIO_SECONDS → 422 audio_too_long
 │       ► OTORITAS: 8b. Metadata hanya jalan cepat, pernah tidak ada, pernah bohong.
 │       ► Overshoot terukur: satu frame (~13 ms). Tidak pernah men-decode klip panjang
 │         sampai habis lalu menolak.
 │
 ├─ 9. LANGUAGE ......................... bila FSA_STT_ALLOWED_LANGUAGES diisi
 │       di luar daftar → 422 language_not_supported
 │
 ├─ 10. SEMAPHORE ....................... async with get_stt_limiter()
 │
 ├─ 11. INFERENCE ....................... await to_thread.run_sync(engine.transcribe, path, opts)
 │        Model dimuat DI SINI bila belum (double-checked lock). Tidak pernah lebih awal.
 │        engine gagal dimuat → 503 speech_engine_unavailable
 │
 └─ 12. FINALLY ......................... temp_file menghapus file, sukses maupun gagal
```

Lapisan 3, 5, 6, 7, 8b adalah gerbang yang **tidak bisa dilewati dengan memanipulasi header
atau nama file**. Lapisan 4 bisa dilewati dan karena itu tidak pernah sendirian.

### E.2 Bukti empiris (PyAV 17.1.0, py3.11)

| Kasus uji | Hasil terukur | Lapis yang menangkap |
|---|---|---|
| webm/opus 8 s (bentuk MediaRecorder) | container=8,008 s, **stream duration = `None`** | 8b memberi 8,0 s |
| webm/opus, cap 2 s | **abort di 2,0135 s setelah 101 dari 401 frame** | 8b, early abort terbukti |
| wav 3 s | container=3,0 stream=3,0 decoded=3,0 | konsisten |
| mp4 video-only (container sah, tanpa audio) | `no_audio_stream` | **7** |
| WebM malformed (magic byte benar, isi acak) | `av.error.InvalidDataError` | **6** |
| `ID3` + teks (menyamar MP3) | lolos lapis 5, `InvalidDataError` di lapis 6 | **6** |
| PNG di-rename `.wav` | **`av.open` BERHASIL**, `no_audio_stream` | **7** |

Baris terakhir adalah pembenaran paling kuat untuk arsitektur berlapis yang Anda minta: "bisa
dibuka oleh decoder" **bukan** bukti bahwa sebuah file adalah audio.

### E.3 Keputusan: satu decode atau dua

Validasi (lapis 6–8b) men-decode untuk memvalidasi, lalu `engine.transcribe(path)` men-decode
lagi di dalam faster-whisper. Dua decode. Alternatifnya menyimpan ndarray hasil decode dan
menyerahkannya ke engine — satu decode, tapi Protocol jadi membawa bentuk data yang hanya
cocok untuk provider lokal. **Lihat DECISION REQUIRED #3.**

---

## F. Exact TTS pipeline

```
POST /api/v1/speech/synthesize   (application/json)
 │
 ├─ 1. GuardDep ......................... 401 / 429 / 503
 ├─ 2. FSA_TTS_ENABLED .................. mati → 503 tts_disabled
 │
 ├─ 3. VALIDASI REQUEST (pydantic + eksplisit)
 │      text       wajib, strip, kosong → 422 invalid_request
 │                 len > FSA_TTS_MAX_TEXT_CHARS → 413 payload_too_large
 │      voice      opsional; kosong → FSA_TTS_DEFAULT_VOICE
 │      rate       opsional; regex ^[+-]\d{1,3}%$   → 422 invalid_request
 │      volume     opsional; regex ^[+-]\d{1,3}%$   → 422 invalid_request
 │      ► rate/volume DIVALIDASI KETAT, bukan diteruskan mentah: provider Edge
 │        menyisipkannya ke atribut SSML, jadi nilai bebas adalah jalur SSML injection.
 │      ► TIDAK ADA field `format` (§6). Output selalu MP3.
 │
 ├─ 4. RESOLUSI VOICE ................... terhadap katalog (cache TTL) / alias / fallback
 │       tidak ada → 422 voice_not_available
 │       ► Route tidak pernah tahu bentuk id voice. Resolusi ada di service layer.
 │
 ├─ 5. SEMAPHORE ........................ async with get_tts_limiter()   [Semaphore(4)]
 │
 ├─ 6. SINTESIS .......................... await engine.synthesize(req)   [TANPA to_thread]
 │       with anyio.fail_after(FSA_TTS_TIMEOUT_SECONDS)
 │         timeout                → 504 speech_provider_timeout
 │         error provider         → 502 speech_provider_failed
 │         chunk terakumulasi > FSA_TTS_MAX_OUTPUT_BYTES → ABORT → 502 tts_output_too_large
 │       ► Buffer penuh di memori, BERBATAS. Streaming tidak dipakai di v1.
 │       ► Byte 0 audio → 502 speech_provider_failed (jangan kirim MP3 kosong sebagai 200)
 │
 └─ 7. RESPONSE ......................... Response(content=bytes, media_type="audio/mpeg")
         Content-Length: <n>
         Content-Disposition: attachment; filename="speech.mp3"
         X-Speech-Voice: <voice yang benar-benar dipakai>
         ► Status 200 dikirim SETELAH seluruh audio ada di tangan. Kegagalan upstream
           selalu jadi AppError JSON, tidak pernah MP3 terpotong.
         ► Tidak menyentuh disk. Tidak ada tempfile. Tidak ada URL.
```

Bila Laravel butuh persistence: Laravel yang menyimpan/upload. Service ini tetap stateless.

---

## G. Concurrency model

```
┌─ FACE RUNTIME :8001 ──────────────────────────────────────────────────┐
│  get_inference_limiter()      anyio.Semaphore(FSA_MAX_CONCURRENT_INFERENCES = 2)
│      └─ VerificationService.verify()   → to_thread.run_sync(_run)      │
│      └─ LivenessService.verify()       → to_thread.run_sync(_run)      │
│  SEMANTIK TIDAK DIUBAH. Tidak ada beban Speech yang pernah menyentuhnya.│
└───────────────────────────────────────────────────────────────────────┘

┌─ SPEECH RUNTIME :8002 ────────────────────────────────────────────────┐
│  get_stt_limiter()            anyio.Semaphore(FSA_STT_MAX_CONCURRENT = 1)
│      └─ SpeechToTextService.transcribe()                               │
│           async with limiter → await to_thread.run_sync(engine.transcribe)
│           CPU-bound lokal, pola identik Face                           │
│                                                                        │
│  get_tts_limiter()            anyio.Semaphore(FSA_TTS_MAX_CONCURRENT = 4)
│      └─ TextToSpeechService.synthesize()                               │
│           async with limiter → with fail_after(timeout) → await provider
│           TANPA to_thread: network-bound, bukan CPU                     │
└───────────────────────────────────────────────────────────────────────┘
```

**Tiga objek Semaphore berbeda, tidak pernah ada yang dipakai bersama.** Di produksi bahkan
berada di proses berbeda, jadi berbagi secara fisik tidak mungkin.

Detail yang wajib benar:

- **Model load race.** `FasterWhisperEngine.load()` double-checked di bawah `threading.Lock`,
  persis `InsightFaceEngine.load()`: beberapa thread bisa lolos semaphore(1) berurutan dan
  request pertama bisa berlomba dengan apa pun yang memanggil `load()`.
- **`FSA_STT_CPU_THREADS` berkali dengan semaphore.** Anggaran:
  `stt_max_concurrent × max(1, stt_cpu_threads) ≤ core Speech runtime`.
- **TTS tidak boleh masuk `to_thread`.** Mendorong coroutine aiohttp ke thread akan membuat
  event loop kedua atau blocking — keduanya salah.
- **Cancellation.** `to_thread.run_sync` tidak bisa diinterupsi; klien yang memutus tetap
  membayar. Dibatasi `FSA_STT_MAX_AUDIO_SECONDS`. Sudah berlaku untuk Face hari ini.
- **Thread limiter default anyio = 40**, dipakai bersama dalam satu proses. Dengan semaphore
  1 dan 2 tidak pernah jadi pengikat; dicatat agar tidak mengejutkan nanti.

---

## H. Production process isolation — satu codebase, dua runtime

Yang membuat ini bekerja tanpa duplikasi kode: **`create_app()` sudah membaca `Settings` untuk
memutuskan apa yang dipasang.** Router speech dipasang bila `stt_enabled or tts_enabled`. Itu
satu-satunya mekanisme baru yang dibutuhkan — tidak ada entrypoint kedua, tidak ada `app_speech.py`,
tidak ada package kedua.

```python
# app/main.py — inti perubahannya, ~4 baris
if settings.stt_enabled or settings.tts_enabled:
    app.include_router(speech.router, prefix=settings.api_prefix)
```

### H.1 systemd — dua unit, satu direktori

```ini
# /etc/systemd/system/fsa-face.service
[Service]
WorkingDirectory=/opt/fsa
EnvironmentFile=/opt/fsa/.env.face
ExecStart=/opt/fsa/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8001 --workers 2
```

```ini
# /etc/systemd/system/fsa-speech.service
[Service]
WorkingDirectory=/opt/fsa
EnvironmentFile=/opt/fsa/.env.speech
ExecStart=/opt/fsa/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8002 --workers 1
MemoryMax=3G        # batas keras: Speech tidak boleh memakan RAM server
```

`.env.face`: `FSA_STT_ENABLED=false`, `FSA_TTS_ENABLED=false`
`.env.speech`: `FSA_STT_ENABLED=true`, `FSA_TTS_ENABLED=true`

`MemoryMax` pada unit Speech adalah jaring terakhir prinsip §1: kalau Speech membengkak, cgroup
yang membunuhnya, bukan OOM killer kernel yang memilih korban sendiri — dan korban itu bisa
saja Face.

### H.2 nginx — batas sebenarnya

```nginx
limit_req_zone $http_x_api_key zone=face_zone:10m   rate=120r/m;
limit_req_zone $http_x_api_key zone=speech_zone:10m rate=60r/m;

location /api/v1/face/     { limit_req zone=face_zone   burst=20; proxy_pass http://127.0.0.1:8001; }
location /api/v1/liveness/ { limit_req zone=face_zone   burst=20; proxy_pass http://127.0.0.1:8001; }
location /api/v1/health    {                                      proxy_pass http://127.0.0.1:8001; }
location /api/v1/speech/   { limit_req zone=speech_zone burst=5;
                             proxy_read_timeout 120s;             proxy_pass http://127.0.0.1:8002; }
location /demo             { return 404; }
location /                 { return 404; }
```

**Dua zona `limit_req` terpisah adalah bagian dari keputusan, bukan detail.** Satu zona yang
di-key pada API key untuk kedua location akan mengembalikan persis masalah §11 yang ingin
dihindari: klien TTS menghabiskan jatah absensi, kali ini di nginx bukan di Python.

### H.3 Bagaimana §11 Option A terpenuhi

| Requirement | Cara terpenuhi | Bisa diverifikasi dengan |
|---|---|---|
| Speech tidak menghabiskan rate budget Face | Bucket `FixedWindowLimiter` in-memory per proses; dua proses = dua hitungan | `ps` menunjukkan dua PID; zona nginx terpisah |
| `throttle.py` tidak di-refactor | Nol perubahan pada file itu | `git diff` kosong untuk file itu |
| Failure Speech tidak menjatuhkan Face | Batas proses + `MemoryMax` | Bunuh `fsa-speech`, Face tetap menjawab |
| Resource Speech tidak mengambil quota Face | Semaphore terpisah, proses terpisah | Test `test_speech_isolation.py` |

**Acceptance criteria go-live (wajib, bukan catatan):** Speech tidak boleh masuk produksi
sebelum `/api/v1/speech/*` dirutekan ke runtime terpisah dengan zona `limit_req` sendiri.
Menjalankan Speech dalam proses Face di produksi adalah pelanggaran §11, bukan trade-off.

---

## I. Error catalogue

Semua subclass `AppError`, semua lewat handler existing → `{error:{code,message,details}}`.

### I.1 Baru

| Class | Status | `code` | Kapan |
|---|---|---|---|
| `SttDisabledError` | 503 | `stt_disabled` | `FSA_STT_ENABLED=false` |
| `TtsDisabledError` | 503 | `tts_disabled` | `FSA_TTS_ENABLED=false` |
| `AudioDecodeError` | 422 | `audio_decode_failed` | Lapis 6: container tidak bisa dibaca |
| `NoAudioStreamError` | 422 | `no_audio_stream` | Lapis 7: container sah, tanpa stream audio |
| `AudioTooLongError` | 422 | `audio_too_long` | Lapis 8a/8b |
| `LanguageNotSupportedError` | 422 | `language_not_supported` | Di luar `FSA_STT_ALLOWED_LANGUAGES` |
| `SpeechEngineUnavailableError` | 503 | `speech_engine_unavailable` | Model/library gagal dimuat |
| `VoiceNotAvailableError` | 422 | `voice_not_available` | Voice tidak ada di katalog |
| `SpeechProviderError` | 502 | `speech_provider_failed` | Provider menolak / 0 byte / output kebesaran |
| `SpeechProviderTimeoutError` | 504 | `speech_provider_timeout` | `FSA_TTS_TIMEOUT_SECONDS` terlampaui |
| `InvalidSpeechRequestError` | 422 | `invalid_request` | Teks kosong, `rate`/`volume` tidak sesuai pola |

`stt_disabled` dan `tts_disabled` dipisah karena Speech runtime bisa menyalakan salah satu saja,
dan pemanggil harus tahu fitur mana yang mati.

`speech_engine_unavailable` dipisah dari `engine_unavailable` mengikuti preseden
`ReferenceFacesUnusableError`: dua 422 dipecah justru karena pemanggil harus bisa membedakannya.
Pemanggil yang mem-fallback antar fitur perlu tahu mesin **mana** yang mati.

502/504 adalah status baru di repo ini, dibenarkan karena kegagalannya milik pihak ketiga dan
pemanggil harus me-retry-nya berbeda dari 503. `anyio.fail_after` melempar `TimeoutError`;
wrapper provider yang mengonversi — route tidak pernah melihat exception provider.

### I.2 Dipakai ulang tanpa perubahan

`PayloadTooLargeError` (413) audio & teks · `UnsupportedMediaError` (415) content-type ·
`InvalidUploadError` (422) kosong/magic byte tak dikenali · `UnauthorizedError` (401) ·
`NotConfiguredError` (503) · `RateLimitedError` (429).

### I.3 Sengaja tidak dibuat

Handler `RequestValidationError`. 422 milik FastAPI memang tidak memakai envelope, dan itu sudah
berlaku untuk `/liveness/challenge` dan batas `ge/le` di `/face/verify`. Menambahkannya
**mengubah respons existing** → pelanggaran prinsip §1.

---

## J. Test matrix

`FakeSttEngine` / `FakeTtsEngine` disuntik lewat `dependency_overrides`. Provider asli tidak
pernah di-import oleh suite.

### J.1 Audio validation (§19)

| # | Kasus | Harapan | Lapis |
|---|---|---|---|
| 1 | Content-Type absen, audio **valid** | 200 | lolos 4, ditegakkan 5–8 |
| 2 | Content-Type absen, binary **invalid** | 422 `invalid_upload` / `audio_decode_failed` | 5 / 6 |
| 3 | Content-Type salah (`image/png`) | 415 | 4 |
| 4 | Extension `.wav`, isi PNG | 422 `no_audio_stream` | **7** |
| 5 | Extension `.mp3`, isi teks ber-`ID3` | 422 `audio_decode_failed` | 6 |
| 6 | Container sah tanpa audio stream (mp4 video-only) | 422 `no_audio_stream` | 7 |
| 7 | WebM malformed | 422 `audio_decode_failed` | 6 |
| 8 | WebM valid **tanpa** duration metadata | 200, durasi dari decode | 8b |
| 9 | Durasi metadata > cap | 422 `audio_too_long` **tanpa decode** | 8a |
| 10 | Metadata bohong/absen, decode > cap | 422 `audio_too_long` | 8b |
| 11 | Decode berhenti saat melewati cap | frame yang di-decode **jauh lebih sedikit** dari total | 8b |
| 12 | File kosong | 422 `invalid_upload` | 3 |
| 13 | Melewati byte cap | 413 | 3 |
| 14 | Filename tidak pernah jadi penentu | `.txt` berisi WAV sah → 200 | — |

Kasus 11 diassert pada **jumlah frame/durasi ter-decode**, bukan pada wall clock — sebuah
assertion waktu akan flaky di CI.

### J.2 Isolation (§19)

| # | Kasus | Harapan |
|---|---|---|
| 15 | `get_stt_limiter() is not get_inference_limiter()` | True |
| 16 | `get_tts_limiter() is not get_inference_limiter()` | True |
| 17 | `get_stt_limiter() is not get_tts_limiter()` | True |
| 18 | Ketiganya honor angka settings masing-masing | ukuran berbeda |
| 19 | Request speech tidak mengubah state limiter Face | counter Face tak tersentuh |
| 20 | **cv2 dan av hidup bersama** (§L-1) | tulis+decode mp4 lewat cv2 **setelah** `import av` berhasil, kedua urutan import |

### J.3 Lazy loading (§19)

| # | Kasus | Harapan |
|---|---|---|
| 21 | `import app.main` tanpa `faster_whisper` terpasang | sukses |
| 22 | `import app.main` tanpa `edge_tts` terpasang | sukses |
| 23 | `app.openapi()` dihasilkan tanpa memuat model | `engine.load` tidak pernah dipanggil |
| 24 | `GET /health` tidak memuat model STT | idem |
| 25 | `GET /speech/capabilities` tidak memuat model **dan tidak ada network** | idem |
| 26 | `FSA_STT_ENABLED=false` → provider tidak pernah dikonstruksi | idem |
| 27 | Model dimuat **hanya** pada transkripsi pertama | `load` dipanggil tepat sekali |
| 28 | Kedua flag false → prefix `/speech/` tidak ada | 404 |

Kasus 21–22 diimplementasikan dengan menyisipkan modul sentinel yang melempar `ImportError`
ke `sys.modules`, bukan dengan meng-uninstall paket.

### J.4 TTS bounded output (§19)

| # | Kasus | Harapan |
|---|---|---|
| 29 | Output > `FSA_TTS_MAX_OUTPUT_BYTES` | 502 `tts_output_too_large`, **bukan** body separuh |
| 30 | Provider error sebelum byte pertama | 502 `speech_provider_failed` envelope JSON |
| 31 | Provider error di tengah aliran | 502, bukan 200 terpotong |
| 32 | Timeout | 504 `speech_provider_timeout` |
| 33 | Provider mengembalikan 0 byte | 502, bukan 200 kosong |
| 34 | Sukses | `audio/mpeg`, `Content-Length` benar, `Content-Disposition` ada |
| 35 | Teks > `FSA_TTS_MAX_TEXT_CHARS` | 413 |
| 36 | Teks kosong / hanya spasi | 422 `invalid_request` |
| 37 | `rate` = `"drop table"` / SSML | 422 `invalid_request` (anti SSML injection) |
| 38 | Tidak ada field `format` di schema | `format` dikirim → diabaikan/422, output tetap MP3 |

### J.5 Access control

| # | Kasus | Harapan |
|---|---|---|
| 39–42 | Keempat route tanpa key | 401 |
| 43 | Key salah | 401 |
| 44 | Key benar | 200 |
| 45 | Tanpa key terkonfigurasi | 503 `api_keys_not_configured` |
| 46 | Rate limit per key | 429 |
| 47 | Rate limit per `X-Tenant` | bucket terpisah |
| 48 | `/health` tetap terbuka | 200 |
| 49 | `voices` dan `capabilities` **terjaga** (GET mudah terlupa) | 401 tanpa key |

### J.6 Kontrak & tempfile

| # | Kasus | Harapan |
|---|---|---|
| 50 | Bentuk `TranscriptionResponse` | sesuai §G-17 |
| 51 | `include_segments=false` | `segments` kosong/absen |
| 52 | Bentuk `capabilities` | mencerminkan `Settings`, bukan konstanta |
| 53 | **Tempfile terhapus setelah sukses** | temp dir kosong |
| 54 | **Tempfile terhapus setelah gagal** (tiap lapis 5–11) | temp dir kosong |
| 55 | TTS tidak pernah membuat tempfile | temp dir kosong |
| 56 | `.env.example` tetap bisa dimuat | `test_settings.py` |
| 57 | 4 field CSV baru terbaca comma-separated | `CSV_FIELDS` diperluas |

### J.7 Regresi Face — wajib tetap hijau

116 test existing tanpa modifikasi, plus: bentuk respons `/face/verify` dan `/face/verify-image`
byte-identik sebelum/sesudah untuk input yang sama, dan `/health` tidak berubah.

---

## K. Remaining decisions

```
DECISION REQUIRED #1  —  Identifier voice: kontradiksi §6 vs §7
Context:
  §6 menetapkan request body dengan "voice": "id-ID-ArdiNeural". Itu identifier milik
  Edge TTS. §7 menetapkan provider tidak boleh bocor ke API contract supaya bisa diganti
  tanpa mengubah kontrak Laravel/mobile. Kedua instruksi tidak bisa dipenuhi sekaligus
  apa adanya: kalau Laravel mengirim "id-ID-ArdiNeural", pindah ke ElevenLabs
  membatalkan setiap caller yang menyimpan nilai itu.
Option A:  voice tetap id provider mentah, didokumentasikan "hanya sah untuk provider
           yang sedang dikonfigurasi". Paling sederhana, literal sesuai §6.
           Pindah provider = perubahan berkoordinasi di Laravel.
Option B:  tambah FSA_TTS_VOICE_ALIASES (CSV "alias:voice_provider"). Laravel mengirim
           "default" / "female" / "male"; server memetakan. Id mentah tetap diterima
           sebagai escape hatch, jadi bentuk request §6 tidak berubah sama sekali.
           Pindah provider = ubah satu env var, nol perubahan Laravel.
Recommendation: Option B.
  Satu setting config, nol field request baru, §6 tetap valid apa adanya, dan §7 benar-
  benar terpenuhi bukan hanya dinyatakan. Biayanya satu lapis lookup.
Impact:
  A → risiko vendor lock di sisi caller; migrasi provider jadi proyek lintas tim.
  B → +1 field CSV di Settings (total field CSV jadi 4), +1 lapis resolusi di service,
      +2 test. Tidak ada perubahan kontrak.
```

```
DECISION REQUIRED #2  —  Sumber katalog /speech/voices
Context:
  Edge TTS mengambil daftar suara lewat network call ke Microsoft. §18 melarang network
  call pada /speech/capabilities, tapi tidak menyebut /speech/voices. Kalau voices
  memanggil network, availability endpoint itu bergantung pihak ketiga.
Option A:  Network call + cache TTL (FSA_TTS_VOICE_CACHE_TTL_SECONDS) + timeout, dan bila
           gagal jatuh ke FSA_TTS_VOICE_FALLBACK sehingga endpoint tidak pernah hard-fail.
Option B:  Sepenuhnya statis dari FSA_TTS_VOICE_FALLBACK. Nol network. Daftar bisa basi.
Recommendation: Option A dengan fallback wajib terisi.
  Katalog akurat berguna bagi klien yang membangun dropdown, dan fallback membuat
  kegagalan provider tidak pernah menjatuhkan endpoint. Prinsip §1 tetap aman karena
  ini ada di Speech runtime.
Impact:
  A → +1 field CSV, +cache berkunci, +2 test (cache hit, fallback saat provider mati).
  B → paling sederhana, tapi daftar suara harus dipelihara manual di env.
```

```
DECISION REQUIRED #3  —  Satu decode atau dua pada jalur STT
Context:
  Validasi lapis 6-8b harus men-decode audio untuk menegakkan durasi (§3). faster-whisper
  kemudian men-decode lagi secara internal. Dua decode untuk satu request.
Option A:  Dua decode. Validator membuang sampel, engine menerima Path.
           Protocol tetap bersih: transcribe(path, options). Provider cloud (OpenAI/Azure)
           cukup mengunggah file itu — cocok untuk semuanya.
Option B:  Satu decode. Validator menyimpan ndarray 16 kHz mono, Protocol membawa
           DecodedAudio. Menghemat satu decode, tapi bentuk datanya hanya berguna bagi
           provider lokal; provider cloud harus merekonstruksi file.
Recommendation: Option A.
  Decode audio ±0,1-0,3 s untuk klip 5 menit, sementara inference puluhan detik — dua
  decode adalah overhead ~1%. Membayar 1% untuk Protocol yang tidak perlu dirombak saat
  provider kedua masuk adalah pertukaran yang benar, dan §7 menjadikan pergantian
  provider tujuan eksplisit.
Impact:
  A → ~1% CPU tambahan per request STT. Protocol stabil.
  B → hemat ~1%. Protocol harus berubah saat provider cloud ditambahkan, dan perubahan
      Protocol menyentuh setiap provider sekaligus.
```

```
DECISION REQUIRED #4  —  Apakah Face runtime perlu flag untuk menolak route speech
Context:
  Di topologi §12, nginx yang memisahkan. Tapi kalau seseorang salah konfigurasi dan
  request speech mendarat di Face runtime, router speech tidak ter-mount (kedua flag
  false) sehingga hasilnya 404. Sebaliknya request face yang mendarat di Speech runtime
  AKAN dilayani, dan di sana cv2 + av termuat bersama (§L-1).
Option A:  Tidak menambah apa pun. nginx adalah batasnya. Face runtime sudah aman karena
           flag-nya false; Speech runtime melayani face hanya kalau nginx salah.
Option B:  Tambah FSA_FACE_ENABLED (default true) supaya Speech runtime bisa mematikan
           route face secara eksplisit.
Recommendation: Option A.
  Option B menambah satu flag yang, kalau salah diset di Face runtime, mematikan absensi
  — risiko yang ditambahkannya lebih besar daripada risiko yang dihilangkannya. Salah
  konfigurasi nginx terdeteksi oleh smoke check go-live.
Impact:
  A → nol perubahan. Bergantung pada kebenaran konfigurasi nginx.
  B → satu jalur konfigurasi baru yang bisa mematikan production-critical workload.
```

```
DECISION REQUIRED #5  —  Nilai FSA_STT_MODEL untuk produksi
Context:
  small/base/medium berbeda akurasi dan memori. Angka memori nyata belum diukur (§14
  menuntut pengukuran, bukan estimasi), dan akurasi bahasa Indonesia berbeda nyata
  antar ukuran.
Option A:  small int8. Default rencana ini. Kompromi.
Option B:  base int8. Paling ringan, akurasi Indonesia turun.
Option C:  medium int8. Akurasi terbaik, risiko memori tertinggi.
Recommendation: mulai small, PUTUSKAN setelah benchmark §14.
  Rencana ini tidak memilih final. Yang saya minta: setujui small sebagai titik awal
  benchmark, bukan sebagai keputusan produksi.
Impact:
  Menentukan sizing server Speech runtime dan MemoryMax pada unit systemd.
```

```
DECISION REQUIRED #6  —  Timeout Laravel untuk /speech/transcribe
Context:
  Transkripsi bisa memakan puluhan detik untuk audio panjang. nginx contoh memakai
  proxy_read_timeout 120s. FaceApiClient existing memakai timeout(120).
Option A:  Samakan 120 s, batasi FSA_STT_MAX_AUDIO_SECONDS ke 300 dan ukur apakah
           audio 300 s benar-benar selesai di bawah 120 s pada hardware target.
           Kalau tidak, TURUNKAN cap durasi, jangan naikkan timeout.
Option B:  Naikkan timeout untuk speech.
Recommendation: Option A.
  Membiarkan request hidup lebih lama berarti thread yang tidak bisa dibatalkan menahan
  slot semaphore lebih lama. Membatasi panjang audio lebih baik daripada memperpanjang
  kesabaran.
Impact:
  Angka final FSA_STT_MAX_AUDIO_SECONDS bergantung benchmark §14.
```

---

## L. Blockers & kontradiksi teknis

### L.1 TEMUAN BARU — PyAV dan OpenCV membundel FFmpeg masing-masing

**Bukan blocker untuk rencana ini, tapi wajib Anda ketahui karena mengubah apa yang harus
dites dan memperkuat keputusan §12.**

Terukur pada py3.11 dengan `av 17.1.0` + `opencv-python-headless 4.14.0`:

```
objc[29403]: Class AVFFrameReceiver is implemented in both
  .../av/.dylibs/libavdevice.62.3.101.dylib and
  .../cv2/.dylibs/libavdevice.61.3.100.dylib
  This may cause spurious casting failures and mysterious crashes.
  One of the duplicates must be removed or renamed.
```

Kedua paket membawa salinan FFmpeg sendiri (av: libavdevice 62, cv2: libavdevice 61). Memuat
keduanya dalam satu proses menghasilkan simbol duplikat.

**Dampak nyata yang saya ukur:** tidak merusak. Tulis + decode mp4 lewat `cv2.VideoWriter` /
`cv2.VideoCapture` (persis yang dilakukan `OpenCVFrameSampler` dan `test_video_sampler.py`)
tetap benar — 60 frame, fps 30, count 60 — pada **kedua** urutan import, dan `av.open` juga
bekerja di proses yang sama. Jadi statusnya: **peringatan runtime, bukan kerusakan yang
teramati.** Tapi teks peringatannya sendiri menyebut "mysterious crashes", jadi ini risiko
laten, bukan surat sehat.

PyAV tidak bisa dihindari: OpenCV tidak punya dukungan audio sama sekali, `soundfile`/libsndfile
tidak bisa membaca webm/opus/mp4, dan biner ffmpeg dikecualikan oleh §20. faster-whisper sendiri
memakai `av` secara internal, jadi `av` termuat begitu STT berjalan — dengan atau tanpa
validator kita.

Konsekuensi:

1. **Memperkuat §12.** Di produksi tabrakan ini **tidak pernah terjadi**: Face runtime tidak
   pernah meng-import `av` (STT mati, import lazy), Speech runtime tidak pernah meng-import
   `cv2` (tidak ada request video). Ini argumen independen untuk dua runtime yang belum ada di
   Revisi 1 — memisahkan proses bukan cuma soal memori, tapi menjaga dua salinan FFmpeg keluar
   dari satu address space.
2. **Dev/test memuat keduanya**, jadi wajib ada test regresi (J.2 #20) yang membuktikan jalur
   video Face masih benar setelah `av` di-import. Kalau suatu saat test itu merah, itu sinyal
   paling awal yang bisa kita punya.
3. **Verifikasi Linux wajib** (D.7). Peringatan di atas milik runtime ObjC macOS; Linux tidak
   akan mencetaknya, jadi tidak adanya peringatan di Ubuntu **bukan** bukti tidak ada masalah.
   Yang membuktikan adalah test J.2 #20 berjalan hijau di Linux.

### L.2 Kontradiksi instruksi yang sudah saya resolusi (konfirmasi saja)

| Kontradiksi | Resolusi |
|---|---|
| §6 `voice` id provider vs §7 no provider leak | DECISION #1 |
| §6 `rate`/`volume` `"+0%"` adalah sintaks prosodi SSML Edge | Diterima sebagai format kontrak, **divalidasi regex ketat** karena nilainya disisipkan ke SSML (jalur injection). Provider lain memetakan. |
| §18 `capabilities` memuat `"provider": "faster-whisper"` vs §7 | Diterima: `capabilities` permukaan operator, didokumentasikan **bukan untuk dibercabangi klien**. Anda yang memilih ini secara eksplisit. |
| §9 default OFF vs §12 Speech runtime ON | Tidak bertabrakan. `.env.example` OFF; dokumentasi deployment menunjukkan `.env.speech` secara eksplisit. |
| §10 model tidak dimuat saat "test startup" vs `av` di requirements-test | Tidak bertabrakan. `av` adalah decoder, bukan model. faster-whisper/edge-tts tetap tidak terpasang di test env, dan itu justru yang memungkinkan test J.3 #21–22. |
| §3 `FSA_STT_MAX_AUDIO_SECONDS` vs konvensi Face tanpa namespace | Nama Anda dipakai. Deviasi disengaja dan didokumentasikan (§C). |

### L.3 Tidak ada blocker yang menghentikan implementasi

Semua requirement §1–§20 feasible pada repo ini. Yang menghalangi hanyalah **6 keputusan di
§K** dan **verifikasi Linux D.7**, keduanya bisa diselesaikan sebelum baris kode pertama.

### L.4 Yang saya minta sebelum Prompt 2

1. Keputusan #1–#6.
2. Konfirmasi bahwa pin `av>=17.1,<18` diterima (berarti av 17.1.0 di ketiga versi Python).
3. Konfirmasi topologi §H.1/§H.2 (dua unit systemd, dua zona nginx) sesuai kondisi server.
4. Izin bahwa verifikasi D.7 pada Ubuntu 3.10/3.11/3.12 dikerjakan **sebelum** merge, bukan
   sebelum scaffolding — supaya implementasi tidak terhenti menunggu VM.

---

## M. Urutan implementasi (revisi)

Tidak ada langkah sebelum #9 yang bisa mempengaruhi deployment Face yang berjalan.

| # | Langkah | Gerbang |
|---|---|---|
| 0 | Keputusan §K + verifikasi Linux D.7 dimulai | **Approval Anda** |
| 1 | `core/errors.py` — subclass baru | test hijau |
| 2 | `core/config.py` — settings + `NoDecode` + `_split_csv`; perluas `test_settings.py` | boot dari `.env.example` |
| 3 | `.env.example` — blok Speech, kedua flag `false` | test #56 |
| 4 | `services/speech/audio.py` + `tests/test_audio_validation.py` | **14 kasus J.1** |
| 5 | `services/speech/stt.py`, `tts.py` — Protocol, service, limiter, timeout | unit test terhadap fake |
| 6 | `schemas/speech.py` | |
| 7 | `tests/conftest.py` — fake engine, fixture, cache clear | |
| 8 | `tests/test_speech_isolation.py` — limiter, lazy, **cv2+av** | J.2 + J.3 |
| 9 | `api/deps.py` + `routes/speech.py` + mount di `main.py` | **pertama yang mengubah app**; J.4–J.6 |
| 10 | `providers/faster_whisper.py` | verifikasi manual, klip nyata |
| 11 | `providers/edge.py` | verifikasi manual |
| 12 | `requirements.txt` / `requirements-test.txt` | D.7 selesai |
| 13 | **Benchmark §14** — tabel RSS di bawah diisi | angka nyata |
| 14 | `README.md` (Indonesia) + `AGENTS.md` | |
| 15 | systemd 2 unit + nginx 2 zona + smoke check | acceptance §H.3 |

### M.1 Benchmark §14 — tabel yang harus diisi dengan angka nyata

Prosedur: `ps -o rss= -p <pid>` pada titik-titik berikut, bukan estimasi.

| Runtime | Titik ukur | RSS (diisi) |
|---|---|---|
| Face | idle, sesudah boot | |
| Face | buffalo_l termuat | |
| Face | saat verifikasi berjalan | |
| Speech | idle, sesudah boot (model belum dimuat) | |
| Speech | model STT termuat | |
| Speech | puncak, satu transkripsi | |
| Speech | sesudah transkripsi selesai | |

Diukur pada **1 worker, `FSA_STT_MAX_CONCURRENT=1`**. Multiple Speech worker tidak diaktifkan
sebelum angka-angka ini ada. Sizing server dan `MemoryMax` diturunkan dari tabel ini, bukan dari
dugaan.
