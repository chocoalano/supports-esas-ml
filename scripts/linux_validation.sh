#!/usr/bin/env bash
# Linux validation for the Speech module: the merge gate (step 13), the STT
# benchmark (step 14), and a staging deployment smoke test (steps 17-18).
#
# Run from a checkout of feature/speech-module on Ubuntu 24.04 x86_64 - the
# production platform. Nothing here is macOS-meaningful: results from any other
# machine are not the merge gate.
#
#   scripts/linux_validation.sh matrix [python3.10 python3.11 python3.12]
#   scripts/linux_validation.sh benchmark [python3.12]
#   sudo CONFIRM_THROWAWAY_HOST=yes scripts/linux_validation.sh staging
#
#   matrix     per interpreter, in fresh venvs under $WORK: full install
#              (insightface must build from its sdist), a package-by-package
#              comparison against the base commit's Face-only install, pip
#              check, imports and versions, ruff, the Speech, Face and full
#              suites, the real Face smoke (base against feature, both import
#              orders) and the offline real Speech smoke. Changes nothing
#              outside $WORK and the download caches.
#   benchmark  base and small, int8, on real Indonesian speech, at every core
#              (and 2 threads when there are more); memory, CPU, latency
#              model, WER/CER. FACE_MEMORY=1 also measures buffalo_l. Run it on
#              a spare VM of the production instance type, not on the live
#              server: it saturates the CPU and loads a model of its own.
#   staging    DEPLOYS ONTO THIS HOST as the README documents - user faceapi,
#              /opt/face-api, both systemd units, nginx - then runs
#              scripts/smoke_deploy.py --systemd. For a throwaway VM only.
#              Validates the templates as shipped: no resource limits applied.
#
# Python 3.10 and 3.11 on Ubuntu 24.04 come from the deadsnakes PPA:
#   sudo add-apt-repository ppa:deadsnakes/ppa
#   sudo apt install python3.10-venv python3.10-dev python3.11-venv python3.11-dev
#
# Results: validation-results/<timestamp>/ - send that directory back, whole.
set -euo pipefail

MODE="${1:-}"
shift || true
REPO="$(cd "$(dirname "$0")/.." && pwd)"
BASE_COMMIT="${BASE_COMMIT:-f1f002f}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS="${RESULTS:-$REPO/validation-results/$STAMP}"
WORK="${WORK:-$HOME/.cache/fsa-validation}"
CORPUS_CACHE="${CORPUS_CACHE:-$WORK/corpus-cache}"
STT_MODEL_ROOT="${STT_MODEL_ROOT:-$WORK/models}"
mkdir -p "$RESULTS" "$WORK" "$CORPUS_CACHE" "$STT_MODEL_ROOT"
cd "$REPO"

log() { printf '\n=== %s\n' "$*"; }

record_host() {
    {
        echo "date: $STAMP"
        echo "commit: $(git rev-parse HEAD)"
        echo "branch: $(git rev-parse --abbrev-ref HEAD)"
        echo "dirty: $(git status --porcelain | wc -l) files"
        uname -a
        grep PRETTY_NAME /etc/os-release || true
        lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core|Socket' || true
        free -h | head -2 || true
    } > "$RESULTS/host.txt"
}

corpus() {
    local python="$1"
    if [ ! -f "$WORK/corpus/manifest.json" ]; then
        "$python" scripts/build_stt_corpus.py "$WORK/corpus" --cache "$CORPUS_CACHE" > "$RESULTS/corpus.txt"
    fi
    cp "$WORK/corpus/manifest.json" "$RESULTS/corpus-manifest.json"
}

install_feature() {  # venv python-binary out-dir
    local venv="$1" python="$2" out="$3"
    rm -rf "$venv"
    "$python" -m venv "$venv"
    # As the Makefile does. No cache: a cached wheel would hide whether
    # insightface still compiles from its sdist on this interpreter.
    "$venv/bin/pip" install -q --no-cache-dir --upgrade pip "cython<3.1" "numpy<2.3"
    "$venv/bin/pip" install --no-cache-dir -r requirements-dev.txt > "$out/install.log" 2>&1
    grep -E "Building wheel for insightface" "$out/install.log" | tee "$out/insightface-build.txt"
    "$venv/bin/pip" freeze > "$out/freeze-feature.txt"
}

install_base() {  # venv python-binary out-dir
    local venv="$1" python="$2" out="$3" requirements="$WORK/base-requirements"
    mkdir -p "$requirements"
    git show "$BASE_COMMIT:requirements.txt" > "$requirements/requirements.txt"
    git show "$BASE_COMMIT:requirements-dev.txt" > "$requirements/requirements-dev.txt"
    rm -rf "$venv"
    "$python" -m venv "$venv"
    "$venv/bin/pip" install -q --upgrade pip "cython<3.1" "numpy<2.3"
    "$venv/bin/pip" install -q -r "$requirements/requirements-dev.txt"
    "$venv/bin/pip" freeze > "$out/freeze-base.txt"
}

no_drift() {  # out-dir
    python3 - "$1" <<'EOF'
import json, sys
out = sys.argv[1]
def read(name):
    pins = {}
    for line in open(f"{out}/{name}"):
        if "==" in line:
            package, version = line.strip().split("==", 1)
            pins[package.lower().replace("_", "-")] = version
    return pins
base, feature = read("freeze-base.txt"), read("freeze-feature.txt")
drift = {p: [v, feature.get(p)] for p, v in base.items() if feature.get(p) != v}
report = {"drift": drift, "added_by_speech": sorted(set(feature) - set(base))}
json.dump(report, open(f"{out}/drift.json", "w"), indent=2)
print(json.dumps(report, indent=2))
sys.exit(1 if drift else 0)
EOF
}

versions() {  # venv
    "$1/bin/python" - <<'EOF'
import json, platform
from importlib.metadata import version
import av, cv2, edge_tts, faster_whisper, insightface, onnxruntime  # noqa: F401
packages = [
    "numpy", "onnxruntime", "opencv-python-headless", "insightface", "av", "faster-whisper",
    "ctranslate2", "edge-tts", "huggingface-hub", "tokenizers", "fastapi", "starlette",
    "pydantic", "pydantic-settings", "uvicorn", "anyio",
]
print(json.dumps({"python": platform.python_version(), "platform": platform.platform(),
                  **{name: version(name) for name in packages}}, indent=2))
EOF
}

face_regression() {  # venv out-dir
    local venv="$1" out="$2" tree="$WORK/base-tree"
    [ -d "$tree" ] || git worktree add -f "$tree" "$BASE_COMMIT" > /dev/null
    PYTHONPATH="$tree" "$venv/bin/python" scripts/smoke_face.py --output "$out/face-base.json" > /dev/null
    PYTHONPATH="$REPO" "$venv/bin/python" scripts/smoke_face.py --output "$out/face-feature.json" > /dev/null
    PYTHONPATH="$REPO" "$venv/bin/python" scripts/smoke_face.py --preimport av \
        --output "$out/face-feature-av-first.json" > /dev/null
    python3 - "$out" <<'EOF'
import json, sys
out = sys.argv[1]
base, feature, av_first = (json.load(open(f"{out}/{n}.json"))["cases"]
                           for n in ("face-base", "face-feature", "face-feature-av-first"))
verdict = {"base_equals_feature": base == feature, "av_first_equals_feature": av_first == feature}
json.dump(verdict, open(f"{out}/face-regression.json", "w"), indent=2)
print(verdict)
sys.exit(0 if all(verdict.values()) else 1)
EOF
}

speech_smoke() {  # venv out-dir
    local venv="$1" out="$2" env_file="$WORK/speech.env"
    cat > "$env_file" <<EOF
FSA_STT_ENABLED=true
FSA_STT_MODEL=small
FSA_STT_COMPUTE_TYPE=int8
FSA_STT_MODEL_ROOT=$STT_MODEL_ROOT
FSA_STT_DEFAULT_LANGUAGE=id
EOF
    "$venv/bin/python" scripts/provision_stt_model.py --env-file "$env_file" > "$out/provision.json"
    HF_HUB_OFFLINE=1 FSA_STT_MODEL=small FSA_STT_MODEL_ROOT="$STT_MODEL_ROOT" \
        "$venv/bin/python" scripts/smoke_speech.py \
        --audio "$WORK/corpus/pendek_pria/02.wav" --reference "$WORK/corpus/pendek_pria/02.txt" \
        --output "$out/speech-smoke.json" > /dev/null
}

run_matrix() {
    local pythons=("$@") failed=()
    [ ${#pythons[@]} -gt 0 ] || pythons=(python3.10 python3.11 python3.12)
    record_host
    for name in "${pythons[@]}"; do
        local python out venv status
        python="$(command -v "$name" || true)"
        if [ -z "$python" ]; then
            echo "$name: not installed - see the header of this script" | tee -a "$RESULTS/summary.txt"
            failed+=("$name:missing")
            continue
        fi
        out="$RESULTS/matrix-$name"
        venv="$WORK/venv-$name"
        mkdir -p "$out"
        log "$name: install"
        # Not `if ( ... )`: bash ignores `set -e` inside anything whose status is
        # being tested, so a failed step would not stop the ones after it.
        set +e
        (
            set -e
            install_feature "$venv" "$python" "$out"
            install_base "$WORK/base-venv-$name" "$python" "$out"
            log "$name: dependency drift"
            no_drift "$out"
            log "$name: pip check, imports"
            "$venv/bin/pip" check | tee "$out/pip-check.txt"
            versions "$venv" | tee "$out/versions.json"
            log "$name: ruff, tests"
            "$venv/bin/ruff" check . | tee "$out/ruff.txt"
            "$venv/bin/python" -m pytest -p no:cacheprovider tests/test_speech_*.py \
                tests/test_audio_validation.py 2>&1 | tail -1 | tee "$out/pytest-speech.txt"
            "$venv/bin/python" -m pytest -p no:cacheprovider tests/test_access.py \
                tests/test_challenge.py tests/test_liveness.py tests/test_scoring.py \
                tests/test_settings.py tests/test_verify.py tests/test_verify_image.py \
                tests/test_video_sampler.py 2>&1 | tail -1 | tee "$out/pytest-face.txt"
            "$venv/bin/python" -m pytest -p no:cacheprovider 2>&1 | tail -1 | tee "$out/pytest-full.txt"
            log "$name: real Face smoke, base against feature, both import orders"
            face_regression "$venv" "$out"
            log "$name: real Speech smoke, offline"
            corpus "$venv/bin/python"
            speech_smoke "$venv" "$out"
        ) 2>&1 | tee "$out/run.log"
        status=${PIPESTATUS[0]}
        set -e
        if [ "$status" -eq 0 ]; then
            echo "$name: passed" | tee -a "$RESULTS/summary.txt"
        else
            echo "$name: FAILED (see $out/run.log)" | tee -a "$RESULTS/summary.txt"
            failed+=("$name")
        fi
    done
    [ ${#failed[@]} -eq 0 ] || { echo "matrix failed: ${failed[*]}"; exit 1; }
}

run_benchmark() {
    local python="${1:-python3.12}" venv="$WORK/venv-benchmark" out="$RESULTS/benchmark"
    record_host
    mkdir -p "$out"
    log "benchmark: install ($python)"
    rm -rf "$venv"
    "$python" -m venv "$venv"
    "$venv/bin/pip" install -q --upgrade pip "cython<3.1" "numpy<2.3"
    "$venv/bin/pip" install -q -r requirements-dev.txt
    corpus "$venv/bin/python"
    # Thread counts the host can actually run: every core, and 2 when there
    # are more than 2. On a 1 vCPU host that is 1 - benchmarking 2 threads on
    # one core measures contention, not the model.
    local threads="$(nproc)"
    [ "$(nproc)" -gt 2 ] && threads="2 $(nproc)"
    log "benchmark: ${MODELS:-base small}, int8, threads $threads"
    # shellcheck disable=SC2086  # word splitting of the two lists is intended
    "$venv/bin/python" scripts/benchmark_stt.py "$WORK/corpus" \
        --models ${MODELS:-base small} --cpu-threads $threads --compute-type int8 \
        --model-root "$STT_MODEL_ROOT" --language id --repeat "${REPEAT:-3}" --show-text \
        --output "$out/benchmark.md" --json "$out/benchmark.json"
    # Loads buffalo_l (~700 MB). Opt-in: on a small host already running the
    # Face service, a second copy of the model is how the OOM killer gets
    # involved.
    if [ "${FACE_MEMORY:-0}" = "1" ]; then
        log "benchmark: Face memory (real buffalo_l)"
        PYTHONPATH="$REPO" "$venv/bin/python" scripts/smoke_face.py --output "$out/face-memory.json" > /dev/null
    fi
}

run_staging() {
    [ "$(id -u)" -eq 0 ] || { echo "staging needs root (sudo)"; exit 1; }
    [ "${CONFIRM_THROWAWAY_HOST:-}" = "yes" ] || {
        echo "staging installs users, systemd units and nginx on THIS host."
        echo "Run it on a throwaway VM with CONFIRM_THROWAWAY_HOST=yes."
        exit 1
    }
    local model="${STAGED_MODEL:-small}" key out="$RESULTS/staging"
    key="staging-$(python3 -c 'import secrets; print(secrets.token_urlsafe(16))')"
    mkdir -p "$out"

    log "staging: packages (README step 1)"
    apt-get update -q
    apt-get install -y -q python3.12 python3.12-venv python3.12-dev build-essential \
        libglib2.0-0 libgomp1 nginx strace rsync

    log "staging: code and venv (README step 2), runtime requirements only"
    id faceapi > /dev/null 2>&1 || useradd -r -m -d /opt/face-api -s /usr/sbin/nologin faceapi
    mkdir -p /opt/face-api/app /opt/face-api/speech /opt/face-api/models/stt
    rsync -a --delete --exclude .git --exclude validation-results ./ /opt/face-api/app/
    chown -R faceapi:faceapi /opt/face-api
    sudo -u faceapi python3.12 -m venv /opt/face-api/venv
    sudo -u faceapi /opt/face-api/venv/bin/pip install -q --upgrade pip "cython<3.1" "numpy<2.3"
    sudo -u faceapi /opt/face-api/venv/bin/pip install -q -r /opt/face-api/app/requirements.txt
    sudo -u faceapi /opt/face-api/venv/bin/pip install -q httpx  # the smoke test's client

    log "staging: configuration (README step 3), one key on both runtimes"
    # One key on purpose: then only the separate nginx zones keep Speech off Face's budget.
    sudo -u faceapi cp /opt/face-api/app/.env.example /opt/face-api/app/.env
    sed -i -e "s|^FSA_API_KEYS=.*|FSA_API_KEYS=laravel:$key|" \
        -e "s|^FSA_CHALLENGE_SECRET=.*|FSA_CHALLENGE_SECRET=$key-hmac|" /opt/face-api/app/.env
    echo "FSA_MODEL_ROOT=/opt/face-api/models" >> /opt/face-api/app/.env
    sed -e "s|^FSA_API_KEYS=.*|FSA_API_KEYS=laravel:$key|" -e "s|^FSA_STT_MODEL=.*|FSA_STT_MODEL=$model|" \
        deploy/env/speech.env.example > /opt/face-api/speech.env
    chown faceapi:faceapi /opt/face-api/app/.env /opt/face-api/speech.env
    chmod 600 /opt/face-api/app/.env /opt/face-api/speech.env

    log "staging: provision models"
    sudo -u faceapi /opt/face-api/venv/bin/python -c "from insightface.app import FaceAnalysis; \
FaceAnalysis(name='buffalo_l', root='/opt/face-api/models', \
allowed_modules=['detection','recognition','landmark_3d_68']).prepare(ctx_id=-1)"
    (cd /opt/face-api/app && sudo -u faceapi /opt/face-api/venv/bin/python \
        scripts/provision_stt_model.py --env-file /opt/face-api/speech.env) | tee "$out/provision.json"

    log "staging: systemd units as shipped"
    cp deploy/systemd/fsa-face.service deploy/systemd/fsa-speech.service /etc/systemd/system/
    systemd-analyze verify /etc/systemd/system/fsa-face.service /etc/systemd/system/fsa-speech.service
    systemctl daemon-reload
    systemctl enable --now fsa-face fsa-speech
    for port in 8001 8002; do
        for _ in $(seq 1 120); do curl -sf "http://127.0.0.1:$port/api/v1/health" && break; sleep 1; done
        echo
    done

    log "staging: nginx with a self-signed certificate"
    mkdir -p /etc/letsencrypt/live/face-api.example.com
    [ -f /etc/letsencrypt/live/face-api.example.com/fullchain.pem ] || \
        openssl req -x509 -nodes -newkey rsa:2048 -days 1 -subj "/CN=face-api.example.com" \
            -keyout /etc/letsencrypt/live/face-api.example.com/privkey.pem \
            -out /etc/letsencrypt/live/face-api.example.com/fullchain.pem
    cp deploy/nginx/fsa-zones.conf /etc/nginx/conf.d/
    cp deploy/nginx/fsa.conf /etc/nginx/sites-available/fsa.conf
    rm -f /etc/nginx/sites-enabled/default
    ln -sf /etc/nginx/sites-available/fsa.conf /etc/nginx/sites-enabled/fsa.conf
    nginx -t
    systemctl reload nginx

    log "staging: smoke test (step 18)"
    corpus /opt/face-api/venv/bin/python
    status=0
    /opt/face-api/venv/bin/python scripts/smoke_deploy.py \
        --url https://127.0.0.1 --insecure --face-key "$key" --speech-key "$key" \
        --audio "$WORK/corpus/pendek_pria/02.wav" --systemd --output "$out/smoke.json" \
        | tee "$out/smoke.txt" || status=$?

    for unit in fsa-face fsa-speech; do
        systemctl show "$unit" -p ActiveState -p MemoryMax -p MemoryPeak -p MemoryCurrent \
            -p CPUWeight -p NRestarts -p Environment | grep -v '^Environment=.*KEY' > "$out/$unit.show" || true
        journalctl -u "$unit" --no-pager > "$out/$unit.journal" || true
    done
    cp /var/log/nginx/error.log "$out/nginx-error.log" || true
    exit "$status"
}

case "$MODE" in
    matrix) run_matrix "$@" ;;
    benchmark) run_benchmark "$@" ;;
    staging) run_staging ;;
    *) sed -n '2,32p' "$0"; exit 2 ;;
esac
echo
echo "results: $RESULTS"
