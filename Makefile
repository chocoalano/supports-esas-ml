# Python 3.10–3.12 (batasan wheel onnxruntime + insightface, lihat pyproject).
#
# Dicari, bukan dipatok pada satu versi: mesin yang satu punya 3.12 dari brew,
# yang lain 3.11 dari apt, dan `python3.11: command not found` adalah kegagalan
# pertama yang dilihat orang baru di repo ini. Timpa dengan `make install
# PY=/path/ke/python3.11` kalau interpreternya di luar PATH.
#
# Direktori venv yang sedang aktif dibuang dari PATH sebelum pencarian, bukan
# kandidatnya yang dilewati satu per satu: sebuah venv menaruh `python3.x`
# miliknya paling depan, dan melewati kandidat pertama akan membuang seluruh
# versi itu alih-alih melanjutkan ke interpreter berikutnya di PATH.
#
# Alasannya sendiri: menjalankan `make install` sementara venv proyek lain aktif
# akan membangun `.venv` di sini dari interpreter proyek itu — persis kekacauan
# yang seluruh berkas ini ada untuk mencegahnya, dan yang gagalnya muncul jauh
# dari sebabnya.
#
# Satu jebakan make di sini tidak terlihat sebagai kesalahan shell: make
# menghitung tanda kurung untuk menemukan ujung `$(shell ...)`, sehingga `)`
# milik sebuah pola `case` akan menutup panggilan itu di tengah jalan. Karena
# itu tidak ada `case` di bawah.
PY ?= $(shell \
	if [ -n "$$VIRTUAL_ENV" ]; then \
		PATH=$$(printf '%s' "$$PATH" | tr ':' '\n' | grep -vxF "$$VIRTUAL_ENV/bin" | paste -sd: -); \
		export PATH; \
	fi; \
	for v in 3.12 3.11 3.10; do command -v python$$v 2>/dev/null && break; done)

VENV := .venv
BIN := $(VENV)/bin

# 8000 dipakai Laravel pada pemasangan gabungan (esas-app berjalan di sana),
# dan dua proses yang berebut satu port gagal dengan pesan yang tidak
# menyebut siapa lawannya. Timpa dengan `make run PORT=8002` bila perlu.
PORT ?= 8001
HOST ?= 127.0.0.1

.PHONY: install install-test run serve test lint fmt clean require-python fresh-venv normalise-prompt

require-python:
	@test -n "$(PY)" || { \
		echo "Tidak menemukan python3.12, python3.11, atau python3.10 di PATH."; \
		echo "  macOS : brew install python@3.12"; \
		echo "  Ubuntu: sudo apt install python3.12 python3.12-venv python3.12-dev"; \
		echo "  Atau  : make install PY=/path/ke/python3.11"; \
		exit 1; \
	}
	@{ test -f "$(PY)" && test -x "$(PY)"; } || { \
		echo "PY tidak menunjuk ke interpreter yang bisa dijalankan:"; \
		echo "    [$(PY)]"; \
		echo ""; \
		echo "Tanpa pesan ini, shell hanya menjawab 'is a directory' atau"; \
		echo "'No such file' untuk potongan pertama path-nya, dan make berhenti"; \
		echo "dengan Error 126 tanpa menyebut apa pun."; \
		echo ""; \
		echo "  venv proyek lain aktif?  deactivate, lalu ulangi"; \
		echo "  path-nya benar?          make install PY=/opt/homebrew/bin/python3.12"; \
		exit 1; \
	}
	@"$(PY)" -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)' || { \
		echo "$$("$(PY)" --version 2>&1) di luar rentang yang didukung (3.10–3.12)."; \
		echo "Batasannya milik wheel onnxruntime + insightface, bukan kode ini."; \
		echo "  make install PY=/opt/homebrew/bin/python3.12"; \
		exit 1; \
	}
	@echo "Memakai $$("$(PY)" --version 2>&1) dari $(PY)"

# Sebuah venv menyimpan path absolutnya di dalam shebang setiap skrip konsol dan
# di dalam `activate`. Kalau foldernya disalin atau dipindah — repo ini sendiri
# pernah pindah dari `filament/esas-app/` — `.venv` yang ikut terbawa tetap
# menunjuk lokasi lamanya: `uvicorn` diam-diam menjalankan interpreter proyek
# lain, dan `source .venv/bin/activate` memasang direktori yang tidak ada ke
# PATH sehingga `python` jatuh kembali ke 3.9 milik sistem. Keduanya gagal
# dengan pesan yang tidak menyebut sebabnya, jadi venv asing dibuang di sini.
fresh-venv: require-python
	@if [ -n "$$VIRTUAL_ENV" ] && [ "$$VIRTUAL_ENV" != "$(CURDIR)/$(VENV)" ]; then \
		echo "Catatan: venv lain sedang aktif — $$VIRTUAL_ENV"; \
		echo "  Interpreter untuk membangun $(VENV) dicari di luar venv itu."; \
		echo "  Jalankan 'deactivate' dulu kalau ada yang terasa aneh."; \
	fi
	@if [ -f $(BIN)/pip ] && ! grep -qF "$(CURDIR)" $(BIN)/pip 2>/dev/null; then \
		echo "$(VENV) dibuat di direktori lain — dibuat ulang."; \
		rm -rf $(VENV); \
	fi
	@test -d $(VENV) || "$(PY)" -m venv $(VENV)
	@$(MAKE) --no-print-directory normalise-prompt

# Bug pada template `activate` milik build CPython tertentu — Homebrew 3.12.13
# salah satunya. `__VENV_PROMPT__` sudah berisi `(.venv) ` lengkap dengan
# kurungnya, lalu templatenya membungkusnya sekali lagi, sehingga promptnya
# terbaca `((.venv) )`. `python3` bawaan Apple menghasilkan bentuk yang benar,
# dan `--prompt` tidak bisa menghindarinya: apa pun nilainya tetap dibungkus dua
# kali.
#
# Barisnya ditulis ulang memakai `VIRTUAL_ENV_PROMPT` — variabel yang sudah
# diekspor beberapa baris di atasnya, dan bentuk yang memang dimaksudkan
# upstream. Dijaga oleh grep, jadi ini tidak melakukan apa pun pada venv yang
# templatenya sudah benar, termasuk kelak setelah bug-nya diperbaiki.
normalise-prompt:
	@if [ -f $(BIN)/activate ] && grep -q 'PS1="("' $(BIN)/activate; then \
		sed 's|.*PS1="(".*|    PS1="$${VIRTUAL_ENV_PROMPT}$${PS1:-}"|' $(BIN)/activate > $(BIN)/activate.new \
			&& mv $(BIN)/activate.new $(BIN)/activate \
			&& echo "prompt activate dirapikan: ((.venv) ) -> (.venv)"; \
	fi

install: fresh-venv ## Full runtime (downloads ONNX models on first request)
	$(BIN)/pip install --upgrade pip "cython<3.1" "numpy<2.3"
	$(BIN)/pip install -r requirements-dev.txt

install-test: fresh-venv ## Light install: enough to run the suite, no ONNX models
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements-test.txt

run: ## Development server with autoreload
	$(BIN)/uvicorn app.main:app --reload --host $(HOST) --port $(PORT)

serve: ## Production-style run; see README for systemd/nginx
	$(BIN)/uvicorn app.main:app --host $(HOST) --port $(PORT) --workers 2

test:
	$(BIN)/python -m pytest

lint:
	$(BIN)/ruff check .

fmt:
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
