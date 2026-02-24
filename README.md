# HRM FlashRenderer

Deterministischer Retrieval‑und‑Rendering‑Stack für lokale, VRAM‑arme Inferenz.

Dieses Repository kombiniert:
- **HRM Core (C++17)** für deterministische Retrieval- und Ranking-Schritte.
- **Python-Orchestrierung (`hrm_flash`)** für Prompt-Bau, HRM-Abfragen, Daemon/HTTP-Service.
- **FlashAttention Custom CUDA Extension** für schnelle Inferenzpfade.
- **Tensor-Parallel Engine** für lokale HF‑Modelle mit World Size 2/3/4.

## Ziele

- Große Wissensbasis lokal halten (SSD/RAM/SQLite + Router-Index).
- Deterministische Quellenwahl über HRM.
- Rendering mit lokalem LLM, optional per persistentem Daemon.
- Produktionsnahe Schnittstellen via CLI und HTTP.

## Repository-Struktur

- `hrm_core/` – C++ Bibliothek + CLI (`prep`, `build`, `query`) für HRM.
- `hrm_flash/` – Python-CLI, HTTP-Service, Daemon-Client/-Server, Prompt-Budgeting.
- `engine/` – Tensor-Parallel Generierung und Gewichtslader.
- `flashattention_custom/` + `csrc/` – CUDA/C++ Extension (`flashattention_custom._ext`).
- `renderer/` – Python Renderer-Einstiegspunkte/Abhängigkeiten.
- `scripts/` – Build-/Serve-/Bootstrap-Helfer.
- `docs/INTEGRATION.md` – Integrationspfade (HTTP, Daemon, Embedded HRM).

## Voraussetzungen

### System
- Linux
- Python 3.10+
- CUDA-fähige Umgebung für die Custom-Extension (wenn genutzt)
- CMake + C++17 Toolchain für `hrm_core`
- SQLite3 dev libs für `hrm_core`

### Python
Installiere je nach Bedarf:

```bash
python -m pip install -r requirements.prod.txt
python -m pip install -r requirements.server.txt
```

## Installation

Build/Install als Paket (inkl. Entry-Points):

```bash
python -m pip install -e .
```

Das registriert u. a.:
- `hrm-flash`
- `hrm-flashd`
- `hrm-flash-serve`
- `flash-kernel-test`
- `flash-append-test`

## HRM Core bauen

```bash
make build
make test
```

Alternativ direkt mit CMake:

```bash
cmake -S hrm_core -B hrm_core/build -DCMAKE_BUILD_TYPE=Release
cmake --build hrm_core/build -j
ctest --test-dir hrm_core/build
```

## Minimales End-to-End Beispiel

### 1) Payloads erzeugen

```bash
hrm_core/build/hrm prep --input input.txt --out payloads.jsonl --cluster-size 200
```

### 2) Modellindex bauen

```bash
hrm_core/build/hrm build --payloads payloads.jsonl --outdir model
```

### 3) Generierung über CLI

```bash
hrm-flash generate \
  --hrm_model ./model \
  --llm_model /pfad/zum/hf-modell \
  --prompt "Was steht in den Quellen?" \
  --world 2 \
  --max_new_tokens 128
```

## Betriebsarten

### A) Persistent Daemon

```bash
hrm-flash daemon --model /pfad/zum/hf-modell --world 2 --port 5555 --local_files_only
```

Dann per Client:

```bash
export HRM_FLASH_DAEMON=127.0.0.1:5555
export HRM_FLASH_AUTHKEY=hrmflash
hrm-flash generate --hrm_model ./model --llm_model /pfad/zum/hf-modell --prompt "..." --use_daemon
```

### B) HTTP API

```bash
hrm-flash serve \
  --hrm_model ./model \
  --llm_model /pfad/zum/hf-modell \
  --world 2 \
  --port 8080 \
  --local_files_only
```

Healthcheck:

```bash
curl -s http://127.0.0.1:8080/v1/health
```

Generate:

```bash
curl -s http://127.0.0.1:8080/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"...","max_new_tokens":128}'
```

## Determinismus & Prompt-Budgeting

- HRM-Retrieval ist deterministisch ausgelegt.
- Prompt-Budgeting reduziert Kontextüberschreitungen (`max_seq_len`, `reserve_prompt_tokens`).
- Bei fehlenden Quellen kann die API deterministisch mit **"I don't know."** antworten.

## Wichtige CLI-Parameter

- Retrieval: `--top_k`, `--top_m`, `--k`
- Quellenbudget: `--max_sources`, `--max_chars_per_source`
- Sequenzbudget: `--max_seq_len`, `--reserve_prompt_tokens`, `--disable_token_budget`
- Decodingbudget: `--max_new_tokens`, `--prefill_chunk_size`
- Deployment: `--world`, `--local_files_only`, `--max_concurrent`

## Troubleshooting

- **TP-World Fehler**: Prüfe Modell-Head/Shard-Kompatibilität und `--world` (2/3/4).
- **Keine HRM-Binary gefunden**: `--hrm_bin` setzen oder `hrm_core` bauen.
- **CUDA Extension Build failt**: CUDA Toolkit, NVCC/Arch (`sm_75`) und PyTorch-Version abgleichen.

## Rechtliches / Copyright

- Dieses Repository enthält urheberrechtlich geschützten Quellcode.
- Lizenz- und Nutzungsbedingungen stehen in `LICENSE`.
- Zusätzliche Rechte- und Namensangaben stehen in `COPYRIGHT.md`.
- Drittanbieter-Komponenten behalten ihre jeweiligen Lizenzen.

## Weiterführende Dokumentation

- Integrationsleitfaden: `docs/INTEGRATION.md`
- C++ HRM Core: `hrm_core/README.md`
- Architekturüberblick: `STACK.md`
