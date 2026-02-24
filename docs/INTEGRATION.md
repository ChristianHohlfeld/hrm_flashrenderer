# Integration Guide (Production)

This project is designed as a **frontstack**:

- HRM = deterministic local retrieval over a large knowledge mass (SSD/RAM)
- FlashRenderer = small local HF model (7B–9B) using SM75 fused attention + TP

You integrate via one of these stable interfaces:

## 1) HTTP API (recommended)

Start:

```bash
hrm-flash serve --hrm_model ./model --llm_model /path/to/llm --world 2 --port 8080 --local_files_only
```

Request:

```bash
curl -s http://127.0.0.1:8080/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"...","max_new_tokens":128}'
```

Response:

```json
{
  "ok": true,
  "text": "...",
  "sources": [
    {"sid":"0001#s0000","cid":"0001","rel":123,"txt":"...","txt_c":"..."}
  ]
}
```

Notes:
- Deterministic no-sources: returns "I don't know." with empty sources.
- Prompt budgeting prevents overshooting max_seq_len.

## 2) Persistent daemon + CLI

Daemon:

```bash
export HRM_FLASH_DAEMON=127.0.0.1:5555
export HRM_FLASH_AUTHKEY=hrmflash
hrm-flash daemon --model /path/to/llm --world 2 --port 5555 --local_files_only
```

Client:

```bash
hrm-flash generate --hrm_model ./model --llm_model /path/to/llm --prompt "..." --use_daemon
```

## 3) Embedding HRM retrieval without subprocess

If you build `hrm_core/build/libhrm_api.so`, Python will use it automatically.
Otherwise, `hrm` subprocess is used as a fallback.

Build:

```bash
cmake -S hrm_core -B hrm_core/build -DCMAKE_BUILD_TYPE=Release
cmake --build hrm_core/build -j
```
