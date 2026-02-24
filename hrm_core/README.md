# HRM C++ (FULL REDO) — deterministic local retrieval & ranking core

This is a **from-scratch redo** of the HRM v3 core in C++17, structured like a small real project:

- `hrm_core` library (static) with public headers under `include/hrm/*`
- `hrm` CLI with `prep / build / query`
- Unit tests (`ctest`) to validate determinism-critical components

### Core properties
- **0 RNG, 0 floats, 0 matrices**
- Signature: **word 3-grams + char 5-grams**, FNV-1a 64-bit, 2048 bins, quantized 0..15
- Router index: cache-friendly **flat postings + offsets** (binary `HRMIDX2`)
- Snippet store: SQLite3 (`snips`) with nibble-packed qbins (1024 bytes)

## Build
Requirements: CMake >= 3.16, C++17 compiler, SQLite3 dev package.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build
```

## Usage

### 1) Prepare payloads.jsonl from a text file (Tiny Shakespeare example)
```bash
curl -L -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
build/hrm prep --input input.txt --out payloads.jsonl --cluster-size 200
```

### 2) Build model (router_index.bin + index.sqlite + meta.json)
```bash
build/hrm build --payloads payloads.jsonl --outdir model
```

### 3) Query (REPL)
```bash
build/hrm query --model model --top-k 5 --top-m 400 --k 8 --format text
```

JSON output:
```bash
build/hrm query --model model --format json
```

## Notes
- Output is deterministic for the same inputs.
- Tie-breaks are explicit (score desc, id asc).
