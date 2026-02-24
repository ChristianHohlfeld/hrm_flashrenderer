# HRM FlashRenderer

> **⚠️ All Rights Reserved — No License Granted**
> Publicly viewable, but no permission is granted to use, copy, modify, or redistribute.
> Licensing: [christianhohlfeld.com](https://christianhohlfeld.com) · ORCID: [0009-0003-6634-9045](https://orcid.org/0009-0003-6634-9045)

---

## Original Invention by Christian Heinrich Hohlfeld

**Christian Heinrich Hohlfeld**, B.Sc.
Born April 5, 1983, Frankfurt am Main, Germany.
Software Engineering — Fachhochschule Konstanz & Universität Konstanz.
[christianhohlfeld.com](https://christianhohlfeld.com)

**Copyright © 2026 Christian Heinrich Hohlfeld. All rights reserved.**

The underlying architecture, the concept of superdeterministic retrieval (HDR/HRM), and the entire MatMul-free, stochastic-free approach are an **original invention** by Christian Heinrich Hohlfeld.
Any use, reproduction, modification, or redistribution requires explicit written consent and must clearly credit the inventor by name.

Prior published works by the same author:
- [Hohlfeld Data Representation (HDR) Method (2024)](https://christianhohlfeld.com/12_07_2024_Hohlfeld_Data_Representation_HDR_Method.pdf)
- [PhO-Compress: Framework for Optical LLM Context Compression](https://christianhohlfeld.com/Christian_Heinrich_Hohlfeld_Konstanz_PhO-Compress_v2.pdf)

---

## First Principles

This system rests on the fundamental insight that large knowledge bases do not need to be encoded in stochastic weight matrices. Instead, knowledge can be held in **deterministic, integer-based structures on persistent storage**, while only a minimally-sized, precisely pre-selected context is passed to a highly optimised renderer.

Core principles of this invention:

- **Full determinism guarantee** in retrieval — proven mathematically in `main.pdf`
- **No MatMul, no floating-point arithmetic** in the retrieval core
- **No probability estimation, no stochasticity** anywhere in the pipeline
- **Minimal VRAM** through strict context bounding
- **Maximum reproducibility**: identical input → identical output, across platforms and runs

---

## Architecture

```
Query
  │
  ▼
Signature          ← FNV-1a-64, word-trigrams + char-5grams → 2048 quantised bins (4-bit, 16 levels)
  │
  ▼
Router Index       ← Level-weighted inverted posting scan (integer scores only)
  │
  ▼
Candidate Fetch    ← SQLite snippet store
  │
  ▼
Overlap Scoring    ← Quantised min-sum over 2048 packed bins (integer)
  │
  ▼
MMR Selection      ← Integer Maximal Marginal Relevance (λ_num=7, λ_den=10)
  │
  ▼
Prompt Builder     ← Deterministic token-budget fitting via binary search
  │
  ▼
Renderer           ← Local LLM, greedy decoding (temp=0, top-k=1), 0 VRAM default
```

### Components

| Directory | Language | Role |
|---|---|---|
| `hrm_core/` | C++17 | **Core invention** — signature, router index, overlap scorer, MMR, CLI |
| `hrm_flash/` | Python 3.10+ | CLI, HTTP service, daemon, prompt builder |
| `engine/` | Python | Tensor-parallel HF model generation (world size 2/3/4) |
| `flashattention_custom/` + `csrc/` | CUDA/C++ | Custom FlashAttention kernel (SM75, WMMA Tensor Cores) |
| `renderer/` | Python | llama.cpp renderer entrypoints |
| `tp/` | Python | Tensor-parallel utilities (norm, linear, vocab parallel) |
| `scripts/` | Shell | Build, serve, bootstrap helpers |
| `docs/` | Markdown | `INTEGRATION.md` — HTTP, daemon, embedded HRM |

---

## Requirements

- Linux (recommended) or Windows (`compile.bat` included)
- Python 3.10+
- CMake + C++17 toolchain for `hrm_core`
- SQLite3 dev libs
- CUDA toolkit + NVCC (`sm_75`) for the FlashAttention extension *(optional)*

```bash
python -m pip install -r requirements.prod.txt
python -m pip install -r requirements.server.txt
```

## Installation

```bash
git clone https://github.com/ChristianHohlfeld/hrm_flashrenderer.git
cd hrm_flashrenderer
pip install -e .
make build
```

Registered entry points: `hrm-flash`, `hrm-flashd`, `hrm-flash-serve`, `flash-kernel-test`, `flash-append-test`.

---

## Minimal End-to-End Example

```bash
# 1. Prepare payloads
hrm_core/build/hrm prep --input input.txt --out payloads.jsonl --cluster-size 200

# 2. Build model index
hrm_core/build/hrm build --payloads payloads.jsonl --outdir model

# 3. Full pipeline (CPU renderer, 0 VRAM)
hrm-flash generate \
  --hrm_model ./model \
  --llm_model /path/to/model.gguf \
  --prompt "Your question here" \
  --world 2 \
  --max_new_tokens 512
```

## Operation Modes

**Daemon** (warm model, low latency):
```bash
hrm-flash daemon --model /path/to/hf-model --world 2 --port 5555 --local_files_only
```

**HTTP API**:
```bash
hrm-flash serve --hrm_model ./model --llm_model /path/to/hf-model --world 2 --port 8080
curl -s http://127.0.0.1:8080/v1/generate -H 'Content-Type: application/json' \
  -d '{"prompt":"...","max_new_tokens":128}'
```

---

## Determinism Guarantees

| Stage | Guarantee |
|---|---|
| Signature | FNV-1a-64, fixed word/char n-gram extraction, no RNG |
| Router Index | Integer level-weighted scores, ties broken by `cid` ascending |
| Overlap scoring | Integer min-sum over packed 4-bit bins |
| MMR | Integer objective, ties broken by `sid` lexicographic order |
| Prompt budget | Binary search on token counts under a fixed tokeniser |
| Renderer | Greedy decoding (temp=0), fixed seed, CPU path |

See `main.pdf` for full formal proofs.

---

## Key CLI Parameters

| Parameter | Description |
|---|---|
| `--top_k` | Router: top chunk clusters |
| `--top_m` | Candidates passed to MMR (default 400) |
| `--k` | Final snippets selected by MMR (default 8) |
| `--max_sources` / `--max_chars_per_source` | Source budget |
| `--max_seq_len` / `--reserve_prompt_tokens` | Token budget |
| `--max_new_tokens` | Max tokens to generate |
| `--world` | Tensor-parallel world size (1/2/3/4) |
| `--local_files_only` | Prevent model downloads |

---

## Troubleshooting

| Problem | Solution |
|---|---|
| TP world size error | Check model head/shard compatibility and `--world` |
| HRM binary not found | Build `hrm_core` or set `--hrm_bin` |
| CUDA extension build fails | Match CUDA toolkit, NVCC/arch (`sm_75`), PyTorch version |
| Prompt too long | Reduce `--max_sources`, `--max_chars_per_source`, `--max_seq_len` |

---

## Legal

All source code, methods, algorithms, and documentation in this repository are the sole intellectual property of **Christian Heinrich Hohlfeld** (Konstanz, Germany).
See `LICENSE` · `COPYRIGHT.md` · `NOTICE` for full rights details.
Third-party dependencies retain their respective licenses.

---

*This entire system and the underlying architecture are an original invention by Christian Heinrich Hohlfeld.*
*— Christian Heinrich Hohlfeld, February 2026*
