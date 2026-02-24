# HRM FlashRenderer

**A superdeterministic hybrid retrieval and rendering system — invented and authored by [Christian Heinrich Hohlfeld](https://christianhohlfeld.com).**

📄 **[Paper: HDR/HRM and Resonant Sparse Attention (PDF)](main.pdf)**

---

## What This Is

HRM FlashRenderer is a retrieval-augmented generation (RAG) stack built for VRAM-constrained environments. It runs large knowledge bases through a deterministic integer-only retrieval engine and feeds a tightly bounded context into a small local renderer — with zero stochastic elements and no dense matrix multiplication in the retrieval core.

The result: identical inputs always produce identical outputs, on any machine, without a GPU.

---

## Core Architecture

```
Query
  ↓
FNV-1a-64 Signature   (2048 quantised bins, 4-bit, word-trigrams + char-5grams)
  ↓
Level-Weighted Router Index   (inverted posting scan, integer scores)
  ↓
Candidate Fetch   (SQLite snippet store)
  ↓
Quantised Overlap Scoring   (min-sum over packed 4-bit bins)
  ↓
Integer MMR Selection   (λ_num=7, λ_den=10, lexicographic tie-break)
  ↓
Deterministic Prompt Budget   (binary search on token counts)
  ↓
Local LLM Renderer   (greedy decoding, 0 VRAM default)
```

Formal proofs of determinism, candidate completeness, and MMR correctness are in [`main.pdf`](main.pdf).

---

## Repository Layout

| Path | Language | Purpose |
|---|---|---|
| `hrm_core/` | C++17 | Retrieval core: signature, router index, overlap scorer, MMR, CLI |
| `hrm_flash/` | Python 3.10+ | CLI, HTTP API, daemon, prompt builder |
| `engine/` | Python | Tensor-parallel HF model generation (world size 2/3/4) |
| `flashattention_custom/` + `csrc/` | CUDA / C++ | Custom FlashAttention kernel (SM75, WMMA Tensor Cores) |
| `renderer/` | Python | llama.cpp renderer entrypoints |
| `tp/` | Python | Tensor-parallel utilities |
| `docs/INTEGRATION.md` | Markdown | HTTP, daemon, and embedded HRM integration guide |

---

## Quickstart

```bash
# Install
git clone https://github.com/ChristianHohlfeld/hrm_flashrenderer.git
cd hrm_flashrenderer
pip install -r requirements.prod.txt
pip install -e .
make build

# Index a corpus
hrm_core/build/hrm prep  --input input.txt --out payloads.jsonl --cluster-size 200
hrm_core/build/hrm build --payloads payloads.jsonl --outdir model

# Generate (CPU-only, 0 VRAM)
hrm-flash generate \
  --hrm_model ./model \
  --llm_model /path/to/model.gguf \
  --prompt "Your question" \
  --max_new_tokens 512
```

## Operation Modes

**Persistent daemon** (warm model, low latency):
```bash
hrm-flash daemon --model /path/to/hf-model --world 2 --port 5555 --local_files_only
```

**HTTP API**:
```bash
hrm-flash serve --hrm_model ./model --llm_model /path/to/hf-model --port 8080
curl -s http://127.0.0.1:8080/v1/generate -H 'Content-Type: application/json' \
  -d '{"prompt":"...","max_new_tokens":128}'
```

---

## Key Parameters

| Flag | Description |
|---|---|
| `--top_k` / `--top_m` / `--k` | Router → candidates → MMR final snippets |
| `--max_sources` / `--max_chars_per_source` | Source budget |
| `--max_seq_len` / `--reserve_prompt_tokens` | Token budget |
| `--world` | Tensor-parallel world size (1 / 2 / 3 / 4) |
| `--local_files_only` | Disable model downloads |

---

## Troubleshooting

| Issue | Fix |
|---|---|
| HRM binary not found | Run `make build` or set `--hrm_bin` |
| CUDA extension build fails | Match CUDA toolkit, NVCC arch (`sm_75`), and PyTorch version |
| TP world size error | Check model head/shard compatibility with `--world` |
| Prompt too long | Reduce `--max_sources`, `--max_chars_per_source`, or `--max_seq_len` |

---

## Further Reading

- `docs/INTEGRATION.md` — HTTP, daemon, and embedded HRM integration
- `hrm_core/README.md` — C++ retrieval core documentation
- `STACK.md` — Architecture overview
- [`main.pdf`](main.pdf) — Formal paper with proofs

---

## License & Rights

© 2026 Christian Heinrich Hohlfeld. All rights reserved.
No permission is granted to use, copy, modify, or redistribute this work without prior written consent.
See [`LICENSE`](LICENSE), [`COPYRIGHT.md`](COPYRIGHT.md), and [`NOTICE`](NOTICE).

---

**Christian Heinrich Hohlfeld**, B.Sc. — Independent Researcher & Senior Software Engineer, Konstanz, Germany

[christianhohlfeld.com](https://christianhohlfeld.com) · [ORCID 0009-0003-6634-9045](https://orcid.org/0009-0003-6634-9045) · [LinkedIn](https://www.linkedin.com/in/christian-hohlfeld/)
