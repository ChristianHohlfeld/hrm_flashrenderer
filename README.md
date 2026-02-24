# HRM FlashRenderer

**A superdeterministic hybrid retrieval and rendering system — invented and authored by [Christian Heinrich Hohlfeld](https://christianhohlfeld.com).**

📄 **[Paper: HDR/HRM and Resonant Sparse Attention (PDF)](main.pdf)**

---

## What This Is

HRM FlashRenderer is a retrieval-augmented generation (RAG) stack built for VRAM-constrained environments. The retrieval core is fully deterministic and integer-only. A tightly bounded context is passed to a small local renderer — keeping VRAM usage minimal and outputs reproducible.

**Scope of "MatMul-free" and "deterministic":**

> The **HDR/HRM retrieval and routing core** (`hrm_core/`) contains no dense matrix multiplication and no floating-point arithmetic. All operations are bounded integer arithmetic.
>
> The **renderer** (LLM inference stage) may use standard matrix operations internally. Determinism of the renderer output is guaranteed only under the CPU path with greedy decoding and a fixed seed; GPU inference may vary across hardware and driver versions.

Formal proofs of retrieval determinism, candidate completeness, and MMR correctness are in [`main.pdf`](main.pdf).

---

## Core Pipeline

```
Query
  ↓
FNV-1a-64 Signature        word-trigrams + char-5grams → 2048 quantised bins (4-bit, 16 levels)
  ↓
Level-Weighted Router       inverted posting scan, integer scores, deterministic tie-break
  ↓
Candidate Fetch             SQLite snippet store
  ↓
Quantised Overlap Scoring   min-sum over packed 4-bit bins (integer)
  ↓
Integer MMR Selection       λ_num=7 / λ_den=10, lexicographic tie-break
  ↓
Deterministic Prompt Budget binary search on token counts
  ↓
Local LLM Renderer          greedy decoding; CPU path = 0 VRAM, GPU path optional
```

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

## Other Operation Modes

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
| `--max_sources` / `--max_chars_per_source` | Source budget per prompt |
| `--max_seq_len` / `--reserve_prompt_tokens` | Token budget |
| `--world` | Tensor-parallel world size (1 / 2 / 3 / 4) |
| `--local_files_only` | Disable model downloads |

---

## Determinism Scope

| Component | Determinism |
|---|---|
| Signature (FNV-1a-64, n-grams) | ✅ Hard — identical output on any conforming platform |
| Router index (level-weighted scan) | ✅ Hard — integer scores, tie-break by `cid` |
| Overlap scoring (min-sum, 4-bit) | ✅ Hard — purely integer |
| MMR selection | ✅ Hard — integer objective, tie-break by `sid` |
| Prompt budget (binary search) | ✅ Hard — deterministic under fixed tokeniser |
| Renderer — CPU path (greedy, fixed seed) | ✅ Strong — bit-identical under same model weights |
| Renderer — GPU path (TP, CUDA) | ⚠️ Best-effort — may vary across hardware/drivers |

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

- [`main.pdf`](main.pdf) — Formal paper with proofs
- [`THIRD_PARTY.md`](THIRD_PARTY.md) — Third-party component licenses
- [`docs/INTEGRATION.md`](docs/INTEGRATION.md) — HTTP, daemon, and embedded HRM guide
- [`hrm_core/README.md`](hrm_core/README.md) — C++ retrieval core documentation
- [`STACK.md`](STACK.md) — Architecture overview

---

## License & Rights

© 2026 Christian Heinrich Hohlfeld. All rights reserved.

The original source code authored by Christian Heinrich Hohlfeld (retrieval core, orchestration, documentation) is proprietary. No permission is granted to use, copy, modify, or redistribute without prior written consent.

Third-party components (PyTorch, Transformers, FastAPI, CUDA, etc.) retain their respective licenses — see [`THIRD_PARTY.md`](THIRD_PARTY.md).

See also: [`LICENSE`](LICENSE) · [`COPYRIGHT.md`](COPYRIGHT.md) · [`NOTICE`](NOTICE)

---

**Christian Heinrich Hohlfeld**, B.Sc. — Independent Researcher & Senior Software Engineer, Konstanz, Germany

[christianhohlfeld.com](https://christianhohlfeld.com) · [ORCID 0009-0003-6634-9045](https://orcid.org/0009-0003-6634-9045) · [LinkedIn](https://www.linkedin.com/in/christian-hohlfeld/)
