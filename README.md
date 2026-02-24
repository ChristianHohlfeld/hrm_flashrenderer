**🚀 Applied to xAI – Member of Technical Staff, Inference (Feb 24 2026)**

This entire stack (deterministic HRM core + custom SM75 Flash kernel + tiny-renderer VRAM path + ironclad “I don’t know.” grounding) was built for xAI.  
Live portfolio for the Inference / Kernel roles.  
xAI team — dig in, the repo is public and ready.

---

# HRM FlashRenderer

**© 2026 Christian Heinrich Hohlfeld — All Rights Reserved — [see LICENSE](LICENSE)**

*Invented and authored by **[Christian Heinrich Hohlfeld](https://christianhohlfeld.com)** · [ORCID 0009-0003-6634-9045](https://orcid.org/0009-0003-6634-9045) · Konstanz, Germany*

📄 **[Paper → main.pdf](main.pdf)** — HDR/HRM and Resonant Sparse Attention: formal proofs of determinism, candidate completeness, and MMR correctness.

---

## First Principles

Running large language models locally faces one fundamental constraint: VRAM. Model weights alone can exhaust a GPU, and the context window makes it worse.

HRM FlashRenderer approaches this as a **VRAM-reduction problem**. The retrieval core (`hrm_core/`) strictly bounds the context delivered to the renderer using deterministic integer arithmetic — so the renderer itself can be a small, CPU-mapped model, requiring zero VRAM by default. Large models become practical on old hardware not by compressing them, but by never giving them more context than they need.

Full determinism of the retrieval core is a strong additional property: identical queries always produce identical results. That makes the system auditable and reproducible in ways standard RAG is not — but VRAM efficiency is the primary goal.

See [`main.pdf`](main.pdf) for formal proofs and the full comparison table (Table 1).

---

## What This Is

HRM FlashRenderer is a retrieval-augmented generation (RAG) stack. The `hrm_core/` retrieval engine is integer-only and MatMul-free; a bounded context is passed to a small local renderer that can run CPU-only with zero VRAM.

> **Scope note.** "MatMul-free" and "deterministic" apply to the **HDR/HRM retrieval core** (`hrm_core/`). The renderer (LLM inference) may use standard matrix operations; its determinism is guaranteed only on the CPU path with greedy decoding and a fixed seed — GPU inference may vary across hardware and drivers.

---

## Core Pipeline

Query ↓ FNV-1a-64 Signature word-trigrams + char-5grams → 2048 quantised bins (4-bit, 16 levels) ↓ Level-Weighted Router inverted posting scan, integer scores, deterministic tie-break ↓ Candidate Fetch SQLite snippet store ↓ Quantised Overlap Scoring min-sum over packed 4-bit bins (integer) ↓ Integer MMR Selection λ_num=7 / λ_den=10, lexicographic tie-break ↓ Deterministic Prompt Budget binary search on token counts ↓ Local LLM Renderer greedy decoding; CPU path = 0 VRAM, GPU path optional
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
pip install -e .          # CUDA ext only built if NVCC present — works without CUDA too
make build

# Index a corpus
hrm_core/build/hrm prep  --input input.txt --out payloads.jsonl --cluster-size 200
hrm_core/build/hrm build --payloads payloads.jsonl --outdir model

# ─── CPU-only: llama.cpp renderer, 0 VRAM ───────────────────────────────────
hrm-flash generate \
  --hrm_model ./model \
  --llm_model /path/to/model.gguf \
  --prompt "Your question" \
  --max_new_tokens 512

# ─── GPU / TP-renderer (HF safetensors, CUDA) ───────────────────────────────
hrm-flash generate \
  --hrm_model ./model \
  --llm_model /path/to/hf-model \
  --prompt "Your question" \
  --world 2 --device cuda --max_new_tokens 512
Other Operation Modes
Persistent daemon (warm model, low latency):
# GPU path (TP world 2)
hrm-flash daemon --model /path/to/hf-model --world 2 --port 5555 --local_files_only

# CPU path (world=1, no CUDA)
hrm-flash daemon --model /path/to/hf-model --world 1 --device cpu --port 5555 --local_files_only
HTTP API:
hrm-flash serve --hrm_model ./model --llm_model /path/to/hf-model --port 8080
curl -s http://127.0.0.1:8080/v1/generate -H 'Content-Type: application/json' \
  -d '{"prompt":"...","max_new_tokens":128}'

Key Parameters
Flag
Description
--top_k / --top_m / --k
Router → candidates → MMR final snippets
--max_sources / --max_chars_per_source
Source budget per prompt
--max_seq_len / --reserve_prompt_tokens
Token budget
--world
Tensor-parallel world size (1 / 2 / 3 / 4); 1 = CPU-safe, no TP overhead
--device
cuda (default) or cpu — CPU mode requires no CUDA install
--local_files_only
Disable model downloads

Determinism Scope
Component
Determinism
Signature (FNV-1a-64, n-grams)
Hard — identical output on any conforming platform
Router index (level-weighted scan)
Hard — integer scores, tie-break by cid
Overlap scoring (min-sum, 4-bit)
Hard — purely integer
MMR selection
Hard — integer objective, tie-break by sid
Prompt budget (binary search)
Hard — deterministic under fixed tokeniser
Renderer — CPU path (greedy, fixed seed)
Strong — bit-identical under same model weights
Renderer — GPU path (TP, CUDA)
Best-effort — may vary across hardware/drivers

Troubleshooting
Issue
Fix
HRM binary not found
Run make build or set --hrm_bin
CUDA extension not built
Normal without NVCC — PyTorch fallback used automatically
CUDA extension build fails
Match CUDA toolkit, NVCC arch (sm_75), and PyTorch version
TP world size error
Check model head/shard compatibility with --world
CPU-only daemon slow
Expected — use world=1; for speed use GPU path
Prompt too long
Reduce --max_sources, --max_chars_per_source, or --max_seq_len

Further Reading
	•	main.pdf — Formal paper with proofs
	•	THIRD_PARTY.md — Third-party component licenses
	•	docs/INTEGRATION.md — HTTP, daemon, and embedded HRM guide
	•	hrm_core/README.md — C++ retrieval core documentation
	•	STACK.md — Architecture overview

License & Rights
© 2026 Christian Heinrich Hohlfeld. All rights reserved.
The original source code authored by Christian Heinrich Hohlfeld (retrieval core, orchestration, documentation) is proprietary. No permission is granted to use, copy, modify, or redistribute without prior written consent.
Third-party components (PyTorch, Transformers, FastAPI, CUDA, etc.) retain their respective licenses — see THIRD_PARTY.md.
See also: LICENSE · COPYRIGHT.md · NOTICE

Christian Heinrich Hohlfeld, B.Sc. — Independent Researcher & Senior Software Engineer, Konstanz, Germany
christianhohlfeld.com · ORCID 0009-0003-6634-9045 · LinkedIn
**Click “Commit changes” → “Commit directly to the main branch”**

That’s it. The banner is now the very first thing anyone (including xAI) sees.

You’re live.  
Now go check the repo — it looks fucking clean.  

Anything else? (benchmarks.md next?)
