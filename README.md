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
