# HRM Frontstack (Production) — VRAM-minimal by design

**Goal (your original intent):**
- Keep the *knowledge mass* in a deterministic on-disk index (SSD/RAM via OS page cache).
- Use a *small* renderer model (GGUF) that can run CPU-only (**0 VRAM**) or with limited GPU offload.
- Never require a huge Transformer to fully reside in GPU VRAM.

This repo contains:
- `hrm_core/` (C++17): deterministic retrieval core (0 RNG, 0 floats, 0 matrices)
  - builds `router_index.bin` + `index.sqlite`
  - supports one-shot JSON query (`hrm query --prompt "..." --format json`)
- `renderer/hrm_render.py`: deterministic HRM->LLM pipeline
  - uses HRM for routing + MMR selection
  - feeds top snippets into a small local GGUF model (llama.cpp)
  - validates strict JSON + exact quotes; deterministic fallback if invalid

## Build HRM core
```bash
make build
make test
```

## Build an index
```bash
# Example: Tiny Shakespeare
curl -L -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
hrm_core/build/hrm prep --input input.txt --out payloads.jsonl --cluster-size 200
hrm_core/build/hrm build --payloads payloads.jsonl --outdir model
```

## Run the renderer (CPU-only = 0 VRAM)
```bash
python -m pip install -r renderer/requirements.txt
python renderer/hrm_render.py --model model --llm /path/to/small.gguf --n_gpu_layers 0
```

## Optional: limited GPU offload
- increase `--n_gpu_layers` gradually to trade VRAM for speed.
- You still never need the full model in VRAM.

## Why this addresses VRAM limits
Classic LLM stacks keep two big VRAM consumers:
1) **Model weights** (static, huge)
2) **KV cache** (grows with context)

This stack changes the problem:
- The context you feed the renderer is hard-bounded by HRM selection.
- The renderer can be small and/or CPU-mapped, so weights do **not** have to sit in VRAM.

Determinism:
- HRM retrieval is deterministic.
- Renderer is configured to greedy decoding (temperature=0, top_k=1). CPU-only + fixed seed gives strongest stability.
