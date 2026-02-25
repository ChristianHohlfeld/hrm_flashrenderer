HRM FlashRenderer
=================

**Mission: Extreme VRAM Reduction for Large Models on Legacy GPUs**

This repository presents my invention: the **HRM (Hash Retrieval Model)**.

The core objective is to solve a strict hardware constraint: running massive LLMs (32B+ parameters) on old, VRAM-starved GPUs such as 2×11 GB RTX 2080 Ti.

It achieves this through a fundamental reorientation: Instead of trying to make heavy matrix multiplications fit into tiny VRAM, the entire architecture shifts the burden of knowledge retrieval away from the GPU entirely.

### Core Philosophy

*   **HRM Core & Context Bounding** The HRM Core selects context so precisely that we can work with a hard prompt budget. This is the real lever that makes 30B+ models possible on 11 GB VRAM.
    
*   **VRAM-Reduction Layer** The GGUF path is not a fallback — it is the current realization of the “Local Renderer” for maximum efficiency on legacy hardware.
    
*   **Vision: Resonant Sparse Attention (RSA)** The long-term goal is to eventually replace dense MatMuls with associative recall. GPTQ-Int4 is only a temporary bridge; the true direction is sparse, resonant, memory-efficient attention.
    

This entire architecture was realized through Agentic Coding: translating my conceptual paper and strict implementation path into high-performance, bare-metal code using LLMs.

### Why HRM-Flash?

This stack is built on three pillars to maximize VRAM efficiency and architectural reliability:

1.  **Zero-VRAM Retrieval (HRM Core):** Standard vector RAG wastes VRAM on embeddings and floating-point operations. The C++17 HRM Core replaces this with deterministic, integer-only indexing — completely offloading retrieval to CPU/RAM/SSD.
    
2.  **Legacy-Tuned Native CUDA Optimization:** A custom SM75 FlashAttention kernel, specifically tuned for Turing GPUs (2080 Ti, T4, etc.) with paged KV-cache and optimized append operations.
    
3.  **Decoupled Knowledge & Compute:** Knowledge base (Index) and reasoning engine (LLM) are strictly separated. This allows running retrieval on zero-VRAM CPU while the LLM operates at the absolute hardware limit on GPU.
    

### Reality Check

Running large models (32B+) on 11 GB per GPU is an extreme edge case. It requires strict prompt budgeting, adjusted parameters, and precise memory management.

### Two Execution Paths

**1\. High-Performance Path (Custom Kernel)** Uses my tensor-parallel engine and custom Turing-optimized FlashAttention kernel.

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   hrm-flash generate \    --hrm_model model_index \    --llm_model models/Qwen2.5-32B-Instruct-GPTQ-Int4 \    --world 2 \    --prompt "Your question"   `

**2\. Extreme Low-VRAM Fallback (for 2×11 GB GPUs)** Uses GGUF + llama.cpp with tensor\_split.

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python renderer/hrm_render.py \    --model model_index \    --hrm_bin hrm_core/build/hrm \    --llm /path/to/model.gguf \    --n_gpu_layers 50 \    --tensor_split 0.5 0.5   `

Requirements
------------

**System:**

*   Ubuntu / Linux
    
*   CUDA Toolkit (for full stack)
    

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   sudo apt install libsqlite3-dev build-essential cmake   `

**Python:**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python3 -m pip install -r requirements.prod.txt  python3 -m pip install -e .   `

**Build:**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   make build   `

Troubleshooting
---------------

**Problem**

**Solution**

**Could NOT find SQLite3**

sudo apt install libsqlite3-dev + make build

**No space left on device**

Check df -h, avoid keeping GGUF and GPTQ at the same time

**HRM query failed (code=2)**

Rebuild index (hrm\_core/build/hrm build ...)

**Model path does not exist**

Use absolute path to the model

**OOM in full stack**

Reduce --max\_seq\_len to 512 or lower

**hrm binary not found**

Explicitly set --hrm\_bin hrm\_core/build/hrm

Quick Start (Local Renderer)
----------------------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   # Build index  hrm_core/build/hrm prep --input your_data.txt --out payloads.jsonl  hrm_core/build/hrm build --payloads payloads.jsonl --outdir model_index  # Run  CUDA_VISIBLE_DEVICES=0,1 python renderer/hrm_render.py \    --model model_index \    --hrm_bin hrm_core/build/hrm \    --llm /path/to/model.gguf \    --prompt "Your question" \    --n_gpu_layers 50 \    --n_ctx 4096 \    --max_tokens 512 \    --top_k 4   `

Disclaimer
==========

I’m Christian Heinrich Hohlfeld, B.Sc. Software Engineering.

**Full honesty:** I’m not a traditional CUDA kernel veteran or ninja. What I do really well is guide AI precisely towards my goals — and turn ideas into clean, working, performant code very fast.

To be completely transparent: I built and open-sourced **hrm\_flashrenderer** using my 15+ years of software engineering experience combined with Agentic Coding to demonstrate this approach, successfully generating a custom SM75 FlashAttention kernel with paged KV + append.

I want to bring this direct, pragmatic way of working to **xAI**. Ready to relocate to Bay Area / Seattle tomorrow.

**Let’s talk.**

[christianhohlfeld.com](https://christianhohlfeld.com) | [GitHub: ChristianHohlfeld](https://github.com/ChristianHohlfeld)
