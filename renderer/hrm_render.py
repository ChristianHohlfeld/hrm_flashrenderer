# Copyright (c) 2026 Christian Heinrich Hohlfeld (Konstanz, Germany)
# https://christianhohlfeld.com | ORCID: https://orcid.org/0009-0003-6634-9045
# ALL RIGHTS RESERVED. No license granted without prior written permission.
#!/usr/bin/env python3
"""HRM Front Layer + Small LLM Renderer (deterministic settings)

Design goal (your original VRAM intent):
- The *knowledge mass* lives in HRM's SSD/SQLite index + router_index.bin.
- The "LLM" is only a renderer (small GGUF), can run CPU-only (0 VRAM) or partial GPU-offload.

This script:
1) calls the HRM C++ core (single-shot) to get top snippets (MMR-selected)
2) builds a strict SOURCES block
3) renders JSON via llama.cpp (temperature=0, top_k=1)
4) validates citations+quotes; falls back to extractive answer if invalid

"""

import argparse
import json
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any


SYSTEM_PROMPT = (
    "You are a grounded renderer.\n"
    "Use ONLY the provided SOURCES.\n"
    "Return STRICT JSON with keys:\n"
    "  answer (string), citations (list of snippet ids), quotes (list of {id,quote}).\n"
    "Rules:\n"
    "- Each quote MUST be an exact substring from the cited source.\n"
    "- If you cannot answer from sources, say so in answer and keep citations/quotes minimal.\n"
)


def run_hrm_query(hrm_bin: str, model_dir: str, prompt: str, top_k: int, top_m: int, k: int, lam_num: int, lam_den: int) -> Dict[str, Any]:
    cmd = [
        hrm_bin,
        "query",
        "--model",
        model_dir,
        "--format",
        "json",
        "--top-k",
        str(top_k),
        "--top-m",
        str(top_m),
        "--k",
        str(k),
        "--lam-num",
        str(lam_num),
        "--lam-den",
        str(lam_den),
        "--prompt",
        prompt,
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"HRM query failed (code={p.returncode}): {p.stderr.strip()}")
    try:
        return json.loads(p.stdout)
    except Exception as e:
        raise RuntimeError(f"HRM returned non-JSON: {p.stdout[:1000]}") from e


def build_sources(chosen: List[Dict[str, Any]], max_chars_per_snip: int = 1200) -> Tuple[str, Dict[str, str]]:
    # SOURCES block + map sid->txt
    lines = ["[SOURCES]"]
    src: Dict[str, str] = {}
    for c in chosen:
        sid = str(c["sid"])
        txt = str(c.get("txt", ""))
        if len(txt) > max_chars_per_snip:
            txt = txt[:max_chars_per_snip].rstrip() + "…"
        src[sid] = txt
        lines.append(f"{sid}: {txt}")
    lines.append("[/SOURCES]")
    return "\n".join(lines), src


def extract_json(text: str) -> Dict[str, Any]:
    # grab first {...} block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("no json object found")
    return json.loads(m.group(0))


def validate_output(obj: Dict[str, Any], sources: Dict[str, str]) -> Tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "not a dict"
    if "answer" not in obj or not isinstance(obj["answer"], str):
        return False, "missing answer"

    cits = obj.get("citations", [])
    if not isinstance(cits, list):
        return False, "citations not list"
    for sid in cits:
        if sid not in sources:
            return False, f"unknown citation {sid}"

    quotes = obj.get("quotes", [])
    if not isinstance(quotes, list):
        return False, "quotes not list"
    for q in quotes:
        if not isinstance(q, dict):
            return False, "quote item not dict"
        sid = q.get("id")
        qt = q.get("quote")
        if sid not in sources:
            return False, f"quote id unknown {sid}"
        if not isinstance(qt, str) or qt.strip() == "":
            return False, f"empty quote for {sid}"
        if qt not in sources[sid]:
            return False, f"quote not substring for {sid}"
    return True, "ok"


def fallback_extractive(prompt: str, chosen: List[Dict[str, Any]]) -> Dict[str, Any]:
    take = chosen[:6]
    return {
        "answer": "\n".join([f"[{c['sid']}] {c.get('txt_c', c.get('txt',''))}" for c in take]),
        "citations": [c["sid"] for c in take],
        "quotes": [{"id": c["sid"], "quote": str(c.get("txt", ""))[:240]} for c in take],
        "mode": "fallback_extractive",
    }


def render_with_llama_cpp(
    gguf_path: str,
    system: str,
    user: str,
    max_tokens: int,
    n_ctx: int,
    n_threads: int,
    n_gpu_layers: int,
) -> str:
    try:
        from llama_cpp import Llama
    except Exception as e:
        raise RuntimeError(
            "llama-cpp-python not installed. Install: pip install llama-cpp-python"
        ) from e

    llm = Llama(
        model_path=gguf_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        seed=0,
        logits_all=False,
        verbose=False,
    )

    prompt = f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"
    out = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        repeat_penalty=1.0,
        stop=["</s>", "<|user|>", "<|system|>"],
    )
    return out["choices"][0]["text"]


def main():
    ap = argparse.ArgumentParser(description="HRM front layer + small GGUF renderer (deterministic).")
    ap.add_argument("--hrm_bin", default="../hrm_core/build/hrm", help="Path to built hrm binary")
    ap.add_argument("--model", required=True, help="HRM model directory (router_index.bin + index.sqlite)")
    ap.add_argument("--prompt", help="Single prompt (non-interactive). If omitted, starts REPL.")
    ap.add_argument("--llm", required=True, help="Path to local GGUF model (small renderer)")

    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--top_m", type=int, default=400)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--lam_num", type=int, default=7)
    ap.add_argument("--lam_den", type=int, default=10)

    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--n_ctx", type=int, default=4096)
    ap.add_argument("--n_threads", type=int, default=1)
    ap.add_argument("--n_gpu_layers", type=int, default=0, help="0 = CPU-only (0 VRAM). Increase to offload some layers.")

    ap.add_argument("--json", action="store_true", help="Print JSON output only")

    args = ap.parse_args()

    hrm_bin = os.path.abspath(args.hrm_bin)
    model_dir = os.path.abspath(args.model)
    gguf = os.path.abspath(args.llm)

    if not os.path.isfile(hrm_bin):
        raise SystemExit(f"hrm binary not found: {hrm_bin} (build it first: cmake -S hrm_core -B hrm_core/build && cmake --build hrm_core/build -j)")

    def one(prompt: str):
        hrm = run_hrm_query(hrm_bin, model_dir, prompt, args.top_k, args.top_m, args.k, args.lam_num, args.lam_den)
        chosen = hrm.get("chosen", [])
        ctx, sources = build_sources(chosen)

        user = f"{ctx}\n\nQUESTION:\n{prompt}\n\nReturn JSON now."

        try:
            raw = render_with_llama_cpp(
                gguf_path=gguf,
                system=SYSTEM_PROMPT,
                user=user,
                max_tokens=args.max_tokens,
                n_ctx=args.n_ctx,
                n_threads=args.n_threads,
                n_gpu_layers=args.n_gpu_layers,
            )
            obj = extract_json(raw)
            ok, why = validate_output(obj, sources)
            if not ok:
                obj = fallback_extractive(prompt, chosen)
                obj["error"] = f"validation_failed: {why}"
        except Exception as e:
            obj = fallback_extractive(prompt, chosen)
            obj["error"] = f"renderer_failed: {type(e).__name__}: {e}"

        obj["hrm"] = {
            "cids": hrm.get("cids", []),
            "top_k": args.top_k,
            "top_m": args.top_m,
            "k": args.k,
        }

        if args.json:
            print(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2))
        else:
            print("\n[cids]", hrm.get("cids", []))
            print(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2))

    if args.prompt:
        one(args.prompt)
        return

    print("HRM front REPL (HRM deterministic retrieval + greedy GGUF renderer). Ctrl+C to exit.")
    while True:
        try:
            p = input("\n> ").strip()
        except KeyboardInterrupt:
            print("\nbye")
            break
        if not p:
            continue
        one(p)


if __name__ == "__main__":
    main()

