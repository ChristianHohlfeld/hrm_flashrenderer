# Copyright (c) 2026 Christian Heinrich Hohlfeld (Konstanz, Germany)
# https://christianhohlfeld.com | ORCID: https://orcid.org/0009-0003-6634-9045
# ALL RIGHTS RESERVED. No license granted without prior written permission.
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any, Optional

from hrm_flash.hrm_client import run_hrm_query
from hrm_flash.prompt_builder import (
    build_prompt_for_mode,
    build_sources,
    fit_prompt_to_token_budget,
    normalize_mode,
)
from hrm_flash.utils import find_hrm_binary


def _require_fastapi():
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel
        return FastAPI, HTTPException, JSONResponse, BaseModel
    except Exception as e:
        raise SystemExit(
            "ERR: Missing server dependencies. Install with:\n"
            "  python -m pip install fastapi uvicorn pydantic\n"
            f"Details: {e}"
        )


class _State:
    def __init__(self):
        self.repo_root = Path(__file__).resolve().parents[1]
        self.hrm_bin: Path | None = None
        self.hrm_model: Path | None = None
        self.tokenizer_source: str | None = None
        self.world: int | None = None
        self.max_seq_len: int = 8192
        self.prefill_chunk_size: int = 1024
        self.max_new_tokens: int = 256
        self.local_files_only: bool = True
        self.max_sources: int = 16
        self.max_chars_per_source: int = 1200
        self.reserve_prompt_tokens: int = 16
        self.disable_token_budget: bool = False
        self.expose_sources: bool = False
        self.sem: asyncio.Semaphore | None = None
        self.backend: str = "deepseek_int8"
        self.deepseek_engine = None
        self.native_request_timeout_s: float = 180.0
        self.requests_total: int = 0
        self.mode_counts: dict[str, int] = {"mixed": 0, "retrieval": 0, "deepseek_only": 0}
        self.hrm_query_calls_total: int = 0
        self.hrm_query_calls_by_mode: dict[str, int] = {"mixed": 0, "retrieval": 0, "deepseek_only": 0}
        self.prompt_build_failures: int = 0


STATE = _State()


def _build_prompt(prompt: str, mode: str) -> tuple[str, list[Any], str, bool]:
    resolved_mode = normalize_mode(mode)
    if resolved_mode == "deepseek_only":
        return build_prompt_for_mode(prompt, [], mode=resolved_mode), [], resolved_mode, False

    # Audit: explicit counter for every live HRM query invocation.
    STATE.hrm_query_calls_total += 1
    STATE.hrm_query_calls_by_mode[resolved_mode] = STATE.hrm_query_calls_by_mode.get(resolved_mode, 0) + 1

    # HRM query (prefer API if built, else subprocess)
    r = run_hrm_query(
        hrm_bin=STATE.hrm_bin,
        model_dir=STATE.hrm_model,
        prompt=prompt,
        top_k=16,
        top_m=400,
        k=16,
        prefer_api=True,
        repo_root=STATE.repo_root,
        timeout_s=1.8,
    )

    sources = build_sources(r.raw, max_sources=STATE.max_sources, max_chars_per_source=STATE.max_chars_per_source)
    if not sources:
        return "", [], resolved_mode, True

    q_for_prompt = prompt
    sources_for_prompt = sources

    if not STATE.disable_token_budget:
        tok_src = STATE.tokenizer_source
        if not tok_src:
            return "", [], resolved_mode, True
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(tok_src), local_files_only=bool(STATE.local_files_only))
        max_prompt_tokens = int(STATE.max_seq_len) - int(STATE.max_new_tokens) - int(STATE.reserve_prompt_tokens)
        q_fit, s_fit = fit_prompt_to_token_budget(
            q_for_prompt,
            sources_for_prompt,
            tok,
            max_prompt_tokens,
            mode=resolved_mode,
        )
        if not s_fit:
            return "", [], resolved_mode, True
        q_for_prompt, sources_for_prompt = q_fit, s_fit

    prompt_text = build_prompt_for_mode(q_for_prompt, sources_for_prompt, mode=resolved_mode)
    return prompt_text, sources_for_prompt, resolved_mode, True


def main():
    FastAPI, HTTPException, JSONResponse, BaseModel = _require_fastapi()

    ap = argparse.ArgumentParser(prog="hrm-flash-serve", description="HRM FlashRenderer HTTP service (HRM retrieval + native DeepSeek inference).")
    ap.add_argument("--hrm_model", required=True)
    ap.add_argument("--llm_model", required=True, help="Local HF safetensors model dir or HF repo ID")
    ap.add_argument("--world", type=int, choices=[1, 2, 3, 4], required=True)
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)

    ap.add_argument("--max_seq_len", type=int, default=8192)
    ap.add_argument("--prefill_chunk_size", type=int, default=1024)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--local_files_only", action="store_true")

    ap.add_argument("--max_sources", type=int, default=16)
    ap.add_argument("--max_chars_per_source", type=int, default=1200)
    ap.add_argument("--reserve_prompt_tokens", type=int, default=16)
    ap.add_argument("--disable_token_budget", action="store_true")
    ap.add_argument("--expose_sources", action="store_true", help="Debug only: include source texts in HTTP response")

    ap.add_argument("--hrm_bin", default=None)
    ap.add_argument("--max_concurrent", type=int, default=1)
    ap.add_argument("--backend", type=str, choices=["deepseek_int8"], default="deepseek_int8")
    ap.add_argument("--model_quant", type=str, choices=["q8", "q4"], default="q8", help="Native model quantization mode")
    ap.add_argument("--model_bin", type=str, default=None, help="Native DeepSeek model bin path (q8/q4)")
    ap.add_argument("--tokenizer_model", type=str, default=None, help="Tokenizer source for deepseek_int8 backend")
    ap.add_argument("--native_engine_bin", type=str, default=None, help="Path to deepseek_engine binary")
    ap.add_argument("--native_startup_timeout_s", type=float, default=120.0)
    ap.add_argument("--native_request_timeout_s", type=float, default=180.0)

    args = ap.parse_args()

    STATE.hrm_model = Path(args.hrm_model).resolve()
    if not (STATE.hrm_model / "router_index.bin").is_file() or not (STATE.hrm_model / "index.sqlite").is_file():
        raise SystemExit(
            f"ERR: HRM model directory must contain router_index.bin and index.sqlite: {STATE.hrm_model}"
        )
    STATE.backend = str(args.backend)
    STATE.world = int(args.world)
    STATE.max_seq_len = int(args.max_seq_len)
    STATE.prefill_chunk_size = int(args.prefill_chunk_size)
    STATE.max_new_tokens = int(args.max_new_tokens)
    STATE.local_files_only = bool(args.local_files_only)
    STATE.max_sources = int(args.max_sources)
    STATE.max_chars_per_source = int(args.max_chars_per_source)
    STATE.reserve_prompt_tokens = int(args.reserve_prompt_tokens)
    STATE.disable_token_budget = bool(args.disable_token_budget)
    STATE.expose_sources = bool(args.expose_sources)
    STATE.sem = asyncio.Semaphore(max(1, int(args.max_concurrent)))
    STATE.native_request_timeout_s = float(args.native_request_timeout_s)

    # Locate HRM binary (only needed if hrm_api not built)
    STATE.hrm_bin = find_hrm_binary(repo_root=STATE.repo_root, explicit=args.hrm_bin)

    from hrm_flash.deepseek_native import (
        DeepSeekNativeEngine,
        ensure_deepseek_model,
        resolve_deepseek_engine_bin,
    )

    try:
        model_bin, tok_src = ensure_deepseek_model(
            args.llm_model,
            model_bin=args.model_bin,
            model_quant=str(args.model_quant),
            local_files_only=bool(args.local_files_only),
            project_root=STATE.repo_root,
        )
    except RuntimeError as e:
        raise SystemExit(f"ERR: Failed to resolve/export deepseek model: {e}") from e
    tokenizer_source = str(args.tokenizer_model or tok_src)
    try:
        engine_bin = resolve_deepseek_engine_bin(STATE.repo_root, explicit=args.native_engine_bin)
    except RuntimeError as e:
        raise SystemExit(f"ERR: {e}") from e

    STATE.tokenizer_source = tokenizer_source
    STATE.deepseek_engine = DeepSeekNativeEngine(
        repo_root=STATE.repo_root,
        model_bin=model_bin,
        tokenizer_source=tokenizer_source,
        engine_bin=engine_bin,
        runtime_name=f"serve-{int(args.port)}",
        local_files_only=bool(args.local_files_only),
        max_new_tokens=int(args.max_new_tokens),
        startup_timeout_s=float(args.native_startup_timeout_s),
        request_timeout_s=float(args.native_request_timeout_s),
    )
    STATE.deepseek_engine.start()

    app = FastAPI(title="hrm-flash", version="5.1.0")

    class GenerateReq(BaseModel):
        prompt: str
        max_new_tokens: Optional[int] = None
        mode: Optional[str] = None
        show_sources: Optional[bool] = None

    @app.get("/v1/health")
    async def health():
        deepseek_running = False
        if STATE.deepseek_engine is not None:
            try:
                deepseek_running = bool(STATE.deepseek_engine.is_running())
            except Exception:
                deepseek_running = False
        return {
            "ok": bool(deepseek_running),
            "backend": STATE.backend,
            "deepseek_running": deepseek_running,
            "world": STATE.world,
            "audit": {
                "requests_total": int(STATE.requests_total),
                "mode_counts": dict(STATE.mode_counts),
                "hrm_query_calls_total": int(STATE.hrm_query_calls_total),
                "hrm_query_calls_by_mode": dict(STATE.hrm_query_calls_by_mode),
                "prompt_build_failures": int(STATE.prompt_build_failures),
            },
        }

    @app.post("/v1/generate")
    async def generate(req: GenerateReq):
        if not req.prompt:
            raise HTTPException(status_code=400, detail="prompt required")
        try:
            mode = normalize_mode(req.mode)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        async with STATE.sem:
            STATE.requests_total += 1
            STATE.mode_counts[mode] = STATE.mode_counts.get(mode, 0) + 1
            prompt_text, sources, mode, hrm_active = _build_prompt(req.prompt, mode=mode)
            prompt_template = "deepseek_only_raw"
            if mode == "mixed":
                prompt_template = "mixed_silent"
            elif mode == "retrieval":
                prompt_template = "retrieval_explicit"
            show_sources = bool(req.show_sources) if req.show_sources is not None else (mode == "retrieval")
            if not prompt_text:
                STATE.prompt_build_failures += 1
                payload = {
                    "ok": True,
                    "text": "I don't know.",
                    "source_count": 0,
                    "mode": mode,
                    "hrm_active": hrm_active,
                    "audit": {
                        "mode_resolved": mode,
                        "hrm_called": bool(hrm_active),
                        "source_injected_count": 0,
                        "prompt_template": prompt_template,
                    },
                }
                if show_sources or STATE.expose_sources:
                    payload["sources"] = []
                return JSONResponse(payload)
            max_new = int(req.max_new_tokens) if req.max_new_tokens is not None else STATE.max_new_tokens
            if STATE.deepseek_engine is None:
                raise HTTPException(status_code=500, detail="DeepSeek engine not initialized")
            try:
                text = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: STATE.deepseek_engine.generate(
                        prompt_text,
                        max_new_tokens=max_new,
                        timeout_s=STATE.native_request_timeout_s,
                    ),
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
            payload = {
                "ok": True,
                "text": text,
                "source_count": len(sources),
                "mode": mode,
                "hrm_active": hrm_active,
                "audit": {
                    "mode_resolved": mode,
                    "hrm_called": bool(hrm_active),
                    "source_injected_count": int(len(sources)),
                    "prompt_template": prompt_template,
                },
            }
            if (show_sources or STATE.expose_sources) and mode != "deepseek_only":
                payload["sources"] = [{"sid": s.sid, "txt": s.txt} for s in sources]
            return JSONResponse(payload)

    @app.on_event("shutdown")
    def _shutdown():
        if STATE.deepseek_engine is not None:
            try:
                STATE.deepseek_engine.stop()
            except Exception:
                pass

    import uvicorn
    uvicorn.run(app, host=str(args.host), port=int(args.port), log_level="info")


if __name__ == "__main__":
    main()

