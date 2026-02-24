# Copyright (c) 2026 Christian Heinrich Hohlfeld (Konstanz, Germany)
# https://christianhohlfeld.com | ORCID: https://orcid.org/0009-0003-6634-9045
# ALL RIGHTS RESERVED. No license granted without prior written permission.
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from hrm_flash.hrm_client import run_hrm_query
from hrm_flash.prompt_builder import build_sources, build_renderer_prompt, fit_prompt_to_token_budget
from hrm_flash.utils import find_hrm_binary, validate_tp_world
from hrm_flash.flash_daemon_client import FlashDaemonAddr, parse_daemon_addr, generate as daemon_generate


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
        self.llm_model: Path | None = None
        self.world: int | None = None
        self.max_seq_len: int = 8192
        self.prefill_chunk_size: int = 1024
        self.max_new_tokens: int = 256
        self.local_files_only: bool = True
        self.max_sources: int = 8
        self.max_chars_per_source: int = 1200
        self.reserve_prompt_tokens: int = 16
        self.disable_token_budget: bool = False
        self.daemon_addr: FlashDaemonAddr | None = None
        self.daemon_proc: subprocess.Popen | None = None
        self.sem: asyncio.Semaphore | None = None
        self.device: str = 'cuda'


STATE = _State()


def _start_daemon_if_needed() -> FlashDaemonAddr:
    addr = parse_daemon_addr()
    if addr is not None:
        return addr

    # Start our own daemon process
    # Choose port and publish via env so client can read it.
    import socket
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    os.environ["HRM_FLASH_DAEMON"] = f"127.0.0.1:{port}"
    os.environ["HRM_FLASH_AUTHKEY"] = os.environ.get("HRM_FLASH_AUTHKEY", "hrmflash")
    authkey = os.environ["HRM_FLASH_AUTHKEY"].encode("utf-8")
    addr = FlashDaemonAddr(host="127.0.0.1", port=port, authkey=authkey)

    cmd = [
        sys.executable,
        "-m",
        "hrm_flash.flash_daemon",
        "--model",
        str(STATE.llm_model),
        "--world",
        str(int(STATE.world)),
        "--max_seq_len",
        str(int(STATE.max_seq_len)),
        "--prefill_chunk_size",
        str(int(STATE.prefill_chunk_size)),
        "--host",
        addr.host,
        "--port",
        str(addr.port),
        "--authkey",
        os.environ["HRM_FLASH_AUTHKEY"],
        "--device",
        STATE.device,
    ]
    if STATE.local_files_only:
        cmd.append("--local_files_only")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(STATE.repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    p = subprocess.Popen(cmd, env=env, cwd=str(STATE.repo_root), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    STATE.daemon_proc = p

    # wait for READY line or timeout
    t0 = time.time()
    ready = False
    while time.time() - t0 < 30.0:
        line = p.stderr.readline()
        if not line:
            time.sleep(0.05)
            continue
        if "READY" in line:
            ready = True
            break
    if not ready:
        try:
            err = p.stderr.read()
        except Exception:
            err = ""
        raise SystemExit(f"ERR: flash daemon failed to start.\n{err}")

    return addr


def _build_prompt(prompt: str) -> tuple[str, list[dict[str, Any]]]:
    # HRM query (prefer API if built, else subprocess)
    r = run_hrm_query(
        hrm_bin=STATE.hrm_bin,
        model_dir=STATE.hrm_model,
        prompt=prompt,
        top_k=8,
        top_m=400,
        k=8,
        prefer_api=True,
        repo_root=STATE.repo_root,
    )

    sources = build_sources(r.raw, max_sources=STATE.max_sources, max_chars_per_source=STATE.max_chars_per_source)
    if not sources:
        return "", []

    q_for_prompt = prompt
    sources_for_prompt = sources

    if not STATE.disable_token_budget:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(STATE.llm_model), local_files_only=bool(STATE.local_files_only))
        max_prompt_tokens = int(STATE.max_seq_len) - int(STATE.max_new_tokens) - int(STATE.reserve_prompt_tokens)
        q_fit, s_fit = fit_prompt_to_token_budget(q_for_prompt, sources_for_prompt, tok, max_prompt_tokens)
        if not s_fit:
            return "", []
        q_for_prompt, sources_for_prompt = q_fit, s_fit

    prompt_text = build_renderer_prompt(q_for_prompt, sources_for_prompt)
    return prompt_text, sources_for_prompt


def main():
    FastAPI, HTTPException, JSONResponse, BaseModel = _require_fastapi()

    ap = argparse.ArgumentParser(prog="hrm-flash-serve", description="HRM FlashRenderer HTTP service (HRM retrieval + persistent FlashAttention TP daemon).")
    ap.add_argument("--hrm_model", required=True)
    ap.add_argument("--llm_model", required=True)
    ap.add_argument("--world", type=int, choices=[2, 3, 4], required=True)
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)

    ap.add_argument("--max_seq_len", type=int, default=8192)
    ap.add_argument("--prefill_chunk_size", type=int, default=1024)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--local_files_only", action="store_true")

    ap.add_argument("--max_sources", type=int, default=8)
    ap.add_argument("--max_chars_per_source", type=int, default=1200)
    ap.add_argument("--reserve_prompt_tokens", type=int, default=16)
    ap.add_argument("--disable_token_budget", action="store_true")

    ap.add_argument("--hrm_bin", default=None)
    ap.add_argument("--max_concurrent", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                    help="Device to run on. Use 'cpu' for CPU-only mode (no CUDA required).")

    args = ap.parse_args()

    STATE.hrm_model = Path(args.hrm_model).resolve()
    STATE.llm_model = Path(args.llm_model).resolve()
    STATE.world = int(args.world)
    STATE.max_seq_len = int(args.max_seq_len)
    STATE.prefill_chunk_size = int(args.prefill_chunk_size)
    STATE.max_new_tokens = int(args.max_new_tokens)
    STATE.local_files_only = bool(args.local_files_only)
    STATE.max_sources = int(args.max_sources)
    STATE.max_chars_per_source = int(args.max_chars_per_source)
    STATE.reserve_prompt_tokens = int(args.reserve_prompt_tokens)
    STATE.disable_token_budget = bool(args.disable_token_budget)
    STATE.sem = asyncio.Semaphore(max(1, int(args.max_concurrent)))
    STATE.device = str(args.device)

    # Locate HRM binary (only needed if hrm_api not built)
    STATE.hrm_bin = find_hrm_binary(repo_root=STATE.repo_root, explicit=args.hrm_bin)

    # Validate TP world up-front
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(str(STATE.llm_model), local_files_only=bool(STATE.local_files_only))
    validate_tp_world(cfg, int(STATE.world))

    # Ensure daemon running
    STATE.daemon_addr = _start_daemon_if_needed()

    app = FastAPI(title="hrm-flash", version="5.1.0")

    class GenerateReq(BaseModel):
        prompt: str
        max_new_tokens: Optional[int] = None

    @app.get("/v1/health")
    async def health():
        return {"ok": True, "daemon": os.environ.get("HRM_FLASH_DAEMON"), "world": STATE.world}

    @app.post("/v1/generate")
    async def generate(req: GenerateReq):
        if not req.prompt:
            raise HTTPException(status_code=400, detail="prompt required")
        async with STATE.sem:
            prompt_text, sources = _build_prompt(req.prompt)
            if not prompt_text:
                return JSONResponse({"ok": True, "text": "I don't know.", "sources": []})
            max_new = int(req.max_new_tokens) if req.max_new_tokens is not None else STATE.max_new_tokens
            try:
                text = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: daemon_generate(STATE.daemon_addr, prompt_text, max_new, STATE.prefill_chunk_size),
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
            return JSONResponse({"ok": True, "text": text, "sources": sources})

    @app.on_event("shutdown")
    def _shutdown():
        # Best-effort: stop daemon we started
        p = STATE.daemon_proc
        if p is not None:
            try:
                p.terminate()
            except Exception:
                pass

    import uvicorn
    uvicorn.run(app, host=str(args.host), port=int(args.port), log_level="info")


if __name__ == "__main__":
    main()

