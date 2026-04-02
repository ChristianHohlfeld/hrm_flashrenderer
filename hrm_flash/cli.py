# Copyright (c) 2026 Christian Heinrich Hohlfeld (Konstanz, Germany)
# https://christianhohlfeld.com | ORCID: https://orcid.org/0009-0003-6634-9045
# ALL RIGHTS RESERVED. No license granted without prior written permission.
import argparse
import os
import sys
from pathlib import Path

from hrm_flash.hrm_client import run_hrm_query
from hrm_flash.prompt_builder import build_prompt_for_mode, build_sources, fit_prompt_to_token_budget, normalize_mode
from hrm_flash.utils import (
    find_hrm_binary,
)


def main():
    ap = argparse.ArgumentParser(prog="hrm-flash", description="HRM-FlashRenderer v5.1.0 (HRM retrieval + native DeepSeek INT8 inference)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("daemon", help="Start persistent TP FlashAttention daemon (loads renderer once)")
    d.add_argument("--model", required=True, help="Local HF safetensors model dir or HF repo ID")
    d.add_argument("--world", type=int, choices=[1, 2, 3, 4], required=True)
    d.add_argument("--max_seq_len", type=int, default=8192)
    d.add_argument("--prefill_chunk_size", type=int, default=1024)
    d.add_argument("--local_files_only", action="store_true")
    d.add_argument("--host", type=str, default="127.0.0.1")
    d.add_argument("--port", type=int, default=0)
    d.add_argument("--authkey", type=str, default="hrmflash")
    d.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                   help="Device to run on. Use 'cpu' for CPU-only mode (no CUDA required).")

    s = sub.add_parser("serve", help="Start HTTP service (HRM retrieval + native DeepSeek INT8/INT4 engine)")
    s.add_argument("--hrm_model", required=True)
    s.add_argument("--llm_model", required=True, help="Local HF safetensors model dir or HF repo ID")
    s.add_argument("--world", type=int, choices=[1, 2, 3, 4], required=True)
    s.add_argument("--host", type=str, default="0.0.0.0")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--max_seq_len", type=int, default=8192)
    s.add_argument("--prefill_chunk_size", type=int, default=1024)
    s.add_argument("--max_new_tokens", type=int, default=256)
    s.add_argument("--local_files_only", action="store_true")
    s.add_argument("--max_sources", type=int, default=16)
    s.add_argument("--max_chars_per_source", type=int, default=1200)
    s.add_argument("--reserve_prompt_tokens", type=int, default=16)
    s.add_argument("--disable_token_budget", action="store_true")
    s.add_argument("--expose_sources", action="store_true", help="Debug only: include source texts in HTTP response")
    s.add_argument("--hrm_bin", default=None)
    s.add_argument("--max_concurrent", type=int, default=1)
    s.add_argument("--backend", type=str, choices=["deepseek_int8"], default="deepseek_int8")
    s.add_argument("--model_quant", type=str, choices=["q8", "q4"], default="q8", help="Native model quantization mode")
    s.add_argument("--model_bin", type=str, default=None, help="Native DeepSeek model bin path (q8/q4)")
    s.add_argument("--tokenizer_model", type=str, default=None, help="Tokenizer source for deepseek_int8 backend")
    s.add_argument("--native_engine_bin", type=str, default=None, help="Path to deepseek_engine binary")
    s.add_argument("--native_startup_timeout_s", type=float, default=120.0)
    s.add_argument("--native_request_timeout_s", type=float, default=180.0)

    rt = sub.add_parser("router", help="Start heterogenous multi-endpoint router for native hrm-flash services")
    rt.add_argument("--host", type=str, default="0.0.0.0")
    rt.add_argument("--port", type=int, default=8090)
    rt.add_argument("--endpoint_solo_22gb", type=str, default="http://127.0.0.1:8081")
    rt.add_argument("--endpoint_nvlink_pair", type=str, default="http://127.0.0.1:8082")
    rt.add_argument("--endpoint_solo_3080", type=str, default="http://127.0.0.1:8083")
    rt.add_argument("--short_prompt_tokens", type=int, default=256)
    rt.add_argument("--medium_prompt_tokens", type=int, default=1200)
    rt.add_argument("--short_max_new_tokens", type=int, default=192)
    rt.add_argument("--medium_max_new_tokens", type=int, default=384)
    rt.add_argument("--long_max_new_tokens", type=int, default=768)
    rt.add_argument("--request_timeout_s", type=float, default=180.0)
    rt.add_argument("--health_timeout_s", type=float, default=1.5)
    rt.add_argument("--max_concurrent", type=int, default=8)
    rt.add_argument("--tokenizer_model", type=str, default=None, help="Optional tokenizer source for accurate prompt token estimation")
    rt.add_argument("--chars_per_token", type=float, default=4.0, help="Fallback token estimate when tokenizer is unavailable")
    rt.add_argument("--local_files_only", action="store_true")
    rt.add_argument("--disable_tokenizer", action="store_true")

    g = sub.add_parser("generate", help="Retrieve with HRM, then answer with native DeepSeek INT8/INT4 engine")
    g.add_argument("--hrm_model", required=True, help="HRM model dir (router_index.bin + index.sqlite)")
    g.add_argument("--llm_model", required=True, help="Local HF safetensors model dir or HF repo ID (renderer)")
    g.add_argument("--prompt", required=True, help="User question")
    g.add_argument("--mode", type=str, choices=["retrieval", "mixed", "deepseek_only"], default="mixed")

    g.add_argument("--hrm_bin", default=None, help="Path to HRM binary. If omitted: uses HRM_BIN env, PATH, or repo build.")

    g.add_argument("--world", type=int, choices=[1, 2, 3, 4], default=None, help="Tensor parallel world size")

    g.add_argument("--top_k", type=int, default=16)
    g.add_argument("--top_m", type=int, default=400)
    g.add_argument("--k", type=int, default=16)

    g.add_argument("--max_sources", type=int, default=16)
    g.add_argument("--max_chars_per_source", type=int, default=1200)

    g.add_argument("--reserve_prompt_tokens", type=int, default=16, help="Safety margin for prompt budget")
    g.add_argument("--disable_token_budget", action="store_true", help="Disable tokenizer-based prompt budgeting")

    g.add_argument("--max_new_tokens", type=int, default=256)
    g.add_argument("--max_seq_len", type=int, default=8192)
    g.add_argument("--prefill_chunk_size", type=int, default=1024)
    g.add_argument("--local_files_only", action="store_true")
    g.add_argument("--backend", type=str, choices=["deepseek_int8"], default="deepseek_int8")
    g.add_argument("--model_quant", type=str, choices=["q8", "q4"], default="q8", help="Native model quantization mode")
    g.add_argument("--model_bin", type=str, default=None, help="Native DeepSeek model bin path (q8/q4)")
    g.add_argument("--tokenizer_model", type=str, default=None, help="Tokenizer source for deepseek_int8 backend")
    g.add_argument("--native_engine_bin", type=str, default=None, help="Path to deepseek_engine binary")
    g.add_argument("--native_startup_timeout_s", type=float, default=120.0)
    g.add_argument("--native_request_timeout_s", type=float, default=180.0)

    g.add_argument("--print_prompt", action="store_true", help="Print the built renderer prompt and exit")
    g.add_argument("--use_daemon", action="store_true", help="Send to HRM_FLASH_DAEMON instead of spawning renderer")
    g.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                   help="Device to run on. Use 'cpu' for CPU-only mode (no CUDA required).")

    args = ap.parse_args()

    if args.cmd == "daemon":
        from hrm_flash.flash_daemon import main as dmain
        sys.argv = ["hrm-flash daemon"] + [
            "--model", args.model,
            "--world", str(args.world),
            "--max_seq_len", str(args.max_seq_len),
            "--prefill_chunk_size", str(args.prefill_chunk_size),
            "--host", args.host,
            "--port", str(args.port),
            "--authkey", args.authkey,
        ] + (["--local_files_only"] if args.local_files_only else [])
        if hasattr(args, 'device'):
            sys.argv.append("--device")
            sys.argv.append(args.device)
        dmain()
        return

    if args.cmd == "serve":
        from hrm_flash.serve import main as smain
        sys.argv = ["hrm-flash serve"] + [
            "--hrm_model", args.hrm_model,
            "--llm_model", args.llm_model,
            "--world", str(args.world),
            "--host", args.host,
            "--port", str(args.port),
            "--max_seq_len", str(args.max_seq_len),
            "--prefill_chunk_size", str(args.prefill_chunk_size),
            "--max_new_tokens", str(args.max_new_tokens),
            "--max_sources", str(args.max_sources),
            "--max_chars_per_source", str(args.max_chars_per_source),
            "--reserve_prompt_tokens", str(args.reserve_prompt_tokens),
            "--max_concurrent", str(args.max_concurrent),
            "--backend", str(args.backend),
            "--model_quant", str(args.model_quant),
            "--native_startup_timeout_s", str(args.native_startup_timeout_s),
            "--native_request_timeout_s", str(args.native_request_timeout_s),
        ] + (["--local_files_only"] if args.local_files_only else []) + (["--disable_token_budget"] if args.disable_token_budget else []) + (["--hrm_bin", args.hrm_bin] if args.hrm_bin else [])
        if args.expose_sources:
            sys.argv += ["--expose_sources"]
        if args.model_bin:
            sys.argv += ["--model_bin", args.model_bin]
        if args.tokenizer_model:
            sys.argv += ["--tokenizer_model", args.tokenizer_model]
        if args.native_engine_bin:
            sys.argv += ["--native_engine_bin", args.native_engine_bin]
        smain()
        return

    if args.cmd == "router":
        from hrm_flash.router import main as rmain
        sys.argv = ["hrm-flash router"] + [
            "--host", args.host,
            "--port", str(args.port),
            "--endpoint_solo_22gb", args.endpoint_solo_22gb,
            "--endpoint_nvlink_pair", args.endpoint_nvlink_pair,
            "--endpoint_solo_3080", args.endpoint_solo_3080,
            "--short_prompt_tokens", str(args.short_prompt_tokens),
            "--medium_prompt_tokens", str(args.medium_prompt_tokens),
            "--short_max_new_tokens", str(args.short_max_new_tokens),
            "--medium_max_new_tokens", str(args.medium_max_new_tokens),
            "--long_max_new_tokens", str(args.long_max_new_tokens),
            "--request_timeout_s", str(args.request_timeout_s),
            "--health_timeout_s", str(args.health_timeout_s),
            "--max_concurrent", str(args.max_concurrent),
            "--chars_per_token", str(args.chars_per_token),
        ] + (["--tokenizer_model", args.tokenizer_model] if args.tokenizer_model else []) + (["--local_files_only"] if args.local_files_only else []) + (["--disable_tokenizer"] if args.disable_tokenizer else [])
        rmain()
        return

    if args.cmd == "generate":
        repo_root = Path(__file__).resolve().parents[1]
        hrm_bin = find_hrm_binary(repo_root=repo_root, explicit=args.hrm_bin)
        hrm_model = Path(args.hrm_model).resolve()

        # Fail fast: validate HRM model directory early
        if not (hrm_model / "router_index.bin").is_file() or not (hrm_model / "index.sqlite").is_file():
            raise SystemExit(f"ERR: HRM model directory must contain router_index.bin and index.sqlite: {hrm_model}")

        tokenizer_model_for_budget = None
        deepseek_ctx = None

        from hrm_flash.deepseek_native import ensure_deepseek_model, resolve_deepseek_engine_bin

        try:
            model_bin, tok_src = ensure_deepseek_model(
                args.llm_model,
                model_bin=args.model_bin,
                model_quant=str(args.model_quant),
                local_files_only=bool(args.local_files_only),
                project_root=repo_root,
            )
        except RuntimeError as e:
            raise SystemExit(f"ERR: Failed to resolve/export deepseek model: {e}") from e
        tokenizer_model_for_budget = str(args.tokenizer_model or tok_src)
        try:
            engine_bin = resolve_deepseek_engine_bin(repo_root, explicit=args.native_engine_bin)
        except RuntimeError as e:
            raise SystemExit(f"ERR: {e}") from e
        deepseek_ctx = {
            "model_bin": model_bin,
            "tokenizer_source": tokenizer_model_for_budget,
            "engine_bin": engine_bin,
        }

        mode = normalize_mode(args.mode)
        if mode == "deepseek_only":
            sources = []
        else:
            r = run_hrm_query(
                hrm_bin=hrm_bin,
                model_dir=hrm_model,
                prompt=args.prompt,
                top_k=args.top_k,
                top_m=args.top_m,
                k=args.k,
                prefer_api=True,
                repo_root=repo_root,
                timeout_s=1.8,
            )
            sources = build_sources(r.raw, max_sources=args.max_sources, max_chars_per_source=args.max_chars_per_source)

        # Hard, deterministic no-sources behavior (prevents hallucinations).
        # If --print_prompt is set, we still print the prompt (with empty SOURCES).
        if (mode != "deepseek_only") and (not sources) and (not args.print_prompt):
            print("I don't know. (HRM returned no sources.)")
            return

        q_for_prompt = args.prompt
        sources_for_prompt = sources

        # Token-budget the prompt (deterministic shrink) so we never exceed max_seq_len.
        # If sources are empty, budgeting is unnecessary and we'd still want --print_prompt to work.
        if (not args.disable_token_budget) and sources_for_prompt:
            try:
                from transformers import AutoTokenizer
            except ImportError as e:
                raise SystemExit(f"ERR: transformers is required for token budgeting: {e}")
            except Exception as e:
                raise SystemExit(f"ERR: Failed to initialize tokenizer: {e}")
            tok = AutoTokenizer.from_pretrained(str(tokenizer_model_for_budget), local_files_only=bool(args.local_files_only))
            max_prompt_tokens = int(args.max_seq_len) - int(args.max_new_tokens) - int(args.reserve_prompt_tokens)
            if max_prompt_tokens <= 64:
                raise SystemExit("ERR: max_seq_len too small for given max_new_tokens")
            q_fit, s_fit = fit_prompt_to_token_budget(
                q_for_prompt,
                sources_for_prompt,
                tok,
                max_prompt_tokens,
                mode=mode,
            )
            if not s_fit:
                print("I don't know. (Prompt too large even after budgeting.)")
                return
            q_for_prompt, sources_for_prompt = q_fit, s_fit

        prompt_text = build_prompt_for_mode(q_for_prompt, sources_for_prompt, mode=mode)

        if args.print_prompt:
            print(prompt_text)
            return

        if args.use_daemon:
            raise SystemExit("ERR: --use_daemon is not supported in deepseek-only production mainline.")

        from hrm_flash.deepseek_native import DeepSeekNativeEngine

        assert deepseek_ctx is not None
        runner = DeepSeekNativeEngine(
            repo_root=repo_root,
            model_bin=deepseek_ctx["model_bin"],
            tokenizer_source=deepseek_ctx["tokenizer_source"],
            engine_bin=deepseek_ctx["engine_bin"],
            runtime_name=f"generate-{os.getpid()}",
            local_files_only=bool(args.local_files_only),
            max_new_tokens=max(1, int(args.max_new_tokens)),
            startup_timeout_s=float(args.native_startup_timeout_s),
            request_timeout_s=float(args.native_request_timeout_s),
        )
        try:
            text = runner.generate(
                prompt_text,
                max_new_tokens=int(args.max_new_tokens),
                timeout_s=float(args.native_request_timeout_s),
            )
        finally:
            runner.stop()
        print(text)
        return


if __name__ == "__main__":
    main()
