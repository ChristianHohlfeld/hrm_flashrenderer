# Copyright (c) 2026 Christian Heinrich Hohlfeld (Konstanz, Germany)
# https://christianhohlfeld.com | ORCID: https://orcid.org/0009-0003-6634-9045
# ALL RIGHTS RESERVED. No license granted without prior written permission.
import argparse
import sys
from pathlib import Path

from hrm_flash.hrm_client import run_hrm_query
from hrm_flash.prompt_builder import build_sources, build_renderer_prompt, fit_prompt_to_token_budget
from hrm_flash.flash_runner import run_flash_generate
from hrm_flash.flash_daemon_client import parse_daemon_addr, generate as daemon_generate
from hrm_flash.utils import find_hrm_binary, validate_tp_world


def _ensure_llm_model(model_str: str, local_files_only: bool = False) -> Path:
    """Ensure model_str points to a local directory. If not, try HF download."""
    p = Path(model_str)
    if p.is_dir():
        return p.resolve()

    # Not a local dir, try HF Hub download
    try:
        from huggingface_hub import snapshot_download
        print(f"[*] Model '{model_str}' not found locally. Attempting HF Hub download...")
        # We download config + safetensors.
        path = snapshot_download(
            repo_id=model_str,
            local_files_only=local_files_only,
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"]
        )
        return Path(path).resolve()
    except Exception as e:
        if local_files_only:
            raise SystemExit(f"ERR: Model '{model_str}' not found locally and --local_files_only is set.")
        raise SystemExit(f"ERR: Failed to resolve/download model '{model_str}': {e}")


def main():
    ap = argparse.ArgumentParser(prog="hrm-flash", description="HRM-FlashRenderer v5.1.0 (HRM retrieval + persistent FlashAttention TP renderer)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("daemon", help="Start persistent TP FlashAttention daemon (loads renderer once)")
    d.add_argument("--model", required=True, help="Local HF safetensors model dir")
    d.add_argument("--world", type=int, choices=[1, 2, 3, 4], required=True)
    d.add_argument("--max_seq_len", type=int, default=8192)
    d.add_argument("--prefill_chunk_size", type=int, default=1024)
    d.add_argument("--local_files_only", action="store_true")
    d.add_argument("--host", type=str, default="127.0.0.1")
    d.add_argument("--port", type=int, default=0)
    d.add_argument("--authkey", type=str, default="hrmflash")
    d.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                   help="Device to run on. Use 'cpu' for CPU-only mode (no CUDA required).")

    s = sub.add_parser("serve", help="Start HTTP service (HRM retrieval + persistent TP daemon)")
    s.add_argument("--hrm_model", required=True)
    s.add_argument("--llm_model", required=True)
    s.add_argument("--world", type=int, choices=[2, 3, 4], required=True)
    s.add_argument("--host", type=str, default="0.0.0.0")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--max_seq_len", type=int, default=8192)
    s.add_argument("--prefill_chunk_size", type=int, default=1024)
    s.add_argument("--max_new_tokens", type=int, default=256)
    s.add_argument("--local_files_only", action="store_true")
    s.add_argument("--max_sources", type=int, default=8)
    s.add_argument("--max_chars_per_source", type=int, default=1200)
    s.add_argument("--reserve_prompt_tokens", type=int, default=16)
    s.add_argument("--disable_token_budget", action="store_true")
    s.add_argument("--hrm_bin", default=None)
    s.add_argument("--max_concurrent", type=int, default=1)

    g = sub.add_parser("generate", help="Retrieve with HRM, then render with FlashAttention TP engine")
    g.add_argument("--hrm_model", required=True, help="HRM model dir (router_index.bin + index.sqlite)")
    g.add_argument("--llm_model", required=True, help="Local HF safetensors model dir (renderer, e.g. Llama-2/3 7B)")
    g.add_argument("--prompt", required=True, help="User question")

    g.add_argument("--hrm_bin", default=None, help="Path to HRM binary. If omitted: uses HRM_BIN env, PATH, or repo build.")

    g.add_argument("--world", type=int, choices=[1, 2, 3, 4], default=None, help="Tensor parallel world size")

    g.add_argument("--top_k", type=int, default=8)
    g.add_argument("--top_m", type=int, default=400)
    g.add_argument("--k", type=int, default=8)

    g.add_argument("--max_sources", type=int, default=8)
    g.add_argument("--max_chars_per_source", type=int, default=1200)

    g.add_argument("--reserve_prompt_tokens", type=int, default=16, help="Safety margin for prompt budget")
    g.add_argument("--disable_token_budget", action="store_true", help="Disable tokenizer-based prompt budgeting")

    g.add_argument("--max_new_tokens", type=int, default=256)
    g.add_argument("--max_seq_len", type=int, default=8192)
    g.add_argument("--prefill_chunk_size", type=int, default=1024)
    g.add_argument("--local_files_only", action="store_true")

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
        ] + (["--local_files_only"] if args.local_files_only else []) + (["--disable_token_budget"] if args.disable_token_budget else []) + (["--hrm_bin", args.hrm_bin] if args.hrm_bin else [])
        smain()
        return

    if args.cmd == "generate":
        repo_root = Path(__file__).resolve().parents[1]
        hrm_bin = find_hrm_binary(repo_root=repo_root, explicit=args.hrm_bin)
        hrm_model = Path(args.hrm_model).resolve()

        # Fail fast: validate HRM model directory early
        if not (hrm_model / "router_index.bin").is_file() or not (hrm_model / "index.sqlite").is_file():
            raise SystemExit(f"ERR: HRM model directory must contain router_index.bin and index.sqlite: {hrm_model}")

        # Fail fast: check CUDA if not explicitly using CPU
        if args.device != "cpu":
            try:
                import torch
                if not torch.cuda.is_available():
                    print("WARN: CUDA not available. Results might be slow or fail if GPU is required.")
                elif torch.cuda.device_count() == 0:
                    print("WARN: torch reports 0 CUDA devices.")
            except ImportError:
                raise SystemExit("ERR: torch is not installed. Required for LLM operations.")
            except Exception as e:
                print(f"WARN: Error checking CUDA: {e}")

        llm_model = _ensure_llm_model(args.llm_model, local_files_only=bool(args.local_files_only))

        # Validate TP compatibility early (production behavior: fail fast with clear error)
        if args.world is not None:
            try:
                from transformers import AutoConfig
            except ImportError as e:
                raise SystemExit(f"ERR: transformers is required for --world validation: {e}")
            except Exception as e:
                raise SystemExit(f"ERR: Failed to initialize transformers: {e}")
            cfg = AutoConfig.from_pretrained(str(llm_model), local_files_only=bool(args.local_files_only))
            validate_tp_world(cfg, int(args.world))

        r = run_hrm_query(
            hrm_bin=hrm_bin,
            model_dir=hrm_model,
            prompt=args.prompt,
            top_k=args.top_k,
            top_m=args.top_m,
            k=args.k,
            prefer_api=True,
            repo_root=repo_root,
        )
        sources = build_sources(r.raw, max_sources=args.max_sources, max_chars_per_source=args.max_chars_per_source)

        # Hard, deterministic no-sources behavior (prevents hallucinations).
        # If --print_prompt is set, we still print the prompt (with empty SOURCES).
        if not sources and not args.print_prompt:
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
            tok = AutoTokenizer.from_pretrained(str(llm_model), local_files_only=bool(args.local_files_only))
            max_prompt_tokens = int(args.max_seq_len) - int(args.max_new_tokens) - int(args.reserve_prompt_tokens)
            if max_prompt_tokens <= 64:
                raise SystemExit("ERR: max_seq_len too small for given max_new_tokens")
            q_fit, s_fit = fit_prompt_to_token_budget(q_for_prompt, sources_for_prompt, tok, max_prompt_tokens)
            if not s_fit:
                print("I don't know. (Prompt too large even after budgeting.)")
                return
            q_for_prompt, sources_for_prompt = q_fit, s_fit

        prompt_text = build_renderer_prompt(q_for_prompt, sources_for_prompt)

        if args.print_prompt:
            print(prompt_text)
            return

        if args.use_daemon:
            addr = parse_daemon_addr()
            if addr is None:
                raise SystemExit("ERR: --use_daemon set but HRM_FLASH_DAEMON is not configured (host:port)")
            text = daemon_generate(addr, prompt_text, args.max_new_tokens, args.prefill_chunk_size)
            print(text)
            return

        code = run_flash_generate(
            repo_root=repo_root,
            llm_model_dir=llm_model,
            prompt_text=prompt_text,
            world=args.world,
            max_new_tokens=args.max_new_tokens,
            max_seq_len=args.max_seq_len,
            prefill_chunk_size=args.prefill_chunk_size,
            local_files_only=bool(args.local_files_only),
            device=args.device,
        )
        raise SystemExit(code)


if __name__ == "__main__":
    main()
