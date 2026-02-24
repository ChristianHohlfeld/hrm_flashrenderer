# Copyright (c) 2026 Christian Heinrich Hohlfeld (Konstanz, Germany)
# https://christianhohlfeld.com | ORCID: https://orcid.org/0009-0003-6634-9045
# ALL RIGHTS RESERVED. No license granted without prior written permission.
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def run_flash_generate(
    repo_root: Path,
    llm_model_dir: Path,
    prompt_text: str,
    world: int | None,
    max_new_tokens: int,
    max_seq_len: int,
    prefill_chunk_size: int,
    local_files_only: bool,
    device: str = "cuda",
) -> int:
    cmd = [
        sys.executable,
        "-m",
        "engine.hf_tp_generate",
        "--model",
        str(llm_model_dir),
        "--prompt_stdin",
        "--max_new_tokens",
        str(int(max_new_tokens)),
        "--max_seq_len",
        str(int(max_seq_len)),
        "--prefill_chunk_size",
        str(int(prefill_chunk_size)),
    ]
    if local_files_only:
        cmd.append("--local_files_only")
    if world is not None:
        cmd += ["--world", str(int(world))]
    cmd += ["--device", device]

    env = os.environ.copy()
    # If running from a repo checkout (not installed), ensure repo root on PYTHONPATH.
    # If installed as a package, this is harmless.
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")

    # Concurrency-safe run directory for optional debugging.
    # We do NOT write the prompt to disk by default.
    run_root = repo_root / ".run"
    try:
        run_root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        tmp_dir = tempfile.mkdtemp(prefix=f"run-{stamp}-", dir=str(run_root))
        env["HRM_FLASH_RUN_DIR"] = tmp_dir
    except Exception:
        # best-effort; never fail execution due to logging folder
        pass

    p = subprocess.Popen(
        cmd,
        env=env,
        cwd=str(repo_root),
        stdin=subprocess.PIPE,
    )
    try:
        p.communicate(prompt_text.encode("utf-8"), timeout=None)
    except Exception:
        p.kill()
        p.wait()
        raise
    return int(p.returncode)

