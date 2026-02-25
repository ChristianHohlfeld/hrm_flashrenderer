# Copyright (c) 2026 Christian Heinrich Hohlfeld (Konstanz, Germany)
# https://christianhohlfeld.com | ORCID: https://orcid.org/0009-0003-6634-9045
# ALL RIGHTS RESERVED. No license granted without prior written permission.
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from hrm_flash.utils import find_hrm_binary


@dataclass(frozen=True)
class HRMQueryResult:
    raw: Dict[str, Any]


def run_hrm_query_via_api(repo_root: Path | None, model_dir: Path, prompt: str, top_k: int, top_m: int, k: int) -> HRMQueryResult | None:
    if not (model_dir / "router_index.bin").is_file() or not (model_dir / "index.sqlite").is_file():
        return None

    try:
        from hrm_flash.hrm_api import HRMHandle
    except Exception:
        return None

    try:
        h = HRMHandle(model_dir=model_dir, repo_root=repo_root)
    except Exception:
        return None

    try:
        raw = json.loads(h.query_json(prompt, top_k=top_k, top_m=top_m, k=k))
        return HRMQueryResult(raw=raw)
    finally:
        h.close()


def run_hrm_query(
    hrm_bin: Path | None,
    model_dir: Path,
    prompt: str,
    top_k: int = 8,
    top_m: int = 400,
    k: int = 8,
    prefer_api: bool = True,
    repo_root: Path | None = None,
) -> HRMQueryResult:
    if prefer_api:
        r = run_hrm_query_via_api(repo_root=repo_root, model_dir=model_dir, prompt=prompt, top_k=top_k, top_m=top_m, k=k)
        if r is not None:
            return r
    hrm_bin = find_hrm_binary(explicit=str(hrm_bin) if hrm_bin is not None else None)
    if not hrm_bin.is_file():
        raise FileNotFoundError(
            "HRM binary not found. Provide --hrm_bin, or set HRM_BIN, or ensure `hrm` is on PATH.\n"
            "Build in a repo checkout with:\n"
            "  cmake -S hrm_core -B hrm_core/build -DCMAKE_BUILD_TYPE=Release\n"
            "  cmake --build hrm_core/build -j\n"
            f"Tried: {hrm_bin}"
        )
    if not (model_dir / "router_index.bin").is_file() or not (model_dir / "index.sqlite").is_file():
        raise FileNotFoundError(f"HRM model dir must contain router_index.bin and index.sqlite: {model_dir}")

    cmd = [
        str(hrm_bin),
        "query",
        "--model", str(model_dir),
        "--format", "json",
        "--top-k", str(int(top_k)),
        "--top-m", str(int(top_m)),
        "--k", str(int(k)),
        "--prompt-stdin",
    ]

    p = subprocess.run(cmd, input=prompt.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(f"HRM query failed (code {p.returncode}):\n{p.stderr.decode('utf-8', errors='replace')}")

    try:
        raw = json.loads(p.stdout.decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"HRM returned invalid JSON: {e}\n--- stdout ---\n{p.stdout.decode('utf-8', errors='replace')}\n--- stderr ---\n{p.stderr.decode('utf-8', errors='replace')}")

    return HRMQueryResult(raw=raw)

