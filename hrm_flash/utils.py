# Copyright (c) 2026 Christian Heinrich Hohlfeld (Konstanz, Germany)
# https://christianhohlfeld.com | ORCID: https://orcid.org/0009-0003-6634-9045
# ALL RIGHTS RESERVED. No license granted without prior written permission.
from __future__ import annotations

import os
import shutil
from pathlib import Path


def find_hrm_binary(repo_root: Path | None = None, explicit: str | None = None) -> Path:
    """Locate the HRM CLI binary.

    Order:
      1) explicit path argument
      2) env HRM_BIN
      3) PATH lookup ("hrm")
      4) repo_root/hrm_core/build/hrm (dev checkout)
      5) CWD-relative ./hrm_core/build/hrm
    """
    if explicit:
        p = Path(explicit).expanduser().resolve()
        return p

    envp = os.environ.get("HRM_BIN")
    if envp:
        return Path(envp).expanduser().resolve()

    which = shutil.which("hrm")
    if which:
        return Path(which).resolve()

    if repo_root is not None:
        cand = (repo_root / "hrm_core" / "build" / "hrm")
        if cand.is_file():
            return cand.resolve()

    cand = (Path.cwd() / "hrm_core" / "build" / "hrm")
    if cand.is_file():
        return cand.resolve()

    # also try one directory up (common when running inside subfolder)
    cand2 = (Path.cwd().parent / "hrm_core" / "build" / "hrm")
    if cand2.is_file():
        return cand2.resolve()

    # last-resort: return repo_root guess if provided (will fail with good error later)
    if repo_root is not None:
        return (repo_root / "hrm_core" / "build" / "hrm").resolve()

    return Path("hrm")


def validate_tp_world(config, world: int) -> None:
    """Validate whether TP world size is compatible with a Llama-like config."""
    hidden = int(getattr(config, "hidden_size"))
    heads = int(getattr(config, "num_attention_heads"))
    inter = int(getattr(config, "intermediate_size", hidden * 4))
    vocab = int(getattr(config, "vocab_size"))

    if hidden % world != 0:
        raise ValueError(f"TP world={world} invalid: hidden_size={hidden} not divisible by world")
    if heads % world != 0:
        raise ValueError(f"TP world={world} invalid: num_attention_heads={heads} not divisible by world")
    if inter % world != 0:
        raise ValueError(f"TP world={world} invalid: intermediate_size={inter} not divisible by world")

    # Current vocab-parallel embedding/head require exact divisibility.
    if vocab % world != 0:
        raise ValueError(f"TP world={world} invalid: vocab_size={vocab} not divisible by world")

    # KV heads can be replicated; no hard constraint

