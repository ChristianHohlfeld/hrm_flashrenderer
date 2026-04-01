# Copyright (c) 2026 Christian Heinrich Hohlfeld (Konstanz, Germany)
# https://christianhohlfeld.com | ORCID: https://orcid.org/0009-0003-6634-9045
# ALL RIGHTS RESERVED. No license granted without prior written permission.
from __future__ import annotations

import json
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


def ensure_local_llm_model(model_str: str, local_files_only: bool = False, project_root: Path | None = None) -> Path:
    """Ensure model_str resolves to a local safetensors model directory.

    If model_str is already a local directory, it is returned as-is.
    Otherwise we try project-local cache (`<project_root>/llm_models`) and then
    Hugging Face Hub download.
    """
    p = Path(model_str)
    if p.is_dir():
        return p.resolve()

    root = project_root or Path(__file__).resolve().parents[1]
    local_models_dir = root / "llm_models"
    safe_name = model_str.replace("/", "--").replace("\\", "--")
    target_dir = local_models_dir / safe_name

    if target_dir.joinpath("config.json").exists():
        return target_dir.resolve()

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise RuntimeError(f"huggingface_hub is required to resolve non-local models: {e}") from e

    try:
        path = snapshot_download(
            repo_id=model_str,
            local_files_only=True,
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"],
        )
        return Path(path).resolve()
    except Exception:
        if local_files_only:
            raise RuntimeError(f"Model '{model_str}' not found in local cache and --local_files_only is set.")

    local_models_dir.mkdir(exist_ok=True)
    path = snapshot_download(
        repo_id=model_str,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"],
    )
    return Path(path).resolve()


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


def validate_native_model_config(config) -> None:
    """Validate model config for the native hrm-flash TP path.

    This path currently supports dense decoder-only transformer layouts that map
    to the expected Llama/Qwen-style weight keys.
    """
    hidden = int(getattr(config, "hidden_size"))
    heads = int(getattr(config, "num_attention_heads"))
    if hidden % heads != 0:
        raise ValueError(f"Invalid model config: hidden_size={hidden} not divisible by num_attention_heads={heads}")
    head_dim = hidden // heads
    if head_dim not in (64, 128):
        raise ValueError(
            f"Unsupported head_dim={head_dim}. Native flashattention kernel supports head_dim 64 or 128 only."
        )

    archs = [str(a).lower() for a in (getattr(config, "architectures", None) or [])]
    if any("moe" in a for a in archs):
        raise ValueError(
            "MoE architecture detected. Native hrm-flash path currently supports dense models only."
        )
    if any("deepseekv3" in a or "deepseek_v3" in a for a in archs):
        raise ValueError(
            "DeepSeek-V3 style architecture detected. Native hrm-flash currently supports DeepSeek distill (dense) variants, not full MoE V3."
        )

    # Common MoE signals across configs.
    for name in ("num_local_experts", "n_routed_experts", "num_experts"):
        if hasattr(config, name):
            try:
                val = int(getattr(config, name))
            except Exception:
                continue
            if val > 1:
                raise ValueError(
                    f"MoE config detected via {name}={val}. Native hrm-flash path currently supports dense models only."
                )


def validate_native_weight_layout(model_dir: Path) -> None:
    """Validate that local weights look compatible with dense native TP loader."""
    model_dir = Path(model_dir).resolve()
    index_path = model_dir / "model.safetensors.index.json"
    single_path = model_dir / "model.safetensors"
    if not index_path.is_file() and not single_path.is_file():
        raise ValueError(
            "No native safetensors weights found. Native hrm-flash requires model.safetensors(.index.json). "
            "GGUF belongs to renderer/hrm_render.py."
        )

    required_keys = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.norm.weight",
    ]

    keys: set[str] = set()
    if index_path.is_file():
        with index_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        weight_map = payload.get("weight_map") or {}
        if not isinstance(weight_map, dict):
            raise ValueError("Invalid model.safetensors.index.json: weight_map missing or invalid.")
        keys = set(weight_map.keys())
    else:
        try:
            from safetensors.torch import safe_open

            with safe_open(str(single_path), framework="pt", device="cpu") as f:
                keys = set(f.keys())
        except Exception as e:
            raise ValueError(f"Failed to inspect model.safetensors: {e}") from e

    missing = [k for k in required_keys if k not in keys]
    if missing:
        raise ValueError(
            "Model weights are not compatible with native dense TP loader (missing required keys). "
            f"First missing key: {missing[0]}. This often means GPTQ/AWQ/MoE weights; use dense safetensors."
        )

