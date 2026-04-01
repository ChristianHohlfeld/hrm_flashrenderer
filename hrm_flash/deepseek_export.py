# Copyright (c) 2026 Christian Heinrich Hohlfeld (Konstanz, Germany)
# https://christianhohlfeld.com | ORCID: https://orcid.org/0009-0003-6634-9045
# ALL RIGHTS RESERVED. No license granted without prior written permission.
from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Dict

import numpy as np


def _bf16_to_fp32(raw: bytes, shape: list[int]) -> np.ndarray:
    bf = np.frombuffer(raw, dtype=np.uint16)
    return (bf.astype(np.uint32) << 16).view(np.float32).reshape(shape)


def _load_safetensors(path: Path) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    with path.open("rb") as f:
        hs = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hs))
        base = 8 + hs
        for k, m in hdr.items():
            if k == "__metadata__":
                continue
            s, e = m["data_offsets"]
            f.seek(base + s)
            raw = f.read(e - s)
            shape = m["shape"]
            dt = m["dtype"]
            if dt == "BF16":
                out[k] = _bf16_to_fp32(raw, shape)
            elif dt == "F16":
                out[k] = np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)
            elif dt == "F32":
                out[k] = np.frombuffer(raw, dtype=np.float32).reshape(shape)
            else:
                out[k] = np.frombuffer(raw, dtype=np.float32).reshape(shape)
    return out


def _q8_row(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    absmax = np.abs(t).max(axis=-1)
    absmax = np.maximum(absmax, 1e-8)
    scale = absmax / 127.0
    qi = np.clip(np.round(t / scale[..., None]), -128, 127).astype(np.int8)
    return qi, scale.astype(np.float32)


def _resolve_model_dir(model_source: str, local_files_only: bool = False) -> Path:
    p = Path(model_source).expanduser()
    if p.is_dir():
        return p.resolve()

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise RuntimeError(f"huggingface_hub is required to resolve model source '{model_source}': {e}") from e

    try:
        cached = snapshot_download(
            repo_id=model_source,
            local_files_only=True,
            allow_patterns=["*.safetensors", "config.json"],
        )
        return Path(cached).resolve()
    except Exception:
        if local_files_only:
            raise RuntimeError(f"Model '{model_source}' not found in local cache and local_files_only=True.")

    downloaded = snapshot_download(
        repo_id=model_source,
        allow_patterns=["*.safetensors", "config.json"],
    )
    return Path(downloaded).resolve()


def export_dsi8_v3(model_source: str, out_path: Path, local_files_only: bool = False) -> Path:
    model_dir = _resolve_model_dir(model_source, local_files_only=local_files_only)
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise RuntimeError(f"Missing config.json in model source: {model_dir}")

    with config_path.open("r", encoding="utf-8") as cf:
        cfg = json.load(cf)

    L = int(cfg.get("num_hidden_layers", 28))
    D = int(cfg.get("hidden_size", 1536))
    H = int(cfg.get("num_attention_heads", 12))
    KVH = int(cfg.get("num_key_value_heads", 2))
    Dh = D // H
    F = int(cfg.get("intermediate_size", 8960))
    V = int(cfg.get("vocab_size", 151936))
    Tmax = 4096
    rope_theta = float(cfg.get("rope_theta", 10000.0))
    rms_eps = float(cfg.get("rms_norm_eps", 1e-6))
    bos = int(cfg.get("bos_token_id", 151643))
    eos = int(cfg.get("eos_token_id", 151643))

    tensors: Dict[str, np.ndarray] = {}
    for fn in sorted(os.listdir(model_dir)):
        if fn.endswith(".safetensors"):
            tensors.update(_load_safetensors(model_dir / fn))
    if not tensors:
        raise RuntimeError(f"No safetensors found in model source: {model_dir}")

    keys_2d = ["model.embed_tokens.weight"]
    keys_1d: list[str] = []
    for l in range(L):
        pfx = f"model.layers.{l}"
        keys_1d.append(f"{pfx}.input_layernorm.weight")
        keys_2d.extend(
            [
                f"{pfx}.self_attn.q_proj.weight",
                f"{pfx}.self_attn.k_proj.weight",
                f"{pfx}.self_attn.v_proj.weight",
                f"{pfx}.self_attn.o_proj.weight",
                f"{pfx}.mlp.gate_proj.weight",
                f"{pfx}.mlp.up_proj.weight",
                f"{pfx}.mlp.down_proj.weight",
            ]
        )
        keys_1d.append(f"{pfx}.post_attention_layernorm.weight")
        for bp in ("self_attn.q_proj.bias", "self_attn.k_proj.bias", "self_attn.v_proj.bias"):
            k = f"{pfx}.{bp}"
            if k in tensors:
                keys_1d.append(k)
    keys_1d.append("model.norm.weight")
    keys_2d.append("lm_head.weight")

    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    TENSOR_TYPE_Q8 = 0
    TENSOR_TYPE_F32 = 1

    with out_path.open("wb") as f:
        f.write(b"DSI8")
        f.write(struct.pack("<I", 3))
        f.write(struct.pack("<IIIIIIII", D, H, KVH, Dh, F, V, L, Tmax))
        f.write(struct.pack("<d", rope_theta))
        f.write(struct.pack("<f", rms_eps))
        f.write(struct.pack("<II", bos, eos))

        for key in keys_2d:
            if key not in tensors:
                continue
            t = tensors[key].astype(np.float32)
            if t.ndim == 1:
                t = t.reshape(1, -1)
            qi, sc = _q8_row(t)
            nb = key.encode()
            f.write(struct.pack("<I", len(nb)))
            f.write(nb)
            f.write(struct.pack("<I", TENSOR_TYPE_Q8))
            f.write(struct.pack("<I", len(qi.shape)))
            for d in qi.shape:
                f.write(struct.pack("<I", int(d)))
            f.write(sc.tobytes())
            f.write(qi.tobytes())

        for key in keys_1d:
            if key not in tensors:
                continue
            t = tensors[key].astype(np.float32).ravel()
            nb = key.encode()
            f.write(struct.pack("<I", len(nb)))
            f.write(nb)
            f.write(struct.pack("<I", TENSOR_TYPE_F32))
            f.write(struct.pack("<I", 1))
            f.write(struct.pack("<I", int(t.shape[0])))
            f.write(t.tobytes())

    return out_path
