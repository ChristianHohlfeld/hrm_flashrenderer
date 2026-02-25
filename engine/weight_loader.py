# Copyright (c) 2026 Christian Heinrich Hohlfeld (Konstanz, Germany)
# https://christianhohlfeld.com | ORCID: https://orcid.org/0009-0003-6634-9045
# ALL RIGHTS RESERVED. No license granted without prior written permission.
import os, json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import torch
from safetensors.torch import safe_open

@dataclass
class WeightIndex:
    base_dir: str
    weight_map: Dict[str, str]

def _find_index(model_dir: str) -> Optional[WeightIndex]:
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.isfile(idx):
        with open(idx, "r", encoding="utf-8") as f:
            j = json.load(f)
        return WeightIndex(base_dir=model_dir, weight_map=j["weight_map"])
    single = os.path.join(model_dir, "model.safetensors")
    if os.path.isfile(single):
        return WeightIndex(base_dir=model_dir, weight_map={})
    return None

SliceSpec = Tuple[slice, ...]  # e.g. (slice(a,b), slice(None))

class WeightLoader:
    """Production loader with *sharded slicing*.

    Key property: **avoid materializing full tensors** on CPU when only a slice is needed.
    Uses safetensors `get_slice` when available; falls back to full tensor then slice.

    Expected local folder:
      - model.safetensors
      - or model.safetensors.index.json + shard files
    """
    def __init__(self, model_dir: str):
        if not os.path.isdir(model_dir):
            raise RuntimeError(f"WeightLoader expects a local model directory, got: {model_dir}")
        self.model_dir = model_dir
        self.index = _find_index(model_dir)
        if self.index is None:
            raise RuntimeError(
                f"No safetensors weights found in {model_dir}.\n"
                f"Make sure the directory contains 'model.safetensors' or 'model.safetensors.index.json'.\n"
                f"Note: GGUF models are not supported by hrm-flash generate (use renderer/hrm_render.py instead)."
            )
        self._open = {}

    def _file_for_key(self, key: str) -> str:
        if self.index.weight_map == {}:
            return "model.safetensors"
        if key not in self.index.weight_map:
            raise KeyError(f"Key not found in index: {key}")
        return self.index.weight_map[key]

    def _open_file(self, fn: str):
        path = os.path.join(self.model_dir, fn)
        if path not in self._open:
            self._open[path] = safe_open(path, framework="pt", device="cpu")
        return self._open[path]

    def get(self, key: str) -> torch.Tensor:
        fn = self._file_for_key(key)
        f = self._open_file(fn)
        if key not in f.keys():
            raise KeyError(f"Key not found in {fn}: {key}")
        return f.get_tensor(key)

    def get_slice(self, key: str, slices: SliceSpec) -> torch.Tensor:
        fn = self._file_for_key(key)
        f = self._open_file(fn)
        if key not in f.keys():
            raise KeyError(f"Key not found in {fn}: {key}")
        # Try safetensors slicing API (zero-copy from mmap where possible)
        try:
            s = f.get_slice(key)
            return s[slices]
        except Exception:
            t = f.get_tensor(key)
            return t[slices]

    def rows(self, key: str, start: int, end: int) -> torch.Tensor:
        return self.get_slice(key, (slice(start, end), slice(None)))

    def cols(self, key: str, start: int, end: int) -> torch.Tensor:
        return self.get_slice(key, (slice(None), slice(start, end)))

    def close(self):
        for f in self._open.values():
            try:
                f.__exit__(None, None, None)
            except Exception:
                pass
        self._open.clear()

