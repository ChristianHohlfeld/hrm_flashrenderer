# © 2026 Christian Heinrich Hohlfeld (Konstanz, Germany), christianhohlfeld.com, ORCID 0009-0003-6634-9045.
import os, sys, struct
import numpy as np
from safetensors import safe_open
from huggingface_hub import snapshot_download

def q8_rowwise(t: np.ndarray):
    # t: [out, in]
    absmax = np.maximum(np.abs(t).max(axis=1), 1e-8)
    scale = absmax / 127.0
    qi = np.clip(np.round(t / scale[:, None]), -128, 127).astype(np.int8)
    # scale_q12 = scale * 4096
    scale_q12 = np.round(scale * 4096.0).astype(np.int32)
    return qi, scale_q12

def write_tensor(f, name, qi, scale_q12):
    nb = name.encode("utf-8")
    f.write(struct.pack("<I", len(nb)))
    f.write(nb)
    f.write(struct.pack("<II", qi.shape[0], qi.shape[1]))
    f.write(scale_q12.tobytes())   # per-row scale
    f.write(qi.tobytes())

def load_file_numpy(path):
    tensors = {}
    with safe_open(path, framework="np", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 hf_export.py <repo_id> <out.bin>")
        sys.exit(1)
    
    repo = sys.argv[1]
    outp = sys.argv[2]
    print(f"[*] Downloading/Loading {repo}...")
    mp = snapshot_download(repo, allow_patterns=["*.safetensors", "config.json"])
    tensors = {}
    for fn in os.listdir(mp):
        if fn.endswith(".safetensors"):
            print(f"[*] Loading {fn}...")
            tensors.update(load_file_numpy(os.path.join(mp, fn)))
            
    print(f"[*] Exporting to {outp}...")
    with open(outp, "wb") as f:
        f.write(b"DSI8")              # magic
        f.write(struct.pack("<I", 1)) # version
        
        keys_to_export = [
            "model.embed_tokens.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.norm.weight",
            "lm_head.weight"
        ]
        
        for key in keys_to_export:
            if key not in tensors:
                print(f"[!] Warning: {key} not found in model.")
                continue
            t = tensors[key]
            if t.ndim == 1:
                # Layernorm weights are 1D, write as floats since it's used in RMS
                nb = key.encode("utf-8")
                f.write(struct.pack("<I", len(nb)))
                f.write(nb)
                f.write(struct.pack("<II", t.shape[0], 1))
                # Write empty scale (1 element)
                f.write(struct.pack("<i", 4096)) 
                f.write(t.astype(np.float32).tobytes())
                continue
                
            if t.ndim != 2:
                continue
            qi, sc = q8_rowwise(t)
            write_tensor(f, key, qi, sc)
            
    print("[*] Done!")

if __name__ == "__main__":
    main()
