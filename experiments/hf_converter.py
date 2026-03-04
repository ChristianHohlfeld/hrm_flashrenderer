import os, sys, struct, hashlib, glob
try:
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    
    def find_tensor(filename, tensor_name):
        with safe_open(filename, framework="pt") as f:
            if tensor_name in f.keys():
                return f.get_tensor(tensor_name)
        return None

except ImportError:
    print("FATAL: Please install dependencies: pip install torch huggingface_hub safetensors")
    sys.exit(1)

except ImportError:
    print("FATAL: Please install dependencies: pip install numpy huggingface_hub safetensors")
    sys.exit(1)

with open("/tmp/deepseek_status.txt", "w") as f:
    f.write(f"DOWNLOADING {sys.argv[1]}...")
    
repo_id = sys.argv[1]
workdir = sys.argv[2]
pair_k1 = int(sys.argv[3])
pair_k2 = int(sys.argv[4])
index_bin_path = sys.argv[5]
print(f"[*] Downloading metadata for {repo_id}...")
model_path = snapshot_download(repo_id, allow_patterns=["*.safetensors", "config.json"])

with open("/tmp/deepseek_status.txt", "w") as f:
    f.write("LOADING TENSORS...")
import json
with open(os.path.join(model_path, "config.json")) as f:
    config = json.load(f)
D = config.get("hidden_size", 256)
H = config.get("num_attention_heads", 8)
L = config.get("num_hidden_layers", 6)
F = config.get("intermediate_size", 1024)
Tmax = config.get("max_position_embeddings", 512)
# Write out the env overrides for the Bash script to source
env_path = os.path.join(workdir, "hf_env.sh")
with open(env_path, "w") as f:
    f.write(f"export DMODEL={D}\n")
    f.write(f"export NHEAD={H}\n")
    f.write(f"export NLAY={L}\n")
    f.write(f"export FFN={F}\n")
    f.write(f"export TMAX={Tmax}\n")
# Calculate custom hash to see if we already built this .bin
bin_name = "llm_engine"
v_pad = ((256 + pair_k1 + pair_k2 + 15) // 16) * 16
hash_str = f"{bin_name}_K{pair_k1+pair_k2}_D{D}_H{H}_L{L}_F{F}_T{Tmax}"
ckpt_hash = hashlib.md5(hash_str.encode()).hexdigest()[:8]
ckpt_path = os.path.join(workdir, f"ckpt_{bin_name}_{ckpt_hash}.bin")
with open(env_path, "a") as f:
    f.write(f"export CKPT_HASH={ckpt_hash}\n")
    f.write(f"export DEFAULT_CKPT_FILE={ckpt_path}\n")
if os.path.exists(ckpt_path):
    print(f"[*] Checkpoint {ckpt_path} already exists. Skipping tensor mapping.")
    sys.exit(0)
print(f"[*] Mapping HuggingFace Safetensors to Elite C++ Layout...")
safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))

# Reversible architecture requires Dhf = D/2.
Dhf = D // 2
def get_t(name, expected_shape=None):
    t_pt = None
    for file in safetensors_files:
        t_pt = find_tensor(file, name)
        if t_pt is not None:
            break
            
    if t_pt is None:
        print(f"Warning: {name} not found. Filling with zeros.")
        if expected_shape: return torch.zeros(expected_shape, dtype=torch.float32)
        return torch.zeros(1, dtype=torch.float32)
        
    t = t_pt.float()
    del t_pt
    
    if expected_shape and list(t.shape) != list(expected_shape):
        print(f"Warning: {name} shape mismatch. Expected {expected_shape}, got {t.shape}")
        out = torch.zeros(expected_shape, dtype=torch.float32)
        slices = tuple(slice(0, min(d_out, d_in)) for d_out, d_in in zip(expected_shape, t.shape))
        out[slices] = t[slices]
        del t
        return out
        
    return t
    
def write_tensor(f, t):
    t_int = (t * 1000.0).to(torch.int32).flatten().numpy()
    f.write(t_int.tobytes())
    del t_int
    del t
with open(index_bin_path, "rb") as idx_f:
    index_data = idx_f.read()

pow2 = struct.unpack("<I", index_data[16:20])[0]

with open(ckpt_path, "wb") as f:
    f.write(struct.pack("<II", 0x43484452, 11))
    f.write(struct.pack("<II", pair_k1, pair_k2))
    f.write(struct.pack("<IIIII", D, H, L, F, Tmax))
   
    id2pair_len = pair_k1 * 2
    id2pair2_len = pair_k2 * 4
    f.write(index_data[24 : 24+id2pair_len])
    f.write(index_data[24+id2pair_len : 24+id2pair_len+id2pair2_len])
    f.write(struct.pack("<I", pow2))
    f.write(index_data[24+id2pair_len+id2pair2_len:])
   
    print("[*] Quantizing and writing tensors (INT32)...")
    write_tensor(f, get_t("model.embed_tokens.weight", (v_pad, D)))
    write_tensor(f, torch.zeros((Tmax, D), dtype=torch.float32)) # wpe
   
    for l in range(L):
        write_tensor(f, get_t(f"model.layers.{l}.input_layernorm.weight", (Dhf,)))
        write_tensor(f, get_t(f"model.layers.{l}.self_attn.q_proj.weight", (Dhf, Dhf)))
        write_tensor(f, get_t(f"model.layers.{l}.self_attn.k_proj.weight", (Dhf, Dhf)))
        write_tensor(f, get_t(f"model.layers.{l}.self_attn.v_proj.weight", (Dhf, Dhf)))
        write_tensor(f, get_t(f"model.layers.{l}.self_attn.o_proj.weight", (Dhf, Dhf)))
       
        write_tensor(f, get_t(f"model.layers.{l}.post_attention_layernorm.weight", (Dhf,)))
        write_tensor(f, get_t(f"model.layers.{l}.mlp.gate_proj.weight", (Dhf, F)))
        write_tensor(f, get_t(f"model.layers.{l}.mlp.down_proj.weight", (F, Dhf)))
    write_tensor(f, get_t("model.norm.weight", (D,)))
    write_tensor(f, get_t("lm_head.weight", (D, v_pad)))
print(f"[*] Successfully built Elite INT32 checkpoint: {ckpt_path}")
