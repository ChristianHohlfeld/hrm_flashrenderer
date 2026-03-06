import os, sys, struct, hashlib, glob, json
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

def find_tensor(filename, tensor_name):
    try:
        with safe_open(filename, framework="pt") as f:
            if tensor_name in f.keys():
                return f.get_tensor(tensor_name)
    except: pass
    return None

repo_id = sys.argv[1]
workdir = sys.argv[2]
pair_k1 = int(sys.argv[3])
pair_k2 = int(sys.argv[4])
index_bin_path = sys.argv[5]

print(f"[*] Downloading metadata for {repo_id}...")
model_path = snapshot_download(repo_id, allow_patterns=["*.safetensors", "config.json"])

with open(os.path.join(model_path, "config.json")) as f:
    config = json.load(f)

# RevNet Engine assumes D_engine = 2 * hidden_size to fit llama layers in Dhf slots
D_orig = config.get("hidden_size", 4096)
D = D_orig * 2
H_q = config.get("num_attention_heads", 32)
H_kv = config.get("num_key_value_heads", 8)
L = config.get("num_hidden_layers", 32)
F = config.get("intermediate_size", 11008)
Tmax_embed = config.get("max_position_embeddings", 2048)
v_orig = config.get("vocab_size", 32000)

v_pad = ((v_orig + pair_k1 + pair_k2 + 15) // 16) * 16
Dhf = D_orig

env_path = os.path.join(workdir, "hf_env.sh")
with open(env_path, "w") as f:
    f.write(f"export DMODEL={D}\n")
    f.write(f"export NHEAD={H_q}\n")
    f.write(f"export NLAY={L}\n")
    f.write(f"export FFN={F}\n")
    f.write(f"export TMAX={Tmax_embed}\n")

bin_name = "llm_engine"
hash_str = f"{bin_name}_K{pair_k1+pair_k2}_D{D}_H{H_q}_L{L}_F{F}_T{Tmax_embed}"
ckpt_hash = hashlib.md5(hash_str.encode()).hexdigest()[:8]
ckpt_path = os.path.join(workdir, f"ckpt_{bin_name}_{ckpt_hash}.bin")

with open(env_path, "a") as f:
    f.write(f"export CKPT_HASH={ckpt_hash}\n")
    f.write(f"export DEFAULT_CKPT_FILE={ckpt_path}\n")

if os.path.exists(ckpt_path):
    print(f"[*] Checkpoint {ckpt_path} exists. Skipping.")
    sys.exit(0)

safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))

def get_t(name, shape=None, transpose=False, repeat_heads=1):
    t = None
    for f in safetensors_files:
        t = find_tensor(f, name)
        if t is not None: break
    if t is None:
        print(f"Warning: {name} not found.")
        return torch.zeros(shape if shape else (1,), dtype=torch.float32)
    
    t = t.float()
    if repeat_heads > 1:
        # GQA: Repeat KV heads (cur: H_kv, target: H_q)
        # Weight shape is (H_kv*head_dim, D_orig)
        tdim = t.shape[0]
        t = t.view(H_kv, tdim // H_kv, -1)
        t = t.repeat_interleave(repeat_heads, dim=0)
        t = t.view(-1, t.shape[-1])
    
    if transpose: t = t.t()
    
    if shape:
        out = torch.zeros(shape, dtype=torch.float32)
        s0, s1 = min(shape[0], t.shape[0]), min(shape[1], t.shape[1]) if len(shape)>1 else 0
        if len(shape)==1: out[:s0] = t[:s0]
        else: out[:s0, :s1] = t[:s0, :s1]
        t = out
    return t

def write_tensor(f, t):
    t_int = (t * 10000.0).to(torch.int32).flatten().numpy()
    f.write(t_int.tobytes())

print(f"[*] Mapping {repo_id} to {ckpt_path} (D={D}, Dhf={Dhf}, F={F})...")
with open(index_bin_path, "rb") as idx_f:
    index_data = idx_f.read()
pow2 = struct.unpack("<I", index_data[16:20])[0]

with open(ckpt_path, "wb") as f:
    f.write(struct.pack("<II", 0x43484452, 11))
    f.write(struct.pack("<II", pair_k1, pair_k2))
    f.write(struct.pack("<IIIII", D, H_q, L, F, Tmax_embed))
    id2pair_len = pair_k1 * 2
    id2pair2_len = pair_k2 * 4
    f.write(index_data[24 : 24+id2pair_len])
    f.write(index_data[24+id2pair_len : 24+id2pair_len+id2pair2_len])
    f.write(struct.pack("<I", pow2))
    f.write(index_data[24+id2pair_len+id2pair2_len:])
    
    # wte: Map Llama hidden_size to engine first-half(Dhf)
    print("[*] Writing embeddings...")
    write_tensor(f, get_t("model.embed_tokens.weight", (v_pad, D)))
    write_tensor(f, torch.zeros((Tmax_embed, D), dtype=torch.float32)) # wpe
    
    for l in range(L):
        print(f"[*] Layer {l}/{L}...")
        write_tensor(f, get_t(f"model.layers.{l}.input_layernorm.weight", (Dhf,)))
        write_tensor(f, get_t(f"model.layers.{l}.self_attn.q_proj.weight", (Dhf, Dhf), transpose=True))
        write_tensor(f, get_t(f"model.layers.{l}.self_attn.k_proj.weight", (Dhf, Dhf), transpose=True, repeat_heads=H_q//H_kv))
        write_tensor(f, get_t(f"model.layers.{l}.self_attn.v_proj.weight", (Dhf, Dhf), transpose=True, repeat_heads=H_q//H_kv))
        write_tensor(f, get_t(f"model.layers.{l}.self_attn.o_proj.weight", (Dhf, Dhf), transpose=True))
        
        write_tensor(f, get_t(f"model.layers.{l}.post_attention_layernorm.weight", (Dhf,)))
        # MLP: Gate and Down. (Llama has Gate, Up, Down. We use static bridge for minimal change)
        write_tensor(f, get_t(f"model.layers.{l}.mlp.gate_proj.weight", (F, Dhf), transpose=True))
        write_tensor(f, get_t(f"model.layers.{l}.mlp.down_proj.weight", (Dhf, F), transpose=True))
        
    write_tensor(f, get_t("model.norm.weight", (D_orig,))) # Pad to D later via get_t shape if needed
    write_tensor(f, get_t("lm_head.weight", (v_pad, D_orig), transpose=True)) # Pad to D later
print(f"[*] Done.")
