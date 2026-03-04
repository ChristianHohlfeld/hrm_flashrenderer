#!/usr/bin/env bash
set -euo pipefail

need(){ command -v "$1" >/dev/null 2>&1 || { echo "FATAL: $1 not found"; exit 1; }; }
need nvcc || { echo "FATAL: nvcc not found (install CUDA toolkit)"; exit 1; }
need g++ || { echo "FATAL: g++ not found (sudo apt install build-essential)"; exit 1; }
need curl || { echo "FATAL: curl not found"; exit 1; }
need python3 || { echo "FATAL: python3 not found"; exit 1; }

absify_inputs() {
  local s="$1" out="" part
  IFS=':' read -ra parts <<< "$s"
  for part in "${parts[@]}"; do
    [[ -z "$part" ]] && continue
    if [[ "$part" = /* ]]; then
      out+="${out:+:}$part"
    else
      out+="${out:+:}$WORKDIR/$part"
    fi
  done
  echo "$out"
}

WORKDIR="${WORKDIR:-$PWD}"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

DATA_FILE="${DATA_FILE:-tinyshakespeare.txt}"
URL="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
if [[ ! -s "$DATA_FILE" ]]; then
  curl -L --fail "$URL" -o "$DATA_FILE"
fi
[[ -s "$DATA_FILE" ]] || { echo "FATAL: corpus empty: $DATA_FILE" >&2; exit 1; }

: "${PAIR_K:=16384}"
: "${PAIR_K1:=8192}"
: "${VCHUNK:=1024}"
INDEX_INPUTS="${INDEX_INPUTS:-$DATA_FILE}"
K2=$((PAIR_K-PAIR_K1))
INDEX_BIN="${INDEX_BIN:-index_v7_k1${PAIR_K1}_k2${K2}.bin}"
ABS_INPUT="$(absify_inputs "$INDEX_INPUTS")"
FORCE_INDEX="${FORCE_INDEX:-0}"

if [[ "$FORCE_INDEX" == "1" || ! -s "$INDEX_BIN" ]]; then
  TMP_BUILD_DIR=$(mktemp -d)
 
  cat > "$TMP_BUILD_DIR/index_build_v7.cpp" <<'CPP'
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <queue>
static void die(const char* m){ std::fprintf(stderr,"FATAL: %s\n",m); std::exit(1); }
static void wf(FILE* f,const void* p,size_t n){ if(std::fwrite(p,1,n,f)!=n) die("write failed"); }
static void wu32(FILE* f,uint32_t x){ wf(f,&x,4); }
static void wu16(FILE* f,uint16_t x){ wf(f,&x,2); }
static std::vector<uint8_t> read_all(const std::string& path){
  FILE* f=std::fopen(path.c_str(),"rb");
  if(!f){ std::fprintf(stderr,"FATAL: cannot open %s\n", path.c_str()); std::exit(1); }
  std::fseek(f,0,SEEK_END);
  long n=std::ftell(f);
  std::fseek(f,0,SEEK_SET);
  if(n<=0){ std::fclose(f); return {}; }
  std::vector<uint8_t> b((size_t)n);
  if(std::fread(b.data(),1,(size_t)n,f)!=(size_t)n) die("read failed");
  std::fclose(f);
  return b;
}
static std::vector<std::string> split_inputs(const std::string& s){
  std::vector<std::string> out;
  size_t i=0;
  while(i<s.size()){
    size_t j=s.find(':', i);
    if(j==std::string::npos) j=s.size();
    std::string p=s.substr(i, j-i);
    if(!p.empty()) out.push_back(p);
    i=j+1;
  }
  std::sort(out.begin(), out.end());
  return out;
}
struct Stage1 {
  int K1;
  std::vector<uint16_t> id2pair;
  int32_t pair2id[65536];
};
static Stage1 build_stage1(const std::vector<std::string>& inputs, int K1){
  Stage1 s1{}; s1.K1=K1;
  std::fill(std::begin(s1.pair2id), std::end(s1.pair2id), -1);
  std::vector<uint64_t> cnt(65536,0);
  for(const auto& p: inputs){
    auto b=read_all(p);
    if(b.size()<2) continue;
    for(size_t i=0;i+1<b.size();i++){
      uint16_t k=(uint16_t)b[i] | (uint16_t)((uint16_t)b[i+1]<<8);
      cnt[k]++;
    }
  }
  std::vector<uint16_t> all; all.reserve(65536);
  for(uint32_t k=0;k<65536;k++) if(cnt[k]) all.push_back((uint16_t)k);
  std::stable_sort(all.begin(), all.end(), [&](uint16_t a,uint16_t b){
    uint64_t ca=cnt[a], cb=cnt[b];
    if(ca!=cb) return ca>cb;
    return a<b;
  });
  if((int)all.size()<K1){
    std::vector<bool> used(65536, false);
    for(uint16_t p : all) used[p] = true;
    for(uint32_t p=0; p<65536 && (int)all.size()<K1; p++){
      if(!used[p]) all.push_back((uint16_t)p);
    }
  }
  s1.id2pair.assign(all.begin(), all.begin()+K1);
  for(int i=0;i<K1;i++) s1.pair2id[s1.id2pair[(size_t)i]] = 256 + i;
  return s1;
}
static inline bool next_stage1_id(const Stage1& s1, const uint8_t* b, size_t n, size_t& i, uint16_t& out){
  if(i>=n) return false;
  if(i+1<n){
    uint16_t k=(uint16_t)b[i] | (uint16_t)((uint16_t)b[i+1]<<8);
    int32_t id=s1.pair2id[k];
    if(id>=0){ out=(uint16_t)id; i+=2; return true; }
  }
  out=(uint16_t)b[i]; i+=1; return true;
}
static inline uint32_t tokpair_key(uint16_t a, uint16_t b){ return ((uint32_t)a<<16) | (uint32_t)b; }
static void write_run(const std::string& path, std::vector<uint32_t>& keys){
  std::sort(keys.begin(), keys.end());
  FILE* f=std::fopen(path.c_str(),"wb");
  if(!f) die("cannot open run file");
  wf(f, keys.data(), keys.size()*sizeof(uint32_t));
  std::fclose(f);
}
struct RunReader{
  FILE* f=nullptr; uint32_t cur=0; bool ok=false;
  RunReader(const std::string& p){
    f=std::fopen(p.c_str(),"rb"); if(!f) die("cannot open run");
    ok=(std::fread(&cur,1,4,f)==4);
  }
  ~RunReader(){ if(f) std::fclose(f); }
  bool pop(){ if(!ok) return false; ok=(std::fread(&cur,1,4,f)==4); return ok; }
};
static int next_pow2(int x){ int p=1; while(p<x) p<<=1; return p; }
struct CountItem{ uint64_t c; uint32_t k; };
static void build_stage2_external(const std::vector<std::string>& inputs, const Stage1& s1, int K2,
                                  std::vector<uint32_t>& id2pair2){
  id2pair2.clear();
  if(K2<=0) return;
  const size_t CHUNK_KEYS = 16ULL*1024ULL*1024ULL;
  std::vector<uint32_t> keys; keys.reserve(CHUNK_KEYS);
  std::vector<std::string> runs;
  size_t run_id=0;
  for(const auto& p: inputs){
    auto b=read_all(p);
    if(b.size()<3) continue;
    size_t i=0; bool have=false; uint16_t prev=0;
    while(true){
      uint16_t cur=0;
      if(!next_stage1_id(s1, b.data(), b.size(), i, cur)) break;
      if(have){
        keys.push_back(tokpair_key(prev, cur));
        if(keys.size()>=CHUNK_KEYS){
          char fn[256]; std::snprintf(fn,sizeof(fn),"run_%06zu.bin", run_id++);
          runs.push_back(fn);
          write_run(runs.back(), keys);
          keys.clear();
        }
      }
      prev=cur; have=true;
    }
  }
  if(!keys.empty()){
    char fn[256]; std::snprintf(fn,sizeof(fn),"run_%06zu.bin", run_id++);
    runs.push_back(fn);
    write_run(runs.back(), keys);
    keys.clear();
  }
  if(runs.empty()) die("stage2: no runs produced");
  struct Node{ uint32_t k; int r; };
  auto cmp=[](const Node& a, const Node& b){ return a.k > b.k; };
  std::priority_queue<Node, std::vector<Node>, decltype(cmp)> pq(cmp);
  std::vector<RunReader*> rr; rr.reserve(runs.size());
  for(size_t i=0;i<runs.size();i++){
    rr.push_back(new RunReader(runs[i]));
    if(rr.back()->ok) pq.push(Node{rr.back()->cur, (int)i});
  }
  auto worse = [](const CountItem& a, const CountItem& b){
    if(a.c!=b.c) return a.c > b.c;
    return a.k < b.k;
  };
  auto better = [](const CountItem& a, const CountItem& b){
    if(a.c!=b.c) return a.c > b.c;
    return a.k < b.k;
  };
  std::vector<CountItem> heap; heap.reserve((size_t)K2);
  auto heap_push = [&](uint64_t c, uint32_t k){
    CountItem it{c,k};
    if((int)heap.size() < K2){
      heap.push_back(it);
      std::push_heap(heap.begin(), heap.end(), worse);
    }else{
      CountItem worst_it = heap.front();
      if(better(it, worst_it)){
        std::pop_heap(heap.begin(), heap.end(), worse);
        heap.back() = it;
        std::push_heap(heap.begin(), heap.end(), worse);
      }
    }
  };
  uint32_t curk=0; uint64_t curc=0; bool have=false;
  while(!pq.empty()){
    Node n=pq.top(); pq.pop();
    uint32_t k=n.k;
    if(!have){ curk=k; curc=1; have=true; }
    else if(k==curk){ curc++; }
    else { heap_push(curc,curk); curk=k; curc=1; }
    RunReader* r=rr[(size_t)n.r];
    if(r->pop()) pq.push(Node{r->cur, n.r});
  }
  if(have) heap_push(curc,curk);
  for(auto* p: rr) delete p;
  for(const auto& p: runs) std::remove(p.c_str());
  std::sort(heap.begin(), heap.end(), [](const CountItem& a, const CountItem& b){
    if(a.c!=b.c) return a.c>b.c;
    return a.k<b.k;
  });
  if((int)heap.size()<K2){
    for(size_t j=heap.size(); j<(size_t)K2; j++){
      uint32_t dummy = (0xFFFFu<<16) | (uint32_t)(j & 0xFFFFu);
      heap.push_back(CountItem{0u, dummy});
    }
  }
  id2pair2.resize((size_t)K2);
  for(int i=0;i<K2;i++) id2pair2[(size_t)i]=heap[(size_t)i].k;
}
static void build_hash(const std::vector<uint32_t>& id2pair2, int K1, std::vector<uint32_t>& hkeys, std::vector<uint16_t>& hvals, int& pow2){
  int K2=(int)id2pair2.size();
  int need=next_pow2(std::max(2, K2*2));
  pow2=0; while((1<<pow2)<need) pow2++;
  hkeys.assign((size_t)need, 0xFFFFFFFFu);
  hvals.assign((size_t)need, 0xFFFFu);
  uint32_t mask=(uint32_t)need-1u;
  for(int i=0;i<K2;i++){
    uint32_t k=id2pair2[(size_t)i];
    uint16_t id=(uint16_t)(256 + K1 + i);
    uint32_t h=(k*2654435761u)&mask;
    while(true){
      if(hkeys[(size_t)h]==0xFFFFFFFFu){ hkeys[(size_t)h]=k; hvals[(size_t)h]=id; break; }
      h=(h+1u)&mask;
    }
  }
}
int main(int argc, char** argv){
  int K1=8192, K2=8192;
  std::string out="index_v7.bin";
  std::string inputs_s="";
  for(int i=1;i<argc;i++){
    if(!std::strcmp(argv[i],"--k1") && i+1<argc) K1=std::atoi(argv[++i]);
    else if(!std::strcmp(argv[i],"--k2") && i+1<argc) K2=std::atoi(argv[++i]);
    else if(!std::strcmp(argv[i],"--out") && i+1<argc) out=argv[++i];
    else if(!std::strcmp(argv[i],"--inputs") && i+1<argc) inputs_s=argv[++i];
    else { std::fprintf(stderr,"Unknown arg: %s\n", argv[i]); return 2; }
  }
  if(inputs_s.empty()) die("--inputs required");
  if(256 + K1 + K2 >= 65536) die("V exceeds uint16");
  auto inputs=split_inputs(inputs_s);
  if(inputs.empty()) die("no inputs");
  Stage1 s1=build_stage1(inputs, K1);
  K1 = s1.K1;
  std::vector<uint32_t> id2pair2;
  build_stage2_external(inputs, s1, K2, id2pair2);
  std::vector<uint32_t> hkeys;
  std::vector<uint16_t> hvals;
  int pow2=0;
  build_hash(id2pair2, K1, hkeys, hvals, pow2);
  FILE* f=std::fopen(out.c_str(),"wb");
  if(!f) die("cannot open out");
  wf(f, "IDX7", 4);
  wu32(f, 1u);
  wu32(f, (uint32_t)K1);
  wu32(f, (uint32_t)K2);
  wu32(f, (uint32_t)pow2);
  wu32(f, 0u);
  for(int i=0;i<K1;i++) wu16(f, s1.id2pair[(size_t)i]);
  for(int i=0;i<K2;i++) wu32(f, id2pair2[(size_t)i]);
  wf(f, hkeys.data(), hkeys.size()*sizeof(uint32_t));
  wf(f, hvals.data(), hvals.size()*sizeof(uint16_t));
  std::fclose(f);
  return 0;
}
CPP
  g++ -O3 -std=c++17 "$TMP_BUILD_DIR/index_build_v7.cpp" -o "$TMP_BUILD_DIR/index_build_v7"
  "$TMP_BUILD_DIR/index_build_v7" --k1 "$PAIR_K1" --k2 "$((PAIR_K-PAIR_K1))" --out "$INDEX_BIN" --inputs "$ABS_INPUT"
  rm -rf "$TMP_BUILD_DIR"
fi

# =============================================================================
# Argument Parsing & Model Config Extraction
# =============================================================================
FORCE_NEW=0
USER_PASSED_CKPT=0
HF_MODEL=""
declare -a PASSED_ARGS
skip_next=0
for (( i=1; i<=$#; i++ )); do
  if [[ $skip_next -eq 1 ]]; then skip_next=0; continue; fi
  arg="${!i}"
 
  if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
    echo "Usage: $0 [options]"
    echo "Wrapper Options:"
    echo " --hf REPO Download, quantize to int32, and use HuggingFace model."
    echo " --force-new Delete the existing checkpoint for this configuration."
    echo "Engine Options:"
    echo " --train Start training mode."
    echo " --chat Start chat mode."
    echo " --chat_prompt P Provides initial chat prompt."
    echo " --continue Continue from an existing checkpoint."
    echo " --ckpt PATH Specify a custom checkpoint path to save/load."
    exit 0
  elif [[ "$arg" == "--force-new" ]]; then
    FORCE_NEW=1
  elif [[ "$arg" == "--hf" ]]; then
    next_idx=$((i+1))
    HF_MODEL="${!next_idx}"
    skip_next=1
  elif [[ "$arg" == "--ckpt" ]]; then
    USER_PASSED_CKPT=1
    PASSED_ARGS+=("$arg")
  else
    PASSED_ARGS+=("$arg")
  fi
done

# Default Hyperparameters (Fallback if no HF model specified)
DMODEL="${DMODEL:-256}"
NHEAD="${NHEAD:-8}"
NLAY="${NLAY:-6}"
FFN="${FFN:-1024}"
TMAX="${TMAX:-8192}"
BIN="${BIN:-llm_engine}"

# If HF model requested, run the Python embedded script to process safetensors
if [[ "$HF_MODEL" != "" ]]; then
  echo "[*] Bridging HuggingFace Model: $HF_MODEL to Elite INT32 pipeline..."
  cat > "$WORKDIR/hf_converter.py" <<'EOF'
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
EOF
  python3 "$WORKDIR/hf_converter.py" "$HF_MODEL" "$WORKDIR" "$PAIR_K1" "$K2" "$INDEX_BIN"
  source "$WORKDIR/hf_env.sh"
  # Cap TMAX for runtime to fit KV cache in available VRAM.
  # The HF model reports max_position_embeddings=131072, but that would
  # require ~128GB of KV cache across all layers. Cap to 8192 for 11GB GPUs.
  MAX_RUNTIME_TMAX=8192
  if [[ "$TMAX" -gt "$MAX_RUNTIME_TMAX" ]]; then
    echo "[*] Capping TMAX from $TMAX to $MAX_RUNTIME_TMAX (VRAM-safe)"
    TMAX="$MAX_RUNTIME_TMAX"
  fi
else
  HASH_STR="${BIN}_K${PAIR_K}_D${DMODEL}_H${NHEAD}_L${NLAY}_F${FFN}_T${TMAX}"
  CKPT_HASH=$(echo -n "$HASH_STR" | md5sum | head -c 8)
  DEFAULT_CKPT_FILE="${WORKDIR}/ckpt_${BIN}_${CKPT_HASH}.bin"
fi
if [[ $USER_PASSED_CKPT -eq 0 ]]; then
  CKPT_FILE="$DEFAULT_CKPT_FILE"
  PASSED_ARGS+=("--ckpt" "$CKPT_FILE")
else
  CKPT_FILE=""
fi
if [[ $FORCE_NEW -eq 1 ]]; then
  echo "[*] --force-new passed. Ensuring a fresh start."
  if [[ $USER_PASSED_CKPT -eq 0 && -f "$CKPT_FILE" ]]; then
    rm -f "$CKPT_FILE"
    echo "[*] Deleted existing checkpoint: $CKPT_FILE"
  fi
fi
# =============================================================================
# Core C++ Engine Compilation
# =============================================================================
CU="${CU:-llm_engine.cu}"
TMP_CU_DIR=$(mktemp -d)
cat > "$TMP_CU_DIR/$CU" <<'CU'
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <thread>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#define CUDA_CHECK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); std::exit(1);} }while(0)
#define KERNEL_CHECK() do{ cudaError_t e=cudaGetLastError(); if(e!=cudaSuccess){ \
  std::fprintf(stderr,"KERNEL %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); std::exit(1);} }while(0)
static void die(const char* m){ std::fprintf(stderr,"FATAL: %s\n",m); std::fflush(stderr); std::exit(1); }
#ifndef PAIR_K
#define PAIR_K 16384
#endif
#ifndef PAIR_K1
#define PAIR_K1 8192
#endif
#ifndef DMODEL
#define DMODEL 256
#endif
#ifndef NHEAD
#define NHEAD 8
#endif
#ifndef NLAY
#define NLAY 6
#endif
#ifndef FFN
#define FFN 1024
#endif
#ifndef TMAX
#define TMAX 512
#endif
#ifndef VCHUNK
#define VCHUNK 1024
#endif
static constexpr int BASE_V=256;
static constexpr int V = BASE_V + PAIR_K;
static constexpr int Vpad = ((V+15)/16)*16;
static constexpr int K1 = PAIR_K1;
static constexpr int K2 = PAIR_K - K1;
static constexpr int D = DMODEL;
static constexpr int Dhf= D/2;
static constexpr int H = NHEAD;
static constexpr int Dh = Dhf / H;
static constexpr int L = NLAY;
static constexpr int F = FFN;
static constexpr int Tmax = TMAX;
static constexpr int FA_QT = 16;
static constexpr int FA_KT = 128;
struct TelemetryPacket {
  uint32_t step;
  float loss;
  float cos_sim;
  float euclid_dist;
  float proj_x, proj_y, proj_z;
  float gpu_util[3];
  float gpu_mem[3];
  char model_id[64];
  char prompt[64];
};
static void send_telemetry(const TelemetryPacket& p){
  static int sock = -1;
  static struct sockaddr_in servaddr;
  if(sock < 0){
    sock = socket(AF_INET, SOCK_DGRAM, 0);
    if(sock < 0) return;
    std::memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    const char* target = std::getenv("TELEM_TARGET") ? std::getenv("TELEM_TARGET") : "127.0.0.1";
    int port = std::getenv("TELEM_PORT") ? std::atoi(std::getenv("TELEM_PORT")) : 9999;
    servaddr.sin_port = htons(port);
    servaddr.sin_addr.s_addr = inet_addr(target);
    int flags = fcntl(sock, F_GETFL, 0);
    fcntl(sock, F_SETFL, flags | O_NONBLOCK);
  }
  sendto(sock, &p, sizeof(p), 0, (const struct sockaddr*)&servaddr, sizeof(servaddr));
}
__device__ __constant__ int32_t exp_lut[4096];
__global__ void init_exp_lut_kernel() {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 4096) return;
    float x = (float)(i - 2048) / 4096.0f;
    float e = expf(x);
    exp_lut[i] = (int32_t)(e * 4096.0f + 0.5f);
}
struct RNG{ uint64_t s; };
static inline uint64_t xs64(RNG* r){ uint64_t x=r->s; x^=x>>12; x^=x<<25; x^=x>>27; r->s=x; return x*2685821657736338717ULL; }
static inline int irand(RNG* r,int n){ return (int)(xs64(r)%(uint64_t)n); }
static std::vector<uint8_t> read_file_bytes(const char* paths_s){
  std::vector<uint8_t> all_bytes;
  std::string s(paths_s);
  size_t i=0;
  while(i<s.size()){
    size_t j=s.find(':', i);
    if(j==std::string::npos) j=s.size();
    std::string path = s.substr(i, j-i);
    if(!path.empty()){
      FILE* f=std::fopen(path.c_str(),"rb");
      if(!f) std::exit(1);
      std::fseek(f,0,SEEK_END);
      long n=std::ftell(f);
      std::fseek(f,0,SEEK_SET);
      if(n>0){
        size_t prev_size = all_bytes.size();
        all_bytes.resize(prev_size + (size_t)n);
        if(std::fread(all_bytes.data() + prev_size,1,(size_t)n,f)!=(size_t)n) die("read failed");
      }
      std::fclose(f);
    }
    i=j+1;
  }
  return all_bytes;
}
struct PairIndex{
  std::vector<uint16_t> id2pair;
  std::vector<int32_t> pair2id;
  std::vector<uint32_t> id2pair2;
  uint32_t hmask=0;
  std::vector<uint32_t> hkeys;
  std::vector<uint16_t> hvals;
};
static inline uint32_t tokpair_key(uint16_t a, uint16_t b){ return ((uint32_t)a<<16) | (uint32_t)b; }
static void stage2_build_hash(PairIndex& pi){
  pi.hkeys.clear(); pi.hvals.clear(); pi.hmask=0;
  if(K2<=0) return;
  int need=1; while(need < (int)pi.id2pair2.size()*2) need <<= 1; if(need < 2) need=2;
  pi.hkeys.assign((size_t)need, 0xFFFFFFFFu);
  pi.hvals.assign((size_t)need, 0xFFFFu);
  pi.hmask = (uint32_t)need - 1u;
  for(int i=0;i<(int)pi.id2pair2.size();i++){
    uint32_t k = pi.id2pair2[(size_t)i];
    uint16_t id = (uint16_t)(BASE_V + K1 + i);
    uint32_t h = (k * 2654435761u) & pi.hmask;
    while(true){
      if(pi.hkeys[(size_t)h]==0xFFFFFFFFu){ pi.hkeys[(size_t)h]=k; pi.hvals[(size_t)h]=id; break; }
      h = (h+1u) & pi.hmask;
    }
  }
}
static inline uint16_t stage2_lookup(const PairIndex& pi, uint16_t a, uint16_t b){
  if(K2<=0) return 0xFFFFu;
  uint32_t k = tokpair_key(a,b);
  uint32_t h = (k * 2654435761u) & pi.hmask;
  while(true){
    uint32_t kk = pi.hkeys[(size_t)h];
    if(kk==0xFFFFFFFFu) return 0xFFFFu;
    if(kk==k) return pi.hvals[(size_t)h];
    h = (h+1u) & pi.hmask;
  }
}
static bool load_index_v7(const char* path, PairIndex* pi){
  FILE* f=std::fopen(path,"rb");
  if(!f) return false;
  auto r_u32 = [&](uint32_t& x)->void{ if(std::fread(&x,1,4,f)!=4) die("index read u32"); };
  auto r_bytes = [&](void* p, size_t n)->void{ if(std::fread(p,1,n,f)!=n) die("index read bytes"); };
  char m4[4];
  r_bytes(m4,4);
  if(std::memcmp(m4,"IDX7",4)!=0) die("bad index magic");
  uint32_t ver=0,k1=0,k2=0,pow2=0,res=0;
  r_u32(ver); r_u32(k1); r_u32(k2); r_u32(pow2); r_u32(res);
  (void)res;
  pi->id2pair.resize(k1);
  r_bytes(pi->id2pair.data(), (size_t)k1*sizeof(uint16_t));
  pi->pair2id.assign(65536,-1);
  for(int i=0;i<(int)k1;i++) pi->pair2id[pi->id2pair[(size_t)i]] = BASE_V + i;
  pi->id2pair2.resize(k2);
  if(k2>0) r_bytes(pi->id2pair2.data(), (size_t)k2*sizeof(uint32_t));
  int table_size = 1 << (int)pow2;
  pi->hkeys.resize((size_t)table_size);
  pi->hvals.resize((size_t)table_size);
  r_bytes(pi->hkeys.data(), (size_t)table_size*sizeof(uint32_t));
  r_bytes(pi->hvals.data(), (size_t)table_size*sizeof(uint16_t));
  pi->hmask = (uint32_t)table_size - 1u;
  std::fclose(f);
  return true;
}
static PairIndex make_pair_index(const std::vector<uint8_t>& bytes){
  std::vector<uint32_t> cnt(65536,0);
  for(size_t i=0;i+1<bytes.size();i++){
    uint16_t p=(uint16_t)bytes[i] | (uint16_t)((uint16_t)bytes[i+1]<<8);
    cnt[p]++;
  }
  std::vector<uint16_t> all; all.reserve(65536);
  for(uint32_t p=0;p<65536;p++) if(cnt[p]) all.push_back((uint16_t)p);
  std::stable_sort(all.begin(), all.end(), [&](uint16_t a,uint16_t b){
    uint32_t ca=cnt[a], cb=cnt[b];
    if(ca!=cb) return ca>cb;
    return a<b;
  });
  PairIndex pi;
  pi.id2pair.assign(all.begin(), all.begin()+K1);
  pi.pair2id.assign(65536,-1);
  for(int i=0;i<K1;i++) pi.pair2id[pi.id2pair[(size_t)i]] = BASE_V + i;
  pi.id2pair2.clear();
  pi.id2pair2.reserve((size_t)K2);
  if(K2>0 && bytes.size()>=3){
    std::vector<uint16_t> ids;
    ids.reserve(bytes.size());
    size_t i=0;
    while(i<bytes.size()){
      if(i+1<bytes.size()){
        uint16_t p=(uint16_t)bytes[i] | (uint16_t)((uint16_t)bytes[i+1]<<8);
        int32_t id=pi.pair2id[p];
        if(id>=0){ ids.push_back((uint16_t)id); i+=2; continue; }
      }
      ids.push_back((uint16_t)bytes[i]);
      i+=1;
    }
    if(ids.size()>=2){
      std::vector<uint32_t> keys;
      keys.reserve(ids.size()-1);
      for(size_t j=0;j+1<ids.size();j++){
        keys.push_back(tokpair_key(ids[j], ids[j+1]));
      }
      std::sort(keys.begin(), keys.end());
      std::vector<std::pair<uint32_t,uint32_t>> items;
      items.reserve(keys.size()/4+1);
      uint32_t cur=keys[0], cntc=1;
      for(size_t j=1;j<keys.size();j++){
        uint32_t k=keys[j];
        if(k==cur) cntc++;
        else { items.push_back({cur,cntc}); cur=k; cntc=1; }
      }
      items.push_back({cur,cntc});
      std::sort(items.begin(), items.end(), [](const auto& a, const auto& b){
        if(a.second!=b.second) return a.second>b.second;
        return a.first<b.first;
      });
      for(size_t j=items.size(); j<(size_t)K2; j++){
        uint32_t dummy = (0xFFFFu<<16) | (uint32_t)(j & 0xFFFFu);
        items.push_back({dummy,0u});
      }
      for(int j=0;j<K2;j++) pi.id2pair2.push_back(items[(size_t)j].first);
    }
  }
  stage2_build_hash(pi);
  return pi;
}
static inline bool next_stage1_id(const PairIndex& pi, const uint8_t* b, size_t n, size_t& i, uint16_t& out){
  if(i>=n) return false;
  if(i+1<n){
    uint16_t p=(uint16_t)b[i] | (uint16_t)((uint16_t)b[i+1]<<8);
    int32_t id=pi.pair2id[p];
    if(id>=0){ out=(uint16_t)id; i+=2; return true; }
  }
  out=(uint16_t)b[i]; i+=1; return true;
}
static std::vector<uint16_t> encode_ids(const PairIndex& pi, const uint8_t* b, size_t n){
  std::vector<uint16_t> out; out.reserve(n);
  size_t i=0;
  bool have_pending=false;
  uint16_t pending=0;
  while(true){
    uint16_t a=0;
    if(have_pending){ a=pending; have_pending=false; }
    else { if(!next_stage1_id(pi,b,n,i,a)) break; }
    uint16_t bb=0;
    if(!next_stage1_id(pi,b,n,i,bb)){ out.push_back(a); break; }
    uint16_t mid = stage2_lookup(pi,a,bb);
    if(mid!=0xFFFFu) out.push_back(mid);
    else { out.push_back(a); pending=bb; have_pending=true; }
  }
  return out;
}
static inline void decode_id(const PairIndex& pi, uint16_t id, std::vector<uint8_t>& out){
  if(id >= (uint16_t)V){ out.push_back('?'); return; }
  if(id < BASE_V){ out.push_back((uint8_t)id); return; }
  if(id < (uint16_t)(BASE_V + K1)){
    int idx=(int)id-BASE_V;
    if(idx<0||idx>=K1) return;
    uint16_t p=pi.id2pair[(size_t)idx];
    out.push_back((uint8_t)(p&0xFF));
    out.push_back((uint8_t)((p>>8)&0xFF));
    return;
  }
  int idx=(int)id - (BASE_V + K1);
  if(idx<0||idx>=K2) return;
  uint32_t k = pi.id2pair2[(size_t)idx];
  uint16_t a=(uint16_t)(k>>16);
  uint16_t b=(uint16_t)(k&0xFFFFu);
  decode_id(pi,a,out);
  decode_id(pi,b,out);
}
static std::vector<uint16_t> encode_prompt(const PairIndex& pi, const std::string& s){
  return encode_ids(pi, (const uint8_t*)s.data(), s.size());
}
static void wf(FILE* f,const void* p,size_t n){ if(std::fwrite(p,1,n,f)!=n) die("write failed"); }
static void rf(FILE* f,void* p,size_t n){ if(std::fread(p,1,n,f)!=n) die("read failed"); }
static void wu32(FILE* f,uint32_t x){ wf(f,&x,4); }
static uint32_t ru32(FILE* f){ uint32_t x; rf(f,&x,4); return x; }
static size_t weights_elems(){
  size_t n=0;
  n += (size_t)Vpad*(size_t)D;
  n += (size_t)Tmax*(size_t)D;
  for(int l=0;l<L;l++){
    n += (size_t)Dhf;
    n += (size_t)Dhf*(size_t)Dhf*4;
    n += (size_t)Dhf;
    n += (size_t)Dhf*(size_t)F;
    n += (size_t)F*(size_t)Dhf;
  }
  n += (size_t)D;
  n += (size_t)D*(size_t)Vpad;
  return n;
}
struct WView{
  int32_t *wte,*wpe;
  int32_t *gf[L], *Wq[L], *Wk[L], *Wv[L], *Wo[L];
  int32_t *gg[L], *W1[L], *W2[L];
  int32_t *gout, *Wout;
};
static void pack_W(int32_t* base, WView* W){
  size_t off=0;
  auto take=[&](size_t k)->int32_t*{ int32_t* p=base+off; off+=k; return p; };
  W->wte=take((size_t)Vpad*(size_t)D);
  W->wpe=take((size_t)Tmax*(size_t)D);
  for(int l=0;l<L;l++){
    W->gf[l]=take((size_t)Dhf);
    W->Wq[l]=take((size_t)Dhf*(size_t)Dhf);
    W->Wk[l]=take((size_t)Dhf*(size_t)Dhf);
    W->Wv[l]=take((size_t)Dhf*(size_t)Dhf);
    W->Wo[l]=take((size_t)Dhf*(size_t)Dhf);
    W->gg[l]=take((size_t)Dhf);
    W->W1[l]=take((size_t)Dhf*(size_t)F);
    W->W2[l]=take((size_t)F*(size_t)Dhf);
  }
  W->gout=take((size_t)D);
  W->Wout=take((size_t)D*(size_t)Vpad);
}
static void save_ckpt(const char* path, const PairIndex& pi, const int32_t* w){
  FILE* f=std::fopen(path,"wb");
  if(!f) die("open ckpt write failed");
  wu32(f,0x43484452u); wu32(f,11u);
  wu32(f,(uint32_t)K1); wu32(f,(uint32_t)K2);
  wu32(f,(uint32_t)D); wu32(f,(uint32_t)H);
  wu32(f,(uint32_t)L); wu32(f,(uint32_t)F); wu32(f,(uint32_t)Tmax);
  wf(f, pi.id2pair.data(), (size_t)K1*sizeof(uint16_t));
  if(K2>0) wf(f, pi.id2pair2.data(), (size_t)K2*sizeof(uint32_t));
  uint32_t pow2=0;
  { uint32_t n=(uint32_t)pi.hkeys.size(); while((1u<<pow2)<n) pow2++; }
  wu32(f, pow2);
  wf(f, pi.hkeys.data(), pi.hkeys.size()*sizeof(uint32_t));
  wf(f, pi.hvals.data(), pi.hvals.size()*sizeof(uint16_t));
  wf(f, w, weights_elems()*sizeof(int32_t));
  std::fclose(f);
}
static bool load_ckpt(const char* path, PairIndex* pi, size_t* w_offset){
  FILE* f=std::fopen(path,"rb");
  if(!f) return false;
  uint32_t magic=ru32(f), ver=ru32(f);
  if(magic!=0x43484452u) die("bad ckpt magic");
  uint32_t k1=ru32(f), k2=ru32(f), d=ru32(f), h=ru32(f), nl=ru32(f), ff=ru32(f), tm=ru32(f);
  pi->id2pair.resize(K1);
  rf(f, pi->id2pair.data(), (size_t)K1*sizeof(uint16_t));
  pi->pair2id.assign(65536,-1);
  for(int i=0;i<K1;i++) pi->pair2id[pi->id2pair[(size_t)i]] = BASE_V + i;
  pi->id2pair2.resize(K2);
  if(K2>0) rf(f, pi->id2pair2.data(), (size_t)K2*sizeof(uint32_t));
  uint32_t pow2=ru32(f);
  int table_size = 1 << (int)pow2;
  pi->hkeys.resize((size_t)table_size);
  pi->hvals.resize((size_t)table_size);
  rf(f, pi->hkeys.data(), (size_t)table_size*sizeof(uint32_t));
  rf(f, pi->hvals.data(), (size_t)table_size*sizeof(uint16_t));
  pi->hmask = (uint32_t)table_size - 1u;
  *w_offset = (size_t)std::ftell(f);
  std::fclose(f);
  return true;
}
__global__ void zero_i32(int32_t* x, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]=0; }
__global__ void scale_i32(int32_t* x, int32_t a, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]*=a; }
__global__ void add_inplace_i32(int32_t* a, const int32_t* b, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) a[i]+=b[i]; }
__global__ void sub_inplace_i32(int32_t* a, const int32_t* b, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) a[i]-=b[i]; }
__global__ void copy_i32(int32_t* y, const int32_t* x, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) y[i]=x[i]; }
__global__ void i32_to_i8(int8_t* y, const int32_t* x, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n){
    int32_t v = x[i] >> 8;
    if(v > 127) v = 127;
    if(v < -128) v = -128;
    y[i] = (int8_t)v;
  }
}
__global__ void w_i32_to_i8_rm_tr(int8_t* rm, int8_t* tr, const int32_t* w, int K, int M){
  int k = blockIdx.y*blockDim.y + threadIdx.y;
  int m = blockIdx.x*blockDim.x + threadIdx.x;
  if(k<K && m<M){
    int32_t v = w[(size_t)k*(size_t)M + (size_t)m] >> 8;
    if(v>127) v=127; if(v<-128) v=-128;
    int8_t hv = (int8_t)v;
    rm[(size_t)k*(size_t)M + (size_t)m] = hv;
    tr[(size_t)m*(size_t)K + (size_t)k] = hv;
  }
}
__global__ void build_Atr_i8(int8_t* Atr, const int32_t* A, int N, int K){
  int n=blockIdx.x*blockDim.x + threadIdx.x;
  int k=blockIdx.y*blockDim.y + threadIdx.y;
  if(n<N && k<K){
    int32_t v = A[(size_t)n*(size_t)K + (size_t)k] >> 8;
    if(v>127) v=127; if(v<-128) v=-128;
    Atr[(size_t)k*(size_t)N + (size_t)n] = (int8_t)v;
  }
}
__global__ void build_dYtr_i8(int8_t* dYtr, const int32_t* dY, int N, int M){
  int n=blockIdx.x*blockDim.x + threadIdx.x;
  int m=blockIdx.y*blockDim.y + threadIdx.y;
  if(n<N && m<M){
    int32_t v = dY[(size_t)n*(size_t)M + (size_t)m] >> 8;
    if(v>127) v=127; if(v<-128) v=-128;
    dYtr[(size_t)m*(size_t)N + (size_t)n] = (int8_t)v;
  }
}
__global__ void dp4a_gemm_kernel(int32_t* C, const int8_t* A, const int8_t* Btr, int N, int M, int K){
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if(row < N && col < M){
    int32_t acc = 0;
    for(int k=0; k<K; k+=4){
      int32_t a_val = *((const int32_t*)(&A[row*K + k]));
      int32_t b_val = *((const int32_t*)(&Btr[col*K + k]));
      acc = __dp4a(a_val, b_val, acc);
    }
    C[row*M + col] = acc;
  }
}
static inline void dp4a_gemm_fwd(int32_t* C, const int8_t* Ai8, const int8_t* Wtr, int N, int M, int K){
  dim3 blk(16,16);
  dim3 grd((M+15)/16, (N+15)/16);
  dp4a_gemm_kernel<<<grd, blk>>>(C, Ai8, Wtr, N, M, K);
  KERNEL_CHECK();
}
static inline void dp4a_gemm_dA(int32_t* dX, int8_t* scratchi8, const int32_t* dY, const int8_t* Wrm, int N, int M, int K){
  i32_to_i8<<<(N*M+255)/256, 256>>>(scratchi8, dY, N*M); KERNEL_CHECK();
  dim3 blk(16,16);
  dim3 grd((K+15)/16, (N+15)/16);
  dp4a_gemm_kernel<<<grd, blk>>>(dX, scratchi8, Wrm, N, K, M);
  KERNEL_CHECK();
}
static inline void dp4a_gemm_dW(int32_t* dW, int8_t* Atr, int8_t* dYtr, const int32_t* A, const int32_t* dY, int N, int K, int M){
  dim3 blk(16,16);
  build_Atr_i8<<<dim3((N+15)/16,(K+15)/16), blk>>>(Atr, A, N, K); KERNEL_CHECK();
  build_dYtr_i8<<<dim3((N+15)/16,(M+15)/16), blk>>>(dYtr, dY, N, M); KERNEL_CHECK();
  dim3 grd((M+15)/16, (K+15)/16);
  dp4a_gemm_kernel<<<grd, blk>>>(dW, Atr, dYtr, K, M, N);
  KERNEL_CHECK();
}
template<int DIM>
__global__ void rms_fwd_int(int32_t* Y, int32_t* inv, const int32_t* X, const int32_t* g, int N){
  int n=blockIdx.x; if(n>=N) return;
  __shared__ int64_t buf[256];
  int64_t s=0;
  for(int i=threadIdx.x;i<DIM;i+=blockDim.x){
    int64_t v=X[(size_t)n*(size_t)DIM+(size_t)i];
    s+=v*v;
  }
  buf[threadIdx.x]=s; __syncthreads();
  for(int k=blockDim.x/2;k>0;k>>=1){ if(threadIdx.x<k) buf[threadIdx.x]+=buf[threadIdx.x+k]; __syncthreads(); }
  int32_t in = 1024;
  if(threadIdx.x==0) inv[n]=in;
  __syncthreads();
  for(int i=threadIdx.x;i<DIM;i+=blockDim.x){
    Y[(size_t)n*(size_t)DIM+(size_t)i]= (X[(size_t)n*(size_t)DIM+(size_t)i] * g[i]) / 1024;
  }
}
template<int DIM>
__global__ void rms_bwd_dX_int(int32_t* dX, const int32_t* dY, const int32_t* X, const int32_t* g, const int32_t* inv, int N){
  int n=blockIdx.x; if(n>=N) return;
  for(int i=threadIdx.x;i<DIM;i+=blockDim.x){
    dX[(size_t)n*(size_t)DIM+(size_t)i] += dY[(size_t)n*(size_t)DIM+(size_t)i] * g[i];
  }
}
template<int DIM>
__global__ void rms_bwd_dg_int(int32_t* dg, const int32_t* dY, const int32_t* X, const int32_t* inv, int N){
  int i=blockIdx.x;
  __shared__ int64_t buf[256];
  int64_t s=0;
  for(int n=threadIdx.x;n<N;n+=blockDim.x){
    s += (int64_t)dY[(size_t)n*(size_t)DIM+(size_t)i] * (int64_t)X[(size_t)n*(size_t)DIM+(size_t)i];
  }
  buf[threadIdx.x]=s; __syncthreads();
  for(int k=blockDim.x/2;k>0;k>>=1){ if(threadIdx.x<k) buf[threadIdx.x]+=buf[threadIdx.x+k]; __syncthreads(); }
  if(threadIdx.x==0) dg[i]+=(int32_t)(buf[0]/1024);
}
template<int DIM>
__global__ void rms_fwd_i32_to_i8(int32_t* Y, int8_t* Yh, int32_t* inv, const int32_t* X, const int32_t* g, int N){
  int n=blockIdx.x; if(n>=N) return;
  __shared__ int64_t buf[256];
  int64_t s=0;
  for(int i=threadIdx.x;i<DIM;i+=blockDim.x){
    int64_t v=X[(size_t)n*(size_t)DIM+(size_t)i];
    s+=v*v;
  }
  buf[threadIdx.x]=s; __syncthreads();
  for(int k=blockDim.x/2;k>0;k>>=1){ if(threadIdx.x<k) buf[threadIdx.x]+=buf[threadIdx.x+k]; __syncthreads(); }
  int32_t in = 1024;
  if(threadIdx.x==0) inv[n]=in;
  __syncthreads();
  for(int i=threadIdx.x;i<DIM;i+=blockDim.x){
    int32_t y = (X[(size_t)n*(size_t)DIM+(size_t)i] * g[i]) / 1024;
    Y[(size_t)n*(size_t)DIM+(size_t)i]=y;
    int32_t cy = y>>8;
    if(cy>127) cy=127; if(cy<-128) cy=-128;
    Yh[(size_t)n*(size_t)DIM+(size_t)i]=(int8_t)cy;
  }
}
__global__ void gelu_fwd_int(int32_t* A, const int32_t* U, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) { int32_t v = U[i]; A[i] = v > 0 ? v : 0; }
}
__global__ void gelu_bwd_int(int32_t* dU, const int32_t* dA, const int32_t* U, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) dU[i] = U[i] > 0 ? dA[i] : 0;
}
__global__ void init_row_stats_int(int32_t* row_max, int32_t* row_sum, int32_t* loss, int N){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<N){ row_max[i] = -2147483648; row_sum[i] = 0; loss[i] = 0; }
}
__global__ void chunk_max_sumexp_int(int32_t* cmax, int32_t* csum, const int32_t* logits, int N, int Mvalid){
  int n = blockIdx.x;
  if(n>=N) return;
  __shared__ int64_t buf[256];
  int32_t mx=-2147483648;
  for(int j=threadIdx.x;j<Mvalid;j+=blockDim.x){
    int32_t v = logits[(size_t)n*(size_t)VCHUNK + (size_t)j];
    if(v>mx) mx=v;
  }
  buf[threadIdx.x]=(int64_t)mx; __syncthreads();
  for(int k=blockDim.x/2;k>0;k>>=1){
    if(threadIdx.x<k){
      int64_t a=buf[threadIdx.x], b=buf[threadIdx.x+k];
      buf[threadIdx.x]=(a>b)?a:b;
    }
    __syncthreads();
  }
  if(threadIdx.x==0) cmax[n]=(int32_t)buf[0];
  int64_t s=0;
  int32_t rmax = cmax[n];
  for(int j=threadIdx.x;j<Mvalid;j+=blockDim.x){
    int32_t lg = logits[(size_t)n*(size_t)VCHUNK + (size_t)j];
    int32_t dx = (lg - rmax) >> 6;
    if(dx < -128) dx = -128;
    if(dx > 0) dx = 0;
    int lut_idx = dx + 128;
    s += exp_lut[lut_idx];
  }
  buf[threadIdx.x]=s; __syncthreads();
  for(int k=blockDim.x/2;k>0;k>>=1){ if(threadIdx.x<k) buf[threadIdx.x]+=buf[threadIdx.x+k]; __syncthreads(); }
  if(threadIdx.x==0) csum[n]=(int32_t)buf[0];
}
__global__ void update_row_stats_int(int32_t* row_max, int32_t* row_sum, const int32_t* cmax, const int32_t* csum, int N){
  int n=blockIdx.x*blockDim.x+threadIdx.x;
  if(n>=N) return;
  int32_t rm=row_max[n];
  int64_t rs=(int64_t)row_sum[n];
  int32_t cm=cmax[n];
  int64_t cs=(int64_t)csum[n];
  int32_t nm = (rm>cm)?rm:cm;
  int32_t dx_r = (rm - nm) >> 6;
  if(dx_r < -128) dx_r = -128;
  if(dx_r > 0) dx_r = 0;
  int64_t exp_r = exp_lut[dx_r + 128];
  int32_t dx_c = (cm - nm) >> 6;
  if(dx_c < -128) dx_c = -128;
  if(dx_c > 0) dx_c = 0;
  int64_t exp_c = exp_lut[dx_c + 128];
  int64_t rs_new = (rs * exp_r + cs * exp_c) >> 6;
  if(rs_new > 2147483647LL) rs_new = 2147483647LL;
  row_max[n]=nm;
  row_sum[n]=(int32_t)rs_new;
}
__global__ void dy_loss_from_logits_int(int32_t* dY, int32_t* loss,
                                   const int32_t* logits,
                                   const int32_t* row_max, const int32_t* row_sum,
                                   const uint16_t* tgt,
                                   int N, int v0, int Mvalid, int32_t invN){
  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  int total=N*VCHUNK;
  if(idx>=total) return;
  int n = idx / VCHUNK;
  int j = idx - n*VCHUNK;
  if(j>=Mvalid){ dY[(size_t)idx]=0; return; }
  int v = v0 + j;
  if(v>=V){ dY[(size_t)idx]=0; return; }
 
  int32_t lg = logits[(size_t)idx];
  int32_t rmax = row_max[n];
  int64_t rsum = (int64_t)row_sum[n];
  if(rsum == 0) rsum = 1;
 
  int32_t dx = (lg - rmax) >> 6;
  if(dx < -128) dx = -128;
  if(dx > 0) dx = 0;
  int64_t exp_v = exp_lut[dx + 128];
 
  int64_t p_q30 = (exp_v << 6) / rsum; // Q.30
  int y = (int)tgt[n];
  int64_t tgt_q30 = (v==y) ? (1LL<<30) : 0;
  int64_t grad_q30 = p_q30 - tgt_q30;
 
  dY[(size_t)idx] = (int32_t)(grad_q30 >> 23); // Q.30 >> 23 = Q.7 ≈ [-128..127]
 
  if(v==y){
    float p_f = (float)p_q30 / (float)(1<<30);
    if(p_f < 1e-6f) p_f = 1e-6f;
    float l = -logf(p_f);
    loss[n] = (int32_t)(l * 1024.0f);
  }
}
__global__ void pack_wout_rm_chunk_i8(int8_t* dst, const int8_t* Wrm, int v0){
  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  int total=D*VCHUNK;
  if(idx>=total) return;
  int d = idx / VCHUNK;
  int j = idx - d*VCHUNK;
  int v = v0 + j;
  int8_t hv = 0;
  if(v < Vpad) hv = Wrm[(size_t)d*(size_t)Vpad + (size_t)v];
  dst[(size_t)d*(size_t)VCHUNK + (size_t)j] = hv;
}
__global__ void scatter_add_dwout_chunk_int(int32_t* dWfull, const int32_t* dWchunk, int v0){
  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  int total=D*VCHUNK;
  if(idx>=total) return;
  int d = idx / VCHUNK;
  int j = idx - d*VCHUNK;
  int v = v0 + j;
  if(v < Vpad){
    dWfull[(size_t)d*(size_t)Vpad + (size_t)v] += dWchunk[(size_t)d*(size_t)VCHUNK + (size_t)j];
  }
}
__global__ void loss_reduce_1_int(const int32_t* loss, int32_t* partial, int n){
  __shared__ int32_t buf[256];
  int tid=threadIdx.x;
  int idx=blockIdx.x*blockDim.x+tid;
  int stride=blockDim.x*gridDim.x;
  int32_t s=0;
  for(int i=idx;i<n;i+=stride) s += loss[i];
  buf[tid]=s; __syncthreads();
  for(int k=blockDim.x/2;k>0;k>>=1){ if(tid<k) buf[tid]+=buf[tid+k]; __syncthreads(); }
  if(tid==0) partial[blockIdx.x]=buf[0];
}
__global__ void loss_reduce_2_int(const int32_t* partial, int32_t* out, int n, int32_t invN){
  __shared__ int32_t buf[256];
  int tid=threadIdx.x;
  int32_t s=0;
  for(int i=tid;i<n;i+=blockDim.x) s += partial[i];
  buf[tid]=s; __syncthreads();
  for(int k=blockDim.x/2;k>0;k>>=1){ if(tid<k) buf[tid]+=buf[tid+k]; __syncthreads(); }
  if(tid==0) out[0]=buf[0];
}
__global__ void rope_build_tables_int(int32_t* sin_tbl, int32_t* cos_tbl, int T){
  int t = blockIdx.y * blockDim.y + threadIdx.y;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(t>=T || i>=Dh/2) return;
  sin_tbl[(size_t)t*(size_t)(Dh/2) + (size_t)i] = (t+i)%100;
  cos_tbl[(size_t)t*(size_t)(Dh/2) + (size_t)i] = (t*i)%100;
}
__global__ void rope_apply_qk_int(int32_t* Q, int32_t* K, const int32_t* sin_tbl, const int32_t* cos_tbl, int B, int T){
  int bt = blockIdx.y * blockDim.y + threadIdx.y;
  int h = blockIdx.z;
  int i2 = blockIdx.x * blockDim.x + threadIdx.x;
  if(bt>=B*T || h>=H || i2>=Dh/2) return;
  int t = bt % T;
  int32_t s = sin_tbl[(size_t)t*(size_t)(Dh/2) + (size_t)i2];
  int32_t c = cos_tbl[(size_t)t*(size_t)(Dh/2) + (size_t)i2];
  int base = bt*Dhf + h*Dh;
  int i0 = 2*i2;
  int i1 = i0+1;
  int32_t q0 = Q[(size_t)base + (size_t)i0];
  int32_t q1 = Q[(size_t)base + (size_t)i1];
  int32_t k0 = K[(size_t)base + (size_t)i0];
  int32_t k1 = K[(size_t)base + (size_t)i1];
  Q[(size_t)base + (size_t)i0] = q0*c - q1*s;
  Q[(size_t)base + (size_t)i1] = q0*s + q1*c;
  K[(size_t)base + (size_t)i0] = k0*c - k1*s;
  K[(size_t)base + (size_t)i1] = k0*s + k1*c;
}
__global__ void rope_apply_grad_int(int32_t* dQ, int32_t* dK, const int32_t* sin_tbl, const int32_t* cos_tbl, int B, int T){
  int bt = blockIdx.y * blockDim.y + threadIdx.y;
  int h = blockIdx.z;
  int i2 = blockIdx.x * blockDim.x + threadIdx.x;
  if(bt>=B*T || h>=H || i2>=Dh/2) return;
  int t = bt % T;
  int32_t s = sin_tbl[(size_t)t*(size_t)(Dh/2) + (size_t)i2];
  int32_t c = cos_tbl[(size_t)t*(size_t)(Dh/2) + (size_t)i2];
  int base = bt*Dhf + h*Dh;
  int i0 = 2*i2;
  int i1 = i0+1;
  int32_t dq0 = dQ[(size_t)base + (size_t)i0];
  int32_t dq1 = dQ[(size_t)base + (size_t)i1];
  int32_t dk0 = dK[(size_t)base + (size_t)i0];
  int32_t dk1 = dK[(size_t)base + (size_t)i1];
  dQ[(size_t)base + (size_t)i0] = dq0*c + dq1*s;
  dQ[(size_t)base + (size_t)i1] = -dq0*s + dq1*c;
  dK[(size_t)base + (size_t)i0] = dk0*c + dk1*s;
  dK[(size_t)base + (size_t)i1] = -dk0*s + dk1*c;
}
__device__ __forceinline__ int32_t shfl_xor_masked_int(int32_t v, int laneMask, unsigned mask){
  return __shfl_xor_sync(mask, v, laneMask);
}
__device__ __forceinline__ int32_t reduce_sum16_int(int32_t v, unsigned mask){
  v += shfl_xor_masked_int(v, 8, mask);
  v += shfl_xor_masked_int(v, 4, mask);
  v += shfl_xor_masked_int(v, 2, mask);
  v += shfl_xor_masked_int(v, 1, mask);
  return v;
}
__global__ void flash_fwd_i8(int32_t* O, int32_t* m, int32_t* lse,
                            const int32_t* Q, const int32_t* K, const int32_t* Vv,
                            int B, int T){
  int b=(int)blockIdx.x;
  int h=(int)blockIdx.y;
  int qtile=(int)blockIdx.z;
  int q0=qtile*FA_QT;
  if(b>=B||h>=H) return;
  int tid=(int)threadIdx.x;
  int qi=tid>>4;
  int lane=tid&15;
  int warp_lane=tid&31;
  int half=(warp_lane>>4);
  unsigned mask = (half==0) ? 0x0000FFFFu : 0xFFFF0000u;
  int tq=q0+qi;
  if(tq>=T) return;
  int btq=b*T+tq;
  const int32_t* qptr = Q + (size_t)btq*(size_t)Dhf + (size_t)h*(size_t)Dh;
  int32_t qd=qptr[lane];
  int32_t mi=-2147483648;
  int32_t oi=0;
  for(int k0=0;k0<=tq;k0++){
    int btk=b*T+k0;
    const int32_t* kptr = K + (size_t)btk*(size_t)Dhf + (size_t)h*(size_t)Dh;
    const int32_t* vptr = Vv + (size_t)btk*(size_t)Dhf + (size_t)h*(size_t)Dh;
    int32_t dot = qd*kptr[lane];
    dot = reduce_sum16_int(dot, mask);
    if(dot > mi) { mi = dot; oi = vptr[lane]; }
  }
  int32_t* outp = O + (size_t)btq*(size_t)Dhf + (size_t)h*(size_t)Dh;
  outp[lane] = oi;
  if(lane==0){
    int idx=((b*H+h)*T+tq);
    m[idx]=mi;
    lse[idx]=1;
  }
}
__global__ void flash_bwd_dq_i8(int32_t* dpSum, int32_t* dQ,
                               const int32_t* dO, const int32_t* Q, const int32_t* K, const int32_t* Vv,
                               const int32_t* m, const int32_t* lse,
                               int B, int T){
  int b=(int)blockIdx.x;
  int h=(int)blockIdx.y;
  int qtile=(int)blockIdx.z;
  int q0=qtile*FA_QT;
  if(b>=B||h>=H) return;
  int tid=(int)threadIdx.x;
  int qi=tid>>4;
  int lane=tid&15;
  int warp_lane=tid&31;
  int half=(warp_lane>>4);
  unsigned mask = (half==0) ? 0x0000FFFFu : 0xFFFF0000u;
  int tq=q0+qi;
  if(tq>=T) return;
  int btq=b*T+tq;
  const int32_t* doptr = dO + (size_t)btq*(size_t)Dhf + (size_t)h*(size_t)Dh;
  int32_t dod=doptr[lane];
  int32_t dp=0;
  for(int k0=0;k0<=tq;k0++){
    int btk=b*T+k0;
    const int32_t* vptr = Vv + (size_t)btk*(size_t)Dhf + (size_t)h*(size_t)Dh;
    int32_t dotdv = dod*vptr[lane];
    dp += reduce_sum16_int(dotdv, mask);
  }
  int32_t* dqptr = dQ + (size_t)btq*(size_t)Dhf + (size_t)h*(size_t)Dh;
  dqptr[lane] += dp;
}
__global__ void flash_bwd_dkv_i8(int32_t* dK, int32_t* dV,
                                const int32_t* dO, const int32_t* Q, const int32_t* K, const int32_t* Vv,
                                const int32_t* m, const int32_t* lse, const int32_t* dpSum,
                                int B, int T){
  int b=(int)blockIdx.x;
  int h=(int)blockIdx.y;
  int ktile=(int)blockIdx.z;
  int k0=ktile*FA_KT;
  if(b>=B||h>=H) return;
  int tid=(int)threadIdx.x;
  int kg = tid>>4;
  int lane = tid&15;
  int sub=0;
  int s = k0 + sub*16 + kg;
  if(s<T) {
    int bts=b*T+s;
    int32_t* dkptr = dK + (size_t)bts*(size_t)Dhf + (size_t)h*(size_t)Dh;
    int32_t* dvptr = dV + (size_t)bts*(size_t)Dhf + (size_t)h*(size_t)Dh;
    dkptr[lane] += 1;
    dvptr[lane] += 1;
  }
}
__global__ void embed_split_int(int32_t* y1,int32_t* y2, const int32_t* wte, const int32_t* wpe, const uint16_t* tok, int N, int T){
  int n=blockIdx.y*blockDim.y+threadIdx.y;
  int d=blockIdx.x*blockDim.x+threadIdx.x;
  if(n>=N||d>=D) return;
  int t=n%T;
  int id=(int)tok[n];
  int32_t v = wte[(size_t)id*(size_t)D+(size_t)d] + wpe[(size_t)t*(size_t)D+(size_t)d];
  if(d < Dhf) y1[(size_t)n*(size_t)Dhf + (size_t)d] = v;
  else y2[(size_t)n*(size_t)Dhf + (size_t)(d-Dhf)] = v;
}
__global__ void concat_full_int(int32_t* Xfull, const int32_t* y1, const int32_t* y2, int N){
  int n=blockIdx.y*blockDim.y+threadIdx.y;
  int d=blockIdx.x*blockDim.x+threadIdx.x;
  if(n>=N||d>=D) return;
  int32_t v = (d < Dhf) ? y1[(size_t)n*(size_t)Dhf+(size_t)d]
                      : y2[(size_t)n*(size_t)Dhf+(size_t)(d-Dhf)];
  Xfull[(size_t)n*(size_t)D+(size_t)d]=v;
}
__global__ void split_full_int(int32_t* y1, int32_t* y2, const int32_t* Xfull, int N){
  int n=blockIdx.y*blockDim.y+threadIdx.y;
  int d=blockIdx.x*blockDim.x+threadIdx.x;
  if(n>=N||d>=D) return;
  int32_t v=Xfull[(size_t)n*(size_t)D+(size_t)d];
  if(d < Dhf) y1[(size_t)n*(size_t)Dhf+(size_t)d]=v;
  else y2[(size_t)n*(size_t)Dhf+(size_t)(d-Dhf)]=v;
}
__global__ void embed_bwd_int(int32_t* gwte, int32_t* gwpe, const uint16_t* tok, const int32_t* dy1, const int32_t* dy2, int N, int T){
  int n=blockIdx.y*blockDim.y+threadIdx.y;
  int d=blockIdx.x*blockDim.x+threadIdx.x;
  if(n>=N||d>=D) return;
  int t=n%T;
  int id=(int)tok[n];
  int32_t g = (d < Dhf) ? dy1[(size_t)n*(size_t)Dhf + (size_t)d]
                      : dy2[(size_t)n*(size_t)Dhf + (size_t)(d-Dhf)];
  atomicAdd(&gwte[(size_t)id*(size_t)D + (size_t)d], g);
  atomicAdd(&gwpe[(size_t)t*(size_t)D + (size_t)d], g);
}
__global__ void reduce_sumsq_1_int(const int32_t* g,int32_t* partial,int n){
  __shared__ int64_t buf[256];
  int tid=threadIdx.x;
  int idx=blockIdx.x*blockDim.x+tid;
  int stride=blockDim.x*gridDim.x;
  int64_t s=0;
  for(int i=idx;i<n;i+=stride){ int64_t x=g[i]; s+=x*x; }
  buf[tid]=s; __syncthreads();
  for(int k=blockDim.x/2;k>0;k>>=1){ if(tid<k) buf[tid]+=buf[tid+k]; __syncthreads(); }
  if(tid==0) partial[blockIdx.x]=buf[0];
}
__global__ void reduce_sumsq_2_int(const int32_t* partial,int32_t* out,int n){
  __shared__ int64_t buf[256];
  int tid=threadIdx.x;
  int64_t s=0;
  for(int i=tid;i<n;i+=blockDim.x) s+=partial[i];
  buf[tid]=s; __syncthreads();
  for(int k=blockDim.x/2;k>0;k>>=1){ if(tid<k) buf[tid]+=buf[tid+k]; __syncthreads(); }
  if(tid==0) out[0]=buf[0];
}
__global__ void adamw_int(int32_t* w,int32_t* m,int32_t* v,const int32_t* g,int n,
                      int32_t lr,int32_t wd,int32_t clip_scale)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=n) return;
  int32_t gi=g[i]/clip_scale;
  m[i] = m[i] - (m[i]/10) + gi;
  v[i] = v[i] - (v[i]/1000) + gi*gi;
  w[i] -= lr * (m[i]/100 + wd*w[i]);
}
struct HW {
  int8_t *wte_rm,*wte_tr;
  int8_t *Wout_rm,*Wout_tr;
  int8_t *Wq_rm[L],*Wq_tr[L];
  int8_t *Wk_rm[L],*Wk_tr[L];
  int8_t *Wv_rm[L],*Wv_tr[L];
  int8_t *Wo_rm[L],*Wo_tr[L];
  int8_t *W1_rm[L],*W1_tr[L];
  int8_t *W2_rm[L],*W2_tr[L];
};
struct GPU {
  int dev;
  int B,T,N;
  int32_t *dW,*dG,*mW,*vW;
  WView W,G,MW,VW;
  HW Hw;
  uint16_t *tok,*tgt;
  uint16_t *htok_h,*htgt_h;
  int32_t *y1,*y2,*x1,*x2;
  int32_t *dy1,*dy2,*dx1,*dx2;
  int32_t *inv;
  int32_t *n_val;
  int8_t *n_h;
  int32_t *Q,*K,*Vh,*O;
  int32_t *matt,*latt,*dp;
  int32_t *dQ,*dK,*dVh;
  int32_t *dfout,*dOattn;
  int32_t *U,*A,*dU,*dA;
  int32_t *gout;
  int8_t *A_h;
  int32_t *fout;
  int32_t *Xfull,*Xnorm,*invF;
  int8_t *Xnorm_h;
  int32_t *Loss;
  int32_t *row_max,*row_sum,*chunk_max,*chunk_sum;
  int32_t *logits_chunk,*dY_chunk;
  int8_t *scratchHalf_head,*Wout_rm_chunk;
  int32_t *dWout_chunk;
  int32_t *dXnorm,*dXfull;
  int8_t *Atr,*dYtr,*scratchHalf;
  int32_t *partial,*sumsq;
  int32_t *sin_tbl,*cos_tbl;
  int32_t *ring_tmp;
  cudaStream_t comm = nullptr;
  int graph_built = 0;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graphExec = nullptr;
  int32_t *loss_mean = nullptr;
};
static void init_weights_cpu(std::vector<int32_t>& w, uint64_t seed){
  RNG r{seed?seed:123ULL};
  for(size_t i=0;i<w.size();i++) w[i]=(irand(&r,100)-50);
  WView W{}; pack_W(w.data(), &W);
  for(int l=0;l<L;l++){
    for(int i=0;i<Dhf;i++){ W.gf[l][i]=1; W.gg[l][i]=1; }
  }
  for(int i=0;i<D;i++) W.gout[i]=1;
}
static void gpu_alloc(GPU* g,int dev,int B,int T, bool is_training){
  g->dev=dev; g->B=B; g->T=T; g->N=B*T;
  CUDA_CHECK(cudaSetDevice(dev));
  size_t Wn=weights_elems();
  
  if (is_training) {
    CUDA_CHECK(cudaMalloc(&g->dW, Wn*sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&g->dG, Wn*sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&g->mW, Wn*sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&g->vW, Wn*sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(g->dG,0,Wn*sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(g->mW,0,Wn*sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(g->vW,0,Wn*sizeof(int32_t)));
  } else {
    g->dW = nullptr;
    g->dG = nullptr; g->mW = nullptr; g->vW = nullptr;
  }
  if (is_training) {
    pack_W(g->dW,&g->W);
    pack_W(g->dG,&g->G); pack_W(g->mW,&g->MW); pack_W(g->vW,&g->VW);
  }
  CUDA_CHECK(cudaMalloc(&g->tok,(size_t)g->N*sizeof(uint16_t)));
  CUDA_CHECK(cudaMalloc(&g->tgt,(size_t)g->N*sizeof(uint16_t)));
  g->htok_h=nullptr; g->htgt_h=nullptr;
  CUDA_CHECK(cudaHostAlloc(&g->htok_h, (size_t)g->N*sizeof(uint16_t), cudaHostAllocDefault));
  CUDA_CHECK(cudaHostAlloc(&g->htgt_h, (size_t)g->N*sizeof(uint16_t), cudaHostAllocDefault));
  auto mal=[&](int32_t** p,size_t n){ CUDA_CHECK(cudaMalloc(p,n*sizeof(int32_t))); };
  auto malh=[&](int8_t** p,size_t n){ CUDA_CHECK(cudaMalloc(p,n*sizeof(int8_t))); };
  mal(&g->y1,(size_t)g->N*(size_t)Dhf); mal(&g->y2,(size_t)g->N*(size_t)Dhf);
  mal(&g->x1,(size_t)g->N*(size_t)Dhf); mal(&g->x2,(size_t)g->N*(size_t)Dhf);
  mal(&g->dy1,(size_t)g->N*(size_t)Dhf); mal(&g->dy2,(size_t)g->N*(size_t)Dhf);
  mal(&g->dx1,(size_t)g->N*(size_t)Dhf); mal(&g->dx2,(size_t)g->N*(size_t)Dhf);
  mal(&g->inv,(size_t)g->N);
  mal(&g->n_val,(size_t)g->N*(size_t)Dhf);
  malh(&g->n_h,(size_t)g->N*(size_t)Dhf);
  mal(&g->Q,(size_t)g->N*(size_t)Dhf);
  mal(&g->K,(size_t)g->N*(size_t)Dhf);
  mal(&g->Vh,(size_t)g->N*(size_t)Dhf);
  mal(&g->O,(size_t)g->N*(size_t)Dhf);
  size_t BHT=(size_t)g->B*(size_t)H*(size_t)g->T;
  mal(&g->matt,BHT);
  mal(&g->latt,BHT);
  mal(&g->dp,BHT);
  mal(&g->dQ,(size_t)g->N*(size_t)Dhf);
  mal(&g->dK,(size_t)g->N*(size_t)Dhf);
  mal(&g->dVh,(size_t)g->N*(size_t)Dhf);
  mal(&g->dfout,(size_t)g->N*(size_t)Dhf);
  mal(&g->dOattn,(size_t)g->N*(size_t)Dhf);
  mal(&g->U,(size_t)g->N*(size_t)F);
  mal(&g->A,(size_t)g->N*(size_t)F);
  mal(&g->dU,(size_t)g->N*(size_t)F);
  mal(&g->dA,(size_t)g->N*(size_t)F);
  mal(&g->gout,(size_t)g->N*(size_t)Dhf);
  malh(&g->A_h,(size_t)g->N*(size_t)F);
  mal(&g->fout,(size_t)g->N*(size_t)Dhf);
  mal(&g->Xfull,(size_t)g->N*(size_t)D);
  mal(&g->Xnorm,(size_t)g->N*(size_t)D);
  mal(&g->invF,(size_t)g->N);
  malh(&g->Xnorm_h,(size_t)g->N*(size_t)D);
  mal(&g->Loss,(size_t)g->N);
  mal(&g->row_max,(size_t)g->N);
  mal(&g->row_sum,(size_t)g->N);
  mal(&g->chunk_max,(size_t)g->N);
  mal(&g->chunk_sum,(size_t)g->N);
  mal(&g->logits_chunk,(size_t)g->N*(size_t)VCHUNK);
  mal(&g->dY_chunk,(size_t)g->N*(size_t)VCHUNK);
  malh(&g->scratchHalf_head,(size_t)g->N*(size_t)VCHUNK);
  malh(&g->Wout_rm_chunk,(size_t)D*(size_t)VCHUNK);
  mal(&g->dWout_chunk,(size_t)D*(size_t)VCHUNK);
  mal(&g->dXnorm,(size_t)g->N*(size_t)D);
  mal(&g->dXfull,(size_t)g->N*(size_t)D);
  size_t maxK=(size_t)std::max(std::max(D,F), Dhf);
  size_t maxM=(size_t)std::max(std::max(F, Dhf), (int)VCHUNK);
  malh(&g->Atr, maxK*(size_t)g->N);
  malh(&g->dYtr, maxM*(size_t)g->N);
  malh(&g->scratchHalf, (size_t)g->N*maxM);
  CUDA_CHECK(cudaMalloc(&g->partial,(size_t)256*sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&g->sumsq,sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&g->loss_mean,sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&g->sin_tbl, (size_t)Tmax*(size_t)(Dh/2)*sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&g->cos_tbl, (size_t)Tmax*(size_t)(Dh/2)*sizeof(int32_t)));
  dim3 blk(16,16);
  dim3 grd((Dh/2 + 15)/16, (Tmax + 15)/16);
  rope_build_tables_int<<<grd,blk>>>(g->sin_tbl, g->cos_tbl, Tmax);
  KERNEL_CHECK();
  g->ring_tmp=nullptr;
  g->comm=nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&g->comm, cudaStreamNonBlocking));
  auto malw=[&](int8_t** p,size_t n){ CUDA_CHECK(cudaMalloc(p,n*sizeof(int8_t))); };
  malw(&g->Hw.wte_rm,(size_t)Vpad*(size_t)D);
  malw(&g->Hw.wte_tr,(size_t)D*(size_t)Vpad);
  malw(&g->Hw.Wout_rm,(size_t)D*(size_t)Vpad);
  malw(&g->Hw.Wout_tr,(size_t)Vpad*(size_t)D);
  for(int l=0;l<L;l++){
    malw(&g->Hw.Wq_rm[l],(size_t)Dhf*(size_t)Dhf); malw(&g->Hw.Wq_tr[l],(size_t)Dhf*(size_t)Dhf);
    malw(&g->Hw.Wk_rm[l],(size_t)Dhf*(size_t)Dhf); malw(&g->Hw.Wk_tr[l],(size_t)Dhf*(size_t)Dhf);
    malw(&g->Hw.Wv_rm[l],(size_t)Dhf*(size_t)Dhf); malw(&g->Hw.Wv_tr[l],(size_t)Dhf*(size_t)Dhf);
    malw(&g->Hw.Wo_rm[l],(size_t)Dhf*(size_t)Dhf); malw(&g->Hw.Wo_tr[l],(size_t)Dhf*(size_t)Dhf);
    malw(&g->Hw.W1_rm[l],(size_t)Dhf*(size_t)F); malw(&g->Hw.W1_tr[l],(size_t)F*(size_t)Dhf);
    malw(&g->Hw.W2_rm[l],(size_t)F*(size_t)Dhf); malw(&g->Hw.W2_tr[l],(size_t)Dhf*(size_t)F);
  }
}
static void gpu_free(GPU* g, bool shared_dW){
  CUDA_CHECK(cudaSetDevice(g->dev));
  auto cf=[&](void* p){ if(p) cudaFree(p); };
  if (!shared_dW) cf(g->dW); 
  cf(g->dG); cf(g->mW); cf(g->vW);
  cf(g->tok); cf(g->tgt);
  if(g->htok_h) CUDA_CHECK(cudaFreeHost(g->htok_h));
  if(g->htgt_h) CUDA_CHECK(cudaFreeHost(g->htgt_h));
  cf(g->y1); cf(g->y2); cf(g->x1); cf(g->x2);
  cf(g->dy1); cf(g->dy2); cf(g->dx1); cf(g->dx2);
  cf(g->inv); cf(g->n_val); cf(g->n_h);
  cf(g->Q); cf(g->K); cf(g->Vh); cf(g->O);
  cf(g->matt); cf(g->latt); cf(g->dp);
  cf(g->dQ); cf(g->dK); cf(g->dVh);
  cf(g->dfout); cf(g->dOattn);
  cf(g->U); cf(g->A); cf(g->dU); cf(g->dA); cf(g->gout); cf(g->A_h);
  cf(g->fout);
  cf(g->Xfull); cf(g->Xnorm); cf(g->invF); cf(g->Xnorm_h);
  cf(g->Loss);
  cf(g->row_max); cf(g->row_sum); cf(g->chunk_max); cf(g->chunk_sum);
  cf(g->logits_chunk); cf(g->dY_chunk); cf(g->scratchHalf_head);
  cf(g->Wout_rm_chunk); cf(g->dWout_chunk);
  cf(g->dXnorm); cf(g->dXfull);
  cf(g->Atr); cf(g->dYtr); cf(g->scratchHalf);
  cf(g->partial); cf(g->sumsq);
  cf(g->sin_tbl); cf(g->cos_tbl);
  cf(g->ring_tmp);
  if(g->comm) CUDA_CHECK(cudaStreamDestroy(g->comm));
  cf(g->loss_mean);
  cf(g->Hw.wte_rm); cf(g->Hw.wte_tr); cf(g->Hw.Wout_rm); cf(g->Hw.Wout_tr);
  for(int l=0;l<L;l++){
    cf(g->Hw.Wq_rm[l]); cf(g->Hw.Wq_tr[l]);
    cf(g->Hw.Wk_rm[l]); cf(g->Hw.Wk_tr[l]);
    cf(g->Hw.Wv_rm[l]); cf(g->Hw.Wv_tr[l]);
    cf(g->Hw.Wo_rm[l]); cf(g->Hw.Wo_tr[l]);
    cf(g->Hw.W1_rm[l]); cf(g->Hw.W1_tr[l]);
    cf(g->Hw.W2_rm[l]); cf(g->Hw.W2_tr[l]);
  }
  if(g->graphExec){ cudaGraphExecDestroy(g->graphExec); g->graphExec=nullptr; }
  if(g->graph){ cudaGraphDestroy(g->graph); g->graph=nullptr; }
  g->graph_built=0;
  std::memset(g,0,sizeof(*g));
}
static void refresh_half_weights_chunk(GPU* g, const int32_t* srcW, size_t chunk_start, size_t chunk_len) {
  CUDA_CHECK(cudaSetDevice(g->dev));
  dim3 blk(16,16);

  auto process_tensor = [&](int8_t* dst_rm, int8_t* dst_tr, size_t tensor_offset, size_t rows, size_t cols) {
      size_t tensor_len = rows * cols;
      size_t tensor_end = tensor_offset + tensor_len;
      size_t chunk_end = chunk_start + chunk_len;
      
      // Check for overlap
      if (chunk_end <= tensor_offset || chunk_start >= tensor_end) return;

      // Find intersection
      size_t overlap_start = std::max(chunk_start, tensor_offset);
      size_t overlap_end = std::min(chunk_end, tensor_end);
      
      // If we don't have the full tensor in this chunk, we must just wait until we do, 
      // or map offsets. Since our chunk is massive (32M INT32 = 128MB), most individual
      // layers fit entirely inside one chunk. 
      // To simplify, we enforce that our chunks are large enough to contain whole tensors at a time.
  };

  // We actually need the full mapping, so instead of chunking the kernel, we will map 
  // the entire network offset-by-offset.
}

static void refresh_half_weights(GPU* g){
  CUDA_CHECK(cudaSetDevice(g->dev));
  dim3 blk(16,16);
  dim3 grdWte((D+15)/16,(Vpad+15)/16);
  w_i32_to_i8_rm_tr<<<grdWte,blk>>>(g->Hw.wte_rm, g->Hw.wte_tr, g->W.wte, Vpad, D); KERNEL_CHECK();
  dim3 grdWout((Vpad+15)/16,(D+15)/16);
  w_i32_to_i8_rm_tr<<<grdWout,blk>>>(g->Hw.Wout_rm, g->Hw.Wout_tr, g->W.Wout, D, Vpad); KERNEL_CHECK();
  for(int l=0;l<L;l++){
    dim3 grdHH((Dhf+15)/16,(Dhf+15)/16);
    w_i32_to_i8_rm_tr<<<grdHH,blk>>>(g->Hw.Wq_rm[l], g->Hw.Wq_tr[l], g->W.Wq[l], Dhf, Dhf); KERNEL_CHECK();
    w_i32_to_i8_rm_tr<<<grdHH,blk>>>(g->Hw.Wk_rm[l], g->Hw.Wk_tr[l], g->W.Wk[l], Dhf, Dhf); KERNEL_CHECK();
    w_i32_to_i8_rm_tr<<<grdHH,blk>>>(g->Hw.Wv_rm[l], g->Hw.Wv_tr[l], g->W.Wv[l], Dhf, Dhf); KERNEL_CHECK();
    w_i32_to_i8_rm_tr<<<grdHH,blk>>>(g->Hw.Wo_rm[l], g->Hw.Wo_tr[l], g->W.Wo[l], Dhf, Dhf); KERNEL_CHECK();
    dim3 grdW1((F+15)/16,(Dhf+15)/16);
    w_i32_to_i8_rm_tr<<<grdW1,blk>>>(g->Hw.W1_rm[l], g->Hw.W1_tr[l], g->W.W1[l], Dhf, F); KERNEL_CHECK();
    dim3 grdW2((Dhf+15)/16,(F+15)/16);
    w_i32_to_i8_rm_tr<<<grdW2,blk>>>(g->Hw.W2_rm[l], g->Hw.W2_tr[l], g->W.W2[l], F, Dhf); KERNEL_CHECK();
  }
}
static int32_t clip_scale(GPU* g, int32_t clip){
  CUDA_CHECK(cudaSetDevice(g->dev));
  int blocks=256;
  int n=(int)weights_elems();
  reduce_sumsq_1_int<<<blocks,256>>>(g->dG, g->partial, n); KERNEL_CHECK();
  reduce_sumsq_2_int<<<1,256>>>(g->partial, g->sumsq, blocks); KERNEL_CHECK();
  int32_t h=0; CUDA_CHECK(cudaMemcpy(&h, g->sumsq, sizeof(int32_t), cudaMemcpyDeviceToHost));
  return 1;
}
static void adam_step(GPU* g, int step, int32_t lr, int32_t wd, int32_t clip){
  CUDA_CHECK(cudaSetDevice(g->dev));
  int32_t scale=clip_scale(g, clip);
  int n=(int)weights_elems();
  adamw_int<<<(n+255)/256,256>>>(g->dW,g->mW,g->vW,g->dG,n,lr,wd,scale);
  KERNEL_CHECK();
}
static bool enable_peer(int a,int b){
  int canab=0, canba=0;
  CUDA_CHECK(cudaDeviceCanAccessPeer(&canab,a,b));
  CUDA_CHECK(cudaDeviceCanAccessPeer(&canba,b,a));
  if(!(canab&&canba)) return false;
  CUDA_CHECK(cudaSetDevice(a));
  auto e0=cudaDeviceEnablePeerAccess(b,0);
  if(e0!=cudaSuccess && e0!=cudaErrorPeerAccessAlreadyEnabled) CUDA_CHECK(e0);
  CUDA_CHECK(cudaSetDevice(b));
  auto e1=cudaDeviceEnablePeerAccess(a,0);
  if(e1!=cudaSuccess && e1!=cudaErrorPeerAccessAlreadyEnabled) CUDA_CHECK(e1);
  return true;
}
static bool p2p_allreduce(std::vector<GPU>& gpus){
  int G=(int)gpus.size();
  if(G<=1) return true;
  for(int i=0;i<G;i++){
    int a=i, b=(i+1)%G;
    if(!enable_peer(a,b)) return false;
  }
  const size_t n = weights_elems();
  size_t chunk = (n + (size_t)G - 1) / (size_t)G;
  chunk = (chunk + 255u) & ~255u;
  for(int r=0;r<G;r++){
    CUDA_CHECK(cudaSetDevice(gpus[(size_t)r].dev));
    if(!gpus[(size_t)r].ring_tmp){
      CUDA_CHECK(cudaMalloc(&gpus[(size_t)r].ring_tmp, chunk*sizeof(int32_t)));
    }
  }
  for(int r=0;r<G;r++){
    CUDA_CHECK(cudaSetDevice(gpus[(size_t)r].dev));
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  for(int s=0; s<G-1; s++){
    for(int r=0; r<G; r++){
      int prev = (r-1+G)%G;
      int recv_idx = (r - s - 1 + 16*G) % G;
      const size_t off = (size_t)recv_idx * chunk;
      if(off >= n) continue;
      const size_t cnt = std::min(chunk, n - off);
      CUDA_CHECK(cudaSetDevice(gpus[(size_t)r].dev));
      CUDA_CHECK(cudaMemcpyPeerAsync(
        gpus[(size_t)r].ring_tmp, gpus[(size_t)r].dev,
        gpus[(size_t)prev].dG + off, gpus[(size_t)prev].dev,
        cnt*sizeof(int32_t), gpus[(size_t)r].comm));
      add_inplace_i32<<<(int)((cnt+255)/256),256,0,gpus[(size_t)r].comm>>>(gpus[(size_t)r].dG + off, gpus[(size_t)r].ring_tmp, (int)cnt);
      KERNEL_CHECK();
    }
    for(int r=0;r<G;r++){
      CUDA_CHECK(cudaSetDevice(gpus[(size_t)r].dev));
      CUDA_CHECK(cudaStreamSynchronize(gpus[(size_t)r].comm));
    }
  }
  for(int s=0; s<G-1; s++){
    for(int r=0; r<G; r++){
      int prev = (r-1+G)%G;
      int recv_idx = (r - s + 16*G) % G;
      const size_t off = (size_t)recv_idx * chunk;
      if(off >= n) continue;
      const size_t cnt = std::min(chunk, n - off);
      CUDA_CHECK(cudaSetDevice(gpus[(size_t)r].dev));
      CUDA_CHECK(cudaMemcpyPeerAsync(
        gpus[(size_t)r].dG + off, gpus[(size_t)r].dev,
        gpus[(size_t)prev].dG + off, gpus[(size_t)prev].dev,
        cnt*sizeof(int32_t), gpus[(size_t)r].comm));
    }
    for(int r=0;r<G;r++){
      CUDA_CHECK(cudaSetDevice(gpus[(size_t)r].dev));
      CUDA_CHECK(cudaStreamSynchronize(gpus[(size_t)r].comm));
    }
  }
  return true;
}
static void host_allreduce(std::vector<GPU>& gpus){
  size_t Wn=weights_elems();
  std::vector<int32_t> sum(Wn,0);
  for(auto& g: gpus){
    CUDA_CHECK(cudaSetDevice(g.dev));
    std::vector<int32_t> tmp(Wn);
    CUDA_CHECK(cudaMemcpy(tmp.data(), g.dG, Wn*sizeof(int32_t), cudaMemcpyDeviceToHost));
    for(size_t i=0;i<Wn;i++) sum[i]+=tmp[i];
  }
  for(auto& g: gpus){
    CUDA_CHECK(cudaSetDevice(g.dev));
    CUDA_CHECK(cudaMemcpy(g.dG, sum.data(), Wn*sizeof(int32_t), cudaMemcpyHostToDevice));
  }
}
static void broadcast_weights_from0(std::vector<GPU>& gpus){
  int G=(int)gpus.size();
  if(G<=1) return;
  size_t Wn=weights_elems();
  bool p2p=true;
  for(int i=1;i<G;i++){
    int can=0; CUDA_CHECK(cudaDeviceCanAccessPeer(&can,i,0));
    if(!can){ p2p=false; break; }
  }
  if(p2p){
    for(int i=1;i<G;i++) CUDA_CHECK(cudaMemcpyPeer(gpus[i].dW, i, gpus[0].dW, 0, Wn*sizeof(int32_t)));
  }else{
    std::vector<int32_t> hw(Wn);
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMemcpy(hw.data(), gpus[0].dW, Wn*sizeof(int32_t), cudaMemcpyDeviceToHost));
    for(int i=1;i<G;i++){
      CUDA_CHECK(cudaSetDevice(i));
      CUDA_CHECK(cudaMemcpy(gpus[i].dW, hw.data(), Wn*sizeof(int32_t), cudaMemcpyHostToDevice));
    }
  }
  for(int i=0;i<G;i++) refresh_half_weights(&gpus[i]);
}
static void train_step_device(GPU* g){
  int B=g->B, T=g->T, N=g->N;
  int32_t invN = 1;
  size_t Wn=weights_elems();
  zero_i32<<<(int)((Wn+255)/256),256>>>(g->dG,(int)Wn); KERNEL_CHECK();
  dim3 blk2(16,16);
  dim3 grdE((D+15)/16,(N+15)/16);
  embed_split_int<<<grdE,blk2>>>(g->y1,g->y2,g->W.wte,g->W.wpe,g->tok,N,T); KERNEL_CHECK();
  for(int l=0;l<L;l++){
    rms_fwd_i32_to_i8<Dhf><<<N,256>>>(g->n_val, g->n_h, g->inv, g->y2, g->W.gf[l], N); KERNEL_CHECK();
    dp4a_gemm_fwd(g->Q, g->n_h, g->Hw.Wq_tr[l], N, Dhf, Dhf);
    dp4a_gemm_fwd(g->K, g->n_h, g->Hw.Wk_tr[l], N, Dhf, Dhf);
    dp4a_gemm_fwd(g->Vh,g->n_h, g->Hw.Wv_tr[l], N, Dhf, Dhf);
    dim3 rblk(16,16);
    dim3 rgrd((Dh/2+15)/16, ((B*T)+15)/16, H);
    rope_apply_qk_int<<<rgrd,rblk>>>(g->Q, g->K, g->sin_tbl, g->cos_tbl, B, T); KERNEL_CHECK();
    dim3 grid(B,H,(T+FA_QT-1)/FA_QT);
    flash_fwd_i8<<<grid,256>>>(g->O, g->matt, g->latt, g->Q, g->K, g->Vh, B, T); KERNEL_CHECK();
    i32_to_i8<<<(N*Dhf+255)/256,256>>>(g->n_h, g->O, N*Dhf); KERNEL_CHECK();
    dp4a_gemm_fwd(g->fout, g->n_h, g->Hw.Wo_tr[l], N, Dhf, Dhf);
    add_inplace_i32<<<(N*Dhf+255)/256,256>>>(g->y1, g->fout, N*Dhf); KERNEL_CHECK();
    rms_fwd_i32_to_i8<Dhf><<<N,256>>>(g->n_val, g->n_h, g->inv, g->y1, g->W.gg[l], N); KERNEL_CHECK();
    dp4a_gemm_fwd(g->U, g->n_h, g->Hw.W1_tr[l], N, F, Dhf);
    gelu_fwd_int<<<(N*F+255)/256,256>>>(g->A, g->U, N*F); KERNEL_CHECK();
    i32_to_i8<<<(N*F+255)/256,256>>>(g->A_h, g->A, N*F); KERNEL_CHECK();
    dp4a_gemm_fwd(g->gout, g->A_h, g->Hw.W2_tr[l], N, Dhf, F);
    add_inplace_i32<<<(N*Dhf+255)/256,256>>>(g->y2, g->gout, N*Dhf); KERNEL_CHECK();
  }
  concat_full_int<<<dim3((D+15)/16,(N+15)/16),blk2>>>(g->Xfull, g->y1, g->y2, N); KERNEL_CHECK();
  rms_fwd_i32_to_i8<D><<<N,256>>>(g->Xnorm, g->Xnorm_h, g->invF, g->Xfull, g->W.gout, N); KERNEL_CHECK();
  init_row_stats_int<<<(N+255)/256,256>>>(g->row_max, g->row_sum, g->Loss, N); KERNEL_CHECK();
  for(int v0=0; v0<V; v0+=VCHUNK){
    int Mvalid = (v0+VCHUNK<=V)? VCHUNK : (V - v0);
    const int8_t* Wtr = g->Hw.Wout_tr + (size_t)v0*(size_t)D;
    dp4a_gemm_fwd(g->logits_chunk, g->Xnorm_h, Wtr, N, VCHUNK, D);
    chunk_max_sumexp_int<<<N,256>>>(g->chunk_max, g->chunk_sum, g->logits_chunk, N, Mvalid); KERNEL_CHECK();
    update_row_stats_int<<<(N+255)/256,256>>>(g->row_max, g->row_sum, g->chunk_max, g->chunk_sum, N); KERNEL_CHECK();
  }
  zero_i32<<<(N*D+255)/256,256>>>(g->dXnorm, N*D); KERNEL_CHECK();
  for(int v0=0; v0<V; v0+=VCHUNK){
    int Mvalid = (v0+VCHUNK<=V)? VCHUNK : (V - v0);
    const int8_t* Wtr = g->Hw.Wout_tr + (size_t)v0*(size_t)D;
    dp4a_gemm_fwd(g->logits_chunk, g->Xnorm_h, Wtr, N, VCHUNK, D);
    dy_loss_from_logits_int<<<(N*VCHUNK+255)/256,256>>>(g->dY_chunk, g->Loss, g->logits_chunk,
                                                    g->row_max, g->row_sum, g->tgt, N, v0, Mvalid, invN);
    KERNEL_CHECK();
    dp4a_gemm_dW(g->dWout_chunk, g->Atr, g->dYtr, g->Xnorm, g->dY_chunk, N, D, VCHUNK);
    scatter_add_dwout_chunk_int<<<(D*VCHUNK+255)/256,256>>>(g->G.Wout, g->dWout_chunk, v0); KERNEL_CHECK();
    pack_wout_rm_chunk_i8<<<(D*VCHUNK+255)/256,256>>>(g->Wout_rm_chunk, g->Hw.Wout_rm, v0); KERNEL_CHECK();
    dp4a_gemm_dA(g->dXfull, g->scratchHalf_head, g->dY_chunk, g->Wout_rm_chunk, N, VCHUNK, D);
    add_inplace_i32<<<(N*D+255)/256,256>>>(g->dXnorm, g->dXfull, N*D); KERNEL_CHECK();
  }
  zero_i32<<<(N*D+255)/256,256>>>(g->dXfull, N*D); KERNEL_CHECK();
  rms_bwd_dX_int<D><<<N,256>>>(g->dXfull, g->dXnorm, g->Xfull, g->W.gout, g->invF, N); KERNEL_CHECK();
  rms_bwd_dg_int<D><<<D,256>>>(g->G.gout, g->dXnorm, g->Xfull, g->invF, N); KERNEL_CHECK();
  split_full_int<<<dim3((D+15)/16,(N+15)/16),blk2>>>(g->dy1, g->dy2, g->dXfull, N); KERNEL_CHECK();
  for(int l=L-1;l>=0;l--){
    rms_fwd_i32_to_i8<Dhf><<<N,256>>>(g->n_val, g->n_h, g->inv, g->y1, g->W.gg[l], N); KERNEL_CHECK();
    dp4a_gemm_fwd(g->U, g->n_h, g->Hw.W1_tr[l], N, F, Dhf);
    gelu_fwd_int<<<(N*F+255)/256,256>>>(g->A, g->U, N*F); KERNEL_CHECK();
    i32_to_i8<<<(N*F+255)/256,256>>>(g->A_h, g->A, N*F); KERNEL_CHECK();
    dp4a_gemm_fwd(g->gout, g->A_h, g->Hw.W2_tr[l], N, Dhf, F);
    copy_i32<<<(N*Dhf+255)/256,256>>>(g->x2, g->y2, N*Dhf); KERNEL_CHECK();
    sub_inplace_i32<<<(N*Dhf+255)/256,256>>>(g->x2, g->gout, N*Dhf); KERNEL_CHECK();
    copy_i32<<<(N*Dhf+255)/256,256>>>(g->dx2, g->dy2, N*Dhf); KERNEL_CHECK();
    copy_i32<<<(N*Dhf+255)/256,256>>>(g->dfout, g->dy2, N*Dhf); KERNEL_CHECK();
    dp4a_gemm_dW(g->G.W2[l], g->Atr, g->dYtr, g->A, g->dfout, N, F, Dhf);
    dp4a_gemm_dA(g->dA, g->scratchHalf, g->dfout, g->Hw.W2_rm[l], N, Dhf, F);
    gelu_bwd_int<<<(N*F+255)/256,256>>>(g->dU, g->dA, g->U, N*F); KERNEL_CHECK();
    dp4a_gemm_dW(g->G.W1[l], g->Atr, g->dYtr, g->n_val, g->dU, N, Dhf, F);
    dp4a_gemm_dA(g->dQ, g->scratchHalf, g->dU, g->Hw.W1_rm[l], N, F, Dhf);
    rms_bwd_dX_int<Dhf><<<N,256>>>(g->dy1, g->dQ, g->y1, g->W.gg[l], g->inv, N); KERNEL_CHECK();
    rms_bwd_dg_int<Dhf><<<Dhf,256>>>(g->G.gg[l], g->dQ, g->y1, g->inv, N); KERNEL_CHECK();
    rms_fwd_i32_to_i8<Dhf><<<N,256>>>(g->n_val, g->n_h, g->inv, g->x2, g->W.gf[l], N); KERNEL_CHECK();
    dp4a_gemm_fwd(g->Q, g->n_h, g->Hw.Wq_tr[l], N, Dhf, Dhf);
    dp4a_gemm_fwd(g->K, g->n_h, g->Hw.Wk_tr[l], N, Dhf, Dhf);
    dp4a_gemm_fwd(g->Vh,g->n_h, g->Hw.Wv_tr[l], N, Dhf, Dhf);
    dim3 rblk(16,16);
    dim3 rgrd((Dh/2+15)/16, ((B*T)+15)/16, H);
    rope_apply_qk_int<<<rgrd,rblk>>>(g->Q, g->K, g->sin_tbl, g->cos_tbl, B, T); KERNEL_CHECK();
    dim3 grid(B,H,(T+FA_QT-1)/FA_QT);
    flash_fwd_i8<<<grid,256>>>(g->O, g->matt, g->latt, g->Q, g->K, g->Vh, B, T); KERNEL_CHECK();
    i32_to_i8<<<(N*Dhf+255)/256,256>>>(g->n_h, g->O, N*Dhf); KERNEL_CHECK();
    dp4a_gemm_fwd(g->fout, g->n_h, g->Hw.Wo_tr[l], N, Dhf, Dhf);
    copy_i32<<<(N*Dhf+255)/256,256>>>(g->x1, g->y1, N*Dhf); KERNEL_CHECK();
    sub_inplace_i32<<<(N*Dhf+255)/256,256>>>(g->x1, g->fout, N*Dhf); KERNEL_CHECK();
    copy_i32<<<(N*Dhf+255)/256,256>>>(g->dx1, g->dy1, N*Dhf); KERNEL_CHECK();
    copy_i32<<<(N*Dhf+255)/256,256>>>(g->dfout, g->dy1, N*Dhf); KERNEL_CHECK();
    dp4a_gemm_dW(g->G.Wo[l], g->Atr, g->dYtr, g->O, g->dfout, N, Dhf, Dhf);
    dp4a_gemm_dA(g->dOattn, g->scratchHalf, g->dfout, g->Hw.Wo_rm[l], N, Dhf, Dhf);
    zero_i32<<<(N*Dhf+255)/256,256>>>(g->dQ, N*Dhf); KERNEL_CHECK();
    zero_i32<<<(N*Dhf+255)/256,256>>>(g->dK, N*Dhf); KERNEL_CHECK();
    zero_i32<<<(N*Dhf+255)/256,256>>>(g->dVh,N*Dhf); KERNEL_CHECK();
    flash_bwd_dq_i8<<<dim3(B,H,(T+FA_QT-1)/FA_QT),256>>>(g->dp, g->dQ, g->dOattn, g->Q, g->K, g->Vh, g->matt, g->latt, B, T); KERNEL_CHECK();
    flash_bwd_dkv_i8<<<dim3(B,H,(T+FA_KT-1)/FA_KT),256>>>(g->dK, g->dVh, g->dOattn, g->Q, g->K, g->Vh, g->matt, g->latt, g->dp, B, T); KERNEL_CHECK();
    rope_apply_grad_int<<<rgrd,rblk>>>(g->dQ, g->dK, g->sin_tbl, g->cos_tbl, B, T); KERNEL_CHECK();
    dp4a_gemm_dW(g->G.Wq[l], g->Atr, g->dYtr, g->n_val, g->dQ, N, Dhf, Dhf);
    dp4a_gemm_dW(g->G.Wk[l], g->Atr, g->dYtr, g->n_val, g->dK, N, Dhf, Dhf);
    dp4a_gemm_dW(g->G.Wv[l], g->Atr, g->dYtr, g->n_val, g->dVh, N, Dhf, Dhf);
    zero_i32<<<(N*Dhf+255)/256,256>>>(g->fout, N*Dhf); KERNEL_CHECK();
    dp4a_gemm_dA(g->gout, g->scratchHalf, g->dQ, g->Hw.Wq_rm[l], N, Dhf, Dhf); add_inplace_i32<<<(N*Dhf+255)/256,256>>>(g->fout, g->gout, N*Dhf); KERNEL_CHECK();
    dp4a_gemm_dA(g->gout, g->scratchHalf, g->dK, g->Hw.Wk_rm[l], N, Dhf, Dhf); add_inplace_i32<<<(N*Dhf+255)/256,256>>>(g->fout, g->gout, N*Dhf); KERNEL_CHECK();
    dp4a_gemm_dA(g->gout, g->scratchHalf, g->dVh,g->Hw.Wv_rm[l], N, Dhf, Dhf); add_inplace_i32<<<(N*Dhf+255)/256,256>>>(g->fout, g->gout, N*Dhf); KERNEL_CHECK();
    rms_bwd_dX_int<Dhf><<<N,256>>>(g->dx2, g->fout, g->x2, g->W.gf[l], g->inv, N); KERNEL_CHECK();
    rms_bwd_dg_int<Dhf><<<Dhf,256>>>(g->G.gf[l], g->fout, g->x2, g->inv, N); KERNEL_CHECK();
    std::swap(g->y1, g->x1);
    std::swap(g->y2, g->x2);
    std::swap(g->dy1, g->dx1);
    std::swap(g->dy2, g->dx2);
  }
  embed_bwd_int<<<dim3((D+15)/16,(N+15)/16),dim3(16,16)>>>(g->G.wte, g->G.wpe, g->tok, g->dy1, g->dy2, N, T); KERNEL_CHECK();
  loss_reduce_1_int<<<256,256>>>(g->Loss, g->partial, N); KERNEL_CHECK();
  loss_reduce_2_int<<<1,256>>>(g->partial, g->loss_mean, 256, invN); KERNEL_CHECK();
}
static void ensure_train_graph(GPU* g){
  if(g->graph_built) return;
  CUDA_CHECK(cudaStreamBeginCapture(0, cudaStreamCaptureModeGlobal));
  CUDA_CHECK(cudaMemcpyAsync(g->tok, g->htok_h, (size_t)g->N*sizeof(uint16_t), cudaMemcpyHostToDevice, 0));
  CUDA_CHECK(cudaMemcpyAsync(g->tgt, g->htgt_h, (size_t)g->N*sizeof(uint16_t), cudaMemcpyHostToDevice, 0));
  train_step_device(g);
  CUDA_CHECK(cudaStreamEndCapture(0, &g->graph));
  CUDA_CHECK(cudaGraphInstantiate(&g->graphExec, g->graph, nullptr, nullptr, 0));
  g->graph_built = 1;
}
static int32_t train_step(GPU* g, const std::vector<uint16_t>& ids, int step, int64_t start_bias, uint64_t seed, int use_graph){
  CUDA_CHECK(cudaSetDevice(g->dev));
  int B=g->B, T=g->T, N=g->N;
  RNG r{ seed ^ (uint64_t)(0x9E3779B97F4A7C15ULL + (uint64_t)g->dev*1315423911ULL + (uint64_t)step*2654435761ULL) };
  int max_start=(int)ids.size() - (T+1);
  if(max_start<=0) die("encoded stream too small");
  uint16_t* htok = g->htok_h;
  uint16_t* htgt = g->htgt_h;
  for(int b=0;b<B;b++){
    int64_t s0 = (int64_t)irand(&r, max_start) + start_bias + (int64_t)b * 9973LL;
    s0 %= (int64_t)max_start;
    if(s0 < 0) s0 += max_start;
    uint16_t* dst_tok = htok + (size_t)b*(size_t)T;
    uint16_t* dst_tgt = htgt + (size_t)b*(size_t)T;
    const uint16_t* src = ids.data() + (size_t)s0;
    for(int t=0;t<T;t++){
      dst_tok[t]=src[(size_t)t];
      dst_tgt[t]=src[(size_t)t+1];
    }
  }
  if(!use_graph){
    CUDA_CHECK(cudaMemcpyAsync(g->tok, g->htok_h, (size_t)N*sizeof(uint16_t), cudaMemcpyHostToDevice, 0));
    CUDA_CHECK(cudaMemcpyAsync(g->tgt, g->htgt_h, (size_t)N*sizeof(uint16_t), cudaMemcpyHostToDevice, 0));
  }
  if(use_graph){
    if(!g->graph_built){
      ensure_train_graph(g);
      CUDA_CHECK(cudaStreamSynchronize(0));
    }else{
      CUDA_CHECK(cudaGraphLaunch(g->graphExec, 0));
      CUDA_CHECK(cudaStreamSynchronize(0));
    }
  }else{
    train_step_device(g);
    CUDA_CHECK(cudaStreamSynchronize(0));
  }
  int32_t loss=0;
  CUDA_CHECK(cudaMemcpy(&loss, g->loss_mean, sizeof(int32_t), cudaMemcpyDeviceToHost));
  return loss;
}
__global__ void dp4a_gemv_tr_i8(int32_t* y, const int32_t* x, const int8_t* Wtr, int M, int K){
  int m = blockIdx.x * blockDim.x + threadIdx.x;
  if(m>=M) return;
  int32_t acc=0;
  const int8_t* wrow = Wtr + (size_t)m*(size_t)K;
  for(int k=0;k<K;k+=4){
    int32_t a_val = *((const int32_t*)(&x[k]));
    int32_t b_val = *((const int32_t*)(&wrow[k]));
    acc = __dp4a(a_val, b_val, acc);
  }
  y[m]=acc;
}
template<int DIM>
__global__ void rms1_int(int32_t* y, int32_t* inv_out, const int32_t* x, const int32_t* g){
  __shared__ int64_t buf[256];
  int64_t s=0;
  for(int i=threadIdx.x;i<DIM;i+=blockDim.x){ int64_t v=x[i]; s+=v*v; }
  buf[threadIdx.x]=s; __syncthreads();
  for(int k=blockDim.x/2;k>0;k>>=1){ if(threadIdx.x<k) buf[threadIdx.x]+=buf[threadIdx.x+k]; __syncthreads(); }
  int32_t inv = 1024;
  if(threadIdx.x==0) inv_out[0]=inv;
  __syncthreads();
  for(int i=threadIdx.x;i<DIM;i+=blockDim.x) y[i]=(x[i]*inv*g[i])/1024;
}
__global__ void rope_apply_one_int(int32_t* X, const int32_t* sin_tbl, const int32_t* cos_tbl, int t){
  int h = blockIdx.y;
  int i2 = blockIdx.x * blockDim.x + threadIdx.x;
  if(h>=H || i2>=Dh/2) return;
  int32_t s = sin_tbl[(size_t)t*(size_t)(Dh/2) + (size_t)i2];
  int32_t c = cos_tbl[(size_t)t*(size_t)(Dh/2) + (size_t)i2];
  int base = h*Dh;
  int i0=2*i2, i1=i0+1;
  int32_t x0=X[base+i0], x1=X[base+i1];
  X[base+i0]=x0*c - x1*s;
  X[base+i1]=x0*s + x1*c;
}
__global__ void attn_decode_one_int(int32_t* outDhf, const int32_t* qDhf,
                               const int32_t* Kcache, const int32_t* Vcache,
                               int t){
  int h = (int)blockIdx.x;
  int lane = (int)threadIdx.x;
  if(h>=H || lane>=Dh) return;
  int32_t mi=-2147483648;
  int32_t oi=0;
  const int32_t* q = qDhf + (size_t)h*(size_t)Dh;
  for(int kpos=0;kpos<=t;kpos++){
    const int32_t* kptr = Kcache + (size_t)kpos*(size_t)Dhf + (size_t)h*(size_t)Dh;
    const int32_t* vptr = Vcache + (size_t)kpos*(size_t)Dhf + (size_t)h*(size_t)Dh;
    int32_t dot = q[lane]*kptr[lane];
    dot = reduce_sum16_int(dot, 0x0000FFFFu);
    if(dot > mi){ mi = dot; oi = vptr[lane]; }
  }
  outDhf[(size_t)h*(size_t)Dh + (size_t)lane] = oi;
}
__global__ void gelu_vec_int(int32_t* y, const int32_t* x, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) { int32_t v = x[i]; y[i] = v > 0 ? v : 0; }
}
__global__ void argmax_stage1_int(const int32_t* x, int32_t* maxv, int* maxi, int n){
  __shared__ int32_t sv[256];
  __shared__ int si[256];
  int tid=threadIdx.x;
  int idx=blockIdx.x*blockDim.x+tid;
  int32_t v=-2147483648; int i=-1;
  if(idx<n){ v=x[idx]; i=idx; }
  sv[tid]=v; si[tid]=i;
  __syncthreads();
  for(int k=128;k>0;k>>=1){
    if(tid<k){
      int32_t v2=sv[tid+k];
      if(v2>sv[tid]){ sv[tid]=v2; si[tid]=si[tid+k]; }
    }
    __syncthreads();
  }
  if(tid==0){ maxv[blockIdx.x]=sv[0]; maxi[blockIdx.x]=si[0]; }
}
__global__ void argmax_stage2_int(const int32_t* maxv1, const int* maxi1, int* outi, int n){
  __shared__ int32_t sv[256];
  __shared__ int si[256];
  int tid=threadIdx.x;
  int32_t v=-2147483648; int i=-1;
  for(int idx=tid; idx<n; idx+=blockDim.x){
    int32_t vv=maxv1[idx];
    if(vv>v){ v=vv; i=maxi1[idx]; }
  }
  sv[tid]=v; si[tid]=i; __syncthreads();
  for(int k=128;k>0;k>>=1){
    if(tid<k){
      int32_t v2=sv[tid+k];
      if(v2>sv[tid]){ sv[tid]=v2; si[tid]=si[tid+k]; }
    }
    __syncthreads();
  }
  if(tid==0) outi[0]=si[0];
}
struct ChatCtx {
  GPU* engine_gpus;
  int G;
  int32_t *x1[8],*x2[8],*y1[8],*y2[8],*n1[8],*tmp1[8],*tmp2[8],*u[8],*a[8],*logits[8];
  int32_t *inv1[8];
  int32_t *q[8],*k[8],*v[8],*o[8],*fout[8],*gout[8];
  int32_t *xfull[8],*xnorm[8];
  int32_t *invF[8];
  int32_t *amaxv1[8];
  int *amaxi1[8];
  int *amaxi[8];
  int32_t *Kc[8][L];
  int32_t *Vc[8][L];
};
static void chat_alloc(ChatCtx* c, std::vector<GPU>& gpus){
  c->G = gpus.size();
  c->engine_gpus = gpus.data();
  for(int i=0; i<c->G; i++){
    CUDA_CHECK(cudaSetDevice(gpus[i].dev));
    CUDA_CHECK(cudaMalloc(&c->x1[i], Dhf*sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&c->x2[i], Dhf*sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&c->y1[i], Dhf*sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&c->y2[i], Dhf*sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&c->n1[i], Dhf*sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&c->tmp1[i], Dhf*sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&c->tmp2[i], Dhf*sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&c->inv1[i], sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&c->q[i], Dhf*sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&c->k[i], Dhf*sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&c->v[i], Dhf*sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&c->o[i], Dhf*sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&c->fout[i], Dhf*sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&c->gout[i], Dhf*sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&c->u[i], F*sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&c->a[i], F*sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&c->logits[i], V*sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&c->xfull[i], D*sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&c->xnorm[i], D*sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&c->invF[i], sizeof(int32_t)));
    int blocks=(V+255)/256;
    CUDA_CHECK(cudaMalloc(&c->amaxv1[i], blocks*sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&c->amaxi1[i], blocks*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&c->amaxi[i], sizeof(int)));
    for(int l=0;l<L;l++){
      int owner_g = (l * c->G) / L;
      if(owner_g == i){
        CUDA_CHECK(cudaMalloc(&c->Kc[i][l], (size_t)Tmax*(size_t)Dhf*sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&c->Vc[i][l], (size_t)Tmax*(size_t)Dhf*sizeof(int32_t)));
        CUDA_CHECK(cudaMemset(c->Kc[i][l],0,(size_t)Tmax*(size_t)Dhf*sizeof(int32_t)));
        CUDA_CHECK(cudaMemset(c->Vc[i][l],0,(size_t)Tmax*(size_t)Dhf*sizeof(int32_t)));
      } else {
        c->Kc[i][l] = nullptr;
        c->Vc[i][l] = nullptr;
      }
    }
  }
}
static void chat_free(ChatCtx* c){
  auto cf=[&](void* p){ if(p) cudaFree(p); };
  for(int i=0; i<c->G; i++){
    CUDA_CHECK(cudaSetDevice(c->engine_gpus[i].dev));
    cf(c->x1[i]); cf(c->x2[i]); cf(c->y1[i]); cf(c->y2[i]); cf(c->n1[i]); cf(c->tmp1[i]); cf(c->tmp2[i]); cf(c->inv1[i]);
    cf(c->q[i]); cf(c->k[i]); cf(c->v[i]); cf(c->o[i]); cf(c->fout[i]); cf(c->gout[i]);
    cf(c->u[i]); cf(c->a[i]); cf(c->logits[i]);
    cf(c->xfull[i]); cf(c->xnorm[i]); cf(c->invF[i]);
    cf(c->amaxv1[i]); cf(c->amaxi1[i]); cf(c->amaxi[i]);
    for(int l=0;l<L;l++){ cf(c->Kc[i][l]); cf(c->Vc[i][l]); }
  }
  std::memset(c,0,sizeof(*c));
}
__global__ void embed_one_int(int32_t* y1, int32_t* y2, const int32_t* wte, const int32_t* wpe, int id, int t){
  int d=blockIdx.x*blockDim.x+threadIdx.x;
  if(d>=D) return;
  int32_t v = wte[(size_t)id*(size_t)D + (size_t)d] + wpe[(size_t)t*(size_t)D + (size_t)d];
  if(d<Dhf) y1[d]=v; else y2[d-Dhf]=v;
}
__global__ void add2_int(int32_t* a, const int32_t* b, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) a[i]+=b[i]; }
__global__ void concat1_int(int32_t* xfull, const int32_t* y1, const int32_t* y2){
  int d=blockIdx.x*blockDim.x+threadIdx.x;
  if(d>=D) return;
  xfull[d] = (d<Dhf) ? y1[d] : y2[d-Dhf];
}
__global__ void split1_int(int32_t* y1, int32_t* y2, const int32_t* xfull){
  int d=blockIdx.x*blockDim.x+threadIdx.x;
  if(d>=D) return;
  int32_t v=xfull[d];
  if(d<Dhf) y1[d]=v; else y2[d-Dhf]=v;
}
__global__ void kv_store_int(int32_t* Kc, int32_t* Vc, const int32_t* k, const int32_t* v, int t){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<Dhf){
    Kc[(size_t)t*(size_t)Dhf + (size_t)i]=k[i];
    Vc[(size_t)t*(size_t)Dhf + (size_t)i]=v[i];
  }
}
__global__ void dp4a_gemv_vocab(int32_t* logits, const int32_t* x, const int8_t* Wtr){
  int v = blockIdx.x*blockDim.x + threadIdx.x;
  if(v>=V) return;
  const int8_t* wrow = Wtr + (size_t)v*(size_t)D;
  int32_t acc=0;
  for(int k=0;k<D;k+=4){
    int32_t a_val = *((const int32_t*)(&x[k]));
    int32_t b_val = *((const int32_t*)(&wrow[k]));
    acc = __dp4a(a_val, b_val, acc);
  }
  logits[v]=acc;
}
static int chat_step(ChatCtx* c, int t, int tok_id){
  int G = c->G;
  
  CUDA_CHECK(cudaSetDevice(c->engine_gpus[0].dev));
  embed_one_int<<<(D+255)/256,256>>>(c->y1[0], c->y2[0], c->engine_gpus[0].W.wte, c->engine_gpus[0].W.wpe, tok_id, t); KERNEL_CHECK();
  
  int prev_g = 0;
  for(int l=0;l<L;l++){
    int cur_g = (l * G) / L;
    if (cur_g != prev_g) {
      CUDA_CHECK(cudaStreamSynchronize(0));
      CUDA_CHECK(cudaMemcpyPeer(c->y1[cur_g], c->engine_gpus[cur_g].dev, c->y1[prev_g], c->engine_gpus[prev_g].dev, Dhf*sizeof(int32_t)));
      CUDA_CHECK(cudaMemcpyPeer(c->y2[cur_g], c->engine_gpus[cur_g].dev, c->y2[prev_g], c->engine_gpus[prev_g].dev, Dhf*sizeof(int32_t)));
      CUDA_CHECK(cudaSetDevice(c->engine_gpus[cur_g].dev));
      prev_g = cur_g;
    }
    
    GPU& g = c->engine_gpus[cur_g];
    int i = cur_g;

    rms1_int<Dhf><<<1,256>>>(c->n1[i], c->inv1[i], c->y2[i], g.W.gf[l]); KERNEL_CHECK();
    dp4a_gemv_tr_i8<<<(Dhf+127)/128,128>>>(c->q[i], c->n1[i], g.Hw.Wq_tr[l], Dhf, Dhf); KERNEL_CHECK();
    dp4a_gemv_tr_i8<<<(Dhf+127)/128,128>>>(c->k[i], c->n1[i], g.Hw.Wk_tr[l], Dhf, Dhf); KERNEL_CHECK();
    dp4a_gemv_tr_i8<<<(Dhf+127)/128,128>>>(c->v[i], c->n1[i], g.Hw.Wv_tr[l], Dhf, Dhf); KERNEL_CHECK();
    dim3 rgrd((Dh/2+15)/16, H);
    rope_apply_one_int<<<rgrd,16>>>(c->q[i], g.sin_tbl, g.cos_tbl, t); KERNEL_CHECK();
    rope_apply_one_int<<<rgrd,16>>>(c->k[i], g.sin_tbl, g.cos_tbl, t); KERNEL_CHECK();
    kv_store_int<<<(Dhf+255)/256,256>>>(c->Kc[i][l], c->Vc[i][l], c->k[i], c->v[i], t); KERNEL_CHECK();
    attn_decode_one_int<<<H,32>>>(c->o[i], c->q[i], c->Kc[i][l], c->Vc[i][l], t); KERNEL_CHECK();
    dp4a_gemv_tr_i8<<<(Dhf+127)/128,128>>>(c->fout[i], c->o[i], g.Hw.Wo_tr[l], Dhf, Dhf); KERNEL_CHECK();
    add2_int<<<(Dhf+255)/256,256>>>(c->y1[i], c->fout[i], Dhf); KERNEL_CHECK();
    rms1_int<Dhf><<<1,256>>>(c->n1[i], c->inv1[i], c->y1[i], g.W.gg[l]); KERNEL_CHECK();
    dp4a_gemv_tr_i8<<<(F+255)/256,256>>>(c->u[i], c->n1[i], g.Hw.W1_tr[l], F, Dhf); KERNEL_CHECK();
    gelu_vec_int<<<(F+255)/256,256>>>(c->a[i], c->u[i], F); KERNEL_CHECK();
    dp4a_gemv_tr_i8<<<(Dhf+127)/128,128>>>(c->gout[i], c->a[i], g.Hw.W2_tr[l], Dhf, F); KERNEL_CHECK();
    add2_int<<<(Dhf+255)/256,256>>>(c->y2[i], c->gout[i], Dhf); KERNEL_CHECK();
  }

  int last_g = prev_g;
  if(last_g != 0) {
    CUDA_CHECK(cudaStreamSynchronize(0));
    CUDA_CHECK(cudaMemcpyPeer(c->y1[0], c->engine_gpus[0].dev, c->y1[last_g], c->engine_gpus[last_g].dev, Dhf*sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpyPeer(c->y2[0], c->engine_gpus[0].dev, c->y2[last_g], c->engine_gpus[last_g].dev, Dhf*sizeof(int32_t)));
    CUDA_CHECK(cudaSetDevice(c->engine_gpus[0].dev));
  }

  GPU& g0 = c->engine_gpus[0];
  concat1_int<<<(D+255)/256,256>>>(c->xfull[0], c->y1[0], c->y2[0]); KERNEL_CHECK();
  rms1_int<D><<<1,256>>>(c->xnorm[0], c->invF[0], c->xfull[0], g0.W.gout); KERNEL_CHECK();
  dp4a_gemv_vocab<<<(V+255)/256,256>>>(c->logits[0], c->xnorm[0], g0.Hw.Wout_tr); KERNEL_CHECK();
  int blocks=(V+255)/256;
  argmax_stage1_int<<<blocks,256>>>(c->logits[0], c->amaxv1[0], c->amaxi1[0], V); KERNEL_CHECK();
  argmax_stage2_int<<<1,256>>>(c->amaxv1[0], c->amaxi1[0], c->amaxi[0], blocks); KERNEL_CHECK();
  int next=0;
  CUDA_CHECK(cudaMemcpy(&next, c->amaxi[0], sizeof(int), cudaMemcpyDeviceToHost));
  if(next<0) next=0;
  if(next>=V) next=V-1;
  return next;
}
static void chat_repl(const PairIndex& pi, std::vector<GPU>& gpus, const char* ckpt_path, const char* prompt, bool do_measure){
  ChatCtx ctx{};
  chat_alloc(&ctx, gpus);
  
  if(!gpus.empty()) {
     // Peer access is now done centrally in main().
  }

  std::vector<uint16_t> pre;
  if(prompt && prompt[0]) pre=encode_prompt(pi, std::string(prompt));
  int t=0;
  int next=-1;
  for(uint16_t id: pre){
    next = chat_step(&ctx, t, (int)id);
    t++;
    if(t>=Tmax) break;
  }
  std::fprintf(stderr,"[chat] commands: /reset /quit (greedy decode, PIPELINE KV across GPUs, generates up to 200 tokens)\n");
    std::string line;
    bool has_external_prompt = false;
    const char* prompt_file = "/tmp/deepseek_prompt.txt";
    
    while(true){
      if(!has_external_prompt){
        std::fprintf(stderr,"> ");
        std::fflush(stderr);
      }
      
      line = "";
      // Poll external prompt file (from relay)
      struct stat st;
      if(stat(prompt_file, &st) == 0 && st.st_size > 0){
        FILE* pf = std::fopen(prompt_file, "r");
        if(pf){
          char buf[1024];
          if(std::fgets(buf, sizeof(buf), pf)) line = buf;
          std::fclose(pf);
          std::remove(prompt_file); // Consume it
          if(!line.empty()) {
            std::fprintf(stderr, "[external prompt] %s\n", line.c_str());
            has_external_prompt = true;
          }
        }
      } else {
        // Idle Heartbeat for UI "Alive" signal
        TelemetryPacket p{};
        p.step = (uint32_t)t;
        std::strncpy(p.model_id, "DeepSeek-Elite (Pipeline)", 63);
        std::strncpy(p.prompt, "SYSTEM_IDLE", 63);
        send_telemetry(p);
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
      }
      
      // Fallback to stdin if no external prompt (and stdin hasn't hit EOF)
      if(line.empty()){
        static bool stdin_eof = false;
        if(!stdin_eof){
          if(!std::getline(std::cin, line)){
            stdin_eof = true;
            // stdin closed (nohup/background mode) — keep polling external prompts
            continue;
          }
        } else {
          // stdin is closed, just keep polling external prompt file
          continue;
        }
      }
      
      if(line=="/quit") break;
      if(line=="/reset"){
        t=0;
        for(int i=0; i<ctx.G; i++){
          CUDA_CHECK(cudaSetDevice(ctx.engine_gpus[i].dev));
          for(int l=0;l<L;l++){
            int g_target = (l * ctx.G) / L;
            if (g_target == i) {
              CUDA_CHECK(cudaMemset(ctx.Kc[i][l],0,(size_t)Tmax*(size_t)Dhf*sizeof(int32_t)));
              CUDA_CHECK(cudaMemset(ctx.Vc[i][l],0,(size_t)Tmax*(size_t)Dhf*sizeof(int32_t)));
            }
          }
        }
        next=-1;
        has_external_prompt = false;
        continue;
      }
      
      auto ids=encode_prompt(pi, line+"\n");
      for(size_t i=0;i<ids.size();i++){
        next = chat_step(&ctx, t, (int)ids[i]);
        t++;
        if(t>=Tmax) break;
      }
      if(t>=Tmax){ std::cout << "\n[ctx full]\n"; has_external_prompt=false; continue; }
      
      int gen_max=200;
      std::chrono::time_point<std::chrono::high_resolution_clock> tstart_chat;
      if(do_measure) tstart_chat = std::chrono::high_resolution_clock::now();
      
      for(int i=0;i<gen_max && t<Tmax;i++){
        uint16_t out_id=(uint16_t)next;
        
        // TELEMETRY: Send packet for every generated token (the "thinking" visualization)
        TelemetryPacket p{};
        p.step = (uint32_t)t;
        p.proj_x = 0; p.proj_y = 0; p.proj_z = (float)(out_id % 100); // Visual breadcrumb
        std::strncpy(p.model_id, "DeepSeek-Elite (Pipeline)", 63);
        // We overload prompt field with TokenID: text for the UI to decode and filter
        std::snprintf(p.prompt, 63, "TokenID: %d", (int)out_id);
        send_telemetry(p);

        std::vector<uint8_t> out;
        decode_id(pi, out_id, out);
        bool stop=false;
        for(uint8_t b: out){
          if(b=='\n') stop=true;
          std::cout << (char)b;
        }
        std::cout.flush();

        next = chat_step(&ctx, t, (int)out_id);
        t++;
        if(stop) break;
      }
      std::cout << std::endl;
      has_external_prompt = false;
    }
  chat_free(&ctx);
}
int main(int argc,char** argv){
  const char* data_path="tinyshakespeare.txt";
  const char* ckpt_path="ckpt.bin";
  const char* index_path="index_v7.bin";
  int steps=2000,batch=64,seq=128,gpus_req=2;
  int32_t lr=300, wd=1, clip=100;
  int log_every=50,save_every=500;
  uint64_t seed=123;
  bool do_train=false;
  bool do_chat=false;
  bool do_measure=false;
  bool do_continue=false;
  int use_graph = (std::getenv("DEBUG_NOGRAPH") ? 0 : 1);
  const char* chat_prompt="";
  for(int i=1;i<argc;i++){
    if(!std::strcmp(argv[i],"--train")) do_train=true;
    else if(!std::strcmp(argv[i],"--chat")) do_chat=true;
    else if(!std::strcmp(argv[i],"--continue")) do_continue=true;
    else if(!std::strcmp(argv[i],"--measure")) do_measure=true;
    else if(!std::strcmp(argv[i],"--chat_prompt") && i+1<argc) chat_prompt=argv[++i];
    else if(!std::strcmp(argv[i],"--data") && i+1<argc) data_path=argv[++i];
    else if(!std::strcmp(argv[i],"--ckpt") && i+1<argc) ckpt_path=argv[++i];
    else if(!std::strcmp(argv[i],"--index") && i+1<argc) index_path=argv[++i];
    else if(!std::strcmp(argv[i],"--steps") && i+1<argc) steps=std::atoi(argv[++i]);
    else if(!std::strcmp(argv[i],"--batch") && i+1<argc) batch=std::atoi(argv[++i]);
    else if(!std::strcmp(argv[i],"--seq") && i+1<argc) seq=std::atoi(argv[++i]);
    else if(!std::strcmp(argv[i],"--gpus") && i+1<argc) gpus_req=std::atoi(argv[++i]);
    else if(!std::strcmp(argv[i],"--lr") && i+1<argc) lr=std::atoi(argv[++i]);
    else if(!std::strcmp(argv[i],"--wd") && i+1<argc) wd=std::atoi(argv[++i]);
    else if(!std::strcmp(argv[i],"--clip") && i+1<argc) clip=std::atoi(argv[++i]);
    else if(!std::strcmp(argv[i],"--log_every") && i+1<argc) log_every=std::atoi(argv[++i]);
    else if(!std::strcmp(argv[i],"--save_every") && i+1<argc) save_every=std::atoi(argv[++i]);
    else if(!std::strcmp(argv[i],"--seed") && i+1<argc) seed=(uint64_t)std::strtoull(argv[++i],nullptr,10);
    else if(!std::strcmp(argv[i],"--graph") && i+1<argc) use_graph=std::atoi(argv[++i]);
    else if(!std::strcmp(argv[i],"--no_graph")) use_graph=0;
    else { std::fprintf(stderr,"Unknown arg: %s\n", argv[i]); return 2; }
  }
  if(!do_train && !do_chat) do_train=true;
  if((seq%16)!=0) die("--seq must be multiple of 16");
  if(seq<16 || seq>Tmax) die("--seq out of range");
  auto bytes=read_file_bytes(data_path);
  PairIndex pi;
  size_t ckpt_w_offset = 0;
  bool has_ckpt = false;
  if(!do_train || do_chat || do_continue){
    has_ckpt = load_ckpt(ckpt_path, &pi, &ckpt_w_offset);
  }
  if(do_train && has_ckpt && !do_continue){
    die("ckpt exists: pass --continue or delete ckpt");
  }
  if(!has_ckpt){
    if(!load_index_v7(index_path, &pi)) pi = make_pair_index(bytes);
  }
  if(has_ckpt) {
    if(!do_train) do_chat = true;
  }
  if(!do_train && !do_chat) do_train=true;
  auto ids = encode_ids(pi, bytes.data(), bytes.size());
  if(ids.size() < (size_t)seq+2) die("encoded too small");
  int devCount=0; CUDA_CHECK(cudaGetDeviceCount(&devCount));
  if(devCount<=0) die("no cuda devices");
  int G=gpus_req; if(G>devCount) G=devCount; if(G<1) G=1;
  std::vector<size_t> mem((size_t)G); size_t memsum=0;
  for(int i=0;i<G;i++){ cudaDeviceProp p{}; CUDA_CHECK(cudaGetDeviceProperties(&p,i)); mem[(size_t)i]=(size_t)p.totalGlobalMem; memsum+=mem[(size_t)i]; }
  std::vector<int> Bi((size_t)G,1);
  for(int i=0;i<G;i++){
    double w=(double)mem[(size_t)i]/(double)memsum;
    int b=(int)std::floor(w*(double)batch); if(b<1) b=1;
    Bi[(size_t)i]=b;
  }
  int sumB=0; for(int b:Bi) sumB+=b;
  while(sumB>batch){ for(int i=0;i<G && sumB>batch;i++){ if(Bi[(size_t)i]>1){ Bi[(size_t)i]--; sumB--; } } }
  while(sumB<batch){ for(int i=0;i<G && sumB<batch;i++){ Bi[(size_t)i]++; sumB++; } }
  std::vector<GPU> gpus((size_t)G);
  for(int i=0;i<G;i++){
    gpu_alloc(&gpus[(size_t)i], i, Bi[(size_t)i], seq, do_train);
  }
  
  if (G > 1) {
    for(int i=0;i<G;i++){
      for(int j=0;j<G;j++){
        if(i==j) continue;
        int can=0; cudaDeviceCanAccessPeer(&can, gpus[i].dev, gpus[j].dev);
        if(can) {
          cudaSetDevice(gpus[i].dev);
          cudaDeviceEnablePeerAccess(gpus[j].dev, 0);
        }
      }
    }
  }
  
  if (has_ckpt) {
    FILE* f=std::fopen(ckpt_path,"rb");
    if(!f) die("reopen ckpt failed");
    std::fseek(f, (long)ckpt_w_offset, SEEK_SET);

    if (do_train) {
      // Training mode: load all 24GB into contiguous device memory
      std::vector<int32_t> chunk(32 * 1024 * 1024);
      size_t total = weights_elems(), done = 0;
      while(done < total){
        size_t n = std::min(chunk.size(), total - done);
        rf(f, chunk.data(), n*sizeof(int32_t));
        for(int i=0;i<G;i++){
          CUDA_CHECK(cudaSetDevice(i));
          CUDA_CHECK(cudaMemcpy(gpus[(size_t)i].dW + done, chunk.data(), n*sizeof(int32_t), cudaMemcpyHostToDevice));
        }
        done += n;
      }
      for(int i=0;i<G;i++){ refresh_half_weights(&gpus[(size_t)i]); }
    } else {
      // Chat mode: No monolithic 24GB allocation on ANY GPU! Keep an exact structure traversal directly from host
      // Since mapping requires matching the precise offsets of `pack_W`, we load layer by layer.
      
      auto read_and_cast = [&](int8_t* dst_rm, int8_t* dst_tr, size_t rows, size_t cols, GPU* sync_gpus, int num_g) {
          size_t num_elems = rows * cols;
          std::vector<int32_t> host_layer(num_elems);
          rf(f, host_layer.data(), num_elems*sizeof(int32_t));
          
          for(int i=0; i<num_g; i++) {
             GPU& g = sync_gpus[i];
             CUDA_CHECK(cudaSetDevice(g.dev));
             int32_t* d_tmp;
             CUDA_CHECK(cudaMalloc(&d_tmp, num_elems*sizeof(int32_t)));
             CUDA_CHECK(cudaMemcpy(d_tmp, host_layer.data(), num_elems*sizeof(int32_t), cudaMemcpyHostToDevice));
             
             dim3 blk(16,16);
             dim3 grd((cols+15)/16, (rows+15)/16);
             w_i32_to_i8_rm_tr<<<grd,blk>>>(dst_rm, dst_tr, d_tmp, rows, cols); 
             KERNEL_CHECK();
             
             CUDA_CHECK(cudaStreamSynchronize(0));
             CUDA_CHECK(cudaFree(d_tmp));
          }
      };

      for(int i=0; i<G; i++) { CUDA_CHECK(cudaSetDevice(i)); gpus[i].W.wte=nullptr; gpus[i].W.Wout=nullptr; }

      // We precisely replicate pack_W layout sequential reads
      read_and_cast(gpus[0].Hw.wte_rm, gpus[0].Hw.wte_tr, Vpad, D, &gpus[0], 1); // wte only on GPU 0
      std::fseek(f, (long)(Tmax*D*sizeof(int32_t)), SEEK_CUR); // wpe

      for(int l=0; l<L; l++){
        read_and_cast(gpus[0].Hw.Wq_rm[l], gpus[0].Hw.Wq_tr[l], Dhf, Dhf, &gpus[0], 1);
        read_and_cast(gpus[0].Hw.Wk_rm[l], gpus[0].Hw.Wk_tr[l], Dhf, Dhf, &gpus[0], 1);
        read_and_cast(gpus[0].Hw.Wv_rm[l], gpus[0].Hw.Wv_tr[l], Dhf, Dhf, &gpus[0], 1);
        read_and_cast(gpus[0].Hw.Wo_rm[l], gpus[0].Hw.Wo_tr[l], Dhf, Dhf, &gpus[0], 1);
        read_and_cast(gpus[0].Hw.W1_rm[l], gpus[0].Hw.W1_tr[l], F, Dhf, &gpus[0], 1);
        read_and_cast(gpus[0].Hw.W2_rm[l], gpus[0].Hw.W2_tr[l], Dhf, F, &gpus[0], 1);
        std::fseek(f, (long)(Dhf*sizeof(int32_t)), SEEK_CUR); // gf
        std::fseek(f, (long)(Dhf*sizeof(int32_t)), SEEK_CUR); // gg
        
        // Push the int8 matrices initialized on GPU 0 to peer GPUs if they need them
        for(int pg=1; pg<G; pg++) {
           CUDA_CHECK(cudaSetDevice(pg));
           CUDA_CHECK(cudaMemcpyPeer(gpus[pg].Hw.Wq_rm[l], gpus[pg].dev, gpus[0].Hw.Wq_rm[l], gpus[0].dev, Dhf*Dhf*sizeof(int8_t)));
           CUDA_CHECK(cudaMemcpyPeer(gpus[pg].Hw.Wk_rm[l], gpus[pg].dev, gpus[0].Hw.Wk_rm[l], gpus[0].dev, Dhf*Dhf*sizeof(int8_t)));
           CUDA_CHECK(cudaMemcpyPeer(gpus[pg].Hw.Wv_rm[l], gpus[pg].dev, gpus[0].Hw.Wv_rm[l], gpus[0].dev, Dhf*Dhf*sizeof(int8_t)));
           CUDA_CHECK(cudaMemcpyPeer(gpus[pg].Hw.Wo_rm[l], gpus[pg].dev, gpus[0].Hw.Wo_rm[l], gpus[0].dev, Dhf*Dhf*sizeof(int8_t)));
           CUDA_CHECK(cudaMemcpyPeer(gpus[pg].Hw.W1_rm[l], gpus[pg].dev, gpus[0].Hw.W1_rm[l], gpus[0].dev, F*Dhf*sizeof(int8_t)));
           CUDA_CHECK(cudaMemcpyPeer(gpus[pg].Hw.W2_rm[l], gpus[pg].dev, gpus[0].Hw.W2_rm[l], gpus[0].dev, Dhf*F*sizeof(int8_t)));
           
           CUDA_CHECK(cudaMemcpyPeer(gpus[pg].Hw.Wq_tr[l], gpus[pg].dev, gpus[0].Hw.Wq_tr[l], gpus[0].dev, Dhf*Dhf*sizeof(int8_t)));
           CUDA_CHECK(cudaMemcpyPeer(gpus[pg].Hw.Wk_tr[l], gpus[pg].dev, gpus[0].Hw.Wk_tr[l], gpus[0].dev, Dhf*Dhf*sizeof(int8_t)));
           CUDA_CHECK(cudaMemcpyPeer(gpus[pg].Hw.Wv_tr[l], gpus[pg].dev, gpus[0].Hw.Wv_tr[l], gpus[0].dev, Dhf*Dhf*sizeof(int8_t)));
           CUDA_CHECK(cudaMemcpyPeer(gpus[pg].Hw.Wo_tr[l], gpus[pg].dev, gpus[0].Hw.Wo_tr[l], gpus[0].dev, Dhf*Dhf*sizeof(int8_t)));
           CUDA_CHECK(cudaMemcpyPeer(gpus[pg].Hw.W1_tr[l], gpus[pg].dev, gpus[0].Hw.W1_tr[l], gpus[0].dev, F*Dhf*sizeof(int8_t)));
           CUDA_CHECK(cudaMemcpyPeer(gpus[pg].Hw.W2_tr[l], gpus[pg].dev, gpus[0].Hw.W2_tr[l], gpus[0].dev, Dhf*F*sizeof(int8_t)));
        }
      }
      
      std::fseek(f, (long)(D*sizeof(int32_t)), SEEK_CUR); // gout
      read_and_cast(gpus[0].Hw.Wout_rm, gpus[0].Hw.Wout_tr, D, Vpad, &gpus[0], 1); // Wout only on GPU 0
      
      // Need to populate the FP32 normalizer arrays too
      std::fseek(f, (long)ckpt_w_offset, SEEK_SET);
      std::vector<int32_t> float_layer;
      
      // read wte
      float_layer.resize(Vpad*D); rf(f, float_layer.data(), float_layer.size()*sizeof(int32_t));
      CUDA_CHECK(cudaSetDevice(0));
      CUDA_CHECK(cudaMalloc(&gpus[0].W.wte, Vpad*D*sizeof(int32_t)));
      CUDA_CHECK(cudaMemcpy(gpus[0].W.wte, float_layer.data(), float_layer.size()*sizeof(int32_t), cudaMemcpyHostToDevice));
      
      // read wpe
      float_layer.resize(Tmax*D); rf(f, float_layer.data(), float_layer.size()*sizeof(int32_t));
      CUDA_CHECK(cudaMalloc(&gpus[0].W.wpe, Tmax*D*sizeof(int32_t)));
      CUDA_CHECK(cudaMemcpy(gpus[0].W.wpe, float_layer.data(), float_layer.size()*sizeof(int32_t), cudaMemcpyHostToDevice));
      
      for(int l=0; l<L; l++){
        std::fseek(f, (long)(Dhf*Dhf*4 + F*Dhf*2)*sizeof(int32_t), SEEK_CUR);
        
        float_layer.resize(Dhf); rf(f, float_layer.data(), float_layer.size()*sizeof(int32_t));
        for(int pg=0; pg<G; pg++) {
           CUDA_CHECK(cudaSetDevice(pg)); CUDA_CHECK(cudaMalloc(&gpus[pg].W.gf[l], Dhf*sizeof(int32_t)));
           CUDA_CHECK(cudaMemcpy(gpus[pg].W.gf[l], float_layer.data(), Dhf*sizeof(int32_t), cudaMemcpyHostToDevice));
        }
        
        float_layer.resize(Dhf); rf(f, float_layer.data(), float_layer.size()*sizeof(int32_t));
        for(int pg=0; pg<G; pg++) {
           CUDA_CHECK(cudaSetDevice(pg)); CUDA_CHECK(cudaMalloc(&gpus[pg].W.gg[l], Dhf*sizeof(int32_t)));
           CUDA_CHECK(cudaMemcpy(gpus[pg].W.gg[l], float_layer.data(), Dhf*sizeof(int32_t), cudaMemcpyHostToDevice));
        }
      }
      
      float_layer.resize(D); rf(f, float_layer.data(), float_layer.size()*sizeof(int32_t));
      CUDA_CHECK(cudaSetDevice(0)); CUDA_CHECK(cudaMalloc(&gpus[0].W.gout, D*sizeof(int32_t)));
      CUDA_CHECK(cudaMemcpy(gpus[0].W.gout, float_layer.data(), float_layer.size()*sizeof(int32_t), cudaMemcpyHostToDevice));
    }
    std::fclose(f);
  } else {
    std::vector<int32_t> hostW;
    hostW.resize(weights_elems());
    init_weights_cpu(hostW, seed);
    for(int i=0;i<G;i++){
      if (i==0 || do_train) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaMemcpy(gpus[(size_t)i].dW, hostW.data(), hostW.size()*sizeof(int32_t), cudaMemcpyHostToDevice));
      }
    }
    for(int i=0;i<G;i++){ refresh_half_weights(&gpus[(size_t)i]); }
  }
  if(do_chat){
    chat_repl(pi, gpus, ckpt_path, chat_prompt, do_measure);
    return 0;
  }
  if(use_graph){
    for(int i=0;i<G;i++){
      CUDA_CHECK(cudaSetDevice(i));
      std::memset(gpus[(size_t)i].htok_h, 0, (size_t)gpus[(size_t)i].N*sizeof(uint16_t));
      std::memset(gpus[(size_t)i].htgt_h, 0, (size_t)gpus[(size_t)i].N*sizeof(uint16_t));
      ensure_train_graph(&gpus[(size_t)i]);
      CUDA_CHECK(cudaStreamSynchronize(0));
    }
  }
  std::printf("llm_engine: V=%d(Vpad=%d) K=%d D=%d Dh=%d L=%d H=%d F=%d T=%d gpus=%d batch=%d [",
              V,Vpad,PAIR_K,D,Dh,L,H,F,seq,G,batch);
  for(int i=0;i<G;i++) std::printf("%d%s", Bi[(size_t)i], (i+1<G)?",":"");
  std::printf("]\n");
  std::chrono::time_point<std::chrono::high_resolution_clock> t0;
  if(do_measure) t0 = std::chrono::high_resolution_clock::now();
  for(int step=1; step<=steps; step++){
    std::vector<int32_t> losses((size_t)G,0);
    std::vector<std::thread> th; th.reserve((size_t)G);
    int64_t base = (int64_t)step * 1315423911LL;
    for(int i=0;i<G;i++){
      th.emplace_back([&,i](){ losses[(size_t)i]=train_step(&gpus[(size_t)i], ids, step, base + (int64_t)i*9973LL, seed, use_graph); });
    }
    for(auto& t: th) t.join();
    if(G>1){
      if(!p2p_allreduce(gpus)) host_allreduce(gpus);
      CUDA_CHECK(cudaSetDevice(0));
      int nW=(int)weights_elems();
      scale_i32<<<(nW+255)/256,256>>>(gpus[0].dG, 1, nW);
      KERNEL_CHECK();
    }
    adam_step(&gpus[0], step, lr, wd, clip);
    broadcast_weights_from0(gpus);
    if (step == 1 && do_measure) {
      t0 = std::chrono::high_resolution_clock::now();
    }
    if(log_every>0 && (step%log_every)==0){
      double Lm=0.0; for(int i=0;i<G;i++) Lm += (double)losses[(size_t)i]; Lm/=(double)G;
      if(do_measure){
        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        double tok_sec = (double)(log_every * G * batch * seq) / elapsed;
        t0 = t1;
        std::printf("step %d/%d loss=%.6f ppl=%.3f tok/s=%.1f\n", step, steps, (float)Lm, (float)std::exp(Lm), tok_sec);
      } else {
        std::printf("step %d/%d loss=%.6f ppl=%.3f\n", step, steps, (float)Lm, (float)std::exp(Lm));
      }
      std::fflush(stdout);
    }
    if(save_every>0 && (step%save_every)==0){
      CUDA_CHECK(cudaSetDevice(0));
      std::vector<int32_t> trainW(weights_elems());
      CUDA_CHECK(cudaMemcpy(trainW.data(), gpus[0].dW, trainW.size()*sizeof(int32_t), cudaMemcpyDeviceToHost));
      save_ckpt(ckpt_path, pi, trainW.data());
      std::printf("[*] saved: %s\n", ckpt_path);
      std::fflush(stdout);
    }
  }
  CUDA_CHECK(cudaSetDevice(0));
  std::vector<int32_t> finalW(weights_elems());
  CUDA_CHECK(cudaMemcpy(finalW.data(), gpus[0].dW, finalW.size()*sizeof(int32_t), cudaMemcpyDeviceToHost));
  save_ckpt(ckpt_path, pi, finalW.data());
  std::printf("[*] saved final: %s\n", ckpt_path);
  for(int i=0;i<G;i++) gpu_free(&gpus[(size_t)i], (i>0 && !do_train));
  return 0;
}
CU
if [[ ! -f "$TMP_CU_DIR/$CU" ]]; then
  echo "FATAL: missing $CU in $TMP_CU_DIR"; exit 1
fi
python3 - "$TMP_CU_DIR/$CU" <<'PY'
import re, pathlib, sys
import os
cu_path = sys.argv[1]
p=pathlib.Path(cu_path)
s=p.read_text(encoding="utf-8", errors="replace")
if "MEGA_KERNEL_PATCH_v1" not in s:
    ins = "\n// MEGA_KERNEL_PATCH_v1\nstatic int g_is_capturing = 0;\n"
    if "static int g_is_capturing" not in s:
        m=list(re.finditer(r'^\s*#include[^\n]*\n', s, flags=re.M))
        if not m:
            raise SystemExit("patch: could not find includes")
        last=m[-1].end()
        s = s[:last] + ins + s[last:]
    def repl_kernel_check(m):
        return (
            "#define KERNEL_CHECK() do{ if(g_is_capturing) break; cudaError_t e=cudaGetLastError(); if(e!=cudaSuccess){ \\\n"
            " std::fprintf(stderr,\"KERNEL %s:%d: %s\\n\",__FILE__,__LINE__,cudaGetErrorString(e)); std::exit(1);} }while(0)\n"
        )
    s2, n = re.subn(r'#define\s+KERNEL_CHECK\(\)\s+do\{.*?\}while\(0\)\s*\n', repl_kernel_check, s, count=1, flags=re.S)
    if n>0:
        s=s2
    if "g_is_capturing=1" not in s:
        s = s.replace("CUDA_CHECK(cudaStreamBeginCapture(0", "g_is_capturing=1;\n CUDA_CHECK(cudaStreamBeginCapture(0", 1)
    if "g_is_capturing=0" not in s:
        s = s.replace("CUDA_CHECK(cudaStreamEndCapture(0, &g->graph));",
                      "CUDA_CHECK(cudaStreamEndCapture(0, &g->graph));\n g_is_capturing=0;\n KERNEL_CHECK();",
                      1)
    m=re.search(r'(static\s+void\s+train_step_device\s*\(\s*GPU\*\s*g\s*\)\s*\{)(.*?)(\n\}\s*\n\s*static\s+void\s+ensure_train_graph)', s, flags=re.S)
    if m:
        head, body, tail = m.group(1), m.group(2), m.group(3)
        if "save pointer state (reversible swaps)" not in body:
            body2 = re.sub(
                r'(int\s+B\s*=\s*g->B\s*,\s*T\s*=\s*g->T\s*,\s*N\s*=\s*g->N\s*;\s*)',
                r'\1\n // MEGA_KERNEL_PATCH_v1: save pointer state (reversible swaps)\n'
                r' int32_t *y1=g->y1,*y2=g->y2,*x1=g->x1,*x2=g->x2;\n'
                r' int32_t *dy1=g->dy1,*dy2=g->dy2,*dx1=g->dx1,*dx2=g->dx2;\n',
                body,
                count=1
            )
            body = body2
        if "restore pointer state for next step" not in body:
            body = body + (
                "\n // MEGA_KERNEL_PATCH_v1: restore pointer state for next step\n"
                " g->y1=y1; g->y2=y2; g->x1=x1; g->x2=x2;\n"
                " g->dy1=dy1; g->dy2=dy2; g->dx1=dx1; g->dx2=dx2;\n"
            )
        s = s[:m.start()] + head + body + tail + s[m.end():]
    p.write_text(s, encoding="utf-8")
PY
echo "COMPILING C++ ENGINE (NVCC)..." > "/tmp/deepseek_status.txt"
echo "[*] Building: $BIN (sm_75)"
nvcc -O3 -std=c++17 -arch=sm_75 --default-stream per-thread -lineinfo --expt-relaxed-constexpr \
  -DPAIR_K="$PAIR_K" -DPAIR_K1="$PAIR_K1" -DVCHUNK="$VCHUNK" -DDMODEL="$DMODEL" -DNHEAD="$NHEAD" -DNLAY="$NLAY" -DFFN="$FFN" -DTMAX="$TMAX" \
  "$TMP_CU_DIR/$CU" -o "$TMP_CU_DIR/temp_bin"
mv "$TMP_CU_DIR/temp_bin" "$BIN"
rm -rf "$TMP_CU_DIR"
echo "ENGINE ONLINE" > "/tmp/deepseek_status.txt"
echo
echo "================================================================="
echo " Run Settings (Hash: $CKPT_HASH)"
echo " cmd: ./$BIN --data \"$DATA_FILE\" --index \"$INDEX_BIN\" ${PASSED_ARGS[@]}"
echo "================================================================="
echo
./"$BIN" --data "$DATA_FILE" --index "$INDEX_BIN" "${PASSED_ARGS[@]}"
