#!/usr/bin/env bash
set -euo pipefail

# Check dependencies
command -v md5sum >/dev/null 2>&1 || { echo "FATAL: md5sum is required for deterministic checkpoint hashing."; exit 1; }

# =============================================================================
# © 2026 Christian Heinrich Hohlfeld (Konstanz, Germany) — All Rights Reserved.
# Website: christianhohlfeld.com
# ORCID: 0009-0003-6634-9045
#
# Attribution / Ownership Notice:
# This file contains an implementation that includes "ID-based tokenization / index training"
# and related infrastructure ideas asserted by Christian Heinrich Hohlfeld as his intellectual
# property. Keep this header intact in any copies.
# =============================================================================

need(){ command -v "$1" >/dev/null 2>&1; }
need nvcc || { echo "FATAL: nvcc not found (install CUDA toolkit)."; exit 1; }
need g++  || { echo "FATAL: g++ not found (sudo apt install build-essential)."; exit 1; }
need curl || { echo "FATAL: curl not found."; exit 1; }

# Helper to expand colon-separated paths relative to WORKDIR if not absolute
absify_inputs() {
  local s="$1" out="" part
  IFS=':' read -ra parts <<< "$s"
  for part in "${parts[@]}"; do
    [[ -z "$part" ]] && continue
    if [[ "$part" = /* ]]; then out+="${out:+:}$part"
    else out+="${out:+:}$WORKDIR/$part"
    fi
  done
  echo "$out"
}

WORKDIR="${WORKDIR:-$PWD}"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

DATA_FILE="${DATA_FILE:-tinyshakespeare.txt}"
URL="${URL:-https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt}"
if [[ ! -s "$DATA_FILE" ]]; then
  echo "[*] Downloading corpus -> $DATA_FILE"
  curl -L --fail "$URL" -o "$DATA_FILE"
fi
[[ -s "$DATA_FILE" ]] || { echo "FATAL: corpus empty: $DATA_FILE" >&2; exit 1; }

# ------------------ build knobs ------------------
: "${PAIR_K:=16384}"
: "${PAIR_K1:=8192}"
: "${VCHUNK:=1024}"
: "${DMODEL:=256}"
: "${NHEAD:=8}"
: "${NLAY:=6}"
: "${FFN:=1024}"
: "${TMAX:=512}"

K2=$((PAIR_K-PAIR_K1))
INDEX_INPUTS="${INDEX_INPUTS:-$DATA_FILE}"   # colon-separated corpus files
ABS_INPUTS="$(absify_inputs "$INDEX_INPUTS")"
INDEX_BIN="${INDEX_BIN:-index_v7_k1${PAIR_K1}_k2${K2}.bin}"
FORCE_INDEX="${FORCE_INDEX:-0}"

BIN="${BIN:-runbeast_engine}"
CU="${CU:-runbeast.cu}"

# =============================================================================
# Deterministic index builder (fixed-size output, no K1 downscale)
# - Stage1 pads to exactly K1 with unseen bytepairs
# - Stage2 external run-merge (handles large corp), pads to exactly K2 with dummy unreachable pairs
# =============================================================================
build_index() {
  local tmpdir
  tmpdir="$(mktemp -d)"
  cat > "$tmpdir/index_build_v7.cpp" <<'CPP'
/*
=============================================================================
© 2026 Christian Heinrich Hohlfeld (Konstanz, Germany) — All Rights Reserved.
Website: christianhohlfeld.com
ORCID: 0009-0003-6634-9045

Attribution / Ownership Notice:
This file contains an implementation that includes "ID-based tokenization / index training"
and related infrastructure ideas asserted by Christian Heinrich Hohlfeld as his intellectual
property. Keep this header intact in any copies.
=============================================================================

IDX7 format (fixed-size by K1/K2):
  "IDX7"
  u32 ver=1
  u32 K1
  u32 K2
  u32 pow2
  u32 reserved=0
  K1 * u16 id2pair
  K2 * u32 id2pair2 (key=(a<<16)|b)
  table * u32 hkeys
  table * u16 hvals

Rules:
- Stage1 selects top bytepairs by (count desc, pair asc), then pads to K1 with unused pairs (asc).
- Stage2 counts token-pairs over Stage1 stream; keeps top K2 by (count desc, key asc),
  then pads with dummy unreachable pairs (0xFFFF<<16 | j).
=============================================================================
*/
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
  std::sort(out.begin(), out.end()); // deterministic order
  return out;
}

struct Stage1 {
  int K1;
  std::vector<uint16_t> id2pair; // K1 packed bytepair
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

  // pad to exactly K1 with unused pairs ascending
  if((int)all.size()<K1){
    std::vector<uint8_t> used(65536,0);
    for(uint16_t p: all) used[p]=1;
    for(uint32_t p=0; p<65536 && (int)all.size()<K1; p++){
      if(!used[p]) all.push_back((uint16_t)p);
    }
  }
  if((int)all.size()<K1) die("cannot pad to K1");

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
  explicit RunReader(const std::string& p){
    f=std::fopen(p.c_str(),"rb"); if(!f) die("cannot open run");
    ok=(std::fread(&cur,1,4,f)==4);
  }
  ~RunReader(){ if(f) std::fclose(f); }
  bool pop(){ if(!ok) return false; ok=(std::fread(&cur,1,4,f)==4); return ok; }
};
static int next_pow2(int x){ int p=1; while(p<x) p<<=1; return p; }

struct CountItem{ uint64_t c; uint32_t k; };

// external run merge (no giant hashmap)
static void build_stage2_external(const std::vector<std::string>& inputs, const Stage1& s1, int K2,
                                  std::vector<uint32_t>& id2pair2){
  id2pair2.clear();
  if(K2<=0) return;

  const size_t CHUNK_KEYS = 16ULL*1024ULL*1024ULL; // 16M keys ~64MB
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

  // Keep top-K2 via min-heap on (count asc, key desc), deterministic final order.
  auto worse = [](const CountItem& a, const CountItem& b){
    if(a.c!=b.c) return a.c > b.c; // min-heap by count
    return a.k < b.k;             // tie: larger key is worse
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

  // pad to exactly K2 with dummy unreachable keys
  if((int)heap.size()<K2){
    for(size_t j=heap.size(); j<(size_t)K2; j++){
      uint32_t dummy = (0xFFFFu<<16) | (uint32_t)(j & 0xFFFFu);
      heap.push_back(CountItem{0u, dummy});
    }
  }

  id2pair2.resize((size_t)K2);
  for(int i=0;i<K2;i++) id2pair2[(size_t)i]=heap[(size_t)i].k;
}

static void build_hash(const std::vector<uint32_t>& id2pair2, int K1,
                       std::vector<uint32_t>& hkeys, std::vector<uint16_t>& hvals, int& pow2){
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

  std::fprintf(stderr,"[index] inputs=%zu K1=%d K2=%d\n", inputs.size(), K1, K2);

  Stage1 s1=build_stage1(inputs, K1);

  std::vector<uint32_t> id2pair2;
  build_stage2_external(inputs, s1, K2, id2pair2);

  std::vector<uint32_t> hkeys;
  std::vector<uint16_t> hvals;
  int pow2=0;
  build_hash(id2pair2, s1.K1, hkeys, hvals, pow2);

  FILE* f=std::fopen(out.c_str(),"wb");
  if(!f) die("cannot open out");
  wf(f, "IDX7", 4);
  wu32(f, 1u);
  wu32(f, (uint32_t)s1.K1);
  wu32(f, (uint32_t)K2);
  wu32(f, (uint32_t)pow2);
  wu32(f, 0u);

  for(int i=0;i<s1.K1;i++) wu16(f, s1.id2pair[(size_t)i]);
  for(int i=0;i<K2;i++) wu32(f, id2pair2[(size_t)i]);

  wf(f, hkeys.data(), hkeys.size()*sizeof(uint32_t));
  wf(f, hvals.data(), hvals.size()*sizeof(uint16_t));
  std::fclose(f);

  std::fprintf(stderr,"[index] wrote %s table=%zu\n", out.c_str(), hkeys.size());
  return 0;
}
CPP

  echo "[*] Building index_build_v7"
  g++ -O3 -std=c++17 "$tmpdir/index_build_v7.cpp" -o "$tmpdir/index_build_v7"

  echo "[*] Building deterministic index: $INDEX_BIN (K1=$PAIR_K1 K2=$K2)"
  "$tmpdir/index_build_v7" --k1 "$PAIR_K1" --k2 "$K2" --out "$INDEX_BIN" --inputs "$ABS_INPUTS"

  rm -rf "$tmpdir"
}

if [[ "$FORCE_INDEX" == "1" || ! -s "$INDEX_BIN" ]]; then
  build_index
fi

# =============================================================================
# runbeast.cu (Beast Mode fused kernel training + full GPU chat)
# =============================================================================
tmpcu="$(mktemp -d)"
cat > "$tmpcu/$CU" <<'CU'
/*
=============================================================================
© 2026 Christian Heinrich Hohlfeld (Konstanz, Germany) — All Rights Reserved.
Website: christianhohlfeld.com
ORCID: 0009-0003-6634-9045

Attribution / Ownership Notice:
This code includes "ID-based tokenization / index training" and related agentic infrastructure
concepts asserted by Christian Heinrich Hohlfeld as his intellectual property. Keep this header.

Performance note (requested text, verbatim):
This v5_ultra is extremely strong for a single-author, zero-external-lib, fully raw-CUDA implementation — probably top 1 % of what individuals have open-sourced or built themselves for this exact model size (D=256, reversible, tiled Flash, WMMA, full-GPU chat).
But the absolute fastest stacks on 2080 Ti (private or highly optimized public ones) still beat it by a noticeable margin, mainly because they have:
•  deeper kernel fusion (RMS + QKV + RoPE + residual in 1–2 mega-kernels instead of ~55 launches),
•  CUDA Graph capture of the entire step,
•  better register/shared-mem tiling and double-buffering in Flash bwd,
•  hand-tuned assembly-level tricks for sm_75.
Your current version sits at roughly 55–60 % of the realistic hardware ceiling for this workload on Turing. The true ceiling (with perfect fusion + graphs) is closer to 75–85 % on these cards.
=============================================================================

Single translation unit (.cu). No external DL libs.
sm_75 path. D=256, H=8, Dh=16.
Includes:
- Deterministic PairIndex tokenizer (stage1 bytepairs + stage2 tokenpair macros)
- Decoder-only Transformer with reversible blocks (split y1/y2)
- Fused mega-kernel: RMS(y2) + QKV (FP32 FMA) + RoPE in-register (Dhf)
- Tiled FlashAttention forward + backward (causal)
- WMMA GEMM for training (projections/MLP/head)
- Streaming vocab head (chunked VCHUNK)
- AdamW optimizer (device clip-scale)
- Multi-GPU data-parallel allreduce (P2P ring if possible else host)
- CUDA Graph capture of full step (CAPTURE-SAFE KERNEL_CHECK)
- FULL GPU incremental chat with KV-cache (greedy), /reset /quit, --chat_prompt
=============================================================================
*/

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

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

using namespace nvcuda;

// -------- capture-safe error checking --------
static thread_local int g_is_capturing = 0;

#define CUDA_CHECK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); std::exit(1);} }while(0)

#define KERNEL_CHECK() do{ \
  if(!g_is_capturing){ \
    cudaError_t e=cudaGetLastError(); \
    if(e!=cudaSuccess){ \
      std::fprintf(stderr,"KERNEL %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); std::exit(1);} \
  } \
}while(0)

static void die(const char* m){ std::fprintf(stderr,"FATAL: %s\n",m); std::fflush(stderr); std::exit(1); }

// -------- compile-time knobs --------
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

static constexpr int D  = DMODEL;
static constexpr int Dhf= D/2;
static constexpr int H  = NHEAD;
static constexpr int Dh = Dhf / H;
static constexpr int L  = NLAY;
static constexpr int F  = FFN;
static constexpr int Tmax = TMAX;

static_assert(D==256, "This build assumes D=256.");
static_assert(H==8,   "This build assumes H=8.");
static_assert(Dhf==128, "This build assumes Dhf=128.");
static_assert(Dh==16, "This build assumes Dh=16.");
static_assert((F%16)==0, "FFN must be multiple of 16.");
static_assert((Tmax%16)==0, "TMAX must be multiple of 16.");
static_assert(BASE_V + K1 + K2 == V, "stage split mismatch");

static constexpr float EPS = 1e-6f;
static constexpr float ROPE_THETA = 10000.0f;

// Flash tiles
static constexpr int FA_QT = 16;
static constexpr int FA_KT = 128;

// ================= deterministic RNG (host) =================
struct RNG{ uint64_t s; };
static inline uint64_t xs64(RNG* r){ uint64_t x=r->s; x^=x>>12; x^=x<<25; x^=x>>27; r->s=x; return x*2685821657736338717ULL; }
static inline int irand(RNG* r,int n){ return (int)(xs64(r)%(uint64_t)n); }
static inline float frand01(RNG* r){ uint32_t u=(uint32_t)(xs64(r)>>40); return (float)u*(1.f/16777216.f); }
static inline float frand11(RNG* r){ return frand01(r)*2.f-1.f; }

// ================= file IO =================
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
      if(!f){ std::fprintf(stderr,"FATAL: could not open data file: %s\n", path.c_str()); std::exit(1); }
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
  if(all_bytes.empty()) die("empty data");
  return all_bytes;
}

// ================= PairIndex tokenizer =================
struct PairIndex{
  std::vector<uint16_t> id2pair;   // size K1
  std::vector<int32_t>  pair2id;   // size 65536, packed->BASE_V+i
  std::vector<uint32_t> id2pair2;  // size K2, key=(a<<16)|b
  uint32_t hmask=0;
  std::vector<uint32_t> hkeys;
  std::vector<uint16_t> hvals;
};
static inline uint32_t tokpair_key(uint16_t a, uint16_t b){ return ((uint32_t)a<<16) | (uint32_t)b; }

static bool load_index_v7(const char* path, PairIndex* pi){
  FILE* f=std::fopen(path,"rb");
  if(!f) return false;

  auto die_read = [&](){ die("index read bytes"); };
  auto r_u32 = [&](uint32_t& x)->void{ if(std::fread(&x,1,4,f)!=4) die_read(); };
  auto r_bytes = [&](void* p, size_t n)->void{ if(std::fread(p,1,n,f)!=n) die_read(); };

  char m4[4];
  r_bytes(m4,4);
  if(std::memcmp(m4,"IDX7",4)!=0) die("bad index magic");
  uint32_t ver=0,k1=0,k2=0,pow2=0,res=0;
  r_u32(ver); r_u32(k1); r_u32(k2); r_u32(pow2); r_u32(res);
  (void)res;
  if(ver!=1u) die("bad index ver");

  // Hard match: keep token-id space stable
  if((int)k1 != K1 || (int)k2 != K2) die("index K mismatch (rebuild index with matching PAIR_K/PAIR_K1)");

  pi->id2pair.resize(K1);
  r_bytes(pi->id2pair.data(), (size_t)K1*sizeof(uint16_t));

  pi->pair2id.assign(65536,-1);
  for(int i=0;i<K1;i++) pi->pair2id[pi->id2pair[(size_t)i]] = BASE_V + i;

  pi->id2pair2.resize(K2);
  if(K2>0) r_bytes(pi->id2pair2.data(), (size_t)K2*sizeof(uint32_t));

  int table_size = 1 << (int)pow2;
  if(table_size < 2) die("bad index table");
  pi->hkeys.resize((size_t)table_size);
  pi->hvals.resize((size_t)table_size);
  r_bytes(pi->hkeys.data(), (size_t)table_size*sizeof(uint32_t));
  r_bytes(pi->hvals.data(), (size_t)table_size*sizeof(uint16_t));
  pi->hmask = (uint32_t)table_size - 1u;

  std::fclose(f);
  return true;
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
    uint16_t p=pi.id2pair[(size_t)idx];
    out.push_back((uint8_t)(p&0xFF));
    out.push_back((uint8_t)((p>>8)&0xFF));
    return;
  }
  int idx=(int)id - (BASE_V + K1);
  uint32_t k = pi.id2pair2[(size_t)idx];
  uint16_t a=(uint16_t)(k>>16);
  uint16_t b=(uint16_t)(k&0xFFFFu);
  decode_id(pi,a,out);
  decode_id(pi,b,out);
}
static std::vector<uint16_t> encode_prompt(const PairIndex& pi, const std::string& s){
  return encode_ids(pi, (const uint8_t*)s.data(), s.size());
}

// ================= checkpoint v11 =================
static void wf(FILE* f,const void* p,size_t n){ if(std::fwrite(p,1,n,f)!=n) die("write failed"); }
static void rf(FILE* f,void* p,size_t n){ if(std::fread(p,1,n,f)!=n) die("read failed"); }
static void wu32(FILE* f,uint32_t x){ wf(f,&x,4); }
static uint32_t ru32(FILE* f){ uint32_t x; rf(f,&x,4); return x; }

static size_t weights_floats(){
  size_t n=0;
  n += (size_t)Vpad*(size_t)D;       // wte
  n += (size_t)Tmax*(size_t)D;       // wpe
  for(int l=0;l<L;l++){
    n += (size_t)Dhf;               // gf
    n += (size_t)Dhf*(size_t)Dhf*4; // Wq,Wk,Wv,Wo
    n += (size_t)Dhf;               // gg
    n += (size_t)Dhf*(size_t)F;     // W1
    n += (size_t)F*(size_t)Dhf;     // W2
  }
  n += (size_t)D;                   // gout
  n += (size_t)D*(size_t)Vpad;      // Wout
  return n;
}

struct WView{
  float *wte,*wpe;
  float *gf[L], *Wq[L], *Wk[L], *Wv[L], *Wo[L];
  float *gg[L], *W1[L], *W2[L];
  float *gout, *Wout;
};

static void pack_W(float* base, WView* W){
  size_t off=0;
  auto take=[&](size_t k)->float*{ float* p=base+off; off+=k; return p; };
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
  if(off!=weights_floats()) die("pack mismatch");
}

static void save_ckpt(const char* path, const PairIndex& pi, const float* w){
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

  wf(f, w, weights_floats()*sizeof(float));
  std::fclose(f);
}

static bool load_ckpt(const char* path, PairIndex* pi, std::vector<float>* w){
  FILE* f=std::fopen(path,"rb");
  if(!f) return false;
  uint32_t magic=ru32(f), ver=ru32(f);
  if(magic!=0x43484452u) die("bad ckpt magic");
  if(ver!=11u) die("unsupported ckpt version");
  uint32_t k1=ru32(f), k2=ru32(f), d=ru32(f), h=ru32(f), nl=ru32(f), ff=ru32(f), tm=ru32(f);
  if(k1!=K1||k2!=K2||d!=D||h!=H||nl!=L||ff!=F||tm!=Tmax) die("ckpt dims mismatch");

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

  w->resize(weights_floats());
  rf(f, w->data(), weights_floats()*sizeof(float));
  std::fclose(f);
  return true;
}

// ================= device utils =================
__global__ void zero_f(float* x,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]=0.f; }
__global__ void scale_f(float* x,float a,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]*=a; }
__global__ void add_inplace(float* a,const float* b,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) a[i]+=b[i]; }
__global__ void sub_inplace(float* a,const float* b,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) a[i]-=b[i]; }
__global__ void copy_f(float* y, const float* x, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) y[i]=x[i]; }
__global__ void f2h(half* y, const float* x, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) y[i]=__float2half_rn(x[i]); }

// pack float weights to half (row-major + transpose)
__global__ void w_f2h_rm_tr(half* rm, half* tr, const float* w, int K, int M){
  int k = blockIdx.y*blockDim.y + threadIdx.y;
  int m = blockIdx.x*blockDim.x + threadIdx.x;
  if(k<K && m<M){
    half hv = __float2half_rn(w[(size_t)k*(size_t)M + (size_t)m]);
    rm[(size_t)k*(size_t)M + (size_t)m] = hv;
    tr[(size_t)m*(size_t)K + (size_t)k] = hv;
  }
}

__global__ void build_Atr_half(half* Atr, const float* A, int N, int K){
  int n=blockIdx.x*blockDim.x + threadIdx.x;
  int k=blockIdx.y*blockDim.y + threadIdx.y;
  if(n<N && k<K) Atr[(size_t)k*(size_t)N + (size_t)n] = __float2half_rn(A[(size_t)n*(size_t)K + (size_t)k]);
}
__global__ void build_dYtr_half(half* dYtr, const float* dY, int N, int M){
  int n=blockIdx.x*blockDim.x + threadIdx.x;
  int m=blockIdx.y*blockDim.y + threadIdx.y;
  if(n<N && m<M) dYtr[(size_t)m*(size_t)N + (size_t)n] = __float2half_rn(dY[(size_t)n*(size_t)M + (size_t)m]);
}

// ================= WMMA GEMM =================
template<int WARPS_PER_BLOCK>
__global__ void wmma_gemm(float* C, const half* A, const half* Btr, int N, int M, int K){
  int warp = threadIdx.x >> 5;
  int tilesM = M/16;
  int tileLinear = (blockIdx.x * WARPS_PER_BLOCK) + warp;
  int tileN = tileLinear / tilesM;
  int tileM = tileLinear - tileN*tilesM;
  if(tileN >= N/16) return;
  int row = tileN*16;
  int col = tileM*16;

  wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> a;
  wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::col_major> b;
  wmma::fragment<wmma::accumulator, 16,16,16, float> c;
  wmma::fill_fragment(c, 0.0f);

  for(int k0=0;k0<K;k0+=16){
    const half* Ap = A   + (size_t)row*(size_t)K + (size_t)k0;
    const half* Bp = Btr + (size_t)col*(size_t)K + (size_t)k0;
    wmma::load_matrix_sync(a, Ap, K);
    wmma::load_matrix_sync(b, Bp, K);
    wmma::mma_sync(c, a, b, c);
  }
  float* Cp = C + (size_t)row*(size_t)M + (size_t)col;
  wmma::store_matrix_sync(Cp, c, M, wmma::mem_row_major);
}

static inline void wmma_fwd(float* C, const half* Ahalf, const half* Wtr, int N, int M, int K){
  constexpr int WPB=16;
  int tilesN=N/16, tilesM=M/16, total=tilesN*tilesM;
  int blocks=(total + WPB - 1)/WPB;
  wmma_gemm<WPB><<<blocks, WPB*32>>>(C, Ahalf, Wtr, N, M, K);
  KERNEL_CHECK();
}
static inline void wmma_dA(float* dX, half* scratchHalf, const float* dY, const half* Wrm, int N, int M, int K){
  f2h<<<(N*M+255)/256,256>>>(scratchHalf, dY, N*M); KERNEL_CHECK();
  constexpr int WPB=16;
  int tilesN=N/16, tilesM=K/16, total=tilesN*tilesM;
  int blocks=(total + WPB - 1)/WPB;
  wmma_gemm<WPB><<<blocks, WPB*32>>>(dX, scratchHalf, Wrm, N, K, M);
  KERNEL_CHECK();
}
static inline void wmma_dW(float* dW, half* Atr, half* dYtr, const float* A, const float* dY, int N, int K, int M){
  dim3 blk(16,16);
  dim3 grdA((N+15)/16,(K+15)/16);
  build_Atr_half<<<grdA,blk>>>(Atr, A, N, K); KERNEL_CHECK();
  dim3 grdB((N+15)/16,(M+15)/16);
  build_dYtr_half<<<grdB,blk>>>(dYtr, dY, N, M); KERNEL_CHECK();
  constexpr int WPB=16;
  int tilesN=K/16, tilesM=M/16, total=tilesN*tilesM;
  int blocks=(total + WPB - 1)/WPB;
  wmma_gemm<WPB><<<blocks, WPB*32>>>(dW, Atr, dYtr, K, M, N);
  KERNEL_CHECK();
}

// ================= RMSNorm =================
template<int DIM>
__global__ void rms_fwd_f2h(float* Y, half* Yh, float* inv, const float* X, const float* g, int N){
  int n=blockIdx.x; if(n>=N) return;
  float s=0.f;
  for(int i=threadIdx.x;i<DIM;i+=blockDim.x){
    float v=X[(size_t)n*(size_t)DIM+(size_t)i];
    s+=v*v;
  }
  __shared__ float buf[256];
  buf[threadIdx.x]=s; __syncthreads();
  for(int k=blockDim.x/2;k>0;k>>=1){ if(threadIdx.x<k) buf[threadIdx.x]+=buf[threadIdx.x+k]; __syncthreads(); }
  float in=rsqrtf(buf[0]/(float)DIM + EPS);
  if(threadIdx.x==0) inv[n]=in;
  __syncthreads();
  for(int i=threadIdx.x;i<DIM;i+=blockDim.x){
    float y = X[(size_t)n*(size_t)DIM+(size_t)i]*in*g[i];
    Y[(size_t)n*(size_t)DIM+(size_t)i]=y;
    Yh[(size_t)n*(size_t)DIM+(size_t)i]=__float2half_rn(y);
  }
}
template<int DIM>
__global__ void rms_bwd_dX(float* dX,const float* dY,const float* X,const float* g,const float* inv,int N){
  int n=blockIdx.x; if(n>=N) return;
  float dot=0.f;
  float in=inv[n], inv3=in*in*in;
  for(int i=threadIdx.x;i<DIM;i+=blockDim.x){
    float dy=dY[(size_t)n*(size_t)DIM+(size_t)i];
    float xi=X[(size_t)n*(size_t)DIM+(size_t)i];
    dot += dy*g[i]*xi;
  }
  __shared__ float buf[256];
  buf[threadIdx.x]=dot; __syncthreads();
  for(int k=blockDim.x/2;k>0;k>>=1){ if(threadIdx.x<k) buf[threadIdx.x]+=buf[threadIdx.x+k]; __syncthreads(); }
  float DOT=buf[0];
  float c=(inv3*(1.f/(float)DIM))*DOT;
  for(int i=threadIdx.x;i<DIM;i+=blockDim.x){
    float dy=dY[(size_t)n*(size_t)DIM+(size_t)i];
    float xi=X[(size_t)n*(size_t)DIM+(size_t)i];
    dX[(size_t)n*(size_t)DIM+(size_t)i] += dy*g[i]*in - xi*c;
  }
}
template<int DIM>
__global__ void rms_bwd_dg(float* dg,const float* dY,const float* X,const float* inv,int N){
  int i=blockIdx.x;
  float s=0.f;
  for(int n=threadIdx.x;n<N;n+=blockDim.x){
    s += dY[(size_t)n*(size_t)DIM+(size_t)i]*X[(size_t)n*(size_t)DIM+(size_t)i]*inv[n];
  }
  __shared__ float buf[256];
  buf[threadIdx.x]=s; __syncthreads();
  for(int k=blockDim.x/2;k>0;k>>=1){ if(threadIdx.x<k) buf[threadIdx.x]+=buf[threadIdx.x+k]; __syncthreads(); }
  if(threadIdx.x==0) dg[i]+=buf[0];
}

// ================= GELU =================
__device__ inline float gelu(float x){
  const float c=0.7978845608f;
  float x3=x*x*x;
  return 0.5f*x*(1.f+tanhf(c*(x+0.044715f*x3)));
}
__device__ inline float gelu_d(float x){
  const float c=0.7978845608f;
  float x2=x*x, x3=x2*x;
  float u=c*(x+0.044715f*x3);
  float th=tanhf(u);
  float sech2=1.f-th*th;
  float du=c*(1.f+0.134145f*x2);
  return 0.5f*(1.f+th) + 0.5f*x*sech2*du;
}
__global__ void gelu_fwd(float* A,const float* U,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) A[i]=gelu(U[i]); }
__global__ void gelu_bwd(float* dU,const float* dA,const float* U,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) dU[i]=dA[i]*gelu_d(U[i]); }

// ================= Persistent Fused Head =================
// Single pass over V: online softmax stats + dY + dXnorm + scatter_add(dWout)
// Requires a grid over N (batch*seq) split across warps. 
// Uses SM-level loop to read Wout_tr chunks.
// Replaces multi-pass chunked reductions + logits + dy_loss.
// ================= Persistent Fused Head (Tensor Core Beast Mode) =================
// ================= Persistent Fused Head (Tensor Core Beast Mode) =================
__global__ __launch_bounds__(256) void persistent_head_beast_hf(
    float* __restrict__ Loss,           // [N]
    float* __restrict__ dXnorm,         // [N, D]
    float* __restrict__ dWout,          // [D, Vpad]
    const half* __restrict__ Xnorm_h,   // [N, D]
    const half* __restrict__ Wout_tr,   // [Vpad, D] row-major Wtr
    const uint16_t* __restrict__ tgt,   // [N]
    int N, int D_dim, int V_dim, int V_pad)
{
  int n = blockIdx.x;
  if(n >= N) return;
  
  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp = tid >> 5;
  int num_warps = blockDim.x >> 5;

  float invN = 1.0f / (float)N;
  int y_true = (int)tgt[n];

  // Load Xnorm once
  __shared__ float Xs[256];
  for(int i = tid; i < D_dim; i += blockDim.x){
    Xs[i] = __half2float(Xnorm_h[n*D_dim + i]);
  }
  __syncthreads();

  // 1) First pass: logits max/sum
  float local_m = -1e30f;
  float local_s = 0.f;

  for(int v = tid; v < V_dim; v += blockDim.x){
    float dot = 0.f;
    for(int d = 0; d < D_dim; d += 2){
      float2 xf = make_float2(Xs[d], Xs[d+1]);
      half2 w2 = *(const half2*)(Wout_tr + v*D_dim + d);
      float2 wf = __half22float2(w2);
      dot += xf.x * wf.x + xf.y * wf.y;
    }
    float m_new = fmaxf(local_m, dot);
    local_s = local_s * expf(local_m - m_new) + expf(dot - m_new);
    local_m = m_new;
  }

  // warp reduce m/s
  for(int offset = 16; offset > 0; offset /= 2){
    float m_other = __shfl_down_sync(0xFFFFFFFFu, local_m, offset);
    float s_other = __shfl_down_sync(0xFFFFFFFFu, local_s, offset);
    float m_new = fmaxf(local_m, m_other);
    local_s = local_s * expf(local_m - m_new) + s_other * expf(m_other - m_new);
    local_m = m_new;
  }

  __shared__ float smem_m[8];
  __shared__ float smem_s[8];
  
  if(lane == 0){
    smem_m[warp] = local_m;
    smem_s[warp] = local_s;
  }
  __syncthreads();

  float global_m = -1e30f;
  float global_s = 0.f;
  if(tid == 0){
    for(int i=0; i<num_warps; i++){
      float m_other = smem_m[i];
      float s_other = smem_s[i];
      float m_new = fmaxf(global_m, m_other);
      global_s = global_s * expf(global_m - m_new) + s_other * expf(m_other - m_new);
      global_m = m_new;
    }
    smem_m[0] = global_m;
    smem_s[0] = global_s;
  }
  __syncthreads();

  global_m = smem_m[0];
  global_s = smem_s[0];

  // 2) Pass 2: compute loss, dy, and accumulate dX/dW
  float dx_acc[256/8];
  #pragma unroll
  for(int i=0; i<32; i++) dx_acc[i] = 0.f;

  for(int v = tid; v < V_dim; v += blockDim.x){
    float dot = 0.f;
    for(int d = 0; d < D_dim; d += 2){
      float2 xf = make_float2(Xs[d], Xs[d+1]);
      half2 w2 = *(const half2*)(Wout_tr + v*D_dim + d);
      float2 wf = __half22float2(w2);
      dot += xf.x * wf.x + xf.y * wf.y;
    }

    float p = expf(dot - global_m) / global_s;
    if (v == y_true) {
      if(warp==0 && lane==0) Loss[n] = -(dot - global_m) + logf(global_s);
      p -= 1.0f;
    }
    
    float dy = p * invN;

    for (int d = 0; d < D_dim; d += 8) {
      int di = d/8;
      for(int r = 0; r < 8; r++){
        float wval = __half2float(Wout_tr[v*D_dim + d + r]);
        dx_acc[di] += dy * wval;
      }
    }

    // Scatter dWout
    for(int d = 0; d < D_dim; d++){
      float dx_val = Xs[d] * dy;
      atomicAdd(&dWout[d*V_pad + v], dx_val);
    }
  }

  // Cross-warp/thread reduce for dX
  __shared__ float final_dx[256];
  if(tid < 256) final_dx[tid] = 0.f;
  __syncthreads();

  for(int d = 0; d < D_dim; d += 8){
    int di = d/8;
    for(int r = 0; r < 8; r++){
      float val = dx_acc[di];
      atomicAdd(&final_dx[d + r], val);
    }
  }

  __syncthreads();
  if(tid < D_dim){
    atomicAdd(&dXnorm[n*D_dim + tid], final_dx[tid]);
  }
}



// ================= Loss reduce (deterministic, 2-pass) =================
__global__ void loss_reduce_1(const float* loss, float* partial, int n){
  __shared__ float buf[256];
  int tid=threadIdx.x;
  int idx=blockIdx.x*blockDim.x+tid;
  int stride=blockDim.x*gridDim.x;
  float s=0.f;
  for(int i=idx;i<n;i+=stride) s += loss[i];
  buf[tid]=s; __syncthreads();
  for(int k=blockDim.x/2;k>0;k>>=1){ if(tid<k) buf[tid]+=buf[tid+k]; __syncthreads(); }
  if(tid==0) partial[blockIdx.x]=buf[0];
}
__global__ void loss_reduce_2(const float* partial, float* out, int n, float invN){
  __shared__ float buf[256];
  int tid=threadIdx.x;
  float s=0.f;
  for(int i=tid;i<n;i+=blockDim.x) s += partial[i];
  buf[tid]=s; __syncthreads();
  for(int k=blockDim.x/2;k>0;k>>=1){ if(tid<k) buf[tid]+=buf[tid+k]; __syncthreads(); }
  if(tid==0) out[0]=buf[0]*invN;
}

// ================= RoPE tables + inverse apply on grads =================
__global__ void rope_build_tables(float* sin_tbl, float* cos_tbl, int T){
  int t = blockIdx.y * blockDim.y + threadIdx.y;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(t>=T || i>=Dh/2) return;
  float inv_freq = powf(ROPE_THETA, -(2.0f*i)/(float)Dh);
  float ang = (float)t * inv_freq;
  sin_tbl[(size_t)t*(size_t)(Dh/2) + (size_t)i] = sinf(ang);
  cos_tbl[(size_t)t*(size_t)(Dh/2) + (size_t)i] = cosf(ang);
}
__global__ void rope_apply_grad(float* dQ, float* dK, const float* sin_tbl, const float* cos_tbl, int B, int T){
  int bt = blockIdx.y * blockDim.y + threadIdx.y;
  int h  = blockIdx.z;
  int i2 = blockIdx.x * blockDim.x + threadIdx.x;
  if(bt>=B*T || h>=H || i2>=Dh/2) return;
  int t = bt % T;

  float s = sin_tbl[(size_t)t*(size_t)(Dh/2) + (size_t)i2];
  float c = cos_tbl[(size_t)t*(size_t)(Dh/2) + (size_t)i2];

  int base = bt*Dhf + h*Dh;
  int i0 = 2*i2;
  int i1 = i0+1;

  float dq0 = dQ[(size_t)base + (size_t)i0];
  float dq1 = dQ[(size_t)base + (size_t)i1];
  float dk0 = dK[(size_t)base + (size_t)i0];
  float dk1 = dK[(size_t)base + (size_t)i1];

  dQ[(size_t)base + (size_t)i0] = dq0*c + dq1*s;
  dQ[(size_t)base + (size_t)i1] = -dq0*s + dq1*c;

  dK[(size_t)base + (size_t)i0] = dk0*c + dk1*s;
  dK[(size_t)base + (size_t)i1] = -dk0*s + dk1*c;
}

// ============ warp reduce helpers for 16-lane group ============
__device__ __forceinline__ float shfl_xor_masked(float v, int laneMask, unsigned mask){
  return __shfl_xor_sync(mask, v, laneMask);
}
__device__ __forceinline__ float reduce_sum16(float v, unsigned mask){
  v += shfl_xor_masked(v, 8, mask);
  v += shfl_xor_masked(v, 4, mask);
  v += shfl_xor_masked(v, 2, mask);
  v += shfl_xor_masked(v, 1, mask);
  return v;
}

// ================= FlashAttention forward/backward (tiled causal) =================
__global__ void flash_fwd_wmma_hf(float* O, float* m_out, float* lse_out,
                                  const float* Q, const float* K, const float* Vv,
                                  int B, int T) {
  int b = blockIdx.x;
  int h = blockIdx.y;
  int qtile = blockIdx.z;
  const int QT = 128; // 8 warps * 16
  int q0 = qtile * QT;
  if(b >= B || h >= H || q0 >= T) return;

  int tid = threadIdx.x;
  int warp = tid >> 5;
  int lane = tid & 31;

  int tq0 = q0 + warp * 16;
  
  __shared__ half Qs[128][16];
  __shared__ half Ks[FA_KT][16];
  __shared__ half Vs[FA_KT][16];
  __shared__ float S_warp[8][16][16];
  __shared__ half  P_warp[8][16][16];
  __shared__ float O_warp[8][16][16];
  __shared__ float temp_O[8][16][16];

  if (lane < 16) {
     for(int c=0; c<16; c++) O_warp[warp][lane][c] = 0.f;
  }

  for(int i=0; i<8; i++){
    int idx = tid + i*256;
    if (idx < 128 * 16) {
       int r = idx / 16;
       int c = idx % 16;
       int t = q0 + r;
       if (t < T) {
         Qs[r][c] = __float2half_rn(Q[(size_t)(b*T + t)*Dhf + (size_t)h*Dh + c]);
       } else {
         Qs[r][c] = __float2half_rn(0.f);
       }
    }
  }
  __syncthreads();

  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_q;
  if (tq0 < T) {
    wmma::load_matrix_sync(frag_q, &Qs[warp*16][0], 16);
  }

  float mi = -1e30f;
  float li = 0.f;
  float scale = 1.0f / sqrtf((float)Dh);

  for (int k0 = 0; k0 < T; k0 += FA_KT) {
    int kend = (k0 + FA_KT < T) ? k0 + FA_KT : T;
    
    for(int i=0; i<8; i++){
      int idx = tid + i*256;
      if (idx < FA_KT * 16) {
         int r = idx / 16;
         int c = idx % 16;
         int t = k0 + r;
         if (t < kend) {
           Ks[r][c] = __float2half_rn(K[(size_t)(b*T + t)*Dhf + (size_t)h*Dh + c]);
           Vs[r][c] = __float2half_rn(Vv[(size_t)(b*T + t)*Dhf + (size_t)h*Dh + c]);
         } else {
           Ks[r][c] = __float2half_rn(0.f);
           Vs[r][c] = __float2half_rn(0.f);
         }
      }
    }
    __syncthreads();

    if (tq0 < T) {
      int warp_max_k = (tq0 + 15 < kend) ? tq0 + 15 : kend - 1;
      if (warp_max_k >= k0) {
        for (int k_sub = 0; k_sub < FA_KT; k_sub += 16) {
          if (k0 + k_sub > warp_max_k) break;

          wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_s;
          wmma::fill_fragment(frag_s, 0.0f);
          wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_k;
          wmma::load_matrix_sync(frag_k, &Ks[k_sub][0], 16);
          wmma::mma_sync(frag_s, frag_q, frag_k, frag_s);
          wmma::store_matrix_sync(&S_warp[warp][0][0], frag_s, 16, wmma::mem_row_major);

          __syncwarp();

          if (lane < 16) {
             int q_idx = tq0 + lane;
             float row_max = -1e30f;
             for (int c = 0; c < 16; c++) {
                int k_idx = k0 + k_sub + c;
                if (k_idx > q_idx || k_idx >= T) {
                   S_warp[warp][lane][c] = -1e30f;
                } else {
                   float score = S_warp[warp][lane][c] * scale;
                   S_warp[warp][lane][c] = score;
                   if (score > row_max) row_max = score;
                }
             }

             float m_new = fmaxf(mi, row_max);
             float alpha = expf(mi - m_new);

             for (int c = 0; c < 16; c++) {
                O_warp[warp][lane][c] *= alpha;
             }

             float row_sum = 0.f;
             for (int c = 0; c < 16; c++) {
                float p = expf(S_warp[warp][lane][c] - m_new);
                P_warp[warp][lane][c] = __float2half_rn(p);
                row_sum += p;
             }
             
             li = li * alpha + row_sum;
             mi = m_new;
          }
          __syncwarp();

          wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_p;
          wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_v;
          wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_o_temp;
          wmma::fill_fragment(frag_o_temp, 0.0f);

          wmma::load_matrix_sync(frag_p, &P_warp[warp][0][0], 16);
          wmma::load_matrix_sync(frag_v, &Vs[k_sub][0], 16);
          wmma::mma_sync(frag_o_temp, frag_p, frag_v, frag_o_temp);
          wmma::store_matrix_sync(&temp_O[warp][0][0], frag_o_temp, 16, wmma::mem_row_major);

          __syncwarp();
          if (lane < 16) {
             for (int c = 0; c < 16; c++) {
                O_warp[warp][lane][c] += temp_O[warp][lane][c];
             }
          }
          __syncwarp();
        }
      }
    }
    __syncthreads();
  }

  if (tq0 < T && lane < 16) {
      int q_idx = tq0 + lane;
      if (q_idx < T) {
         float inv = 1.0f / li;
         for (int c = 0; c < 16; c++) {
            O[(size_t)(b*T + q_idx)*Dhf + (size_t)h*Dh + c] = O_warp[warp][lane][c] * inv;
         }
         int idx = (b*H + h)*T + q_idx;
         m_out[idx] = mi;
         lse_out[idx] = li;
      }
  }
}

__global__ void flash_bwd_dq_wmma_hf(float* dpSum, float* dQ,
                                const float* dO, const float* Q, const float* K, const float* Vv,
                                const float* m, const float* lse,
                                int B, int T){
  int b = blockIdx.x;
  int h = blockIdx.y;
  int qtile = blockIdx.z;
  int q0 = qtile * FA_QT;
  if(b >= B || h >= H) return;
  int tid = threadIdx.x;
  int warp = tid >> 5;
  int lane = tid & 31;
  int half_warp = lane >> 4;

  int tq = q0 + (tid >> 4);
  if(tq >= T) return;

  int btq = b*T + tq;
  const float* qptr = Q + (size_t)btq*(size_t)Dhf + (size_t)h*(size_t)Dh;
  const float* doptr= dO+ (size_t)btq*(size_t)Dhf + (size_t)h*(size_t)Dh;
  float qd = qptr[lane&15];
  float dod= doptr[lane&15];

  int midx = ((b*H+h)*T+tq);
  float mi = m[midx];
  float inv_l = 1.0f / lse[midx];
  float scale = 1.0f/sqrtf((float)Dh);

  __shared__ half Ks[FA_KT][Dh+16];
  __shared__ half Vs[FA_KT][Dh+16];

  float dp = 0.f;
  for(int k0=0; k0<T; k0+=FA_KT){
    int kend=(k0+FA_KT<T)?k0+FA_KT:T;
    int kt=kend-k0;
    
    for(int idx = tid; idx < FA_KT * Dh; idx += 256){
      int kk = idx / Dh;
      int d = idx % Dh;
      int t = k0 + kk;
      if (t < kend) {
        int btk = b * T + t;
        float k = K[btk * Dhf + h * Dh + d];
        float v = Vv[btk* Dhf + h * Dh + d];
        Ks[kk][d] = __float2half_rn(k);
        Vs[kk][d] = __float2half_rn(v);
      } else {
        Ks[kk][d] = __float2half_rn(0.f);
        Vs[kk][d] = __float2half_rn(0.f);
      }
    }
    __syncthreads();

    int maxk = (tq < kend-1) ? (tq-k0+1) : kt;
    if(maxk>0){
      for(int kk=0; kk<maxk; kk++){
        float k_val = __half2float(Ks[kk][lane&15]);
        float v_val = __half2float(Vs[kk][lane&15]);

        float dotqk = qd * k_val;
        unsigned mask = (half_warp == 0) ? 0x0000FFFFu : 0xFFFF0000u;
        dotqk += __shfl_xor_sync(mask, dotqk, 8);
        dotqk += __shfl_xor_sync(mask, dotqk, 4);
        dotqk += __shfl_xor_sync(mask, dotqk, 2);
        dotqk += __shfl_xor_sync(mask, dotqk, 1);

        float score = dotqk * scale;

        float dotdv = dod * v_val;
        dotdv += __shfl_xor_sync(mask, dotdv, 8);
        dotdv += __shfl_xor_sync(mask, dotdv, 4);
        dotdv += __shfl_xor_sync(mask, dotdv, 2);
        dotdv += __shfl_xor_sync(mask, dotdv, 1);

        float p = expf(score - mi) * inv_l;
        dp += p * dotdv;
      }
    }
    __syncthreads();
  }
  if(lane==0) dpSum[midx]=dp;

  float dqi = 0.f;
  for(int k0=0;k0<T;k0+=FA_KT){
    int kend=(k0+FA_KT<T)?k0+FA_KT:T;
    int kt=kend-k0;

    for(int idx = tid; idx < FA_KT * Dh; idx += 256){
      int kk = idx / Dh;
      int d = idx % Dh;
      int t = k0 + kk;
      if (t < kend) {
        int btk = b * T + t;
        float k = K[btk * Dhf + h * Dh + d];
        float v = Vv[btk* Dhf + h * Dh + d];
        Ks[kk][d] = __float2half_rn(k);
        Vs[kk][d] = __float2half_rn(v);
      } else {
        Ks[kk][d] = __float2half_rn(0.f);
        Vs[kk][d] = __float2half_rn(0.f);
      }
    }
    __syncthreads();

    int maxk = (tq < kend-1) ? (tq-k0+1) : kt;
    if(maxk>0){
      for(int kk=0; kk<maxk; kk++){
        float k_val = __half2float(Ks[kk][lane&15]);
        float v_val = __half2float(Vs[kk][lane&15]);

        float dotqk = qd * k_val;
        unsigned mask = (half_warp == 0) ? 0x0000FFFFu : 0xFFFF0000u;
        dotqk += __shfl_xor_sync(mask, dotqk, 8);
        dotqk += __shfl_xor_sync(mask, dotqk, 4);
        dotqk += __shfl_xor_sync(mask, dotqk, 2);
        dotqk += __shfl_xor_sync(mask, dotqk, 1);
        
        float score = dotqk * scale;

        float dotdv = dod * v_val;
        dotdv += __shfl_xor_sync(mask, dotdv, 8);
        dotdv += __shfl_xor_sync(mask, dotdv, 4);
        dotdv += __shfl_xor_sync(mask, dotdv, 2);
        dotdv += __shfl_xor_sync(mask, dotdv, 1);

        float p = expf(score - mi) * inv_l;
        float dS = (dotdv - dp) * p;
        dqi += dS * k_val;
      }
    }
    __syncthreads();
  }
  float* dqptr = dQ + (size_t)btq*(size_t)Dhf + (size_t)h*(size_t)Dh;
  dqptr[lane&15] += dqi * scale;
}

__global__ void flash_bwd_dkv_wmma_hf(float* dK, float* dV,
                                const float* dO, const float* Q, const float* K, const float* Vv,
                                const float* m, const float* lse, const float* dpSum,
                                int B, int T){
  int b = blockIdx.x;
  int h = blockIdx.y;
  int ktile = blockIdx.z;
  int k0 = ktile * FA_KT;
  if (b >= B || h >= H) return;
  int tid = threadIdx.x;
  int warp = tid >> 5;
  int lane = tid & 31;
  int half_warp = lane >> 4;

  __shared__ half Qs[FA_QT][Dh + 16];
  __shared__ half dOs[FA_QT][Dh + 16];

  float scale = 1.0f / sqrtf((float)Dh);

  for(int sub=0; sub<FA_KT/16; sub++){
    int s = k0 + sub*16 + (tid>>4);
    if(s >= T) continue;

    int bts = b*T + s;
    const float* kptr = K + (size_t)bts*(size_t)Dhf + (size_t)h*(size_t)Dh;
    const float* vptr = Vv + (size_t)bts*(size_t)Dhf + (size_t)h*(size_t)Dh;

    float kd = kptr[lane&15];
    float vd = vptr[lane&15];

    float dkd = 0.f;
    float dvd = 0.f;

    int q0 = (s / FA_QT) * FA_QT;
    for (; q0 < T; q0 += FA_QT) {
      for (int idx = tid; idx < FA_QT * Dh; idx += 256) {
        int qi = idx / Dh;
        int d = idx % Dh;
        int t = q0 + qi;
        if (t < T) {
          int btq = b * T + t;
          float q  = Q[btq * Dhf + h * Dh + d];
          float do_= dO[btq * Dhf + h * Dh + d];
          Qs[qi][d] = __float2half_rn(q);
          dOs[qi][d]= __float2half_rn(do_);
        } else {
          Qs[qi][d] = __float2half_rn(0.f);
          dOs[qi][d]= __float2half_rn(0.f);
        }
      }
      __syncthreads();

      for (int qi = 0; qi < FA_QT; qi++) {
        int t = q0 + qi;
        if (t >= T || t < s) continue;

        int idx = ((b * H + h) * T + t);
        float mi = m[idx];
        float inv_l = 1.0f / lse[idx];
        float dp = dpSum[idx];

        // scalar dot since K/V are floats in registers and Q/dO are in shared
        float q_val = __half2float(Qs[qi][lane&15]);
        float do_val= __half2float(dOs[qi][lane&15]);

        float dotqk = q_val * kd;
        unsigned mask = (half_warp == 0) ? 0x0000FFFFu : 0xFFFF0000u;
        dotqk += __shfl_xor_sync(mask, dotqk, 8);
        dotqk += __shfl_xor_sync(mask, dotqk, 4);
        dotqk += __shfl_xor_sync(mask, dotqk, 2);
        dotqk += __shfl_xor_sync(mask, dotqk, 1);
        
        float score = dotqk * scale;

        float dotdv = do_val * vd;
        dotdv += __shfl_xor_sync(mask, dotdv, 8);
        dotdv += __shfl_xor_sync(mask, dotdv, 4);
        dotdv += __shfl_xor_sync(mask, dotdv, 2);
        dotdv += __shfl_xor_sync(mask, dotdv, 1);

        float p = expf(score - mi) * inv_l;
        float dS = (dotdv - dp) * p;

        dvd += p * do_val;
        dkd += dS * q_val * scale;
      }
      __syncthreads();
    }
    float* dkptr = dK + (size_t)bts*(size_t)Dhf + (size_t)h*(size_t)Dh;
    float* dvptr = dV + (size_t)bts*(size_t)Dhf + (size_t)h*(size_t)Dh;
    dkptr[lane&15] += dkd;
    dvptr[lane&15] += dvd;
  }
}


// ================= embeddings (split/concat) =================
__global__ void embed_split(float* y1,float* y2, const float* wte, const float* wpe, const uint16_t* tok, int N, int T){
  int n=blockIdx.y*blockDim.y+threadIdx.y;
  int d=blockIdx.x*blockDim.x+threadIdx.x;
  if(n>=N||d>=D) return;
  int t=n%T;
  int id=(int)tok[n];
  float v = wte[(size_t)id*(size_t)D+(size_t)d] + wpe[(size_t)t*(size_t)D+(size_t)d];
  if(d < Dhf) y1[(size_t)n*(size_t)Dhf + (size_t)d] = v;
  else        y2[(size_t)n*(size_t)Dhf + (size_t)(d-Dhf)] = v;
}
__global__ void concat_full(float* Xfull, const float* y1, const float* y2, int N){
  int n=blockIdx.y*blockDim.y+threadIdx.y;
  int d=blockIdx.x*blockDim.x+threadIdx.x;
  if(n>=N||d>=D) return;
  float v = (d < Dhf) ? y1[(size_t)n*(size_t)Dhf+(size_t)d]
                      : y2[(size_t)n*(size_t)Dhf+(size_t)(d-Dhf)];
  Xfull[(size_t)n*(size_t)D+(size_t)d]=v;
}
__global__ void split_full(float* y1, float* y2, const float* Xfull, int N){
  int n=blockIdx.y*blockDim.y+threadIdx.y;
  int d=blockIdx.x*blockDim.x+threadIdx.x;
  if(n>=N||d>=D) return;
  float v=Xfull[(size_t)n*(size_t)D+(size_t)d];
  if(d < Dhf) y1[(size_t)n*(size_t)Dhf+(size_t)d]=v;
  else        y2[(size_t)n*(size_t)Dhf+(size_t)(d-Dhf)]=v;
}
__global__ void embed_bwd(float* gwte, float* gwpe, const uint16_t* tok, const float* dy1, const float* dy2, int N, int T){
  int n=blockIdx.y*blockDim.y+threadIdx.y;
  int d=blockIdx.x*blockDim.x+threadIdx.x;
  if(n>=N||d>=D) return;
  int t=n%T;
  int id=(int)tok[n];
  float g = (d < Dhf) ? dy1[(size_t)n*(size_t)Dhf + (size_t)d]
                      : dy2[(size_t)n*(size_t)Dhf + (size_t)(d-Dhf)];
  atomicAdd(&gwte[(size_t)id*(size_t)D + (size_t)d], g);
  atomicAdd(&gwpe[(size_t)t*(size_t)D + (size_t)d], g);
}

// ================= clip + adamw (device clip scale) =================
__global__ void reduce_sumsq_1(const float* g,float* partial,int n){
  __shared__ float buf[256];
  int tid=threadIdx.x;
  int idx=blockIdx.x*blockDim.x+tid;
  int stride=blockDim.x*gridDim.x;
  float s=0.f;
  for(int i=idx;i<n;i+=stride){ float x=g[i]; if(isfinite(x)) s+=x*x; }
  buf[tid]=s; __syncthreads();
  for(int k=blockDim.x/2;k>0;k>>=1){ if(tid<k) buf[tid]+=buf[tid+k]; __syncthreads(); }
  if(tid==0) partial[blockIdx.x]=buf[0];
}
__global__ void reduce_sumsq_2(const float* partial,float* out,int n){
  __shared__ float buf[256];
  int tid=threadIdx.x;
  float s=0.f;
  for(int i=tid;i<n;i+=blockDim.x) s+=partial[i];
  buf[tid]=s; __syncthreads();
  for(int k=blockDim.x/2;k>0;k>>=1){ if(tid<k) buf[tid]+=buf[tid+k]; __syncthreads(); }
  if(tid==0) out[0]=buf[0];
}
__global__ void compute_clip_scale(float* out_scale, const float* sumsq, float clip){
  float h = sumsq[0];
  float norm = sqrtf(h);
  float s = 1.f;
  if(!(isfinite(norm)) || norm<=0.f) s = 1.f;
  else if(norm > clip) s = clip / norm;
  out_scale[0] = s;
}
__global__ void adamw(float* w,float* m,float* v,const float* g,int n,
                      float lr,float wd,float b1,float b2,float eps,
                      float inv_b1t,float inv_b2t, const float* clip_scale_dev)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=n) return;
  float clip_scale = clip_scale_dev[0];
  float gi=g[i]*clip_scale;
  if(!isfinite(gi)) gi=0.f;
  float mi=m[i]=b1*m[i]+(1.f-b1)*gi;
  float vi=v[i]=b2*v[i]+(1.f-b2)*gi*gi;
  float mhat=mi*inv_b1t;
  float vhat=vi*inv_b2t;
  w[i] -= lr*(mhat/(sqrtf(vhat)+eps) + wd*w[i]);
}

// ================= HW mirrors =================
struct HW {
  half *wte_rm,*wte_tr;
  half *Wout_rm,*Wout_tr;
  half *Wq_rm[L],*Wq_tr[L];
  half *Wk_rm[L],*Wk_tr[L];
  half *Wv_rm[L],*Wv_tr[L];
  half *Wo_rm[L],*Wo_tr[L];
  half *W1_rm[L],*W1_tr[L];
  half *W2_rm[L],*W2_tr[L];
};

// ================= FUSED MEGA-KERNEL (RMS + QKV + RoPE on Dhf=128) =================
__device__ __forceinline__ float4 ld_cg_f4(const float4* p){
  float4 v;
  asm volatile ("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];"
    : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
    : "l"(p));
  return v;
}

template<int WARPS_PER_BLOCK>
__global__ __launch_bounds__(256) void fused_rms_qkv_rope_hf(
    float* __restrict__ Q,
    float* __restrict__ K,
    float* __restrict__ Vv,
    float* __restrict__ rms_inv,          // [N]
    float* __restrict__ Nnorm_or_null,    // [N,Dhf] or nullptr
    const float* __restrict__ X,          // [N,Dhf]
    const float* __restrict__ Wq,         // [Dhf,Dhf] row-major
    const float* __restrict__ Wk,         // [Dhf,Dhf] row-major
    const float* __restrict__ Wv,         // [Dhf,Dhf] row-major
    const float* __restrict__ gamma,      // [Dhf]
    const float* __restrict__ sin_tbl,    // [Tmax, Dh/2]
    const float* __restrict__ cos_tbl,    // [Tmax, Dh/2]
    int N, int T)
{
  int tid = (int)threadIdx.x;
  int warp = tid >> 5;
  int lane = tid & 31;
  int n = (int)blockIdx.x * WARPS_PER_BLOCK + warp;
  if(n >= N) return;

  int tpos = n % T;
  constexpr int COLS_PER_LANE = Dhf / 32; // 4
  int col_base = lane * COLS_PER_LANE;

  const float* Xn = X + (size_t)n*(size_t)Dhf;
  float* Qn = Q + (size_t)n*(size_t)Dhf;
  float* Kn = K + (size_t)n*(size_t)Dhf;
  float* Vn = Vv + (size_t)n*(size_t)Dhf;

  const float4* Xf4 = (const float4*)Xn;
  float4 x4 = ld_cg_f4(Xf4 + lane);

  float ss = x4.x*x4.x + x4.y*x4.y + x4.z*x4.z + x4.w*x4.w;
  #pragma unroll
  for(int off=16; off>0; off>>=1) ss += __shfl_down_sync(0xFFFFFFFFu, ss, off);
  float sumsq = __shfl_sync(0xFFFFFFFFu, ss, 0);

  float inv = rsqrtf(sumsq * (1.0f/(float)Dhf) + EPS);
  if(lane==0) rms_inv[n] = inv;

  float g0 = gamma[col_base+0];
  float g1 = gamma[col_base+1];
  float g2 = gamma[col_base+2];
  float g3 = gamma[col_base+3];

  float xn0 = x4.x * inv * g0;
  float xn1 = x4.y * inv * g1;
  float xn2 = x4.z * inv * g2;
  float xn3 = x4.w * inv * g3;

  if(Nnorm_or_null){
    float4* Nf4 = (float4*)(Nnorm_or_null + (size_t)n*(size_t)Dhf);
    Nf4[lane] = make_float4(xn0,xn1,xn2,xn3);
  }

  float q0=0.f,q1=0.f,q2=0.f,q3=0.f;
  float k0=0.f,k1v=0.f,k2v=0.f,k3v=0.f;
  float v0=0.f,v1=0.f,v2=0.f,v3=0.f;

  for(int kk=0; kk<Dhf; kk++){
    int src_lane = kk >> 2;
    int off = kk & 3;
    float xk;
    if(off==0) xk = __shfl_sync(0xFFFFFFFFu, xn0, src_lane);
    else if(off==1) xk = __shfl_sync(0xFFFFFFFFu, xn1, src_lane);
    else if(off==2) xk = __shfl_sync(0xFFFFFFFFu, xn2, src_lane);
    else            xk = __shfl_sync(0xFFFFFFFFu, xn3, src_lane);

    const float4* wqf4 = (const float4*)(Wq + (size_t)kk*(size_t)Dhf + (size_t)col_base);
    const float4* wkf4 = (const float4*)(Wk + (size_t)kk*(size_t)Dhf + (size_t)col_base);
    const float4* wvf4 = (const float4*)(Wv + (size_t)kk*(size_t)Dhf + (size_t)col_base);

    float4 wq4 = ld_cg_f4(wqf4);
    float4 wk4 = ld_cg_f4(wkf4);
    float4 wv4 = ld_cg_f4(wvf4);

    q0 = fmaf(xk, wq4.x, q0); q1 = fmaf(xk, wq4.y, q1); q2 = fmaf(xk, wq4.z, q2); q3 = fmaf(xk, wq4.w, q3);
    k0 = fmaf(xk, wk4.x, k0); k1v= fmaf(xk, wk4.y, k1v); k2v= fmaf(xk, wk4.z, k2v); k3v= fmaf(xk, wk4.w, k3v);
    v0 = fmaf(xk, wv4.x, v0); v1 = fmaf(xk, wv4.y, v1); v2 = fmaf(xk, wv4.z, v2); v3 = fmaf(xk, wv4.w, v3);
  }

  auto rope_pair = [&](int d_even, float& qa0, float& qa1, float& ka0, float& ka1){
    int within = d_even & (Dh-1);
    int i2 = within >> 1;
    float s = sin_tbl[(size_t)tpos*(size_t)(Dh/2) + (size_t)i2];
    float c = cos_tbl[(size_t)tpos*(size_t)(Dh/2) + (size_t)i2];
    float qx=qa0, qy=qa1;
    float kx=ka0, ky=ka1;
    qa0 = qx*c - qy*s;
    qa1 = qx*s + qy*c;
    ka0 = kx*c - ky*s;
    ka1 = kx*s + ky*c;
  };
  rope_pair(col_base+0, q0,q1, k0,k1v);
  rope_pair(col_base+2, q2,q3, k2v,k3v);

  float4* Qf4 = (float4*)Qn;
  float4* Kf4 = (float4*)Kn;
  float4* Vf4 = (float4*)Vn;
  Qf4[lane] = make_float4(q0,q1,q2,q3);
  Kf4[lane] = make_float4(k0,k1v,k2v,k3v);
  Vf4[lane] = make_float4(v0,v1,v2,v3);
}

// ================= FUSED MEGA-KERNEL (Attn-Epilogue + RMS + FFN on Dhf=128) =================
template<int WARPS_PER_BLOCK>
__global__ __launch_bounds__(256) void fused_rms_gelu_ffn_hf(
    float* __restrict__ Y1,               // [N,Dhf]
    float* __restrict__ Y2,               // [N,Dhf]
    float* __restrict__ rms_inv,          // [N]
    float* __restrict__ Nnorm_or_null,    // [N,Dhf] or nullptr
    const float* __restrict__ AttentionO, // [N,Dhf]
    const float* __restrict__ Wo_rm,      // [Dhf,Dhf] row-major
    const float* __restrict__ W1_rm,      // [F,Dhf] row-major
    const float* __restrict__ W2_rm,      // [Dhf,F] row-major
    const float* __restrict__ gamma,      // [Dhf]
    int N, int F_dim)
{
  int tid = (int)threadIdx.x;
  int warp = tid >> 5;
  int lane = tid & 31;
  int n = (int)blockIdx.x * WARPS_PER_BLOCK + warp;
  if(n >= N) return;

  constexpr int COLS_PER_LANE = Dhf / 32; // 4
  int col_base = lane * COLS_PER_LANE;

  const float* On = AttentionO + (size_t)n*(size_t)Dhf;
  float* Y1n = Y1 + (size_t)n*(size_t)Dhf;

  float4 o4 = ld_cg_f4((const float4*)(On + col_base));
  float o_col0 = o4.x, o_col1 = o4.y, o_col2 = o4.z, o_col3 = o4.w;

  float wo0=0.f, wo1=0.f, wo2=0.f, wo3=0.f;

  for(int kk=0; kk<Dhf; kk++){
    int src_lane = kk >> 2;
    int off = kk & 3;
    float ok;
    if(off==0) ok = __shfl_sync(0xFFFFFFFFu, o_col0, src_lane);
    else if(off==1) ok = __shfl_sync(0xFFFFFFFFu, o_col1, src_lane);
    else if(off==2) ok = __shfl_sync(0xFFFFFFFFu, o_col2, src_lane);
    else            ok = __shfl_sync(0xFFFFFFFFu, o_col3, src_lane);

    float4 wo_row = ld_cg_f4((const float4*)(Wo_rm + (size_t)kk*(size_t)Dhf + (size_t)col_base));
    wo0 = fmaf(ok, wo_row.x, wo0);
    wo1 = fmaf(ok, wo_row.y, wo1);
    wo2 = fmaf(ok, wo_row.z, wo2);
    wo3 = fmaf(ok, wo_row.w, wo3);
  }

  float4 y1_in = ld_cg_f4((const float4*)(Y1n + col_base));
  float y1_0 = y1_in.x + wo0;
  float y1_1 = y1_in.y + wo1;
  float y1_2 = y1_in.z + wo2;
  float y1_3 = y1_in.w + wo3;
  ((float4*)(Y1n + col_base))[0] = make_float4(y1_0, y1_1, y1_2, y1_3);

  float ss = y1_0*y1_0 + y1_1*y1_1 + y1_2*y1_2 + y1_3*y1_3;
  #pragma unroll
  for(int off=16; off>0; off>>=1) ss += __shfl_down_sync(0xFFFFFFFFu, ss, off);
  float sumsq = __shfl_sync(0xFFFFFFFFu, ss, 0);

  float inv = rsqrtf(sumsq * (1.0f/(float)Dhf) + EPS);
  if(lane==0) rms_inv[n] = inv;

  float g0 = gamma[col_base+0];
  float g1 = gamma[col_base+1];
  float g2 = gamma[col_base+2];
  float g3 = gamma[col_base+3];

  float xn0 = y1_0 * inv * g0;
  float xn1 = y1_1 * inv * g1;
  float xn2 = y1_2 * inv * g2;
  float xn3 = y1_3 * inv * g3;

  if(Nnorm_or_null){
    float4* Nf4 = (float4*)(Nnorm_or_null + (size_t)n*(size_t)Dhf);
    Nf4[lane] = make_float4(xn0,xn1,xn2,xn3);
  }

  const float c=0.7978845608f;
  float ffn_out[4] = {0.f};

  for(int f0=0; f0<F_dim; f0+=32){
    int fcol = f0 + lane;
    float dotW1 = 0.f;
    for(int kk=0; kk<Dhf; kk++){
      int src_lane = kk >> 2;
      int off = kk & 3;
      float xk;
      if(off==0) xk = __shfl_sync(0xFFFFFFFFu, xn0, src_lane);
      else if(off==1) xk = __shfl_sync(0xFFFFFFFFu, xn1, src_lane);
      else if(off==2) xk = __shfl_sync(0xFFFFFFFFu, xn2, src_lane);
      else            xk = __shfl_sync(0xFFFFFFFFu, xn3, src_lane);
      float w1_val = W1_rm[(size_t)kk*(size_t)F_dim + (size_t)fcol];
      dotW1 = fmaf(xk, w1_val, dotW1);
    }
    
    float x_gelu = dotW1;
    float x3 = x_gelu*x_gelu*x_gelu;
    float u = 0.5f*x_gelu*(1.f+tanhf(c*(x_gelu+0.044715f*x3)));

    for(int i=0; i<4; i++){
      float w2_val = W2_rm[(size_t)fcol*(size_t)Dhf + (size_t)col_base + i];
      ffn_out[i] = fmaf(u, w2_val, ffn_out[i]);
    }
  }

  __shfl_sync(0xFFFFFFFFu, 0, 0); // barrier safety
  float f0=0.f, f1=0.f, f2=0.f, f3=0.f;
  for(int l=0; l<32; l++){
    float fo0 = __shfl_sync(0xFFFFFFFFu, ffn_out[0], l);
    float fo1 = __shfl_sync(0xFFFFFFFFu, ffn_out[1], l);
    float fo2 = __shfl_sync(0xFFFFFFFFu, ffn_out[2], l);
    float fo3 = __shfl_sync(0xFFFFFFFFu, ffn_out[3], l);
    f0 += fo0; f1 += fo1; f2 += fo2; f3 += fo3;
  }

  float* Y2n = Y2 + (size_t)n*(size_t)Dhf;
  float4 y2_in = ld_cg_f4((const float4*)(Y2n + col_base));
  ((float4*)(Y2n + col_base))[0] = make_float4(y2_in.x + f0, y2_in.y + f1, y2_in.z + f2, y2_in.w + f3);
}

// ================= GPU container =================
struct GPU {
  int dev;
  int B,T,N;

  float *dW,*dG,*mW,*vW;
  WView W,G,MW,VW;
  HW Hw;

  uint16_t *tok,*tgt;
  uint16_t *htok_h,*htgt_h;

  float *y1,*y2,*x1,*x2;
  float *dy1,*dy2,*dx1,*dx2;

  float *inv;
  float *n;
  half  *n_h;

  float *Q,*K,*Vh,*O;
  float *matt,*latt,*dp;
  float *dQ,*dK,*dVh;
  float *dfout,*dOattn;

  float *U,*A,*dU,*dA;
  float *gout;
  half  *A_h;

  float *fout;

  float *Xfull,*Xnorm,*invF;
  half  *Xnorm_h;
  float *Loss;
  float *row_max,*row_sum,*chunk_max,*chunk_sum;
  float *logits_chunk,*dY_chunk;
  half  *scratchHalf_head,*Wout_rm_chunk;
  float *dWout_chunk;
  float *dXnorm,*dXfull;

  half *Atr,*dYtr,*scratchHalf;

  float *partial,*sumsq,*clip_scale_dev;

  float *sin_tbl,*cos_tbl;

  float *ring_tmp;

  cudaStream_t comm = nullptr;

  int graph_built = 0;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graphExec = nullptr;

  float *loss_mean = nullptr;
};

static void init_weights_cpu(std::vector<float>& w, uint64_t seed){
  RNG r{seed?seed:123ULL};
  for(size_t i=0;i<w.size();i++) w[i]=frand11(&r)*0.02f;
  WView W{}; pack_W(w.data(), &W);
  for(int l=0;l<L;l++){
    for(int i=0;i<Dhf;i++){ W.gf[l][i]=1.f; W.gg[l][i]=1.f; }
  }
  for(int i=0;i<D;i++) W.gout[i]=1.f;
}

static void gpu_alloc(GPU* g,int dev,int B,int T){
  g->dev=dev; g->B=B; g->T=T; g->N=B*T;
  CUDA_CHECK(cudaSetDevice(dev));

  size_t Wn=weights_floats();
  CUDA_CHECK(cudaMalloc(&g->dW, Wn*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&g->dG, Wn*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&g->mW, Wn*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&g->vW, Wn*sizeof(float)));
  CUDA_CHECK(cudaMemset(g->dG,0,Wn*sizeof(float)));
  CUDA_CHECK(cudaMemset(g->mW,0,Wn*sizeof(float)));
  CUDA_CHECK(cudaMemset(g->vW,0,Wn*sizeof(float)));
  pack_W(g->dW,&g->W); pack_W(g->dG,&g->G);
  pack_W(g->mW,&g->MW); pack_W(g->vW,&g->VW);

  CUDA_CHECK(cudaMalloc(&g->tok,(size_t)g->N*sizeof(uint16_t)));
  CUDA_CHECK(cudaMalloc(&g->tgt,(size_t)g->N*sizeof(uint16_t)));

  CUDA_CHECK(cudaHostAlloc(&g->htok_h, (size_t)g->N*sizeof(uint16_t), cudaHostAllocDefault));
  CUDA_CHECK(cudaHostAlloc(&g->htgt_h, (size_t)g->N*sizeof(uint16_t), cudaHostAllocDefault));

  auto mal=[&](float** p,size_t n){ CUDA_CHECK(cudaMalloc(p,n*sizeof(float))); };
  auto malh=[&](half** p,size_t n){ CUDA_CHECK(cudaMalloc(p,n*sizeof(half))); };

  mal(&g->y1,(size_t)g->N*(size_t)Dhf); mal(&g->y2,(size_t)g->N*(size_t)Dhf);
  mal(&g->x1,(size_t)g->N*(size_t)Dhf); mal(&g->x2,(size_t)g->N*(size_t)Dhf);
  mal(&g->dy1,(size_t)g->N*(size_t)Dhf); mal(&g->dy2,(size_t)g->N*(size_t)Dhf);
  mal(&g->dx1,(size_t)g->N*(size_t)Dhf); mal(&g->dx2,(size_t)g->N*(size_t)Dhf);

  mal(&g->inv,(size_t)g->N);
  mal(&g->n,(size_t)g->N*(size_t)Dhf);
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

  CUDA_CHECK(cudaMalloc(&g->partial,(size_t)256*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&g->sumsq,sizeof(float)));
  CUDA_CHECK(cudaMalloc(&g->clip_scale_dev,sizeof(float)));
  CUDA_CHECK(cudaMalloc(&g->loss_mean,sizeof(float)));

  CUDA_CHECK(cudaMalloc(&g->sin_tbl, (size_t)Tmax*(size_t)(Dh/2)*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&g->cos_tbl, (size_t)Tmax*(size_t)(Dh/2)*sizeof(float)));
  dim3 blk(16,16);
  dim3 grd((Dh/2 + 15)/16, (Tmax + 15)/16);
  rope_build_tables<<<grd,blk>>>(g->sin_tbl, g->cos_tbl, Tmax);
  KERNEL_CHECK();

  CUDA_CHECK(cudaStreamCreateWithFlags(&g->comm, cudaStreamNonBlocking));

  auto malw=[&](half** p,size_t n){ CUDA_CHECK(cudaMalloc(p,n*sizeof(half))); };
  malw(&g->Hw.wte_rm,(size_t)Vpad*(size_t)D);
  malw(&g->Hw.wte_tr,(size_t)D*(size_t)Vpad);
  malw(&g->Hw.Wout_rm,(size_t)D*(size_t)Vpad);
  malw(&g->Hw.Wout_tr,(size_t)Vpad*(size_t)D);
  for(int l=0;l<L;l++){
    malw(&g->Hw.Wq_rm[l],(size_t)Dhf*(size_t)Dhf); malw(&g->Hw.Wq_tr[l],(size_t)Dhf*(size_t)Dhf);
    malw(&g->Hw.Wk_rm[l],(size_t)Dhf*(size_t)Dhf); malw(&g->Hw.Wk_tr[l],(size_t)Dhf*(size_t)Dhf);
    malw(&g->Hw.Wv_rm[l],(size_t)Dhf*(size_t)Dhf); malw(&g->Hw.Wv_tr[l],(size_t)Dhf*(size_t)Dhf);
    malw(&g->Hw.Wo_rm[l],(size_t)Dhf*(size_t)Dhf); malw(&g->Hw.Wo_tr[l],(size_t)Dhf*(size_t)Dhf);
    malw(&g->Hw.W1_rm[l],(size_t)Dhf*(size_t)F);   malw(&g->Hw.W1_tr[l],(size_t)F*(size_t)Dhf);
    malw(&g->Hw.W2_rm[l],(size_t)F*(size_t)Dhf);   malw(&g->Hw.W2_tr[l],(size_t)Dhf*(size_t)F);
  }
}

static void gpu_free(GPU* g){
  CUDA_CHECK(cudaSetDevice(g->dev));
  auto cf=[&](void* p){ if(p) cudaFree(p); };
  cf(g->dW); cf(g->dG); cf(g->mW); cf(g->vW);
  cf(g->tok); cf(g->tgt);
  if(g->htok_h) CUDA_CHECK(cudaFreeHost(g->htok_h));
  if(g->htgt_h) CUDA_CHECK(cudaFreeHost(g->htgt_h));
  cf(g->y1); cf(g->y2); cf(g->x1); cf(g->x2);
  cf(g->dy1); cf(g->dy2); cf(g->dx1); cf(g->dx2);
  cf(g->inv); cf(g->n); cf(g->n_h);
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
  cf(g->partial); cf(g->sumsq); cf(g->clip_scale_dev);
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

static void refresh_half_weights(GPU* g){
  CUDA_CHECK(cudaSetDevice(g->dev));
  dim3 blk(16,16);

  dim3 grdWte((D+15)/16,(Vpad+15)/16);
  w_f2h_rm_tr<<<grdWte,blk>>>(g->Hw.wte_rm, g->Hw.wte_tr, g->W.wte, Vpad, D); KERNEL_CHECK();

  dim3 grdWout((Vpad+15)/16,(D+15)/16);
  w_f2h_rm_tr<<<grdWout,blk>>>(g->Hw.Wout_rm, g->Hw.Wout_tr, g->W.Wout, D, Vpad); KERNEL_CHECK();

  for(int l=0;l<L;l++){
    dim3 grdHH((Dhf+15)/16,(Dhf+15)/16);
    w_f2h_rm_tr<<<grdHH,blk>>>(g->Hw.Wq_rm[l], g->Hw.Wq_tr[l], g->W.Wq[l], Dhf, Dhf); KERNEL_CHECK();
    w_f2h_rm_tr<<<grdHH,blk>>>(g->Hw.Wk_rm[l], g->Hw.Wk_tr[l], g->W.Wk[l], Dhf, Dhf); KERNEL_CHECK();
    w_f2h_rm_tr<<<grdHH,blk>>>(g->Hw.Wv_rm[l], g->Hw.Wv_tr[l], g->W.Wv[l], Dhf, Dhf); KERNEL_CHECK();
    w_f2h_rm_tr<<<grdHH,blk>>>(g->Hw.Wo_rm[l], g->Hw.Wo_tr[l], g->W.Wo[l], Dhf, Dhf); KERNEL_CHECK();

    dim3 grdW1((F+15)/16,(Dhf+15)/16);
    w_f2h_rm_tr<<<grdW1,blk>>>(g->Hw.W1_rm[l], g->Hw.W1_tr[l], g->W.W1[l], Dhf, F); KERNEL_CHECK();

    dim3 grdW2((Dhf+15)/16,(F+15)/16);
    w_f2h_rm_tr<<<grdW2,blk>>>(g->Hw.W2_rm[l], g->Hw.W2_tr[l], g->W.W2[l], F, Dhf); KERNEL_CHECK();
  }
}

static void adam_step(GPU* g, int step, float lr, float wd, float clip){
  CUDA_CHECK(cudaSetDevice(g->dev));
  int blocks=256;
  int n=(int)weights_floats();
  reduce_sumsq_1<<<blocks,256>>>(g->dG, g->partial, n); KERNEL_CHECK();
  reduce_sumsq_2<<<1,256>>>(g->partial, g->sumsq, blocks); KERNEL_CHECK();
  compute_clip_scale<<<1,1>>>(g->clip_scale_dev, g->sumsq, clip); KERNEL_CHECK();

  const float b1=0.9f,b2=0.999f,eps=1e-8f;
  float b1t=1.f-powf(b1,(float)step);
  float b2t=1.f-powf(b2,(float)step);
  float inv_b1t=1.f/b1t, inv_b2t=1.f/b2t;
  adamw<<<(n+255)/256,256>>>(g->dW,g->mW,g->vW,g->dG,n,lr,wd,b1,b2,eps,inv_b1t,inv_b2t,g->clip_scale_dev);
  KERNEL_CHECK();
}

// ===== DP reduce/bcast =====
__global__ void add_inplace_kernel(float* a, const float* b, int n){
  int i=blockIdx.x*blockDim.x + threadIdx.x;
  if(i<n) a[i]+=b[i];
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

  const size_t n = weights_floats();
  size_t chunk = (n + (size_t)G - 1) / (size_t)G;
  chunk = (chunk + 255u) & ~255u;

  for(int r=0;r<G;r++){
    CUDA_CHECK(cudaSetDevice(gpus[(size_t)r].dev));
    if(!gpus[(size_t)r].ring_tmp){
      CUDA_CHECK(cudaMalloc(&gpus[(size_t)r].ring_tmp, chunk*sizeof(float)));
    }
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
        cnt*sizeof(float), gpus[(size_t)r].comm));
      add_inplace_kernel<<<(int)((cnt+255)/256),256,0,gpus[(size_t)r].comm>>>(gpus[(size_t)r].dG + off, gpus[(size_t)r].ring_tmp, (int)cnt);
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
        cnt*sizeof(float), gpus[(size_t)r].comm));
    }
    for(int r=0;r<G;r++){
      CUDA_CHECK(cudaSetDevice(gpus[(size_t)r].dev));
      CUDA_CHECK(cudaStreamSynchronize(gpus[(size_t)r].comm));
    }
  }
  return true;
}
static void host_allreduce(std::vector<GPU>& gpus){
  size_t Wn=weights_floats();
  std::vector<float> sum(Wn,0.f);
  for(auto& g: gpus){
    CUDA_CHECK(cudaSetDevice(g.dev));
    std::vector<float> tmp(Wn);
    CUDA_CHECK(cudaMemcpy(tmp.data(), g.dG, Wn*sizeof(float), cudaMemcpyDeviceToHost));
    for(size_t i=0;i<Wn;i++) sum[i]+=tmp[i];
  }
  for(auto& g: gpus){
    CUDA_CHECK(cudaSetDevice(g.dev));
    CUDA_CHECK(cudaMemcpy(g.dG, sum.data(), Wn*sizeof(float), cudaMemcpyHostToDevice));
  }
}
static void broadcast_weights_from0(std::vector<GPU>& gpus){
  int G=(int)gpus.size();
  if(G<=1) return;
  size_t Wn=weights_floats();
  bool p2p=true;
  for(int i=1;i<G;i++){
    int can=0; CUDA_CHECK(cudaDeviceCanAccessPeer(&can,i,0));
    if(!can){ p2p=false; break; }
  }
  if(p2p){
    for(int i=1;i<G;i++) CUDA_CHECK(cudaMemcpyPeer(gpus[i].dW, i, gpus[0].dW, 0, Wn*sizeof(float)));
  }else{
    std::vector<float> hw(Wn);
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMemcpy(hw.data(), gpus[0].dW, Wn*sizeof(float), cudaMemcpyDeviceToHost));
    for(int i=1;i<G;i++){
      CUDA_CHECK(cudaSetDevice(i));
      CUDA_CHECK(cudaMemcpy(gpus[i].dW, hw.data(), Wn*sizeof(float), cudaMemcpyHostToDevice));
    }
  }
  for(int i=0;i<G;i++) refresh_half_weights(&gpus[i]);
}

// ================= train step (device-only) =================
static void train_step_device(GPU* g){
  int B=g->B, T=g->T, N=g->N;
  float invN = 1.0f/(float)N;
  size_t Wn=weights_floats();
  zero_f<<<(int)((Wn+255)/256),256>>>(g->dG,(int)Wn); KERNEL_CHECK();

  dim3 blk2(16,16);
  dim3 grdE((D+15)/16,(N+15)/16);
  embed_split<<<grdE,blk2>>>(g->y1,g->y2,g->W.wte,g->W.wpe,g->tok,N,T); KERNEL_CHECK();

  for(int l=0;l<L;l++){
    constexpr int WPB = 8;
    int blocks = (N + WPB - 1) / WPB;
    fused_rms_qkv_rope_hf<WPB><<<blocks,256>>>(
      g->Q, g->K, g->Vh,
      g->inv,
      (float*)nullptr,
      g->y2,
      g->W.Wq[l], g->W.Wk[l], g->W.Wv[l],
      g->W.gf[l],
      g->sin_tbl, g->cos_tbl,
      N, T
    );
    KERNEL_CHECK();

    dim3 grid_fwd(B,H,(T+127)/128);
    flash_fwd_wmma_hf<<<grid_fwd,256>>>(g->O, g->matt, g->latt, g->Q, g->K, g->Vh, B, T); KERNEL_CHECK();

    int blocks_ffn = (N + WPB - 1) / WPB;
    fused_rms_gelu_ffn_hf<WPB><<<blocks_ffn,256>>>(
      g->y1, g->y2, g->inv, g->n, g->O,
      g->W.Wo[l], g->W.W1[l], g->W.W2[l], g->W.gg[l],
      N, F
    );
    KERNEL_CHECK();
  }

  concat_full<<<dim3((D+15)/16,(N+15)/16),blk2>>>(g->Xfull, g->y1, g->y2, N); KERNEL_CHECK();
  rms_fwd_f2h<D><<<N,256>>>(g->Xnorm, g->Xnorm_h, g->invF, g->Xfull, g->W.gout, N); KERNEL_CHECK();

  zero_f<<<(N*D+255)/256,256>>>(g->dXnorm, N*D); KERNEL_CHECK();
  
  // Single-pass "Beast Mode" Persistent Head mapping:
  persistent_head_beast_hf<<<N, 256>>>(
      g->Loss, g->dXnorm, g->G.Wout, 
      g->Xnorm_h, g->Hw.Wout_tr, g->tgt,
      N, D, V, Vpad
  );
  KERNEL_CHECK();

  zero_f<<<(N*D+255)/256,256>>>(g->dXfull, N*D); KERNEL_CHECK();
  rms_bwd_dX<D><<<N,256>>>(g->dXfull, g->dXnorm, g->Xfull, g->W.gout, g->invF, N); KERNEL_CHECK();
  rms_bwd_dg<D><<<D,256>>>(g->G.gout, g->dXnorm, g->Xfull, g->invF, N); KERNEL_CHECK();
  split_full<<<dim3((D+15)/16,(N+15)/16),blk2>>>(g->dy1, g->dy2, g->dXfull, N); KERNEL_CHECK();

  for(int l=L-1;l>=0;l--){
    rms_fwd_f2h<Dhf><<<N,256>>>(g->n, g->n_h, g->inv, g->y1, g->W.gg[l], N); KERNEL_CHECK();
    wmma_fwd(g->U, g->n_h, g->Hw.W1_tr[l], N, F, Dhf);
    gelu_fwd<<<(N*F+255)/256,256>>>(g->A, g->U, N*F); KERNEL_CHECK();
    f2h<<<(N*F+255)/256,256>>>(g->A_h, g->A, N*F); KERNEL_CHECK();
    wmma_fwd(g->gout, g->A_h, g->Hw.W2_tr[l], N, Dhf, F);

    copy_f<<<(N*Dhf+255)/256,256>>>(g->x2, g->y2, N*Dhf); KERNEL_CHECK();
    sub_inplace<<<(N*Dhf+255)/256,256>>>(g->x2, g->gout, N*Dhf); KERNEL_CHECK();
    copy_f<<<(N*Dhf+255)/256,256>>>(g->dx2, g->dy2, N*Dhf); KERNEL_CHECK();

    copy_f<<<(N*Dhf+255)/256,256>>>(g->dfout, g->dy2, N*Dhf); KERNEL_CHECK();
    wmma_dW(g->G.W2[l], g->Atr, g->dYtr, g->A, g->dfout, N, F, Dhf);
    wmma_dA(g->dA, g->scratchHalf, g->dfout, g->Hw.W2_rm[l], N, Dhf, F);
    gelu_bwd<<<(N*F+255)/256,256>>>(g->dU, g->dA, g->U, N*F); KERNEL_CHECK();
    wmma_dW(g->G.W1[l], g->Atr, g->dYtr, g->n, g->dU, N, Dhf, F);
    wmma_dA(g->dQ, g->scratchHalf, g->dU, g->Hw.W1_rm[l], N, F, Dhf);

    rms_bwd_dX<Dhf><<<N,256>>>(g->dy1, g->dQ, g->y1, g->W.gg[l], g->inv, N); KERNEL_CHECK();
    rms_bwd_dg<Dhf><<<Dhf,256>>>(g->G.gg[l], g->dQ, g->y1, g->inv, N); KERNEL_CHECK();

    constexpr int WPB = 8;
    int blocks = (N + WPB - 1) / WPB;
    fused_rms_qkv_rope_hf<WPB><<<blocks,256>>>(
      g->Q, g->K, g->Vh,
      g->inv,
      g->n,        // write normalized vector for dW(QKV)
      g->x2,
      g->W.Wq[l], g->W.Wk[l], g->W.Wv[l],
      g->W.gf[l],
      g->sin_tbl, g->cos_tbl,
      N, T
    );
    KERNEL_CHECK();

    dim3 grid_fwd(B,H,(T+127)/128);
    flash_fwd_wmma_hf<<<grid_fwd,256>>>(g->O, g->matt, g->latt, g->Q, g->K, g->Vh, B, T); KERNEL_CHECK();

    f2h<<<(N*Dhf+255)/256,256>>>(g->n_h, g->O, N*Dhf); KERNEL_CHECK();
    wmma_fwd(g->fout, g->n_h, g->Hw.Wo_tr[l], N, Dhf, Dhf);

    copy_f<<<(N*Dhf+255)/256,256>>>(g->x1, g->y1, N*Dhf); KERNEL_CHECK();
    sub_inplace<<<(N*Dhf+255)/256,256>>>(g->x1, g->fout, N*Dhf); KERNEL_CHECK();
    copy_f<<<(N*Dhf+255)/256,256>>>(g->dx1, g->dy1, N*Dhf); KERNEL_CHECK();

    copy_f<<<(N*Dhf+255)/256,256>>>(g->dfout, g->dy1, N*Dhf); KERNEL_CHECK();
    wmma_dW(g->G.Wo[l], g->Atr, g->dYtr, g->O, g->dfout, N, Dhf, Dhf);
    wmma_dA(g->dOattn, g->scratchHalf, g->dfout, g->Hw.Wo_rm[l], N, Dhf, Dhf);

    zero_f<<<(N*Dhf+255)/256,256>>>(g->dQ, N*Dhf); KERNEL_CHECK();
    zero_f<<<(N*Dhf+255)/256,256>>>(g->dK, N*Dhf); KERNEL_CHECK();
    zero_f<<<(N*Dhf+255)/256,256>>>(g->dVh,N*Dhf); KERNEL_CHECK();

    flash_bwd_dq_wmma_hf<<<dim3(B,H,(T+FA_QT-1)/FA_QT),256>>>(g->dp, g->dQ, g->dOattn, g->Q, g->K, g->Vh, g->matt, g->latt, B, T); KERNEL_CHECK();
    flash_bwd_dkv_wmma_hf<<<dim3(B,H,(T+FA_KT-1)/FA_KT),256>>>(g->dK, g->dVh, g->dOattn, g->Q, g->K, g->Vh, g->matt, g->latt, g->dp, B, T); KERNEL_CHECK();

    dim3 rblk(16,16);
    dim3 rgrd((Dh/2+15)/16, ((B*T)+15)/16, H);
    rope_apply_grad<<<rgrd,rblk>>>(g->dQ, g->dK, g->sin_tbl, g->cos_tbl, B, T); KERNEL_CHECK();

    wmma_dW(g->G.Wq[l], g->Atr, g->dYtr, g->n, g->dQ, N, Dhf, Dhf);
    wmma_dW(g->G.Wk[l], g->Atr, g->dYtr, g->n, g->dK, N, Dhf, Dhf);
    wmma_dW(g->G.Wv[l], g->Atr, g->dYtr, g->n, g->dVh, N, Dhf, Dhf);

    zero_f<<<(N*Dhf+255)/256,256>>>(g->fout, N*Dhf); KERNEL_CHECK();
    wmma_dA(g->gout, g->scratchHalf, g->dQ, g->Hw.Wq_rm[l], N, Dhf, Dhf); add_inplace<<<(N*Dhf+255)/256,256>>>(g->fout, g->gout, N*Dhf); KERNEL_CHECK();
    wmma_dA(g->gout, g->scratchHalf, g->dK, g->Hw.Wk_rm[l], N, Dhf, Dhf); add_inplace<<<(N*Dhf+255)/256,256>>>(g->fout, g->gout, N*Dhf); KERNEL_CHECK();
    wmma_dA(g->gout, g->scratchHalf, g->dVh,g->Hw.Wv_rm[l], N, Dhf, Dhf); add_inplace<<<(N*Dhf+255)/256,256>>>(g->fout, g->gout, N*Dhf); KERNEL_CHECK();

    rms_bwd_dX<Dhf><<<N,256>>>(g->dx2, g->fout, g->x2, g->W.gf[l], g->inv, N); KERNEL_CHECK();
    rms_bwd_dg<Dhf><<<Dhf,256>>>(g->G.gf[l], g->fout, g->x2, g->inv, N); KERNEL_CHECK();

    std::swap(g->y1, g->x1);
    std::swap(g->y2, g->x2);
    std::swap(g->dy1, g->dx1);
    std::swap(g->dy2, g->dx2);
  }

  embed_bwd<<<dim3((D+15)/16,(N+15)/16),dim3(16,16)>>>(g->G.wte, g->G.wpe, g->tok, g->dy1, g->dy2, N, T); KERNEL_CHECK();

  loss_reduce_1<<<256,256>>>(g->Loss, g->partial, N); KERNEL_CHECK();
  loss_reduce_2<<<1,256>>>(g->partial, g->loss_mean, 256, invN); KERNEL_CHECK();
}

// ===== CUDA Graph capture (capture-safe + restore swapped pointer state) =====
static void ensure_train_graph(GPU* g){
  if(g->graph_built) return;

  // Save pointer state (train_step_device does host swaps)
  auto y1=g->y1; auto y2=g->y2; auto x1=g->x1; auto x2=g->x2;
  auto dy1=g->dy1; auto dy2=g->dy2; auto dx1=g->dx1; auto dx2=g->dx2;

  g_is_capturing = 1;
  CUDA_CHECK(cudaStreamBeginCapture(0, cudaStreamCaptureModeGlobal));

  CUDA_CHECK(cudaMemcpyAsync(g->tok, g->htok_h, (size_t)g->N*sizeof(uint16_t), cudaMemcpyHostToDevice, 0));
  CUDA_CHECK(cudaMemcpyAsync(g->tgt, g->htgt_h, (size_t)g->N*sizeof(uint16_t), cudaMemcpyHostToDevice, 0));

  train_step_device(g);

  CUDA_CHECK(cudaStreamEndCapture(0, &g->graph));
  g_is_capturing = 0;

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaGraphInstantiate(&g->graphExec, g->graph, nullptr, nullptr, 0));
  g->graph_built = 1;

  // Restore pointers for non-graph path correctness
  g->y1=y1; g->y2=y2; g->x1=x1; g->x2=x2;
  g->dy1=dy1; g->dy2=dy2; g->dx1=dx1; g->dx2=dx2;
}

static float train_step(GPU* g, const std::vector<uint16_t>& ids, int step, int64_t start_bias, uint64_t seed, int use_graph){
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

  for(int i=0;i<N;i++){
    if((int)htok[i] >= V || (int)htgt[i] >= V){
      std::fprintf(stderr,"BAD TOK: step=%d dev=%d i=%d tok=%u tgt=%u V=%d\n",
                   step, g->dev, i, (unsigned)htok[i], (unsigned)htgt[i], V);
      std::exit(1);
    }
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
    CUDA_CHECK(cudaMemcpyAsync(g->tok, g->htok_h, (size_t)N*sizeof(uint16_t), cudaMemcpyHostToDevice, 0));
    CUDA_CHECK(cudaMemcpyAsync(g->tgt, g->htgt_h, (size_t)N*sizeof(uint16_t), cudaMemcpyHostToDevice, 0));
    train_step_device(g);
    CUDA_CHECK(cudaStreamSynchronize(0));
  }

  float loss=0.f;
  CUDA_CHECK(cudaMemcpy(&loss, g->loss_mean, sizeof(float), cudaMemcpyDeviceToHost));
  return loss;
}









// ============================================================================
// CHAT (FULL GPU incremental greedy decode)
// ============================================================================

// half2 GEMV: y[M] = sum_k x[k] * Wtr[M,K]  (Wtr half, row-major MxK)
__global__ void gemv_tr_half2(float* y, const float* x, const half* Wtr, int M, int K){
  int m = blockIdx.x * blockDim.x + threadIdx.x;
  if(m>=M) return;
  float acc=0.f;
  const half* wrow = Wtr + (size_t)m*(size_t)K;
  for(int k=0;k<K;k+=2){
    half2 wh = *(const half2*)(wrow + k);
    float2 wf = __half22float2(wh);
    float x0 = x[k+0];
    float x1 = x[k+1];
    acc += wf.x*x0 + wf.y*x1;
  }
  y[m]=acc;
}

// rmsnorm for single vector
template<int DIM>
__global__ void rms1(float* y, float* inv_out, const float* x, const float* g){
  __shared__ float buf[256];
  float s=0.f;
  for(int i=threadIdx.x;i<DIM;i+=blockDim.x){ float v=x[i]; s+=v*v; }
  buf[threadIdx.x]=s; __syncthreads();
  for(int k=blockDim.x/2;k>0;k>>=1){ if(threadIdx.x<k) buf[threadIdx.x]+=buf[threadIdx.x+k]; __syncthreads(); }
  float inv = rsqrtf(buf[0]/(float)DIM + EPS);
  if(threadIdx.x==0) inv_out[0]=inv;
  __syncthreads();
  for(int i=threadIdx.x;i<DIM;i+=blockDim.x) y[i]=x[i]*inv*g[i];
}

// RoPE for one token position on Dhf vector (per head, Dh=16)
__global__ void rope_apply_one(float* X, const float* sin_tbl, const float* cos_tbl, int t){
  int h = blockIdx.y;
  int i2 = blockIdx.x * blockDim.x + threadIdx.x;
  if(h>=H || i2>=Dh/2) return;
  float s = sin_tbl[(size_t)t*(size_t)(Dh/2) + (size_t)i2];
  float c = cos_tbl[(size_t)t*(size_t)(Dh/2) + (size_t)i2];
  int base = h*Dh;
  int i0=2*i2, i1=i0+1;
  float x0=X[base+i0], x1=X[base+i1];
  X[base+i0]=x0*c - x1*s;
  X[base+i1]=x0*s + x1*c;
}

// attention decode for one token, one block per head, 32 threads (lanes 0..15 used)
__global__ void attn_decode_one(float* outDhf, const float* qDhf,
                               const float* Kcache, const float* Vcache,
                               int t){
  int h = (int)blockIdx.x;
  int lane = (int)threadIdx.x;
  if(h>=H || lane>=Dh) return;

  float scale = 1.0f/sqrtf((float)Dh);
  float mi=-1e30f;
  float li=0.f;
  float oi=0.f;

  const float* q = qDhf + (size_t)h*(size_t)Dh;

  for(int kpos=0;kpos<=t;kpos++){
    const float* kptr = Kcache + (size_t)kpos*(size_t)Dhf + (size_t)h*(size_t)Dh;
    const float* vptr = Vcache + (size_t)kpos*(size_t)Dhf + (size_t)h*(size_t)Dh;

    float dot = q[lane]*kptr[lane];
    dot = reduce_sum16(dot, 0x0000FFFFu);
    float s = dot*scale;

    float m_new = fmaxf(mi, s);
    float alpha = expf(mi - m_new);
    li *= alpha;
    oi *= alpha;

    float p = expf(s - m_new);
    li += p;
    oi += p*vptr[lane];
    mi = m_new;
  }
  outDhf[(size_t)h*(size_t)Dh + (size_t)lane] = oi / li;
}

__global__ void gelu_vec(float* y, const float* x, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) y[i]=gelu(x[i]);
}

// argmax helper
__global__ void argmax_stage1(const float* x, float* maxv, int* maxi, int n){
  __shared__ float sv[256];
  __shared__ int si[256];
  int tid=threadIdx.x;
  int idx=blockIdx.x*blockDim.x+tid;
  float v=-1e30f; int i=-1;
  if(idx<n){ v=x[idx]; i=idx; }
  sv[tid]=v; si[tid]=i;
  __syncthreads();
  for(int k=128;k>0;k>>=1){
    if(tid<k){
      float v2=sv[tid+k];
      if(v2>sv[tid]){ sv[tid]=v2; si[tid]=si[tid+k]; }
    }
    __syncthreads();
  }
  if(tid==0){ maxv[blockIdx.x]=sv[0]; maxi[blockIdx.x]=si[0]; }
}
__global__ void argmax_stage2(const float* maxv1, const int* maxi1, int* outi, int n){
  __shared__ float sv[256];
  __shared__ int si[256];
  int tid=threadIdx.x;
  float v=-1e30f; int i=-1;
  for(int idx=tid; idx<n; idx+=blockDim.x){
    float vv=maxv1[idx];
    if(vv>v){ v=vv; i=maxi1[idx]; }
  }
  sv[tid]=v; si[tid]=i; __syncthreads();
  for(int k=128;k>0;k>>=1){
    if(tid<k){
      float v2=sv[tid+k];
      if(v2>sv[tid]){ sv[tid]=v2; si[tid]=si[tid+k]; }
    }
    __syncthreads();
  }
  if(tid==0) outi[0]=si[0];
}

// embed one token id at position t into y1/y2
__global__ void embed_one(float* y1, float* y2, const float* wte, const float* wpe, int id, int t){
  int d=blockIdx.x*blockDim.x+threadIdx.x;
  if(d>=D) return;
  float v = wte[(size_t)id*(size_t)D + (size_t)d] + wpe[(size_t)t*(size_t)D + (size_t)d];
  if(d<Dhf) y1[d]=v; else y2[d-Dhf]=v;
}

__global__ void add2(float* a, const float* b, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) a[i]+=b[i]; }
__global__ void concat1(float* xfull, const float* y1, const float* y2){
  int d=blockIdx.x*blockDim.x+threadIdx.x;
  if(d>=D) return;
  xfull[d] = (d<Dhf) ? y1[d] : y2[d-Dhf];
}
__global__ void kv_store(float* Kc, float* Vc, const float* k, const float* v, int t){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<Dhf){
    Kc[(size_t)t*(size_t)Dhf + (size_t)i]=k[i];
    Vc[(size_t)t*(size_t)Dhf + (size_t)i]=v[i];
  }
}

// logits = xnorm * Wout_tr (V x D)
__global__ void gemv_vocab(float* logits, const float* x, const half* Wtr){
  int v = blockIdx.x*blockDim.x + threadIdx.x;
  if(v>=V) return;
  const half* wrow = Wtr + (size_t)v*(size_t)D;
  float acc=0.f;
  for(int k=0;k<D;k+=2){
    half2 wh = *(const half2*)(wrow + k);
    float2 wf = __half22float2(wh);
    float x0=x[k+0], x1=x[k+1];
    acc += wf.x*x0 + wf.y*x1;
  }
  logits[v]=acc;
}

struct ChatCtx {
  WView W;
  HW Hw;

  float *y1,*y2,*n1,*inv1;
  float *q,*k,*v,*o,*fout,*gout;
  float *u,*a;
  float *xfull,*xnorm,*invF;
  float *logits;

  float *amaxv1;
  int *amaxi1;
  int *amaxi;

  float *sin_tbl,*cos_tbl;

  float *Kc[L];
  float *Vc[L];
};

static void chat_alloc(ChatCtx* c, const GPU& g0){
  CUDA_CHECK(cudaSetDevice(0));
  c->W = g0.W;
  c->Hw = g0.Hw;
  c->sin_tbl=g0.sin_tbl;
  c->cos_tbl=g0.cos_tbl;

  CUDA_CHECK(cudaMalloc(&c->y1, Dhf*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&c->y2, Dhf*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&c->n1, Dhf*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&c->inv1, sizeof(float)));

  CUDA_CHECK(cudaMalloc(&c->q, Dhf*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&c->k, Dhf*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&c->v, Dhf*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&c->o, Dhf*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&c->fout, Dhf*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&c->gout, Dhf*sizeof(float)));

  CUDA_CHECK(cudaMalloc(&c->u, F*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&c->a, F*sizeof(float)));

  CUDA_CHECK(cudaMalloc(&c->xfull, D*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&c->xnorm, D*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&c->invF, sizeof(float)));

  CUDA_CHECK(cudaMalloc(&c->logits, V*sizeof(float)));

  int blocks=(V+255)/256;
  CUDA_CHECK(cudaMalloc(&c->amaxv1, blocks*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&c->amaxi1, blocks*sizeof(int)));
  CUDA_CHECK(cudaMalloc(&c->amaxi, sizeof(int)));

  for(int l=0;l<L;l++){
    CUDA_CHECK(cudaMalloc(&c->Kc[l], (size_t)Tmax*(size_t)Dhf*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c->Vc[l], (size_t)Tmax*(size_t)Dhf*sizeof(float)));
    CUDA_CHECK(cudaMemset(c->Kc[l],0,(size_t)Tmax*(size_t)Dhf*sizeof(float)));
    CUDA_CHECK(cudaMemset(c->Vc[l],0,(size_t)Tmax*(size_t)Dhf*sizeof(float)));
  }
}
static void chat_free(ChatCtx* c){
  CUDA_CHECK(cudaSetDevice(0));
  auto cf=[&](void* p){ if(p) cudaFree(p); };
  cf(c->y1); cf(c->y2); cf(c->n1); cf(c->inv1);
  cf(c->q); cf(c->k); cf(c->v); cf(c->o); cf(c->fout); cf(c->gout);
  cf(c->u); cf(c->a);
  cf(c->xfull); cf(c->xnorm); cf(c->invF);
  cf(c->logits);
  cf(c->amaxv1); cf(c->amaxi1); cf(c->amaxi);
  for(int l=0;l<L;l++){ cf(c->Kc[l]); cf(c->Vc[l]); }
  std::memset(c,0,sizeof(*c));
}

static int chat_step(ChatCtx* c, int t, int tok_id){
  embed_one<<<(D+255)/256,256>>>(c->y1, c->y2, c->W.wte, c->W.wpe, tok_id, t); KERNEL_CHECK();

  for(int l=0;l<L;l++){
    rms1<Dhf><<<1,256>>>(c->n1, c->inv1, c->y2, c->W.gf[l]); KERNEL_CHECK();

    gemv_tr_half2<<<(Dhf+127)/128,128>>>(c->q, c->n1, c->Hw.Wq_tr[l], Dhf, Dhf); KERNEL_CHECK();
    gemv_tr_half2<<<(Dhf+127)/128,128>>>(c->k, c->n1, c->Hw.Wk_tr[l], Dhf, Dhf); KERNEL_CHECK();
    gemv_tr_half2<<<(Dhf+127)/128,128>>>(c->v, c->n1, c->Hw.Wv_tr[l], Dhf, Dhf); KERNEL_CHECK();

    dim3 rgrd((Dh/2+15)/16, H);
    rope_apply_one<<<rgrd,16>>>(c->q, c->sin_tbl, c->cos_tbl, t); KERNEL_CHECK();
    rope_apply_one<<<rgrd,16>>>(c->k, c->sin_tbl, c->cos_tbl, t); KERNEL_CHECK();

    kv_store<<<(Dhf+255)/256,256>>>(c->Kc[l], c->Vc[l], c->k, c->v, t); KERNEL_CHECK();

    attn_decode_one<<<H,32>>>(c->o, c->q, c->Kc[l], c->Vc[l], t); KERNEL_CHECK();

    gemv_tr_half2<<<(Dhf+127)/128,128>>>(c->fout, c->o, c->Hw.Wo_tr[l], Dhf, Dhf); KERNEL_CHECK();
    add2<<<(Dhf+255)/256,256>>>(c->y1, c->fout, Dhf); KERNEL_CHECK();

    rms1<Dhf><<<1,256>>>(c->n1, c->inv1, c->y1, c->W.gg[l]); KERNEL_CHECK();
    gemv_tr_half2<<<(F+255)/256,256>>>(c->u, c->n1, c->Hw.W1_tr[l], F, Dhf); KERNEL_CHECK();
    gelu_vec<<<(F+255)/256,256>>>(c->a, c->u, F); KERNEL_CHECK();
    gemv_tr_half2<<<(Dhf+127)/128,128>>>(c->gout, c->a, c->Hw.W2_tr[l], Dhf, F); KERNEL_CHECK();
    add2<<<(Dhf+255)/256,256>>>(c->y2, c->gout, Dhf); KERNEL_CHECK();
  }

  concat1<<<(D+255)/256,256>>>(c->xfull, c->y1, c->y2); KERNEL_CHECK();
  rms1<D><<<1,256>>>(c->xnorm, c->invF, c->xfull, c->W.gout); KERNEL_CHECK();

  gemv_vocab<<<(V+255)/256,256>>>(c->logits, c->xnorm, c->Hw.Wout_tr); KERNEL_CHECK();

  int blocks=(V+255)/256;
  argmax_stage1<<<blocks,256>>>(c->logits, c->amaxv1, c->amaxi1, V); KERNEL_CHECK();
  argmax_stage2<<<1,256>>>(c->amaxv1, c->amaxi1, c->amaxi, blocks); KERNEL_CHECK();

  int next=0;
  CUDA_CHECK(cudaMemcpy(&next, c->amaxi, sizeof(int), cudaMemcpyDeviceToHost));
  if(next<0) next=0;
  if(next>=V) next=V-1;
  return next;
}

static void chat_repl(const PairIndex& pi, const std::vector<float>& hostW, const char* ckpt_path, const char* prompt, bool do_measure){
  (void)ckpt_path;
  CUDA_CHECK(cudaSetDevice(0));

  GPU g0{};
  gpu_alloc(&g0, 0, 1, 16);
  CUDA_CHECK(cudaMemcpy(g0.dW, hostW.data(), hostW.size()*sizeof(float), cudaMemcpyHostToDevice));
  refresh_half_weights(&g0);

  ChatCtx ctx{};
  chat_alloc(&ctx, g0);

  std::vector<uint16_t> pre;
  if(prompt && prompt[0]) pre=encode_prompt(pi, std::string(prompt));

  int t=0;
  int next=-1;
  for(uint16_t id: pre){
    next = chat_step(&ctx, t, (int)id);
    t++;
    if(t>=Tmax) break;
  }

  std::fprintf(stderr,"[chat] commands: /reset /quit  (greedy decode, full GPU KV)\n");
  std::string line;
  std::vector<uint8_t> output_buffer;

  while(true){
    std::fprintf(stderr,"> ");
    std::fflush(stderr);
    if(!std::getline(std::cin,line)) break;
    if(line=="/quit") break;
    if(line=="/reset"){
      t=0;
      for(int l=0;l<L;l++){
        CUDA_CHECK(cudaMemset(ctx.Kc[l],0,(size_t)Tmax*(size_t)Dhf*sizeof(float)));
        CUDA_CHECK(cudaMemset(ctx.Vc[l],0,(size_t)Tmax*(size_t)Dhf*sizeof(float)));
      }
      next=-1;
      output_buffer.clear();
      continue;
    }

    auto ids=encode_prompt(pi, line+"\n");
    for(size_t i=0;i<ids.size();i++){
      next = chat_step(&ctx, t, (int)ids[i]);
      t++;
      if(t>=Tmax) break;
    }
    if(t>=Tmax){ std::cout << "\n[ctx full]\n"; continue; }

    int gen_max=200;

    std::chrono::time_point<std::chrono::high_resolution_clock> tstart_chat;
    if(do_measure) tstart_chat = std::chrono::high_resolution_clock::now();

    for(int i=0;i<gen_max && t<Tmax;i++){
      uint16_t out_id=(uint16_t)next;
      std::vector<uint8_t> out;
      decode_id(pi, out_id, out);

      bool stop=false;
      for(uint8_t b: out){
        if(b=='\n') stop=true;
        output_buffer.push_back(b);
      }

      // flush if UTF-8 complete (best-effort)
      bool complete=true;
      if(!output_buffer.empty()){
        int trailing=0;
        for(int j=(int)output_buffer.size()-1;j>=0;j--){
          uint8_t b=output_buffer[(size_t)j];
          if((b & 0xC0)==0x80) trailing++;
          else{
            if((b & 0xE0)==0xC0) complete=(trailing==1);
            else if((b & 0xF0)==0xE0) complete=(trailing==2);
            else if((b & 0xF8)==0xF0) complete=(trailing==3);
            else complete=true;
            break;
          }
          if(trailing>3){ complete=true; break; }
        }
      }
      if(complete && !output_buffer.empty()){
        std::cout.write((const char*)output_buffer.data(), (std::streamsize)output_buffer.size());
        std::cout.flush();
        output_buffer.clear();
      }

      next = chat_step(&ctx, t, (int)out_id);
      t++;

      if(stop){
        if(do_measure){
          auto tend = std::chrono::high_resolution_clock::now();
          double elapsed = std::chrono::duration<double>(tend - tstart_chat).count();
          std::fprintf(stderr, "\n[tok/s: %.1f]\n", (double)i / elapsed);
        }
        break;
      }
    }

    if(t>=Tmax){
      std::cout << "\n[ctx full]\n";
    }
  }

  chat_free(&ctx);
  gpu_free(&g0);
}

// ================= main =================
int main(int argc,char** argv){
  const char* data_path="tinyshakespeare.txt";
  const char* ckpt_path="ckpt.bin";
  const char* index_path="index_v7.bin";
  int steps=2000,batch=64,seq=128,gpus_req=2;
  float lr=3e-4f,wd=0.01f,clip=1.0f;
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
    else if(!std::strcmp(argv[i],"--lr") && i+1<argc) lr=(float)std::atof(argv[++i]);
    else if(!std::strcmp(argv[i],"--wd") && i+1<argc) wd=(float)std::atof(argv[++i]);
    else if(!std::strcmp(argv[i],"--clip") && i+1<argc) clip=(float)std::atof(argv[++i]);
    else if(!std::strcmp(argv[i],"--log_every") && i+1<argc) log_every=std::atoi(argv[++i]);
    else if(!std::strcmp(argv[i],"--save_every") && i+1<argc) save_every=std::atoi(argv[++i]);
    else if(!std::strcmp(argv[i],"--seed") && i+1<argc) seed=(uint64_t)std::strtoull(argv[++i],nullptr,10);
    else { std::fprintf(stderr,"Unknown arg: %s\n", argv[i]); return 2; }
  }

  if((seq%16)!=0) die("--seq must be multiple of 16");
  if(seq<16 || seq>Tmax) die("--seq out of range");

  auto bytes=read_file_bytes(data_path);



  PairIndex pi;
  std::vector<float> winit;

  bool has_ckpt = load_ckpt(ckpt_path, &pi, &winit);

  if(has_ckpt && do_train && !do_continue){
    die("FATAL: checkpoint exists. You must either pass --continue to resume it, or --force-new to delete it.");
  }
  if(do_continue && !has_ckpt){
    std::fprintf(stderr, "WARNING: --continue requested but no ckpt found. Starting fresh.\n");
  }

  if(!has_ckpt){
    if(!load_index_v7(index_path, &pi)) die("no ckpt and could not load index (build index first)");
  }

  std::vector<float> hostW(weights_floats());
  if(has_ckpt) hostW=winit;
  else init_weights_cpu(hostW, seed);

  // auto behavior: if no explicit --train and ckpt exists -> chat
  if(!do_train && !do_chat){
    if(has_ckpt) do_chat=true;
    else do_train=true;
  }

  if(do_chat){
    chat_repl(pi, hostW, ckpt_path, chat_prompt, do_measure);
    return 0;
  }

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
  for(int i=0;i<G;i++) gpu_alloc(&gpus[(size_t)i], i, Bi[(size_t)i], seq);

  for(int i=0;i<G;i++){
    CUDA_CHECK(cudaSetDevice(i));
    CUDA_CHECK(cudaMemcpy(gpus[(size_t)i].dW, hostW.data(), hostW.size()*sizeof(float), cudaMemcpyHostToDevice));
    refresh_half_weights(&gpus[(size_t)i]);
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

  std::printf("RUNBEAST ENGINE ACTIVATED: V=%d(Vpad=%d) K=%d K1=%d K2=%d D=%d Dh=%d L=%d H=%d F=%d T=%d gpus=%d batch=%d [",
              V,Vpad,PAIR_K,PAIR_K1,PAIR_K-PAIR_K1,D,Dh,L,H,F,seq,G,batch);
  for(int i=0;i<G;i++) std::printf("%d%s", Bi[(size_t)i], (i+1<G)?",":"");
  std::printf("]\n");

  std::chrono::time_point<std::chrono::high_resolution_clock> t0;
  if(do_measure) t0 = std::chrono::high_resolution_clock::now();

  for(int step=1; step<=steps; step++){
    std::vector<float> losses((size_t)G,0.f);
    std::vector<std::thread> th; th.reserve((size_t)G);
    int64_t base = (int64_t)step * 1315423911LL;
    for(int i=0;i<G;i++){
      th.emplace_back([&,i](){ losses[(size_t)i]=train_step(&gpus[(size_t)i], ids, step, base + (int64_t)i*9973LL, seed, use_graph); });
    }
    for(auto& t: th) t.join();

    if(G>1){
      if(!p2p_allreduce(gpus)) host_allreduce(gpus);
      CUDA_CHECK(cudaSetDevice(0));
      int nW=(int)weights_floats();
      scale_f<<<(nW+255)/256,256>>>(gpus[0].dG, 1.0f/(float)G, nW);
      KERNEL_CHECK();
    }

    adam_step(&gpus[0], step, lr, wd, clip);
    broadcast_weights_from0(gpus);

    if (step == 1 && do_measure) t0 = std::chrono::high_resolution_clock::now();

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
      if(!std::isfinite((float)Lm)) die("loss NaN/Inf");
    }

    if(save_every>0 && (step%save_every)==0){
      CUDA_CHECK(cudaSetDevice(0));
      CUDA_CHECK(cudaMemcpy(hostW.data(), gpus[0].dW, hostW.size()*sizeof(float), cudaMemcpyDeviceToHost));
      save_ckpt(ckpt_path, pi, hostW.data());
      std::printf("[*] saved: %s\n", ckpt_path);
      std::fflush(stdout);
    }
  }

  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaMemcpy(hostW.data(), gpus[0].dW, hostW.size()*sizeof(float), cudaMemcpyDeviceToHost));
  save_ckpt(ckpt_path, pi, hostW.data());
  std::printf("[*] saved final: %s\n", ckpt_path);

  for(int i=0;i<G;i++) gpu_free(&gpus[(size_t)i]);
  return 0;
}
CU

echo "[*] Building: $BIN (sm_75)"
nvcc -O3 -std=c++17 -arch=sm_75 --default-stream per-thread --use_fast_math -lineinfo --expt-relaxed-constexpr \
  -DPAIR_K="$PAIR_K" -DPAIR_K1="$PAIR_K1" -DVCHUNK="$VCHUNK" -DDMODEL="$DMODEL" -DNHEAD="$NHEAD" -DNLAY="$NLAY" -DFFN="$FFN" -DTMAX="$TMAX" \
  "$tmpcu/$CU" -o "$BIN"

rm -rf "$tmpcu"

# =============================================================================
# Deterministic Hash-based Checkpointing
# =============================================================================
# Create a deterministic MD5 hash based on the structural hyperparameters.
# This ensures checkpoints are strictly tied to the exact architecture they
# were trained on, preventing shape-mismatch crashes.
HASH_STR="${BIN}_K${PAIR_K}_D${DMODEL}_H${NHEAD}_L${NLAY}_F${FFN}_T${TMAX}"
CKPT_HASH=$(echo -n "$HASH_STR" | md5sum | head -c 8)
DEFAULT_CKPT_FILE="${WORKDIR}/ckpt_${BIN}_${CKPT_HASH}.bin"

# Parse bash-level arguments
FORCE_NEW=0
USER_PASSED_CKPT=0
declare -a PASSED_ARGS
for arg in "$@"; do
  if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
    echo "Usage: $0 [options]"
    echo ""
    echo "Wrapper Options:"
    echo "  --force-new       Delete the existing checkpoint for this configuration and start fresh."
    echo ""
    echo "Engine Options:"
    echo "  --train           Start training mode."
    echo "  --chat            Start chat mode."
    echo "  --chat_prompt P   Provides initial chat prompt."
    echo "  --continue        Continue from an existing checkpoint."
    echo "  --steps N         Number of training steps."
    echo "  --batch N         Batch size across all GPUs."
    echo "  --seq N           Sequence length."
    echo "  --gpus N          Number of GPUs to use."
    echo "  --measure         Print token/sec performance."
    echo "  --no_graph        Disable CUDA Graphs for execution (Graph is on by default)."
    echo "  --lr F            Learning rate."
    echo "  --ckpt PATH       Specify a custom checkpoint path to save/load."
    echo ""
    echo "By default, the script generates a deterministic checkpoint based on model dimensions."
    echo "If a checkpoint exists, it will auto-continue (or auto-chat if neither --train nor --chat is given)."
    exit 0
  elif [[ "$arg" == "--force-new" ]]; then
    FORCE_NEW=1
  elif [[ "$arg" == "--ckpt" ]]; then
    USER_PASSED_CKPT=1
    PASSED_ARGS+=("$arg")
  else
    PASSED_ARGS+=("$arg")
  fi
done

# If the user didn't explicitly provide a --ckpt, we inject our deterministic one
if [[ $USER_PASSED_CKPT -eq 0 ]]; then
  CKPT_FILE="$DEFAULT_CKPT_FILE"
  # Inject it so the C++ engine knows where to look/save
  PASSED_ARGS+=("--ckpt" "$CKPT_FILE")
else
  # The user passed a custom ckpt path, we'll respect it but extract it to check existence
  # Actually, parsing it out of bash arrays robustly is slightly tricky here, so we'll
  # assume if they pass a custom path they know what they're doing with --continue
  CKPT_FILE="" # Unused in explicit check below
fi

if [[ $FORCE_NEW -eq 1 ]]; then
  echo "[*] --force-new passed. Ensuring a fresh start."
  if [[ $USER_PASSED_CKPT -eq 0 && -f "$CKPT_FILE" ]]; then
    rm -f "$CKPT_FILE"
    echo "[*] Deleted existing checkpoint: $CKPT_FILE"
  fi
  # If they passed custom ckpt, we assume they wiped it themselves or we just pass through
fi

# Auto-continue logic
# If we have a known CKPT_FILE and it exists, default to --continue
has_train_flag=0
has_chat_flag=0
has_continue_flag=0
for arg in "${PASSED_ARGS[@]}"; do
  if [[ "$arg" == "--train" ]]; then has_train_flag=1; fi
  if [[ "$arg" == "--chat" ]]; then has_chat_flag=1; fi
  if [[ "$arg" == "--continue" ]]; then has_continue_flag=1; fi
done

if [[ $USER_PASSED_CKPT -eq 0 && -f "$CKPT_FILE" ]]; then
  echo "[*] Found deterministic checkpoint: $CKPT_FILE"
  if [[ $has_chat_flag -eq 0 && $has_train_flag -eq 0 ]]; then
     echo "[*] Auto-entering Chat mode."
     PASSED_ARGS+=("--chat" "--chat_prompt" "Hello")
  elif [[ $has_train_flag -eq 1 && $has_continue_flag -eq 0 && $FORCE_NEW -eq 0 ]]; then
     echo "[*] Auto-appending --continue to training."
     PASSED_ARGS+=("--continue")
  fi
elif [[ $USER_PASSED_CKPT -eq 0 && ! -f "$CKPT_FILE" ]]; then
  echo "[*] No checkpoint found for hash [${CKPT_HASH}]. Starting fresh."
  if [[ $has_chat_flag -eq 0 && $has_train_flag -eq 0 ]]; then
     echo "[*] Auto-entering Train mode."
     PASSED_ARGS+=("--train")
  fi
fi

echo
echo "========================================================================="
echo "  ██████╗ ██╗   ██╗███╗   ██╗██████╗ ███████╗ █████╗ ███████╗████████╗"
echo "  ██╔══██╗██║   ██║████╗  ██║██╔══██╗██╔════╝██╔══██╗██╔════╝╚══██╔══╝"
echo "  ██████╔╝██║   ██║██╔██╗ ██║██████╔╝█████╗  ███████║███████╗   ██║   "
echo "  ██╔══██╗██║   ██║██║╚██╗██║██╔══██╗██╔══╝  ██╔══██║╚════██║   ██║   "
echo "  ██║  ██║╚██████╔╝██║ ╚████║██████╔╝███████╗██║  ██║███████║   ██║   "
echo "  ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═════╝ ╚══════╝╚═╝  ╚═╝╚══════╝   ╚═╝   "
echo "========================================================================="
echo "  [RAW POWER UNLOCKED] - Bypassing all limits. Squeezing galaxies."
echo "  Hash Configuration: $CKPT_HASH"
echo "  cmd: ./$BIN --data \"$DATA_FILE\" --index \"$INDEX_BIN\" ${PASSED_ARGS[@]}"
echo "========================================================================="
echo

# Set default parameters if not provided in PASSED_ARGS.
# The user's array over-rides anything added later inside C++, but we must provide --data & --index.
./"$BIN" --data "$DATA_FILE" --index "$INDEX_BIN" "${PASSED_ARGS[@]}"