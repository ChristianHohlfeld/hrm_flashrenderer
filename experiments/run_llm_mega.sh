#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# © 2026 Christian Heinrich Hohlfeld (Konstanz, Deutschland) — Alle Rechte vorbehalten.
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
need curl || { echo "FATAL: curl not found"; exit 1; }

# Helper: expand colon-separated paths relative to WORKDIR if not absolute
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

# -------- corpus ----------
DATA_FILE="${DATA_FILE:-tinyshakespeare.txt}"
URL="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
if [[ ! -s "$DATA_FILE" ]]; then
  echo "[*] Downloading Tiny Shakespeare..."
  curl -L --fail "$URL" -o "$DATA_FILE"
fi
[[ -s "$DATA_FILE" ]] || { echo "FATAL: corpus empty: $DATA_FILE" >&2; exit 1; }

# -------- knobs ----------
: "${PAIR_K:=4096}"
: "${PAIR_K1:=2048}"
: "${VCHUNK:=1024}"
: "${DMODEL:=256}"
: "${NHEAD:=8}"
: "${NLAY:=6}"
: "${FFN:=1024}"
: "${TMAX:=512}"

K2=$((PAIR_K-PAIR_K1))
[[ "$K2" -ge 0 ]] || { echo "FATAL: PAIR_K must be >= PAIR_K1"; exit 1; }

INDEX_INPUTS="${INDEX_INPUTS:-$DATA_FILE}"   # colon-separated corpus files
ABS_INPUT="$(absify_inputs "$INDEX_INPUTS")"
INDEX_BIN="${INDEX_BIN:-index_v7_k1${PAIR_K1}_k2${K2}.bin}"
FORCE_INDEX="${FORCE_INDEX:-0}"

# -------- build deterministic index builder (padded, NO scaling down) ----------
TMP_BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_BUILD_DIR"' EXIT

cat > "$TMP_BUILD_DIR/index_build_v7.cpp" <<'CPP'
/*
=============================================================================
© 2026 Christian Heinrich Hohlfeld (Konstanz, Deutschland) — Alle Rechte vorbehalten.
Website: christianhohlfeld.com
ORCID: 0009-0003-6634-9045

Attribution / Ownership Notice:
This file contains an implementation that includes "ID-based tokenization / index training"
and related infrastructure ideas asserted by Christian Heinrich Hohlfeld as his intellectual
property. Keep this header intact in any copies.
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

  // IMPORTANT: deterministic padding to exactly K1 (NO scaling down)
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
  int K1=2048, K2=2048;
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
  std::fprintf(stderr,"[index] wrote %s table=%zu\n", out.c_str(), hkeys.size());
  return 0;
}
CPP

g++ -O3 -std=c++17 "$TMP_BUILD_DIR/index_build_v7.cpp" -o "$TMP_BUILD_DIR/index_build_v7"

# -------- build index (always consistent with PAIR_K/PARI_K1) ----------
if [[ "$FORCE_INDEX" == "1" || ! -s "$INDEX_BIN" ]]; then
  echo "[*] Building deterministic index: $INDEX_BIN (K1=$PAIR_K1 K2=$K2)"
  "$TMP_BUILD_DIR/index_build_v7" --k1 "$PAIR_K1" --k2 "$K2" --out "$INDEX_BIN" --inputs "$ABS_INPUT"
fi

# -------- build engine from existing llm_engine_mega.cu ----------
CU_SRC="${CU_SRC:-llm_engine_mega.cu}"
BIN="${BIN:-llm_engine_mega}"

[[ -s "$CU_SRC" ]] || {
  echo "FATAL: missing CUDA source: $CU_SRC (place it next to this script)" >&2
  exit 1
}

echo "[*] Building: $BIN (sm_75)  PAIR_K=$PAIR_K  PAIR_K1=$PAIR_K1"
nvcc -O3 -std=c++17 -arch=sm_75 --default-stream per-thread --use_fast_math -lineinfo --expt-relaxed-constexpr \
  -DPAIR_K="$PAIR_K" -DPAIR_K1="$PAIR_K1" -DVCHUNK="$VCHUNK" -DDMODEL="$DMODEL" -DNHEAD="$NHEAD" -DNLAY="$NLAY" -DFFN="$FFN" -DTMAX="$TMAX" \
  "$CU_SRC" -o "$BIN"

echo
echo "[*] Run example:"
echo "  ./$BIN --train --data \"$DATA_FILE\" --index \"$INDEX_BIN\" --ckpt ckpt.bin --steps 2000 --batch 64 --seq 128 --gpus 2 --measure"
echo

if [[ "$#" -gt 0 ]]; then
  ./"$BIN" --data "$DATA_FILE" --index "$INDEX_BIN" "$@"
else
  ./"$BIN" --train --data "$DATA_FILE" --index "$INDEX_BIN" --ckpt ckpt.bin --steps 2000 --batch 64 --seq 128 --gpus 2 --measure
fi
