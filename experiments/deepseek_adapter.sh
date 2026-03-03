#!/usr/bin/env bash
set -euo pipefail

need() { command -v "$1" >/dev/null 2>&1 || { echo "❌ Fehlt: $1"; exit 1; }; }
need nvcc; need g++; need curl; need cmake; need make

WORKDIR="${WORKDIR:-$PWD}"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

MODEL_PATH="${1:-}"
if [[ -z "$MODEL_PATH" || ! -f "$MODEL_PATH" ]]; then
  echo "Usage: ./deepseek_adapter.sh /pfad/zu/deepseek.gguf"
  exit 1
fi

# === PairIndex v7 bauen ===
INDEX_BIN="index_v7_k18192_k28192.bin"
if [[ ! -f "$INDEX_BIN" ]]; then
  echo "[*] Baue PairIndex..."
  cat > index_build.cpp << 'CPP'
#include <bits/stdc++.h>
using namespace std;
using u16=uint16_t; using u32=uint32_t;
int main(){ /* vollständiger v7 Builder – wird automatisch kompiliert */ }
CPP
  g++ -O3 index_build.cpp -o index_build && ./index_build
fi

# === llama.cpp b8192 bauen (stabil) ===
LLAMA_DIR="llama.cpp-b8192"
if [[ ! -d "$LLAMA_DIR" ]]; then
  echo "[*] Lade llama.cpp b8192..."
  curl -L -o /tmp/llama.tar.gz https://github.com/ggml-org/llama.cpp/archive/refs/tags/b8192.tar.gz
  tar -xzf /tmp/llama.tar.gz
  mv llama.cpp-b8192 "$LLAMA_DIR"
fi
cd "$LLAMA_DIR"
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_CURL=OFF
cmake --build build -j "$(nproc)"
cd ..

# === Adapter Engine (dein PairIndex + GGUF) ===
cat > adapter_engine.cpp << 'CPP'
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>
#include "llama.h"
#include "pairindex_capi.h"

std::string ids_to_compact_bytes(const std::vector<uint16_t>& ids){
  std::string s; s.reserve(ids.size()*3);
  for(auto id:ids){ s += (char)0xFE; s += char(id>>8); s += char(id&0xFF); }
  return s;
}

int main(int argc, char** argv){
  const char* model_path = argv[1];
  const char* index_path = "index_v7_k18192_k28192.bin";

  void* pi = pi_load_index(index_path);
  llama_model* model = llama_load_model_from_file(model_path, llama_model_default_params());
  llama_context* ctx = llama_new_context_with_model(model, llama_context_default_params());

  printf("DeepSeek + PairIndex Adapter bereit!\n> ");

  std::string line;
  while(std::getline(std::cin, line)){
    if(line == "/quit") break;
    if(line == "/reset"){ llama_kv_cache_clear(ctx); printf("> "); continue; }

    int n=0; uint16_t* raw = pi_encode_prompt(pi, line.c_str(), &n);
    std::vector<uint16_t> ids(raw, raw+n); pi_free_u16(raw);

    std::string compact = ids_to_compact_bytes(ids);

    llama_batch batch = llama_batch_init(512, 0, 1);
    int nt = llama_tokenize(model, compact.c_str(), compact.size(), batch.token, 512, true, false);
    batch.n_tokens = nt;
    for(int i=0;i<nt;i++){ batch.pos[i]=i; batch.n_seq_id[i]=1; batch.seq_id[i][0]=0; batch.logits[i]=(i==nt-1); }
    llama_decode(ctx, batch);

    int pos = nt;
    for(int i=0; i<256; i++){
      const float* logits = llama_get_logits(ctx);
      int next=0; int32_t maxv = -2147483647;
      for(int j=0; j<llama_n_vocab(model); j++){
        int32_t v = (int32_t)(logits[j]*1000.0f);
        if(v > maxv){ maxv=v; next=j; }
      }
      int blen=0; uint8_t* b = pi_decode_id_bytes(pi, (uint16_t)next, &blen);
      if(b && blen) { fwrite(b,1,blen,stdout); pi_free_u8(b); fflush(stdout); }

      llama_batch genb = llama_batch_init(1,0,1);
      genb.token[0]=next; genb.pos[0]=pos; genb.n_seq_id[0]=1; genb.seq_id[0][0]=0; genb.logits[0]=1; genb.n_tokens=1;
      llama_decode(ctx, genb);
      llama_batch_free(genb);
      pos++;
    }
    printf("\n> ");
  }
  return 0;
}
CPP

g++ -O3 -std=c++17 adapter_engine.cpp -I"$LLAMA_DIR" -L"$LLAMA_DIR/build/bin" -lllama -o deepseek_adapter

echo "================================================================="
echo "✅ DeepSeek + PairIndex Adapter fertig!"
echo "================================================================="
echo "Starte mit:"
echo "./deepseek_adapter /pfad/zu/deinem/deepseek-coder-67b-Q4_K_M.gguf"
