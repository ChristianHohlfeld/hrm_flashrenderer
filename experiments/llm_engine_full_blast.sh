#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# llm_engine_full_blast.sh
#
# Self-contained, CUDA-generating, promptable runtime script.
#
# Writes one .cu, builds it with nvcc, runs it.
#
# Features in the generated runtime:
#   - native checkpoint loader for the bounded small-core format
#   - explicit VRAM preflight / budgeting
#   - single-GPU-first deterministic boot
#   - paged KV with host spill + resident GPU window
#   - bounded decode loop / repeat guard / blank guard
#   - interactive prompt loop
#
# Intended checkpoint format:
#   the bounded/native format used by your small-core runs
#   (magic/version/pair_k1/K2/D/H/L/F/TMAX + fixed trailing weights)
#
# Usage:
#   ./llm_engine_full_blast.sh
#   CKPT=./native_student_train/student_candidate_1.bin ./llm_engine_full_blast.sh
#   INDEX=./index_v7_k18192_k28192.bin ./llm_engine_full_blast.sh
#   GPU=1 CTX_LIMIT=256 MAX_NEW=64 ./llm_engine_full_blast.sh
# =============================================================================

BIN="${BIN:-llm_engine_full_blast}"
CU="${CU:-llm_engine_full_blast.cu}"
WORKDIR="${WORKDIR:-$PWD}"

cat > "$WORKDIR/$CU" <<'CU'
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#define CUDA_OK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);} } while(0)

struct RuntimeConfig {
  int gpu_index = 0;
  int ctx_limit = 256;
  int max_new = 64;
  int kv_page_tokens = 32;
  int kv_gpu_pages_max = 8;
  int safety_mb = 512;
  bool verbose = true;
  bool no_repeat_guard = true;
  bool blank_abort = true;
};

static RuntimeConfig g_rt;

static int getenv_int(const char* name, int fallback) {
  const char* s = std::getenv(name);
  return (s && *s) ? std::atoi(s) : fallback;
}

static std::string getenv_str(const char* name, const char* fallback) {
  const char* s = std::getenv(name);
  return (s && *s) ? std::string(s) : std::string(fallback);
}

struct MemoryBudget {
  size_t free_bytes=0,total_bytes=0,safety_bytes=0;
  size_t weights_bytes=0,kv_bytes=0,logits_bytes=0,scratch_bytes=0,misc_bytes=0;
  size_t projected_total() const { return weights_bytes+kv_bytes+logits_bytes+scratch_bytes+misc_bytes; }
  bool safe() const { return projected_total()+safety_bytes <= free_bytes; }
};

static MemoryBudget make_budget(size_t wb,size_t kb,size_t lb,size_t sb,size_t mb) {
  MemoryBudget b{};
  CUDA_OK(cudaMemGetInfo(&b.free_bytes,&b.total_bytes));
  b.safety_bytes = (size_t)g_rt.safety_mb * 1024ull * 1024ull;
  b.weights_bytes = wb; b.kv_bytes=kb; b.logits_bytes=lb; b.scratch_bytes=sb; b.misc_bytes=mb;
  return b;
}

static void print_budget(const MemoryBudget& b) {
  auto mb=[](size_t x){ return double(x)/(1024.0*1024.0); };
  fprintf(stderr,
    "[budget] free=%.1fMB total=%.1fMB safety=%.1fMB\n"
    "[budget] weights=%.1fMB kv=%.1fMB logits=%.1fMB scratch=%.1fMB misc=%.1fMB projected=%.1fMB\n",
    mb(b.free_bytes), mb(b.total_bytes), mb(b.safety_bytes),
    mb(b.weights_bytes), mb(b.kv_bytes), mb(b.logits_bytes), mb(b.scratch_bytes), mb(b.misc_bytes), mb(b.projected_total()));
}

static void preflight_gpu() {
  int count=0;
  CUDA_OK(cudaGetDeviceCount(&count));
  if(count<=0) { fprintf(stderr,"FATAL: no CUDA devices\n"); exit(1); }
  if(g_rt.gpu_index < 0 || g_rt.gpu_index >= count) { fprintf(stderr,"FATAL: bad GPU=%d count=%d\n", g_rt.gpu_index, count); exit(1); }
  CUDA_OK(cudaSetDevice(g_rt.gpu_index));
  cudaDeviceProp prop{};
  CUDA_OK(cudaGetDeviceProperties(&prop, g_rt.gpu_index));
  fprintf(stderr,"[runtime] gpu=%d name=%s totalGlobalMem=%.1fMB\n",
          g_rt.gpu_index, prop.name, double(prop.totalGlobalMem)/(1024.0*1024.0));
}

struct CheckpointHeader {
  uint32_t magic0;
  uint32_t version;
  uint32_t pair_k1;
  uint32_t k2;
  uint32_t D;
  uint32_t H;
  uint32_t L;
  uint32_t F;
  uint32_t T;
};

struct Model {
  // meta
  uint32_t pair_k1=0, k2=0;
  int D=0,H=0,L=0,F=0,T=0,Dhf=0,V=0,Vpad=0;
  // host
  std::vector<int32_t> wte, wpe, final_norm, lm_head;
  struct LayerH {
    std::vector<int32_t> r1,q,k,v,o,r2,gate,down;
  };
  std::vector<LayerH> layers;
  // device
  int32_t *d_wte=nullptr,*d_final_norm=nullptr,*d_lm_head=nullptr;
  struct LayerD {
    int32_t *r1=nullptr,*q=nullptr,*k=nullptr,*v=nullptr,*o=nullptr,*r2=nullptr,*gate=nullptr,*down=nullptr;
  };
  std::vector<LayerD> dlayers;
};

static size_t model_weight_bytes_est(int D, int H, int L, int F, int Vpad, int T, int Dhf) {
  size_t elems = 0;
  elems += (size_t)Vpad * D;     // wte
  elems += (size_t)T * D;        // wpe
  for(int i=0;i<L;i++) {
    elems += Dhf;                // r1
    elems += 4ull * Dhf * Dhf;   // qkv o
    elems += Dhf;                // r2
    elems += (size_t)F * Dhf;    // gate
    elems += (size_t)Dhf * F;    // down
  }
  elems += D;                    // final norm
  elems += (size_t)D * Vpad;     // lm_head
  return elems * sizeof(int32_t);
}

static void read_exact(std::ifstream& in, void* ptr, size_t n) {
  in.read(reinterpret_cast<char*>(ptr), (std::streamsize)n);
  if((size_t)in.gcount() != n) {
    fprintf(stderr, "FATAL: short read (%zu bytes wanted)\n", n);
    exit(1);
  }
}

static void load_checkpoint(const std::string& path, Model& M) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if(!in) { fprintf(stderr, "FATAL: cannot open ckpt: %s\n", path.c_str()); exit(1); }
  std::streamsize fsz = in.tellg();
  in.seekg(0, std::ios::beg);

  CheckpointHeader h{};
  read_exact(in, &h, sizeof(h));
  if(h.magic0 != 0x43484452u) {
    fprintf(stderr, "FATAL: bad ckpt magic\n");
    exit(1);
  }

  M.pair_k1 = h.pair_k1;
  M.k2      = h.k2;
  M.D       = (int)h.D;
  M.H       = (int)h.H;
  M.L       = (int)h.L;
  M.F       = (int)h.F;
  M.T       = (int)h.T;
  M.Dhf     = M.D / 2;
  M.V       = 256 + (int)M.pair_k1 + (int)M.k2;
  M.Vpad    = ((M.V + 15) / 16) * 16;

  if(M.D != 256 || M.H != 8) {
    fprintf(stderr, "FATAL: this runtime expects bounded core D=256 H=8; got D=%d H=%d\n", M.D, M.H);
    exit(1);
  }

  size_t weight_bytes = model_weight_bytes_est(M.D, M.H, M.L, M.F, M.Vpad, M.T, M.Dhf);
  if((size_t)fsz < weight_bytes) {
    fprintf(stderr, "FATAL: ckpt too small for expected trailing weights\n");
    exit(1);
  }

  size_t weights_off = (size_t)fsz - weight_bytes;
  in.seekg((std::streamoff)weights_off, std::ios::beg);

  auto read_vec = [&](std::vector<int32_t>& v, size_t n){
    v.resize(n);
    read_exact(in, v.data(), n * sizeof(int32_t));
  };

  read_vec(M.wte, (size_t)M.Vpad * M.D);
  read_vec(M.wpe, (size_t)M.T * M.D);

  M.layers.resize(M.L);
  for(int i=0;i<M.L;i++) {
    read_vec(M.layers[i].r1,   M.Dhf);
    read_vec(M.layers[i].q,    (size_t)M.Dhf * M.Dhf);
    read_vec(M.layers[i].k,    (size_t)M.Dhf * M.Dhf);
    read_vec(M.layers[i].v,    (size_t)M.Dhf * M.Dhf);
    read_vec(M.layers[i].o,    (size_t)M.Dhf * M.Dhf);
    read_vec(M.layers[i].r2,   M.Dhf);
    read_vec(M.layers[i].gate, (size_t)M.F * M.Dhf);
    read_vec(M.layers[i].down, (size_t)M.Dhf * M.F);
  }

  read_vec(M.final_norm, M.D);
  read_vec(M.lm_head,    (size_t)M.D * M.Vpad);

  fprintf(stderr, "[ckpt] loaded D=%d H=%d L=%d F=%d T=%d Vpad=%d\n", M.D, M.H, M.L, M.F, M.T, M.Vpad);
}

static void upload_model(Model& M) {
  auto up = [](int32_t*& d, const std::vector<int32_t>& h){
    CUDA_OK(cudaMalloc(&d, h.size() * sizeof(int32_t)));
    CUDA_OK(cudaMemcpy(d, h.data(), h.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
  };
  up(M.d_wte, M.wte);
  up(M.d_final_norm, M.final_norm);
  up(M.d_lm_head, M.lm_head);
  M.dlayers.resize(M.L);
  for(int i=0;i<M.L;i++) {
    up(M.dlayers[i].r1,   M.layers[i].r1);
    up(M.dlayers[i].q,    M.layers[i].q);
    up(M.dlayers[i].k,    M.layers[i].k);
    up(M.dlayers[i].v,    M.layers[i].v);
    up(M.dlayers[i].o,    M.layers[i].o);
    up(M.dlayers[i].r2,   M.layers[i].r2);
    up(M.dlayers[i].gate, M.layers[i].gate);
    up(M.dlayers[i].down, M.layers[i].down);
  }
}

__global__ void gather_embedding_kernel(const int32_t* wte, int Vpad, int D, int tok, float scale, float* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < D) out[i] = (float)wte[tok * D + i] / scale;
}

__global__ void mul_vec_kernel(float* x, const int32_t* s, int n, float scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n) x[i] *= ((float)s[i] / scale);
}

__global__ void copy_first_half_kernel(const float* h, float* x, int Dhf) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < Dhf) x[i] = h[i];
}

__global__ void write_first_half_kernel(float* h, const float* x, int Dhf) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < Dhf) h[i] = x[i];
}

__global__ void matvec_i32_kernel(const int32_t* W, const float* x, float* y, int rows, int cols, float scale) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if(r >= rows) return;
  float acc = 0.f;
  const int32_t* wr = W + (size_t)r * cols;
  for(int c=0;c<cols;c++) acc += ((float)wr[c] / scale) * x[c];
  y[r] = acc;
}

__global__ void add_vec_kernel(float* a, const float* b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n) a[i] += b[i];
}

__global__ void tanh_kernel(float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n) x[i] = tanhf(x[i]);
}

__global__ void gelu_kernel(float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n) {
    float v = x[i];
    x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
  }
}

__global__ void attention_scores_window_kernel(const float* q, const float* kbuf, int tokens, int Dhf, float* scores) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if(t >= tokens) return;
  const float* k = kbuf + (size_t)t * Dhf;
  float acc = 0.f;
  for(int i=0;i<Dhf;i++) acc += q[i] * k[i];
  scores[t] = acc / sqrtf((float)Dhf);
}

__global__ void softmax_kernel(float* scores, int n) {
  // single block version for n<=512
  __shared__ float mx;
  __shared__ float sm;
  if(threadIdx.x == 0) { mx = -1e30f; sm = 0.f; }
  __syncthreads();

  int i = threadIdx.x;
  float v = (i < n) ? scores[i] : -1e30f;
  atomicMax((int*)&mx, __float_as_int(v));
  __syncthreads();

  float e = 0.f;
  if(i < n) {
    e = expf(scores[i] - mx);
    scores[i] = e;
  }
  __syncthreads();

  if(i < n) atomicAdd(&sm, scores[i]);
  __syncthreads();

  if(i < n) scores[i] /= (sm + 1e-9f);
}

__global__ void weighted_sum_window_kernel(const float* probs, const float* vbuf, int tokens, int Dhf, float* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= Dhf) return;
  float acc = 0.f;
  for(int t=0;t<tokens;t++) acc += probs[t] * vbuf[(size_t)t * Dhf + i];
  out[i] = acc;
}

__global__ void logits_kernel(const float* h, const int32_t* lm_head, int D, int Vpad, float scale, float* logits) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if(v >= Vpad) return;
  float acc = 0.f;
  for(int i=0;i<D;i++) acc += h[i] * ((float)lm_head[(size_t)i * Vpad + v] / scale);
  logits[v] = acc;
}

struct PagedLayerKV {
  int Dhf=0;
  int page_tokens=0;
  int gpu_pages_max=0;
  int resident_tokens=0;
  int total_tokens=0;
  int ctx_limit=0;

  // resident GPU window [resident_tokens x Dhf]
  float *k_gpu=nullptr, *v_gpu=nullptr;

  // host spill for all tokens (pinned)
  float *k_host=nullptr, *v_host=nullptr;
};

struct PagedKV {
  int L=0;
  int Dhf=0;
  int ctx_limit=0;
  int page_tokens=0;
  int gpu_pages_max=0;
  std::vector<PagedLayerKV> layers;
};

static size_t estimate_paged_kv_gpu_bytes(int L, int Dhf) {
  size_t resident_tokens = (size_t)g_rt.kv_page_tokens * (size_t)g_rt.kv_gpu_pages_max;
  return (size_t)L * 2ull * resident_tokens * (size_t)Dhf * sizeof(float);
}

static void paged_kv_init(PagedKV& kv, int L, int Dhf, int ctx_limit) {
  kv.L = L; kv.Dhf = Dhf; kv.ctx_limit = ctx_limit;
  kv.page_tokens = g_rt.kv_page_tokens;
  kv.gpu_pages_max = g_rt.kv_gpu_pages_max;
  kv.layers.resize(L);

  int resident_tokens = kv.page_tokens * kv.gpu_pages_max;
  for(int li=0; li<L; ++li) {
    auto& x = kv.layers[li];
    x.Dhf = Dhf;
    x.page_tokens = kv.page_tokens;
    x.gpu_pages_max = kv.gpu_pages_max;
    x.resident_tokens = resident_tokens;
    x.total_tokens = 0;
    x.ctx_limit = ctx_limit;

    CUDA_OK(cudaMalloc(&x.k_gpu, (size_t)resident_tokens * Dhf * sizeof(float)));
    CUDA_OK(cudaMalloc(&x.v_gpu, (size_t)resident_tokens * Dhf * sizeof(float)));
    CUDA_OK(cudaHostAlloc(&x.k_host, (size_t)ctx_limit * Dhf * sizeof(float), cudaHostAllocPortable));
    CUDA_OK(cudaHostAlloc(&x.v_host, (size_t)ctx_limit * Dhf * sizeof(float), cudaHostAllocPortable));
    std::memset(x.k_host, 0, (size_t)ctx_limit * Dhf * sizeof(float));
    std::memset(x.v_host, 0, (size_t)ctx_limit * Dhf * sizeof(float));
  }
}

static void paged_kv_append(PagedKV& kv, int layer, int token_pos, const float* d_k, const float* d_v) {
  auto& x = kv.layers[layer];
  if(token_pos >= x.ctx_limit) return;

  size_t row_bytes = (size_t)x.Dhf * sizeof(float);

  // always keep host canonical
  CUDA_OK(cudaMemcpy(x.k_host + (size_t)token_pos * x.Dhf, d_k, row_bytes, cudaMemcpyDeviceToHost));
  CUDA_OK(cudaMemcpy(x.v_host + (size_t)token_pos * x.Dhf, d_v, row_bytes, cudaMemcpyDeviceToHost));

  int resident_cap = x.resident_tokens;
  int window_start = std::max(0, token_pos - resident_cap + 1);
  int local_pos = token_pos - window_start;

  // refresh resident window from host when it slides
  int active = token_pos - window_start + 1;
  CUDA_OK(cudaMemcpy(x.k_gpu, x.k_host + (size_t)window_start * x.Dhf, (size_t)active * x.Dhf * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(x.v_gpu, x.v_host + (size_t)window_start * x.Dhf, (size_t)active * x.Dhf * sizeof(float), cudaMemcpyHostToDevice));
  x.total_tokens = token_pos + 1;
}

static int paged_kv_active_tokens(const PagedKV& kv, int layer) {
  const auto& x = kv.layers[layer];
  return std::min(x.total_tokens, x.resident_tokens);
}

static const float* paged_kv_kptr(const PagedKV& kv, int layer) { return kv.layers[layer].k_gpu; }
static const float* paged_kv_vptr(const PagedKV& kv, int layer) { return kv.layers[layer].v_gpu; }

struct DecodeGuards {
  int blank_steps = 0;
  int repeat_steps = 0;
  int last_tok = -1;
};

static bool printable_token(int tok) {
  return tok >= 32 && tok < 127;
}

static bool should_abort_decode(DecodeGuards& g, int tok) {
  if(tok == g.last_tok) g.repeat_steps++;
  else g.repeat_steps = 0;
  if(!printable_token(tok)) g.blank_steps++;
  else g.blank_steps = 0;
  g.last_tok = tok;
  if(g_rt.blank_abort && g.blank_steps >= 8) return true;
  if(g_rt.no_repeat_guard && g.repeat_steps >= 16) return true;
  return false;
}

static int best_token_from_logits(const std::vector<float>& logits, int V) {
  int best = 0;
  float bv = -std::numeric_limits<float>::infinity();
  for(int i=0;i<V;i++) {
    float x = logits[i];
    if(i < 256) x -= 0.15f; // mild control character suppression
    if(x > bv) { bv = x; best = i; }
  }
  return best;
}

static std::vector<int> encode_prompt_ids(const std::string& s, int V) {
  std::vector<int> out;
  for(unsigned char c : s) {
    if(c < 256) out.push_back((int)c);
  }
  if(out.empty()) out.push_back((int)'?');
  return out;
}

static std::string render_tokens(const std::vector<int>& toks) {
  std::string out;
  char buf[32];
  for(int t : toks) {
    if(t >= 32 && t < 127) out.push_back((char)t);
    else { std::snprintf(buf, sizeof(buf), "<%d>", t); out += buf; }
  }
  return out;
}

static int generate_one(Model& M, PagedKV& kv, int token_pos, int input_tok) {
  const float SCALE = 1000.f;
  int D = M.D, Dhf = M.Dhf, F = M.F, Vpad = M.Vpad;

  float *d_h=nullptr,*d_x=nullptr,*d_qv=nullptr,*d_kv=nullptr,*d_vv=nullptr,*d_tmp=nullptr,*d_ff=nullptr,*d_scores=nullptr,*d_logits=nullptr;
  CUDA_OK(cudaMalloc(&d_h, D * sizeof(float)));
  CUDA_OK(cudaMalloc(&d_x, Dhf * sizeof(float)));
  CUDA_OK(cudaMalloc(&d_qv, Dhf * sizeof(float)));
  CUDA_OK(cudaMalloc(&d_kv, Dhf * sizeof(float)));
  CUDA_OK(cudaMalloc(&d_vv, Dhf * sizeof(float)));
  CUDA_OK(cudaMalloc(&d_tmp, Dhf * sizeof(float)));
  CUDA_OK(cudaMalloc(&d_ff, F * sizeof(float)));
  CUDA_OK(cudaMalloc(&d_scores, std::max(1, g_rt.ctx_limit) * sizeof(float)));
  CUDA_OK(cudaMalloc(&d_logits, Vpad * sizeof(float)));

  gather_embedding_kernel<<<(D+255)/256,256>>>(M.d_wte, Vpad, D, input_tok % Vpad, SCALE, d_h);

  for(int li=0; li<M.L; ++li) {
    const auto& Ld = M.dlayers[li];

    copy_first_half_kernel<<<(Dhf+255)/256,256>>>(d_h, d_x, Dhf);
    mul_vec_kernel<<<(Dhf+255)/256,256>>>(d_x, Ld.r1, Dhf, SCALE);

    matvec_i32_kernel<<<(Dhf+255)/256,256>>>(Ld.q, d_x, d_qv, Dhf, Dhf, SCALE);
    matvec_i32_kernel<<<(Dhf+255)/256,256>>>(Ld.k, d_x, d_kv, Dhf, Dhf, SCALE);
    matvec_i32_kernel<<<(Dhf+255)/256,256>>>(Ld.v, d_x, d_vv, Dhf, Dhf, SCALE);

    paged_kv_append(kv, li, token_pos, d_kv, d_vv);

    int active = paged_kv_active_tokens(kv, li);
    const float* kbuf = paged_kv_kptr(kv, li);
    const float* vbuf = paged_kv_vptr(kv, li);

    attention_scores_window_kernel<<<(active+255)/256,256>>>(d_qv, kbuf, active, Dhf, d_scores);
    softmax_kernel<<<1,512>>>(d_scores, active);
    weighted_sum_window_kernel<<<(Dhf+255)/256,256>>>(d_scores, vbuf, active, Dhf, d_tmp);

    matvec_i32_kernel<<<(Dhf+255)/256,256>>>(Ld.o, d_tmp, d_x, Dhf, Dhf, SCALE);
    mul_vec_kernel<<<(Dhf+255)/256,256>>>(d_x, Ld.r2, Dhf, SCALE);

    matvec_i32_kernel<<<(F+255)/256,256>>>(Ld.gate, d_x, d_ff, F, Dhf, SCALE);
    gelu_kernel<<<(F+255)/256,256>>>(d_ff, F);
    matvec_i32_kernel<<<(Dhf+255)/256,256>>>(Ld.down, d_ff, d_tmp, Dhf, F, SCALE);
    add_vec_kernel<<<(Dhf+255)/256,256>>>(d_x, d_tmp, Dhf);
    tanh_kernel<<<(Dhf+255)/256,256>>>(d_x, Dhf);

    write_first_half_kernel<<<(Dhf+255)/256,256>>>(d_h, d_x, Dhf);
  }

  mul_vec_kernel<<<(D+255)/256,256>>>(d_h, M.d_final_norm, D, SCALE);
  logits_kernel<<<(Vpad+255)/256,256>>>(d_h, M.d_lm_head, D, Vpad, SCALE, d_logits);

  std::vector<float> h_logits(Vpad);
  CUDA_OK(cudaMemcpy(h_logits.data(), d_logits, (size_t)Vpad * sizeof(float), cudaMemcpyDeviceToHost));
  int next_tok = best_token_from_logits(h_logits, M.V);

  cudaFree(d_h); cudaFree(d_x); cudaFree(d_qv); cudaFree(d_kv); cudaFree(d_vv);
  cudaFree(d_tmp); cudaFree(d_ff); cudaFree(d_scores); cudaFree(d_logits);
  return next_tok;
}

int main() {
  g_rt.gpu_index        = getenv_int("GPU", 0);
  g_rt.ctx_limit        = getenv_int("CTX_LIMIT", 256);
  g_rt.max_new          = getenv_int("MAX_NEW", 64);
  g_rt.kv_page_tokens   = getenv_int("KV_PAGE_TOKENS", 32);
  g_rt.kv_gpu_pages_max = getenv_int("KV_GPU_PAGES_MAX", 8);
  g_rt.safety_mb        = getenv_int("SAFETY_MB", 512);

  std::string ckpt = getenv_str("CKPT", "./ckpt_llm_engine_b957ab5a.bin");
  std::string index = getenv_str("INDEX", "./index_v7_k18192_k28192.bin");
  (void)index; // kept for interface parity

  preflight_gpu();

  Model M;
  load_checkpoint(ckpt, M);

  size_t wb = model_weight_bytes_est(M.D, M.H, M.L, M.F, M.Vpad, M.T, M.Dhf);
  size_t kb = estimate_paged_kv_gpu_bytes(M.L, M.Dhf);
  size_t lb = (size_t)M.Vpad * sizeof(float);
  size_t sb = (size_t)(M.D + 6ull*M.Dhf + M.F + g_rt.ctx_limit) * sizeof(float);
  size_t mb = 128ull * 1024ull * 1024ull;
  auto bud = make_budget(wb, kb, lb, sb, mb);
  print_budget(bud);
  if(!bud.safe()) {
    fprintf(stderr, "FATAL: unsafe VRAM budget; refusing to launch.\n");
    return 2;
  }

  upload_model(M);

  PagedKV kv;
  paged_kv_init(kv, M.L, M.Dhf, std::min(g_rt.ctx_limit, M.T));

  fprintf(stderr, "[chat] promptable full-blast runtime. /quit to exit\n");

  for(;;) {
    std::cout << "> " << std::flush;
    std::string line;
    if(!std::getline(std::cin, line)) break;
    if(line == "/quit") break;

    auto toks = encode_prompt_ids(line, M.V);
    std::vector<int> out;
    int token_pos = 0;

    // prime context with prompt tokens
    for(int t : toks) {
      int next = generate_one(M, kv, token_pos, t);
      (void)next;
      token_pos++;
      if(token_pos >= g_rt.ctx_limit) break;
    }

    DecodeGuards dg{};
    int cur = toks.empty() ? (int)'?' : toks.back();
    for(int step=0; step<g_rt.max_new && token_pos < g_rt.ctx_limit; ++step) {
      int nxt = generate_one(M, kv, token_pos, cur);
      if(should_abort_decode(dg, nxt)) {
        fprintf(stderr, "[decode] guard abort\n");
        break;
      }
      out.push_back(nxt);
      cur = nxt;
      token_pos++;
    }

    if(token_pos >= g_rt.ctx_limit) {
      std::cout << "\n[ctx full]\n";
    } else {
      std::cout << render_tokens(out) << "\n";
    }
  }

  return 0;
}
CU

echo "[*] Building $BIN from $CU"
nvcc -O3 -std=c++17 -arch=sm_75 -lineinfo "$WORKDIR/$CU" -o "$WORKDIR/$BIN"

echo "[*] Launching $BIN"
exec "$WORKDIR/$BIN"
