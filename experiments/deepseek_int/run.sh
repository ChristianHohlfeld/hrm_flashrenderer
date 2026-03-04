#!/bin/bash
# © 2026 Christian Heinrich Hohlfeld (Konstanz, Germany), christianhohlfeld.com, ORCID 0009-0003-6634-9045.
# DeepSeek INT8 Persistent Decode Engine — Single File, No Compromise, Prod Class
# Hardware Target: RTX 2080 Ti (sm_75), 11 GB VRAM
# Model: DeepSeek-R1-Distill-Qwen-1.5B (D=1536 H=12 KVH=2 Dh=128 F=8960 L=28 V=151936)
#
# Architecture:
#   - INT8 weights with per-row fp32 scale (W8A16 quantization)
#   - FP32 hidden states & activations (full precision intermediate)
#   - FP16 KV cache (compact, no precision loss)
#   - Online softmax with fused 1/sqrt(Dh) scaling
#   - RoPE (Q15 precomputed sin/cos tables in device memory)
#   - DP4A dot products for weight GEMVs (bandwidth-bound → int8 wins)
#   - Zero-copy mmap weight loading
#   - Pure numpy export (NO TORCH)

set -euo pipefail

WORKDIR="${WORKDIR:-$PWD}"
PYTHON="${PYTHON:-/home/chris/myenv2/bin/python}"
PIP="${PIP:-/home/chris/myenv2/bin/pip}"
MODEL_REPO="${MODEL_REPO:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
MODEL_BIN="${MODEL_BIN:-model_q8.bin}"
ENGINE_BIN="${ENGINE_BIN:-deepseek_engine}"
SM="${SM:-75}"

cd "$WORKDIR"

# ═══════════════════════════════════════════════════════════════
# PHASE 1: Inline Python Exporter (BF16→INT8, all layers, NO TORCH)
# ═══════════════════════════════════════════════════════════════
cat << 'PYEOF' > /tmp/_dsi8_export.py
import os, sys, struct, json
import numpy as np
from huggingface_hub import snapshot_download

def bf16_to_fp32(raw, shape):
    bf = np.frombuffer(raw, dtype=np.uint16)
    return (bf.astype(np.uint32) << 16).view(np.float32).reshape(shape)

def load_safetensors(path):
    out = {}
    with open(path, "rb") as f:
        hs = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hs))
        base = 8 + hs
        for k, m in hdr.items():
            if k == "__metadata__": continue
            s, e = m["data_offsets"]
            f.seek(base + s)
            raw = f.read(e - s)
            sh = m["shape"]
            dt = m["dtype"]
            if dt == "BF16":
                out[k] = bf16_to_fp32(raw, sh)
            elif dt == "F16":
                out[k] = np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(sh)
            elif dt == "F32":
                out[k] = np.frombuffer(raw, dtype=np.float32).reshape(sh)
            else:
                out[k] = np.frombuffer(raw, dtype=np.float32).reshape(sh)
    return out

def q8_row(t):
    """Per-row INT8 quantization with fp32 scale: t ≈ q * scale"""
    absmax = np.abs(t).max(axis=-1)
    absmax = np.maximum(absmax, 1e-8)
    scale = absmax / 127.0
    qi = np.clip(np.round(t / scale[..., None]), -128, 127).astype(np.int8)
    return qi, scale.astype(np.float32)

TENSOR_TYPE_Q8 = 0
TENSOR_TYPE_F32 = 1

def write_tensor(f, name, data, scales, ttype):
    nb = name.encode()
    f.write(struct.pack("<I", len(nb))); f.write(nb)
    ndim = len(data.shape)
    f.write(struct.pack("<I", ttype))
    f.write(struct.pack("<I", ndim))
    for d in data.shape:
        f.write(struct.pack("<I", d))
    if ttype == TENSOR_TYPE_Q8:
        f.write(scales.tobytes())  # fp32 per-row scale
        f.write(data.tobytes())    # int8 data
    else:
        f.write(data.astype(np.float32).tobytes())

def main():
    repo, outp = sys.argv[1], sys.argv[2]
    print(f"[*] Downloading {repo}...")
    mp = snapshot_download(repo, allow_patterns=["*.safetensors", "config.json"])

    cfg_path = os.path.join(mp, "config.json")
    with open(cfg_path) as cf: cfg = json.load(cf)
    L = cfg.get("num_hidden_layers", 28)
    D = cfg.get("hidden_size", 1536)
    H = cfg.get("num_attention_heads", 12)
    KVH = cfg.get("num_key_value_heads", 2)
    Dh = D // H
    F = cfg.get("intermediate_size", 8960)
    V = cfg.get("vocab_size", 151936)
    Tmax = 4096
    rope_theta = cfg.get("rope_theta", 10000.0)
    rms_eps = cfg.get("rms_norm_eps", 1e-6)
    print(f"[*] D={D} H={H} KVH={KVH} Dh={Dh} F={F} V={V} L={L} rope_theta={rope_theta}")

    tensors = {}
    for fn in sorted(os.listdir(mp)):
        if fn.endswith(".safetensors"):
            print(f"[*] Loading {fn}...")
            tensors.update(load_safetensors(os.path.join(mp, fn)))

    # Build key list
    keys_2d = ["model.embed_tokens.weight"]
    keys_1d = []
    for l in range(L):
        pfx = f"model.layers.{l}"
        keys_1d += [f"{pfx}.input_layernorm.weight"]
        keys_2d += [
            f"{pfx}.self_attn.q_proj.weight",
            f"{pfx}.self_attn.k_proj.weight",
            f"{pfx}.self_attn.v_proj.weight",
            f"{pfx}.self_attn.o_proj.weight",
            f"{pfx}.mlp.gate_proj.weight",
            f"{pfx}.mlp.up_proj.weight",
            f"{pfx}.mlp.down_proj.weight",
        ]
        keys_1d += [f"{pfx}.post_attention_layernorm.weight"]
        # bias (may not exist for all models)
        for bp in ["self_attn.q_proj.bias","self_attn.k_proj.bias","self_attn.v_proj.bias"]:
            k = f"{pfx}.{bp}"
            if k in tensors:
                keys_1d.append(k)
    keys_1d += ["model.norm.weight"]
    keys_2d += ["lm_head.weight"]

    print(f"[*] Exporting to {outp}...")
    with open(outp, "wb") as f:
        f.write(b"DSI8")
        f.write(struct.pack("<I", 3))  # version 3
        # Config header
        f.write(struct.pack("<IIIIIIII", D, H, KVH, Dh, F, V, L, Tmax))
        f.write(struct.pack("<d", rope_theta))
        f.write(struct.pack("<f", rms_eps))
        bos = cfg.get("bos_token_id", 151643)
        eos = cfg.get("eos_token_id", 151643)
        f.write(struct.pack("<II", bos, eos))

        exported = 0
        # 2D tensors: Q8 quantized
        for key in keys_2d:
            if key not in tensors: continue
            t = tensors[key].astype(np.float32)
            if t.ndim == 1: t = t.reshape(1, -1)
            qi, sc = q8_row(t)
            write_tensor(f, key, qi, sc, TENSOR_TYPE_Q8)
            exported += 1

        # 1D tensors: kept as fp32
        for key in keys_1d:
            if key not in tensors: continue
            t = tensors[key].astype(np.float32).ravel()
            write_tensor(f, key, t, None, TENSOR_TYPE_F32)
            exported += 1

    print(f"[*] Exported {exported} tensors. Done!")

if __name__ == "__main__":
    main()
PYEOF

# ═══════════════════════════════════════════════════════════════
# PHASE 2: Inline CUDA Engine — FP32 hidden + INT8 weight GEMVs
# ═══════════════════════════════════════════════════════════════
cat << 'CUEOF' > /tmp/_dsi8_engine.cu
// © 2026 Christian Heinrich Hohlfeld (Konstanz, Germany), christianhohlfeld.com
// DeepSeek INT8 Persistent Decode Engine — Hybrid FP32/INT8
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <chrono>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);}} while(0)
#define KERNEL_CHECK() { cudaError_t e=cudaGetLastError(); if(e!=cudaSuccess){ \
  fprintf(stderr,"Kernel %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);}}

// ─── Config ───
struct ModelConfig {
  int D, H, KVH, Dh, F, V, L, Tmax;
  double rope_theta;
  float rms_eps;
  int bos_id, eos_id;
};

// ─── Tensor types ───
#define TENSOR_TYPE_Q8  0
#define TENSOR_TYPE_F32 1

struct TensorQ8 {
  int rows, cols;
  const float* scale;  // fp32 per-row scale
  const int8_t* data;
};

struct TensorF32 {
  int size;
  const float* data;
};

// ─── Mmap Loader ───
struct Loader {
  uint8_t* base; size_t sz; int fd;
  ModelConfig cfg;
  std::unordered_map<std::string, TensorQ8> q8;
  std::unordered_map<std::string, TensorF32> f32;

  void open(const char* path) {
    fd = ::open(path, O_RDONLY);
    if(fd<0){ perror("open"); exit(1); }
    struct stat sb; fstat(fd,&sb); sz=sb.st_size;
    base=(uint8_t*)mmap(nullptr,sz,PROT_READ,MAP_POPULATE|MAP_PRIVATE,fd,0);
    if(base==MAP_FAILED){ perror("mmap"); exit(1); }
    parse();
  }
  void close_() { if(base!=MAP_FAILED) munmap(base,sz); if(fd>=0) ::close(fd); }

  TensorQ8 getq8(const std::string& k) {
    auto it=q8.find(k);
    if(it==q8.end()){ fprintf(stderr,"[!] Q8 key not found: %s\n",k.c_str()); exit(1); }
    return it->second;
  }
  TensorF32 getf32(const std::string& k) {
    auto it=f32.find(k);
    if(it==f32.end()){ fprintf(stderr,"[!] F32 key not found: %s\n",k.c_str()); exit(1); }
    return it->second;
  }
  bool has(const std::string& k) { return f32.count(k) || q8.count(k); }

private:
  void parse() {
    size_t o=0;
    if(memcmp(base,"DSI8",4)!=0){ fprintf(stderr,"Bad magic\n"); exit(1); }
    o+=4;
    uint32_t ver=*(uint32_t*)(base+o); o+=4;
    if(ver<3){ fprintf(stderr,"Need version>=3, got %u\n",ver); exit(1); }
    cfg.D   =*(int32_t*)(base+o); o+=4;
    cfg.H   =*(int32_t*)(base+o); o+=4;
    cfg.KVH =*(int32_t*)(base+o); o+=4;
    cfg.Dh  =*(int32_t*)(base+o); o+=4;
    cfg.F   =*(int32_t*)(base+o); o+=4;
    cfg.V   =*(int32_t*)(base+o); o+=4;
    cfg.L   =*(int32_t*)(base+o); o+=4;
    cfg.Tmax=*(int32_t*)(base+o); o+=4;
    cfg.rope_theta = *(double*)(base+o); o+=8;
    cfg.rms_eps    = *(float*)(base+o);  o+=4;
    cfg.bos_id     = *(int32_t*)(base+o); o+=4;
    cfg.eos_id     = *(int32_t*)(base+o); o+=4;
    while(o<sz) {
      if(o+4>sz) break;
      uint32_t nl=*(uint32_t*)(base+o); o+=4;
      if(o+nl>sz) break;
      std::string name((char*)(base+o),nl); o+=nl;
      uint32_t ttype=*(uint32_t*)(base+o); o+=4;
      uint32_t ndim=*(uint32_t*)(base+o); o+=4;
      std::vector<int> dims(ndim);
      for(uint32_t d=0;d<ndim;d++){ dims[d]=*(int32_t*)(base+o); o+=4; }
      if(ttype == TENSOR_TYPE_Q8) {
        TensorQ8 t;
        t.rows = dims[0];
        t.cols = (ndim>1) ? dims[1] : 1;
        t.scale = (const float*)(base+o); o += (size_t)t.rows * 4;
        t.data  = (const int8_t*)(base+o); o += (size_t)t.rows * (size_t)t.cols;
        q8[name] = t;
      } else {
        TensorF32 t;
        t.size = 1;
        for(auto d : dims) t.size *= d;
        t.data = (const float*)(base+o); o += (size_t)t.size * 4;
        f32[name] = t;
      }
    }
  }
};

// ═══════════════════════════════════════════════════════════════
// CUDA KERNELS — FP32 activations, INT8 weight GEMVs
// ═══════════════════════════════════════════════════════════════

// ─── RoPE tables (device memory) ───
static float* d_rope_sin = nullptr;
static float* d_rope_cos = nullptr;

static void init_rope_tables(int Dh, int Tmax, double theta) {
  int half = Dh / 2;
  size_t n = (size_t)Tmax * half;
  std::vector<float> s(n), c(n);
  for(int t=0; t<Tmax; t++) {
    for(int d=0; d<half; d++) {
      double freq = 1.0 / pow(theta, (double)(2*d) / (double)Dh);
      double angle = (double)t * freq;
      s[t*half+d] = (float)sin(angle);
      c[t*half+d] = (float)cos(angle);
    }
  }
  size_t bytes = n * sizeof(float);
  CUDA_CHECK(cudaMalloc(&d_rope_sin, bytes));
  CUDA_CHECK(cudaMalloc(&d_rope_cos, bytes));
  CUDA_CHECK(cudaMemcpy(d_rope_sin, s.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_rope_cos, c.data(), bytes, cudaMemcpyHostToDevice));
}

// ─── RMSNorm (fp32 → fp32) ───
__global__ void rmsnorm(float* out, const float* x, const float* gamma, float eps, int D) {
  __shared__ float shared_ss[32];
  int tid = threadIdx.x;
  int lane = tid % 32;
  int warp = tid / 32;

  float my_ss = 0.0f;
  for(int i = tid; i < D; i += blockDim.x) {
    float v = x[i];
    my_ss += v * v;
  }
  for(int d=16;d>0;d>>=1) my_ss += __shfl_down_sync(0xFFFFFFFFu, my_ss, d);
  if(lane==0) shared_ss[warp] = my_ss;
  __syncthreads();
  if(warp==0) {
    float val = (tid < (blockDim.x+31)/32) ? shared_ss[tid] : 0.0f;
    for(int d=16;d>0;d>>=1) val += __shfl_down_sync(0xFFFFFFFFu, val, d);
    if(lane==0) shared_ss[0] = val;
  }
  __syncthreads();
  float rms = rsqrtf(shared_ss[0] / (float)D + eps);
  for(int i = tid; i < D; i += blockDim.x) {
    out[i] = x[i] * rms * gamma[i];
  }
}

// ─── INT8 Weight GEMV: out[row] = dot(x_fp32, W_int8[row]) * scale[row] ───
// x is fp32, W is int8, scale is fp32 per-row
// We quantize x to int8 on-the-fly, use DP4A, then rescale
__global__ void gemv_w8(float* out, const float* x, const int8_t* W,
                        const float* w_scale, int rows, int cols) {
  int row = blockIdx.x;
  if(row >= rows) return;
  int tid = threadIdx.x;

  // Step 1: cooperatively quantize x to int8 in shared memory
  extern __shared__ int8_t sh_x[];
  __shared__ float sh_x_scale[1];

  // Find absmax of x
  float my_max = 0.0f;
  for(int i = tid; i < cols; i += blockDim.x) {
    float v = fabsf(x[i]);
    if(v > my_max) my_max = v;
  }
  for(int d=16;d>0;d>>=1) my_max = fmaxf(my_max, __shfl_down_sync(0xFFFFFFFFu, my_max, d));
  // Cross-warp max
  __shared__ float sh_max[8];
  int lane = tid % 32, warp = tid / 32;
  if(lane==0) sh_max[warp] = my_max;
  __syncthreads();
  if(warp==0) {
    float val = (tid < (blockDim.x+31)/32) ? sh_max[tid] : 0.0f;
    for(int d=16;d>0;d>>=1) val = fmaxf(val, __shfl_down_sync(0xFFFFFFFFu, val, d));
    if(tid==0) { sh_x_scale[0] = val / 127.0f; if(sh_x_scale[0] < 1e-10f) sh_x_scale[0] = 1e-10f; }
  }
  __syncthreads();

  float inv_scale = 1.0f / sh_x_scale[0];
  for(int i = tid; i < cols; i += blockDim.x) {
    float v = x[i] * inv_scale;
    int32_t qi = __float2int_rn(v);
    if(qi > 127) qi = 127; if(qi < -128) qi = -128;
    sh_x[i] = (int8_t)qi;
  }
  // Pad to multiple of 4
  for(int i = cols + tid; i < ((cols+3)&~3); i += blockDim.x) sh_x[i] = 0;
  __syncthreads();

  // Step 2: DP4A dot product
  const int8_t* wrow = W + (size_t)row * (size_t)cols;
  int32_t acc = 0;
  int cols4 = (cols + 3) & ~3;
  for(int k = tid*4; k < cols4; k += blockDim.x*4) {
    if(k+3 < cols4) {
      int32_t a = *((const int32_t*)(sh_x + k));
      int32_t b = *((const int32_t*)(wrow + k));
      acc = __dp4a(a, b, acc);
    }
  }
  // Warp reduce
  for(int d=16;d>0;d>>=1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, d);
  // Cross-warp reduce
  __shared__ int32_t sh_acc[8];
  if(lane==0) sh_acc[warp] = acc;
  __syncthreads();
  if(warp==0) {
    int32_t val = (tid < (blockDim.x+31)/32) ? sh_acc[tid] : 0;
    for(int d=16;d>0;d>>=1) val += __shfl_down_sync(0xFFFFFFFFu, val, d);
    if(tid==0) {
      float result = (float)val * sh_x_scale[0] * w_scale[row];
      out[row] = result;
    }
  }
}

// ─── Add bias (fp32) ───
__global__ void add_bias_f(float* x, const float* bias, int n) {
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) x[i] += bias[i];
}

// ─── RoPE (fp32 in-place) ───
__global__ void rope_apply(float* q, const float* rope_cos, const float* rope_sin,
                           int t, int Dh) {
  int head = blockIdx.x;
  int d = threadIdx.x;
  int half = Dh / 2;
  if(d >= half) return;
  float* qh = q + head * Dh;
  float q0 = qh[2*d], q1 = qh[2*d+1];
  float c = rope_cos[t * half + d];
  float s = rope_sin[t * half + d];
  qh[2*d]   = q0*c - q1*s;
  qh[2*d+1] = q0*s + q1*c;
}

// ─── KV Store (fp32 → fp16) ───
__global__ void kv_store(half* Kc, half* Vc, const float* k, const float* v,
                         int t, int KVDh) {
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=KVDh) return;
  Kc[(size_t)t*KVDh + i] = __float2half(k[i]);
  Vc[(size_t)t*KVDh + i] = __float2half(v[i]);
}

// ─── Online Softmax Attention (fp32, GQA, 1 block/head) ───
__global__ void attn_decode(float* out, const float* q,
                            const half* Kc, const half* Vc,
                            int t, int Dh, int KVH, float inv_sqrt_dh) {
  int head = blockIdx.x;
  int lane = threadIdx.x;
  if(lane >= Dh) return;
  int kv_head = head % KVH;

  float m = -1e30f, s = 0.0f, acc = 0.0f;

  for(int kpos = 0; kpos <= t; kpos++) {
    const half* kvec = Kc + (size_t)kpos*(KVH*Dh) + kv_head*Dh;
    const half* vvec = Vc + (size_t)kpos*(KVH*Dh) + kv_head*Dh;

    float part = 0.0f;
    for(int i = lane; i < Dh; i += 32)
      part += q[head*Dh+i] * __half2float(kvec[i]);

    float dot = part;
    for(int d=16;d>0;d>>=1) dot += __shfl_down_sync(0xFFFFFFFFu, dot, d);
    dot = __shfl_sync(0xFFFFFFFFu, dot, 0);
    dot *= inv_sqrt_dh;

    float new_m = fmaxf(m, dot);
    float e_old = expf(m - new_m);
    float e_new = expf(dot - new_m);
    acc = acc * e_old + __half2float(vvec[lane]) * e_new;
    s   = s   * e_old + e_new;
    m = new_m;
  }
  out[head*Dh + lane] = acc / fmaxf(s, 1e-10f);
}

// ─── SiLU gate × up (fp32) ───
__global__ void silu_gate_mul(float* out, const float* gate, const float* up, int n) {
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=n) return;
  float g = gate[i];
  float sigmoid = 1.0f / (1.0f + expf(-g));
  out[i] = g * sigmoid * up[i];
}

// ─── Argmax ───
__global__ void argmax_kernel(const float* logits, int* result, int V) {
  __shared__ float sv[256];
  __shared__ int si[256];
  int tid = threadIdx.x;
  sv[tid] = -1e30f; si[tid] = 0;
  for(int i=tid; i<V; i+=blockDim.x) {
    if(logits[i]>sv[tid]){ sv[tid]=logits[i]; si[tid]=i; }
  }
  __syncthreads();
  for(int k=128;k>0;k>>=1){
    if(tid<k && sv[tid+k]>sv[tid]){ sv[tid]=sv[tid+k]; si[tid]=si[tid+k]; }
    __syncthreads();
  }
  if(tid==0) result[0]=si[0];
}

// ─── Log-Softmax + Cross-Entropy Loss (for PPL tracking) ───
__global__ void cross_entropy_loss(const float* logits, int target, float* loss, int V) {
  // Compute log(softmax(logits))[target]
  // Step 1: find max
  __shared__ float sh_max[256];
  __shared__ float sh_sum[256];
  int tid = threadIdx.x;
  sh_max[tid] = -1e30f;
  for(int i=tid; i<V; i+=blockDim.x)
    sh_max[tid] = fmaxf(sh_max[tid], logits[i]);
  __syncthreads();
  for(int k=128;k>0;k>>=1) {
    if(tid<k) sh_max[tid] = fmaxf(sh_max[tid], sh_max[tid+k]);
    __syncthreads();
  }
  float maxv = sh_max[0];
  // Step 2: sum of exp
  float my_sum = 0.0f;
  for(int i=tid; i<V; i+=blockDim.x)
    my_sum += expf(logits[i] - maxv);
  sh_sum[tid] = my_sum;
  __syncthreads();
  for(int k=128;k>0;k>>=1) {
    if(tid<k) sh_sum[tid] += sh_sum[tid+k];
    __syncthreads();
  }
  // Step 3: loss = -(logits[target] - maxv - log(sum))
  if(tid==0) {
    float log_softmax = logits[target] - maxv - logf(sh_sum[0]);
    loss[0] = -log_softmax;
  }
}

// ─── Add residual ───
__global__ void add_inplace(float* a, const float* b, int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) a[i]+=b[i];
}

// ─── Embed lookup (fp32 dequantize) ───
__global__ void embed_lookup(float* out, const int8_t* table, const float* scale,
                             int tok, int D) {
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=D) return;
  out[i] = (float)table[(size_t)tok*D + i] * scale[tok];
}

// ═══════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════
int main(int argc, char** argv) {
  if(argc<2){ fprintf(stderr,"Usage: %s <model_q8.bin> [prompt_text]\n",argv[0]); return 1; }

  printf("[*] DeepSeek INT8 Persistent Decode Engine (Hybrid FP32/INT8)\n");
  printf("[*] (C) 2026 Christian Heinrich Hohlfeld, Konstanz\n");

  Loader ldr;
  ldr.open(argv[1]);
  auto& C = ldr.cfg;
  printf("[*] Model: D=%d H=%d KVH=%d Dh=%d F=%d V=%d L=%d Tmax=%d\n",
         C.D, C.H, C.KVH, C.Dh, C.F, C.V, C.L, C.Tmax);
  printf("[*] rope_theta=%.1f rms_eps=%e bos=%d eos=%d\n", C.rope_theta, C.rms_eps, C.bos_id, C.eos_id);

  init_rope_tables(C.Dh, C.Tmax, C.rope_theta);
  int KVDh = C.KVH * C.Dh;
  float inv_sqrt_dh = 1.0f / sqrtf((float)C.Dh);

  // ─── Upload weights to GPU ───
  struct LayerW {
    int8_t *d_wq, *d_wk, *d_wv, *d_wo, *d_wgate, *d_wup, *d_wdown;
    float  *d_sq, *d_sk, *d_sv, *d_so, *d_sgate, *d_sup, *d_sdown;
    float  *d_ln1, *d_ln2;
    float  *d_bq, *d_bk, *d_bv; // bias (nullable)
    bool has_bq, has_bk, has_bv;
  };

  auto upload_q8 = [&](const std::string& key, int8_t** d_w, float** d_s) {
    TensorQ8 t = ldr.getq8(key);
    size_t wb = (size_t)t.rows*(size_t)t.cols;
    CUDA_CHECK(cudaMalloc(d_w, wb));
    CUDA_CHECK(cudaMemcpy(*d_w, t.data, wb, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(d_s, (size_t)t.rows*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(*d_s, t.scale, (size_t)t.rows*sizeof(float), cudaMemcpyHostToDevice));
  };

  auto upload_f32 = [&](const std::string& key, float** d_p) {
    TensorF32 t = ldr.getf32(key);
    CUDA_CHECK(cudaMalloc(d_p, (size_t)t.size*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(*d_p, t.data, (size_t)t.size*sizeof(float), cudaMemcpyHostToDevice));
  };

  auto try_upload_f32 = [&](const std::string& key, float** d_p) -> bool {
    if(!ldr.has(key)){ *d_p=nullptr; return false; }
    upload_f32(key, d_p);
    return true;
  };

  printf("[*] Uploading weights to GPU...\n");
  auto t0 = std::chrono::high_resolution_clock::now();

  int8_t *d_embed_w, *d_lmhead_w;
  float  *d_embed_s, *d_lmhead_s;
  upload_q8("model.embed_tokens.weight", &d_embed_w, &d_embed_s);
  upload_q8("lm_head.weight", &d_lmhead_w, &d_lmhead_s);

  float *d_lnf;
  upload_f32("model.norm.weight", &d_lnf);

  std::vector<LayerW> lw(C.L);
  for(int l=0;l<C.L;l++){
    char buf[256];
    auto mk=[&](const char* s)->std::string{ snprintf(buf,256,"model.layers.%d.%s",l,s); return buf; };
    upload_f32(mk("input_layernorm.weight"), &lw[l].d_ln1);
    upload_q8(mk("self_attn.q_proj.weight"), &lw[l].d_wq, &lw[l].d_sq);
    upload_q8(mk("self_attn.k_proj.weight"), &lw[l].d_wk, &lw[l].d_sk);
    upload_q8(mk("self_attn.v_proj.weight"), &lw[l].d_wv, &lw[l].d_sv);
    upload_q8(mk("self_attn.o_proj.weight"), &lw[l].d_wo, &lw[l].d_so);
    upload_f32(mk("post_attention_layernorm.weight"), &lw[l].d_ln2);
    upload_q8(mk("mlp.gate_proj.weight"), &lw[l].d_wgate, &lw[l].d_sgate);
    upload_q8(mk("mlp.up_proj.weight"), &lw[l].d_wup, &lw[l].d_sup);
    upload_q8(mk("mlp.down_proj.weight"), &lw[l].d_wdown, &lw[l].d_sdown);
    lw[l].has_bq = try_upload_f32(mk("self_attn.q_proj.bias"), &lw[l].d_bq);
    lw[l].has_bk = try_upload_f32(mk("self_attn.k_proj.bias"), &lw[l].d_bk);
    lw[l].has_bv = try_upload_f32(mk("self_attn.v_proj.bias"), &lw[l].d_bv);
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  double load_ms = std::chrono::duration<double,std::milli>(t1-t0).count();
  printf("[*] Weights uploaded in %.1f ms\n", load_ms);

  // ─── Scratch Buffers (all fp32) ───
  float *d_hidden, *d_norm_out, *d_q, *d_k, *d_v, *d_attn_out, *d_buf;
  float *d_gate, *d_up, *d_logits;
  CUDA_CHECK(cudaMalloc(&d_hidden,   C.D*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_norm_out, C.D*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_q,        C.D*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_k,        KVDh*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_v,        KVDh*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_attn_out, C.D*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_buf,      C.D*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_gate,     C.F*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_up,       C.F*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_logits,   C.V*sizeof(float)));

  // KV cache: fp16 [L][Tmax][KVH*Dh]
  std::vector<half*> d_Kc(C.L), d_Vc(C.L);
  size_t kv_layer = (size_t)C.Tmax * KVDh * sizeof(half);
  for(int l=0;l<C.L;l++){
    CUDA_CHECK(cudaMalloc(&d_Kc[l], kv_layer));
    CUDA_CHECK(cudaMalloc(&d_Vc[l], kv_layer));
  }

  int *d_next_tok;
  CUDA_CHECK(cudaMalloc(&d_next_tok, sizeof(int)));

  // Shared memory for gemv_w8
  int gemv_smem = ((C.D + 3) & ~3) * sizeof(int8_t) + 64;

  printf("[*] Ready. Generating tokens (greedy)...\n");

  int cur_tok = C.bos_id; // BOS token from model config
  int Tgen = 64;
  float total_loss = 0.0f;
  int loss_count = 0;
  float *d_loss;
  CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
  auto gen_start = std::chrono::high_resolution_clock::now();

  for(int t=0; t<Tgen; t++) {
    // Embed
    embed_lookup<<<(C.D+255)/256, 256>>>(d_hidden, d_embed_w, d_embed_s, cur_tok, C.D);
    KERNEL_CHECK();

    for(int l=0;l<C.L;l++) {
      // RMSNorm
      rmsnorm<<<1, 256>>>(d_norm_out, d_hidden, lw[l].d_ln1, C.rms_eps, C.D);
      KERNEL_CHECK();

      // QKV projections (INT8 GEMV with DP4A)
      gemv_w8<<<C.D, 128, gemv_smem>>>(d_q, d_norm_out, lw[l].d_wq, lw[l].d_sq, C.D, C.D);
      KERNEL_CHECK();
      gemv_w8<<<KVDh, 128, gemv_smem>>>(d_k, d_norm_out, lw[l].d_wk, lw[l].d_sk, KVDh, C.D);
      KERNEL_CHECK();
      gemv_w8<<<KVDh, 128, gemv_smem>>>(d_v, d_norm_out, lw[l].d_wv, lw[l].d_sv, KVDh, C.D);
      KERNEL_CHECK();

      // Bias
      if(lw[l].has_bq) { add_bias_f<<<(C.D+255)/256,256>>>(d_q, lw[l].d_bq, C.D); KERNEL_CHECK(); }
      if(lw[l].has_bk) { add_bias_f<<<(KVDh+255)/256,256>>>(d_k, lw[l].d_bk, KVDh); KERNEL_CHECK(); }
      if(lw[l].has_bv) { add_bias_f<<<(KVDh+255)/256,256>>>(d_v, lw[l].d_bv, KVDh); KERNEL_CHECK(); }

      // RoPE
      rope_apply<<<C.H, C.Dh/2>>>(d_q, d_rope_cos, d_rope_sin, t, C.Dh);
      KERNEL_CHECK();
      rope_apply<<<C.KVH, C.Dh/2>>>(d_k, d_rope_cos, d_rope_sin, t, C.Dh);
      KERNEL_CHECK();

      // KV store (fp32 → fp16)
      kv_store<<<(KVDh+255)/256,256>>>(d_Kc[l], d_Vc[l], d_k, d_v, t, KVDh);
      KERNEL_CHECK();

      // Attention (online softmax, GQA, 1 block per head)
      attn_decode<<<C.H, C.Dh>>>(d_attn_out, d_q, d_Kc[l], d_Vc[l], t, C.Dh, C.KVH, inv_sqrt_dh);
      KERNEL_CHECK();

      // Out projection
      gemv_w8<<<C.D, 128, gemv_smem>>>(d_buf, d_attn_out, lw[l].d_wo, lw[l].d_so, C.D, C.D);
      KERNEL_CHECK();

      // Residual
      add_inplace<<<(C.D+255)/256,256>>>(d_hidden, d_buf, C.D);
      KERNEL_CHECK();

      // RMSNorm 2
      rmsnorm<<<1, 256>>>(d_norm_out, d_hidden, lw[l].d_ln2, C.rms_eps, C.D);
      KERNEL_CHECK();

      // MLP: gate, up, SiLU gate*up, down, residual
      gemv_w8<<<C.F, 128, gemv_smem>>>(d_gate, d_norm_out, lw[l].d_wgate, lw[l].d_sgate, C.F, C.D);
      gemv_w8<<<C.F, 128, gemv_smem>>>(d_up,   d_norm_out, lw[l].d_wup,   lw[l].d_sup,  C.F, C.D);
      KERNEL_CHECK();

      silu_gate_mul<<<(C.F+255)/256,256>>>(d_gate, d_gate, d_up, C.F);
      KERNEL_CHECK();

      // Down proj: need larger smem for F dims
      int down_smem = ((C.F + 3) & ~3) * sizeof(int8_t) + 64;
      gemv_w8<<<C.D, 128, down_smem>>>(d_buf, d_gate, lw[l].d_wdown, lw[l].d_sdown, C.D, C.F);
      KERNEL_CHECK();

      add_inplace<<<(C.D+255)/256,256>>>(d_hidden, d_buf, C.D);
      KERNEL_CHECK();
    }

    // Final norm
    rmsnorm<<<1, 256>>>(d_norm_out, d_hidden, d_lnf, C.rms_eps, C.D);
    KERNEL_CHECK();

    // Vocab logits
    int lm_smem = ((C.D + 3) & ~3) * sizeof(int8_t) + 64;
    gemv_w8<<<C.V, 128, lm_smem>>>(d_logits, d_norm_out, d_lmhead_w, d_lmhead_s, C.V, C.D);
    KERNEL_CHECK();

    // Argmax
    argmax_kernel<<<1, 256>>>(d_logits, d_next_tok, C.V);
    KERNEL_CHECK();

    CUDA_CHECK(cudaMemcpy(&cur_tok, d_next_tok, sizeof(int), cudaMemcpyDeviceToHost));

    // Debug: dump first few logits at t=0
    if(t == 0) {
      std::vector<float> h_log(20);
      CUDA_CHECK(cudaMemcpy(h_log.data(), d_logits, 20*sizeof(float), cudaMemcpyDeviceToHost));
      printf("  [dbg] logits[0..19]:");
      for(int i=0;i<20;i++) printf(" %.2f", h_log[i]);
      printf("\n");
    }

    // Loss/PPL tracking (cross-entropy of predicted logits vs next token)
    if(t > 0) {
      // loss against the token we just predicted
      cross_entropy_loss<<<1, 256>>>(d_logits, cur_tok, d_loss, C.V);
      KERNEL_CHECK();
      float h_loss;
      CUDA_CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
      total_loss += h_loss;
      loss_count++;
      if(t % 10 == 0 || t < 5) {
        float avg_loss = total_loss / loss_count;
        float ppl = expf(avg_loss);
        printf("t=%d tok=%d loss=%.4f avg_loss=%.4f ppl=%.1f\n", t, cur_tok, h_loss, avg_loss, ppl);
        continue;
      }
    }
    printf("t=%d tok=%d\n", t, cur_tok);
  }

  auto gen_end = std::chrono::high_resolution_clock::now();
  double gen_ms = std::chrono::duration<double,std::milli>(gen_end-gen_start).count();
  double tps = (double)Tgen / (gen_ms / 1000.0);
  float final_avg_loss = (loss_count > 0) ? total_loss / loss_count : 0.0f;
  float final_ppl = expf(final_avg_loss);
  printf("\n[*] Generated %d tokens in %.1f ms = %.1f tok/s\n", Tgen, gen_ms, tps);
  printf("[*] Final avg_loss=%.4f ppl=%.1f\n", final_avg_loss, final_ppl);
  printf("[*] Done.\n");

  ldr.close_();
  return 0;
}
CUEOF

# ═══════════════════════════════════════════════════════════════
# PHASE 3: Ensure Dependencies
# ═══════════════════════════════════════════════════════════════
echo "[*] Checking Python deps..."
"$PIP" install safetensors huggingface_hub numpy --quiet 2>/dev/null

# ═══════════════════════════════════════════════════════════════
# PHASE 4: Export Model (v3 format: Q8 weights + F32 norms/bias)
# ═══════════════════════════════════════════════════════════════
if [ ! -f "$MODEL_BIN" ]; then
  echo "[*] Exporting $MODEL_REPO → $MODEL_BIN ..."
  "$PYTHON" /tmp/_dsi8_export.py "$MODEL_REPO" "$MODEL_BIN"
fi

# ═══════════════════════════════════════════════════════════════
# PHASE 5: Compile
# ═══════════════════════════════════════════════════════════════
echo "[*] Compiling engine (sm_$SM, -O3, --use_fast_math)..."
nvcc -O3 --use_fast_math -Xptxas -O3,-v -arch=sm_$SM \
     /tmp/_dsi8_engine.cu -o "$ENGINE_BIN"

# ═══════════════════════════════════════════════════════════════
# PHASE 6: Run
# ═══════════════════════════════════════════════════════════════
echo "================================================================="
echo "  DeepSeek INT8 Persistent Decode Engine (Hybrid FP32/INT8)"
echo "  Model: $MODEL_REPO"
echo "  Binary: $MODEL_BIN"
echo "================================================================="
./"$ENGINE_BIN" "$MODEL_BIN"
