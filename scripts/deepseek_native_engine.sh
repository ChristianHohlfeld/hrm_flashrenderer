#!/bin/bash
# Â© 2026 Christian Heinrich Hohlfeld (Konstanz, Germany), christianhohlfeld.com, ORCID 0009-0003-6634-9045.
# DeepSeek INT8 Persistent Decode Engine â€” Single File, No Compromise, Prod Class
# Hardware Target: RTX 2080 Ti (sm_75), 11 GB VRAM
# Model Scope: DeepSeek dense distill variants (Qwen/Llama-style safetensors layout)
#
# Architecture:
#   - INT8 weights with per-row fp32 scale (W8A16 quantization)
#   - FP32 hidden states & activations (full precision intermediate)
#   - FP16 KV cache (compact, no precision loss)
#   - Online softmax with fused 1/sqrt(Dh) scaling
#   - RoPE (Q15 precomputed sin/cos tables in device memory)
#   - DP4A dot products for weight GEMVs (bandwidth-bound â†’ int8 wins)
#   - Zero-copy mmap weight loading
#   - Pure numpy export (NO TORCH)

set -euo pipefail

WORKDIR="${WORKDIR:-$PWD}"
PYTHON="${PYTHON:-python3}"
MODEL_REPO="${MODEL_REPO:-deepseek-ai/DeepSeek-R1-Distill-Qwen-32B}"
MODEL_BIN="${MODEL_BIN:-model_q8.bin}"
ENGINE_BIN="${ENGINE_BIN:-$WORKDIR/.run/bin/deepseek_engine}"
SM="${SM:-75}"  # Backward-compatible single-arch override.
CUDA_ARCH_LIST="${CUDA_ARCH_LIST:-75,86}"  # Comma-separated list, e.g. "75,86"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
SKIP_RUN="${SKIP_RUN:-0}"

cd "$WORKDIR"

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PHASE 1: Inline Python Exporter (BF16â†’INT8, all layers, NO TORCH)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
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
    """Per-row INT8 quantization with fp32 scale: t â‰ˆ q * scale"""
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

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PHASE 2: Inline CUDA Engine â€” FP32 hidden + INT8 weight GEMVs
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
cat << 'CUEOF' > /tmp/_dsi8_engine.cu
// Â© 2026 Christian Heinrich Hohlfeld (Konstanz, Germany), christianhohlfeld.com
// DeepSeek INT8 Persistent Decode Engine â€” Hybrid FP32/INT8
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <thread>
#include <chrono>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <nvml.h>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);}} while(0)
#define KERNEL_CHECK() { cudaError_t e=cudaGetLastError(); if(e!=cudaSuccess){ \
  fprintf(stderr,"Kernel %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);}}
#define NVML_CHECK(x) do { nvmlReturn_t r=(x); if(r!=NVML_SUCCESS){ \
  fprintf(stderr,"NVML Error: %s\n",nvmlErrorString(r)); exit(1);}} while(0)

// â”€â”€â”€ Config â”€â”€â”€
struct ModelConfig {
  int D, H, KVH, Dh, F, V, L, Tmax;
  double rope_theta;
  float rms_eps;
  int bos_id, eos_id;
};

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

struct TelemetryCtx {
    int sock;
    struct sockaddr_in addr;
    float* d_proj_matrix;
    float* d_proj_result;
    nvmlDevice_t nvml_handles[3];
    bool enabled;
};

// â”€â”€â”€ Tensor types â”€â”€â”€
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

// â”€â”€â”€ Mmap Loader â”€â”€â”€
struct Loader {
  uint8_t* base; size_t sz; int fd;
  ModelConfig cfg;
  std::unordered_map<std::string, TensorQ8> q8;
  std::unordered_map<std::string, TensorF32> f32;

  void open(const char* path) {
    fd = ::open(path, O_RDONLY);
    if(fd<0){ perror("open"); exit(1); }
    struct stat sb; fstat(fd,&sb); sz=sb.st_size;
    base=(uint8_t*)mmap(nullptr,sz,PROT_READ,MAP_PRIVATE,fd,0);
    if(base==MAP_FAILED){ perror("mmap"); exit(1); }
    printf("[*] Mapped %zu bytes. Parsing...\n", sz); fflush(stdout);
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

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// CUDA KERNELS â€” FP32 activations, INT8 weight GEMVs
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

// â”€â”€â”€ RoPE tables (per-GPU) â”€â”€â”€
static float* d_rope_sin[3] = {nullptr, nullptr, nullptr};
static float* d_rope_cos[3] = {nullptr, nullptr, nullptr};

static void init_rope_tables(int Dh, int Tmax, double theta, int n_gpus) {
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
  for(int i=0; i<n_gpus; i++) {
    cudaSetDevice(i);
    CUDA_CHECK(cudaMalloc(&d_rope_sin[i], bytes));
    CUDA_CHECK(cudaMalloc(&d_rope_cos[i], bytes));
    CUDA_CHECK(cudaMemcpy(d_rope_sin[i], s.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rope_cos[i], c.data(), bytes, cudaMemcpyHostToDevice));
  }
}

// â”€â”€â”€ RMSNorm then INT8 Quantize (Fused) â”€â”€â”€
// out_q: [D] int8, out_scale: [1] float
__global__ void rmsnorm_quant(int8_t* out_q, float* out_scale, float* out_f32, 
                             const float* x, const float* gamma, float eps, int D) {
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
  float inv_rms = rsqrtf(shared_ss[0] / (float)D + eps);
  
  // Now find absmax for quantization
  float my_max = 0.0f;
  for(int i = tid; i < D; i += blockDim.x) {
      float v = fabsf(x[i] * inv_rms * gamma[i]);
      if(out_f32) out_f32[i] = x[i] * inv_rms * gamma[i];
      if(v > my_max) my_max = v;
  }
  for(int d=16;d>0;d>>=1) my_max = fmaxf(my_max, __shfl_down_sync(0xFFFFFFFFu, my_max, d));
  if(lane==0) shared_ss[warp] = my_max;
  __syncthreads();
  if(warp==0) {
      float val = (tid < (blockDim.x+31)/32) ? shared_ss[tid] : 0.0f;
      for(int d=16;d>0;d>>=1) val = fmaxf(val, __shfl_down_sync(0xFFFFFFFFu, val, d));
      if(tid==0) {
          float scale = val / 127.0f;
          if(scale < 1e-10f) scale = 1e-10f;
          out_scale[0] = scale;
      }
  }
  __syncthreads();
  
  float inv_q_scale = 1.0f / out_scale[0];
  for(int i = tid; i < D; i += blockDim.x) {
      float v = (x[i] * inv_rms * gamma[i]) * inv_q_scale;
      int32_t qi = __float2int_rn(v);
      if(qi > 127) qi = 127; if(qi < -128) qi = -128;
      out_q[i] = (int8_t)qi;
  }
}

// â”€â”€â”€ Optimized GEMV using pre-quantized x â”€â”€â”€
__global__ void gemv_w8_prequant(float* out, const int8_t* x_q, float x_scale,
                                const int8_t* W, const float* w_scale, int rows, int cols) {
  int row = blockIdx.x;
  if(row >= rows) return;
  int tid = threadIdx.x;
  
  const int8_t* wrow = W + (size_t)row * (size_t)cols;
  int32_t acc = 0;
  // Assumes cols is multiple of 4
  for(int k = tid*4; k < cols; k += blockDim.x*4) {
      int32_t a = *((const int32_t*)(x_q + k));
      int32_t b = *((const int32_t*)(wrow + k));
      acc = __dp4a(a, b, acc);
  }
  
  int lane = tid % 32;
  for(int d=16;d>0;d>>=1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, d);
  
  __shared__ int32_t sh_acc[8];
  int warp = tid / 32;
  if(lane==0) sh_acc[warp] = acc;
  __syncthreads();
  
  if(warp==0) {
    int32_t val = (tid < (blockDim.x+31)/32) ? sh_acc[tid] : 0;
    for(int d=16;d>0;d>>=1) val += __shfl_down_sync(0xFFFFFFFFu, val, d);
    if(tid==0) {
      out[row] = (float)val * x_scale * w_scale[row];
    }
  }
}

// â”€â”€â”€ Standard INT8 Weight GEMV (quantizes on-the-fly) â”€â”€â”€
__global__ void gemv_w8(float* out, const float* x, const int8_t* W,
                        const float* w_scale, int rows, int cols) {
  int row = blockIdx.x;
  if(row >= rows) return;
  int tid = threadIdx.x;
  extern __shared__ int8_t sh_x[];
  __shared__ float sh_x_scale[1];

  float my_max = 0.0f;
  for(int i = tid; i < cols; i += blockDim.x) {
    float v = fabsf(x[i]);
    if(v > my_max) my_max = v;
  }
  for(int d=16;d>0;d>>=1) my_max = fmaxf(my_max, __shfl_down_sync(0xFFFFFFFFu, my_max, d));
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
  __syncthreads();

  const int8_t* wrow = W + (size_t)row * (size_t)cols;
  int32_t acc = 0;
  for(int k = tid*4; k < cols; k += blockDim.x*4) {
    int32_t a = *((const int32_t*)(sh_x + k));
    int32_t b = *((const int32_t*)(wrow + k));
    acc = __dp4a(a, b, acc);
  }
  for(int d=16;d>0;d>>=1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, d);
  __shared__ int32_t sh_acc[8];
  if(lane==0) sh_acc[warp] = acc;
  __syncthreads();
  if(warp==0) {
    int32_t val = (tid < (blockDim.x+31)/32) ? sh_acc[tid] : 0;
    for(int d=16;d>0;d>>=1) val += __shfl_down_sync(0xFFFFFFFFu, val, d);
    if(tid==0) out[row] = (float)val * sh_x_scale[0] * w_scale[row];
  }
}

// â”€â”€â”€ Add bias (fp32) â”€â”€â”€
__global__ void add_bias_f(float* x, const float* bias, int n) {
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) x[i] += bias[i];
}

// â”€â”€â”€ RoPE (fp32 in-place) â”€â”€â”€
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

// â”€â”€â”€ KV Store (fp32 â†’ fp16) â”€â”€â”€
__global__ void kv_store(half* Kc, half* Vc, const float* k, const float* v,
                         int t, int KVDh) {
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=KVDh) return;
  Kc[(size_t)t*KVDh + i] = __float2half(k[i]);
  Vc[(size_t)t*KVDh + i] = __float2half(v[i]);
}

// â”€â”€â”€ Online Softmax Attention (Corrected Decode) â”€â”€â”€
__global__ void attn_decode(float* out, const float* q,
                             const half* Kc, const half* Vc,
                             int t, int Dh, int KVH, float inv_sqrt_dh) {
  int head = blockIdx.x;
  int tid = threadIdx.x;
  if(tid >= Dh) return;
  int kv_head = head % KVH;

  float m = -1e30f, s = 0.0f, acc = 0.0f;
  const float query_val = q[head * Dh + tid];

  for(int kpos = 0; kpos <= t; kpos++) {
    const half* k_base = Kc + (size_t)kpos * (KVH * Dh) + kv_head * Dh;
    const half* v_base = Vc + (size_t)kpos * (KVH * Dh) + kv_head * Dh;

    float dot = query_val * __half2float(k_base[tid]);
    // Full block reduction (intra-warp then inter-warp)
    for(int d=16; d>0; d>>=1) dot += __shfl_down_sync(0xFFFFFFFFu, dot, d);
    
    __shared__ float sh_dot[32]; 
    int lane = tid % 32;
    int warp = tid / 32;
    if(lane == 0) sh_dot[warp] = dot;
    __syncthreads();
    
    if(warp == 0) {
      float v = (tid < (blockDim.x+31)/32) ? sh_dot[tid] : 0.0f;
      for(int d=16; d>0; d>>=1) v += __shfl_down_sync(0xFFFFFFFFu, v, d);
      if(tid == 0) sh_dot[0] = v * inv_sqrt_dh;
    }
    __syncthreads();
    float final_dot = sh_dot[0];

    float new_m = fmaxf(m, final_dot);
    float e_old = expf(m - new_m);
    float e_new = expf(final_dot - new_m);
    
    acc = acc * e_old + __half2float(v_base[tid]) * e_new;
    s   = s   * e_old + e_new;
    m = new_m;
  }
  out[head * Dh + tid] = acc / fmaxf(s, 1e-10f);
}

// â”€â”€â”€ SiLU gate Ã— up (fp32) â”€â”€â”€
__global__ void silu_gate_mul(float* out, const float* gate, const float* up, int n) {
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=n) return;
  float g = gate[i];
  float sigmoid = 1.0f / (1.0f + expf(-g));
  out[i] = g * sigmoid * up[i];
}

// â”€â”€â”€ Argmax â”€â”€â”€
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

// â”€â”€â”€ Log-Softmax + Cross-Entropy Loss (for PPL tracking) â”€â”€â”€
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

// â”€â”€â”€ Add residual â”€â”€â”€
__global__ void add_inplace(float* a, const float* b, int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) a[i]+=b[i];
}

// â”€â”€â”€ Embed lookup (fp32 dequantize) â”€â”€â”€
__global__ void embed_lookup(float* out, const int8_t* table, const float* scale,
                             int tok, int D) {
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=D) return;
  out[i] = (float)table[(size_t)tok*D + i] * scale[tok];
}

__global__ void random_project_3d_kernel(const float* vec, float* proj, const float* matrix, int D) {
    int dim = blockIdx.x;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        sum += vec[i] * matrix[dim * D + i];
    }
    atomicAdd(&proj[dim], sum);
}

static char g_model_name[64] = "Unknown Model";

static void init_telemetry(TelemetryCtx* ctx, int D, const char* name) {
    if (name) strncpy(g_model_name, name, 63);
    const char* enable = std::getenv("TELEM_ENABLE");
    if (!enable || std::string(enable) != "1") {
        ctx->enabled = false;
        return;
    }
    const char* port_s = std::getenv("TELEM_PORT");
    int port = port_s ? std::atoi(port_s) : 9999;
    const char* target = std::getenv("TELEM_TARGET");
    if (!target) target = "127.0.0.1";

    ctx->sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (ctx->sock < 0) { ctx->enabled = false; return; }
    
    ctx->addr.sin_family = AF_INET;
    ctx->addr.sin_port = htons(port);
    ctx->addr.sin_addr.s_addr = inet_addr(target);
    
    ctx->enabled = true;

    NVML_CHECK(nvmlInit());
    for(int i=0; i<3; i++) {
        NVML_CHECK(nvmlDeviceGetHandleByIndex(i, &ctx->nvml_handles[i]));
    }

    CUDA_CHECK(cudaMalloc(&ctx->d_proj_matrix, 3 * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_proj_result, 3 * sizeof(float)));

    std::vector<float> h_matrix(3 * D);
    for (int i=0; i < 3*D; i++) h_matrix[i] = ((float)rand()/RAND_MAX) * 2.0f - 1.0f;
    CUDA_CHECK(cudaMemcpy(ctx->d_proj_matrix, h_matrix.data(), h_matrix.size() * sizeof(float), cudaMemcpyHostToDevice));
    
    std::printf("[*] Telemetry enabled (with NVML) on %s:%d\n", target, port);
}

static void send_telemetry(TelemetryCtx* ctx, float loss, int step, float* d_vec, int D, const char* prompt_text) {
    if (!ctx->enabled) return;
    
    float h_proj[3] = {0,0,0};
    if (d_vec) {
        CUDA_CHECK(cudaMemset(ctx->d_proj_result, 0, 3 * sizeof(float)));
        random_project_3d_kernel<<<3, 256>>>(d_vec, ctx->d_proj_result, ctx->d_proj_matrix, D);
        CUDA_CHECK(cudaMemcpy(h_proj, ctx->d_proj_result, 3 * sizeof(float), cudaMemcpyDeviceToHost));
    }

    static float last_x = 0, last_y = 0, last_z = 0;
    float dx = h_proj[0] - last_x, dy = h_proj[1] - last_y, dz = h_proj[2] - last_z;
    float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
    float dot = h_proj[0]*last_x + h_proj[1]*last_y + h_proj[2]*last_z;
    float mag1 = std::sqrt(h_proj[0]*h_proj[0] + h_proj[1]*h_proj[1] + h_proj[2]*h_proj[2]);
    float mag2 = std::sqrt(last_x*last_x + last_y*last_y + last_z*last_z);
    float cos_sim = (mag1 > 1e-9 && mag2 > 1e-9) ? (dot / (mag1 * mag2)) : 1.0f;
    last_x = h_proj[0]; last_y = h_proj[1]; last_z = h_proj[2];

    TelemetryPacket p;
    memset(&p, 0, sizeof(p));
    p.step = (uint32_t)step; p.loss = loss; p.cos_sim = cos_sim; p.euclid_dist = dist;
    p.proj_x = h_proj[0]; p.proj_y = h_proj[1]; p.proj_z = h_proj[2];
    
    for(int i=0; i<3; i++) {
        nvmlUtilization_t util;
        nvmlMemory_t mem;
        if(nvmlDeviceGetUtilizationRates(ctx->nvml_handles[i], &util) == NVML_SUCCESS) p.gpu_util[i] = (float)util.gpu;
        if(nvmlDeviceGetMemoryInfo(ctx->nvml_handles[i], &mem) == NVML_SUCCESS) p.gpu_mem[i] = (float)mem.used / (float)mem.total * 100.0f;
    }

    if (prompt_text) strncpy(p.prompt, prompt_text, 63);
    else strncpy(p.prompt, "SYSTEM_IDLE", 63);
    strncpy(p.model_id, g_model_name, 63);
    sendto(ctx->sock, &p, sizeof(p), 0, (struct sockaddr*)&ctx->addr, sizeof(ctx->addr));
}

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// MULTI-GPU & INTERACTIVE INFRASTRUCTURE
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

struct GPUBuffer {
    float *d_hidden;
    float *d_norm_out;
    int8_t *d_x_q;      // Pre-quantized x for GEMVs
    float *d_x_scale;   // Scale for x_q
    float *d_q, *d_k, *d_v;
    float *d_attn_out;
    float *d_buf;
    float *d_gate, *d_up;
    half *d_Kc_ptr, *d_Vc_ptr; // KV cache for this GPU's layers
};

static void check_prompt_file(char* buffer, size_t size) {
    const char* prompt_path = std::getenv("DSI8_PROMPT_FILE");
    if (!prompt_path || !prompt_path[0]) prompt_path = "/tmp/deepseek_prompt.txt";
    FILE* f = fopen(prompt_path, "r");
    if (f) {
        if (fgets(buffer, size, f)) {
            // strip newline
            char* nl = strchr(buffer, '\n');
            if (nl) *nl = 0;
        }
        fclose(f);
        remove(prompt_path);
    } else {
        buffer[0] = 0;
    }
}

static const char* dsi8_tokens_path() {
    const char* p = std::getenv("DSI8_TOKENS_FILE");
    return (p && p[0]) ? p : "/tmp/deepseek_tokens.txt";
}

static const char* dsi8_prompt_tokens_path() {
    const char* p = std::getenv("DSI8_PROMPT_TOKENS_FILE");
    return (p && p[0]) ? p : "/tmp/deepseek_prompt_tokens.txt";
}

static const char* dsi8_done_path() {
    const char* p = std::getenv("DSI8_DONE_FILE");
    return (p && p[0]) ? p : "/tmp/deepseek_done.flag";
}

static int dsi8_tgen_default() {
    const char* s = std::getenv("DSI8_TGEN");
    if (!s || !s[0]) return 64;
    int v = std::atoi(s);
    if (v < 1) v = 1;
    if (v > 2048) v = 2048;
    return v;
}

static void clear_result_files() {
    remove(dsi8_tokens_path());
    remove(dsi8_done_path());
}

static void load_prompt_tokens(std::vector<int>& out) {
    out.clear();
    const char* path = dsi8_prompt_tokens_path();
    FILE* f = fopen(path, "r");
    if (!f) return;
    char line[64];
    while (fgets(line, sizeof(line), f)) {
        int tok = std::atoi(line);
        if (tok >= 0) out.push_back(tok);
    }
    fclose(f);
    remove(path);
}

static void append_token_id(int tok) {
    FILE* f = fopen(dsi8_tokens_path(), "a");
    if (!f) return;
    fprintf(f, "%d\n", tok);
    fclose(f);
}

static void mark_done() {
    FILE* f = fopen(dsi8_done_path(), "w");
    if (!f) return;
    fprintf(f, "done\n");
    fclose(f);
}

static inline int layer_owner(int layer_idx, int n_layers, int n_gpus) {
    if (n_gpus <= 1) return 0;
    if (n_gpus > 3) n_gpus = 3;
    if (n_layers <= 0) return 0;
    if (layer_idx < 0) layer_idx = 0;
    if (layer_idx >= n_layers) layer_idx = n_layers - 1;

    int base = n_layers / n_gpus;
    int rem = n_layers % n_gpus;
    int start = 0;
    for (int g = 0; g < n_gpus; g++) {
        int cnt = base + ((g < rem) ? 1 : 0);
        int end = start + cnt;
        if (layer_idx < end) return g;
        start = end;
    }
    return n_gpus - 1;
}
int main(int argc, char** argv) {
  if(argc<2){ fprintf(stderr,"Usage: %s <model_q8.bin> [prompt_text]\n",argv[0]); return 1; }

  printf("[*] DeepSeek INT8 Persistent Decode Engine (Hybrid Multi-GPU)\n");
  printf("[*] (C) 2026 Christian Heinrich Hohlfeld, Konstanz\n");

  Loader ldr;
  ldr.open(argv[1]);
  auto& C = ldr.cfg;
  
  TelemetryCtx telem;
  init_telemetry(&telem, C.D, argc > 2 ? argv[2] : "DeepSeek-Distill-Q8");
  printf("[*] Model: D=%d H=%d KVH=%d Dh=%d F=%d V=%d L=%d Tmax=%d\n",
         C.D, C.H, C.KVH, C.Dh, C.F, C.V, C.L, C.Tmax);
  fflush(stdout);

  int KVDh = C.KVH * C.Dh;
  float inv_sqrt_dh = 1.0f / sqrtf((float)C.Dh);

  int n_gpus = 0;
  cudaGetDeviceCount(&n_gpus);
  if (n_gpus < 1) {
    fprintf(stderr, "ERR: no CUDA devices visible. Check CUDA_VISIBLE_DEVICES and NVIDIA driver.\n");
    return 1;
  }
  if (n_gpus > 3) n_gpus = 3;
  printf("[*] Using %d GPUs for Layer Splitting\n", n_gpus);
  fflush(stdout);

  std::vector<int> layer_counts((size_t)n_gpus, 0);
  for (int l = 0; l < C.L; l++) layer_counts[(size_t)layer_owner(l, C.L, n_gpus)]++;
  for (int g = 0; g < n_gpus; g++) {
    printf("[*] Layer partition GPU %d: %d layers\n", g, layer_counts[(size_t)g]);
  }
  fflush(stdout);

  init_rope_tables(C.Dh, C.Tmax, C.rope_theta, n_gpus);

  for(int i=0; i<n_gpus; i++) {
    for(int j=0; j<n_gpus; j++) {
      if(i != j) {
        cudaSetDevice(i);
        int can_access = 0;
        cudaDeviceCanAccessPeer(&can_access, i, j);
        if(can_access) {
          cudaDeviceEnablePeerAccess(j, 0);
          printf("[*] P2P: GPU %d -> %d enabled\n", i, j); fflush(stdout);
        }
      }
    }
  }

  struct LayerW {
    int8_t *d_wq, *d_wk, *d_wv, *d_wo, *d_wgate, *d_wup, *d_wdown;
    float  *d_sq, *d_sk, *d_sv, *d_so, *d_sgate, *d_sup, *d_sdown;
    float  *d_ln1, *d_ln2;
    float  *d_bq, *d_bk, *d_bv;
    bool has_bq, has_bk, has_bv;
    half *d_Kc, *d_Vc;
  };

  std::vector<LayerW> lw(C.L);
  GPUBuffer gb[3];

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

  printf("[*] Allocating buffers and uploading weights...\n");
  for(int i=0; i<n_gpus; i++) {
    cudaSetDevice(i);
    CUDA_CHECK(cudaMalloc(&gb[i].d_hidden, C.D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gb[i].d_norm_out, C.D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gb[i].d_x_q, C.D * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&gb[i].d_x_scale, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gb[i].d_q, C.D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gb[i].d_k, KVDh * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gb[i].d_v, KVDh * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gb[i].d_attn_out, C.D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gb[i].d_buf, C.D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gb[i].d_gate, C.F * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gb[i].d_up,   C.F * sizeof(float)));
  }

  size_t kv_layer = (size_t)C.Tmax * KVDh * sizeof(half);
  for(int l=0; l<C.L; l++) {
    int target_gpu = layer_owner(l, C.L, n_gpus);
    cudaSetDevice(target_gpu);

    char pfix[128]; snprintf(pfix, 128, "model.layers.%d.", l);
    upload_f32(std::string(pfix)+"input_layernorm.weight", &lw[l].d_ln1);
    upload_q8(std::string(pfix)+"self_attn.q_proj.weight", &lw[l].d_wq, &lw[l].d_sq);
    upload_q8(std::string(pfix)+"self_attn.k_proj.weight", &lw[l].d_wk, &lw[l].d_sk);
    upload_q8(std::string(pfix)+"self_attn.v_proj.weight", &lw[l].d_wv, &lw[l].d_sv);
    upload_q8(std::string(pfix)+"self_attn.o_proj.weight", &lw[l].d_wo, &lw[l].d_so);
    upload_f32(std::string(pfix)+"post_attention_layernorm.weight", &lw[l].d_ln2);
    upload_q8(std::string(pfix)+"mlp.gate_proj.weight",   &lw[l].d_wgate,&lw[l].d_sgate);
    upload_q8(std::string(pfix)+"mlp.up_proj.weight",     &lw[l].d_wup,  &lw[l].d_sup);
    upload_q8(std::string(pfix)+"mlp.down_proj.weight",   &lw[l].d_wdown,&lw[l].d_sdown);
    lw[l].has_bq = try_upload_f32(std::string(pfix)+"self_attn.q_proj.bias", &lw[l].d_bq);
    lw[l].has_bk = try_upload_f32(std::string(pfix)+"self_attn.k_proj.bias", &lw[l].d_bk);
    lw[l].has_bv = try_upload_f32(std::string(pfix)+"self_attn.v_proj.bias", &lw[l].d_bv);
    
    CUDA_CHECK(cudaMalloc(&lw[l].d_Kc, kv_layer));
    CUDA_CHECK(cudaMalloc(&lw[l].d_Vc, kv_layer));
  }

  cudaSetDevice(0);
  int8_t *d_embed_w, *d_lmhead_w;
  float *d_embed_s, *d_lnf, *d_lmhead_s, *d_logits, *d_loss;
  int *d_next_tok;
  upload_q8("model.embed_tokens.weight", &d_embed_w, &d_embed_s);
  upload_f32("model.norm.weight", &d_lnf);
  upload_q8("lm_head.weight", &d_lmhead_w, &d_lmhead_s);
  CUDA_CHECK(cudaMalloc(&d_logits, (size_t)C.V*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_next_tok, sizeof(int)));

  int gemv_smem = ((C.D + 3) & ~3) * sizeof(int8_t) + 64;
  int lm_smem = ((C.D + 3) & ~3) * sizeof(int8_t) + 64;
  int down_smem = ((C.F + 3) & ~3) * sizeof(int8_t) + 64;

  printf("[*] Ready. Entering interactive loop...\n"); fflush(stdout);
  char prompt_buf[512];
  while(true) {
    check_prompt_file(prompt_buf, sizeof(prompt_buf));
    if (prompt_buf[0] == 0) { 
        send_telemetry(&telem, 0.0f, 0, nullptr, C.D, "AWAITING_PROMPT");
        std::this_thread::sleep_for(std::chrono::milliseconds(200)); 
        continue; 
    }
    printf("> Received Prompt: %s\n", prompt_buf); fflush(stdout);

    std::vector<int> prompt_tokens;
    load_prompt_tokens(prompt_tokens);
    if (prompt_tokens.empty()) prompt_tokens.push_back(C.bos_id);

    int prompt_len = (int)prompt_tokens.size();
    int Tgen = dsi8_tgen_default();
    int total_steps = prompt_len + Tgen;
    if (total_steps > C.Tmax) total_steps = C.Tmax;
    int generated = 0;
    int next_tok = prompt_tokens[0];

    clear_result_files();
    float total_loss = 0.0f;
    int loss_count = 0;
    
    for(int t=0; t<total_steps; t++) {
      int cur_tok = (t < prompt_len) ? prompt_tokens[(size_t)t] : next_tok;
      cudaSetDevice(0);
      embed_lookup<<<(C.D+255)/256, 256>>>(gb[0].d_hidden, d_embed_w, d_embed_s, cur_tok, C.D);

      for(int l=0; l<C.L; l++) {
        int target_gpu = layer_owner(l, C.L, n_gpus);
        int prev_gpu = (l > 0) ? layer_owner(l - 1, C.L, n_gpus) : 0;

        if (target_gpu != prev_gpu) {
            CUDA_CHECK(cudaMemcpy(gb[target_gpu].d_hidden, gb[prev_gpu].d_hidden, C.D * sizeof(float), cudaMemcpyDeviceToDevice));
        }
        
        cudaSetDevice(target_gpu);
        float *h = gb[target_gpu].d_hidden, *n = gb[target_gpu].d_norm_out;
        int8_t *xq = gb[target_gpu].d_x_q; float *xs = gb[target_gpu].d_x_scale;
        float *q = gb[target_gpu].d_q,      *k = gb[target_gpu].d_k, *v = gb[target_gpu].d_v;
        float *ao = gb[target_gpu].d_attn_out, *b = gb[target_gpu].d_buf;
        float *gt = gb[target_gpu].d_gate, *up = gb[target_gpu].d_up;

        // Fused RMS + Quantize
        rmsnorm_quant<<<1, 256>>>(xq, xs, n, h, lw[l].d_ln1, C.rms_eps, C.D);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Use local scale for speed
        float h_xs; CUDA_CHECK(cudaMemcpy(&h_xs, xs, sizeof(float), cudaMemcpyDeviceToHost));

        gemv_w8_prequant<<<C.D, 128>>>(q, xq, h_xs, lw[l].d_wq, lw[l].d_sq, C.D, C.D);
        gemv_w8_prequant<<<KVDh, 128>>>(k, xq, h_xs, lw[l].d_wk, lw[l].d_sk, KVDh, C.D);
        gemv_w8_prequant<<<KVDh, 128>>>(v, xq, h_xs, lw[l].d_wv, lw[l].d_sv, KVDh, C.D);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        if(lw[l].has_bq) add_bias_f<<<(C.D+255)/256,256>>>(q, lw[l].d_bq, C.D);
        if(lw[l].has_bk) add_bias_f<<<(KVDh+255)/256,256>>>(k, lw[l].d_bk, KVDh);
        if(lw[l].has_bv) add_bias_f<<<(KVDh+255)/256,256>>>(v, lw[l].d_bv, KVDh);
        
        rope_apply<<<C.H, C.Dh/2>>>(q, d_rope_cos[target_gpu], d_rope_sin[target_gpu], t, C.Dh);
        rope_apply<<<C.KVH, C.Dh/2>>>(k, d_rope_cos[target_gpu], d_rope_sin[target_gpu], t, C.Dh);
        kv_store<<<(KVDh+255)/256,256>>>(lw[l].d_Kc, lw[l].d_Vc, k, v, t, KVDh);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        attn_decode<<<C.H, C.Dh>>>(ao, q, lw[l].d_Kc, lw[l].d_Vc, t, C.Dh, C.KVH, inv_sqrt_dh);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // O Proj still needs a quant step or we keep it as is (O uses ao as input)
        gemv_w8<<<C.D, 128, gemv_smem>>>(b, ao, lw[l].d_wo, lw[l].d_so, C.D, C.D);
        add_inplace<<<(C.D+255)/256,256>>>(h, b, C.D);
        
        // MLP Part: Fused RMS + Quant
        rmsnorm_quant<<<1, 256>>>(xq, xs, n, h, lw[l].d_ln2, C.rms_eps, C.D);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_xs, xs, sizeof(float), cudaMemcpyDeviceToHost));
        
        gemv_w8_prequant<<<C.F, 128>>>(gt, xq, h_xs, lw[l].d_wgate, lw[l].d_sgate, C.F, C.D);
        gemv_w8_prequant<<<C.F, 128>>>(up, xq, h_xs, lw[l].d_wup,   lw[l].d_sup,  C.F, C.D);
        
        silu_gate_mul<<<(C.F+255)/256,256>>>(gt, gt, up, C.F);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Down projection
        gemv_w8<<<C.D, 128, down_smem>>>(b, gt, lw[l].d_wdown, lw[l].d_sdown, C.D, C.F);
        add_inplace<<<(C.D+255)/256,256>>>(h, b, C.D);
        CUDA_CHECK(cudaDeviceSynchronize());
      }

      int last_gpu = n_gpus - 1;
      if (last_gpu != 0) CUDA_CHECK(cudaMemcpy(gb[0].d_hidden, gb[last_gpu].d_hidden, C.D * sizeof(float), cudaMemcpyDeviceToDevice));
      cudaSetDevice(0);
      
      // Final Output Head: Fused RMS + Quant
      rmsnorm_quant<<<1, 256>>>(gb[0].d_x_q, gb[0].d_x_scale, gb[0].d_norm_out, gb[0].d_hidden, d_lnf, C.rms_eps, C.D);
      float head_xs; CUDA_CHECK(cudaMemcpy(&head_xs, gb[0].d_x_scale, sizeof(float), cudaMemcpyDeviceToHost));
      gemv_w8_prequant<<<C.V, 128>>>(d_logits, gb[0].d_x_q, head_xs, d_lmhead_w, d_lmhead_s, C.V, C.D);
      
      argmax_kernel<<<1, 256>>>(d_logits, d_next_tok, C.V);
      CUDA_CHECK(cudaMemcpy(&next_tok, d_next_tok, sizeof(int), cudaMemcpyDeviceToHost));

      float step_loss = 0.0f;
      if (t > 0) {
          cross_entropy_loss<<<1, 256>>>(d_logits, next_tok, d_loss, C.V);
          CUDA_CHECK(cudaMemcpy(&step_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
          total_loss += step_loss; loss_count++;
      }
      if (t >= prompt_len - 1) {
          append_token_id(next_tok);
          generated++;
      }
      char idbuf[64]; snprintf(idbuf, 64, "TokenID: %d", next_tok);
      send_telemetry(&telem, step_loss, t, gb[0].d_hidden, C.D, idbuf);
      if(t == 0 || t % 10 == 0) printf("t=%d tok=%d\n", t, next_tok);
      if (generated >= Tgen) break;
    }
    mark_done();
    printf("> Generation complete.\n\n> "); fflush(stdout);
  }
  return 0;
}
CUEOF

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PHASE 3: Skip (assume env ready)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PHASE 4: Export Model (v3 format: Q8 weights + F32 norms/bias)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
if [ ! -f "$MODEL_BIN" ]; then
  echo "[*] Exporting $MODEL_REPO â†’ $MODEL_BIN ..."
  "$PYTHON" /tmp/_dsi8_export.py "$MODEL_REPO" "$MODEL_BIN"
fi

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PHASE 5: Compile
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
echo "[*] Compiling engine (-O3, --use_fast_math)..."
if [ "$FORCE_REBUILD" = "1" ] || [ ! -x "$ENGINE_BIN" ]; then
  arch_list="$CUDA_ARCH_LIST"
  if [ -z "${arch_list// /}" ]; then
    arch_list="$SM"
  fi
  IFS=',' read -r -a _archs <<< "$arch_list"
  NVCC_ARCH_FLAGS=()
  LAST_ARCH=""
  for raw_arch in "${_archs[@]}"; do
    arch="${raw_arch// /}"
    if [ -z "$arch" ]; then
      continue
    fi
    NVCC_ARCH_FLAGS+=("-gencode=arch=compute_${arch},code=sm_${arch}")
    LAST_ARCH="$arch"
  done
  if [ -z "$LAST_ARCH" ]; then
    LAST_ARCH="$SM"
    NVCC_ARCH_FLAGS+=("-gencode=arch=compute_${LAST_ARCH},code=sm_${LAST_ARCH}")
  fi
  NVCC_ARCH_FLAGS+=("-gencode=arch=compute_${LAST_ARCH},code=compute_${LAST_ARCH}")
  echo "[*] CUDA_ARCH_LIST=$arch_list"
  mkdir -p "$(dirname "$ENGINE_BIN")"
  nvcc -O3 --use_fast_math -Xptxas -O3,-v "${NVCC_ARCH_FLAGS[@]}" \
       /tmp/_dsi8_engine.cu -o "$ENGINE_BIN" -lnvidia-ml
else
  echo "[*] Reusing existing engine binary: $ENGINE_BIN"
fi

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PHASE 6: Run
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
echo "================================================================="
echo "  DeepSeek INT8 Persistent Decode Engine (Hybrid FP32/INT8)"
echo "  Model: $MODEL_REPO"
echo "  Binary: $MODEL_BIN"
echo "================================================================="
if [ "$SKIP_RUN" = "1" ]; then
  echo "[*] SKIP_RUN=1 -> build/export finished."
  exit 0
fi
"$ENGINE_BIN" "$MODEL_BIN" "$MODEL_REPO"
