// © 2026 Christian Heinrich Hohlfeld (Konstanz, Germany), christianhohlfeld.com, ORCID 0009-0003-6634-9045.

#include <cuda_runtime.h>
#include <stdint.h>
#include <iostream>

#ifndef H
#define H 32
#endif
#ifndef Dh
#define Dh 128
#endif
#ifndef Tmax
#define Tmax 4096
#endif

// Q formats
#define EXP_SHIFT 12
#define LOGIT_SHIFT 10
#define EXP_CLAMP_NEG (-16 << EXP_SHIFT)

// LUT in constant memory (fast + cached)
__constant__ int32_t c_exp_lut[4096];

__device__ __forceinline__ int32_t lut_exp_q12(int32_t dx_q12) {
  if (dx_q12 < EXP_CLAMP_NEG) dx_q12 = EXP_CLAMP_NEG;
  if (dx_q12 > 0) dx_q12 = 0;
  int idx = (dx_q12 - EXP_CLAMP_NEG) >> 4; // map [-16..0] to [0..4095]
  if (idx < 0) idx = 0;
  if (idx > 4095) idx = 4095;
  return c_exp_lut[idx];
}

__device__ __forceinline__ int32_t warp_sum(int32_t v) {
  for (int d=16; d>0; d>>=1) v += __shfl_down_sync(0xFFFFFFFFu, v, d);
  return v;
}

__device__ __forceinline__ int32_t warp_max(int32_t v) {
  for (int d=16; d>0; d>>=1) v = max(v, __shfl_down_sync(0xFFFFFFFFu, v, d));
  return v;
}

// KV cache layout: K[layer][t][head][Dh], int8
// Q is int8, but we often compute q in int16/int32 before dot
// Here: q_i8, k_i8, v_i8 in int8, dp4a dot -> int32
__device__ __forceinline__ int32_t dp4a_dot_i8(const int8_t* a, const int8_t* b) {
  int32_t acc = 0;
  #pragma unroll
  for (int i=0;i<Dh;i+=4) {
    int32_t av = *((const int32_t*)(a+i));
    int32_t bv = *((const int32_t*)(b+i));
    acc = __dp4a(av, bv, acc);
  }
  return acc;
}

// ONLINE SOFTMAX ATTENTION (single head, single token t)
// Produces outDh (int32) for this head.
// - q: int8[Dh]
// - K: int8[(t+1)*Dh]
// - V: int8[(t+1)*Dh]
// - score_scale_q12: fixed-point scale for scores before exp
__device__ void attn_online_one_head(
    int32_t* outDh,
    const int8_t* q,
    const int8_t* Kt,   // base ptr for this head in KV cache (time-major)
    const int8_t* Vt,
    int t,
    int32_t score_scale_q12)
{
  int lane = threadIdx.x; // assume blockDim.x == Dh or at least >=Dh
  if (lane >= Dh) return;

  // streaming stats
  int32_t m = -2147483647;
  int32_t s = 0; // sum exp in Q12

  // accumulation of V weighted (int32)
  int32_t acc = 0;

  // iterate over kpos = 0..t
  for (int kpos = 0; kpos <= t; kpos++) {
    const int8_t* kvec = Kt + (size_t)kpos * (size_t)Dh;
    const int8_t* vvec = Vt + (size_t)kpos * (size_t)Dh;

    // do warp-strided partial dp4a then warp_sum.
    int32_t part = 0;
    for (int i = lane; i < Dh; i += 32) {
      part += (int32_t)q[i] * (int32_t)kvec[i];
    }
    int32_t dot = warp_sum(part);
    dot = __shfl_sync(0xFFFFFFFFu, dot, 0); // broadcast

    // scale to Q12
    int32_t sc = (int32_t)(((int64_t)dot * (int64_t)score_scale_q12) >> 12);

    // online softmax update
    int32_t new_m = max(m, sc);
    int32_t e_old = lut_exp_q12(m - new_m);
    int32_t e_new = lut_exp_q12(sc - new_m);

    // acc = acc * e_old + v * e_new
    // s   = s   * e_old + e_new
    // both in Q12
    acc = (int32_t)(((int64_t)acc * (int64_t)e_old) >> 12) + (int32_t)(((int64_t)(int32_t)vvec[lane] * (int64_t)e_new));
    s   = (int32_t)(((int64_t)s   * (int64_t)e_old) >> 12) + e_new;

    m = new_m;
  }

  if (s == 0) s = 1;
  outDh[lane] = (int32_t)(((int64_t)acc << 12) / (int64_t)s);
}

// PERSISTENT DECODE KERNEL
extern "C" __global__
void persistent_decode_one_seq(
    int32_t* __restrict__ out_logits,   // [Tgen][V] or streamed
    const int8_t* __restrict__ w_qkv,    // packed weights (int8) - MVP placeholder
    const int8_t* __restrict__ w_out,    // vocab projection (int8) - MVP placeholder
    int8_t* __restrict__ Kcache,         // [L][Tmax][H][Dh]
    int8_t* __restrict__ Vcache,
    const int32_t* __restrict__ scales,  // per-layer/head scales in Q12
    int t_start,
    int t_end)
{
  int head = (int)blockIdx.x;
  int lane = (int)threadIdx.x;

  __shared__ int8_t  sh_q[Dh];
  __shared__ int32_t sh_out[Dh];

  // For MVP this kernel only demonstrates attention.
  for (int t = t_start; t < t_end; t++) {

    if (lane < Dh) sh_q[lane] = 1; // TODO: real q
    __syncthreads();

    // pointers into KV cache for this head at time 0..t
    const int8_t* Kt = Kcache + (size_t)head * Dh; // MVP simplicity missing L/stride
    const int8_t* Vt = Vcache + (size_t)head * Dh;

    int32_t score_scale_q12 = scales[head]; 

    // Online attention
    if (lane < Dh) {
      attn_online_one_head(sh_out, sh_q, Kt, Vt, t, score_scale_q12);
    }
    __syncthreads();

    // vocab projection
    if (lane == 0) {
      if (out_logits) {
         out_logits[t] = sh_out[0]; // Placeholder for token out
      }
    }
    __syncthreads();
  }
}

// Host Entry
extern "C" void launch_persistent_decode(
    int32_t* out_logits,
    const int8_t* w_qkv,
    const int8_t* w_out,
    int8_t* Kcache,
    int8_t* Vcache,
    const int32_t* scales,
    int t_start, int t_end) 
{
    // Launch one block per head, 128 threads per block (Dh=128 usually, scaled to max lane)
    persistent_decode_one_seq<<<H, Dh>>>(out_logits, w_qkv, w_out, Kcache, Vcache, scales, t_start, t_end);
}
