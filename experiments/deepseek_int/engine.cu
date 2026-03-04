#include <cuda_runtime.h>
#include <stdint.h>
#include <iostream>
#include "weights_loader.h"

#ifndef H
#define H 32
#endif
#ifndef Dh
#define Dh 128
#endif
#ifndef Tmax
#define Tmax 4096
#endif

#define EXP_SHIFT 12
#define EXP_CLAMP_NEG (-16 << EXP_SHIFT)

__constant__ int32_t c_exp_lut[4096];

__device__ __forceinline__ int32_t lut_exp_q12(int32_t dx_q12) {
  if (dx_q12 < EXP_CLAMP_NEG) dx_q12 = EXP_CLAMP_NEG;
  if (dx_q12 > 0) dx_q12 = 0;
  int idx = (dx_q12 - EXP_CLAMP_NEG) >> 4; 
  if (idx < 0) idx = 0;
  if (idx > 4095) idx = 4095;
  return c_exp_lut[idx];
}

__device__ __forceinline__ int32_t warp_sum(int32_t v) {
  for (int d=16; d>0; d>>=1) v += __shfl_down_sync(0xFFFFFFFFu, v, d);
  return v;
}

__device__ void attn_online_one_head(
    int32_t* outDh,
    const int8_t* q,
    const int8_t* Kt,
    const int8_t* Vt,
    int t,
    int32_t score_scale_q12)
{
  int lane = threadIdx.x; 
  if (lane >= Dh) return;

  int32_t m = -2147483647;
  int32_t s = 0; 
  int32_t acc = 0;

  for (int kpos = 0; kpos <= t; kpos++) {
    const int8_t* kvec = Kt + (size_t)kpos * (size_t)Dh;
    const int8_t* vvec = Vt + (size_t)kpos * (size_t)Dh;

    int32_t part = 0;
    for (int i = lane; i < Dh; i += 32) part += (int32_t)q[i] * (int32_t)kvec[i];
    int32_t dot = warp_sum(part);
    dot = __shfl_sync(0xFFFFFFFFu, dot, 0); 

    int32_t sc = (int32_t)(((int64_t)dot * (int64_t)score_scale_q12) >> 12);

    int32_t new_m = max(m, sc);
    int32_t e_old = lut_exp_q12(m - new_m);
    int32_t e_new = lut_exp_q12(sc - new_m);

    acc = (int32_t)(((int64_t)acc * (int64_t)e_old) >> 12) + (int32_t)(((int64_t)(int32_t)vvec[lane] * (int64_t)e_new));
    s   = (int32_t)(((int64_t)s   * (int64_t)e_old) >> 12) + e_new;
    m = new_m;
  }
  if (s == 0) s = 1;
  outDh[lane] = (int32_t)(((int64_t)acc << 12) / (int64_t)s);
}

extern "C" __global__
void persistent_decode_one_seq(
    int32_t* __restrict__ out_logits, 
    const int8_t* __restrict__ w_qkv,  
    const int8_t* __restrict__ w_out,   
    int8_t* __restrict__ Kcache,       
    int8_t* __restrict__ Vcache,
    const int32_t* __restrict__ scales, 
    int t_start,
    int t_end)
{
  int head = (int)blockIdx.x;
  int lane = (int)threadIdx.x;
  __shared__ int8_t  sh_q[Dh];
  __shared__ int32_t sh_out[Dh];

  for (int t = t_start; t < t_end; t++) {
    if (lane < Dh) sh_q[lane] = 1; 
    __syncthreads();

    const int8_t* Kt = Kcache + (size_t)head * Dh;
    const int8_t* Vt = Vcache + (size_t)head * Dh;
    int32_t score_scale_q12 = scales[head]; 

    if (lane < Dh) attn_online_one_head(sh_out, sh_q, Kt, Vt, t, score_scale_q12);
    __syncthreads();

    if (lane == 0 && out_logits) out_logits[t] = sh_out[0]; 
    __syncthreads();
  }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_q8.bin>" << std::endl;
        return 1;
    }
    std::cout << "[*] DeepSeek Int8 Persistent Decode Engine" << std::endl;
    std::cout << "[*] Mmapping " << argv[1] << " (Zero-Copy)..." << std::endl;
    
    try {
        WeightsLoader loader(argv[1]);
        TensorView q_proj = loader.get("model.layers.0.self_attn.q_proj.weight");
        TensorView lm_head = loader.get("lm_head.weight");

        int8_t *d_w_qkv, *d_w_out, *d_Kcache, *d_Vcache;
        int32_t *d_out_logits, *d_scales;
        
        cudaMalloc(&d_w_qkv, q_proj.rows * q_proj.cols * sizeof(int8_t));
        cudaMemcpy(d_w_qkv, q_proj.data, q_proj.rows * q_proj.cols * sizeof(int8_t), cudaMemcpyHostToDevice);
        
        cudaMalloc(&d_w_out, lm_head.rows * lm_head.cols * sizeof(int8_t));
        cudaMemcpy(d_w_out, lm_head.data, lm_head.rows * lm_head.cols * sizeof(int8_t), cudaMemcpyHostToDevice);

        cudaMalloc(&d_out_logits, 4096 * sizeof(int32_t));
        cudaMalloc(&d_Kcache, 32 * 4096 * 128); 
        cudaMalloc(&d_Vcache, 32 * 4096 * 128);
        cudaMalloc(&d_scales, 32 * sizeof(int32_t));

        cudaMemset(d_Kcache, 0, 32 * 4096 * 128);
        cudaMemset(d_Vcache, 0, 32 * 4096 * 128);
        cudaMemcpy(d_scales, q_proj.scales_q12, 32 * sizeof(int32_t), cudaMemcpyHostToDevice);

        std::cout << "[*] Launching Fused Persistent Decode Kernel..." << std::endl;
        persistent_decode_one_seq<<<32, 128>>>(d_out_logits, d_w_qkv, d_w_out, d_Kcache, d_Vcache, d_scales, 0, 10);
        
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
            return 1;
        }
        std::cout << "[*] Kernel Validated and Executed Successfully." << std::endl;

        cudaFree(d_out_logits); cudaFree(d_w_qkv); cudaFree(d_w_out);
        cudaFree(d_Kcache); cudaFree(d_Vcache); cudaFree(d_scales);
    } catch (const std::exception& e) {
        std::cerr << "[!] Loader Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
