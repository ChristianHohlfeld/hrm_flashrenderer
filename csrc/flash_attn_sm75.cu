// Copyright (c) 2026 Christian Heinrich Hohlfeld (Konstanz, Germany)
// https://christianhohlfeld.com | ORCID: https://orcid.org/0009-0003-6634-9045
// ALL RIGHTS RESERVED. No license granted without prior written permission.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cmath>
#include <stdint.h>

using namespace nvcuda;

__device__ __forceinline__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, off));
    return v;
}
__device__ __forceinline__ void ld16B_to_smem(void* smem_ptr, const void* gmem_ptr, bool pred) {
    uint4 v = pred ? *reinterpret_cast<const uint4*>(gmem_ptr) : uint4{0,0,0,0};
    *reinterpret_cast<uint4*>(smem_ptr) = v;
}

static constexpr int BM = 16;
static constexpr int BN = 64;
static constexpr int DPAD = 128;   // For D=128.

struct ContigAccessor {
    const half* K;
    const half* V;
    int N;
    int D;
    int H;
    int bi;
    int h;

    __device__ __forceinline__ int get_N() const { return N; }
    __device__ __forceinline__ bool valid_kpos(int kpos) const { return (kpos >= 0 && kpos < N); }
    __device__ __forceinline__ const half* k_ptr(int kpos, int d0) const {
        int64_t idx = (((int64_t)(bi * H + h) * N + kpos) * D + d0);
        return K + idx;
    }
    __device__ __forceinline__ const half* v_ptr(int kpos, int d0) const {
        int64_t idx = (((int64_t)(bi * H + h) * N + kpos) * D + d0);
        return V + idx;
    }
};

struct PagedAccessor {
    const half* Kp;
    const half* Vp;
    const int32_t* seqlen;
    const int32_t* head_map;
    int Hk, P, PS, D;
    int bi;
    int h;

    __device__ __forceinline__ int hk() const { return (int)head_map[h]; }
    __device__ __forceinline__ int get_N() const { return (int)seqlen[bi]; }
    __device__ __forceinline__ bool valid_kpos(int kpos) const {
        if (kpos < 0) return false;
        int n = get_N();
        if (kpos >= n) return false;
        int page = kpos / PS;
        return page >= 0 && page < P;
    }
    __device__ __forceinline__ const half* k_ptr(int kpos, int d0) const {
        int page = kpos / PS;
        int off  = kpos - page * PS;
        int hk_ = hk();
        int64_t idx = (((((int64_t)bi * Hk + hk_) * P + page) * PS + off) * D + d0);
        return Kp + idx;
    }
    __device__ __forceinline__ const half* v_ptr(int kpos, int d0) const {
        int page = kpos / PS;
        int off  = kpos - page * PS;
        int hk_ = hk();
        int64_t idx = (((((int64_t)bi * Hk + hk_) * P + page) * PS + off) * D + d0);
        return Vp + idx;
    }
};

// ---------------- D=64 ----------------
template<typename Acc>
__global__ void flash_attn_bn64_d64(
    const half* __restrict__ Q,
    Acc acc,
    half* __restrict__ O,
    int B, int H, int M,
    bool causal,
    int q_offset,
    int k_offset
){
    const int tid  = (int)threadIdx.x;
    const int warp = tid >> 5;   // 0..15
    const int lane = tid & 31;

    const int bi = (int)blockIdx.x;
    const int hi = (int)blockIdx.y;
    const int qb = (int)blockIdx.z;

    acc.bi = bi;
    acc.h  = hi;

    const int q0 = qb * BM;
    const int row = warp;
    const int qi  = q0 + row;

    __shared__ __align__(16) half  shK[BN][64];
    __shared__ __align__(16) half  shV[BN][64];
    __shared__ __align__(16) float shS[BM][BN];

    const float inv_sqrt_d = rsqrtf(64.0f);

    float m_i = -INFINITY;
    float l_i = 0.f;
    float out0 = 0.f, out1 = 0.f;

    const int Ntot = acc.get_N();

    for (int k0 = 0; k0 < Ntot; k0 += BN) {
        const int tile_n = (k0 + BN <= Ntot) ? BN : (Ntot - k0);

        for (int idx = tid; idx < BN * 64; idx += blockDim.x) {
            int tn = idx / 64;
            int d  = idx - tn * 64;
            int kpos = k0 + tn;

            bool pred = (tn < tile_n) && acc.valid_kpos(kpos);
            half kv = __float2half(0.f);
            half vv = __float2half(0.f);
            if (pred) {
                kv = acc.k_ptr(kpos, d)[0];
                vv = acc.v_ptr(kpos, d)[0];
            }
            shK[tn][d] = kv;
            shV[tn][d] = vv;
        }
        __syncthreads();

        if (qi < M) {
            int64_t qbase = (((int64_t)(bi * H + hi) * M + qi) * 64);
            float qd0 = __half2float(Q[qbase + lane]);
            float qd1 = __half2float(Q[qbase + lane + 32]);

            for (int t = 0; t < tile_n; t++) {
                float partial = qd0 * __half2float(shK[t][lane]) + qd1 * __half2float(shK[t][lane + 32]);
                float dot = warp_sum(partial);
                dot = __shfl_sync(0xffffffff, dot, 0);
                if (lane == 0) shS[row][t] = dot * inv_sqrt_d;
            }
            if (lane == 0) for (int t = tile_n; t < BN; t++) shS[row][t] = -INFINITY;
        } else {
            if (lane == 0) for (int t=0; t<BN; t++) shS[row][t] = -INFINITY;
        }
        __syncthreads();

        if (qi < M) {
            const int q_abs = q_offset + qi;

            float row_max = -INFINITY;
            if (lane == 0) {
                for (int t=0; t<tile_n; t++) {
                    int k_abs = k_offset + (k0 + t);
                    if (causal && k_abs > q_abs) continue;
                    row_max = fmaxf(row_max, shS[row][t]);
                }
            }
            row_max = __shfl_sync(0xffffffff, row_max, 0);

            float sum_exp = 0.f;
            float ob0 = 0.f, ob1 = 0.f;

            for (int t=0; t<tile_n; t++) {
                int k_abs = k_offset + (k0 + t);
                if (causal && k_abs > q_abs) continue;

                float s = shS[row][t];
                float w = __expf(s - row_max);
                if (lane == 0) sum_exp += w;

                ob0 += w * __half2float(shV[t][lane]);
                ob1 += w * __half2float(shV[t][lane + 32]);
            }
            sum_exp = __shfl_sync(0xffffffff, sum_exp, 0);

            float m_new = fmaxf(m_i, row_max);
            float alpha = __expf(m_i - m_new);
            float beta  = __expf(row_max - m_new);

            l_i = l_i * alpha + sum_exp * beta;
            out0 = out0 * alpha + ob0 * beta;
            out1 = out1 * alpha + ob1 * beta;
            m_i = m_new;
        }
        __syncthreads();
    }

    if (qi < M) {
        float inv_l = 1.0f / l_i;
        int64_t obase = (((int64_t)(bi * H + hi) * M + qi) * 64);
        O[obase + lane]      = __float2half(out0 * inv_l);
        O[obase + lane + 32] = __float2half(out1 * inv_l);
    }
}

// ---------------- D=128 (WMMA QK + WMMA PV) ----------------
template<typename Acc>
__global__ void flash_attn_bn64_d128_wmma(
    const half* __restrict__ Q,
    Acc acc,
    half* __restrict__ O,
    int B, int H, int M,
    bool causal,
    int q_offset,
    int k_offset
){
    const int tid  = (int)threadIdx.x;
    const int warp = tid >> 5;   // 0..15
    const int lane = tid & 31;

    const int bi = (int)blockIdx.x;
    const int hi = (int)blockIdx.y;
    const int qb = (int)blockIdx.z;

    acc.bi = bi;
    acc.h  = hi;

    const int q0 = qb * BM;
    const int row = warp;        // 0..15
    const int qi  = q0 + row;

    __shared__ __align__(16) half  shQ[BM][DPAD];      // [16][128]
    __shared__ __align__(16) half  shK[BN][DPAD];      // [64][128]
    __shared__ __align__(16) half  shVcol[128][BN];    // [128][64] (V as col_major ld=BN)
    __shared__ __align__(16) float shS[BM][BN];
    __shared__ __align__(16) half  shP[BM][BN];
    __shared__ __align__(16) half  shO[BM][128];

    const float inv_sqrt_d = rsqrtf(128.0f);

    for (int idx = tid; idx < BM * 128; idx += blockDim.x) {
        int rr = idx / 128;
        int cc = idx - rr * 128;
        half vq = __float2half(0.f);
        int qidx = q0 + rr;
        if (qidx < M) {
            int64_t qbase = (((int64_t)(bi * H + hi) * M + qidx) * 128);
            vq = Q[qbase + cc];
        }
        shQ[rr][cc] = vq;
    }
    __syncthreads();

    float m_i = -INFINITY;
    float l_i = 0.f;

    float out0=0.f, out1=0.f, out2=0.f, out3=0.f;

    const int Ntot = acc.get_N();

    for (int k0 = 0; k0 < Ntot; k0 += BN) {
        const int tile_n = (k0 + BN <= Ntot) ? BN : (Ntot - k0);

        constexpr int vecs_per_row = 128 / 8;
        constexpr int total_vecs = BN * vecs_per_row;

        for (int vec = tid; vec < total_vecs; vec += blockDim.x) {
            int tn = vec / vecs_per_row;
            int vcol = vec - tn * vecs_per_row;
            int d0 = vcol * 8;
            int kpos = k0 + tn;

            bool pred = (tn < tile_n) && acc.valid_kpos(kpos);

            const half* gk = pred ? acc.k_ptr(kpos, d0) : (const half*)nullptr;
            const half* gv = pred ? acc.v_ptr(kpos, d0) : (const half*)nullptr;

            ld16B_to_smem(&shK[tn][d0], gk, pred);

            uint4 raw = pred ? *reinterpret_cast<const uint4*>(gv) : uint4{0,0,0,0};
            const half* hv = reinterpret_cast<const half*>(&raw);
            #pragma unroll
            for (int i=0;i<8;i++){
                shVcol[d0 + i][tn] = hv[i];
            }
        }
        __syncthreads();

        if (warp < 4) {
            int colBase = warp * 16;

            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
            wmma::fill_fragment(c, 0.f);

            #pragma unroll
            for (int kk = 0; kk < 128; kk += 16) {
                wmma::load_matrix_sync(a, &shQ[0][kk], DPAD);
                wmma::load_matrix_sync(b, &shK[colBase][kk], DPAD);
                wmma::mma_sync(c, a, b, c);
            }

            float tmp[16*16];
            wmma::store_matrix_sync(tmp, c, 16, wmma::mem_row_major);

            for (int i = lane; i < 16*16; i += 32) {
                int rr = i / 16;
                int cc = i - rr*16;
                int col = colBase + cc;
                float val = tmp[i] * inv_sqrt_d;
                if (col >= tile_n) val = -INFINITY;
                shS[rr][col] = val;
            }
        }
        __syncthreads();

        float beta = 0.f;
        float sum_exp = 0.f;

        if (qi < M) {
            const int q_abs = q_offset + qi;

            float s0 = -INFINITY;
            float s1 = -INFINITY;

            int t0 = lane;
            int t1 = lane + 32;

            if (t0 < tile_n) {
                int k_abs = k_offset + (k0 + t0);
                if (!(causal && k_abs > q_abs)) s0 = shS[row][t0];
            }
            if (t1 < tile_n) {
                int k_abs = k_offset + (k0 + t1);
                if (!(causal && k_abs > q_abs)) s1 = shS[row][t1];
            }

            float row_max = fmaxf(s0, s1);
            row_max = warp_max(row_max);
            row_max = __shfl_sync(0xffffffff, row_max, 0);

            float w0 = (s0 > -INFINITY/2) ? __expf(s0 - row_max) : 0.f;
            float w1 = (s1 > -INFINITY/2) ? __expf(s1 - row_max) : 0.f;

            if (t0 < BN) shP[row][t0] = __float2half_rn(w0);
            if (t1 < BN) shP[row][t1] = __float2half_rn(w1);

            float local_sum = w0 + w1;
            sum_exp = warp_sum(local_sum);
            sum_exp = __shfl_sync(0xffffffff, sum_exp, 0);

            float m_new = fmaxf(m_i, row_max);
            float alpha = __expf(m_i - m_new);
            beta = __expf(row_max - m_new);

            l_i *= alpha;
            out0 *= alpha; out1 *= alpha; out2 *= alpha; out3 *= alpha;

            l_i += sum_exp * beta;
            m_i = m_new;
        } else {
            if (lane == 0) {
                for (int t=0;t<BN;t++) shP[row][t] = __float2half(0.f);
            }
        }
        __syncthreads();

        if (warp < 8) {
            int colBase = warp * 16;

            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
            wmma::fill_fragment(c, 0.f);

            #pragma unroll
            for (int kk = 0; kk < 64; kk += 16) {
                wmma::load_matrix_sync(a, &shP[0][kk], BN);
                wmma::load_matrix_sync(b, &shVcol[colBase][kk], BN);
                wmma::mma_sync(c, a, b, c);
            }

            float tmp[16*16];
            wmma::store_matrix_sync(tmp, c, 16, wmma::mem_row_major);

            for (int i = lane; i < 16*16; i += 32) {
                int rr = i / 16;
                int cc = i - rr*16;
                shO[rr][colBase + cc] = __float2half_rn(tmp[i]);
            }
        }
        __syncthreads();

        if (qi < M) {
            int d = lane;
            float ob0 = __half2float(shO[row][d]);
            float ob1 = __half2float(shO[row][d + 32]);
            float ob2 = __half2float(shO[row][d + 64]);
            float ob3 = __half2float(shO[row][d + 96]);

            out0 += ob0 * beta;
            out1 += ob1 * beta;
            out2 += ob2 * beta;
            out3 += ob3 * beta;
        }
        __syncthreads();
    }

    if (qi < M) {
        float inv_l = 1.0f / l_i;
        int64_t obase = (((int64_t)(bi * H + hi) * M + qi) * 128);
        int d = lane;
        O[obase + d]       = __float2half_rn(out0 * inv_l);
        O[obase + d + 32]  = __float2half_rn(out1 * inv_l);
        O[obase + d + 64]  = __float2half_rn(out2 * inv_l);
        O[obase + d + 96]  = __float2half_rn(out3 * inv_l);
    }
}

// ---------------- Host entrypoints ----------------
torch::Tensor flash_attn_fwd_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal, int64_t q_offset, int64_t k_offset) {
    const int B = (int)q.size(0);
    const int H = (int)q.size(1);
    const int M = (int)q.size(2);
    const int D = (int)q.size(3);
    const int N = (int)k.size(2);

    TORCH_CHECK(D == 64 || D == 128, "Supported head_dim: 64 or 128");

    auto o = torch::empty_like(q);
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    dim3 grid(B, H, (M + BM - 1)/BM);
    dim3 block(512);

    ContigAccessor acc;
    acc.K = (const half*)k.data_ptr<at::Half>();
    acc.V = (const half*)v.data_ptr<at::Half>();
    acc.N = N;
    acc.D = D;
    acc.H = H;
    acc.bi = 0;
    acc.h  = 0;

    if (D == 64) {
        flash_attn_bn64_d64<ContigAccessor><<<grid, block, 0, stream>>>(
            (const half*)q.data_ptr<at::Half>(), acc,
            (half*)o.data_ptr<at::Half>(),
            B, H, M, causal, (int)q_offset, (int)k_offset
        );
    } else {
        flash_attn_bn64_d128_wmma<ContigAccessor><<<grid, block, 0, stream>>>(
            (const half*)q.data_ptr<at::Half>(), acc,
            (half*)o.data_ptr<at::Half>(),
            B, H, M, causal, (int)q_offset, (int)k_offset
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return o;
}

torch::Tensor flash_attn_paged_fwd_cuda(
    torch::Tensor q,
    torch::Tensor k_pages,
    torch::Tensor v_pages,
    torch::Tensor seqlen,
    int64_t page_size,
    torch::Tensor head_map,
    bool causal,
    int64_t q_offset,
    int64_t k_offset
){
    const int B  = (int)q.size(0);
    const int Hq = (int)q.size(1);
    const int M  = (int)q.size(2);
    const int D  = (int)q.size(3);

    const int Hk = (int)k_pages.size(1);
    const int P  = (int)k_pages.size(2);
    const int PS = (int)k_pages.size(3);

    TORCH_CHECK((int)page_size == PS, "page_size mismatch");
    TORCH_CHECK(D == 64 || D == 128, "Supported head_dim: 64 or 128");

    auto o = torch::empty_like(q);
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    dim3 grid(B, Hq, (M + BM - 1)/BM);
    dim3 block(512);

    PagedAccessor acc;
    acc.Kp = (const half*)k_pages.data_ptr<at::Half>();
    acc.Vp = (const half*)v_pages.data_ptr<at::Half>();
    acc.seqlen = (const int32_t*)seqlen.data_ptr<int32_t>();
    acc.head_map = (const int32_t*)head_map.data_ptr<int32_t>();
    acc.Hk = Hk;
    acc.P  = P;
    acc.PS = PS;
    acc.D  = D;
    acc.bi = 0;
    acc.h  = 0;

    if (D == 64) {
        flash_attn_bn64_d64<PagedAccessor><<<grid, block, 0, stream>>>(
            (const half*)q.data_ptr<at::Half>(), acc,
            (half*)o.data_ptr<at::Half>(),
            B, Hq, M, causal, (int)q_offset, (int)k_offset
        );
    } else {
        flash_attn_bn64_d128_wmma<PagedAccessor><<<grid, block, 0, stream>>>(
            (const half*)q.data_ptr<at::Half>(), acc,
            (half*)o.data_ptr<at::Half>(),
            B, Hq, M, causal, (int)q_offset, (int)k_offset
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return o;
}

// ---------------- Append-fast ----------------
__global__ void paged_kv_append_kernel(
    half* __restrict__ Kp, half* __restrict__ Vp,
    const half* __restrict__ Kn, const half* __restrict__ Vn,
    int B, int Hk, int P, int PS, int D, int T,
    int start_pos
){
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int64_t total = (int64_t)B * Hk * T * D;
    if ((int64_t)idx >= total) return;

    int d = idx % D;
    int tmp = idx / D;
    int t = tmp % T;
    tmp /= T;
    int hk = tmp % Hk;
    int b  = tmp / Hk;

    int pos = start_pos + t;
    int page = pos / PS;
    int off  = pos - page * PS;
    if (page < 0 || page >= P) return;

    int64_t src = (((int64_t)b * Hk + hk) * T + t) * D + d;
    int64_t dst = (((((int64_t)b * Hk + hk) * P + page) * PS + off) * D + d);

    Kp[dst] = Kn[src];
    Vp[dst] = Vn[src];
}

void paged_kv_append_cuda(
    torch::Tensor k_pages,
    torch::Tensor v_pages,
    torch::Tensor k_new,
    torch::Tensor v_new,
    int64_t start_pos,
    int64_t page_size
){
    const int B  = (int)k_pages.size(0);
    const int Hk = (int)k_pages.size(1);
    const int P  = (int)k_pages.size(2);
    const int PS = (int)k_pages.size(3);
    const int D  = (int)k_pages.size(4);
    const int T  = (int)k_new.size(2);

    TORCH_CHECK(PS == (int)page_size, "page_size mismatch");

    const int threads = 256;
    int64_t total = (int64_t)B * Hk * T * D;
    int blocks = (int)((total + threads - 1) / threads);

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    paged_kv_append_kernel<<<blocks, threads, 0, stream>>>(
        (half*)k_pages.data_ptr<at::Half>(),
        (half*)v_pages.data_ptr<at::Half>(),
        (const half*)k_new.data_ptr<at::Half>(),
        (const half*)v_new.data_ptr<at::Half>(),
        B, Hk, P, PS, D, T,
        (int)start_pos
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

