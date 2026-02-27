/*
=============================================================================
© 2026 Christian Heinrich Hohlfeld (Konstanz, Deutschland) — Alle Rechte vorbehalten.
Website: christianhohlfeld.com
ORCID: 0009-0003-6634-9045

Attribution / Ownership Notice:
This code includes "ID-based tokenization / index training" and related agentic infrastructure
concepts asserted by Christian Heinrich Hohlfeld as his intellectual property. Keep this header.
=============================================================================

llm_engine.cu — training-only (no PhO), sm_75, D=256/H=8/Dh=16, reversible blocks,
tiled FlashAttention, WMMA for linear layers, streaming vocab head, multi-GPU DP,
CUDA Graph step capture.

FULL-BLAST CHANGE IMPLEMENTED:
- Injected fused_rms_qkv_rope_hf kernel (1 warp = 1 token) to replace:
  rms_fwd_f2h(y2) + wmma_fwd(Q) + wmma_fwd(K) + wmma_fwd(V) + rope_apply_qk
  in BOTH forward pass and backward recompute ("f backward") pass.

Notes:
- Q/K/V projections are computed in FP32 FMA (float weights) inside the fused kernel.
- All other projections remain WMMA/FP16 (tensor cores).
- RoPE is applied in-register inside the fused kernel, matching the existing layout.
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

#define CUDA_CHECK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); std::exit(1);} }while(0)
#define KERNEL_CHECK() do{ cudaError_t e=cudaGetLastError(); if(e!=cudaSuccess){ \
  std::fprintf(stderr,"KERNEL %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); std::exit(1);} }while(0)

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

static_assert(D==256, "This build assumes D=256 (Dh=16).");
static_assert(H==8,   "This build assumes H=8 (Dh=16).");
static_assert(Dh==16, "This build assumes Dh=16.");
static_assert((F%16)==0, "FFN must be multiple of 16");
static_assert((Tmax%16)==0, "TMAX must be multiple of 16");
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
  std::vector<uint16_t> id2pair; // size K1
  std::vector<int32_t>  pair2id; // size 65536, maps packed bytepair -> stage1 id (BASE_V..BASE_V+K1-1)
  std::vector<uint32_t> id2pair2; // size K2, key=(a<<16)|b
  uint32_t hmask=0;
  std::vector<uint32_t> hkeys; // 0xFFFFFFFF empty
  std::vector<uint16_t> hvals; // 0xFFFF empty
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
  if(ver!=1u) die("bad index ver");
  if((int)k1 > K1 || (int)k2 > K2) die("index K mismatch (index larger than max macros)");

  pi->id2pair.resize(k1);
  r_bytes(pi->id2pair.data(), (size_t)k1*sizeof(uint16_t));
  pi->pair2id.assign(65536,-1);
  for(int i=0;i<(int)k1;i++) pi->pair2id[pi->id2pair[(size_t)i]] = BASE_V + i;

  pi->id2pair2.resize(k2);
  if(k2>0) r_bytes(pi->id2pair2.data(), (size_t)k2*sizeof(uint32_t));
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

  if(all.size() < (size_t)K1) die("K1 too large");
  PairIndex pi;
  pi.id2pair.assign(all.begin(), all.begin()+K1);
  pi.pair2id.assign(65536,-1);
  for(int i=0;i<K1;i++) pi.pair2id[pi.id2pair[(size_t)i]] = BASE_V + i;

  // Stage2 (in-memory fallback)
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
      if((int)items.size() < K2){
        for(size_t j=items.size(); j<(size_t)K2; j++){
          uint32_t dummy = (0xFFFFu<<16) | (uint32_t)(j & 0xFFFFu);
          items.push_back({dummy,0u});
        }
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

// ================= checkpoint (v11) =================
static void wf(FILE* f,const void* p,size_t n){ if(std::fwrite(p,1,n,f)!=n) die("write failed"); }
static void rf(FILE* f,void* p,size_t n){ if(std::fread(p,1,n,f)!=n) die("read failed"); }
static void wu32(FILE* f,uint32_t x){ wf(f,&x,4); }
static uint32_t ru32(FILE* f){ uint32_t x; rf(f,&x,4); return x; }

static size_t weights_floats(){
  size_t n=0;
  n += (size_t)Vpad*(size_t)D;      // wte
  n += (size_t)Tmax*(size_t)D;      // wpe
  for(int l=0;l<L;l++){
    n += (size_t)Dhf;               // gf
    n += (size_t)Dhf*(size_t)Dhf*4; // Wq,Wk,Wv,Wo  (Dhf x Dhf)
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

// ================= Streaming Head (chunked vocab) =================
__global__ void init_row_stats(float* row_max, float* row_sum, float* loss, int N){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<N){
    row_max[i] = -1e30f;
    row_sum[i] = 0.f;
    loss[i] = 0.f;
  }
}
__global__ void chunk_max_sumexp(float* cmax, float* csum, const float* logits, int N, int Mvalid){
  int n = blockIdx.x;
  if(n>=N) return;
  __shared__ float buf[256];
  float mx=-1e30f;
  for(int j=threadIdx.x;j<Mvalid;j+=blockDim.x){
    float v = logits[(size_t)n*(size_t)VCHUNK + (size_t)j];
    if(v>mx) mx=v;
  }
  buf[threadIdx.x]=mx; __syncthreads();
  for(int k=blockDim.x/2;k>0;k>>=1){
    if(threadIdx.x<k){
      float a=buf[threadIdx.x], b=buf[threadIdx.x+k];
      buf[threadIdx.x]=(a>b)?a:b;
    }
    __syncthreads();
  }
  float m = buf[0];
  float s=0.f;
  for(int j=threadIdx.x;j<Mvalid;j+=blockDim.x){
    s += expf(logits[(size_t)n*(size_t)VCHUNK + (size_t)j] - m);
  }
  buf[threadIdx.x]=s; __syncthreads();
  for(int k=blockDim.x/2;k>0;k>>=1){ if(threadIdx.x<k) buf[threadIdx.x]+=buf[threadIdx.x+k]; __syncthreads(); }
  if(threadIdx.x==0){ cmax[n]=m; csum[n]=buf[0]; }
}
__global__ void update_row_stats(float* row_max, float* row_sum, const float* cmax, const float* csum, int N){
  int n=blockIdx.x*blockDim.x+threadIdx.x;
  if(n>=N) return;
  float rm=row_max[n];
  float rs=row_sum[n];
  float cm=cmax[n];
  float cs=csum[n];
  float nm = (rm>cm)?rm:cm;
  float rs_new = rs*expf(rm-nm) + cs*expf(cm-nm);
  row_max[n]=nm;
  row_sum[n]=rs_new;
}
__global__ void dy_loss_from_logits(float* dY, float* loss,
                                   const float* logits,
                                   const float* row_max, const float* row_sum,
                                   const uint16_t* tgt,
                                   int N, int v0, int Mvalid, float invN){
  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  int total=N*VCHUNK;
  if(idx>=total) return;
  int n = idx / VCHUNK;
  int j = idx - n*VCHUNK;
  if(j>=Mvalid){ dY[(size_t)idx]=0.f; return; }
  int v = v0 + j;
  if(v>=V){ dY[(size_t)idx]=0.f; return; }
  float lg = logits[(size_t)idx];
  float p = expf(lg - row_max[n]) / row_sum[n];
  int y = (int)tgt[n];
  dY[(size_t)idx] = (p - ((v==y)?1.f:0.f)) * invN;
  if(v==y){
    loss[n] = -((lg - row_max[n]) - logf(row_sum[n]));
  }
}
__global__ void pack_wout_rm_chunk(half* dst, const half* Wrm, int v0){
  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  int total=D*VCHUNK;
  if(idx>=total) return;
  int d = idx / VCHUNK;
  int j = idx - d*VCHUNK;
  int v = v0 + j;
  half hv = __float2half_rn(0.0f);
  if(v < Vpad) hv = Wrm[(size_t)d*(size_t)Vpad + (size_t)v];
  dst[(size_t)d*(size_t)VCHUNK + (size_t)j] = hv;
}
__global__ void scatter_add_dwout_chunk(float* dWfull, const float* dWchunk, int v0){
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

// ================= RoPE tables =================
__global__ void rope_build_tables(float* sin_tbl, float* cos_tbl, int T){
  int t = blockIdx.y * blockDim.y + threadIdx.y;
  int i = blockIdx.x * blockDim.x + threadIdx.x; // i in [0,Dh/2)
  if(t>=T || i>=Dh/2) return;
  float inv_freq = powf(ROPE_THETA, -(2.0f*i)/(float)Dh);
  float ang = (float)t * inv_freq;
  sin_tbl[(size_t)t*(size_t)(Dh/2) + (size_t)i] = sinf(ang);
  cos_tbl[(size_t)t*(size_t)(Dh/2) + (size_t)i] = cosf(ang);
}

// inverse RoPE on grads
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

// ================= FlashAttention forward (tiled, causal, no-P) =================
__global__ void flash_fwd_hf(float* O, float* m, float* lse,
                            const float* Q, const float* K, const float* Vv,
                            int B, int T){
  int b=(int)blockIdx.x;
  int h=(int)blockIdx.y;
  int qtile=(int)blockIdx.z;
  int q0=qtile*FA_QT;
  if(b>=B||h>=H) return;

  int tid=(int)threadIdx.x; // 0..255
  int qi=tid>>4;            // 0..15 query within tile
  int lane=tid&15;          // dim lane
  int warp_lane=tid&31;
  int half=(warp_lane>>4);
  unsigned mask = (half==0) ? 0x0000FFFFu : 0xFFFF0000u;

  int tq=q0+qi;
  if(tq>=T) return;

  int btq=b*T+tq;
  const float* qptr = Q + (size_t)btq*(size_t)Dhf + (size_t)h*(size_t)Dh;
  float qd=qptr[lane];

  __shared__ float Ks[FA_KT][Dh+1];
  __shared__ float Vs[FA_KT][Dh+1];

  float mi=-1e30f;
  float li=0.f;
  float oi=0.f;

  float scale = 1.0f/sqrtf((float)Dh);

  for(int k0=0;k0<T;k0+=FA_KT){
    int kend = (k0+FA_KT<T)? k0+FA_KT : T;
    int kt = kend-k0;

    for(int idx=tid; idx<kt*Dh; idx+=256){
      int kk=idx/Dh;
      int d = idx-kk*Dh;
      int t=k0+kk;
      int btk=b*T+t;
      const float* kptr = K  + (size_t)btk*(size_t)Dhf + (size_t)h*(size_t)Dh;
      const float* vptr = Vv + (size_t)btk*(size_t)Dhf + (size_t)h*(size_t)Dh;
      Ks[kk][d]=kptr[d];
      Vs[kk][d]=vptr[d];
    }
    __syncthreads();

    int maxk = (tq < kend-1) ? (tq-k0+1) : kt;
    if(maxk>0){
      float localMax=-1e30f;
      for(int kk=0; kk<maxk; kk++){
        float dot = qd*Ks[kk][lane];
        dot = reduce_sum16(dot, mask);
        float score = dot*scale;
        localMax = fmaxf(localMax, score);
      }

      float m_new = fmaxf(mi, localMax);
      float alpha = expf(mi - m_new);
      li *= alpha;
      oi *= alpha;

      float l_add=0.f;
      float o_add=0.f;
      for(int kk=0; kk<maxk; kk++){
        float dot = qd*Ks[kk][lane];
        dot = reduce_sum16(dot, mask);
        float score = dot*scale;
        float p = expf(score - m_new);
        l_add += p;
        o_add += p*Vs[kk][lane];
      }
      li += l_add;
      oi += o_add;
      mi = m_new;
    }
    __syncthreads();
  }

  float inv = 1.0f/li;
  float* outp = O + (size_t)btq*(size_t)Dhf + (size_t)h*(size_t)Dh;
  outp[lane] = oi*inv;

  if(lane==0){
    int idx=((b*H+h)*T+tq);
    m[idx]=mi;
    lse[idx]=li;
  }
}

// ================= FlashAttention backward tiled =================
__global__ void flash_bwd_dq_hf(float* dpSum, float* dQ,
                               const float* dO, const float* Q, const float* K, const float* Vv,
                               const float* m, const float* lse,
                               int B, int T){
  int b=(int)blockIdx.x;
  int h=(int)blockIdx.y;
  int qtile=(int)blockIdx.z;
  int q0=qtile*FA_QT;
  if(b>=B||h>=H) return;

  int tid=(int)threadIdx.x; // 0..255
  int qi=tid>>4;
  int lane=tid&15;
  int warp_lane=tid&31;
  int half=(warp_lane>>4);
  unsigned mask = (half==0) ? 0x0000FFFFu : 0xFFFF0000u;

  int tq=q0+qi;
  if(tq>=T) return;

  int btq=b*T+tq;
  const float* qptr = Q + (size_t)btq*(size_t)Dhf + (size_t)h*(size_t)Dh;
  const float* doptr = dO + (size_t)btq*(size_t)Dhf + (size_t)h*(size_t)Dh;
  float qd=qptr[lane];
  float dod=doptr[lane];

  int midx=((b*H+h)*T+tq);
  float mi=m[midx];
  float inv_l = 1.0f / lse[midx];
  float scale = 1.0f/sqrtf((float)Dh);

  __shared__ float Ks[FA_KT][Dh+1];
  __shared__ float Vs[FA_KT][Dh+1];

  float dp=0.f;
  for(int k0=0;k0<T;k0+=FA_KT){
    int kend=(k0+FA_KT<T)?k0+FA_KT:T;
    int kt=kend-k0;

    for(int idx=tid; idx<kt*Dh; idx+=256){
      int kk=idx/Dh;
      int d=idx-kk*Dh;
      int t=k0+kk;
      int btk=b*T+t;
      const float* kptr = K  + (size_t)btk*(size_t)Dhf + (size_t)h*(size_t)Dh;
      const float* vptr = Vv + (size_t)btk*(size_t)Dhf + (size_t)h*(size_t)Dh;
      Ks[kk][d]=kptr[d];
      Vs[kk][d]=vptr[d];
    }
    __syncthreads();

    int maxk = (tq < kend-1) ? (tq-k0+1) : kt;
    if(maxk>0){
      for(int kk=0; kk<maxk; kk++){
        float dotqk = qd*Ks[kk][lane];
        dotqk = reduce_sum16(dotqk, mask);
        float score = dotqk*scale;

        float dotdv = dod*Vs[kk][lane];
        dotdv = reduce_sum16(dotdv, mask);

        float p = expf(score - mi) * inv_l;
        dp += p * dotdv;
      }
    }
    __syncthreads();
  }
  if(lane==0) dpSum[midx]=dp;

  float dqi=0.f;
  for(int k0=0;k0<T;k0+=FA_KT){
    int kend=(k0+FA_KT<T)?k0+FA_KT:T;
    int kt=kend-k0;

    for(int idx=tid; idx<kt*Dh; idx+=256){
      int kk=idx/Dh;
      int d=idx-kk*Dh;
      int t=k0+kk;
      int btk=b*T+t;
      const float* kptr = K  + (size_t)btk*(size_t)Dhf + (size_t)h*(size_t)Dh;
      const float* vptr = Vv + (size_t)btk*(size_t)Dhf + (size_t)h*(size_t)Dh;
      Ks[kk][d]=kptr[d];
      Vs[kk][d]=vptr[d];
    }
    __syncthreads();

    int maxk = (tq < kend-1) ? (tq-k0+1) : kt;
    if(maxk>0){
      for(int kk=0; kk<maxk; kk++){
        float dotqk = qd*Ks[kk][lane];
        dotqk = reduce_sum16(dotqk, mask);
        float score = dotqk*scale;

        float dotdv = dod*Vs[kk][lane];
        dotdv = reduce_sum16(dotdv, mask);

        float p = expf(score - mi) * inv_l;
        float dS = (dotdv - dp) * p;
        dqi += dS * Ks[kk][lane];
      }
    }
    __syncthreads();
  }

  float* dqptr = dQ + (size_t)btq*(size_t)Dhf + (size_t)h*(size_t)Dh;
  dqptr[lane] += dqi*scale;
}

__global__ void flash_bwd_dkv_hf(float* dK, float* dV,
                                const float* dO, const float* Q, const float* K, const float* Vv,
                                const float* m, const float* lse, const float* dpSum,
                                int B, int T){
  int b=(int)blockIdx.x;
  int h=(int)blockIdx.y;
  int ktile=(int)blockIdx.z;
  int k0=ktile*FA_KT;
  if(b>=B||h>=H) return;

  int tid=(int)threadIdx.x; // 0..255
  int kg = tid>>4;          // 0..15
  int lane = tid&15;        // dim lane
  int warp_lane=tid&31;
  int half=(warp_lane>>4);
  unsigned mask = (half==0) ? 0x0000FFFFu : 0xFFFF0000u;

  __shared__ float Qs[FA_QT][Dh+1];
  __shared__ float dOs[FA_QT][Dh+1];

  float scale = 1.0f/sqrtf((float)Dh);

  for(int sub=0; sub<FA_KT/16; sub++){
    int s = k0 + sub*16 + kg;
    if(s>=T) continue;

    int bts=b*T+s;
    const float* kptr = K  + (size_t)bts*(size_t)Dhf + (size_t)h*(size_t)Dh;
    const float* vptr = Vv + (size_t)bts*(size_t)Dhf + (size_t)h*(size_t)Dh;
    float kd=kptr[lane];
    float vd=vptr[lane];

    float dkd=0.f;
    float dvd=0.f;

    int q0 = (s/FA_QT)*FA_QT;
    for(; q0<T; q0+=FA_QT){
      for(int idx=tid; idx<FA_QT*Dh; idx+=256){
        int qi=idx/Dh;
        int d=idx-qi*Dh;
        int t=q0+qi;
        if(t<T){
          int btq=b*T+t;
          const float* q = Q  + (size_t)btq*(size_t)Dhf + (size_t)h*(size_t)Dh;
          const float* do_ = dO + (size_t)btq*(size_t)Dhf + (size_t)h*(size_t)Dh;
          Qs[qi][d]=q[d];
          dOs[qi][d]=do_[d];
        }else{
          Qs[qi][d]=0.f;
          dOs[qi][d]=0.f;
        }
      }
      __syncthreads();

      for(int qi=0; qi<FA_QT; qi++){
        int t=q0+qi;
        if(t>=T || t<s) continue;

        int idx=((b*H+h)*T+t);
        float mi=m[idx];
        float inv_l = 1.0f / lse[idx];
        float dp=dpSum[idx];

        float dotqk = Qs[qi][lane]*kd;
        dotqk = reduce_sum16(dotqk, mask);
        float score = dotqk*scale;

        float dotdv = dOs[qi][lane]*vd;
        dotdv = reduce_sum16(dotdv, mask);

        float p = expf(score - mi) * inv_l;
        float dS = (dotdv - dp) * p;

        dvd += p * dOs[qi][lane];
        dkd += dS * Qs[qi][lane] * scale;
      }
      __syncthreads();
    }

    float* dkptr = dK + (size_t)bts*(size_t)Dhf + (size_t)h*(size_t)Dh;
    float* dvptr = dV + (size_t)bts*(size_t)Dhf + (size_t)h*(size_t)Dh;
    dkptr[lane] += dkd;
    dvptr[lane] += dvd;
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

// ================= FUSED MEGA-KERNEL (Dhf=128) =================
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
  int warp = tid >> 5;       // 0..WARPS_PER_BLOCK-1
  int lane = tid & 31;       // 0..31
  int n = (int)blockIdx.x * WARPS_PER_BLOCK + warp;
  if(n >= N) return;

  int tpos = n % T;
  constexpr int COLS_PER_LANE = Dhf / 32; // 128/32 = 4
  int col_base = lane * COLS_PER_LANE;   // 0..124 step 4

  const float* Xn = X + (size_t)n*(size_t)Dhf;
  float* Qn = Q + (size_t)n*(size_t)Dhf;
  float* Kn = K + (size_t)n*(size_t)Dhf;
  float* Vn = Vv + (size_t)n*(size_t)Dhf;

  // Load X (float4) via L2->REG (cg)
  const float4* Xf4 = (const float4*)Xn;
  float4 x4 = ld_cg_f4(Xf4 + lane);

  // RMS sumsq
  float ss = x4.x*x4.x + x4.y*x4.y + x4.z*x4.z + x4.w*x4.w;
  #pragma unroll
  for(int off=16; off>0; off>>=1) ss += __shfl_down_sync(0xFFFFFFFFu, ss, off);
  float sumsq = __shfl_sync(0xFFFFFFFFu, ss, 0);

  float inv = rsqrtf(sumsq * (1.0f/(float)Dhf) + EPS);
  if(lane==0) rms_inv[n] = inv;

  // Apply gamma (safe scalar loads)
  float g0 = gamma[col_base+0];
  float g1 = gamma[col_base+1];
  float g2 = gamma[col_base+2];
  float g3 = gamma[col_base+3];

  float xn0 = x4.x * inv * g0;
  float xn1 = x4.y * inv * g1;
  float xn2 = x4.z * inv * g2;
  float xn3 = x4.w * inv * g3;

  // Optionally write normalized vector for backward Wq/Wk/Wv gradients
  if(Nnorm_or_null){
    float4* Nf4 = (float4*)(Nnorm_or_null + (size_t)n*(size_t)Dhf);
    Nf4[lane] = make_float4(xn0,xn1,xn2,xn3);
  }

  // Warp-broadcast x[k] for k=0..127
  float q0=0.f,q1=0.f,q2=0.f,q3=0.f;
  float k0=0.f,k1=0.f,k2=0.f,k3=0.f;
  float v0=0.f,v1=0.f,v2=0.f,v3=0.f;

  for(int k=0;k<Dhf;k++){
    int src_lane = k >> 2;      // 0..31
    int off = k & 3;            // 0..3
    float xk;
    if(off==0) xk = __shfl_sync(0xFFFFFFFFu, xn0, src_lane);
    else if(off==1) xk = __shfl_sync(0xFFFFFFFFu, xn1, src_lane);
    else if(off==2) xk = __shfl_sync(0xFFFFFFFFu, xn2, src_lane);
    else            xk = __shfl_sync(0xFFFFFFFFu, xn3, src_lane);

    const float4* wqf4 = (const float4*)(Wq + (size_t)k*(size_t)Dhf + (size_t)col_base);
    const float4* wkf4 = (const float4*)(Wk + (size_t)k*(size_t)Dhf + (size_t)col_base);
    const float4* wvf4 = (const float4*)(Wv + (size_t)k*(size_t)Dhf + (size_t)col_base);

    float4 wq4 = ld_cg_f4(wqf4);
    float4 wk4 = ld_cg_f4(wkf4);
    float4 wv4 = ld_cg_f4(wvf4);

    q0 = fmaf(xk, wq4.x, q0); q1 = fmaf(xk, wq4.y, q1); q2 = fmaf(xk, wq4.z, q2); q3 = fmaf(xk, wq4.w, q3);
    k0 = fmaf(xk, wk4.x, k0); k1 = fmaf(xk, wk4.y, k1); k2 = fmaf(xk, wk4.z, k2); k3 = fmaf(xk, wk4.w, k3);
    v0 = fmaf(xk, wv4.x, v0); v1 = fmaf(xk, wv4.y, v1); v2 = fmaf(xk, wv4.z, v2); v3 = fmaf(xk, wv4.w, v3);
  }

  // RoPE (pairs within head) on (col_base+0,col_base+1) and (col_base+2,col_base+3)
  // Dh=16 => Dh/2=8 per head
  auto rope_pair = [&](int d_even, float& a0, float& a1, float& b0, float& b1){
    int within = d_even & (Dh-1);     // 0..15
    int i2 = within >> 1;            // 0..7
    float s = sin_tbl[(size_t)tpos*(size_t)(Dh/2) + (size_t)i2];
    float c = cos_tbl[(size_t)tpos*(size_t)(Dh/2) + (size_t)i2];
    float qa=a0, qb=a1;
    float ka=b0, kb=b1;
    a0 = qa*c - qb*s;
    a1 = qa*s + qb*c;
    b0 = ka*c - kb*s;
    b1 = ka*s + kb*c;
  };
  rope_pair(col_base+0, q0,q1, k0,k1);
  rope_pair(col_base+2, q2,q3, k2,k3);

  // Store Q/K/V
  float4* Qf4 = (float4*)Qn;
  float4* Kf4 = (float4*)Kn;
  float4* Vf4 = (float4*)Vn;
  Qf4[lane] = make_float4(q0,q1,q2,q3);
  Kf4[lane] = make_float4(k0,k1,k2,k3);
  Vf4[lane] = make_float4(v0,v1,v2,v3);
}

// ================= GPU container =================
struct GPU {
  int dev;
  int B,T,N;

  float *dW,*dG,*mW,*vW;
  WView W,G,MW,VW;
  HW Hw;

  uint16_t *tok,*tgt;

  // pinned host staging for graph-captured H2D copies
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

  float *sin_tbl,*cos_tbl; // [Tmax, Dh/2]

  float *ring_tmp;

  cudaStream_t comm = nullptr;

  // CUDA Graph
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

  g->htok_h=nullptr; g->htgt_h=nullptr;
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

  g->ring_tmp=nullptr;
  g->comm=nullptr;
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

// ================= train step (device-only, capturable) =================
static void train_step_device(GPU* g){
  int B=g->B, T=g->T, N=g->N;
  float invN = 1.0f/(float)N;
  size_t Wn=weights_floats();
  zero_f<<<(int)((Wn+255)/256),256>>>(g->dG,(int)Wn); KERNEL_CHECK();

  dim3 blk2(16,16);
  dim3 grdE((D+15)/16,(N+15)/16);
  embed_split<<<grdE,blk2>>>(g->y1,g->y2,g->W.wte,g->W.wpe,g->tok,N,T); KERNEL_CHECK();

  // Forward
  for(int l=0;l<L;l++){
    // ===== FULL-BLAST FUSED STRIKE (replaces RMS + QKV WMMA + RoPE) =====
    // No need to write g->n in forward; pass nullptr.
    constexpr int WPB = 8;
    int blocks = (N + WPB - 1) / WPB;
    fused_rms_qkv_rope_hf<WPB><<<blocks,256>>>(g->Q, g->K, g->Vh, g->inv, (float*)nullptr,
                                              g->y2,
                                              g->W.Wq[l], g->W.Wk[l], g->W.Wv[l],
                                              g->W.gf[l],
                                              g->sin_tbl, g->cos_tbl,
                                              N, T);
    KERNEL_CHECK();

    dim3 grid(B,H,(T+FA_QT-1)/FA_QT);
    flash_fwd_hf<<<grid,256>>>(g->O, g->matt, g->latt, g->Q, g->K, g->Vh, B, T); KERNEL_CHECK();

    f2h<<<(N*Dhf+255)/256,256>>>(g->n_h, g->O, N*Dhf); KERNEL_CHECK();
    wmma_fwd(g->fout, g->n_h, g->Hw.Wo_tr[l], N, Dhf, Dhf);
    add_inplace<<<(N*Dhf+255)/256,256>>>(g->y1, g->fout, N*Dhf); KERNEL_CHECK();

    rms_fwd_f2h<Dhf><<<N,256>>>(g->n, g->n_h, g->inv, g->y1, g->W.gg[l], N); KERNEL_CHECK();
    wmma_fwd(g->U, g->n_h, g->Hw.W1_tr[l], N, F, Dhf);
    gelu_fwd<<<(N*F+255)/256,256>>>(g->A, g->U, N*F); KERNEL_CHECK();
    f2h<<<(N*F+255)/256,256>>>(g->A_h, g->A, N*F); KERNEL_CHECK();
    wmma_fwd(g->gout, g->A_h, g->Hw.W2_tr[l], N, Dhf, F);
    add_inplace<<<(N*Dhf+255)/256,256>>>(g->y2, g->gout, N*Dhf); KERNEL_CHECK();
  }

  // Head forward (streaming chunks)
  concat_full<<<dim3((D+15)/16,(N+15)/16),blk2>>>(g->Xfull, g->y1, g->y2, N); KERNEL_CHECK();
  rms_fwd_f2h<D><<<N,256>>>(g->Xnorm, g->Xnorm_h, g->invF, g->Xfull, g->W.gout, N); KERNEL_CHECK();

  init_row_stats<<<(N+255)/256,256>>>(g->row_max, g->row_sum, g->Loss, N); KERNEL_CHECK();
  for(int v0=0; v0<V; v0+=VCHUNK){
    int Mvalid = (v0+VCHUNK<=V)? VCHUNK : (V - v0);
    const half* Wtr = g->Hw.Wout_tr + (size_t)v0*(size_t)D;
    wmma_fwd(g->logits_chunk, g->Xnorm_h, Wtr, N, VCHUNK, D);
    chunk_max_sumexp<<<N,256>>>(g->chunk_max, g->chunk_sum, g->logits_chunk, N, Mvalid); KERNEL_CHECK();
    update_row_stats<<<(N+255)/256,256>>>(g->row_max, g->row_sum, g->chunk_max, g->chunk_sum, N); KERNEL_CHECK();
  }

  zero_f<<<(N*D+255)/256,256>>>(g->dXnorm, N*D); KERNEL_CHECK();
  for(int v0=0; v0<V; v0+=VCHUNK){
    int Mvalid = (v0+VCHUNK<=V)? VCHUNK : (V - v0);
    const half* Wtr = g->Hw.Wout_tr + (size_t)v0*(size_t)D;
    wmma_fwd(g->logits_chunk, g->Xnorm_h, Wtr, N, VCHUNK, D);

    dy_loss_from_logits<<<(N*VCHUNK+255)/256,256>>>(g->dY_chunk, g->Loss, g->logits_chunk,
                                                    g->row_max, g->row_sum, g->tgt, N, v0, Mvalid, invN);
    KERNEL_CHECK();

    wmma_dW(g->dWout_chunk, g->Atr, g->dYtr, g->Xnorm, g->dY_chunk, N, D, VCHUNK);
    scatter_add_dwout_chunk<<<(D*VCHUNK+255)/256,256>>>(g->G.Wout, g->dWout_chunk, v0); KERNEL_CHECK();

    pack_wout_rm_chunk<<<(D*VCHUNK+255)/256,256>>>(g->Wout_rm_chunk, g->Hw.Wout_rm, v0); KERNEL_CHECK();
    wmma_dA(g->dXfull, g->scratchHalf_head, g->dY_chunk, g->Wout_rm_chunk, N, VCHUNK, D);
    add_inplace<<<(N*D+255)/256,256>>>(g->dXnorm, g->dXfull, N*D); KERNEL_CHECK();
  }

  zero_f<<<(N*D+255)/256,256>>>(g->dXfull, N*D); KERNEL_CHECK();
  rms_bwd_dX<D><<<N,256>>>(g->dXfull, g->dXnorm, g->Xfull, g->W.gout, g->invF, N); KERNEL_CHECK();
  rms_bwd_dg<D><<<D,256>>>(g->G.gout, g->dXnorm, g->Xfull, g->invF, N); KERNEL_CHECK();
  split_full<<<dim3((D+15)/16,(N+15)/16),blk2>>>(g->dy1, g->dy2, g->dXfull, N); KERNEL_CHECK();

  // Reversible backward
  for(int l=L-1;l>=0;l--){
    // ---- g backward ----
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

    // ---- f backward ----
    // Recompute fused RMS+QKV+RoPE for x2, and WRITE g->n for dW(Q/K/V).
    constexpr int WPB = 8;
    int blocks = (N + WPB - 1) / WPB;
    fused_rms_qkv_rope_hf<WPB><<<blocks,256>>>(g->Q, g->K, g->Vh, g->inv, g->n,
                                              g->x2,
                                              g->W.Wq[l], g->W.Wk[l], g->W.Wv[l],
                                              g->W.gf[l],
                                              g->sin_tbl, g->cos_tbl,
                                              N, T);
    KERNEL_CHECK();

    dim3 grid(B,H,(T+FA_QT-1)/FA_QT);
    flash_fwd_hf<<<grid,256>>>(g->O, g->matt, g->latt, g->Q, g->K, g->Vh, B, T); KERNEL_CHECK();

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

    flash_bwd_dq_hf<<<dim3(B,H,(T+FA_QT-1)/FA_QT),256>>>(g->dp, g->dQ, g->dOattn, g->Q, g->K, g->Vh, g->matt, g->latt, B, T); KERNEL_CHECK();
    flash_bwd_dkv_hf<<<dim3(B,H,(T+FA_KT-1)/FA_KT),256>>>(g->dK, g->dVh, g->dOattn, g->Q, g->K, g->Vh, g->matt, g->latt, g->dp, B, T); KERNEL_CHECK();

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

  float loss=0.f;
  CUDA_CHECK(cudaMemcpy(&loss, g->loss_mean, sizeof(float), cudaMemcpyDeviceToHost));
  return loss;
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
  bool do_measure=false;
  bool do_continue=false;
  int use_graph=1;

  for(int i=1;i<argc;i++){
    if(!std::strcmp(argv[i],"--train")) do_train=true;
    else if(!std::strcmp(argv[i],"--continue")) do_continue=true;
    else if(!std::strcmp(argv[i],"--measure")) do_measure=true;
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
    else if(!std::strcmp(argv[i],"--graph") && i+1<argc) use_graph=std::atoi(argv[++i]);
    else if(!std::strcmp(argv[i],"--no_graph")) use_graph=0;
    else { std::fprintf(stderr,"Unknown arg: %s\n", argv[i]); return 2; }
  }
  if(!do_train) do_train=true;

  if((seq%16)!=0) die("--seq must be multiple of 16");
  if(seq<16 || seq>Tmax) die("--seq out of range");

  auto bytes=read_file_bytes(data_path);

  std::string actual_ckpt_path = ckpt_path;
  if (!do_continue) {
    int suffix = 1;
    while (true) {
      FILE* tst = std::fopen(actual_ckpt_path.c_str(), "rb");
      if (!tst) break;
      std::fclose(tst);
      actual_ckpt_path = std::string(ckpt_path) + "." + std::to_string(suffix++);
    }
  }
  ckpt_path = actual_ckpt_path.c_str();

  PairIndex pi;
  std::vector<float> winit;
  bool has_ckpt = load_ckpt(ckpt_path, &pi, &winit);

  if(!has_ckpt){
    if(!load_index_v7(index_path, &pi)) pi = make_pair_index(bytes);
  }

  std::vector<float> hostW(weights_floats());
  if(has_ckpt) hostW=winit;
  else init_weights_cpu(hostW, seed);

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

  std::printf("llm_engine: V=%d(Vpad=%d) K=%d D=%d Dh=%d L=%d H=%d F=%d T=%d gpus=%d batch=%d [",
              V,Vpad,PAIR_K,D,Dh,L,H,F,seq,G,batch);
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
