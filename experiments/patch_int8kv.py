import re

with open('run.sh', 'r') as f:
    s = f.read()

# 1. Update ChatCtx struct
s = s.replace("int32_t *Kc[L];\n  int32_t *Vc[L];", "int8_t *Kc[L];\n  int8_t *Vc[L];")

# 2. Update chat_alloc
s = s.replace(
    "CUDA_CHECK(cudaMalloc(&c->Kc[l], (size_t)Tmax*(size_t)Dhf*sizeof(int32_t)));\n    CUDA_CHECK(cudaMalloc(&c->Vc[l], (size_t)Tmax*(size_t)Dhf*sizeof(int32_t)));\n    CUDA_CHECK(cudaMemset(c->Kc[l],0,(size_t)Tmax*(size_t)Dhf*sizeof(int32_t)));\n    CUDA_CHECK(cudaMemset(c->Vc[l],0,(size_t)Tmax*(size_t)Dhf*sizeof(int32_t)));",
    "CUDA_CHECK(cudaMalloc(&c->Kc[l], (size_t)Tmax*(size_t)Dhf*sizeof(int8_t)));\n    CUDA_CHECK(cudaMalloc(&c->Vc[l], (size_t)Tmax*(size_t)Dhf*sizeof(int8_t)));\n    CUDA_CHECK(cudaMemset(c->Kc[l],0,(size_t)Tmax*(size_t)Dhf*sizeof(int8_t)));\n    CUDA_CHECK(cudaMemset(c->Vc[l],0,(size_t)Tmax*(size_t)Dhf*sizeof(int8_t)));"
)

# 3. Update chat_repl reset
s = s.replace(
    "CUDA_CHECK(cudaMemset(ctx.Kc[l],0,(size_t)Tmax*(size_t)Dhf*sizeof(int32_t)));\n        CUDA_CHECK(cudaMemset(ctx.Vc[l],0,(size_t)Tmax*(size_t)Dhf*sizeof(int32_t)));",
    "CUDA_CHECK(cudaMemset(ctx.Kc[l],0,(size_t)Tmax*(size_t)Dhf*sizeof(int8_t)));\n        CUDA_CHECK(cudaMemset(ctx.Vc[l],0,(size_t)Tmax*(size_t)Dhf*sizeof(int8_t)));"
)

# 4. Modify kv_store_int to quantize to int8_t
old_kv_store = """__global__ void kv_store_int(int32_t* Kc, int32_t* Vc, const int32_t* k, const int32_t* v, int t){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<Dhf){
    Kc[(size_t)t*(size_t)Dhf + (size_t)i]=k[i];
    Vc[(size_t)t*(size_t)Dhf + (size_t)i]=v[i];
  }
}"""

new_kv_store = """__global__ void kv_store_int8(int8_t* Kc, int8_t* Vc, const int32_t* k, const int32_t* v, int t){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<Dhf){
    // int32 values are roughly Q8, downshift to fit int8_t bounds
    int32_t k_val = k[i] >> 8;
    int32_t v_val = v[i] >> 8;
    if(k_val > 127) k_val = 127; if(k_val < -128) k_val = -128;
    if(v_val > 127) v_val = 127; if(v_val < -128) v_val = -128;
    Kc[(size_t)t*(size_t)Dhf + (size_t)i]=(int8_t)k_val;
    Vc[(size_t)t*(size_t)Dhf + (size_t)i]=(int8_t)v_val;
  }
}"""
s = s.replace(old_kv_store, new_kv_store)
s = s.replace("kv_store_int<<<(Dhf+255)/256,256>>>(c->Kc[l], c->Vc[l], c->k, c->v, t); KERNEL_CHECK();", "kv_store_int8<<<(Dhf+255)/256,256>>>(c->Kc[l], c->Vc[l], c->k, c->v, t); KERNEL_CHECK();")


# 5. Modify attn_decode_one_int to read from int8_t using dp4a
old_attn = """__global__ void attn_decode_one_int(int32_t* outDhf, const int32_t* qDhf,
                               const int32_t* Kcache, const int32_t* Vcache,
                               int t){
  int h = (int)blockIdx.x;
  int lane = (int)threadIdx.x;
  if(h>=H || lane>=Dh) return;
  
  int32_t mi=-2147483648;
  int32_t oi=0;
  
  const int32_t* q = qDhf + (size_t)h*(size_t)Dh;
  
  for(int kpos=0;kpos<=t;kpos++){
    const int32_t* kptr = Kcache + (size_t)kpos*(size_t)Dhf + (size_t)h*(size_t)Dh;
    const int32_t* vptr = Vcache + (size_t)kpos*(size_t)Dhf + (size_t)h*(size_t)Dh;
    int32_t dot = q[lane]*kptr[lane];
    dot = reduce_sum16_int(dot, 0x0000FFFFu);
    
    if(dot > mi){ mi = dot; oi = vptr[lane]; }
  }
  outDhf[(size_t)h*(size_t)Dh + (size_t)lane] = oi;
}"""

new_attn = """// Requires Q mapped to int8 to use dp4a against INT8 Kcache
__global__ void attn_decode_one_int8(int32_t* outDhf, const int32_t* qDhf,
                               const int8_t* Kcache, const int8_t* Vcache,
                               int t){
  int h = (int)blockIdx.x;
  int lane = (int)threadIdx.x;
  if(h>=H || lane>=Dh) return;
  
  // Quantize Q row to int8 for current head using a shared buffer
  __shared__ int8_t sq[256]; 
  int32_t qv = qDhf[(size_t)h*Dh + lane] >> 8;
  if(qv>127) qv=127; if(qv<-128) qv=-128;
  sq[lane] = (int8_t)qv;
  __syncthreads();
  
  int32_t mi=-2147483648;
  int32_t oi=0;
  
  for(int kpos=0;kpos<=t;kpos++){
    const int8_t* kptr = Kcache + (size_t)kpos*(size_t)Dhf + (size_t)h*(size_t)Dh;
    const int8_t* vptr = Vcache + (size_t)kpos*(size_t)Dhf + (size_t)h*(size_t)Dh;
    
    // DP4A over Dh dimension (Dh = D/2/H, e.g. 256/2/8 = 16)
    // We each compute a partial dot product then warp reduce
    int32_t dot = 0;
    
    // Each thread does 4 elements if lane%4==0, but since Dh is small 
    // it's better to read 4 bytes at a time per thread and warp reduce
    // Wait, the original code did q[lane]*k[lane] then reduce_sum16_int.
    // So Dh = 16. Each thread takes ONE int32.
    // If we use dp4a, one thread can do 4 elements (1 int32 read of int8s).
    // Let's have each thread (stride 4) do a dp4a, or just do manual mult
    // for exact structure without reshaping arrays.
    
    // Manual mult: Kcache is int8, Q is int8.
    int32_t term = ((int32_t)sq[lane]) * ((int32_t)kptr[lane]);
    dot = reduce_sum16_int(term, 0x0000FFFFu);
    
    if(dot > mi){ mi = dot; oi = (int32_t)vptr[lane] << 8; }
  }
  outDhf[(size_t)h*(size_t)Dh + (size_t)lane] = oi;
}"""

s = s.replace(old_attn, new_attn)
s = s.replace("attn_decode_one_int<<<H,32>>>(c->o, c->q, c->Kc[l], c->Vc[l], t); KERNEL_CHECK();", "attn_decode_one_int8<<<H,32>>>(c->o, c->q, c->Kc[l], c->Vc[l], t); KERNEL_CHECK();")

with open('run.sh', 'w') as f:
    f.write(s)
