import sys

with open("experiments/run.sh", "r") as f:
    text = f.read()

bad_chunk = """__global__ void chunk_max_sumexp_int(int32_t* cmax, int32_t* csum, const int32_t* logits, int N, int Mvalid){
  int n = blockIdx.x;
  if(n>=N) return;
  __shared__ int32_t buf[256];
  int32_t mx=-2147483648;
  for(int j=threadIdx.x;j<Mvalid;j+=blockDim.x){
    int32_t v = logits[(size_t)n*(size_t)VCHUNK + (size_t)j];
    if(v>mx) mx=v;
  }
  buf[threadIdx.x]=mx; __syncthreads();
  for(int k=blockDim.x/2;k>0;k>>=1){
    if(threadIdx.x<k){
      int32_t a=buf[threadIdx.x], b=buf[threadIdx.x+k];
      buf[threadIdx.x]=(a>b)?a:b;
    }
    __syncthreads();
  }
  if(threadIdx.x==0){ cmax[n]=buf[0]; csum[n]=1; }
}"""

good_chunk = """__global__ void chunk_max_sumexp_int(int32_t* cmax, int32_t* csum, const int32_t* logits, int N, int Mvalid){
  int n = blockIdx.x;
  if(n>=N) return;
  __shared__ float buf[256];
  int32_t mx=-2147483648;
  for(int j=threadIdx.x;j<Mvalid;j+=blockDim.x){
    int32_t v = logits[(size_t)n*(size_t)VCHUNK + (size_t)j];
    if(v>mx) mx=v;
  }
  buf[threadIdx.x]=(float)mx; __syncthreads();
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
    s += expf((float)logits[(size_t)n*(size_t)VCHUNK + (size_t)j] - m);
  }
  buf[threadIdx.x]=s; __syncthreads();
  for(int k=blockDim.x/2;k>0;k>>=1){ if(threadIdx.x<k) buf[threadIdx.x]+=buf[threadIdx.x+k]; __syncthreads(); }
  if(threadIdx.x==0){ cmax[n]=(int32_t)m; csum[n]=*((int32_t*)&buf[0]); } // store float as int bits using float representation
}"""

bad_update = """__global__ void update_row_stats_int(int32_t* row_max, int32_t* row_sum, const int32_t* cmax, const int32_t* csum, int N){
  int n=blockIdx.x*blockDim.x+threadIdx.x;
  if(n>=N) return;
  if(cmax[n] > row_max[n]) { row_max[n] = cmax[n]; }
}"""

good_update = """__global__ void update_row_stats_int(int32_t* row_max, int32_t* row_sum, const int32_t* cmax, const int32_t* csum_bits, int N){
  int n=blockIdx.x*blockDim.x+threadIdx.x;
  if(n>=N) return;
  float rm=(float)row_max[n];
  float rs=*((float*)&row_sum[n]);
  float cm=(float)cmax[n];
  int32_t cbits=csum_bits[n];
  float cs=*((float*)&cbits);
  
  float nm = (rm>cm)?rm:cm;
  float rs_new = rs*expf(rm-nm) + cs*expf(cm-nm);
  row_max[n]=(int32_t)nm;
  row_sum[n]=*((int32_t*)&rs_new); // store float bits
}"""

if bad_chunk in text and bad_update in text:
    text = text.replace(bad_chunk, good_chunk).replace(bad_update, good_update)
    with open("experiments/run.sh", "w") as f:
        f.write(text)
    print("Replaced stats kernels.")
else:
    print("Could not find the kernels to replace.")

