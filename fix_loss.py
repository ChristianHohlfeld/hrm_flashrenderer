import sys

with open("experiments/run.sh", "r") as f:
    text = f.read()

bad_func = """__global__ void dy_loss_from_logits_int(int32_t* dY, int32_t* loss,
                                   const int32_t* logits,
                                   const int32_t* row_max, const int32_t* row_sum,
                                   const uint16_t* tgt,
                                   int N, int v0, int Mvalid, int32_t invN){
  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  int total=N*VCHUNK;
  if(idx>=total) return;
  int n = idx / VCHUNK;
  int j = idx - n*VCHUNK;
  if(j>=Mvalid){ dY[(size_t)idx]=0; return; }
  int v = v0 + j;
  if(v>=V){ dY[(size_t)idx]=0; return; }
  int y = (int)tgt[n];
  dY[(size_t)idx] = (v==y) ? -1 : 0;
  if(v==y) loss[n] = 100;
}"""

good_func = """__global__ void dy_loss_from_logits_int(int32_t* dY, int32_t* loss,
                                   const int32_t* logits,
                                   const int32_t* row_max, const int32_t* row_sum,
                                   const uint16_t* tgt,
                                   int N, int v0, int Mvalid, int32_t invN){
  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  int total=N*VCHUNK;
  if(idx>=total) return;
  int n = idx / VCHUNK;
  int j = idx - n*VCHUNK;
  if(j>=Mvalid){ dY[(size_t)idx]=0; return; }
  int v = v0 + j;
  if(v>=V){ dY[(size_t)idx]=0; return; }
  
  float lg = (float)logits[(size_t)idx];
  float rmax = (float)row_max[n];
  float rsum = (float)row_sum[n];
  float p = expf(lg - rmax) / (rsum + 1e-6f);
  
  int y = (int)tgt[n];
  if((unsigned)y >= (unsigned)V){
    if(v == v0){ loss[n] = 50; } 
    dY[(size_t)idx] = 0;
    return;
  }
  
  // Back to int32 (using scaled integer representation for gradient)
  // Assuming a scale factor of 1024 or similar downstream. 
  // Let's use 1024 to represent 1.0
  float grad = p - ((v==y)?1.f:0.f);
  dY[(size_t)idx] = (int32_t)(grad * 64.0f); // Adjust scale for INT engine
  
  if(v==y){
    float l = -((lg - rmax) - logf(rsum + 1e-6f));
    loss[n] = (int32_t)(l * 1024.0f); // Scaled loss value
  }
}"""

if bad_func in text:
    text = text.replace(bad_func, good_func)
    with open("experiments/run.sh", "w") as f:
        f.write(text)
    print("Replaced dummy loss function.")
else:
    print("Could not find the function to replace.")

