import sys

with open("experiments/run.sh", "r") as f:
    text = f.read()

bad_func2 = """__global__ void dy_loss_from_logits_int(int32_t* dY, int32_t* loss,
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
  float rsum = (float)row_sum[n];"""

good_func2 = """__global__ void dy_loss_from_logits_int(int32_t* dY, int32_t* loss,
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
  int32_t sbits=row_sum[n];
  float rsum = *((float*)&sbits);"""

if bad_func2 in text:
    text = text.replace(bad_func2, good_func2)
    with open("experiments/run.sh", "w") as f:
        f.write(text)
    print("Replaced row_sum bits conversion.")
else:
    print("Could not find the function piece to replace.")

