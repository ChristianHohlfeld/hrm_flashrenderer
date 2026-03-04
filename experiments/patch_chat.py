import re, sys

with open('run.sh', 'r') as f:
    code = f.read()

# 1. Add chat sampling kernel
sampling_kernel = """
__global__ void chat_sample_kernel(const int32_t* logits, int* out_tok, int32_t temp_q16, uint32_t rand_val, int V) {
  int tid = threadIdx.x;
  
  // Warp 0 max reduction
  __shared__ int32_t smax[32];
  int32_t tmax = -2147483648;
  for (int i = tid; i < V; i += blockDim.x) {
    if (logits[i] > tmax) tmax = logits[i];
  }
  
  for (int off = 16; off > 0; off >>= 1) {
    int32_t v = __shfl_xor_sync(0xFFFFFFFFu, tmax, off);
    if (v > tmax) tmax = v;
  }
  if (tid == 0) smax[0] = tmax;
  __syncthreads();
  
  int32_t m = smax[0];
  
  // Calculate unnormalized probs and sum
  __shared__ uint64_t ssum[32];
  uint64_t tsum = 0;
  for (int i = tid; i < V; i += blockDim.x) {
    int32_t dx = logits[i] - m;
    // apply temp_q16: dx / (temp/65536) = dx * 65536 / temp
    if (temp_q16 > 0 && temp_q16 != 65536) {
       dx = (int32_t)(((int64_t)dx * 65536) / temp_q16); 
    }
    int32_t dx_q8 = dx << (DX_Q - SM_SHIFT); 
    if (dx_q8 < DX_MIN_Q) dx_q8 = DX_MIN_Q;
    if (dx_q8 > 0) dx_q8 = 0;
    
    uint32_t p = (uint32_t)exp_dx_q30(dx_q8);
    tsum += p;
  }
  
  for (int off = 16; off > 0; off >>= 1) {
    tsum += __shfl_xor_sync(0xFFFFFFFFu, tsum, off);
  }
  if (tid == 0) ssum[0] = tsum;
  __syncthreads();
  
  uint64_t total_sum = ssum[0];
  if(total_sum==0) total_sum=1;
  
  // rand_val is 32-bit uint. threshold = (rand_val / 2^32) * total_sum
  uint64_t threshold = ((uint64_t)rand_val * total_sum) >> 32;
  
  // Cumsum to find token
  __shared__ uint64_t cumsum[1];
  __shared__ int selected[1];
  if (tid == 0) { cumsum[0] = 0; selected[0] = V - 1; }
  __syncthreads();
  
  // strictly sequential accumulation over V (V is small enough, ~16k-32k)
  if (tid == 0) {
    uint64_t acc = 0;
    for (int i = 0; i < V; ++i) {
       int32_t dx = logits[i] - m;
       if (temp_q16 > 0 && temp_q16 != 65536) dx = (int32_t)(((int64_t)dx * 65536) / temp_q16);
       int32_t dx_q8 = dx << (DX_Q - SM_SHIFT); 
       if (dx_q8 < DX_MIN_Q) dx_q8 = DX_MIN_Q;
       if (dx_q8 > 0) dx_q8 = 0;
       
       uint32_t p = (uint32_t)exp_dx_q30(dx_q8);
       acc += p;
       if (acc > threshold) {
         selected[0] = i;
         break;
       }
    }
    out_tok[0] = selected[0];
  }
}
"""

code = code.replace("static int chat_step(ChatCtx* c, int t, int tok_id){", sampling_kernel + "\nstatic int chat_step(ChatCtx* c, int t, int tok_id, int32_t temp_q16, uint32_t rand_val){")

code = code.replace("int next=0;\n  CUDA_CHECK(cudaMemcpy(&next, c->amaxi, sizeof(int), cudaMemcpyDeviceToHost));", """
  // Replaced greedy argmax with fast sample if temp_q16 > 0
  int next=0;
  if(temp_q16 <= 0){
    CUDA_CHECK(cudaMemcpy(&next, c->amaxi, sizeof(int), cudaMemcpyDeviceToHost));
  } else {
    chat_sample_kernel<<<1, 32>>>(c->logits, c->amaxi, temp_q16, rand_val, V);
    KERNEL_CHECK();
    CUDA_CHECK(cudaMemcpy(&next, c->amaxi, sizeof(int), cudaMemcpyDeviceToHost));
  }
""")

code = code.replace("""next = chat_step(&ctx, t, (int)id);""", """next = chat_step(&ctx, t, (int)id, 0, 0);""")
code = code.replace("""next = chat_step(&ctx, t, (int)ids[i]);""", """next = chat_step(&ctx, t, (int)ids[i], 0, 0);""")

code = code.replace("chat_repl(const PairIndex& pi, const std::vector<int32_t>& hostW, const char* ckpt_path, const char* prompt, bool do_measure)", "chat_repl(const PairIndex& pi, const std::vector<int32_t>& hostW, const char* ckpt_path, const char* prompt, bool do_measure, int32_t temp_q16, uint64_t seed)")

code = code.replace("""std::fprintf(stderr,"[chat] commands: /reset /quit  (greedy decode, incremental KV, generates up to 200 tokens)\\n");""", """std::fprintf(stderr,"[chat] commands: /reset /quit  (Sampling temp_q16=%d, incremental KV, generates up to 200 tokens)\\n", temp_q16);""")

code = re.sub(r'next = chat_step\(&ctx, t, \(int\)out_id\);', r'''
      // Generate pseudo-random value for sampling
      seed ^= seed >> 12; seed ^= seed << 25; seed ^= seed >> 27; uint64_t mult = seed * 2685821657736338717ULL;
      uint32_t rand_val = (uint32_t)(mult >> 32);
      next = chat_step(&ctx, t, (int)out_id, temp_q16, rand_val);
''', code)

code = code.replace("int32_t lr=5, wd=1, clip=5;", "int32_t lr=5, wd=1, clip=5;\n  int32_t temp_q16=0; // 0 = greedy")
code = code.replace("""else if(!std::strcmp(argv[i],"--chat_prompt") && i+1<argc) chat_prompt=argv[++i];""", """else if(!std::strcmp(argv[i],"--chat_prompt") && i+1<argc) chat_prompt=argv[++i];\n    else if(!std::strcmp(argv[i],"--temp") && i+1<argc) temp_q16=(int32_t)(std::atof(argv[++i]) * 65536.0);""")
code = code.replace("chat_repl(pi, hostW, ckpt_path, chat_prompt, do_measure);", "chat_repl(pi, hostW, ckpt_path, chat_prompt, do_measure, temp_q16, seed);")

with open('run.sh', 'w') as f:
    f.write(code)
