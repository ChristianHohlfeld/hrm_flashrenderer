import re

with open('run.sh', 'r') as f:
    s = f.read()

# 1. Update ChatCtx to hold texture object
s = s.replace("int32_t *Kc[L];", "cudaTextureObject_t exp_tex;\n  int8_t *Kc[L];")

# 2. Update GPU struct to hold texture object
# GPU struct has `int32_t *sin_tbl,*cos_tbl; `
s = s.replace("int32_t *sin_tbl,*cos_tbl;", "int32_t *sin_tbl,*cos_tbl;\n  cudaTextureObject_t exp_tex;")

# 3. Alloc texture object in gpu_alloc
tex_alloc_code = """
  // Texture Alloc for exp LUT
  cudaResourceDesc resDesc = {};
  resDesc.resType = cudaResourceTypeLinear;
  
  // c_exp_lut is declared as __constant__ int32_t c_exp_lut[4097]
  // We cannot directly bind __constant__ to texture without getting its device pointer,
  // but it's easier to just copy it to a normal device array and bind that.
  int32_t* dev_exp_lut;
  CUDA_CHECK(cudaMalloc(&dev_exp_lut, 4097 * sizeof(int32_t)));
  CUDA_CHECK(cudaMemcpyFromSymbol(dev_exp_lut, c_exp_lut, 4097 * sizeof(int32_t)));
  
  resDesc.res.linear.devPtr = dev_exp_lut;
  resDesc.res.linear.desc.f = cudaChannelFormatKindSigned;
  resDesc.res.linear.desc.x = 32; // 32-bit int
  resDesc.res.linear.sizeInBytes = 4097 * sizeof(int32_t);

  cudaTextureDesc texDesc = {};
  texDesc.readMode = cudaReadModeElementType;

  CUDA_CHECK(cudaCreateTextureObject(&g->exp_tex, &resDesc, &texDesc, nullptr));
"""
s = s.replace("g->ring_tmp=nullptr;", g->ring_tmp=nullptr;\n" + tex_alloc_code)

# 4. Pass texture object to ChatCtx
s = s.replace("c->Hw = g0.Hw;", "c->Hw = g0.Hw;\n  c->exp_tex = g0.exp_tex;")

# 5. Modify chat_sample_kernel to accept texture object and use tex1Dfetch
func_sig_old = "__global__ void chat_sample_kernel(const int32_t* logits, int* out_tok, int32_t temp_q16, uint32_t rand_val, int V)"
func_sig_new = "__global__ void chat_sample_kernel(const int32_t* logits, int* out_tok, int32_t temp_q16, uint32_t rand_val, int V, cudaTextureObject_t exp_tex)"
s = s.replace(func_sig_old, func_sig_new)

# Replace exp_dx_q30 calls with tex1Dfetch inside chat_sample_kernel
# dx_q8 in [-4096,0] => map [-4096..0] to [0..4096]
tex_fetch = """
       int idx = (dx_q8 - (-4096));
       if (idx < 0) idx = 0; if (idx > 4096) idx = 4096;
       uint32_t p = (uint32_t)tex1Dfetch<int32_t>(exp_tex, idx);
"""
# in unnormalized probs logic:
s = re.sub(r'uint32_t p = \(uint32_t\)exp_dx_q30\(dx_q8\);.*?tsum \+= p;', tex_fetch.strip() + '\n    tsum += p;', s, count=1, flags=re.S)

# in cumsum logic:
s = re.sub(r'uint32_t p = \(uint32_t\)exp_dx_q30\(dx_q8\);.*?acc \+= p;', tex_fetch.strip() + '\n       acc += p;', s, count=1, flags=re.S)

# 6. Update kernel launch in chat_step
s = s.replace("chat_sample_kernel<<<1, 32>>>(c->logits, c->amaxi, temp_q16, rand_val, V);", "chat_sample_kernel<<<1, 32>>>(c->logits, c->amaxi, temp_q16, rand_val, V, c->exp_tex);")

with open('run.sh', 'w') as f:
    f.write(s)
