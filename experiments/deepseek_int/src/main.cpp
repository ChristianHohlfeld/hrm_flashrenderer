// © 2026 Christian Heinrich Hohlfeld (Konstanz, Germany), christianhohlfeld.com, ORCID 0009-0003-6634-9045.
#include <iostream>
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>
#include "weights_loader.h"

extern "C" void launch_persistent_decode(
    int32_t* out_logits,
    const int8_t* w_qkv,
    const int8_t* w_out,
    int8_t* Kcache,
    int8_t* Vcache,
    const int32_t* scales,
    int t_start, int t_end);

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_q8.bin>" << std::endl;
        return 1;
    }

    std::cout << "[*] DeepSeek Int8 Persistent Decode MVP Engine" << std::endl;
    std::cout << "[*] Initializing Zero-Copy Loading for " << argv[1] << "..." << std::endl;

    try {
        WeightsLoader loader(argv[1]);
        
        // Example: load embedding and first layer structures
        TensorView embed = loader.get("model.embed_tokens.weight");
        TensorView q_proj = loader.get("model.layers.0.self_attn.q_proj.weight");
        TensorView k_proj = loader.get("model.layers.0.self_attn.k_proj.weight");
        TensorView v_proj = loader.get("model.layers.0.self_attn.v_proj.weight");
        TensorView o_proj = loader.get("model.layers.0.self_attn.o_proj.weight");
        TensorView lm_head = loader.get("lm_head.weight");

        std::cout << "Successfully mapped QKV and Vocab head to Host virtual memory space." << std::endl;
        
        // For actual MVP execution, we map the host memory directly to GPU device pointers or copy.
        // For extreme performance, memory mapped host memory can be pinned / cudaHostRegister'd
        // but for now we'll do an explicit move to D to satisfy the kernel execution.
        
        int8_t* d_w_qkv;
        int8_t* d_w_out;
        int32_t* d_out_logits;
        int8_t* d_Kcache;
        int8_t* d_Vcache;
        int32_t* d_scales;
        
        cudaMalloc(&d_w_qkv, q_proj.rows * q_proj.cols * sizeof(int8_t));
        cudaMemcpy(d_w_qkv, q_proj.data, q_proj.rows * q_proj.cols * sizeof(int8_t), cudaMemcpyHostToDevice);
        
        cudaMalloc(&d_w_out, lm_head.rows * lm_head.cols * sizeof(int8_t));
        cudaMemcpy(d_w_out, lm_head.data, lm_head.rows * lm_head.cols * sizeof(int8_t), cudaMemcpyHostToDevice);

        cudaMalloc(&d_out_logits, 4096 * sizeof(int32_t));
        cudaMalloc(&d_Kcache, 32 * 4096 * 128); // H * Tmax * Dh
        cudaMalloc(&d_Vcache, 32 * 4096 * 128);
        cudaMalloc(&d_scales, 32 * sizeof(int32_t));

        // Zero out
        cudaMemset(d_Kcache, 0, 32 * 4096 * 128);
        cudaMemset(d_Vcache, 0, 32 * 4096 * 128);
        
        // Push the Q12 scales up
        cudaMemcpy(d_scales, q_proj.scales_q12, 32 * sizeof(int32_t), cudaMemcpyHostToDevice);

        std::cout << "[*] Launching kernel..." << std::endl;
        launch_persistent_decode(d_out_logits, d_w_qkv, d_w_out, d_Kcache, d_Vcache, d_scales, 0, 10);
        
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
            return 1;
        }

        std::cout << "[*] Kernel executed successfully." << std::endl;

        cudaFree(d_out_logits);
        cudaFree(d_w_qkv);
        cudaFree(d_w_out);
        cudaFree(d_Kcache);
        cudaFree(d_Vcache);
        cudaFree(d_scales);

    } catch (const std::exception& e) {
        std::cerr << "[!] Loader Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
