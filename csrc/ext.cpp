#include <torch/extension.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

torch::Tensor flash_attn_fwd_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal, int64_t q_offset, int64_t k_offset);

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
);

void paged_kv_append_cuda(
    torch::Tensor k_pages, torch::Tensor v_pages,
    torch::Tensor k_new, torch::Tensor v_new,
    int64_t start_pos,
    int64_t page_size
);

static torch::Tensor flash_attn_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal, int64_t q_offset, int64_t k_offset) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q/k/v must be CUDA");
    TORCH_CHECK(q.dtype() == torch::kFloat16 && k.dtype() == torch::kFloat16 && v.dtype() == torch::kFloat16, "fp16 required");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "contiguous required");
    TORCH_CHECK(q.dim()==4 && k.dim()==4 && v.dim()==4, "q/k/v must be 4D");
    TORCH_CHECK(q.size(0)==k.size(0) && q.size(0)==v.size(0), "B mismatch");
    TORCH_CHECK(q.size(1)==k.size(1) && q.size(1)==v.size(1), "H mismatch");
    TORCH_CHECK(k.size(2)==v.size(2), "N mismatch");
    TORCH_CHECK(q.size(3)==k.size(3) && q.size(3)==v.size(3), "D mismatch");
    return flash_attn_fwd_cuda(q,k,v,causal,q_offset,k_offset);
}

static torch::Tensor flash_attn_paged_fwd(
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
    TORCH_CHECK(q.is_cuda() && k_pages.is_cuda() && v_pages.is_cuda(), "CUDA required");
    TORCH_CHECK(seqlen.is_cuda() && head_map.is_cuda(), "CUDA required");
    TORCH_CHECK(q.dtype()==torch::kFloat16 && k_pages.dtype()==torch::kFloat16 && v_pages.dtype()==torch::kFloat16, "fp16 required");
    TORCH_CHECK(seqlen.dtype()==torch::kInt32, "seqlen int32 required");
    TORCH_CHECK(head_map.dtype()==torch::kInt32, "head_map int32 required");
    TORCH_CHECK(q.is_contiguous() && k_pages.is_contiguous() && v_pages.is_contiguous(), "contiguous required");
    TORCH_CHECK(seqlen.is_contiguous() && head_map.is_contiguous(), "contiguous required");
    TORCH_CHECK(q.dim()==4, "q [B,Hq,M,D]");
    TORCH_CHECK(k_pages.dim()==5 && v_pages.dim()==5, "k_pages/v_pages [B,Hk,P,PS,D]");
    TORCH_CHECK(k_pages.sizes()==v_pages.sizes(), "k_pages/v_pages mismatch");
    TORCH_CHECK(q.size(0)==k_pages.size(0), "B mismatch");
    TORCH_CHECK(q.size(1)==head_map.size(0), "Hq mismatch");
    TORCH_CHECK(k_pages.size(3)==page_size, "page_size mismatch");
    TORCH_CHECK(k_pages.size(4)==q.size(3), "D mismatch");
    return flash_attn_paged_fwd_cuda(q,k_pages,v_pages,seqlen,page_size,head_map,causal,q_offset,k_offset);
}

static void paged_kv_append(
    torch::Tensor k_pages,
    torch::Tensor v_pages,
    torch::Tensor k_new,
    torch::Tensor v_new,
    int64_t start_pos,
    int64_t page_size
){
    TORCH_CHECK(k_pages.is_cuda() && v_pages.is_cuda() && k_new.is_cuda() && v_new.is_cuda(), "CUDA required");
    TORCH_CHECK(k_pages.dtype()==torch::kFloat16 && v_pages.dtype()==torch::kFloat16 && k_new.dtype()==torch::kFloat16 && v_new.dtype()==torch::kFloat16, "fp16 required");
    TORCH_CHECK(k_pages.is_contiguous() && v_pages.is_contiguous() && k_new.is_contiguous() && v_new.is_contiguous(), "contiguous required");
    TORCH_CHECK(k_pages.dim()==5 && k_new.dim()==4, "shape mismatch");
    TORCH_CHECK(k_pages.sizes()==v_pages.sizes(), "k_pages/v_pages mismatch");
    TORCH_CHECK(k_new.sizes()==v_new.sizes(), "k_new/v_new mismatch");
    TORCH_CHECK(k_pages.size(0)==k_new.size(0), "B mismatch");
    TORCH_CHECK(k_pages.size(1)==k_new.size(1), "Hk mismatch");
    TORCH_CHECK(k_pages.size(3)==page_size, "page_size mismatch");
    TORCH_CHECK(k_pages.size(4)==k_new.size(3), "D mismatch");
    paged_kv_append_cuda(k_pages, v_pages, k_new, v_new, start_pos, page_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_fwd", &flash_attn_fwd,
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("causal") = true,
          py::arg("q_offset") = 0,
          py::arg("k_offset") = 0);

    m.def("flash_attn_paged_fwd", &flash_attn_paged_fwd,
          py::arg("q"),
          py::arg("k_pages"),
          py::arg("v_pages"),
          py::arg("seqlen"),
          py::arg("page_size"),
          py::arg("head_map"),
          py::arg("causal") = true,
          py::arg("q_offset") = 0,
          py::arg("k_offset") = 0);

    m.def("paged_kv_append", &paged_kv_append,
          py::arg("k_pages"),
          py::arg("v_pages"),
          py::arg("k_new"),
          py::arg("v_new"),
          py::arg("start_pos"),
          py::arg("page_size"));
}
