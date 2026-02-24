# Third-Party Components

This repository integrates third-party components that remain governed by their own licenses.
The proprietary "All Rights Reserved" notice in `LICENSE` covers **only** the original source code
authored by Christian Heinrich Hohlfeld (the HRM retrieval core, orchestration layer, and associated
documentation). It does **not** override or supersede the licenses of any third-party components listed below.

---

## Runtime Dependencies (Python)

| Package | License | Source |
|---|---|---|
| [PyTorch](https://github.com/pytorch/pytorch) | BSD 3-Clause | https://github.com/pytorch/pytorch/blob/main/LICENSE |
| [Transformers (Hugging Face)](https://github.com/huggingface/transformers) | Apache 2.0 | https://github.com/huggingface/transformers/blob/main/LICENSE |
| [safetensors](https://github.com/huggingface/safetensors) | Apache 2.0 | https://github.com/huggingface/safetensors/blob/main/LICENSE |
| [FastAPI](https://github.com/fastapi/fastapi) | MIT | https://github.com/fastapi/fastapi/blob/master/LICENSE |
| [uvicorn](https://github.com/encode/uvicorn) | BSD 2-Clause | https://github.com/encode/uvicorn/blob/master/LICENSE.md |
| [pydantic](https://github.com/pydantic/pydantic) | MIT | https://github.com/pydantic/pydantic/blob/main/LICENSE |
| [setuptools](https://github.com/pypa/setuptools) | MIT | https://github.com/pypa/setuptools/blob/main/LICENSE |
| [SQLite3](https://www.sqlite.org) | Public Domain | https://www.sqlite.org/copyright.html |

---

## Build & Compilation Dependencies

| Component | License | Notes |
|---|---|---|
| CUDA Toolkit (NVIDIA) | [CUDA EULA](https://docs.nvidia.com/cuda/eula/) | Required at build time for `flashattention_custom`. Not redistributed. |
| NVCC / WMMA intrinsics | NVIDIA proprietary | Used in `csrc/flash_attn_sm75.cu`. Subject to CUDA EULA. |
| CMake | BSD 3-Clause | Build system for `hrm_core` only. Not redistributed. |

---

## Custom CUDA Kernel (`csrc/`)

The CUDA kernels in `csrc/flash_attn_sm75.cu` and `csrc/ext.cpp` were written from scratch by
Christian Heinrich Hohlfeld and are subject to the proprietary license in `LICENSE`.
They use NVIDIA WMMA/CUDA APIs at compile time; those APIs are NVIDIA's intellectual property
and are governed by the CUDA EULA.

---

## Acknowledgement

Use of any third-party package from the table above must comply with its respective license.
Redistributing this repository does not grant any rights to third-party components beyond what
their original licenses allow.
