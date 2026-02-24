# Copyright (c) 2026 Christian Heinrich Hohlfeld (Konstanz, Germany)
# https://christianhohlfeld.com | ORCID: https://orcid.org/0009-0003-6634-9045
# ALL RIGHTS RESERVED. No license granted without prior written permission.
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="hrm_flashrenderer",
    version="5.1.0",
    author="Christian Heinrich Hohlfeld",
    author_email="contact@christianhohlfeld.com",
    license="Proprietary",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="flashattention_custom._ext",
            sources=["csrc/ext.cpp", "csrc/flash_attn_sm75.cu"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-lineinfo",
                    "-std=c++17",
                    "-arch=sm_75",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    entry_points={
        "console_scripts": [
            "hrm-flash=hrm_flash.cli:main",
            "hrm-flashd=hrm_flash.flash_daemon:main",
            "hrm-flash-serve=hrm_flash.serve:main",
            "flash-kernel-test=flash_tests.test_kernel:main",
            "flash-append-test=flash_tests.test_append_fast:main",
        ]
    },
)

