# © 2026 Christian Heinrich Hohlfeld (Konstanz, Germany), christianhohlfeld.com, ORCID 0009-0003-6634-9045.
cmake_minimum_required(VERSION 3.18)
project(DeepSeekInt8 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES "75") # RTX 2080 Ti

# Optimize for performance
set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "-O3 --use_fast_math -Xptxas -O3,-v")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# Source files
set(SOURCES
    src/main.cpp
    src/engine.cu
)

add_executable(deepseek_engine ${SOURCES})
