#!/usr/bin/env bash
set -euo pipefail
cmake -S hrm_core -B hrm_core/build -DCMAKE_BUILD_TYPE=Release
cmake --build hrm_core/build -j
