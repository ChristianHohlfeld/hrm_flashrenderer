#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cmake -S "$ROOT_DIR/hrm_core" -B "$ROOT_DIR/hrm_core/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$ROOT_DIR/hrm_core/build" -j
