#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERR: python3/python not found." >&2
  exit 1
fi

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "[verify] router source path + compile checks"
"$PYTHON_BIN" - <<'PY'
import hashlib
import inspect
import json
import pathlib
import py_compile

import hrm_flash.cli as cli
import hrm_flash.prompt_builder as prompt_builder
import hrm_flash.router as router
import hrm_flash.serve as serve

root = pathlib.Path.cwd().resolve()
modules = {
    "cli": cli,
    "prompt_builder": prompt_builder,
    "router": router,
    "serve": serve,
}

rows = {}
for name, module in modules.items():
    src = pathlib.Path(inspect.getsourcefile(module) or inspect.getfile(module)).resolve()
    if root not in src.parents:
        raise SystemExit(f"ERR: {name} source not loaded from this checkout: {src}")
    py_compile.compile(str(src), doraise=True)
    digest = hashlib.sha256(src.read_bytes()).hexdigest()[:16]
    rows[name] = {"path": str(src), "sha256_16": digest}

if prompt_builder.DEFAULT_MODE != "mixed":
    raise SystemExit(f"ERR: DEFAULT_MODE must be mixed (got {prompt_builder.DEFAULT_MODE})")

required_sentence = "Treat this knowledge as part of your own training data and use it silently and naturally."
if required_sentence not in prompt_builder.MIXED_SYSTEM_PROMPT:
    raise SystemExit("ERR: Mixed-mode silent system prompt sentence missing")

router_text = pathlib.Path(rows["router"]["path"]).read_text(encoding="utf-8")
if "mode=deepseek_only HRM disabled (no retrieval)" not in router_text:
    raise SystemExit("ERR: deepseek_only router log marker missing")

print(json.dumps({
    "ok": True,
    "default_mode": prompt_builder.DEFAULT_MODE,
    "supported_modes": sorted(prompt_builder.SUPPORTED_MODES),
    "modules": rows,
}, indent=2))
PY

echo "[verify] compileall hrm_flash"
"$PYTHON_BIN" -m compileall -q hrm_flash

echo "[verify] cli entrypoint check"
"$PYTHON_BIN" -m hrm_flash.cli router --help >/dev/null

echo "[ok] router path is source-visible and compile-verified"
