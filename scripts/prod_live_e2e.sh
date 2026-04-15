#!/usr/bin/env bash
set -euo pipefail

# Full production E2E on real hardware:
# - optional bootstrap (build + deps)
# - preflight
# - start native stack
# - verify health
# - run final prompt through router and require real retrieval+inference
#
# Usage:
#   scripts/prod_live_e2e.sh <hrm_model_dir> [final_prompt]
#
# Environment knobs:
#   RUN_BOOTSTRAP=1|0         default: 1
#   EXPECTED_GPUS=<int>       default: selected hardware pool size
#   TOPOLOGY_MODE             default: selected hardware derived topology
#   ROUTER_URL=<url>          default: http://127.0.0.1:8090
#   ROUTER_MODE               default: mixed (retrieval|mixed|deepseek_only)
#   RUN_MODE_MATRIX=1|0       default: 1 (1 = test mixed+retrieval+deepseek_only)
#   E2E_DETERMINISM_RUNS=<n>  default: 3 (same prompt repeated n times per mode)
#   ROUTER_ROUTE_HINT         default: balanced
#   ROUTER_MAX_NEW_TOKENS     default: 256
#   PROMPT_MIXED              default: [final_prompt arg]
#   PROMPT_RETRIEVAL          default: retrieval-specific prompt
#   PROMPT_DEEPSEEK_ONLY      default: deepseek-only prompt
#   EXPECT_SOLO_3080          default: auto (auto|0|1)
#   AUTO_STOP=1|0             default: 0
#   ALLOW_EMPTY_SOURCES=1|0   default: 0 (0 = require non-empty sources => real pipeline)
#   PREFLIGHT_SCRIPT          default: scripts/prod_preflight.sh
#   START_STACK_SCRIPT        default: scripts/start_native_stack.sh
#   STOP_STACK_SCRIPT         default: scripts/stop_native_stack.sh
#   HEALTH_URL_SOLO_22GB      default: http://127.0.0.1:8081/v1/health
#   HEALTH_URL_NVLINK_PAIR    default:
#                             - max_model_fast: alias to solo_22gb URL
#                             - hetero_3lane: http://127.0.0.1:8082/v1/health
#   HEALTH_URL_SOLO_3080      default: http://127.0.0.1:8083/v1/health

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <hrm_model_dir> [final_prompt]" >&2
  exit 2
fi

HRM_MODEL="$1"
FINAL_PROMPT="${2:-Bitte antworte auf Deutsch in 3 kurzen Bulletpoints: 1) Stack-Status 2) Kernaussage 3) Route-Hinweis.}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
HW_LIB="$ROOT_DIR/scripts/hw_profile_lib.sh"

if [[ ! -f "$HW_LIB" ]]; then
  echo "ERR: missing hardware profile helper: $HW_LIB" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$HW_LIB"
load_hw_selection_or_die
derive_hw_runtime_flags
print_hw_selection_summary

RUN_BOOTSTRAP="${RUN_BOOTSTRAP:-1}"
EXPECTED_GPUS="${EXPECTED_GPUS:-$HW_GPU_TOTAL}"
TOPOLOGY_MODE="${TOPOLOGY_MODE:-$HW_DERIVED_TOPOLOGY_MODE}"
ROUTER_URL="${ROUTER_URL:-http://127.0.0.1:8090}"
ROUTER_MODE="${ROUTER_MODE:-mixed}"
RUN_MODE_MATRIX="${RUN_MODE_MATRIX:-1}"
E2E_DETERMINISM_RUNS="${E2E_DETERMINISM_RUNS:-3}"
ROUTER_ROUTE_HINT="${ROUTER_ROUTE_HINT:-balanced}"
ROUTER_MAX_NEW_TOKENS="${ROUTER_MAX_NEW_TOKENS:-256}"
AUTO_STOP="${AUTO_STOP:-0}"
ALLOW_EMPTY_SOURCES="${ALLOW_EMPTY_SOURCES:-0}"
PROMPT_MIXED="${PROMPT_MIXED:-$FINAL_PROMPT}"
PROMPT_RETRIEVAL="${PROMPT_RETRIEVAL:-Nenne die wichtigste Aussage und gib mindestens eine Quellen-ID in eckigen Klammern an.}"
PROMPT_DEEPSEEK_ONLY="${PROMPT_DEEPSEEK_ONLY:-Explain in two short sentences the difference between retrieval and pure model knowledge.}"
EXPECT_SOLO_3080="${EXPECT_SOLO_3080:-auto}"

if [[ "$RUN_MODE_MATRIX" != "0" && "$RUN_MODE_MATRIX" != "1" ]]; then
  echo "ERR: RUN_MODE_MATRIX must be 0 or 1 (got: $RUN_MODE_MATRIX)" >&2
  exit 2
fi
if [[ -z "${E2E_DETERMINISM_RUNS:-}" || "$E2E_DETERMINISM_RUNS" -lt 1 ]]; then
  echo "ERR: E2E_DETERMINISM_RUNS must be >= 1 (got: $E2E_DETERMINISM_RUNS)" >&2
  exit 2
fi
if [[ "$RUN_MODE_MATRIX" == "0" ]]; then
  case "$ROUTER_MODE" in
    retrieval|mixed|deepseek_only) ;;
    *)
      echo "ERR: unsupported ROUTER_MODE=$ROUTER_MODE (supported: retrieval, mixed, deepseek_only)" >&2
      exit 2
      ;;
  esac
fi
if [[ "$EXPECT_SOLO_3080" == "auto" ]]; then
  EXPECT_SOLO_3080="$HW_DERIVED_ENABLE_SOLO_3080"
fi
case "$EXPECT_SOLO_3080" in
  0|1) ;;
  *)
    echo "ERR: EXPECT_SOLO_3080 must be auto, 0 or 1 (got: $EXPECT_SOLO_3080)" >&2
    exit 2
    ;;
esac

LOG_DIR="${LOG_DIR:-$ROOT_DIR/.run/services}"
PREFLIGHT_SCRIPT="${PREFLIGHT_SCRIPT:-$SCRIPT_DIR/prod_preflight.sh}"
START_STACK_SCRIPT="${START_STACK_SCRIPT:-$SCRIPT_DIR/start_native_stack.sh}"
STOP_STACK_SCRIPT="${STOP_STACK_SCRIPT:-$SCRIPT_DIR/stop_native_stack.sh}"
PORT_SOLO_22GB="${PORT_SOLO_22GB:-8081}"
if [[ "$TOPOLOGY_MODE" == "max_model_fast" || "$TOPOLOGY_MODE" == "single_lane" ]]; then
  PORT_NVLINK_PAIR="${PORT_NVLINK_PAIR:-$PORT_SOLO_22GB}"
else
  PORT_NVLINK_PAIR="${PORT_NVLINK_PAIR:-8082}"
fi
PORT_SOLO_3080="${PORT_SOLO_3080:-8083}"
HEALTH_URL_SOLO_22GB="${HEALTH_URL_SOLO_22GB:-http://127.0.0.1:${PORT_SOLO_22GB}/v1/health}"
HEALTH_URL_NVLINK_PAIR="${HEALTH_URL_NVLINK_PAIR:-http://127.0.0.1:${PORT_NVLINK_PAIR}/v1/health}"
HEALTH_URL_SOLO_3080="${HEALTH_URL_SOLO_3080:-http://127.0.0.1:${PORT_SOLO_3080}/v1/health}"

STACK_STARTED=0
cleanup() {
  if [[ "$AUTO_STOP" == "1" && "$STACK_STARTED" == "1" ]]; then
    echo "[cleanup] stopping native stack..."
    bash "$STOP_STACK_SCRIPT" || true
  fi
}
trap cleanup EXIT

require_cmd() {
  local c="$1"
  command -v "$c" >/dev/null 2>&1 || { echo "ERR: missing command: $c" >&2; exit 1; }
}

health_wait_ok() {
  local url="$1"
  local timeout_s="${2:-180}"
  local poll_s="${3:-2}"
  local waited=0
  while (( waited < timeout_s )); do
    local out
    out="$(curl -fsS --max-time 2 "$url" 2>/dev/null || true)"
    if [[ -n "$out" ]]; then
      if printf '%s' "$out" | python3 -c "import json,sys; d=json.load(sys.stdin); raise SystemExit(0 if bool(d.get('ok')) else 1)" >/dev/null 2>&1; then
        return 0
      fi
    fi
    sleep "$poll_s"
    waited=$((waited + poll_s))
  done
  return 1
}

check_backend_deepseek() {
  local name="$1"
  local url="$2"
  local out
  out="$(curl -fsS "$url")"
  printf '%s' "$out" | python3 -c "import json,sys; d=json.load(sys.stdin); \
ok=bool(d.get('ok')); b=d.get('backend'); r=bool(d.get('deepseek_running')); \
raise SystemExit(0 if ok and b=='deepseek_int8' and r else 1)" || {
    echo "ERR: backend health check failed for $name ($url)" >&2
    echo "$out" >&2
    exit 1
  }
  echo "[ok] $name deepseek_int8 running"
}

cd "$ROOT_DIR"
require_cmd bash
require_cmd curl
require_cmd python3
[[ -f "$PREFLIGHT_SCRIPT" ]] || { echo "ERR: missing preflight script: $PREFLIGHT_SCRIPT" >&2; exit 1; }
[[ -f "$START_STACK_SCRIPT" ]] || { echo "ERR: missing start script: $START_STACK_SCRIPT" >&2; exit 1; }
[[ -f "$STOP_STACK_SCRIPT" ]] || { echo "ERR: missing stop script: $STOP_STACK_SCRIPT" >&2; exit 1; }

echo "[1/5] bootstrap (RUN_BOOTSTRAP=$RUN_BOOTSTRAP)"
if [[ "$RUN_BOOTSTRAP" == "1" ]]; then
  require_cmd cmake
  require_cmd ctest
  PROFILE=deepseek_int8 bash "$SCRIPT_DIR/bootstrap.sh"
else
  echo "[skip] bootstrap skipped"
fi

echo "[2/5] production preflight"
bash "$PREFLIGHT_SCRIPT" "$HRM_MODEL" "$EXPECTED_GPUS"

echo "[3/5] start native stack"
export BACKEND=deepseek_int8
export PREPARE_MODELS=1
export TOPOLOGY_MODE="$TOPOLOGY_MODE"
bash "$START_STACK_SCRIPT" "$HRM_MODEL" auto
STACK_STARTED=1

echo "[4/5] verify service health"
health_wait_ok "$HEALTH_URL_SOLO_22GB" 180 2 || { echo "ERR: solo_22gb did not become healthy" >&2; exit 1; }
health_wait_ok "$HEALTH_URL_NVLINK_PAIR" 180 2 || { echo "ERR: nvlink_pair did not become healthy" >&2; exit 1; }
if [[ "$EXPECT_SOLO_3080" == "1" ]]; then
  health_wait_ok "$HEALTH_URL_SOLO_3080" 180 2 || { echo "ERR: solo_3080 did not become healthy" >&2; exit 1; }
else
  echo "[skip] solo_3080 health not required (EXPECT_SOLO_3080=0)"
fi
health_wait_ok "$ROUTER_URL/v1/health" 180 2 || { echo "ERR: router did not become healthy" >&2; exit 1; }

check_backend_deepseek "solo_22gb" "$HEALTH_URL_SOLO_22GB"
check_backend_deepseek "nvlink_pair" "$HEALTH_URL_NVLINK_PAIR"
if [[ "$EXPECT_SOLO_3080" == "1" ]]; then
  check_backend_deepseek "solo_3080" "$HEALTH_URL_SOLO_3080"
fi

echo "[5/5] mode verification + determinism checks"
ROUTER_URL="$ROUTER_URL" \
ROUTER_MODE="$ROUTER_MODE" \
RUN_MODE_MATRIX="$RUN_MODE_MATRIX" \
E2E_DETERMINISM_RUNS="$E2E_DETERMINISM_RUNS" \
ROUTER_ROUTE_HINT="$ROUTER_ROUTE_HINT" \
ROUTER_MAX_NEW_TOKENS="$ROUTER_MAX_NEW_TOKENS" \
ALLOW_EMPTY_SOURCES="$ALLOW_EMPTY_SOURCES" \
LOG_DIR="$LOG_DIR" \
PROMPT_MIXED="$PROMPT_MIXED" \
PROMPT_RETRIEVAL="$PROMPT_RETRIEVAL" \
PROMPT_DEEPSEEK_ONLY="$PROMPT_DEEPSEEK_ONLY" \
python3 - <<'PY'
import hashlib
import json
import os
import re
import time
import urllib.request
from pathlib import Path

router_url = os.environ["ROUTER_URL"].rstrip("/")
route_hint = os.environ["ROUTER_ROUTE_HINT"]
max_new_tokens = int(os.environ["ROUTER_MAX_NEW_TOKENS"])
allow_empty = os.environ.get("ALLOW_EMPTY_SOURCES", "0") == "1"
runs = int(os.environ.get("E2E_DETERMINISM_RUNS", "3"))
run_matrix = os.environ.get("RUN_MODE_MATRIX", "1") == "1"
single_mode = os.environ.get("ROUTER_MODE", "mixed")

if run_matrix:
    modes = ["mixed", "retrieval", "deepseek_only"]
else:
    modes = [single_mode]

prompts = {
    "mixed": os.environ["PROMPT_MIXED"],
    "retrieval": os.environ["PROMPT_RETRIEVAL"],
    "deepseek_only": os.environ["PROMPT_DEEPSEEK_ONLY"],
}

banned_mixed = [
    "according to the documents",
    "retrieved information",
    "laut den quellen",
    "aus den snippets",
    "basierend auf",
    "basierend auf den snippets",
    "aus den bereitgestellten texten",
    "based on the snippets",
    "based on",
    "according to the sources",
]

retrieval_ref_re = re.compile(r"\[s\d{4}\]", re.IGNORECASE)

def post_json(url: str, payload: dict) -> dict:
    raw = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=raw,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=240) as r:
        body = r.read()
    out = json.loads(body.decode("utf-8"))
    if not isinstance(out, dict):
        raise SystemExit("invalid router response shape")
    return out

def source_count_of(resp: dict) -> int:
    sc = resp.get("source_count")
    if isinstance(sc, int):
        return sc
    src = resp.get("sources")
    if isinstance(src, list):
        return len(src)
    return 0

def send_request(*, mode: str | None, prompt: str, show_sources: bool | None, max_new_tokens: int) -> tuple[dict, float]:
    payload = {
        "prompt": prompt,
        "route_hint": route_hint,
        "max_new_tokens": max_new_tokens,
        "allow_failover": True,
    }
    if mode is not None:
        payload["mode"] = mode
    if show_sources is not None:
        payload["show_sources"] = bool(show_sources)
    t0 = time.perf_counter()
    resp = post_json(f"{router_url}/v1/generate", payload)
    dt = time.perf_counter() - t0
    return resp, dt

def assert_quick(name: str, dt: float, limit_s: float = 3.0) -> None:
    if dt > limit_s:
        raise SystemExit(f"{name}: exceeded {limit_s:.1f}s ({dt:.3f}s)")

quick_max_new_tokens = min(max_new_tokens, 32)

print("[fp] 1) mixed-silent-test")
mixed_resp, mixed_dt = send_request(
    mode="mixed",
    prompt=prompts["mixed"],
    show_sources=False,
    max_new_tokens=quick_max_new_tokens,
)
assert_quick("mixed-silent-test", mixed_dt)
mixed_text = str(mixed_resp.get("text", "")).strip()
if not mixed_text:
    raise SystemExit("mixed-silent-test: empty response")
mixed_lower = mixed_text.lower()
for bad in ("quellen", "snippets", "laut den", "basierend auf"):
    if bad in mixed_lower:
        raise SystemExit(f"mixed-silent-test: forbidden phrase found: '{bad}'")

print("[fp] 2) deepseek_only-no-hrm-test")
deepseek_resp, deepseek_dt = send_request(
    mode="deepseek_only",
    prompt=prompts["deepseek_only"],
    show_sources=True,
    max_new_tokens=quick_max_new_tokens,
)
assert_quick("deepseek_only-no-hrm-test", deepseek_dt)
if source_count_of(deepseek_resp) != 0:
    raise SystemExit("deepseek_only-no-hrm-test: expected source_count=0")
if deepseek_resp.get("hrm_active", None) is not False:
    raise SystemExit("deepseek_only-no-hrm-test: expected hrm_active=false")
router_log = Path(os.environ.get("LOG_DIR", "")) / "router.log"
if router_log.is_file():
    router_log_txt = router_log.read_text(encoding="utf-8", errors="replace").lower()
    if "hrm disabled" not in router_log_txt:
        raise SystemExit("deepseek_only-no-hrm-test: router log missing 'HRM disabled' marker")

print("[fp] 3) determinism-test (mixed sources)")
mixed_sources_hash = None
for i in range(1, 4):
    det_resp, det_dt = send_request(
        mode="mixed",
        prompt=prompts["mixed"],
        show_sources=True,
        max_new_tokens=quick_max_new_tokens,
    )
    assert_quick(f"determinism-test run {i}", det_dt)
    sc = source_count_of(det_resp)
    if not allow_empty and sc <= 0:
        raise SystemExit("determinism-test: expected non-empty mixed sources")
    sources = det_resp.get("sources")
    if not isinstance(sources, list):
        sources = []
    src_fingerprint = hashlib.sha256(
        json.dumps(sources, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if mixed_sources_hash is None:
        mixed_sources_hash = src_fingerprint
    elif src_fingerprint != mixed_sources_hash:
        raise SystemExit(
            f"determinism-test: mixed sources changed (baseline={mixed_sources_hash}, got={src_fingerprint})"
        )

print("[fp] 4) retrieval-reference-test")
retrieval_resp, retrieval_dt = send_request(
    mode="retrieval",
    prompt=prompts["retrieval"],
    show_sources=True,
    max_new_tokens=quick_max_new_tokens,
)
assert_quick("retrieval-reference-test", retrieval_dt)
retrieval_text = str(retrieval_resp.get("text", "")).strip()
if not retrieval_text:
    raise SystemExit("retrieval-reference-test: empty response")
retrieval_lower = retrieval_text.lower()
if not (
    "laut den quellen" in retrieval_lower
    or "laut den" in retrieval_lower
    or "according to" in retrieval_lower
    or "quelle" in retrieval_lower
    or "source" in retrieval_lower
    or retrieval_ref_re.search(retrieval_text)
):
    raise SystemExit("retrieval-reference-test: no source reference phrase found")

print("[fp] 5) default-mode-test")
default_resp, default_dt = send_request(
    mode=None,
    prompt=prompts["mixed"],
    show_sources=False,
    max_new_tokens=quick_max_new_tokens,
)
assert_quick("default-mode-test", default_dt)
if str(default_resp.get("mode", "")).strip().lower() != "mixed":
    raise SystemExit("default-mode-test: default mode is not mixed")
default_text = str(default_resp.get("text", "")).strip().lower()
for bad in ("quellen", "snippets", "laut den", "basierend auf"):
    if bad in default_text:
        raise SystemExit(f"default-mode-test: forbidden phrase found: '{bad}'")

for mode in modes:
    prompt = prompts[mode]
    print(f"[mode] {mode} (runs={runs})")
    baseline_hash = None
    baseline_canon = None
    for i in range(1, runs + 1):
        payload = {
            "prompt": prompt,
            "mode": mode,
            "route_hint": route_hint,
            "max_new_tokens": max_new_tokens,
            "allow_failover": True,
        }
        resp = post_json(f"{router_url}/v1/generate", payload)
        if not bool(resp.get("ok", False)):
            raise SystemExit(f"{mode} run {i}: router response ok=false")
        text = str(resp.get("text", "")).strip()
        if not text:
            raise SystemExit(f"{mode} run {i}: empty response text")
        sc = source_count_of(resp)
        hrm_active = resp.get("hrm_active", None)
        lower = text.lower()

        if mode == "mixed":
            if not allow_empty and sc <= 0:
                raise SystemExit(
                    f"{mode} run {i}: source_count is 0 -> expected retrieval+inference path"
                )
            if any(p in lower for p in banned_mixed):
                raise SystemExit(
                    f"{mode} run {i}: mixed output leaked retrieval phrasing"
                )
            if "sources" in lower:
                raise SystemExit(
                    f"{mode} run {i}: mixed output mentioned sources"
                )
            if isinstance(resp.get("sources"), list) and len(resp["sources"]) > 0:
                raise SystemExit(
                    f"{mode} run {i}: mixed response exposed sources by default"
                )
            if hrm_active is not True:
                raise SystemExit(
                    f"{mode} run {i}: mixed must return hrm_active=true"
                )
        elif mode == "retrieval":
            if not allow_empty and sc <= 0:
                raise SystemExit(
                    f"{mode} run {i}: source_count is 0 -> retrieval mode must use sources"
                )
            if not allow_empty:
                src = resp.get("sources")
                if not isinstance(src, list) or len(src) == 0:
                    raise SystemExit(
                        f"{mode} run {i}: retrieval mode should expose sources by default"
                    )
            if not (
                retrieval_ref_re.search(text)
                or "quelle" in lower
                or "source" in lower
                or "according to" in lower
            ):
                raise SystemExit(
                    f"{mode} run {i}: retrieval output did not reference sources"
                )
            if hrm_active is not True:
                raise SystemExit(
                    f"{mode} run {i}: retrieval must return hrm_active=true"
                )
        elif mode == "deepseek_only":
            if sc != 0:
                raise SystemExit(
                    f"{mode} run {i}: deepseek_only must have source_count=0 (got {sc})"
                )
            if hrm_active is not False:
                raise SystemExit(
                    f"{mode} run {i}: deepseek_only must return hrm_active=false"
                )
            src = resp.get("sources")
            if isinstance(src, list) and len(src) > 0:
                raise SystemExit(
                    f"{mode} run {i}: deepseek_only must not return source payload"
                )
        else:
            raise SystemExit(f"unsupported mode in validator: {mode}")

        canon = json.dumps(
            {"text": text, "source_count": sc},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        h = hashlib.sha256(canon.encode("utf-8")).hexdigest()
        if baseline_hash is None:
            baseline_hash = h
            baseline_canon = canon
            route = resp.get("route") if isinstance(resp.get("route"), dict) else {}
            selected = route.get("selected", "?")
            prompt_tokens = route.get("prompt_tokens", "?")
            lat_ms = route.get("latency_ms", "?")
            print(f"[ok] {mode} run {i}: route={selected} prompt_tokens={prompt_tokens} latency_ms={lat_ms}")
        elif h != baseline_hash:
            raise SystemExit(
                f"{mode}: determinism failure between runs. baseline={baseline_hash} current={h}"
            )
    print(f"[deterministic] {mode} hash={baseline_hash}")
    text_preview = json.loads(baseline_canon)["text"]
    print(f"[text:{mode}] {text_preview}")
PY

echo
echo "E2E PASS: DeepSeek stack started and mode checks passed."
if [[ "$AUTO_STOP" == "0" ]]; then
  echo "Stack is still running. Stop with: bash scripts/stop_native_stack.sh"
fi
