#!/usr/bin/env bash
set -euo pipefail

# Comparable DeepSeek benchmark for /v1/generate router path.
#
# Usage:
#   scripts/benchmark_deepseek.sh <router_base_url> [out_dir]
#
# Example:
#   BENCH_MODES=mixed BENCH_ROUTE_HINTS=balanced,quality BENCH_REPEATS=5 \
#   bash scripts/benchmark_deepseek.sh http://127.0.0.1:8090 ./.run/bench/deepseek
#
# Environment knobs:
#   BENCH_MODES               default: mixed,retrieval,deepseek_only
#   BENCH_ROUTE_HINTS         default: balanced,fast,quality
#   BENCH_SCENARIOS           default: short,medium,long
#   BENCH_REPEATS             default: 5
#   BENCH_WARMUP              default: 1
#   BENCH_MAX_NEW_TOKENS      default: 192
#   BENCH_TIMEOUT_S           default: 180
#   BENCH_ALLOW_FAILOVER      default: 1
#   BENCH_SHOW_SOURCES        default: 0
#   BENCH_TOKENIZER_MODEL     default: auto (auto|off|<tokenizer source>)
#   BENCH_CHARS_PER_TOKEN     default: 4.0
#   BENCH_PROMPT_SHORT        optional override
#   BENCH_PROMPT_MEDIUM       optional override
#   BENCH_PROMPT_LONG         optional override

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <router_base_url> [out_dir]" >&2
  exit 2
fi

ROUTER_URL="${1%/}"
OUT_DIR="${2:-./.run/benchmarks/deepseek_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_DIR"

ROUTER_URL="$ROUTER_URL" \
OUT_DIR="$OUT_DIR" \
BENCH_MODES="${BENCH_MODES:-mixed,retrieval,deepseek_only}" \
BENCH_ROUTE_HINTS="${BENCH_ROUTE_HINTS:-balanced,fast,quality}" \
BENCH_SCENARIOS="${BENCH_SCENARIOS:-short,medium,long}" \
BENCH_REPEATS="${BENCH_REPEATS:-5}" \
BENCH_WARMUP="${BENCH_WARMUP:-1}" \
BENCH_MAX_NEW_TOKENS="${BENCH_MAX_NEW_TOKENS:-192}" \
BENCH_TIMEOUT_S="${BENCH_TIMEOUT_S:-180}" \
BENCH_ALLOW_FAILOVER="${BENCH_ALLOW_FAILOVER:-1}" \
BENCH_SHOW_SOURCES="${BENCH_SHOW_SOURCES:-0}" \
BENCH_TOKENIZER_MODEL="${BENCH_TOKENIZER_MODEL:-auto}" \
BENCH_CHARS_PER_TOKEN="${BENCH_CHARS_PER_TOKEN:-4.0}" \
BENCH_PROMPT_SHORT="${BENCH_PROMPT_SHORT:-}" \
BENCH_PROMPT_MEDIUM="${BENCH_PROMPT_MEDIUM:-}" \
BENCH_PROMPT_LONG="${BENCH_PROMPT_LONG:-}" \
python3 - <<'PY'
from __future__ import annotations

import csv
import datetime as dt
import json
import math
import os
import statistics
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path


def split_csv(value: str) -> list[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


router_url = os.environ["ROUTER_URL"].rstrip("/")
out_dir = Path(os.environ["OUT_DIR"])
out_dir.mkdir(parents=True, exist_ok=True)

modes = split_csv(os.environ.get("BENCH_MODES", "mixed,retrieval,deepseek_only"))
hints = split_csv(os.environ.get("BENCH_ROUTE_HINTS", "balanced,fast,quality"))
scenarios = split_csv(os.environ.get("BENCH_SCENARIOS", "short,medium,long"))
repeats = int(os.environ.get("BENCH_REPEATS", "5"))
warmup = int(os.environ.get("BENCH_WARMUP", "1"))
max_new_tokens = int(os.environ.get("BENCH_MAX_NEW_TOKENS", "192"))
timeout_s = float(os.environ.get("BENCH_TIMEOUT_S", "180"))
allow_failover = os.environ.get("BENCH_ALLOW_FAILOVER", "1") == "1"
show_sources = os.environ.get("BENCH_SHOW_SOURCES", "0") == "1"
chars_per_token = max(1.0, float(os.environ.get("BENCH_CHARS_PER_TOKEN", "4.0")))
tokenizer_model = os.environ.get("BENCH_TOKENIZER_MODEL", "auto").strip()

if repeats < 1:
    raise SystemExit("ERR: BENCH_REPEATS must be >= 1")
if warmup < 0:
    raise SystemExit("ERR: BENCH_WARMUP must be >= 0")
if max_new_tokens < 1:
    raise SystemExit("ERR: BENCH_MAX_NEW_TOKENS must be >= 1")

prompt_short = os.environ.get("BENCH_PROMPT_SHORT") or (
    "Nenne in 3 Bulletpoints die wichtigsten Risiken ohne Monitoring bei LLM-Inferenz."
)
prompt_medium = os.environ.get("BENCH_PROMPT_MEDIUM") or (
    "Du betreibst einen Inferenz-Server fuer grosse Sprachmodelle. "
    "Erklaere kompakt: 1) typische p95-Latenz-Treiber, 2) Einfluss von Promptlaenge, "
    "3) welche zwei Metriken fuer stabile Produktion zuerst priorisiert werden sollten."
)
prompt_long = os.environ.get("BENCH_PROMPT_LONG") or (
    "Kontext: Wir benchmarken ein heterogenes 3-GPU-System mit deterministischem Retrieval. "
    "Erstelle eine strukturierte Antwort mit: "
    "A) Performance-Hypothesen, B) Messplan fuer reproduzierbare Runs, "
    "C) Kriterien fuer Go/No-Go in Produktion. "
    "Beziehe explizit Promptlaenge, Routing-Entscheidung, Queueing, TTFT und Durchsatz ein. "
    "Antwort knapp, aber technisch praezise."
)

prompt_map = {
    "short": prompt_short,
    "medium": prompt_medium,
    "long": prompt_long,
}
for sc in scenarios:
    if sc not in prompt_map:
        raise SystemExit(f"ERR: unsupported scenario '{sc}' (supported: short, medium, long)")


def http_json_post(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    out = json.loads(raw.decode("utf-8"))
    if not isinstance(out, dict):
        raise RuntimeError("invalid response payload shape")
    return out


def http_json_get(url: str, timeout: float) -> dict:
    req = urllib.request.Request(url=url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    out = json.loads(raw.decode("utf-8"))
    if not isinstance(out, dict):
        raise RuntimeError("invalid response payload shape")
    return out


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    k = (p / 100.0) * (len(xs) - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(xs[f])
    d0 = xs[f] * (c - k)
    d1 = xs[c] * (k - f)
    return float(d0 + d1)


tok_fn = None
tok_label = "chars"
resolved_tokenizer = ""
if tokenizer_model.lower() == "auto":
    try:
        health = http_json_get(router_url + "/v1/health", timeout=5.0)
        resolved_tokenizer = str((health.get("router") or {}).get("tokenizer") or "").strip()
    except Exception:
        resolved_tokenizer = ""
elif tokenizer_model.lower() == "off":
    resolved_tokenizer = ""
else:
    resolved_tokenizer = tokenizer_model

if resolved_tokenizer:
    try:
        from transformers import AutoTokenizer  # type: ignore

        tok = AutoTokenizer.from_pretrained(resolved_tokenizer, local_files_only=True)

        def tok_count(text: str) -> int:
            ids = tok(text, add_special_tokens=False).get("input_ids", [])
            return max(1, int(len(ids)))

        tok_fn = tok_count
        tok_label = f"tokenizer:{resolved_tokenizer}"
    except Exception:
        tok_fn = None
        tok_label = "chars"


def estimate_output_tokens(text: str) -> int:
    if tok_fn is not None:
        return tok_fn(text)
    return max(1, int(math.ceil(len(text) / chars_per_token)))


raw_rows: list[dict] = []
raw_csv = out_dir / "raw.csv"
summary_csv = out_dir / "summary.csv"
summary_txt = out_dir / "summary.txt"

print(f"[bench] router={router_url}")
print(f"[bench] modes={modes} hints={hints} scenarios={scenarios} repeats={repeats} warmup={warmup}")
print(f"[bench] tokenizer_estimator={tok_label}")

total_cases = len(scenarios) * len(modes) * len(hints)
case_idx = 0

for scenario in scenarios:
    prompt = prompt_map[scenario]
    for mode in modes:
        for hint in hints:
            case_idx += 1
            print(f"[bench] case {case_idx}/{total_cases}: scenario={scenario} mode={mode} route_hint={hint}")
            payload = {
                "prompt": prompt,
                "mode": mode,
                "route_hint": hint,
                "max_new_tokens": max_new_tokens,
                "allow_failover": allow_failover,
            }
            if show_sources:
                payload["show_sources"] = True

            for _ in range(warmup):
                _ = http_json_post(router_url + "/v1/generate", payload, timeout_s)

            for run in range(1, repeats + 1):
                t0 = time.perf_counter()
                resp = http_json_post(router_url + "/v1/generate", payload, timeout_s)
                dt_s = max(1e-9, time.perf_counter() - t0)
                latency_ms = dt_s * 1000.0

                text = str(resp.get("text", "") or "")
                out_tokens = estimate_output_tokens(text)
                tok_s = float(out_tokens) / dt_s
                route = resp.get("route") or {}
                selected = str(route.get("selected", ""))
                prompt_tokens = route.get("prompt_tokens", "")
                source_count = resp.get("source_count", "")
                hrm_active = resp.get("hrm_active", "")

                raw_rows.append(
                    {
                        "ts_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
                        "scenario": scenario,
                        "mode": mode,
                        "route_hint": hint,
                        "run": run,
                        "selected": selected,
                        "latency_ms": f"{latency_ms:.3f}",
                        "prompt_tokens": prompt_tokens,
                        "output_tokens": out_tokens,
                        "tokens_per_s": f"{tok_s:.3f}",
                        "text_chars": len(text),
                        "source_count": source_count,
                        "hrm_active": hrm_active,
                    }
                )
                print(
                    f"[bench] run {run}/{repeats} "
                    f"latency_ms={latency_ms:.1f} tok_s={tok_s:.2f} selected={selected or '?'}"
                )

if not raw_rows:
    raise SystemExit("ERR: benchmark produced no rows")

with raw_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(
        f,
        fieldnames=[
            "ts_utc",
            "scenario",
            "mode",
            "route_hint",
            "run",
            "selected",
            "latency_ms",
            "prompt_tokens",
            "output_tokens",
            "tokens_per_s",
            "text_chars",
            "source_count",
            "hrm_active",
        ],
    )
    w.writeheader()
    w.writerows(raw_rows)

group = defaultdict(list)
for r in raw_rows:
    group[(r["scenario"], r["mode"], r["route_hint"])].append(r)

summary_rows: list[dict] = []
for (scenario, mode, hint), rows in sorted(group.items()):
    lat = [float(x["latency_ms"]) for x in rows]
    tps = [float(x["tokens_per_s"]) for x in rows]
    out_toks = [float(x["output_tokens"]) for x in rows]
    selected = Counter(str(x["selected"]) for x in rows if str(x["selected"]))
    route_majority = selected.most_common(1)[0][0] if selected else ""

    summary_rows.append(
        {
            "scenario": scenario,
            "mode": mode,
            "route_hint": hint,
            "n": len(rows),
            "mean_latency_ms": f"{statistics.mean(lat):.3f}",
            "p50_latency_ms": f"{percentile(lat, 50):.3f}",
            "p95_latency_ms": f"{percentile(lat, 95):.3f}",
            "min_latency_ms": f"{min(lat):.3f}",
            "max_latency_ms": f"{max(lat):.3f}",
            "mean_tokens_per_s": f"{statistics.mean(tps):.3f}",
            "mean_output_tokens": f"{statistics.mean(out_toks):.3f}",
            "route_majority": route_majority,
        }
    )

with summary_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(
        f,
        fieldnames=[
            "scenario",
            "mode",
            "route_hint",
            "n",
            "mean_latency_ms",
            "p50_latency_ms",
            "p95_latency_ms",
            "min_latency_ms",
            "max_latency_ms",
            "mean_tokens_per_s",
            "mean_output_tokens",
            "route_majority",
        ],
    )
    w.writeheader()
    w.writerows(summary_rows)

scored = []
for row in summary_rows:
    mean_tps = float(row["mean_tokens_per_s"])
    p95 = max(1e-6, float(row["p95_latency_ms"]))
    score = mean_tps / p95
    scored.append((score, row))
scored.sort(key=lambda x: x[0], reverse=True)

with summary_txt.open("w", encoding="utf-8") as f:
    f.write("DeepSeek benchmark summary\n")
    f.write(f"router={router_url}\n")
    f.write(f"modes={','.join(modes)} hints={','.join(hints)} scenarios={','.join(scenarios)}\n")
    f.write(f"repeats={repeats} warmup={warmup} max_new_tokens={max_new_tokens}\n")
    f.write(f"token_estimator={tok_label}\n\n")
    f.write("Top 5 by score=mean_tokens_per_s/p95_latency_ms\n")
    f.write("score\tscenario\tmode\troute_hint\tmean_tps\tp95_ms\troute\n")
    for score, row in scored[:5]:
        f.write(
            f"{score:.6f}\t{row['scenario']}\t{row['mode']}\t{row['route_hint']}\t"
            f"{row['mean_tokens_per_s']}\t{row['p95_latency_ms']}\t{row['route_majority']}\n"
        )

print(f"[bench] wrote {raw_csv}")
print(f"[bench] wrote {summary_csv}")
print(f"[bench] wrote {summary_txt}")
PY

