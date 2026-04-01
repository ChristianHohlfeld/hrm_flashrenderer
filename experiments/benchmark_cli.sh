#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PRESET="decision"
REPEATS=2
STEPS=80
LOG_EVERY=10
SAVE_EVERY=1000
GPUS=1
LR_LIST="0.0001,0.0003,0.001"
INCLUDE_BEAST=0
OUT_DIR="$ROOT_DIR/bench_cli"

usage() {
cat <<'EOF'
benchmark_cli.sh - unified benchmark runner

Usage:
  ./benchmark_cli.sh [options]

Options:
  -h, --help                 Show this help
  --preset <name>            matrix | decision | large   (default: decision)
  --repeats <n>              Repeats per config          (default: 2)
  --steps <n>                Training steps per run       (default: 80)
  --log-every <n>            Log interval                 (default: 10)
  --save-every <n>           Save interval                (default: 1000)
  --gpus <n>                 GPU count                    (default: 1)
  --lr-list <a,b,c>          Learning rates               (default: 0.0001,0.0003,0.001)
  --include-beast <0|1>      Include runbeast.sh          (default: 0)
  --out <dir>                Output directory             (default: ./bench_cli)

Examples:
  ./benchmark_cli.sh --preset matrix --steps 60 --repeats 1
  ./benchmark_cli.sh --preset decision --lr-list 0.0003 --repeats 3
  ./benchmark_cli.sh --preset large --steps 120 --repeats 2 --include-beast 1
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --preset) PRESET="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --log-every) LOG_EVERY="$2"; shift 2 ;;
    --save-every) SAVE_EVERY="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --lr-list) LR_LIST="$2"; shift 2 ;;
    --include-beast) INCLUDE_BEAST="$2"; shift 2 ;;
    --out) OUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; usage; exit 2 ;;
  esac
done

mkdir -p "$OUT_DIR/corpora" "$OUT_DIR/logs"
CSV="$OUT_DIR/raw.csv"
echo 'runner,scenario,lr,rep,tok_s,elapsed_sec' > "$CSV"

normalize_lr() {
  python3 - "$1" <<'PY'
import sys
x=float(sys.argv[1])
print(f"{x:.10f}".rstrip('0').rstrip('.'))
PY
}

# corpora
[[ -s "$OUT_DIR/corpora/shakespeare.txt" ]] || curl -fsSL -o "$OUT_DIR/corpora/shakespeare.txt" https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
cat > "$OUT_DIR/corpora/code.txt" <<'TXT'
#include <iostream>
int main(){ std::cout<<"hello"; }
def fib(n):
 a,b=0,1
 for _ in range(n): a,b=b,a+b
TXT
cat > "$OUT_DIR/corpora/german.txt" <<'TXT'
Dies ist ein kurzer deutscher Testkorpus für reproduzierbare Benchmarks.
Wir wollen verschiedene Skripte unter identischen Bedingungen vergleichen.
TXT
[[ -s "$OUT_DIR/corpora/wikitext2_train.txt" ]] || curl -fsSL -o "$OUT_DIR/corpora/wikitext2_train.txt" https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt
{
  for _ in $(seq 1 6); do cat "$OUT_DIR/corpora/wikitext2_train.txt"; echo; done
  for _ in $(seq 1 20); do cat "$OUT_DIR/corpora/shakespeare.txt"; echo; done
} > "$OUT_DIR/corpora/mixed_large.txt"

SCENARIOS=()
case "$PRESET" in
  matrix)
    SCENARIOS+=(
      "shakespeare|$OUT_DIR/corpora/shakespeare.txt|32|8"
      "code|$OUT_DIR/corpora/code.txt|32|8"
      "german|$OUT_DIR/corpora/german.txt|32|8"
    )
    ;;
  decision)
    SCENARIOS+=(
      "shakespeare|$OUT_DIR/corpora/shakespeare.txt|32|8"
      "code|$OUT_DIR/corpora/code.txt|32|8"
      "german|$OUT_DIR/corpora/german.txt|32|8"
      "shakespeare_long|$OUT_DIR/corpora/shakespeare.txt|64|4"
    )
    ;;
  large)
    SCENARIOS+=(
      "tiny|$OUT_DIR/corpora/shakespeare.txt|32|8"
      "medium|$OUT_DIR/corpora/wikitext2_train.txt|64|4"
      "large|$OUT_DIR/corpora/mixed_large.txt|64|4"
    )
    ;;
  *) echo "Invalid preset: $PRESET"; exit 2 ;;
esac

RUNNERS=()
if [[ -f "./run_llm_orig.sh" ]]; then
  RUNNERS+=("orig|./run_llm_orig.sh|")
else
  echo "FATAL: ./run_llm_orig.sh not found" >&2
  exit 1
fi
if [[ -f "./run_llm.sh" ]]; then
  RUNNERS+=("llm|./run_llm.sh|")
  RUNNERS+=("llm_pho|./run_llm.sh|--pho")
else
  echo "WARN: ./run_llm.sh not found, skipping llm/llm_pho runners" >&2
fi
if [[ "$INCLUDE_BEAST" == "1" ]]; then
  if [[ -f "./runbeast.sh" ]]; then
    RUNNERS+=("beast|./runbeast.sh|")
  else
    echo "WARN: --include-beast requested but ./runbeast.sh not found, skipping beast runner" >&2
  fi
fi

run_once(){
  local runner="$1" script="$2" flag="$3" scenario="$4" corpus="$5" seq="$6" batch="$7" lr="$8" rep="$9"
  local log="$OUT_DIR/logs/${runner}__${scenario}__lr${lr}__r${rep}.log"
  local ckpt="$OUT_DIR/logs/ckpt_${runner}__${scenario}__lr${lr}__r${rep}.bin"

  INDEX_INPUTS="$corpus" DATA_FILE="$corpus" /usr/bin/time -f 'ELAPSED_SEC=%e' \
    bash "$script" --train --force-new --steps "$STEPS" --log_every "$LOG_EVERY" --save_every "$SAVE_EVERY" \
    --gpus "$GPUS" --seq "$seq" --batch "$batch" --measure --lr "$lr" --ckpt "$ckpt" $flag > "$log" 2>&1

  local tok sec lr_disp
  tok="$(grep -Eo 'tok/s=[0-9]+(\.[0-9]+)?' "$log" | tail -1 | cut -d= -f2 || echo NA)"
  sec="$(grep -Eo 'ELAPSED_SEC=[0-9]+(\.[0-9]+)?' "$log" | tail -1 | cut -d= -f2 || echo NA)"
  lr_disp="$(normalize_lr "$lr")"
  echo "$runner,$scenario,$lr_disp,$rep,$tok,$sec" >> "$CSV"
  printf '[run] %-8s %-16s lr=%-8s r%s tok/s=%-8s sec=%s\n' "$runner" "$scenario" "$lr_disp" "$rep" "$tok" "$sec"
}

echo "Starting benchmark preset=$PRESET repeats=$REPEATS steps=$STEPS lr_list=$LR_LIST include_beast=$INCLUDE_BEAST"
IFS=',' read -r -a LR_ARR <<< "$LR_LIST"
for s in "${SCENARIOS[@]}"; do
  name="${s%%|*}"; rest="${s#*|}"
  corpus="${rest%%|*}"; rest2="${rest#*|}"
  seq="${rest2%%|*}"; batch="${rest2#*|}"
  for lr in "${LR_ARR[@]}"; do
    for r in "${RUNNERS[@]}"; do
      runner="${r%%|*}"; rr="${r#*|}"
      script="${rr%%|*}"; flag="${rr#*|}"
      for rep in $(seq 1 "$REPEATS"); do
        run_once "$runner" "$script" "$flag" "$name" "$corpus" "$seq" "$batch" "$lr" "$rep"
      done
    done
  done
done

python3 - "$CSV" <<'PY'
import csv,sys
from collections import defaultdict
csv_path=sys.argv[1]
rows=list(csv.DictReader(open(csv_path)))

def ff(x):
    try:return float(x)
    except:return None

def mmss(x):
    m=int(x//60); s=x-m*60
    return f"{m}:{s:05.2f}"

per_key=defaultdict(list)
per_scenario=defaultdict(list)
triple=defaultdict(list)
for r in rows:
    t=ff(r['tok_s']); s=ff(r['elapsed_sec'])
    if t is None or s is None: continue
    rk=f"{r['runner']}@{r['lr']}"
    per_key[rk].append((t,s))
    triple[(r['runner'],r['scenario'],r['lr'])].append((t,s))

for (runner,scenario,lr),vals in triple.items():
    mt=sum(t for t,_ in vals)/len(vals)
    per_scenario[scenario].append((mt,f"{runner}@{lr}"))

wins=defaultdict(int)
for _,cands in per_scenario.items():
    cands.sort(reverse=True)
    if cands: wins[cands[0][1]] += 1

report=[]
for rk,vals in per_key.items():
    mt=sum(t for t,_ in vals)/len(vals)
    ms=sum(s for _,s in vals)/len(vals)
    total=sum(s for _,s in vals)
    report.append((rk,mt,ms,total,wins.get(rk,0)))

mts=[x[1] for x in report]; mss=[x[2] for x in report]
min_t,max_t=min(mts),max(mts)
min_s,max_s=min(mss),max(mss)

def norm(x,a,b): return 0.5 if a==b else (x-a)/(b-a)

scored=[]
for rk,mt,ms,total,w in report:
    score=norm(mt,min_t,max_t)-norm(ms,min_s,max_s)+w*0.08
    scored.append((score,rk,mt,ms,total,w))
scored.sort(reverse=True)

line='+--------------------+------------+----------+----------+------+'
print('\n'+line)
print('| runner@lr          | tok/s      | run      | total    | wins |')
print(line)
for _,rk,mt,ms,total,w in scored[:5]:
    print(f"| {rk:<18} | {mt:10.1f} | {mmss(ms):>8} | {mmss(total):>8} | {w:4d} |")
print(line)
print(f"🏆 WINNER: {scored[0][1]}")
print(f"CSV: {csv_path}")
PY
