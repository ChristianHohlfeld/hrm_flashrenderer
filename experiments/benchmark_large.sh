#!/usr/bin/env bash
set -euo pipefail

# Large-corpus decision benchmark (simple + signal-only)
# Output: only core metrics (tok/s, avg run time, total time, wins)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /home/chris/myenv2/bin/activate
cd "$ROOT_DIR"

REPEATS="${REPEATS:-2}"
STEPS="${STEPS:-120}"
LOG_EVERY="${LOG_EVERY:-20}"
SAVE_EVERY="${SAVE_EVERY:-2000}"
GPUS="${GPUS:-1}"
INCLUDE_BEAST="${INCLUDE_BEAST:-0}"
LR_LIST="${LR_LIST:-0.0001,0.0003,0.001}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/bench_large}"
mkdir -p "$OUT_DIR/corpora" "$OUT_DIR/logs"

# -------- corpora --------
# 1) tiny (baseline)
[[ -s "$OUT_DIR/corpora/tiny_shakespeare.txt" ]] || \
  curl -fsSL -o "$OUT_DIR/corpora/tiny_shakespeare.txt" \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# 2) medium (wikitext-2 raw train)
[[ -s "$OUT_DIR/corpora/wikitext2_train.txt" ]] || \
  curl -fsSL -o "$OUT_DIR/corpora/wikitext2_train.txt" \
  https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt

# 3) larger mixed corpus (repeat + merge small sources deterministically)
python3 - <<'PY'
from pathlib import Path
root=Path('bench_large/corpora')
small=(root/'tiny_shakespeare.txt').read_text(errors='ignore')
wiki=(root/'wikitext2_train.txt').read_text(errors='ignore')
out=root/'mixed_large.txt'
# Build ~10-20MB-ish deterministic mixed corpus without huge downloads
chunks=[]
for i in range(6):
    chunks.append(wiki)
for i in range(20):
    chunks.append(small)
out.write_text('\n'.join(chunks))
print('wrote',out,'bytes',out.stat().st_size)
PY

SCENARIOS=(
  "tiny|$OUT_DIR/corpora/tiny_shakespeare.txt|32|8"
  "medium|$OUT_DIR/corpora/wikitext2_train.txt|64|4"
  "large|$OUT_DIR/corpora/mixed_large.txt|64|4"
)

RUNNERS=(
  "orig|./run_llm_orig.sh|"
  "llm|./run_llm.sh|"
  "llm_pho|./run_llm.sh|--pho"
)
[[ "$INCLUDE_BEAST" == "1" ]] && RUNNERS+=("beast|./runbeast.sh|")

CSV="$OUT_DIR/raw.csv"
echo 'runner,scenario,lr,rep,tok_s,elapsed_sec' > "$CSV"

normalize_lr() {
  python3 - "$1" <<'PYLR'
import sys
x=float(sys.argv[1])
print(f"{x:.10f}".rstrip('0').rstrip('.'))
PYLR
}

run_once(){
  local runner="$1" script="$2" flag="$3" scenario="$4" corpus="$5" seq="$6" batch="$7" lr="$8" rep="$9"
  local log="$OUT_DIR/logs/${runner}__${scenario}__lr${lr}__r${rep}.log"
  local ckpt="$OUT_DIR/logs/ckpt_${runner}__${scenario}__lr${lr}__r${rep}.bin"

  INDEX_INPUTS="$corpus" DATA_FILE="$corpus" /usr/bin/time -f 'ELAPSED_SEC=%e' \
    "$script" --train --force-new --steps "$STEPS" --log_every "$LOG_EVERY" --save_every "$SAVE_EVERY" \
    --gpus "$GPUS" --seq "$seq" --batch "$batch" --measure --lr "$lr" --ckpt "$ckpt" $flag > "$log" 2>&1

  local tok sec lr_disp
  tok="$(grep -Eo 'tok/s=[0-9]+(\.[0-9]+)?' "$log" | tail -1 | cut -d= -f2 || echo NA)"
  sec="$(grep -Eo 'ELAPSED_SEC=[0-9]+(\.[0-9]+)?' "$log" | tail -1 | cut -d= -f2 || echo NA)"
  lr_disp="$(normalize_lr "$lr")"
  echo "$runner,$scenario,$lr_disp,$rep,$tok,$sec" >> "$CSV"
  printf '%-9s %-7s lr=%-8s r%-2s tok/s=%-9s sec=%s\n' "$runner" "$scenario" "$lr_disp" "$rep" "$tok" "$sec"
}

echo "Running large benchmark..."
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

python3 - <<'PY'
import csv, math
from collections import defaultdict

def mmss(x):
    m=int(x//60); s=x-m*60
    return f"{m}:{s:05.2f}"

rows=list(csv.DictReader(open('bench_large/raw.csv')))

def f(x):
    try:return float(x)
    except:return None

by_key=defaultdict(list)  # (runner,lr) -> [(tok,sec)]
by_scenario=defaultdict(list)  # (scenario) -> [(mean_tok, runner@lr)]
per_rs=defaultdict(list) # (runner,scenario,lr)->[(tok,sec)]

for r in rows:
    t=f(r['tok_s']); s=f(r['elapsed_sec'])
    if t is None or s is None: continue
    rk=(r['runner'], r['lr'])
    by_key[rk].append((t,s))
    per_rs[(r['runner'], r['scenario'], r['lr'])].append((t,s))

for (runner,scenario,lr),vals in per_rs.items():
    mt=sum(t for t,_ in vals)/len(vals)
    by_scenario[scenario].append((mt,f"{runner}@{lr}"))

wins=defaultdict(int)
for sc,cands in by_scenario.items():
    cands.sort(reverse=True)
    wins[cands[0][1]] += 1

print('\n=== KERNMETRIKEN (LARGE) ===')
print(f"{'runner@lr':<18} {'avg_tok/s':>10} {'avg_run':>10} {'total':>10} {'wins':>6}")
report=[]
for (runner,lr),vals in sorted(by_key.items()):
    mt=sum(t for t,_ in vals)/len(vals)
    ms=sum(s for _,s in vals)/len(vals)
    total=sum(s for _,s in vals)
    key=f"{runner}@{lr}"
    w=wins.get(key,0)
    report.append((key,mt,ms,total,w))
    print(f"{key:<18} {mt:10.1f} {mmss(ms):>10} {mmss(total):>10} {w:6d}")

# simple decision score: speed + scenario wins - time penalty
mts=[x[1] for x in report]; mss=[x[2] for x in report]
min_t,max_t=min(mts),max(mts)
min_s,max_s=min(mss),max(mss)
def norm(x,a,b): return 0.5 if a==b else (x-a)/(b-a)
scored=[]
for key,mt,ms,total,w in report:
    score = norm(mt,min_t,max_t) - norm(ms,min_s,max_s) + (w*0.08)
    scored.append((score,key))
scored.sort(reverse=True)

print(f"\nWinner: {scored[0][1]}")
print('Raw CSV: bench_large/raw.csv')
PY
