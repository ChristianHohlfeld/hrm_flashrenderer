#!/usr/bin/env bash
set -euo pipefail

# Decision-grade benchmark (small but statistically useful)
# Focus metrics: tok/s, run duration, stability, wins across scenarios.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /home/chris/myenv2/bin/activate
cd "$ROOT_DIR"

REPEATS="${REPEATS:-3}"          # number of fresh runs per scenario
STEPS="${STEPS:-80}"             # keep small for fast iteration
LOG_EVERY="${LOG_EVERY:-10}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
GPUS="${GPUS:-1}"
INCLUDE_BEAST="${INCLUDE_BEAST:-1}"
LR_LIST="${LR_LIST:-3e-4}"   # comma-separated, e.g. 1e-4,3e-4,1e-3
OUT_DIR="${OUT_DIR:-$ROOT_DIR/bench_decision}"
mkdir -p "$OUT_DIR/corpora" "$OUT_DIR/logs"

[[ -s "$OUT_DIR/corpora/shakespeare.txt" ]] || \
  curl -fsSL -o "$OUT_DIR/corpora/shakespeare.txt" https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

cat > "$OUT_DIR/corpora/code.txt" <<'TXT'
#include <iostream>
#include <vector>
int main(){ std::vector<int> v{1,2,3}; for(auto x: v) std::cout<<x<<"\n"; }

def fib(n):
    a,b=0,1
    for _ in range(n):
        a,b=b,a+b
    return a
TXT

cat > "$OUT_DIR/corpora/german.txt" <<'TXT'
Dies ist ein kompakter deutscher Testkorpus zur Vergleichbarkeit von Trainingsläufen.
Wir messen Durchsatz, Laufzeit und Stabilität über mehrere Wiederholungen.
TXT

SCENARIOS=(
  "shakespeare|$OUT_DIR/corpora/shakespeare.txt|32|8"
  "code|$OUT_DIR/corpora/code.txt|32|8"
  "german|$OUT_DIR/corpora/german.txt|32|8"
  "shakespeare_long|$OUT_DIR/corpora/shakespeare.txt|64|4"
)

RUNNERS=(
  "orig|./run_llm_orig.sh|"
  "llm|./run_llm.sh|"
  "llm_pho|./run_llm.sh|--pho"
)
[[ "$INCLUDE_BEAST" == "1" ]] && RUNNERS+=("beast|./runbeast.sh|")

CSV="$OUT_DIR/raw.csv"
echo 'runner,scenario,lr,rep,tok_s,elapsed_sec' > "$CSV"

run_once(){
  local runner="$1" script="$2" flag="$3" scenario="$4" corpus="$5" seq="$6" batch="$7" lr="$8" rep="$9"
  local log="$OUT_DIR/logs/${runner}__${scenario}__r${rep}.log"
  local ckpt="$OUT_DIR/logs/ckpt_${runner}__${scenario}__r${rep}.bin"

  INDEX_INPUTS="$corpus" DATA_FILE="$corpus" /usr/bin/time -f 'ELAPSED_SEC=%e' \
    "$script" --train --force-new --steps "$STEPS" --log_every "$LOG_EVERY" --save_every "$SAVE_EVERY" \
    --gpus "$GPUS" --seq "$seq" --batch "$batch" --measure --lr "$lr" --ckpt "$ckpt" $flag > "$log" 2>&1

  local tok sec
  tok="$(grep -Eo 'tok/s=[0-9]+(\.[0-9]+)?' "$log" | tail -1 | cut -d= -f2 || echo NA)"
  sec="$(grep -Eo 'ELAPSED_SEC=[0-9]+(\.[0-9]+)?' "$log" | tail -1 | cut -d= -f2 || echo NA)"
  echo "$runner,$scenario,$lr,$rep,$tok,$sec" >> "$CSV"
  printf '%-9s %-16s lr=%-7s r%-2s tok/s=%-9s sec=%s\n' "$runner" "$scenario" "$lr" "$rep" "$tok" "$sec"
}

echo "Running decision benchmark..."
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

rows=list(csv.DictReader(open('bench_decision/raw.csv')))

def f(x):
    try: return float(x)
    except: return None

def mmss(x):
    m=int(x//60)
    s=x-m*60
    return f"{m}:{s:05.2f}"

by_rs=defaultdict(list)
for r in rows:
    t=f(r['tok_s']); s=f(r['elapsed_sec'])
    if t is None or s is None:
        continue
    by_rs[(r['runner'], r['scenario'], r['lr'])].append((t,s))

per_runner=defaultdict(lambda:{'tok':[],'sec':[]})
for (runner,_,lr),vals in by_rs.items():
    for t,s in vals:
        per_runner[(runner,lr)]['tok'].append(t)
        per_runner[(runner,lr)]['sec'].append(s)

def stats(vals):
    n=len(vals)
    m=sum(vals)/n
    v=sum((x-m)**2 for x in vals)/n
    sd=math.sqrt(v)
    cv=(sd/m*100) if m else 0.0
    return m,cv

wins=defaultdict(int)
for sc in sorted({k[1] for k in by_rs}):
    cand=[]
    for (r,s,lr),vals in by_rs.items():
        if s!=sc: continue
        mt=sum(t for t,_ in vals)/len(vals)
        cand.append((mt,f'{r}@{lr}'))
    if cand:
        cand.sort(reverse=True)
        wins[cand[0][1]] += 1

print('\n=== KERNMETRIKEN ===')
print(f"{'runner@lr':<18} {'avg_tok/s':>10} {'avg_run':>10} {'total':>10} {'tok_cv%':>8} {'wins':>6}")
report=[]
for key,data in sorted(per_runner.items()):
    r,lr = key
    mt,cvt=stats(data['tok'])
    ms,cvs=stats(data['sec'])
    total=sum(data['sec'])
    rk=f'{r}@{lr}'
    w=wins.get(rk,0)
    report.append((rk,mt,cvt,ms,cvs,total,w))
    print(f"{rk:<18} {mt:10.1f} {mmss(ms):>10} {mmss(total):>10} {cvt:8.2f} {w:6d}")

mts=[x[1] for x in report]; mss=[x[3] for x in report]
min_t,max_t=min(mts),max(mts)
min_s,max_s=min(mss),max(mss)

def norm(x,a,b):
    return 0.5 if a==b else (x-a)/(b-a)

scored=[]
for r,mt,cvt,ms,cvs,total,w in report:
    speed=norm(mt,min_t,max_t)
    time_pen=norm(ms,min_s,max_s)
    stab_pen=(cvt/100.0)*0.15
    score=speed - time_pen - stab_pen + (w*0.05)
    scored.append((score,r))
scored.sort(reverse=True)

print(f"\nWinner: {scored[0][1]}")
print('Raw CSV: bench_decision/raw.csv')
PY
