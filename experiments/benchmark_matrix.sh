#!/usr/bin/env bash
set -euo pipefail

# Signal-only benchmark: token/s + total duration per run.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /home/chris/myenv2/bin/activate
cd "$ROOT_DIR"

STEPS="${STEPS:-80}"
LOG_EVERY="${LOG_EVERY:-10}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
GPUS="${GPUS:-1}"
SEQ="${SEQ:-32}"
BATCH="${BATCH:-8}"
INCLUDE_BEAST="${INCLUDE_BEAST:-1}"

OUT_DIR="${OUT_DIR:-$ROOT_DIR/bench_out}"
mkdir -p "$OUT_DIR"
CORPUS_DIR="$OUT_DIR/corpora"
mkdir -p "$CORPUS_DIR"

# small corpora
[[ -s "$CORPUS_DIR/shakespeare.txt" ]] || curl -fsSL -o "$CORPUS_DIR/shakespeare.txt" https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
cat > "$CORPUS_DIR/code_snippets.txt" <<'TXT'
#include <iostream>
int main(){ std::cout<<"hello"; }
def fib(n):
 a,b=0,1
 for _ in range(n): a,b=b,a+b
TXT
cat > "$CORPUS_DIR/german_short.txt" <<'TXT'
Dies ist ein kurzer deutscher Testkorpus für reproduzierbare Benchmarks.
Wir wollen verschiedene Skripte unter identischen Bedingungen vergleichen.
TXT

CORPORA=(
  "shakespeare:$CORPUS_DIR/shakespeare.txt"
  "code:$CORPUS_DIR/code_snippets.txt"
  "german:$CORPUS_DIR/german_short.txt"
)

RUNNERS=(
  "orig|./run_llm_orig.sh|"
  "llm|./run_llm.sh|"
  "llm_pho|./run_llm.sh|--pho"
)
[[ "$INCLUDE_BEAST" == "1" ]] && RUNNERS+=("beast|./runbeast.sh|")

run_one(){
  local rname="$1" script="$2" flag="$3" cname="$4" cpath="$5"
  local log="$OUT_DIR/${rname}__${cname}.log"
  local ckpt="$OUT_DIR/ckpt_${rname}__${cname}.bin"

  INDEX_INPUTS="$cpath" DATA_FILE="$cpath" /usr/bin/time -f 'ELAPSED_SEC=%e'     "$script" --train --force-new --steps "$STEPS" --log_every "$LOG_EVERY" --save_every "$SAVE_EVERY"     --gpus "$GPUS" --seq "$SEQ" --batch "$BATCH" --measure --ckpt "$ckpt" $flag > "$log" 2>&1

  local tok sec
  tok="$(grep -Eo 'tok/s=[0-9]+(\.[0-9]+)?' "$log" | tail -1 | cut -d= -f2 || echo NA)"
  sec="$(grep -Eo 'ELAPSED_SEC=[0-9]+(\.[0-9]+)?' "$log" | tail -1 | cut -d= -f2 || echo NA)"
  printf '%-10s %-11s %10s %8s
' "$rname" "$cname" "$tok" "$sec"
  echo "$rname,$cname,$tok,$sec" >> "$OUT_DIR/results_signal.csv"
}

echo 'runner,corpus,last_tok_s,elapsed_sec' > "$OUT_DIR/results_signal.csv"
printf '
%-10s %-11s %10s %8s
' 'runner' 'corpus' 'tok/s' 'sec'
printf '%s
' '---------------------------------------------'
for c in "${CORPORA[@]}"; do
  cname="${c%%:*}"; cpath="${c#*:}"
  for r in "${RUNNERS[@]}"; do
    rname="${r%%|*}"; rest="${r#*|}"; script="${rest%%|*}"; flag="${rest#*|}"
    run_one "$rname" "$script" "$flag" "$cname" "$cpath"
  done
done

python3 - <<'PY2'
import csv
from collections import defaultdict
rows=list(csv.DictReader(open('bench_out/results_signal.csv')))
acc=defaultdict(lambda:[0.0,0.0,0])
for r in rows:
    try:t=float(r['last_tok_s']); s=float(r['elapsed_sec'])
    except: continue
    a=acc[r['runner']]; a[0]+=t; a[1]+=s; a[2]+=1
print('
AVG (signal only):')
print(f"{'runner':<10} {'avg_tok/s':>12} {'avg_sec':>10}")
for k,(t,s,n) in sorted(acc.items()):
    print(f"{k:<10} {t/n:12.1f} {s/n:10.2f}")
PY2
