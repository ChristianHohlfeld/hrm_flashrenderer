import json, os, random, sys
from pathlib import Path

DATASET_ID = os.environ["DATASET_ID"]
DATASET_SPLIT = os.environ["DATASET_SPLIT"]
CALIB_LIMIT = int(os.environ["CALIB_LIMIT"])
CALIB_JSONL = Path(os.environ["CALIB_JSONL"])

def pip_install(pkgs):
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + pkgs)

try:
    from datasets import load_dataset
except Exception:
    pip_install(["datasets"])
    from datasets import load_dataset

random.seed(1234)

def extract_prompt_response(row):
    if "instruction" in row and "response" in row:
        inst = (row.get("instruction") or "").strip()
        ctx = (row.get("context") or "").strip()
        resp = (row.get("response") or "").strip()
        prompt = f"{inst}\n\nContext:\n{ctx}" if ctx else inst
        return prompt, resp
    if "instruction" in row and ("output" in row or "response" in row):
        inst = (row.get("instruction") or "").strip()
        inp = (row.get("input") or row.get("context") or "").strip()
        out = (row.get("output") or row.get("response") or "").strip()
        prompt = f"{inst}\n\nInput:\n{inp}" if inp else inst
        return prompt, out
    if "prompt" in row and ("completion" in row or "response" in row or "chosen" in row):
        p = (row.get("prompt") or "").strip()
        r = (row.get("completion") or row.get("response") or row.get("chosen") or "").strip()
        return p, r
    msgs = row.get("messages")
    if isinstance(msgs, list):
        user_parts=[]; assistant_parts=[]
        for m in msgs:
            if not isinstance(m, dict): continue
            role = (m.get("role") or "").lower()
            content = (m.get("content") or "").strip()
            if role == "user" and content: user_parts.append(content)
            elif role == "assistant" and content: assistant_parts.append(content)
        if user_parts and assistant_parts:
            return "\n\n".join(user_parts), assistant_parts[0]
    return "", ""

print(f"[*] Loading dataset: {DATASET_ID} [{DATASET_SPLIT}]")
ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
items = []
for row in ds:
    prompt, resp = extract_prompt_response(row)
    prompt = prompt.strip()
    resp = resp.strip()
    if len(prompt) < 2 or len(resp) < 2: continue
    if len(prompt) > 4000 or len(resp) > 4000: continue
    items.append({"prompt": prompt, "teacher_text": resp})

random.shuffle(items)
items = items[:CALIB_LIMIT]
with CALIB_JSONL.open("w", encoding="utf-8") as f:
    for obj in items:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
print(f"[*] Wrote {len(items)} rows to {CALIB_JSONL}")
