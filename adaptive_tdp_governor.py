#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
from collections import deque

SUDO_REEXEC_ENV = "ADAPTIVE_TDP_SUDO_REEXEC"
SCHEMA = "adaptive-tdp-governor/v1"


def run(cmd, check=True):
    return subprocess.run(
        cmd,
        shell=isinstance(cmd, str),
        check=check,
        text=True,
        capture_output=True,
        executable="/bin/bash" if isinstance(cmd, str) else None,
    )


def which(name):
    p = run(f"command -v {shlex.quote(name)}", check=False)
    return (p.stdout or "").strip() or None


def is_root():
    return hasattr(os, "geteuid") and os.geteuid() == 0


def ensure_tools():
    if not which("nvidia-smi"):
        print("ERROR: nvidia-smi not found", file=sys.stderr)
        sys.exit(2)
    if not which("sudo"):
        print("ERROR: sudo not found", file=sys.stderr)
        sys.exit(2)


def maybe_reexec_with_sudo():
    if is_root():
        return
    if os.environ.get(SUDO_REEXEC_ENV) == "1":
        print("ERROR: sudo re-exec already attempted, but still not root", file=sys.stderr)
        sys.exit(2)
    print("Re-starting with sudo for GPU power-limit control...")
    env = os.environ.copy()
    env[SUDO_REEXEC_ENV] = "1"
    sudo = which("sudo")
    argv = [sudo, "-E", sys.executable, os.path.abspath(sys.argv[0]), *sys.argv[1:]]
    os.execvpe(sudo, argv, env)


def query_gpus(selected=None):
    p = run(
        "nvidia-smi --query-gpu=index,name,power.limit,power.min_limit,power.max_limit,power.draw,utilization.gpu,clocks.sm,clocks.mem --format=csv,noheader,nounits"
    )
    out = []
    for line in (p.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        a = [x.strip() for x in line.split(",")]
        idx = int(a[0])
        if selected is not None and idx not in selected:
            continue
        out.append({
            "index": idx,
            "name": a[1],
            "power_limit_w": int(round(float(a[2]))),
            "min_w": int(round(float(a[3]))),
            "max_w": int(round(float(a[4]))),
            "power_draw_w": float(a[5]) if a[5] else -1.0,
            "util_gpu_pct": float(a[6]) if a[6] else -1.0,
            "sm_clock_mhz": float(a[7]) if a[7] else -1.0,
            "mem_clock_mhz": float(a[8]) if a[8] else -1.0,
        })
    out.sort(key=lambda x: x["index"])
    return out


def set_pl(gpu_index, watts):
    p = run(f"nvidia-smi -i {gpu_index} -pl {int(watts)}", check=False)
    if p.returncode != 0:
        msg = (p.stderr or p.stdout or "").strip()
        raise RuntimeError(f"GPU {gpu_index}: failed to set power limit to {watts} W: {msg}")


def set_group_pl(power_limits_w):
    for idx_str, watts in power_limits_w.items():
        set_pl(int(idx_str), int(watts))


def atomic_write_json(path, obj):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
    os.replace(tmp, path)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_profile(path):
    if not os.path.exists(path):
        return None
    try:
        obj = load_json(path)
        if isinstance(obj, dict) and obj.get("schema") == SCHEMA:
            return obj
    except Exception:
        pass
    return None


def normalize_phase(x):
    s = str(x or "").strip().lower()
    if s in {"decode", "decoding", "generate", "generating", "inference"}:
        return "decode"
    if s in {"prefill", "prompt", "prompting", "loading"}:
        return "prefill"
    if s in {"idle", "none", "waiting"}:
        return "idle"
    return ""


def parse_token_file(path, min_live_tps):
    raw = ""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    if not raw:
        return {
            "phase": "idle",
            "effective_tps": 0.0,
            "decode_tps": 0.0,
            "inter_token_ms": None,
            "workload_key": "default",
            "raw": {},
        }

    try:
        val = float(raw)
        phase = "decode" if val >= min_live_tps else "idle"
        return {
            "phase": phase,
            "effective_tps": val,
            "decode_tps": val,
            "inter_token_ms": None,
            "workload_key": "default",
            "raw": {"decode_tps": val},
        }
    except ValueError:
        pass

    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise RuntimeError("token file JSON must be an object or a plain number")

    phase = (
        normalize_phase(obj.get("phase"))
        or normalize_phase(obj.get("state"))
        or ("prefill" if any(bool(obj.get(k)) for k in ("prefill", "prefill_active", "prompt_active")) else "")
        or ("decode" if any(bool(obj.get(k)) for k in ("decode", "decode_active", "generating", "generation_active")) else "")
    )

    decode_tps = None
    for k in ("decode_tps", "tps", "tokens_per_second", "current_tps", "tok_s"):
        if k in obj:
            try:
                decode_tps = float(obj[k])
                break
            except Exception:
                pass

    inter_token_ms = None
    for k in ("inter_token_ms", "token_gap_ms", "avg_token_ms", "it_gap_ms"):
        if k in obj:
            try:
                inter_token_ms = float(obj[k])
                break
            except Exception:
                pass

    effective_tps = None
    if decode_tps is not None and decode_tps > 0:
        effective_tps = decode_tps
    elif inter_token_ms is not None and inter_token_ms > 0:
        effective_tps = 1000.0 / inter_token_ms
        decode_tps = effective_tps
    else:
        effective_tps = 0.0
        decode_tps = 0.0

    if not phase:
        phase = "decode" if effective_tps >= min_live_tps else "idle"

    workload_key = (
        str(obj.get("workload_key") or obj.get("model_key") or obj.get("model") or obj.get("profile") or "default")
        .strip()
    ) or "default"

    return {
        "phase": phase,
        "effective_tps": float(effective_tps),
        "decode_tps": float(decode_tps),
        "inter_token_ms": float(inter_token_ms) if inter_token_ms is not None else None,
        "workload_key": workload_key,
        "raw": obj,
    }


def cv(values):
    if len(values) < 2:
        return 999.0
    m = sum(values) / len(values)
    if m <= 0:
        return 999.0
    return statistics.pstdev(values) / m


def median(values):
    return statistics.median(values) if values else 0.0


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def build_empty_profile(token_file, output_file, gpus, settings):
    return {
        "schema": SCHEMA,
        "created_at_unix": int(time.time()),
        "updated_at_unix": int(time.time()),
        "token_file": token_file,
        "output_file": output_file,
        "settings": settings,
        "gpus": [
            {
                "index": g["index"],
                "name": g["name"],
                "min_w": g["min_w"],
                "max_w": g["max_w"],
            }
            for g in gpus
        ],
        "current": {
            "phase": "idle",
            "workload_key": "default",
            "applied_power_limits_w": {str(g["index"]): int(g["power_limit_w"]) for g in gpus},
            "recent_effective_tps": 0.0,
            "probe": None,
            "gpu_snapshot": [],
        },
        "workloads": {},
    }


def ensure_workload(profile, workload_key, gpus):
    workloads = profile.setdefault("workloads", {})
    wl = workloads.get(workload_key)
    if wl is None:
        wl = {
            "decode_power_limits_w": {str(g["index"]): int(g["max_w"]) for g in gpus},
            "prefill_power_limits_w": {str(g["index"]): int(g["max_w"]) for g in gpus},
            "gpu_meta": {
                str(g["index"]): {
                    "last_good_w": int(g["max_w"]),
                    "last_bad_w": None,
                    "accepted_steps": 0,
                    "rejected_steps": 0,
                }
                for g in gpus
            },
            "last_reference_tps": None,
            "last_baseline_tps": None,
            "last_updated_unix": int(time.time()),
        }
        workloads[workload_key] = wl
    return wl


def next_probe_target(gpus, wl, step_w, floor_w, rr_cursor):
    n = len(gpus)
    for off in range(n):
        g = gpus[(rr_cursor + off) % n]
        idx = str(g["index"])
        meta = wl["gpu_meta"][idx]
        cur = int(wl["decode_power_limits_w"][idx])
        floor = max(int(floor_w), int(g["min_w"]))
        cand = cur - int(step_w)
        if cand < floor:
            continue
        last_bad = meta.get("last_bad_w")
        if last_bad is not None and cand <= int(last_bad):
            continue
        return g, cand, (rr_cursor + off + 1) % n
    return None, None, rr_cursor


def main():
    ap = argparse.ArgumentParser(description="Online adaptive TDP governor")
    ap.add_argument("--token-file", default="adaptive_tdp_live.json")
    ap.add_argument("--output", default="adaptive_tdp_profile.json")
    ap.add_argument("--gpus", default=None, help="comma-separated GPU indices, e.g. 0,1,2")
    ap.add_argument("--step-w", type=int, default=10)
    ap.add_argument("--floor-w", type=int, default=110)
    ap.add_argument("--poll-ms", type=int, default=500)
    ap.add_argument("--stable-samples", type=int, default=10)
    ap.add_argument("--stable-cv", type=float, default=0.04)
    ap.add_argument("--min-live-tps", type=float, default=1.0)
    ap.add_argument("--allowed-loss-frac", type=float, default=0.025)
    ap.add_argument("--probe-settle-ms", type=int, default=1200)
    ap.add_argument("--probe-window-ms", type=int, default=2200)
    ap.add_argument("--recover-settle-ms", type=int, default=1000)
    ap.add_argument("--recover-window-ms", type=int, default=1800)
    ap.add_argument("--cooldown-ms", type=int, default=5000)
    ap.add_argument("--gpu-snapshot-ms", type=int, default=1500)
    args = ap.parse_args()

    ensure_tools()
    maybe_reexec_with_sudo()

    selected = [int(x.strip()) for x in args.gpus.split(",")] if args.gpus else None
    gpus = query_gpus(selected)
    if not gpus:
        print("ERROR: no GPUs found", file=sys.stderr)
        sys.exit(2)

    settings = {
        "step_w": args.step_w,
        "floor_w": args.floor_w,
        "poll_ms": args.poll_ms,
        "stable_samples": args.stable_samples,
        "stable_cv": args.stable_cv,
        "min_live_tps": args.min_live_tps,
        "allowed_loss_frac": args.allowed_loss_frac,
        "probe_settle_ms": args.probe_settle_ms,
        "probe_window_ms": args.probe_window_ms,
        "recover_settle_ms": args.recover_settle_ms,
        "recover_window_ms": args.recover_window_ms,
        "cooldown_ms": args.cooldown_ms,
        "gpu_snapshot_ms": args.gpu_snapshot_ms,
    }

    profile = load_profile(args.output) or build_empty_profile(args.token_file, args.output, gpus, settings)

    print("Detected GPUs:")
    for g in gpus:
        print(f"  GPU {g['index']}: {g['name']} | start={g['power_limit_w']} W | min={g['min_w']} W | max={g['max_w']} W")

    current_pl = {str(g["index"]): int(g["power_limit_w"]) for g in gpus}
    recent_decode = deque(maxlen=max(3, args.stable_samples))
    phase_prev = "idle"
    workload_prev = "default"
    rr_cursor = 0
    last_probe_done_ts = 0.0
    last_gpu_snapshot_ts = 0.0
    gpu_snapshot = []
    probe = None

    while True:
        now = time.time()

        try:
            token = parse_token_file(args.token_file, args.min_live_tps)
        except Exception:
            token = {
                "phase": "idle",
                "effective_tps": 0.0,
                "decode_tps": 0.0,
                "inter_token_ms": None,
                "workload_key": "default",
                "raw": {},
            }

        phase = token["phase"]
        workload_key = token["workload_key"]
        wl = ensure_workload(profile, workload_key, gpus)

        decode_pl = {str(k): int(v) for k, v in wl["decode_power_limits_w"].items()}
        prefill_pl = {str(k): int(v) for k, v in wl["prefill_power_limits_w"].items()}

        if phase == "decode":
            recent_decode.append(float(token["effective_tps"]))
            wl["last_baseline_tps"] = round(median(list(recent_decode)), 6)
            wl["last_updated_unix"] = int(now)
        else:
            recent_decode.clear()

        if now - last_gpu_snapshot_ts >= args.gpu_snapshot_ms / 1000.0:
            gpu_snapshot = query_gpus(selected)
            last_gpu_snapshot_ts = now

        if phase == "prefill":
            if probe is not None:
                set_pl(probe["gpu"], probe["orig_w"])
                current_pl[str(probe["gpu"])] = int(probe["orig_w"])
                probe = None
            if current_pl != prefill_pl:
                set_group_pl(prefill_pl)
                current_pl = dict(prefill_pl)

        elif phase == "decode":
            if probe is None and current_pl != decode_pl:
                set_group_pl(decode_pl)
                current_pl = dict(decode_pl)

            stable = (
                len(recent_decode) >= args.stable_samples
                and cv(list(recent_decode)) <= args.stable_cv
                and median(list(recent_decode)) >= args.min_live_tps
            )

            if probe is None and stable and (now - last_probe_done_ts) >= args.cooldown_ms / 1000.0:
                g, cand_w, rr_cursor = next_probe_target(gpus, wl, args.step_w, args.floor_w, rr_cursor)
                if g is not None:
                    idx = str(g["index"])
                    before = median(list(recent_decode))
                    orig_w = int(decode_pl[idx])
                    set_pl(g["index"], cand_w)
                    current_pl[idx] = int(cand_w)
                    probe = {
                        "gpu": g["index"],
                        "gpu_name": g["name"],
                        "workload_key": workload_key,
                        "orig_w": int(orig_w),
                        "cand_w": int(cand_w),
                        "before_tps": float(before),
                        "probe_samples": [],
                        "after_samples": [],
                        "stage": "settle_probe",
                        "stage_started_at": now,
                    }

            elif probe is not None:
                if workload_key != probe["workload_key"] or phase != "decode":
                    set_pl(probe["gpu"], probe["orig_w"])
                    current_pl[str(probe["gpu"])] = int(probe["orig_w"])
                    probe = None
                    last_probe_done_ts = now
                else:
                    metric = float(token["effective_tps"])

                    if probe["stage"] == "settle_probe":
                        if now - probe["stage_started_at"] >= args.probe_settle_ms / 1000.0:
                            probe["stage"] = "collect_probe"
                            probe["stage_started_at"] = now

                    elif probe["stage"] == "collect_probe":
                        if metric >= args.min_live_tps:
                            probe["probe_samples"].append(metric)
                        if now - probe["stage_started_at"] >= args.probe_window_ms / 1000.0:
                            set_pl(probe["gpu"], probe["orig_w"])
                            current_pl[str(probe["gpu"])] = int(probe["orig_w"])
                            probe["probe_tps"] = median(probe["probe_samples"])
                            probe["stage"] = "settle_recover"
                            probe["stage_started_at"] = now

                    elif probe["stage"] == "settle_recover":
                        if now - probe["stage_started_at"] >= args.recover_settle_ms / 1000.0:
                            probe["stage"] = "collect_recover"
                            probe["stage_started_at"] = now

                    elif probe["stage"] == "collect_recover":
                        if metric >= args.min_live_tps:
                            probe["after_samples"].append(metric)
                        if now - probe["stage_started_at"] >= args.recover_window_ms / 1000.0:
                            after_tps = median(probe["after_samples"])
                            baseline = (float(probe["before_tps"]) + float(after_tps)) / 2.0
                            probe_tps = float(probe.get("probe_tps", 0.0))
                            ratio = (probe_tps / baseline) if baseline > 0 else 0.0
                            accepted = ratio >= (1.0 - float(args.allowed_loss_frac))

                            idx = str(probe["gpu"])
                            meta = wl["gpu_meta"][idx]

                            if accepted:
                                wl["decode_power_limits_w"][idx] = int(probe["cand_w"])
                                meta["last_good_w"] = int(probe["cand_w"])
                                meta["accepted_steps"] = int(meta.get("accepted_steps", 0)) + 1
                                set_pl(probe["gpu"], probe["cand_w"])
                                current_pl[idx] = int(probe["cand_w"])
                            else:
                                meta["last_bad_w"] = int(probe["cand_w"])
                                meta["rejected_steps"] = int(meta.get("rejected_steps", 0)) + 1
                                set_pl(probe["gpu"], probe["orig_w"])
                                current_pl[idx] = int(probe["orig_w"])

                            wl["last_reference_tps"] = round(baseline, 6)
                            wl["last_updated_unix"] = int(now)
                            probe = None
                            last_probe_done_ts = now

        else:
            if probe is not None:
                set_pl(probe["gpu"], probe["orig_w"])
                current_pl[str(probe["gpu"])] = int(probe["orig_w"])
                probe = None
                last_probe_done_ts = now

        profile["updated_at_unix"] = int(now)
        profile["current"] = {
            "phase": phase,
            "workload_key": workload_key,
            "applied_power_limits_w": {str(k): int(v) for k, v in current_pl.items()},
            "recent_effective_tps": round(median(list(recent_decode)), 6) if recent_decode else 0.0,
            "probe": probe,
            "gpu_snapshot": gpu_snapshot,
            "token_raw": token["raw"],
        }

        atomic_write_json(args.output, profile)

        phase_prev = phase
        workload_prev = workload_key
        time.sleep(args.poll_ms / 1000.0)


if __name__ == "__main__":
    main()
