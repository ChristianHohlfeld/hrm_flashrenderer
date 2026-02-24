# Copyright (c) 2026 Christian Heinrich Hohlfeld (Konstanz, Germany)
# https://christianhohlfeld.com | ORCID: https://orcid.org/0009-0003-6634-9045
# ALL RIGHTS RESERVED. No license granted without prior written permission.
import argparse
import os
import socket
import sys
import time
from dataclasses import dataclass
from multiprocessing.connection import Listener

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoConfig, AutoTokenizer

from tp.dist_utils import init_dist
from engine.llama_tp import TPLlamaForCausalLM, generate_tp
from engine.weight_loader import WeightLoader
from hrm_flash.utils import validate_tp_world


@dataclass(frozen=True)
class DaemonConfig:
    model_dir: str
    world: int
    max_seq_len: int
    prefill_chunk_size: int
    local_files_only: bool
    host: str
    port: int
    authkey: bytes


def _pick_free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _broadcast_int(v: int, device: torch.device) -> int:
    t = torch.tensor([int(v)], device=device, dtype=torch.int32)
    dist.broadcast(t, src=0)
    return int(t.item())


def _broadcast_tensor(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    # broadcast shape then data
    if dist.get_rank() == 0:
        shape = torch.tensor(list(x.shape), device=device, dtype=torch.int32)
    else:
        shape = torch.empty((x.dim(),), device=device, dtype=torch.int32)
    dist.broadcast(shape, src=0)
    shape_list = [int(v.item()) for v in shape]
    if dist.get_rank() != 0:
        x = torch.empty(shape_list, device=device, dtype=x.dtype)
    dist.broadcast(x, src=0)
    return x


def _worker(rank: int, cfg: DaemonConfig, master_addr: str, master_port: int):
    init_dist(rank, cfg.world, master_addr, master_port, local_rank=rank)
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", rank)))

    # Load tokenizer/config/model once per rank
    config = AutoConfig.from_pretrained(cfg.model_dir, local_files_only=cfg.local_files_only)
    validate_tp_world(config, int(cfg.world))

    model = TPLlamaForCausalLM(config, max_seq_len=int(cfg.max_seq_len)).cuda().eval()
    loader = WeightLoader(cfg.model_dir)
    model.load_from_loader(loader)
    loader.close()

    tokenizer = None
    eos_id = None
    if rank == 0:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_dir, local_files_only=cfg.local_files_only)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        eos_id = tokenizer.eos_token_id

    # broadcast eos_id to all ranks (avoid divergent loops)
    if rank == 0:
        eos_t = torch.tensor([int(eos_id) if eos_id is not None else -1], device=device, dtype=torch.int32)
    else:
        eos_t = torch.empty((1,), device=device, dtype=torch.int32)
    dist.broadcast(eos_t, src=0)
    eos_val = int(eos_t.item())
    eos_id = eos_val if eos_val >= 0 else None

    # Rank0: RPC loop, broadcast commands to all ranks
    if rank == 0:
        listener = Listener((cfg.host, cfg.port), authkey=cfg.authkey)
        try:
            # readiness signal
            sys.stderr.write(f"[flashd] READY {cfg.host}:{cfg.port} world={cfg.world}\n")
            sys.stderr.flush()

            while True:
                conn = listener.accept()
                try:
                    msg = conn.recv()
                    if not isinstance(msg, dict):
                        conn.send({"ok": False, "error": "invalid message"})
                        continue
                    cmd = msg.get("cmd")
                    if cmd == "ping":
                        conn.send({"ok": True, "ts": time.time()})
                        continue
                    if cmd == "shutdown":
                        _broadcast_int(0, device)
                        conn.send({"ok": True})
                        break
                    if cmd != "generate":
                        conn.send({"ok": False, "error": "unknown cmd"})
                        continue

                    prompt = str(msg.get("prompt", ""))
                    max_new = int(msg.get("max_new_tokens", 256))
                    max_new = max(1, min(2048, max_new))
                    prefill_chunk = int(msg.get("prefill_chunk_size", cfg.prefill_chunk_size))
                    prefill_chunk = max(64, min(cfg.max_seq_len, prefill_chunk))

                    inputs = tokenizer(prompt, return_tensors="pt")
                    input_ids = inputs["input_ids"]
                    if input_ids.size(1) > cfg.max_seq_len:
                        input_ids = input_ids[:, -cfg.max_seq_len:]
                    input_ids = input_ids.to(device=device)

                    # broadcast generate cmd + params
                    _broadcast_int(1, device)
                    _broadcast_int(max_new, device)
                    _broadcast_int(prefill_chunk, device)

                    # broadcast input_ids
                    # send dims
                    shape = torch.tensor([input_ids.size(0), input_ids.size(1)], device=device, dtype=torch.int32)
                    dist.broadcast(shape, src=0)
                    dist.broadcast(input_ids, src=0)

                    # reset caches for new request (reuse allocations)
                    model.reset_all_caches()
                    out = generate_tp(model, input_ids, max_new_tokens=max_new, eos_token_id=eos_id, prefill_chunk_size=prefill_chunk)
                    text = tokenizer.decode(out[0].tolist(), skip_special_tokens=True)

                    conn.send({"ok": True, "text": text})
                except Exception as e:
                    conn.send({"ok": False, "error": str(e)})
                finally:
                    try:
                        conn.close()
                    except Exception:
                        pass
        finally:
            try:
                listener.close()
            except Exception:
                pass

    # Non-zero ranks: wait for broadcast commands
    while rank != 0:
        cmd = _broadcast_int(-1, device)
        if cmd == 0:
            break
        if cmd != 1:
            continue

        max_new = _broadcast_int(256, device)
        prefill_chunk = _broadcast_int(cfg.prefill_chunk_size, device)

        shape = torch.empty((2,), device=device, dtype=torch.int32)
        dist.broadcast(shape, src=0)
        B = int(shape[0].item())
        T = int(shape[1].item())
        input_ids = torch.empty((B, T), device=device, dtype=torch.long)
        dist.broadcast(input_ids, src=0)

        model.reset_all_caches()
        _ = generate_tp(model, input_ids, max_new_tokens=max_new, eos_token_id=eos_id, prefill_chunk_size=prefill_chunk)

    dist.barrier()
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser(prog="hrm-flashd", description="Persistent FlashAttention TP daemon (loads model once).")
    ap.add_argument("--model", required=True, help="Local HF safetensors model dir")
    ap.add_argument("--world", type=int, choices=[2, 3, 4], required=True)
    ap.add_argument("--max_seq_len", type=int, default=8192)
    ap.add_argument("--prefill_chunk_size", type=int, default=1024)
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--authkey", type=str, default="hrmflash")
    args = ap.parse_args()

    port = int(args.port) if int(args.port) != 0 else _pick_free_port()

    cfg = DaemonConfig(
        model_dir=str(args.model),
        world=int(args.world),
        max_seq_len=int(args.max_seq_len),
        prefill_chunk_size=int(args.prefill_chunk_size),
        local_files_only=bool(args.local_files_only),
        host=str(args.host),
        port=port,
        authkey=str(args.authkey).encode("utf-8"),
    )

    master_port = _pick_free_port()
    mp.spawn(_worker, args=(cfg, "127.0.0.1", master_port), nprocs=cfg.world, join=True)


if __name__ == "__main__":
    main()

