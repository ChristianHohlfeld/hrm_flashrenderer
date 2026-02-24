# Copyright (c) 2026 Christian Heinrich Hohlfeld (Konstanz, Germany)
# https://christianhohlfeld.com | ORCID: https://orcid.org/0009-0003-6634-9045
# ALL RIGHTS RESERVED. No license granted without prior written permission.
import os
import sys
import socket
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoConfig, AutoTokenizer

from tp.dist_utils import init_dist, init_dist_env
from engine.llama_tp import TPLlamaForCausalLM, generate_tp
from engine.weight_loader import WeightLoader


def _pick_free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _worker_spawn(rank: int, world: int, master_addr: str, master_port: int, args_dict: dict):
    init_dist(rank, world, master_addr, master_port, local_rank=rank)
    _run_generate(rank, args_dict)
    dist.destroy_process_group()


def _read_prompt(args: dict) -> str:
    if args.get("prompt"):
        return args["prompt"]
    pf = args.get("prompt_file")
    if pf:
        with open(pf, "r", encoding="utf-8") as f:
            return f.read()
    if args.get("prompt_stdin"):
        return sys.stdin.read()
    raise RuntimeError("No prompt provided")


def _run_generate(rank: int, args: dict):
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", rank)))

    if rank == 0:
        prompt = _read_prompt(args)
        tokenizer = AutoTokenizer.from_pretrained(args["model"], local_files_only=bool(args.get("local_files_only", False)))
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]

        # Safety: hard clamp to max_seq_len (deterministic).
        max_seq_len = int(args["max_seq_len"])
        if input_ids.size(1) > max_seq_len:
            # keep the most recent tokens (right side)
            input_ids = input_ids[:, -max_seq_len:]

        input_ids = input_ids.to(device=device)
        eos_id = tokenizer.eos_token_id
        shape = torch.tensor([input_ids.size(0), input_ids.size(1)], device=device, dtype=torch.long)
    else:
        tokenizer = None
        input_ids = None
        eos_id = None
        shape = torch.empty((2,), device=device, dtype=torch.long)

    dist.broadcast(shape, src=0)
    B, T = int(shape[0].item()), int(shape[1].item())
    if rank != 0:
        input_ids = torch.empty((B, T), device=device, dtype=torch.long)
    dist.broadcast(input_ids, src=0)

    # Broadcast eos_id to all ranks to avoid divergent generate loops.
    if rank == 0:
        eos_t = torch.tensor([int(eos_id) if eos_id is not None else -1], device=device, dtype=torch.int32)
    else:
        eos_t = torch.empty((1,), device=device, dtype=torch.int32)
    dist.broadcast(eos_t, src=0)
    eos_val = int(eos_t.item())
    eos_id = eos_val if eos_val >= 0 else None

    config = AutoConfig.from_pretrained(args["model"], local_files_only=bool(args.get("local_files_only", False)))
    model = TPLlamaForCausalLM(config, max_seq_len=int(args["max_seq_len"])).cuda().eval()

    loader = WeightLoader(args["model"])
    model.load_from_loader(loader)
    loader.close()

    out = generate_tp(
        model,
        input_ids,
        max_new_tokens=int(args["max_new_tokens"]),
        eos_token_id=eos_id,
        prefill_chunk_size=int(args["prefill_chunk_size"]),
    )

    if rank == 0:
        text = tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
        print(text)


def main():
    ap = argparse.ArgumentParser(description="FlashAttention TP generate (SM75) for local HF safetensors.")
    ap.add_argument("--model", required=True, help="Local model directory with safetensors")

    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--prompt", type=str)
    g.add_argument("--prompt_file", type=str)
    g.add_argument("--prompt_stdin", action="store_true")

    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--max_seq_len", type=int, default=8192)
    ap.add_argument("--prefill_chunk_size", type=int, default=1024)
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--world", type=int, choices=[2, 3, 4], default=None, help="If set, spawn processes (no torchrun).")

    args = ap.parse_args()

    if args.world is None:
        rank, _local_rank, _world = init_dist_env()
        _run_generate(rank, vars(args))
        dist.destroy_process_group()
        return

    port = _pick_free_port()
    mp.spawn(_worker_spawn, args=(args.world, "127.0.0.1", port, vars(args)), nprocs=args.world, join=True)


if __name__ == "__main__":
    main()

