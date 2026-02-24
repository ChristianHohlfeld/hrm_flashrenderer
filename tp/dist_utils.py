# Copyright (c) 2026 Christian Heinrich Hohlfeld (Konstanz, Germany)
# https://christianhohlfeld.com | ORCID: https://orcid.org/0009-0003-6634-9045
# ALL RIGHTS RESERVED. No license granted without prior written permission.
import os
import torch
import torch.distributed as dist

def init_dist_env():
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ.get('LOCAL_RANK', rank))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', rank=rank, world_size=world)
    return rank, local_rank, world

def init_dist(rank: int, world: int, master_addr: str, master_port: int, local_rank: int | None = None):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world)
    os.environ['LOCAL_RANK'] = str(local_rank if local_rank is not None else rank)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group(backend='nccl', rank=rank, world_size=world)
    return dist.get_rank(), int(os.environ['LOCAL_RANK']), dist.get_world_size()

def all_reduce_sum(x: torch.Tensor):
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x

def all_gather_cat(x: torch.Tensor, dim: int):
    world = dist.get_world_size()
    xs = [torch.empty_like(x) for _ in range(world)]
    dist.all_gather(xs, x)
    return torch.cat(xs, dim=dim)

