# Copyright (c) 2026 Christian Heinrich Hohlfeld (Konstanz, Germany)
# https://christianhohlfeld.com | ORCID: https://orcid.org/0009-0003-6634-9045
# ALL RIGHTS RESERVED. No license granted without prior written permission.
import torch
import torch.nn as nn
import torch.distributed as dist
from .dist_utils import all_reduce_sum

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, device='cuda'):
        super().__init__()
        world = dist.get_world_size()
        rank = dist.get_rank()
        assert out_features % world == 0
        self.rank = rank
        self.world = world
        self.local_out = out_features // world
        self.weight = nn.Parameter(torch.empty(self.local_out, in_features, device=device, dtype=torch.float16))
        self.bias = nn.Parameter(torch.empty(self.local_out, device=device, dtype=torch.float16)) if bias else None
        nn.init.normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @torch.no_grad()
    def load_from_full(self, full_weight: torch.Tensor, full_bias=None):
        start = self.rank * self.local_out
        end = start + self.local_out
        self.weight.copy_(full_weight[start:end].to(device=self.weight.device, dtype=torch.float16))
        if self.bias is not None and full_bias is not None:
            self.bias.copy_(full_bias[start:end].to(device=self.weight.device, dtype=torch.float16))

    def forward(self, x):
        y = x.matmul(self.weight.t())
        if self.bias is not None:
            y = y + self.bias
        return y

class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, device='cuda'):
        super().__init__()
        world = dist.get_world_size()
        rank = dist.get_rank()
        assert in_features % world == 0
        self.rank = rank
        self.world = world
        self.local_in = in_features // world
        self.weight = nn.Parameter(torch.empty(out_features, self.local_in, device=device, dtype=torch.float16))
        self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=torch.float16)) if bias else None
        nn.init.normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @torch.no_grad()
    def load_from_full(self, full_weight: torch.Tensor, full_bias=None):
        start = self.rank * self.local_in
        end = start + self.local_in
        self.weight.copy_(full_weight[:, start:end].to(device=self.weight.device, dtype=torch.float16))
        if self.bias is not None and full_bias is not None:
            self.bias.copy_(full_bias.to(device=self.weight.device, dtype=torch.float16))

    def forward(self, x_shard):
        y = x_shard.matmul(self.weight.t())
        if self.bias is not None:
            y = y + self.bias
        all_reduce_sum(y)
        return y


class ReplicatedLinear(nn.Module):
    """Full (non-sharded) weight replicated on every rank — used for KV heads when not divisible by world."""
    def __init__(self, in_features, out_features, bias=False, device='cuda'):
        super().__init__()
        self.rank = dist.get_rank()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=torch.float16))
        self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=torch.float16)) if bias else None
        nn.init.normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @torch.no_grad()
    def load_from_full(self, full_weight: torch.Tensor, full_bias=None):
        self.weight.copy_(full_weight.to(device=self.weight.device, dtype=torch.float16))
        if self.bias is not None and full_bias is not None:
            self.bias.copy_(full_bias.to(device=self.weight.device, dtype=torch.float16))

    def forward(self, x):
        y = x.matmul(self.weight.t())
        if self.bias is not None:
            y = y + self.bias
        return y
