import torch
import torch.nn as nn
import torch.distributed as dist
from .dist_utils import all_reduce_sum

class VocabParallelEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        world = dist.get_world_size()
        rank = dist.get_rank()
        assert vocab_size % world == 0
        self.vocab_size = vocab_size
        self.hidden = hidden_size
        self.rank = rank
        self.world = world
        self.vocab_per_rank = vocab_size // world
        self.vocab_start = rank * self.vocab_per_rank
        self.vocab_end = self.vocab_start + self.vocab_per_rank
        self.weight = nn.Parameter(torch.empty(self.vocab_per_rank, hidden_size, device='cuda', dtype=torch.float16))
        nn.init.normal_(self.weight, std=0.02)

    @torch.no_grad()
    def load_from_full(self, full_weight: torch.Tensor):
        self.weight.copy_(full_weight[self.vocab_start:self.vocab_end].to(device='cuda', dtype=torch.float16))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        ids = input_ids
        mask = (ids >= self.vocab_start) & (ids < self.vocab_end)
        local_ids = (ids - self.vocab_start).clamp(min=0, max=self.vocab_per_rank-1)
        emb = torch.nn.functional.embedding(local_ids, self.weight)
        emb = emb * mask.unsqueeze(-1).to(dtype=emb.dtype)
        all_reduce_sum(emb)
        return emb

class VocabParallelLMHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        world = dist.get_world_size()
        rank = dist.get_rank()
        assert vocab_size % world == 0
        self.rank = rank
        self.world = world
        self.vocab_per_rank = vocab_size // world
        self.vocab_start = rank * self.vocab_per_rank
        self.vocab_end = self.vocab_start + self.vocab_per_rank
        self.weight = nn.Parameter(torch.empty(self.vocab_per_rank, hidden_size, device='cuda', dtype=torch.float16))
        nn.init.normal_(self.weight, std=0.02)

    @torch.no_grad()
    def load_from_full(self, full_weight: torch.Tensor):
        self.weight.copy_(full_weight[self.vocab_start:self.vocab_end].to(device='cuda', dtype=torch.float16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.matmul(self.weight.t())
