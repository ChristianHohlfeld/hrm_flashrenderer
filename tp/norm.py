import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device='cuda', dtype=torch.float16))

    @torch.no_grad()
    def load_from_full(self, full_weight: torch.Tensor):
        self.weight.copy_(full_weight.to(device='cuda', dtype=torch.float16))

    def forward(self, x):
        var = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps).to(dtype=x.dtype)
        return x * self.weight
