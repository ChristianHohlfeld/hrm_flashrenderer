import torch

def build_rope_cache(max_seq_len: int, head_dim: int, theta: float = 10000.0, device: str = "cuda", dtype=torch.float16):
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos = torch.cos(freqs).to(dtype=dtype)
    sin = torch.sin(freqs).to(dtype=dtype)
    return cos, sin

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, positions: torch.Tensor):
    B,H,T,D = x.shape
    half = D // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    c = cos.index_select(0, positions).view(1,1,T,half)
    s = sin.index_select(0, positions).view(1,1,T,half)
    y1 = x1 * c - x2 * s
    y2 = x1 * s + x2 * c
    return torch.cat([y1, y2], dim=-1)
