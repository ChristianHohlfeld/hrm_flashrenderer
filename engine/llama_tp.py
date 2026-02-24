import torch
import torch.nn as nn
import torch.distributed as dist

from flashattention_custom.ops import flash_attn_paged, paged_kv_append
from tp.parallel_linear import ColumnParallelLinear, RowParallelLinear, ReplicatedLinear
from tp.vocab_parallel import VocabParallelEmbedding, VocabParallelLMHead
from tp.norm import RMSNorm
from engine.rope import build_rope_cache, apply_rope


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


class PagedKVCache:
    """True paged KV (no gather). Layout [B,Hk,P,PS,D]."""

    def __init__(
        self,
        bsz: int,
        kv_heads: int,
        head_dim: int,
        page_size: int = 128,
        init_pages: int = 4,
        device: str = "cuda",
        dtype=torch.float16,
    ):
        self.bsz = int(bsz)
        self.kv_heads = int(kv_heads)
        self.head_dim = int(head_dim)
        self.page_size = int(page_size)
        self.device = device
        self.dtype = dtype

        P = max(1, int(init_pages))
        self.k_pages = torch.empty(self.bsz, self.kv_heads, P, self.page_size, self.head_dim, device=device, dtype=dtype).contiguous()
        self.v_pages = torch.empty(self.bsz, self.kv_heads, P, self.page_size, self.head_dim, device=device, dtype=dtype).contiguous()
        self.seqlen = torch.zeros((self.bsz,), device=device, dtype=torch.int32).contiguous()
        self.cur_len = 0

    def _ensure_pages(self, needed_pages: int):
        curP = int(self.k_pages.size(2))
        if needed_pages <= curP:
            return
        newP = _next_pow2(needed_pages)
        k_new = torch.empty(self.bsz, self.kv_heads, newP, self.page_size, self.head_dim, device=self.device, dtype=self.dtype).contiguous()
        v_new = torch.empty(self.bsz, self.kv_heads, newP, self.page_size, self.head_dim, device=self.device, dtype=self.dtype).contiguous()
        k_new[:, :, :curP, :, :] = self.k_pages
        v_new[:, :, :curP, :, :] = self.v_pages
        self.k_pages = k_new
        self.v_pages = v_new

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor, positions: torch.Tensor):
        """Append K/V for positions.

        Fast path: positions [T] contiguous => one CUDA kernel.
        General path: positions [B,T] scatter (rare).
        """
        B, Hk, T, D = k_new.shape
        assert B == self.bsz and Hk == self.kv_heads and D == self.head_dim

        if positions.dim() == 1:
            start = int(positions[0].item())
            end = start + int(T)
            needed_pages = (end + self.page_size - 1) // self.page_size
            self._ensure_pages(needed_pages)

            paged_kv_append(self.k_pages, self.v_pages, k_new.contiguous(), v_new.contiguous(), start_pos=start, page_size=self.page_size)

            self.cur_len = max(self.cur_len, end)
            self.seqlen.fill_(self.cur_len)
            return

        if positions.dim() != 2:
            raise ValueError("positions must be [T] or [B,T]")

        max_pos = int(positions.max().item()) + 1
        needed_pages = (max_pos + self.page_size - 1) // self.page_size
        self._ensure_pages(needed_pages)

        # deterministic scatter fallback
        for b in range(B):
            max_b = int(positions[b].max().item()) + 1
            if max_b > int(self.seqlen[b].item()):
                self.seqlen[b] = max_b
            for t in range(T):
                pos = int(positions[b, t].item())
                p = pos // self.page_size
                off = pos - p * self.page_size
                self.k_pages[b, :, p, off, :] = k_new[b, :, t, :]
                self.v_pages[b, :, p, off, :] = v_new[b, :, t, :]

        self.cur_len = max(self.cur_len, int(self.seqlen.max().item()))

    def get(self):
        return self.k_pages, self.v_pages, self.seqlen

    def reset(self):
        # keep allocated pages for performance; only reset logical length
        self.seqlen.zero_()
        self.cur_len = 0


class TPLlamaAttention(nn.Module):
    """Llama-style attention with TP and TRUE paged KV."""

    def __init__(self, hidden: int, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.hidden = int(hidden)
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)

        world = dist.get_world_size()
        rank = dist.get_rank()
        assert self.num_heads % world == 0, "num_heads must be divisible by world"
        self.rank = rank
        self.world = world
        self.local_heads = self.num_heads // world

        self.q_proj = ColumnParallelLinear(self.hidden, self.num_heads * self.head_dim, bias=False)

        # KV sharding: replicate if num_kv_heads not divisible by world
        if (self.num_kv_heads % world) == 0:
            self.kv_replicated = False
            self.local_kv_heads = self.num_kv_heads // world
            self.k_proj = ColumnParallelLinear(self.hidden, self.num_kv_heads * self.head_dim, bias=False)
            self.v_proj = ColumnParallelLinear(self.hidden, self.num_kv_heads * self.head_dim, bias=False)
            self.kv_head_offset = rank * self.local_kv_heads
        else:
            self.kv_replicated = True
            self.local_kv_heads = self.num_kv_heads
            self.k_proj = ReplicatedLinear(self.hidden, self.num_kv_heads * self.head_dim, bias=False)
            self.v_proj = ReplicatedLinear(self.hidden, self.num_kv_heads * self.head_dim, bias=False)
            self.kv_head_offset = 0

        self.o_proj = RowParallelLinear(self.num_heads * self.head_dim, self.hidden, bias=False)

        # head_map: local query heads -> local kv heads
        rep_global = self.num_heads // self.num_kv_heads
        hm = torch.empty((self.local_heads,), device="cuda", dtype=torch.int32)
        for hq in range(self.local_heads):
            gqh = rank * self.local_heads + hq
            gkh = gqh // rep_global
            lkh = gkh if self.kv_replicated else (gkh - self.kv_head_offset)
            hm[hq] = int(lkh)
        self.register_buffer("head_map", hm.contiguous(), persistent=False)

        if torch.any(hm < 0) or torch.any(hm >= self.local_kv_heads):
            raise RuntimeError("head_map out of range: world_size incompatible with kv layout")

    @torch.no_grad()
    def load_from_loader(self, loader, prefix: str):
        self.q_proj.load_from_full(loader.rows(prefix + "q_proj.weight", self.q_proj.rank * self.q_proj.local_out, (self.q_proj.rank + 1) * self.q_proj.local_out))

        if self.kv_replicated:
            self.k_proj.load_from_full(loader.get(prefix + "k_proj.weight"))
            self.v_proj.load_from_full(loader.get(prefix + "v_proj.weight"))
        else:
            self.k_proj.load_from_full(loader.rows(prefix + "k_proj.weight", self.k_proj.rank * self.k_proj.local_out, (self.k_proj.rank + 1) * self.k_proj.local_out))
            self.v_proj.load_from_full(loader.rows(prefix + "v_proj.weight", self.v_proj.rank * self.v_proj.local_out, (self.v_proj.rank + 1) * self.v_proj.local_out))

        self.o_proj.load_from_full(loader.cols(prefix + "o_proj.weight", self.o_proj.rank * self.o_proj.local_in, (self.o_proj.rank + 1) * self.o_proj.local_in))

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, positions: torch.Tensor, cache: PagedKVCache, prefill: bool):
        B, T, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.local_heads, self.head_dim).transpose(1, 2).contiguous()
        k = k.view(B, T, self.local_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.view(B, T, self.local_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        pos_1d = positions if positions.dim() == 1 else positions[0]
        q = apply_rope(q, cos, sin, pos_1d)
        k = apply_rope(k, cos, sin, pos_1d)

        cache.append(k, v, positions)
        k_pages, v_pages, seqlen = cache.get()

        q_offset = int(pos_1d[0].item())
        out = flash_attn_paged(
            q=q,
            k_pages=k_pages,
            v_pages=v_pages,
            seqlen=seqlen,
            page_size=cache.page_size,
            head_map=self.head_map,
            causal=bool(prefill),
            q_offset=q_offset,
            k_offset=0,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, self.local_heads * self.head_dim)
        return self.o_proj(out)


class TPLlamaMLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate = ColumnParallelLinear(hidden, intermediate, bias=False)
        self.up = ColumnParallelLinear(hidden, intermediate, bias=False)
        self.down = RowParallelLinear(intermediate, hidden, bias=False)

    @torch.no_grad()
    def load_from_loader(self, loader, prefix: str):
        self.gate.load_from_full(loader.rows(prefix + "gate_proj.weight", self.gate.rank * self.gate.local_out, (self.gate.rank + 1) * self.gate.local_out))
        self.up.load_from_full(loader.rows(prefix + "up_proj.weight", self.up.rank * self.up.local_out, (self.up.rank + 1) * self.up.local_out))
        self.down.load_from_full(loader.cols(prefix + "down_proj.weight", self.down.rank * self.down.local_in, (self.down.rank + 1) * self.down.local_in))

    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))


class TPLlamaBlock(nn.Module):
    def __init__(self, hidden: int, num_heads: int, num_kv_heads: int, head_dim: int, intermediate: int, eps: float):
        super().__init__()
        self.n1 = RMSNorm(hidden, eps)
        self.attn = TPLlamaAttention(hidden, num_heads, num_kv_heads, head_dim)
        self.n2 = RMSNorm(hidden, eps)
        self.mlp = TPLlamaMLP(hidden, intermediate)

    @torch.no_grad()
    def load_from_loader(self, loader, prefix: str):
        self.n1.load_from_full(loader.get(prefix + "input_layernorm.weight"))
        self.attn.load_from_loader(loader, prefix + "self_attn.")
        self.n2.load_from_full(loader.get(prefix + "post_attention_layernorm.weight"))
        self.mlp.load_from_loader(loader, prefix + "mlp.")

    def forward(self, x, cos, sin, positions, cache: PagedKVCache, prefill: bool):
        x = x + self.attn(self.n1(x), cos, sin, positions, cache, prefill=prefill)
        x = x + self.mlp(self.n2(x))
        return x


class TPLlamaForCausalLM(nn.Module):
    def __init__(self, config, max_seq_len: int):
        super().__init__()
        self.config = config
        self.vocab_size = int(config.vocab_size)
        self.hidden = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(getattr(config, "num_key_value_heads", self.num_heads))
        self.head_dim = self.hidden // self.num_heads
        self.layers = int(config.num_hidden_layers)
        self.intermediate = int(getattr(config, "intermediate_size", self.hidden * 4))
        self.eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.rope_theta = float(getattr(config, "rope_theta", 10000.0))
        self.max_seq_len = int(max_seq_len)

        self.embed = VocabParallelEmbedding(self.vocab_size, self.hidden)
        self.blocks = nn.ModuleList([
            TPLlamaBlock(self.hidden, self.num_heads, self.num_kv_heads, self.head_dim, self.intermediate, self.eps)
            for _ in range(self.layers)
        ])
        self.norm = RMSNorm(self.hidden, self.eps)
        self.lm_head = VocabParallelLMHead(self.hidden, self.vocab_size)

        self.cos, self.sin = build_rope_cache(self.max_seq_len, self.head_dim, theta=self.rope_theta, device="cuda", dtype=torch.float16)
        self.caches: list[PagedKVCache] | None = None

    @torch.no_grad()
    def load_from_loader(self, loader):
        world = dist.get_world_size()
        rank = dist.get_rank()
        vocab_per = self.vocab_size // world
        v0 = rank * vocab_per
        v1 = v0 + vocab_per

        self.embed.load_from_full(loader.rows("model.embed_tokens.weight", v0, v1))
        for i, blk in enumerate(self.blocks):
            blk.load_from_loader(loader, f"model.layers.{i}.")
        self.norm.load_from_full(loader.get("model.norm.weight"))
        try:
            self.lm_head.load_from_full(loader.rows("lm_head.weight", v0, v1))
        except Exception:
            # some models tie lm_head with embed
            self.lm_head.load_from_full(loader.rows("model.embed_tokens.weight", v0, v1))

    def _ensure_cache(self, bsz: int, init_pages: int, page_size: int):
        if self.caches is not None:
            return
        self.caches = []
        for blk in self.blocks:
            kvh = blk.attn.local_kv_heads
            self.caches.append(PagedKVCache(bsz, kvh, self.head_dim, page_size=page_size, init_pages=init_pages, device="cuda", dtype=torch.float16))

    def reset_all_caches(self):
        # reuse allocations across requests (daemon mode)
        if self.caches is None:
            return
        for c in self.caches:
            c.reset()

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, prefill: bool, init_pages: int = 4, page_size: int = 128):
        B, T = input_ids.shape
        self._ensure_cache(B, init_pages=init_pages, page_size=page_size)
        x = self.embed(input_ids)
        assert self.caches is not None
        for i, blk in enumerate(self.blocks):
            x = blk(x, self.cos, self.sin, positions, self.caches[i], prefill=prefill)
        return self.norm(x)

    def logits_last(self, hidden_last: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_last)


def greedy_select_token_distributed(local_logits: torch.Tensor) -> torch.Tensor:
    """Deterministic distributed argmax without full vocab gather."""
    rank = dist.get_rank()
    local_logits_f = local_logits.float()
    local_max_val, local_max_idx = torch.max(local_logits_f, dim=-1)
    global_max_val = local_max_val.clone()
    dist.all_reduce(global_max_val, op=dist.ReduceOp.MAX)

    eps = 0.0
    is_winner = (local_max_val >= global_max_val - eps)
    winner_rank = torch.full_like(local_max_idx, fill_value=10**9, dtype=torch.long)
    winner_rank[is_winner] = rank
    dist.all_reduce(winner_rank, op=dist.ReduceOp.MIN)

    win_idx = local_max_idx.to(dtype=torch.long)
    for b in range(win_idx.numel()):
        dist.broadcast(win_idx[b:b+1], src=int(winner_rank[b].item()))

    return win_idx + winner_rank * local_logits.size(-1)


@torch.no_grad()
def generate_tp(model: TPLlamaForCausalLM, input_ids: torch.Tensor, max_new_tokens: int, eos_token_id: int | None, prefill_chunk_size: int = 1024):
    # In daemon mode, caches persist across requests. Reset logical lengths.
    model.reset_all_caches()
    B, T0 = input_ids.shape
    cur_len = 0
    last = None
    init_pages = max(4, (T0 + max_new_tokens + 127) // 128)

    # chunked prefill
    while cur_len < T0:
        end = min(T0, cur_len + prefill_chunk_size)
        chunk = input_ids[:, cur_len:end]
        pos = torch.arange(cur_len, end, device=input_ids.device, dtype=torch.long)
        h = model(chunk, pos, prefill=True, init_pages=init_pages, page_size=128)
        last = h[:, -1, :]
        cur_len = end

    assert last is not None
    out_ids = [input_ids]

    # decode
    for _ in range(max_new_tokens):
        local_logits = model.logits_last(last)
        next_id = greedy_select_token_distributed(local_logits)
        out_ids.append(next_id.view(B, 1))

        if eos_token_id is not None and int(next_id[0].item()) == int(eos_token_id):
            break
        if cur_len >= model.max_seq_len:
            break

        pos = torch.tensor([cur_len], device=input_ids.device, dtype=torch.long)
        h1 = model(next_id.view(B, 1), pos, prefill=False, init_pages=init_pages, page_size=128)
        last = h1[:, -1, :]
        cur_len += 1

    return torch.cat(out_ids, dim=1)
