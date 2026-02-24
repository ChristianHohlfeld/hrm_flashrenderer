# Copyright (c) 2026 Christian Heinrich Hohlfeld (Konstanz, Germany)
# https://christianhohlfeld.com | ORCID: https://orcid.org/0009-0003-6634-9045
# ALL RIGHTS RESERVED. No license granted without prior written permission.
import math
import torch

try:
    from . import _ext as _ext
    _HAS_EXT = True
except (ImportError, OSError):
    _ext = None
    _HAS_EXT = False


# ---------------------------------------------------------------------------
# Pure-PyTorch fallbacks (used when CUDA extension not built)
# ---------------------------------------------------------------------------

def _ref_flash_attn(q, k, v, causal=True, q_offset=0, k_offset=0):
    """Standard scaled dot-product attention (CPU/non-CUDA fallback)."""
    B, H, M, D = q.shape
    N = k.shape[2]
    scale = 1.0 / math.sqrt(D)
    scores = q.float() @ k.float().transpose(-1, -2) * scale  # [B,H,M,N]
    if causal:
        q_pos = torch.arange(q_offset, q_offset + M, device=q.device)
        k_pos = torch.arange(k_offset, k_offset + N, device=q.device)
        mask = k_pos.view(1, 1, 1, N) > q_pos.view(1, 1, M, 1)
        scores = scores.masked_fill(mask, float("-inf"))
    p = torch.softmax(scores, dim=-1).to(q.dtype)
    return p @ v


def _ref_flash_attn_paged(q, k_pages, v_pages, seqlen, page_size, head_map,
                          causal=True, q_offset=0, k_offset=0):
    """Paged-KV attention — pure-PyTorch fallback for CPU-only mode."""
    B, Hq, M, D = q.shape
    _B2, Hk, P, PS, _D2 = k_pages.shape
    out = torch.zeros_like(q)
    for b in range(B):
        slen = int(seqlen[b].item())
        # Reconstruct a flat [Hk, slen, D] view from pages
        total_pages = (slen + page_size - 1) // page_size
        k_flat = k_pages[b, :, :total_pages].reshape(Hk, total_pages * page_size, D)[:, :slen, :]
        v_flat = v_pages[b, :, :total_pages].reshape(Hk, total_pages * page_size, D)[:, :slen, :]
        scale = 1.0 / math.sqrt(D)
        for hq in range(Hq):
            hk = int(head_map[hq].item())
            q_h = q[b, hq]          # [M, D]
            k_h = k_flat[hk]        # [slen, D]
            v_h = v_flat[hk]        # [slen, D]
            scores = q_h.float() @ k_h.float().t() * scale  # [M, slen]
            if causal:
                q_pos = torch.arange(q_offset, q_offset + M, device=q.device)
                k_pos = torch.arange(k_offset, k_offset + slen, device=q.device)
                mask = k_pos.unsqueeze(0) > q_pos.unsqueeze(1)
                scores = scores.masked_fill(mask, float("-inf"))
            p = torch.softmax(scores, dim=-1).to(q.dtype)
            out[b, hq] = p @ v_h.to(q.dtype)
    return out


def _ref_paged_kv_append(k_pages, v_pages, k_new, v_new, start_pos, page_size):
    """Write k_new/v_new into paged cache at start_pos (CPU fallback)."""
    B, Hk, T, D = k_new.shape
    for b in range(B):
        for t in range(T):
            pos = start_pos + t
            p = pos // page_size
            off = pos % page_size
            k_pages[b, :, p, off, :] = k_new[b, :, t, :]
            v_pages[b, :, p, off, :] = v_new[b, :, t, :]


# ---------------------------------------------------------------------------
# Public API — routes to CUDA kernel or fallback
# ---------------------------------------------------------------------------

def flash_attn(q, k, v, causal=True, q_offset=0, k_offset=0):
    if _HAS_EXT:
        return _ext.flash_attn_fwd(q, k, v, bool(causal), int(q_offset), int(k_offset))
    return _ref_flash_attn(q, k, v, causal=causal, q_offset=q_offset, k_offset=k_offset)


def flash_attn_paged(q, k_pages, v_pages, seqlen, page_size, head_map,
                     causal=True, q_offset=0, k_offset=0):
    if _HAS_EXT:
        return _ext.flash_attn_paged_fwd(q, k_pages, v_pages, seqlen,
                                         int(page_size), head_map,
                                         bool(causal), int(q_offset), int(k_offset))
    return _ref_flash_attn_paged(q, k_pages, v_pages, seqlen, page_size, head_map,
                                 causal=causal, q_offset=q_offset, k_offset=k_offset)


def paged_kv_append(k_pages, v_pages, k_new, v_new, start_pos, page_size):
    if _HAS_EXT:
        _ext.paged_kv_append(k_pages, v_pages, k_new, v_new, int(start_pos), int(page_size))
    else:
        _ref_paged_kv_append(k_pages, v_pages, k_new, v_new, start_pos, page_size)
