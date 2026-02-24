# Copyright (c) 2026 Christian Heinrich Hohlfeld (Konstanz, Germany)
# https://christianhohlfeld.com | ORCID: https://orcid.org/0009-0003-6634-9045
# ALL RIGHTS RESERVED. No license granted without prior written permission.
from . import _ext


def flash_attn(q, k, v, causal=True, q_offset=0, k_offset=0):
    return _ext.flash_attn_fwd(q, k, v, bool(causal), int(q_offset), int(k_offset))


def flash_attn_paged(q, k_pages, v_pages, seqlen, page_size, head_map, causal=True, q_offset=0, k_offset=0):
    return _ext.flash_attn_paged_fwd(q, k_pages, v_pages, seqlen, int(page_size), head_map, bool(causal), int(q_offset), int(k_offset))


def paged_kv_append(k_pages, v_pages, k_new, v_new, start_pos, page_size):
    _ext.paged_kv_append(k_pages, v_pages, k_new, v_new, int(start_pos), int(page_size))

