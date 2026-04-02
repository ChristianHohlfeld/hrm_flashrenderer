# Copyright (c) 2026 Christian Heinrich Hohlfeld (Konstanz, Germany)
# https://christianhohlfeld.com | ORCID: https://orcid.org/0009-0003-6634-9045
# ALL RIGHTS RESERVED. No license granted without prior written permission.
import math
import unittest

try:
    import torch
    from flashattention_custom.ops import flash_attn, flash_attn_paged
except Exception as e:  # pragma: no cover - environment dependent
    raise unittest.SkipTest(f"flash kernel tests unavailable: {e}")


def ref_attn(q,k,v,causal:bool, q_offset:int=0, k_offset:int=0):
    B,H,M,D = q.shape
    N = k.shape[2]
    scale = 1.0/math.sqrt(D)
    scores = torch.matmul(q, k.transpose(-1,-2)) * scale
    if causal:
        qpos = torch.arange(q_offset, q_offset+M, device=q.device)
        kpos = torch.arange(k_offset, k_offset+N, device=q.device)
        mask = kpos.view(1,1,1,N) > qpos.view(1,1,M,1)
        scores = scores.masked_fill(mask, float("-inf"))
    p = torch.softmax(scores, dim=-1)
    return torch.matmul(p, v)


def main():
    torch.manual_seed(0)
    device="cuda"
    PS=128

    for D in [64,128]:
        B=1; Hq=8; Hk=8; M=128; N=256
        q = torch.randn(B,Hq,M,D, device=device, dtype=torch.float16).contiguous()
        k = torch.randn(B,Hq,N,D, device=device, dtype=torch.float16).contiguous()
        v = torch.randn(B,Hq,N,D, device=device, dtype=torch.float16).contiguous()

        out = flash_attn(q,k,v,causal=True,q_offset=0,k_offset=0)
        ref = ref_attn(q.float(),k.float(),v.float(),causal=True,q_offset=0,k_offset=0).half()
        diff = (out-ref).abs()
        print("contig D",D,"max",diff.max().item(),"mean",diff.mean().item())
        assert diff.max().item() < (0.12 if D==128 else 0.10)

        P = (N + PS - 1)//PS
        k_pages = torch.zeros(B,Hk,P,PS,D, device=device, dtype=torch.float16).contiguous()
        v_pages = torch.zeros(B,Hk,P,PS,D, device=device, dtype=torch.float16).contiguous()
        k_pages.view(B,Hk,P*PS,D)[:, :, :N, :] = k[:, :Hk, :, :]
        v_pages.view(B,Hk,P*PS,D)[:, :, :N, :] = v[:, :Hk, :, :]

        seqlen = torch.tensor([N], device=device, dtype=torch.int32).contiguous()
        head_map = torch.arange(Hq, device=device, dtype=torch.int32).contiguous()

        outp = flash_attn_paged(q, k_pages, v_pages, seqlen, PS, head_map, causal=True, q_offset=0, k_offset=0)
        diffp = (outp-ref).abs()
        print("paged D",D,"max",diffp.max().item(),"mean",diffp.mean().item())
        assert diffp.max().item() < (0.12 if D==128 else 0.10)

        q2 = torch.randn(B,Hq,64,D, device=device, dtype=torch.float16).contiguous()
        outp2 = flash_attn_paged(q2, k_pages, v_pages, seqlen, PS, head_map, causal=True, q_offset=128, k_offset=0)
        ref2 = ref_attn(q2.float(), k.float(), v.float(), causal=True, q_offset=128, k_offset=0).half()
        diffp2 = (outp2-ref2).abs()
        print("paged offset D",D,"max",diffp2.max().item(),"mean",diffp2.mean().item())
        assert diffp2.max().item() < (0.12 if D==128 else 0.10)

    print("OK")


if __name__=="__main__":
    main()

