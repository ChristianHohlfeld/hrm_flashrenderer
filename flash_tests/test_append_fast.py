import torch
from flashattention_custom.ops import paged_kv_append


def main():
    torch.manual_seed(0)
    device = "cuda"
    B=2; Hk=4; P=8; PS=128; D=128; T=96
    start=128+17

    k_pages = torch.zeros(B,Hk,P,PS,D, device=device, dtype=torch.float16).contiguous()
    v_pages = torch.zeros(B,Hk,P,PS,D, device=device, dtype=torch.float16).contiguous()
    k_new = torch.randn(B,Hk,T,D, device=device, dtype=torch.float16).contiguous()
    v_new = torch.randn(B,Hk,T,D, device=device, dtype=torch.float16).contiguous()

    k_ref = k_pages.clone()
    v_ref = v_pages.clone()
    for t in range(T):
        pos = start + t
        p = pos // PS
        off = pos - p*PS
        k_ref[:, :, p, off, :] = k_new[:, :, t, :]
        v_ref[:, :, p, off, :] = v_new[:, :, t, :]

    paged_kv_append(k_pages, v_pages, k_new, v_new, start_pos=start, page_size=PS)

    dk = (k_pages - k_ref).abs().max().item()
    dv = (v_pages - v_ref).abs().max().item()
    print("append-fast max|diff| K:", dk, "V:", dv)
    assert dk == 0.0 and dv == 0.0


if __name__ == "__main__":
    main()
