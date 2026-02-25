import torch
import math
from flashattention_custom.ops import flash_attn_paged, paged_kv_append

def prove_custom_flash():
    print("--- Proving Custom Flash Implementation ---")
    
    # Initialize parameters
    B, Hq, Hk, D = 1, 8, 4, 64
    T = 10
    page_size = 8
    device = "cuda"
    dtype = torch.float32
    
    # Create dummy local q, k, v
    q = torch.randn(B, Hq, T, D, device=device, dtype=dtype)
    k = torch.randn(B, Hk, T, D, device=device, dtype=dtype)
    v = torch.randn(B, Hk, T, D, device=device, dtype=dtype)
    
    # Create Paged KV Cache structures
    P = (T + page_size - 1) // page_size + 1
    k_pages = torch.zeros(B, Hk, P, page_size, D, device=device, dtype=dtype)
    v_pages = torch.zeros(B, Hk, P, page_size, D, device=device, dtype=dtype)
    seqlen = torch.zeros(B, device=device, dtype=torch.int32)
    
    # Create head map (8 query heads -> 4 KV heads)
    head_map = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], device=device, dtype=torch.int32)
    
    print(f"Step 1: Appending {T} tokens to KV cache...")
    paged_kv_append(k_pages, v_pages, k, v, start_pos=0, page_size=page_size)
    seqlen.fill_(T)
    
    print("Step 2: Running paged attention...")
    out = flash_attn_paged(
        q=q,
        k_pages=k_pages,
        v_pages=v_pages,
        seqlen=seqlen,
        page_size=page_size,
        head_map=head_map,
        causal=True
    )
    
    print(f"Step 3: Verification...")
    print(f"Output shape: {out.shape}")
    assert out.shape == q.shape, "Output shape mismatch!"
    
    is_zero = torch.all(out == 0)
    print(f"Is output zero? {is_zero}")
    
    # Check if a different query produces a different result
    q2 = torch.randn_like(q)
    out2 = flash_attn_paged(q2, k_pages, v_pages, seqlen, page_size, head_map, causal=True)
    is_same = torch.all(out == out2)
    print(f"Is output same for different query? {is_same}")
    
    if not is_zero and not is_same:
        print("SUCCESS: Custom Flash Implementation (CPU fallback) is functional.")
    else:
        print("FAILURE: Implementation returned zeros or invariant results.")

if __name__ == "__main__":
    prove_custom_flash()
