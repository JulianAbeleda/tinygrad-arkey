"""Phase 1 Step 2: Blocked online-softmax attention in pure tinygrad Tensor ops.
No UOp construction, no flash_kernels imports, no hand kernel. Just Tensor ops.
The Python loop over KV blocks is intentional — we want to see what the scheduler does."""

import os; os.environ.setdefault('DEV','AMD')
from tinygrad import Tensor, dtypes, Device

def blocked_online_softmax(q: Tensor, k: Tensor, v: Tensor, BLK: int = 128):
    """Online-softmax attention over KV blocks. Pure Tensor ops.

    q: (B, Hkv, G, T, Hd) or (B, H, T, Hd)
    k: (B, Hkv, 1, KV, Hd) or (B, H, KV, Hd)
    v: (B, Hkv, 1, KV, Hd) or (B, H, KV, Hd)
    Returns: attention output, same shape as q
    """
    # Detect shape layout
    if q.ndim == 5:
        B, Hkv, G, T, Hd = q.shape
        KV = k.shape[-2]
        # Reduce to 4D for the block loop (single head group)
        q4 = q.reshape(B * Hkv * G, T, Hd)
        k4 = k.reshape(B * Hkv, 1, KV, Hd).expand(B * Hkv, G, KV, Hd).reshape(B * Hkv * G, KV, Hd)
        v4 = v.reshape(B * Hkv, 1, KV, Hd).expand(B * Hkv, G, KV, Hd).reshape(B * Hkv * G, KV, Hd)
    else:
        B_H, T, Hd = q.shape
        KV = k.shape[-2]
        q4, k4, v4 = q, k, v
        G = 1; Hkv = 1

    scale = Hd ** -0.5

    # Initialize running state
    m = Tensor.full((*q4.shape[:-1], 1), float("-inf"), dtype=dtypes.float32, device=q.device)
    l = Tensor.zeros((*q4.shape[:-1], 1), dtype=dtypes.float32, device=q.device)
    acc = Tensor.zeros((*q4.shape[:-1], Hd), dtype=dtypes.float32, device=q.device)

    # Block loop over KV
    for j in range(0, KV, BLK):
        j_end = min(j + BLK, KV)
        kb = k4[..., j:j_end, :].contiguous()  # (B*H*G, BLK, Hd)
        vb = v4[..., j:j_end, :].contiguous()  # (B*H*G, BLK, Hd)

        # QK^T for this block: (BHG, T, Hd) @ (BHG, Hd, BLK) -> (BHG, T, BLK)
        s = (q4 @ kb.transpose(-1, -2)).float() * scale  # (BHG, T, BLK)

        # Causal mask: additive -inf for positions where query < KV
        rows = Tensor.arange(T, dtype=dtypes.int32, device=q.device).reshape(1, T, 1)
        cols = Tensor.arange(j, j_end, dtype=dtypes.int32, device=q.device).reshape(1, 1, j_end - j)
        causal = rows >= cols  # (1, T, BLK) — True where query can attend
        s = causal.where(s, Tensor.full_like(s, float("-inf")))

        # Online-softmax merge
        s_max = s.max(-1, keepdim=True)  # (BHG, T, 1)
        m_new = m.maximum(s_max)
        corr = (m - m_new).exp()  # correction factor
        p = (s - m_new).exp()  # (BHG, T, BLK)
        l_new = l * corr + p.sum(-1, keepdim=True)
        acc = acc * corr + (p @ vb.float())
        m = m_new
        l = l_new
        # Force realization to prevent graph build-up
        acc.realize(); m.realize(); l.realize()
        Device[Device.DEFAULT].synchronize()

    # Final output: acc / l
    out = (acc / l)

    # Reshape back to input format
    if q.ndim == 5:
        out = out.reshape(B, Hkv, G, T, Hd).cast(dtypes.float16)
    else:
        out = out.cast(dtypes.float16)

    return out


# Quick test
if __name__ == "__main__":
    import numpy as np
    T, KV = 256, 256  # tiny for test
    Hd = 128
    q = Tensor.randn(1, 1, 1, T, Hd, dtype=dtypes.float16).contiguous().realize()
    k = Tensor.randn(1, 1, 1, KV, Hd, dtype=dtypes.float16).contiguous().realize()
    v = Tensor.randn(1, 1, 1, KV, Hd, dtype=dtypes.float16).contiguous().realize()
    out = blocked_online_softmax(q, k, v, BLK=128)
    out.realize(); Device[Device.DEFAULT].synchronize()
    print(f"Blocked softmax output: shape={out.shape}, mean={out.numpy().mean():.4f}")
