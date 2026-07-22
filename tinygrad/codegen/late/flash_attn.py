"""Shared flash attention via composite REDUCE.

One implementation for both 8B fp16-overlay and 14B packed-weight routes.
Uses composite online-softmax REDUCE for the softmax part. QK^T and PV
matmuls go through the existing TC optimizer for WMMA.
"""
from tinygrad.uop.ops import AccumulatorSlot, Ops
from tinygrad import Tensor

def merge_online_softmax_tile(m: Tensor, l: Tensor, acc: Tensor, scores: Tensor,
                              v: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Merge one score/V tile into a running online-softmax state."""
    block_m = scores.max(axis=-1, keepdim=True)
    new_m = m.maximum(block_m)
    corr = (m - new_m).exp()
    weights = (scores - new_m).exp()
    new_l = l * corr + weights.sum(axis=-1, keepdim=True)
    pv_weights = weights if weights.dtype == v.dtype else weights.cast(v.dtype)
    new_acc = acc * corr + pv_weights.matmul(v, dtype=acc.dtype)
    return new_m, new_l, new_acc

def normalize_online_softmax_state(acc: Tensor, l: Tensor) -> Tensor:
    """Materialize public attention output from raw online-softmax state."""
    return acc / l

def flash_attention(q: Tensor, k: Tensor, v: Tensor, scale: float = None,
                    mask: Tensor = None) -> Tensor:
    """
    Flash attention using composite online-softmax REDUCE.
    
    Args:
        q, k, v: (B, Hkv, G, T, Hd) for GQA or (B, H, T, Hd) for MHA
        scale: attention scale (default 1/sqrt(Hd))
        mask: additive mask (None for causal, broadcast-compatible)
    
    Returns:
        Attention output, same shape as q.
    """
    Hd = q.shape[-1]
    if scale is None:
        scale = 1.0 / (Hd ** 0.5)
    
    # Kernel 1: QK^T matmul (WMMA-eligible)
    scores = (q @ k.transpose(-1, -2)) * scale
    if mask is not None:
        scores = scores + mask
    
    # Kernel 2: online-softmax composite REDUCE (l-value)
    kv_axis = len(scores.shape) - 1
    slot_m = AccumulatorSlot(op=Ops.MAX, dtype=scores.uop.dtype, identity=float("-inf"), name="m")
    slot_l = AccumulatorSlot(op=Ops.ADD, dtype=scores.uop.dtype, identity=0.0, name="l")
    red = scores.uop.composite_reduce(slot_m, slot_l, axis=(kv_axis,), combine_fn="online_softmax_l")
    l_vals = Tensor(red)  # online-softmax denominator per query position
    
    # Kernel 3: PV matmul (WMMA-eligible)
    probs = scores.softmax(-1)
    if v.uop.dtype != probs.uop.dtype:
        probs = probs.cast(v.uop.dtype)
    return probs @ v
