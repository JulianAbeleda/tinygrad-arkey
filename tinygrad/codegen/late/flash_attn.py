"""Shared flash attention via composite REDUCE.

One implementation for both 8B fp16-overlay and 14B packed-weight routes.
Uses composite online-softmax REDUCE for the softmax part. QK^T and PV
matmuls go through the existing TC optimizer for WMMA.
"""
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
    # The state reducer carries l as one scalar per (B,H,T), while acc keeps
    # the logical head-dimension lane.  Add exactly that lane axis; relying on
    # generic left-aligned broadcasting is incorrect for Hd > 1.
    if len(acc.shape) == len(l.shape) + 1 and tuple(acc.shape[:-1]) == tuple(l.shape):
      l = l.reshape(*l.shape, 1)
    return acc / l
