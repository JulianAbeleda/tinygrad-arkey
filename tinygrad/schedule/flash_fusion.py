"""
Flash-fusion scheduler primitive (Option B — scheduler-native).
Intercepts in Tensor.realize(), detects attention score-spill BUFFERs
in the LINEAR graph. B-M1: detect. B-M2: merge + WMMA.
"""
from tinygrad.uop.ops import UOp, Ops
from tinygrad.helpers import getenv, DEBUG

_TK = getenv("TINYGRAD_FLASH_TK", 128)

def _safe_key(node: UOp) -> str:
    try: return node.key.hex()[:16]
    except: return str(id(node))

def _has_reduce(sink: UOp) -> bool:
    try: return any(u.op is Ops.REDUCE for u in sink.toposort())
    except: return False

def _find_attention_candidates(linear: UOp) -> list:
    """Find SINK pairs connected by a BUFFER (producer→consumer)."""
    calls = []
    buf_to_calls = {}
    for child in linear.src:
        if child.op is not Ops.CALL: continue
        sink = child.src[0] if child.src else None
        if sink is None or sink.op is Ops.PROGRAM: continue
        bufs = [s for s in child.src[1:] if s.op is Ops.BUFFER]
        calls.append((child, sink, bufs))
        for b in bufs:
            k = _safe_key(b)
            if k not in buf_to_calls: buf_to_calls[k] = []
            buf_to_calls[k].append((child, sink))
    candidates = []
    for k, pairs in buf_to_calls.items():
        if len(pairs) != 2: continue  # BUFFER shared by exactly 2 CALLs
        (pc, ps), (cc, cs) = pairs
        if _has_reduce(ps) and _has_reduce(cs):
            candidates.append({"qk_sink": ps, "pv_sink": cs, "score_buf_key": k})
    return candidates

def flash_fusion_rewrite_linear(linear: UOp) -> UOp:
    candidates = _find_attention_candidates(linear)
    if candidates and DEBUG >= 2:
        print(f"  [flash_fusion] {len(candidates)} attention candidate(s) detected in LINEAR graph")
    return linear  # B-M1: pass-through, no rewrite yet

flash_fusion_rewrite = flash_fusion_rewrite_linear