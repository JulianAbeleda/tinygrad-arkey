"""Central home for the Qwen3 fused prefill-attention feature.

WHY THIS FILE EXISTS
--------------------
The fused-attention logic was smeared across the general compiler (rangeify.py,
indexing.py, composite_combines.py, devectorizer.py, postrange.py, wmma.py) plus
model.py and flash_prefill_attention.py -- ~5.8k lines, interwoven with code that
has nothing to do with attention. That is why pinpointing the "class-2" failure
took so long. This module centralizes the FEATURE (routing, eligibility, and the
custom-kernel injection route) so there is ONE place to read, change, and debug
fused attention. It does NOT refactor the general compiler (too risky, unneeded).

ROUTING (the one decision point)
--------------------------------
The model calls exactly one entry: `route_prefill_attention(q, k, v, ...)`. It
chooses, in order:
  1. CUSTOM-KERNEL INJECTION (this module, `custom_kernel_attention`) -- inject the
     already-proven captured kernel via Tensor.custom_kernel. Attention becomes an
     opaque fp16 buffer-in/buffer-out CALL. The compiler realizes Q/K/V as ordinary
     buffers (the working path); NO composite reduce, so NONE of the class-2
     reach-through / store-forwarding / cycle failures can occur.
  2. COMPOSITE-SEMANTIC (legacy/dormant) -- `shared_prefill_attention` ->
     `q._semantic_attention` -> `lower_attention_semantic` (rangeify.py). This is
     the path that hits class-2; kept for reference, OFF the critical path.
  3. SDPA FALLBACK -- ordinary `q.scaled_dot_product_attention`. Always correct.

DTYPE IS ORTHOGONAL
-------------------
The injected kernel is a pure fp16 island (half* Q/K/V in, half* out). All
Q4_K/dequant/quant dtype handling stays UPSTREAM in the existing projection
kernels. Do not add dtype lowering here.

MAP OF THE SCATTERED CODE THIS CENTRALIZES / REPLACES
-----------------------------------------------------
- Entry + eligibility (GQA/grid admission): flash_prefill_attention.py:shared_prefill_attention
- Model call site + candidate-context build: llm/model.py:600-618 (_attention, prefill_tc_attn branch)
- Geometry/admission spec: uop/ops.py:AMDAttentionGridSpec (+ SharedAttentionCandidateContext)
- (legacy) semantic lowering: schedule/rangeify.py:19-197 lower_attention_semantic
- (legacy) range-assignment V handling: schedule/indexing.py:132 (SCOPED_VALUE branch)  <-- class-2 site
- (legacy) combine + V-lane packing: codegen/late/composite_combines.py (online_softmax_state, _pack_online_softmax_v_lanes)
- (legacy) devectorize V load: codegen/late/devectorizer.py:385-570 (_vectorize_live_v_index, _load_v_at_reduce_pos)
- (legacy) native swap to the hand kernel: codegen/opt/postrange.py:328-361 -> schedule/wmma.py:545
- The proven kernel source + ABI (the "base"): produced by extra/qk/generate_shared_attention_captures
  (emits .hip.cpp/.amdisa.s + JSON; ABI = out[slot0], Q[slot1], K[slot2], V[slot3], scale/causal baked CONST)
- Loud class-2 diagnostic (safety net): uop/ops.py DISALLOW_BROADCAST site (ScopedValueSpec vs rank-0)

custom_kernel CONTRACT (verified, tensor.py:194 / uop/ops.py:1256)
------------------------------------------------------------------
  out_buf.custom_kernel(q, k, v, fxn=emit)[0]
  - each src is .contiguous()'d -> realized to a real buffer (opaque to the kernel)
  - placeholders (one param slot per src) are handed to fxn(*placeholders)
  - fxn(*placeholders).call(*srcs) binds the real buffers and yields the CALL
  - returns [s.after(kernel) for s in srcs]; index [0] (out_buf) is the result
"""
from __future__ import annotations
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import AMDAttentionGridSpec, SharedAttentionCandidateContext

# ADMITTED GEOMETRIES (Hq, Hkv, q_tokens) for which a captured kernel exists / is
# generatable. Extend as the capture matrix grows (see B7 in the scope doc).
ADMITTED_GRIDS: frozenset = frozenset({(32, 8, 512), (40, 8, 512)})


def prefill_grid_spec(q:Tensor, k:Tensor) -> AMDAttentionGridSpec | None:
  """Return the admitted grid spec for (q,k), else None (-> caller falls back)."""
  if not (q.shape[0] == 1 and all(isinstance(x, int) for x in
          (q.shape[-3], q.shape[-2], q.shape[-1], k.shape[-3], k.shape[-2]))):
    return None
  if k.shape[-3] == 0 or q.shape[-3] % k.shape[-3]:
    return None
  spec = AMDAttentionGridSpec(q_tokens=q.shape[-2], q_heads=q.shape[-3], kv_heads=k.shape[-3],
    group_ratio=q.shape[-3] // k.shape[-3], kv_tokens=k.shape[-2], head_dim=q.shape[-1])
  try:
    spec.validate()
  except ValueError:
    return None
  return spec if (spec.q_heads, spec.kv_heads, spec.q_tokens) in ADMITTED_GRIDS else None


def custom_kernel_attention(q:Tensor, k:Tensor, v:Tensor, *, scale:float|None, causal:bool,
                            ctx:SharedAttentionCandidateContext) -> Tensor:
  """Inject the proven fused-attention kernel via Tensor.custom_kernel.

  Q/K/V arrive fp16 (1, H, T, 128); returns fp16 (1, Hq, T, 128). custom_kernel
  .contiguous()'s each input into a real buffer that the kernel consumes opaquely
  -> no composite reduce -> none of the class-2 reach-through/forwarding/cycle.

  Mechanism (verified against postrange.py:328 + Tensor.custom_kernel):
  - The proven UOp builder `amd_gfx1100_q16_grid_hd128_loop_attention(q,k,v,out,...)`
    IS the custom_kernel `fxn` body. It requires BARE PARAM owners with slots
    (Q,K,V,out)=(1,2,3,0), so we pass FLAT 1-D buffers (placeholder_like keeps
    1-D tensors as bare PARAM; multi-dim would become RESHAPE(PARAM) and fail).
  - custom_kernel(out_flat; q_flat,k_flat,v_flat) assigns slots 0,1,2,3 -> exactly
    out=0, Q=1, K=2, V=3.
  """
  from tinygrad.schedule.wmma import amd_gfx1100_q16_grid_hd128_loop_attention
  from tinygrad.uop.ops import KernelInfo
  grid = prefill_grid_spec(q, k)
  if grid is None: raise NotImplementedError("custom_kernel_attention: geometry not admitted")
  Hq, Hkv, T, KV, Hd = grid.q_heads, grid.kv_heads, grid.q_tokens, grid.kv_tokens, grid.head_dim
  # Fail-safe geometry cross-check: the ctx (which drives the causal boundary) must agree with the
  # ACTUAL tensor shapes. For a growing unpadded prefill cache kv_tokens == start_pos + q_tokens. If a
  # padded/ring KV buffer ever violates this, fall back (route_prefill_attention catches -> SDPA) rather
  # than silently mis-mask. Variable kv_tokens is otherwise free (a fresh kernel compiles per length).
  if not (grid.kv_tokens == ctx.kv_tokens == ctx.start_pos + ctx.q_tokens and grid.q_tokens == ctx.q_tokens):
    raise NotImplementedError(f"custom_kernel_attention: ctx/tensor geometry mismatch "
      f"(grid kv={grid.kv_tokens} q={grid.q_tokens}; ctx kv={ctx.kv_tokens} q={ctx.q_tokens} start={ctx.start_pos})")
  sc = float(scale) if scale is not None else 1.0 / (Hd ** 0.5)
  q_flat = q.cast(dtypes.half).reshape(Hq * T * Hd)
  k_flat = k.cast(dtypes.half).reshape(Hkv * KV * Hd)
  v_flat = v.cast(dtypes.half).reshape(Hkv * KV * Hd)

  def emit(out_ph, q_ph, k_ph, v_ph):
    # Thread valid_kv/query_start from ctx explicitly (== the builder's no-padding defaults today, but
    # robust for start_pos>0 chunks: the per-row causal boundary is ctx.start_pos, not an incidental
    # kv_tokens-q_tokens equality).
    return amd_gfx1100_q16_grid_hd128_loop_attention(
      q_ph, k_ph, v_ph, out_ph, q_tokens=T, q_heads=Hq, kv_heads=Hkv, kv_tokens=KV,
      scale=sc, causal=causal, valid_kv=ctx.kv_tokens, query_start=ctx.start_pos,
      output_block_base=ctx.output_block_base, acc_blocks=ctx.acc_blocks,
      kernel_info=KernelInfo(name="amd_gfx1100_q16_grid_hd128_loop_attention"))

  out_flat = Tensor.empty(Hq * T * Hd, dtype=dtypes.half, device=q.device)
  result = out_flat.custom_kernel(q_flat, k_flat, v_flat, fxn=emit)[0]
  return result.reshape(1, Hq, T, Hd)


def sdpa_fallback(q:Tensor, k:Tensor, v:Tensor, *, scale:float|None, mask:Tensor|None) -> Tensor:
  return q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)


def route_prefill_attention(q:Tensor, k:Tensor, v:Tensor, *, scale:float|None=None, mask:Tensor|None=None,
                            causal:bool=False, ctx:SharedAttentionCandidateContext|None=None,
                            use_custom_kernel:bool=False) -> Tensor:
  """THE single entry the model calls. Chooses injection / (legacy) semantic / SDPA.

  q/k/v are fp16 at this boundary (the model casts Q->half; K/V are fp16). Result is
  fp16; the caller casts back to the original dtype (as the SDPA path does today).
  """
  grid = prefill_grid_spec(q, k)
  if use_custom_kernel and grid is not None and ctx is not None:
    try:
      return custom_kernel_attention(q, k, v, scale=scale, causal=causal, ctx=ctx)
    except NotImplementedError:
      pass  # until B1-B4 land, fall through to the proven paths
  return sdpa_fallback(q, k, v, scale=scale, mask=mask)
