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
     proven fused-attention program via Tensor.uop_program. The kernel is built as
     ordinary UOps by FlashPrefillAttentionSpec.emit() through the target-keyed
     emitter seam (per-target fragment geometry comes from the fragment model, not
     captured machine code) and lowered by the renderer like any other program.
     Attention becomes an opaque fp16 buffer-in/buffer-out CALL. The compiler
     realizes Q/K/V as ordinary buffers (the working path); NO composite reduce,
     so NONE of the class-2 reach-through / store-forwarding / cycle failures can
     occur.
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
- Geometry/admission spec: uop/ops.py:AttentionGridSpec (+ SharedAttentionCandidateContext)
- (legacy) semantic lowering: schedule/rangeify.py:19-197 lower_attention_semantic
- (legacy) range-assignment V handling: schedule/indexing.py:132 (SCOPED_VALUE branch)  <-- class-2 site
- (legacy) combine + V-lane packing: codegen/late/composite_combines.py (online_softmax_state, _pack_online_softmax_v_lanes)
- (legacy) devectorize V load: codegen/late/reduce_lowering.py (_vectorize_live_v_index, _load_v_at_reduce_pos)
- (legacy) native swap to the hand kernel: codegen/opt/postrange.py:328-361 -> schedule/wmma.py:545
- The kernel topology + ABI (the "base"): FlashPrefillAttentionSpec
  (schedule/wmma/flash_prefill.py) owns the topology as DATA and builds the kernel
  as UOps through the emitter seam (ABI = out[slot0], Q[slot1], K[slot2], V[slot3],
  scale/causal baked CONST). Per-target fragment decomposition comes from the
  fragment model (codegen/opt/attention_fragment.py), not from captured machine code.
- Loud class-2 diagnostic (safety net): uop/ops.py DISALLOW_BROADCAST site (ScopedValueSpec vs rank-0)

PROMOTED PROGRAM BOUNDARY (verified, tensor.py:194 / uop/ops.py:1256)
-----------------------------------------------------------------
  execute_promoted_program(out_buf, q, k, v, program=program)
  - each src is .contiguous()'d -> realized to a real buffer (opaque to the kernel)
  - placeholders (one param slot per src) are handed to fxn(*placeholders)
  - fxn(*placeholders).call(*srcs) binds the real buffers and yields the CALL
  - returns [s.after(kernel) for s in srcs]; index [0] (out_buf) is the result
"""
from __future__ import annotations
from contextvars import ContextVar
from typing import Any
from tinygrad import Tensor, dtypes
from tinygrad.device import Device
from tinygrad.uop.ops import AttentionGridSpec, SharedAttentionCandidateContext
from extra.llm_research.boltbeam_authority import tickets_for_candidate
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, OutputSpec, execute_promoted_program

# ADMITTED GEOMETRIES (Hq, Hkv, q_tokens) for which the fragment-model kernel exists /
# is generatable. Extend as the proven matrix grows (see B7 in the scope doc). This
# stays the proven-on-GPU shape allowlist; FlashPrefillAttentionSpec.validate() (below)
# is a SECOND, independent geometric-legality gate -- admission requires BOTH, so
# today's admitted set is unchanged (the allowlist is strictly narrower than what
# validate() alone would accept).
ADMITTED_GRIDS: frozenset = frozenset({(32, 8, 512), (40, 8, 512)})

# TARGET-KEYED EMITTER DISPATCH (the multi-GPU seam): a FlashPrefillAttentionSpec
# resolves its custom_kernel-shaped emitter fxn by spec.target. One entry today
# (amd_gfx1100); a second GPU is a new dict entry + a per-target emitter, a modular
# add rather than a rewrite of this routing code. **kw passes through to spec.emit()
# (e.g. kernel_info=...) so every call site -- including postrange.py's AST-swap,
# which needs to inject its own carried-forward KernelInfo -- routes through this
# SAME seam instead of ever calling the gfx1100 builder directly.
_PREFILL_EMITTERS = {"amd_gfx1100": lambda spec, **kw: spec.emit(**kw),
                     "nv_sm120": lambda spec, **kw: spec.emit(**kw)}

# DEVICE -> SPEC TARGET (the fused-attention fragment-model registry key, codegen/opt/attention_fragment.py).
# The spec target is resolved from the LIVE renderer's declared (backend, arch) facts, never from the device
# string: the key is the registry's own naming convention -- backend lowercase + arch with underscores removed
# ("amd_gfx1100", "nv_sm120") -- so a new GPU is a new fragment-model + emitter entry, not a backend branch.
# Resolve once and cache (decode's pattern, decode_routes.py:151: Device[device] cannot be opened inside a
# Tensor Function dispatch). Unknown devices fail closed (ValueError -> NotImplementedError -> SDPA fallback).
_ATTENTION_SPEC_TARGETS: dict[str, str] = {}

def _attention_spec_target(device: str) -> str:
  """Derive the fused-attention spec target key from the live renderer's facts for ``device``."""
  if (cached := _ATTENTION_SPEC_TARGETS.get(device)) is not None: return cached
  try: renderer = Device[device].renderer
  except Exception: renderer = None
  if renderer is None: raise ValueError(f"cannot resolve a fused-attention target for device {device!r}")
  target = f"{renderer.target.device.lower()}_{renderer.target.arch.replace('_', '')}"
  _ATTENTION_SPEC_TARGETS[device] = target
  return target

def warm_attention_spec_target(device: str) -> str:
  """Eagerly resolve and cache the attention spec target for ``device`` at model-load time.

  ``custom_kernel_attention`` runs inside a Tensor Function dispatch (ALLOW_DEVICE_USAGE=0), where
  ``Device[device]`` cannot be opened -- the same constraint decode's ``bind`` resolves once at load
  (decode_routes.py:151). Call this from the eager admission path so the runtime lookup is a cache hit.
  """
  return _attention_spec_target(device)

# RUNTIME DISPATCH TRACE (BoltBeam observability seam)
# --------------------------------------------------
# tinygrad/llm/prefill_graph_gemm.py owns the candidate route census.
# mechanism (record_model_forward_candidate), but it is purpose-built for the
# dense-GEMM packed-WMMA roles: (a) it is a no-op unless one_buffer=True, a flag
# that specifically means "this candidate shares one canonical weight-buffer
# identity across the whole-model forward" -- a packed-WMMA weight concept that
# does not apply to attention's Q/K/V/out buffers; and (b) finalize_candidate_
# route_census cross-checks recorded rows against `registry.admissions`, i.e. the
# dense candidate_set registry -- attention has no such registry entry. Reusing
# that seam for attention would either require lying about one_buffer (to avoid
# the no-op) or would record nothing at all (one_buffer=False, per the no-op
# gate). Rather than overload that seam's semantics, this is a small,
# attention-specific trace: a dispatch counter + last-geometry identity, so
# extra/llm_research/prefill/prefill_whole_synced.py can read (not import extra/llm_research/ eagerly from
# here -- see custom_kernel_attention below) whether/how many times the fused
# custom-kernel route actually fired during a census window.
_CUSTOM_KERNEL_ATTENTION_DISPATCH_COUNT: ContextVar[int] = ContextVar(
  "custom_kernel_attention_dispatch_count", default=0)
_CUSTOM_KERNEL_ATTENTION_LAST_IDENTITY: ContextVar[str | None] = ContextVar(
  "custom_kernel_attention_last_identity", default=None)


def _record_custom_kernel_attention_dispatch(identity: str) -> None:
  _CUSTOM_KERNEL_ATTENTION_DISPATCH_COUNT.set(_CUSTOM_KERNEL_ATTENTION_DISPATCH_COUNT.get() + 1)
  _CUSTOM_KERNEL_ATTENTION_LAST_IDENTITY.set(identity)


def custom_kernel_attention_dispatch_count() -> int:
  """How many times custom_kernel_attention has successfully dispatched (this context)."""
  return _CUSTOM_KERNEL_ATTENTION_DISPATCH_COUNT.get()


def custom_kernel_attention_last_identity() -> str | None:
  """The canonical geometry identity of the most recent successful dispatch, if any."""
  return _CUSTOM_KERNEL_ATTENTION_LAST_IDENTITY.get()


def custom_kernel_attention_trace_snapshot() -> dict[str, Any]:
  """Read-only snapshot for reporting (e.g. prefill_whole_synced.py's report dict)."""
  return {"dispatches": _CUSTOM_KERNEL_ATTENTION_DISPATCH_COUNT.get(),
          "last_identity": _CUSTOM_KERNEL_ATTENTION_LAST_IDENTITY.get()}


def reset_custom_kernel_attention_trace() -> None:
  _CUSTOM_KERNEL_ATTENTION_DISPATCH_COUNT.set(0)
  _CUSTOM_KERNEL_ATTENTION_LAST_IDENTITY.set(None)


def prefill_grid_spec(q:Tensor, k:Tensor) -> AttentionGridSpec | None:
  """Return the admitted grid spec for (q,k), else None (-> caller falls back)."""
  if not (q.shape[0] == 1 and all(isinstance(x, int) for x in
          (q.shape[-3], q.shape[-2], q.shape[-1], k.shape[-3], k.shape[-2]))):
    return None
  if k.shape[-3] == 0 or q.shape[-3] % k.shape[-3]:
    return None
  spec = AttentionGridSpec(q_tokens=q.shape[-2], q_heads=q.shape[-3], kv_heads=k.shape[-3],
    group_ratio=q.shape[-3] // k.shape[-3], kv_tokens=k.shape[-2], head_dim=q.shape[-1])
  try:
    spec.validate()
  except ValueError:
    return None
  return spec if (spec.q_heads, spec.kv_heads, spec.q_tokens) in ADMITTED_GRIDS else None


def custom_kernel_attention(q:Tensor, k:Tensor, v:Tensor, *, scale:float|None, causal:bool,
                            ctx:SharedAttentionCandidateContext) -> Tensor:
  """Inject the proven fused-attention program via Tensor.uop_program.

  Q/K/V arrive fp16 (1, H, T, 128); returns fp16 (1, Hq, T, 128). custom_kernel
  .contiguous()'s each input into a real buffer that the kernel consumes opaquely
  -> no composite reduce -> none of the class-2 reach-through/forwarding/cycle.

  Mechanism (verified against postrange.py:328 + Tensor.uop_program):
  - A FlashPrefillAttentionSpec (tinygrad/schedule/wmma/flash_prefill.py) owns the
    topology as DATA and composes the proven UOp builder
    `amd_gfx1100_q16_grid_hd128_loop_attention(q,k,v,out,...)` via `spec.emit()`,
    resolved through the target-keyed `_PREFILL_EMITTERS` dispatch. It requires BARE
    PARAM owners with slots (Q,K,V,out)=(1,2,3,0), so we pass FLAT 1-D buffers
    (placeholder_like keeps 1-D tensors as bare PARAM; multi-dim would become
    RESHAPE(PARAM) and fail).
  - uop_program(out_flat; q_flat,k_flat,v_flat) assigns slots 0,1,2,3 -> exactly
    out=0, Q=1, K=2, V=3.
  """
  from tinygrad.schedule.wmma.flash_prefill import FlashPrefillAttentionSpec
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

  # The SPEC owns the topology as data: it threads valid_kv/query_start from ctx explicitly (==
  # the builder's no-padding defaults today, but robust for start_pos>0 chunks: the per-row causal
  # boundary is ctx.start_pos, not an incidental kv_tokens-q_tokens equality). Admission is now a
  # BOTH-gate: ADMITTED_GRIDS (proven-shapes allowlist, via prefill_grid_spec above) AND
  # spec.validate() (geometric legality) must both hold; validate() alone accepts any %16<=4096
  # geometry, so ADMITTED_GRIDS is what keeps this to the proven 512 shapes -- unchanged behavior.
  #
  # ctx.acc_blocks defaults to 8 (SharedAttentionCandidateContext's own default, the Hd=128
  # full-accumulator value) when a caller doesn't override it. Forwarding that literal unconditionally
  # would break the full-drain case at any Hd!=128 (hd_blocks != 8) even though the caller meant "give
  # me the full accumulator", not literally 8 blocks. Detect the full-accumulator-default case
  # (output_block_base==0 and acc_blocks==8, i.e. ctx never overrode either) and pass None so
  # FlashPrefillAttentionSpec.__post_init__ resolves it via the SAME hd//16 formula the spec already
  # uses -- byte-identical at Hd=128 (None -> 128//16==8, the exact value that would have been
  # forwarded). An explicit non-full slice (output_block_base!=0, or an acc_blocks the caller
  # intentionally set to something other than 8) is preserved exactly as given.
  ctx_full_default = ctx.output_block_base == 0 and ctx.acc_blocks == 8
  try:
    spec_target = _attention_spec_target(q.device)
  except ValueError as e:
    raise NotImplementedError(f"custom_kernel_attention: {e}") from None
  spec = FlashPrefillAttentionSpec(Hq=Hq, Hkv=Hkv, Hd=Hd, q_tokens=T, kv_tokens=KV, causal=causal, scale=sc,
    valid_kv=ctx.kv_tokens, query_start=ctx.start_pos, output_block_base=ctx.output_block_base,
    acc_blocks=None if ctx_full_default else ctx.acc_blocks, target=spec_target)
  try:
    spec.validate()
  except ValueError as e:
    raise NotImplementedError(f"custom_kernel_attention: spec rejected geometry ({e})")
  try:
    emitter = _PREFILL_EMITTERS[spec.target]
  except KeyError:
    raise NotImplementedError(f"custom_kernel_attention: no emitter registered for target {spec.target!r}")
  fxn = emitter(spec)

  q_flat = q.cast(dtypes.float16).reshape(Hq * T * Hd)
  k_flat = k.cast(dtypes.float16).reshape(Hkv * KV * Hd)
  # V VECTORIZATION (PREFILL_V_TRANSPOSED): uop_program already .contiguous()'s each input into a
  # fresh buffer, so materializing V as [Hkv][Hd][KV] instead of [Hkv][KV][Hd] costs one transposed
  # copy in place of a free reshape -- and turns the emitter's 128 2-byte V gathers per KV tile into
  # 16 b128 loads (see amd_attention_abi.expand_loop_fragment). Element count is identical.
  from tinygrad.helpers import getenv as _getenv
  v_flat = (v.cast(dtypes.float16).permute(0, 1, 3, 2).reshape(Hkv * Hd * KV) if _getenv("PREFILL_V_TRANSPOSED")
            else v.cast(dtypes.float16).reshape(Hkv * KV * Hd))
  identity = (f"{spec.target}_q16_grid_hd128_loop_attention:role=attention_tile,"
              f"Hq={Hq},Hkv={Hkv},q_tokens={T},kv_tokens={KV},Hd={Hd}")
  program = KernelProgram("prefill_flash_attention_generated", f"prefill_flash_attention.{identity}",
    KernelProgramProvenance.MACHINE_SEARCH_GENERATED, fxn,
    output_spec=OutputSpec((Hq * T * Hd,), dtypes.float16),
    boltbeam_ticket=tickets_for_candidate({"family":"flash_prefill.v1","identity":identity,"kv_tokens":KV},
      (("custom_kernel_prefill_attention","flash_prefill_score"),
       ("custom_kernel_prefill_attention","flash_prefill_combine"))))
  result = execute_promoted_program(None, q_flat, k_flat, v_flat, program=program)
  # Record the dispatch AFTER every geometry/spec gate above has passed (i.e. only
  # once we know this call is committed to the fused custom-kernel route, not a
  # NotImplementedError fallback). role="attention_tile" matches the manifest row's
  # role naming convention used elsewhere for prefill roles.
  _record_custom_kernel_attention_dispatch(identity)
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
