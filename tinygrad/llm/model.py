from __future__ import annotations
import collections, functools, itertools, json, os, pathlib
from dataclasses import dataclass, replace
from tinygrad import Tensor, nn, UOp, TinyJit, dtypes, getenv, function, Device
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.helpers import prod
from tinygrad.llm.gguf import gguf_load, gguf_load_with_metadata
from tinygrad.uop.ops import resolve

# Prefill v2 (opt-in, default off; decode 100% untouched when off). Concrete-ubatch fp16 prefill that lets
# tensor cores apply + the loop-found TC schedule warm-start in. See docs/amd-decode-prefill-v2-gate-20260616.md.
# Costs ~fp16-covered-weight-size extra VRAM (it coexists with the Q4_K decode storage; ~+14GB for 8B) -> OOMs
# small cards. `PREFILL_V2=auto` resolves on/off from detected VRAM in from_gguf (see prefill_v2_auto_decision +
# docs/prefill-default-policy-evaluation-result-20260620.md). Explicit 0/1 always wins.
_PREFILL_V2_ENV = os.environ.get("PREFILL_V2", "")
PREFILL_V2_AUTO = _PREFILL_V2_ENV.strip().lower() == "auto"
PREFILL_V2 = False if PREFILL_V2_AUTO else bool(getenv("PREFILL_V2", 0))
def _set_prefill_v2(val:bool):   # auto policy resolves the module global before Transformer() is constructed
  global PREFILL_V2; PREFILL_V2 = val
def _detect_total_vram_bytes() -> int|None:
  # cheap one-shot total-VRAM probe via rocm-smi; None on any failure -> auto stays conservatively OFF.
  try:
    import subprocess
    out = subprocess.run(["rocm-smi", "--showmeminfo", "vram"], capture_output=True, text=True, timeout=10).stdout
    for ln in out.splitlines():
      if "VRAM Total Memory" in ln: return int(ln.split(":")[-1].strip())
  except Exception: return None
  return None
def prefill_v2_auto_decision(total_vram_bytes:int|None, est_fp16_bytes:int, q4_bytes:int, kv_bytes:int,
                             min_total_gb:float=23.0, margin_gb:float=3.0) -> tuple[bool, str]:
  # Conservative first-pass policy: enable PREFILL_V2 only on a clearly large card (>=~24GB) where the full
  # footprint (Q4 storage + realized fp16 covered weights + KV) fits with a margin for activations/scores/decode.
  if total_vram_bytes is None: return (False, "VRAM unknown (rocm-smi unavailable) -> conservative OFF")
  need = q4_bytes + est_fp16_bytes + kv_bytes
  tot_gb, need_gb = total_vram_bytes/1e9, need/1e9
  if total_vram_bytes < min_total_gb*1e9:
    return (False, f"total {tot_gb:.1f}GB < {min_total_gb:.0f}GB floor -> OFF (PREFILL_V2 +fp16 would risk OOM)")
  if total_vram_bytes < need + margin_gb*1e9:
    return (False, f"need {need_gb:.1f}GB + {margin_gb:.0f}GB margin > {tot_gb:.1f}GB total -> OFF")
  return (True, f"need {need_gb:.1f}GB + {margin_gb:.0f}GB margin <= {tot_gb:.1f}GB total -> ON")
PREFILL_UBATCH = getenv("PREFILL_UBATCH", 512)  # concrete token batch; warmstart keys use this N
# Phase-3 routing fix (default ON under PREFILL_V2): route a sub-UBATCH prompt remainder through ONE shifted
# prefill-v2 chunk instead of the slow 32-token symbolic fallback. PREFILL_REMAINDER_FIX=0 reverts. See
# docs/prefill-route-schedule-result-20260620.md.
PREFILL_REMAINDER_FIX = bool(getenv("PREFILL_REMAINDER_FIX", 1))
# Research-only (default off): route eligible PREFILL_V2 prefill matmuls through an extracted rocBLAS Tensile kernel
# via HCQ (external artifact). NOT a default/ship path. See docs/prefill-tensile-a3-inmodel-route-scope-20260619.md.
PREFILL_TENSILE_GEMM = bool(getenv("PREFILL_TENSILE_GEMM", 0))
# Restricted default-ON (within PREFILL_V2): route eligible fp16 prefill matmuls through the dependency-free
# graph-capturable RDNA3 GEMM. Passed all 4 default-on readiness gates (synced 1.61x, 0/256 greedy mismatches,
# 11/11 fallback, 5/5 OOM; docs/prefill-graph-gemm-default-on-readiness-result-20260620.md). Default-on is
# GUARDED to gfx1100 (the validated arch), decided ONCE HERE at import -- NOT in the route, because Device[...]
# access is disallowed during JIT capture. Non-gfx1100/other-device -> default off. Unsupported shapes (T!=512,
# non-tile-divisible, bias, ineligible role) silently fall back to the normal PREFILL_V2 matmul. Only active when
# PREFILL_V2 is on. Set PREFILL_GRAPH_GEMM=0/1 to override. Absolute-parity drift (max_abs_dNLL 0.0176) accepted
# report-only (greedy byte-identical; harmful dNLL <= 0.0094).
def _prefill_graph_gemm_default() -> int:
  if "PREFILL_GRAPH_GEMM" in os.environ: return getenv("PREFILL_GRAPH_GEMM", 0)   # explicit user override
  try:                                                                            # restricted default-on: gfx1100 only
    if Device.DEFAULT != "AMD": return 0
    return 1 if "gfx1100" in str(getattr(Device["AMD"], "arch", "")) else 0
  except Exception: return 0
PREFILL_GRAPH_GEMM = bool(_prefill_graph_gemm_default())

# Route B B4 (default-off): the owned hand-AMDGCN flash-decode tile injected as external precompiled Ops.PROGRAM JIT
# graph nodes. The device/arch guard is decided ONCE at import (Device[...] access is disallowed during JIT capture);
# the per-call route adds only shape checks + the env flag. gfx1100-only (the validated arch). See
# docs/decode-attention-route-b-b4-external-graph-node-result-20260621.md.
def _decode_attn_amdgcn_arch_ok() -> bool:
  try: return Device.DEFAULT == "AMD" and "gfx1100" in str(getattr(Device["AMD"], "arch", ""))
  except Exception: return False
DECODE_ATTN_AMDGCN_ARCH_OK = _decode_attn_amdgcn_arch_ok()
# Concrete-KV prefill (opt-in, default off): pass a CONCRETE start_pos per prefill chunk so KV=start_pos+T is
# concrete -> the attention's reduce tiles/TC fires (symbolic KV blocks it). ~1.24x e2e, byte-identical. Cost: a
# separate concrete prefill jit per distinct start_pos (0,512,...), precompiled at load -> best WARM/server prefill
# but a load-time precompile tax that loses for cold one-shot short prompts. `PREFILL_CONCRETE_KV=auto` enables it
# only under the server profile (see prefill_concrete_kv_auto_decision). See
# docs/prefill-default-policy-evaluation-result-20260620.md, docs/prefill-concrete-kv-policy-result-20260620.md.
# PREFILL_SERVER_PROFILE=1 is a convenience: serve >1 generation / long prompts -> implies PREFILL_V2=auto (if V2
# unset) + concrete-KV on (when V2 ends up on). One-shot short prompts must NOT set it.
PREFILL_SERVER_PROFILE = bool(getenv("PREFILL_SERVER_PROFILE", 0))
_PREFILL_CKV_ENV = os.environ.get("PREFILL_CONCRETE_KV", "")
PREFILL_CONCRETE_KV_AUTO = _PREFILL_CKV_ENV.strip().lower() == "auto"
PREFILL_CONCRETE_KV = False if PREFILL_CONCRETE_KV_AUTO else bool(getenv("PREFILL_CONCRETE_KV", 0))
def _set_prefill_concrete_kv(val:bool):
  global PREFILL_CONCRETE_KV; PREFILL_CONCRETE_KV = val
def prefill_concrete_kv_auto_decision(server_profile:bool, prefill_v2_on:bool) -> tuple[bool, str]:
  # Precompile pays off only across repeated/long generation, which can't be detected at load -> the auto signal
  # is the explicit server profile. (Cold one-shot short prompts should leave it off; the precompile load tax loses.)
  if not prefill_v2_on: return (False, "PREFILL_V2 off -> concrete-KV moot, OFF")
  if server_profile: return (True, "server profile + PREFILL_V2 on -> precompile concrete jits, ON")
  return (False, "no server profile (one-shot assumed) -> OFF; set PREFILL_SERVER_PROFILE=1 or PREFILL_CONCRETE_KV=1")
# P2: explicit TC attention (Q@Kᵀ TC + fp16 scores + softmax + P@V TC, GQA broadcast) for prefill on CONCRETE KV
# (the only regime where the concrete-shape tensor core fires; symbolic KV blocked it -> 0.79x in-model). Needs
# PREFILL_CONCRETE_KV. Research, dNLL-gated. See docs/prefill-concrete-kv-build-scope-20260619.md.
# DEFAULT-ON, GUARDED to gfx1100 (validated arch), decided ONCE at import (not in the route -- Device[...] is
# disallowed during JIT capture). Only active under PREFILL_V2; the route's isinstance(start_pos,int) guard keeps
# it to CONCRETE chunks (start_pos=0 by default), the only validated regime. Set PREFILL_TC_ATTN=0/1 to override.
# Concrete first chunk: cuts attention ~18%->~5%, reproducible ~1.16x whole-forward, BYTE-IDENTICAL (rel_RMSE 0.0,
# dNLL 0.0, greedy-exact, 3 sessions). A FUSION win, NOT tensor cores (WMMA does not fire; no warmstart TC-opt for
# attention shapes, no BEAM). See docs/prefill-branch-b-tc-attention-result-20260620.md.
def _prefill_tc_attn_default() -> int:
  if "PREFILL_TC_ATTN" in os.environ: return getenv("PREFILL_TC_ATTN", 0)   # explicit user override
  try:                                                                       # restricted default-on: gfx1100 only
    if Device.DEFAULT != "AMD": return 0
    return 1 if "gfx1100" in str(getattr(Device["AMD"], "arch", "")) else 0
  except Exception: return 0
PREFILL_TC_ATTN = bool(_prefill_tc_attn_default())
Q8_FFN_HANDWRITTEN = bool(getenv("Q8_FFN_HANDWRITTEN", 0))
DECODE_MMVQ_IMPORT_Q4 = bool(getenv("DECODE_MMVQ_IMPORT_Q4", 0))
# HISTORY: the earlier env `PREFILL_TC_ATTENTION` probe reported ~0.8x "REFUTED in-model" -- that was a BROKEN
# harness: it set the typo'd env `PREFILL_TC_ATTENTION` (model reads PREFILL_TC_ATTN) so both arms ran SDPA, AND
# it bound a symbolic start_pos that fails the concrete-int guard so the path never fired. Overturned 2026-06-20
# (correct concrete-int, same-process interleaved synced A/B). See docs/prefill-branch-b-tc-attention-result-20260620.md.
# The loop-found per-shape TC schedule (gate-validated; NO BEAM -- BEAM hangs gfx1100). Forced onto the
# prefill-v2 fp16 matmuls via _WARMSTART_OPTS by shape key. The contraction-heavy shapes (in>out, e.g.
# ffn_down 4096x12288) want UPCAST(0,4); the rest UPCAST(0,2) -- using one schedule for all drops the chain
# to ~9% (verified). See docs/amd-decode-prefill-v2-gate-20260616.md.
def _prefill_v2_opts(out_f:int, in_f:int) -> tuple:
  # UNROLL(reduce,8): unrolling the K loop makes each thread's global->LDS copy loads contiguous, so they fold
  # from per-element global_load_d16 (+ ~8 v_mov register-init/WMMA) to wide global_load_b128 (~2 v_mov/WMMA).
  # +3.7% pp512, no VGPR spill (UNROLL,4 spills 362), dNLL -0.00013. See docs/prefill-cgw3-copy-unroll-result-20260619.md.
  return (Opt(OptOps.TC, 0, (-1, 2, 1)), Opt(OptOps.UPCAST, 0, 4 if in_f > out_f else 2), Opt(OptOps.UPCAST, 1, 4),
          Opt(OptOps.UNROLL, 0, 8))

# Increment 1 only measured/validated the TC schedule at ubatch 512. The warmstart key encodes the ubatch,
# so a different size would silently reuse a schedule found for 512 -> reject until other sizes are measured.
_PREFILL_V2_VALIDATED_UBATCH = (512,)
def _prefill_v2_validate_ubatch(ubatch:int) -> None:
  if ubatch not in _PREFILL_V2_VALIDATED_UBATCH:
    raise ValueError(f"PREFILL_V2 only validates PREFILL_UBATCH in {_PREFILL_V2_VALIDATED_UBATCH} (got {ubatch}); "
                     f"the warmstart TC schedule is shape-specific. Re-measure per-shape opts for {ubatch} first "
                     f"(extra/qk_prefill_gate.py) and add it to _PREFILL_V2_VALIDATED_UBATCH.")

# fp16 realization coexists with the Q4_K decode storage (~fp16-model-size extra VRAM). Preflight it so an
# oversized model (14B/32B) fails fast with an actionable error instead of OOMing late mid-realize.
def _prefill_v2_realize_bytes(shapes:list[tuple[int,int]]) -> int: return sum(o * i for o, i in shapes) * 2  # fp16

def _pf16(lin, x:Tensor) -> Tensor:
  # prefill v2: a single fp16 matmul (both operands fp16 -> RDNA3 WMMA tensor cores can fire). The primitives'
  # `.weight` is a LAZY Q4_K/Q6_K->fp16 dequant graph (not a realized buffer); using it directly fuses the
  # whole dequant into the matmul -> bandwidth/dequant-bound ~3% peak (no TC win). So we realize a clean fp16
  # weight ONCE per linear (cached as `_pf16_w` by _install_prefill_v2_warmstart) and matmul against that.
  w = getattr(lin, "_pf16_w", None)
  if PREFILL_GRAPH_GEMM and w is not None:
    from extra.qk_prefill_graph_gemm_route import route_pf16_graph_gemm
    routed = route_pf16_graph_gemm(lin, x)
    if routed is not None: return routed
  if PREFILL_TENSILE_GEMM and w is not None:   # research-only external Tensile route (flag-gated, eligible shapes only)
    from extra.qk_tensile_inmodel import route_pf16
    routed = route_pf16(lin, x)
    if routed is not None: return routed       # else fall through to the normal fp16 matmul (silent fallback)
  if w is None: w = lin.weight.cast(dtypes.float16)   # fallback (uncached): lazy, slow -- expect the cache
  b = getattr(lin, "bias", None)
  return x.cast(dtypes.float16).linear(w.transpose(), b.cast(dtypes.float16) if b is not None else None)

def _ffn_tensile_col(block, x:Tensor):
  """Transpose-free Tensile FFN (research, PREFILL_TENSILE_GEMM): keep gate/up/down in [feature,T] (column)
  so the gate/up output-transpose + down input-transpose cancel (the diagnostic-localized prefill win).
  Transpose x ONCE at entry, result ONCE at exit. Returns None (silent fallback) if any role is ineligible."""
  from extra.qk_tensile_inmodel import route_pf16_col
  gw = getattr(block.ffn_gate, "_pf16_w", None)
  if gw is None or x.ndim < 2 or not isinstance(x.shape[-2], int) or x.shape[-2] != 512: return None
  D, T = gw.shape[1], x.shape[-2]
  xT = x.reshape(T, D).cast(dtypes.float16).transpose().contiguous()      # [D, T] (one entry transpose)
  g = route_pf16_col(block.ffn_gate, xT); u = route_pf16_col(block.ffn_up, xT)
  if g is None or u is None: return None
  h = (g.silu() * u).contiguous()                                        # [hidden, T] (down's A, no transpose)
  o = route_pf16_col(block.ffn_down, h)                                  # [dim, T]
  if o is None: return None
  return o.transpose().reshape(*x.shape[:-1], D)                         # [..., dim] (one exit transpose)

@functools.cache
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device:str|None=None) -> Tensor:
  freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))
  freqs = Tensor.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
  return freqs.cos().cat(freqs.sin(), dim=-1).clone(device)

class ExpertWeights:
  """Like nn.Linear but with num_experts dimension. Weight shape: (num_experts, out_features, in_features)."""
  def __init__(self, num_experts:int, in_features:int, out_features:int):
    self.weight = Tensor.zeros(num_experts, out_features, in_features)
  def __call__(self, sel:Tensor, x:Tensor) -> Tensor:
    # sel: (B, T, k), x: (B, T, 1, in) or (B, T, k, in) -> output: (B, T, k, out)
    return (x.unsqueeze(-2) @ self.weight[sel].transpose(-1, -2)).contiguous().squeeze(-2)

class Q4KPrimitiveStorage:
  __slots__ = ("words", "source_bytes", "persistent_bytes", "shared_bytes", "nonpersistent_bytes", "mode")
  def __init__(self, words:Tensor, source_bytes:int, persistent_bytes:int, mode:str,
               shared_bytes:int=0, nonpersistent_bytes:int=0):
    self.words, self.source_bytes, self.persistent_bytes = words, source_bytes, persistent_bytes
    self.shared_bytes, self.nonpersistent_bytes, self.mode = shared_bytes, nonpersistent_bytes, mode

class Q6KPrimitiveStorage:
  __slots__ = ("halfs", "source_bytes", "persistent_bytes", "shared_bytes", "nonpersistent_bytes", "mode")
  def __init__(self, halfs:Tensor, source_bytes:int, persistent_bytes:int, mode:str,
               shared_bytes:int=0, nonpersistent_bytes:int=0):
    self.halfs, self.source_bytes, self.persistent_bytes = halfs, source_bytes, persistent_bytes
    self.shared_bytes, self.nonpersistent_bytes, self.mode = shared_bytes, nonpersistent_bytes, mode

class QKPrimitiveBudget:
  __slots__ = ("cap_bytes", "used_bytes", "strict")
  def __init__(self, cap_bytes:int|None=None, strict:bool=False):
    self.cap_bytes, self.used_bytes, self.strict = cap_bytes, 0, strict

  def reserve(self, name:str, bytes_needed:int, kind:str) -> bool:
    if bytes_needed < 0: raise ValueError(f"{kind} storage bytes must be non-negative for {name}: {bytes_needed}")
    if self.cap_bytes is not None and self.used_bytes + bytes_needed > self.cap_bytes:
      if self.strict:
        raise MemoryError(f"{kind} primitive storage cap exceeded for {name}: used={self.used_bytes} "
                          f"need={bytes_needed} cap={self.cap_bytes}")
      return False
    self.used_bytes += bytes_needed
    return True

class Q4KPrimitiveRegistry:
  __slots__ = ("linears",)
  def __init__(self, linears:list[Q4KPrimitiveLinear|Q6KPrimitiveLinear]|None=None): self.linears = linears or []

_VDOT_QUANT_CACHE: dict = {}  # E0: per-token q8 quant cache keyed by x.uop.key (q/k/v + gate/up share)

class Q4KPrimitiveLinear:
  def __init__(self, weight:Tensor, bias:Tensor|None, words:Tensor, out_features:int, in_features:int, parts:int, opts:tuple,
               name:str, source_bytes:int, persistent_bytes:int, storage_mode:str,
               shared_bytes:int=0, nonpersistent_bytes:int=0, kernel_mode:str="partial"):
    if kernel_mode not in ("partial", "direct_out"): raise ValueError(f"unsupported Q4_K primitive kernel mode {kernel_mode!r}")
    if kernel_mode == "direct_out" and parts != 1: raise ValueError("Q4_K direct_out primitive requires parts=1")
    self.weight, self.bias = weight, bias
    self.q4k_storage = Q4KPrimitiveStorage(words, source_bytes, persistent_bytes, storage_mode, shared_bytes, nonpersistent_bytes)
    self.out_features, self.in_features, self.parts, self.opts, self.name = out_features, in_features, parts, opts, name
    self.kernel_mode = kernel_mode
    self.decode_enabled = False

  def _fallback(self, x:Tensor) -> Tensor:
    return x.linear(self.weight.transpose(), self.bias)

  def __call__(self, x:Tensor) -> Tensor:
    # Decode GEMV (1 token) or batched verify/prefill GEMM (K tokens). Unsupported bias/shape -> normal graph.
    if not self.decode_enabled or self.bias is not None or len(x.shape) != 3 or x.shape[0] != 1 or x.shape[-1] != self.in_features:
      return self._fallback(x)
    K = x.shape[-2]
    if not isinstance(K, int) or K != 1:  # batched (verify/prefill); fall back for large or symbolic K
      if not isinstance(K, int) or K > 32 or self.kernel_mode == "direct_out": return self._fallback(x)
      from extra.q4_k_gemv_primitive import q4k_gemm_kernel, parse_opt
      x_batch = x[0].cast(dtypes.float16).contiguous()  # [K, in_features]
      words = self.q4k_storage.words.to(x.device).contiguous() if self.q4k_storage.mode == "q4_ondemand" else self.q4k_storage.words.to(x.device)
      partials = Tensor.empty(self.out_features, K, self.parts, dtype=dtypes.float32, device=x.device)
      gemm_opts = self.opts + (parse_opt(f"UPCAST:1:{min(K, 16)}"),)  # hoist the dequant across the K columns
      out = partials.custom_kernel(words, x_batch.reshape(K*self.in_features),
        fxn=q4k_gemm_kernel(self.out_features, self.in_features, K, self.parts, "none", gemm_opts))[0]
      return out.sum(axis=2).transpose(0, 1).reshape(1, K, self.out_features)
    # PROMOTED default-on: the generated G3 LaneMap is the default Q4_K decode GEMV route (speed-equivalent to the owned
    # warp kernel, <0.5% across ctx 512-4096; token-matched, route-clean). Directional pure-machine-search move: ship the
    # generated route, keep owned warp one flag away. Rollback to owned warp: BUBBLEBEAM_FUTURESIGHT=0.
    bubblebeam_futuresight = getenv("BUBBLEBEAM_FUTURESIGHT", 1) or getenv("BEAM_COALESCE")
    g3_bubblebeam_shape = (self.in_features // 256) % 4 == 0 and DECODE_ATTN_AMDGCN_ARCH_OK and ((self.in_features == 4096 and self.out_features in (4096, 12288)) or (self.in_features == 12288 and self.out_features == 4096))
    # PROMOTED default-ON 2026-06-30 (rollback = DECODE_Q4K_G3_ANYSHAPE=0): bind the generated G3 lanemap by
    # STRUCTURAL shape eligibility ((in//256)%4==0 and out%32==0) rather than the hardcoded 8B dims, so larger
    # dense Q4_K decode shapes (14B/32B FFN gate/up/down + attn_q/k/o) take the generated route instead of the slow
    # lazy-dequant fallback. Byte-identical (token-matched 8B/14B); W==D 8B +4%, 14B +60%, 32B +78% (paired with
    # DECODE_ROUTE_ATTN_K). Structural class, not a model-dim hardcode. See docs/qwen-14b-32b-attn-k-route-miss-result.
    g3_anyshape = bool(getenv("DECODE_Q4K_G3_ANYSHAPE", 1)) and DECODE_ATTN_AMDGCN_ARCH_OK \
      and (self.in_features // 256) % 4 == 0 and self.out_features % 32 == 0
    if bubblebeam_futuresight and not getenv("Q4K_GEMV_SCHEDULER") and (g3_bubblebeam_shape or g3_anyshape):
      from extra.qk_bubblebeam_futuresight import should_route_q4k_lane_partition
      if g3_anyshape or should_route_q4k_lane_partition(self.out_features, self.in_features):
        _w = self.q4k_storage.words.to(x.device).contiguous() if self.q4k_storage.mode == "q4_ondemand" else self.q4k_storage.words.to(x.device)
        _xv = x[:, 0, :].reshape(self.in_features).cast(dtypes.float16).contiguous()
        # L2 (rollback = DECODE_Q4K_SPLIT_K_KV=0): split-K decode for OCCUPANCY-STARVED G3 GEMVs. A generated
        # kernel that launches only `out_features` workgroups underutilizes the GPU when out_features is small
        # (the KV projections 5120->1024 sit at ~26% occupancy). Split-K launches out_features*parts workgroups
        # and finalizes with a sum over parts. GENERIC: parts = the largest divisor of blocks_per_group
        # ((in//256)//4) that keeps out_features*parts under a workgroup cap; no model/shape hardcode.
        # L2b (rollback = DECODE_Q4K_INKERNEL_COMBINE_KV=0): in-kernel-combine decode for OCCUPANCY-STARVED G3
        # GEMVs. Same occupancy motivation as split-K, but instead of launching more workgroups + an EXTERNAL
        # .sum (L2, which was speed-flat because the added combine reduce offset the gain), it uses a WIDER
        # workgroup (`parts` waves per row) and combines the per-wave partials IN-KERNEL via LDS+barrier ->
        # out[row] directly (no external reduce). GENERIC: parts = largest divisor of blocks_per_group under a
        # workgroup cap; no model/shape hardcode. Takes precedence over the split-K path when enabled.
        if getenv("DECODE_Q4K_INKERNEL_COMBINE_KV", 0) and self.out_features <= getenv("DECODE_SPLIT_K_MAX_ROWS", 2048):
          _bpg = (self.in_features // 256) // 4
          _cap = getenv("DECODE_SPLIT_K_TARGET_WG", 8192)
          _parts = max((p for p in range(1, _bpg + 1) if _bpg % p == 0 and self.out_features * p <= _cap), default=1)
          if _parts > 1:
            from extra.qk_gemv_g3_codegen_lowering import q4k_g3_lanemap_gemv_inkernel_combine_kernel
            _out = Tensor.empty(self.out_features, dtype=dtypes.float32, device=x.device)
            return _out.custom_kernel(_w, _xv, fxn=q4k_g3_lanemap_gemv_inkernel_combine_kernel(self.out_features, self.in_features, _parts))[0].reshape(1, 1, self.out_features)
        if getenv("DECODE_Q4K_SPLIT_K_KV", 0) and self.out_features <= getenv("DECODE_SPLIT_K_MAX_ROWS", 2048):
          _bpg = (self.in_features // 256) // 4
          _cap = getenv("DECODE_SPLIT_K_TARGET_WG", 8192)
          _parts = max((p for p in range(1, _bpg + 1) if _bpg % p == 0 and self.out_features * p <= _cap), default=1)
          if _parts > 1:
            from extra.qk_gemv_g3_codegen_lowering import q4k_g3_lanemap_gemv_splitk_kernel
            _p = Tensor.empty(self.out_features * _parts, dtype=dtypes.float32, device=x.device)
            _p = _p.custom_kernel(_w, _xv, fxn=q4k_g3_lanemap_gemv_splitk_kernel(self.out_features, self.in_features, _parts))[0]
            return _p.reshape(self.out_features, _parts).sum(axis=1).reshape(1, 1, self.out_features)
        from extra.qk_gemv_g3_codegen_lowering import q4k_g3_lanemap_gemv_kernel
        _out = Tensor.empty(self.out_features, dtype=dtypes.float32, device=x.device)
        return _out.custom_kernel(_w, _xv, fxn=q4k_g3_lanemap_gemv_kernel(self.out_features, self.in_features))[0].reshape(1, 1, self.out_features)
    if (getenv("Q4K_GEMV_SCHEDULER") or bubblebeam_futuresight) and self.in_features == 4096 and self.out_features == 12288:
      # M5/M6 research lever (default-off): scheduler-GENERATED matvec for FFN gate/up instead of the owned warp
      # custom_kernel. Two modes:
      #   1 (=_fallback): x.linear(self.weight.T) -- self.weight is a LAZY Q4_K->fp16 dequant graph (model.py:141)
      #     fused into the matmul (reads packed); the matvec heuristic groups the K-reduce, so WARP_REDUCE_LOWERING=1
      #     swaps the LDS-tree group reduce for the ds_bpermute ladder. (M6: cross-lane ~neutral.)
      #   2 (PACKED): word-structured tinygrad-ops dequant (extra/qk_q4k_scheduler_gemv) whose load unit is the
      #     uint32 word -- tests whether a pure-scheduler GEMV can coalesce packed-word loads like the owned kernel.
      #   4 (LANE_PARTITION): explicit research-only custom-kernel bridge fallback using LanePartitionReduce.
      #   5 (G2_LANEMAP): generated Tensor/scheduler route bound to the bridge-independent G2 Q4_K LaneMap. Route-clean
      #     runtime/codegen binding probe; expected to fail speed until codegen exploits the representation.
      #   6 (G3_LANEMAP_CODEGEN): generated named wave32 UOp program from the G2 LaneMap. This is the first lowering
      #     probe for one-word-per-lane in-register dequant without routing through the lane-partition bridge module.
      if bubblebeam_futuresight and not getenv("Q4K_GEMV_SCHEDULER"):
        from extra.qk_bubblebeam_futuresight import should_route_q4k_lane_partition
        if not (should_route_q4k_lane_partition(self.out_features, self.in_features) or g3_bubblebeam_shape): return self._fallback(x)
        from extra.qk_gemv_g3_codegen_lowering import q4k_g3_lanemap_gemv_kernel
        _w = self.q4k_storage.words.to(x.device).contiguous() if self.q4k_storage.mode == "q4_ondemand" else self.q4k_storage.words.to(x.device)
        _xv = x[:, 0, :].reshape(self.in_features).cast(dtypes.float16).contiguous()
        _out = Tensor.empty(self.out_features, dtype=dtypes.float32, device=x.device)
        return _out.custom_kernel(_w, _xv, fxn=q4k_g3_lanemap_gemv_kernel(self.out_features, self.in_features))[0].reshape(1, 1, self.out_features)
      if getenv("Q4K_GEMV_SCHEDULER") == 4:
        from extra.qk_q4k_lane_partition_gemv import q4k_lane_partition_gemv_kernel
        _w = self.q4k_storage.words.to(x.device).contiguous() if self.q4k_storage.mode == "q4_ondemand" else self.q4k_storage.words.to(x.device)
        _xv = x[:, 0, :].reshape(self.in_features).cast(dtypes.float16).contiguous()
        _out = Tensor.empty(self.out_features, dtype=dtypes.float32, device=x.device)
        return _out.custom_kernel(_w, _xv, fxn=q4k_lane_partition_gemv_kernel(self.out_features, self.in_features))[0].reshape(1, 1, self.out_features)
      if getenv("Q4K_GEMV_SCHEDULER") == 6:
        from extra.qk_gemv_g3_codegen_lowering import q4k_g3_lanemap_gemv_kernel
        _w = self.q4k_storage.words.to(x.device).contiguous() if self.q4k_storage.mode == "q4_ondemand" else self.q4k_storage.words.to(x.device)
        _xv = x[:, 0, :].reshape(self.in_features).cast(dtypes.float16).contiguous()
        _out = Tensor.empty(self.out_features, dtype=dtypes.float32, device=x.device)
        return _out.custom_kernel(_w, _xv, fxn=q4k_g3_lanemap_gemv_kernel(self.out_features, self.in_features))[0].reshape(1, 1, self.out_features)
      if getenv("Q4K_GEMV_SCHEDULER") in (2, 3, 5):
        from extra.qk_q4k_scheduler_gemv import q4k_scheduler_matvec, q4k_scheduler_matvec_wordlane, q4k_scheduler_matvec_lanemap
        _w = self.q4k_storage.words.to(x.device)
        _xv = x[:, 0, :].reshape(self.in_features).cast(dtypes.float32)
        _fn = q4k_scheduler_matvec_lanemap if getenv("Q4K_GEMV_SCHEDULER") == 5 else q4k_scheduler_matvec_wordlane if getenv("Q4K_GEMV_SCHEDULER") == 3 else q4k_scheduler_matvec
        return _fn(_w, _xv, self.out_features, self.in_features).reshape(1, 1, self.out_features)
      return self._fallback(x)
    from extra.q4_k_gemv_primitive import q4k_gemv_kernel, q4k_gemv_partial_kernel
    x_vec = x[:, 0, :].reshape(self.in_features).cast(dtypes.float16).contiguous()
    words = self.q4k_storage.words.to(x.device).contiguous() if self.q4k_storage.mode == "q4_ondemand" else self.q4k_storage.words.to(x.device)
    # Cooperative-K Q4_K decode GEMV (MMVQ_COOP) for attn_q/o only: the within-block word index lane4=pos//4
    # becomes a LOCAL lane -> coalesced packed-word loads. Q4_K coop is role-dependent: attn_q/o (4096x4096) is
    # poorly coalesced by default (~19% peak -> ~29%, 1.52x), but ffn_gate/up is already ~41% peak so it is NOT
    # routed. fp-reassoc-tol exact. See docs/qk-mmvq-coop-q4k-attn-*.
    # attn q/o projection work-decomposition warp (same lossless q4k_gemv_warp_kernel as FFN; Q4K_GEMV_WARP_PROJ).
    # q/o is Q4_K 4096x4096, coop-routed by default (~27% -> warp ~36%, 1.32x local). Takes precedence over coop.
    # DEFAULT-ON 2026-06-25: a clock-pinned INTERLEAVED (drift-cancelled) W==D re-test shows +1.58-1.67%/ctx,
    # byte-identical, route fires (q4k_gemv_warp_4096_4096) -- REFUTES the 2026-06-22 "did not transfer" finding,
    # which was a non-interleaved auto-clock confound. Revert with Q4K_GEMV_WARP_PROJ=0.
    # Evidence: bench/qk-proj-gemv-warp/wd.json, docs/decode-proj-gemv-warp-promotion-result-20260625.md.
    if getenv("Q4K_GEMV_WARP_PROJ", 1) and self.parts == 1 and self.out_features == 4096 and self.in_features == 4096 \
       and (self.in_features // 256) % 4 == 0 and DECODE_ATTN_AMDGCN_ARCH_OK:
      try:
        from extra.q4_k_gemv_primitive import q4k_gemv_warp_kernel
        out = Tensor.empty(self.out_features, dtype=dtypes.float32, device=x.device)
        got = out.custom_kernel(words, x_vec, fxn=q4k_gemv_warp_kernel(self.out_features, self.in_features))[0]
        return got.reshape(1, 1, self.out_features)
      except Exception as e:
        if getenv("DEBUG", 0): print(f"Q4K_GEMV_WARP_PROJ fallback: {e}")
    rt4 = getenv("Q4K_COOP_RT", 16)
    if getenv("Q4K_ATTN_QO_COOP", 1) and self.parts == 1 and self.out_features == 4096 and self.in_features == 4096 \
        and self.out_features % rt4 == 0:
      from extra.q4_k_gemv_primitive import q4k_coop_partial_kernel
      partials = Tensor.empty(self.out_features, 8, dtype=dtypes.float32, device=x.device)
      partial = partials.custom_kernel(words, x_vec, fxn=q4k_coop_partial_kernel(self.out_features, self.in_features, rt4))[0]
      return partial.sum(axis=1).reshape(1, 1, self.out_features)
    if self.kernel_mode == "direct_out":
      out = Tensor.empty(self.out_features, dtype=dtypes.float32, device=x.device)
      got = out.custom_kernel(words, x_vec, fxn=q4k_gemv_kernel(self.out_features, self.in_features, "none", self.opts))[0]
      return got.reshape(1, 1, self.out_features)
    partials = Tensor.empty(self.out_features, self.parts, dtype=dtypes.float32, device=x.device)
    if getenv("Q4K_VDOT") and self.parts == 1:  # D1/E0: schedulable builtin v_dot4 (udot4) decode GEMV
      from extra.q4_k_gemv_primitive import q4k_q8_1_vdot_builtin_partial_kernel, q8_1_bias_pack_u32_kernel
      from extra.qk_layout import q8_1_quantize
      amort = bool(getenv("Q4K_VDOT_AMORT"))  # E0: quantize x ONCE/token, shared across q/k/v and gate/up
      ck = x.uop.key if amort else None
      cached = _VDOT_QUANT_CACHE.get(ck) if amort else None
      if cached is None:
        q, scales = q8_1_quantize(x_vec.cast(dtypes.float32))
        q_bias_words = Tensor.empty(self.in_features // 4, dtype=dtypes.uint32, device=x.device).custom_kernel(
          q, fxn=q8_1_bias_pack_u32_kernel(self.in_features))[0]
        if amort: _VDOT_QUANT_CACHE[ck] = (q_bias_words, scales); _VDOT_QUANT_CACHE["m"] = _VDOT_QUANT_CACHE.get("m", 0)+1
      else:
        q_bias_words, scales = cached; _VDOT_QUANT_CACHE["h"] = _VDOT_QUANT_CACHE.get("h", 0)+1
      partial = partials.custom_kernel(words, q_bias_words, scales,
        fxn=q4k_q8_1_vdot_builtin_partial_kernel(self.out_features, self.in_features, 1, "none", ()))[0]
      return partial.sum(axis=1).reshape(1, 1, self.out_features)
    # FFN-GEMV work-decomposition variant: lossless FP 32-thread/row + K-block-parallel + in-kernel warp_reduce_sum
    # (ds_bpermute), one output (vs default 1-thread/row serial). Default-ON for guarded 8B Q4_K FFN gate/up + down
    # after W==D hardening: ~+9.6%@ctx1024 / ~+8.5%@ctx4096, byte-identical. Revert with Q4K_GEMV_WARP=0; disable
    # down separately with Q4K_GEMV_WARP_DOWN=0. Q4K_GEMV_WARP_PROJ (attn q/o) is now ALSO default-on (+1.6%/ctx,
    # byte-identical, see above) -- the earlier "did not transfer" was a non-interleaved clock confound.
    if getenv("Q4K_GEMV_WARP", 1) and (self.in_features // 256) % 4 == 0 and DECODE_ATTN_AMDGCN_ARCH_OK \
       and ((self.in_features == 4096 and self.out_features == 12288 and self.parts == 1)      # FFN gate/up
            or (getenv("Q4K_GEMV_WARP_DOWN", 1) and self.in_features == 12288 and self.out_features == 4096)):  # FFN down (Q4_K)
      try:
        from extra.q4_k_gemv_primitive import q4k_gemv_warp_kernel
        out = Tensor.empty(self.out_features, dtype=dtypes.float32, device=x.device)
        got = out.custom_kernel(words, x_vec, fxn=q4k_gemv_warp_kernel(self.out_features, self.in_features))[0]
        return got.reshape(1, 1, self.out_features)
      except Exception as e:
        if getenv("DEBUG", 0): print(f"Q4K_GEMV_WARP fallback: {e}")
    partial = partials.custom_kernel(words, x_vec, fxn=q4k_gemv_partial_kernel(self.out_features, self.in_features, self.parts, "none", self.opts))[0]
    return partial.sum(axis=1).reshape(1, 1, self.out_features)

class Q4KFusedLinear:
  # B1 horizontal-fusion probe: one Q4_K GEMV over concatenated sibling weight rows (q/k/v or gate/up),
  # then split. Decode-only (uses the fused primitive when decode_enabled); prefill falls back to the
  # separate originals (whose own decode_enabled=False routes them to the dense path).
  def __init__(self, fused:Q4KPrimitiveLinear, originals:list[Q4KPrimitiveLinear], splits:list[int]):
    self.fused, self.originals, self.splits = fused, originals, splits
  def __call__(self, x:Tensor) -> list[Tensor]:
    if self.fused.decode_enabled:
      out = self.fused(x)  # (1,1,sum)
      res, c = [], 0
      for s in self.splits: res.append(out[..., c:c+s]); c += s
      return res
    return [l(x) for l in self.originals]

def _build_fused_q4k(linears:list[Q4KPrimitiveLinear], tag:str) -> Q4KFusedLinear:
  words = linears[0].q4k_storage.words.cat(*[l.q4k_storage.words for l in linears[1:]], dim=0).contiguous().realize()
  out_features, in_features, q4_bytes = sum(l.out_features for l in linears), linears[0].in_features, words.numel()*4
  fused = Q4KPrimitiveLinear(None, None, words, out_features, in_features, 1, linears[0].opts,
                             f"fused_{tag}", q4_bytes, q4_bytes, "sidecar", 0, 0)
  return Q4KFusedLinear(fused, linears, [l.out_features for l in linears])

def _install_q4k_fusions(model) -> None:
  # gated by Q4K_FUSE: fuse q/k/v->attn_qkv and gate/up->ffn_gateup on each dense block; register the
  # fused primitives so decode_enabled gets toggled per step.
  for block in getattr(model, "blk", []):
    if all(isinstance(getattr(block, n, None), Q4KPrimitiveLinear) for n in ("attn_q", "attn_k", "attn_v")):
      block.attn_qkv = _build_fused_q4k([block.attn_q, block.attn_k, block.attn_v], "qkv")
      model._q4k_linears.linears.append(block.attn_qkv.fused)
    if all(isinstance(getattr(block, n, None), Q4KPrimitiveLinear) for n in ("ffn_gate", "ffn_up")):
      block.ffn_gateup = _build_fused_q4k([block.ffn_gate, block.ffn_up], "gateup")
      model._q4k_linears.linears.append(block.ffn_gateup.fused)

class Q6KPrimitiveLinear:
  def __init__(self, weight:Tensor, bias:Tensor|None, halfs:Tensor, out_features:int, in_features:int, parts:int, opts:tuple,
               name:str, source_bytes:int, persistent_bytes:int, storage_mode:str,
               shared_bytes:int=0, nonpersistent_bytes:int=0):
    self.weight, self.bias = weight, bias
    self.q6k_storage = Q6KPrimitiveStorage(halfs, source_bytes, persistent_bytes, storage_mode, shared_bytes, nonpersistent_bytes)
    self.out_features, self.in_features, self.parts, self.opts, self.name = out_features, in_features, parts, opts, name
    self.decode_enabled = False

  def _fallback(self, x:Tensor) -> Tensor:
    return x.linear(self.weight.transpose(), self.bias)

  def __call__(self, x:Tensor) -> Tensor:
    # Q6_K decode GEMV (1 token) or batched verify/prefill GEMM (K tokens).
    if not self.decode_enabled or self.bias is not None or len(x.shape) != 3 or x.shape[0] != 1 or x.shape[-1] != self.in_features:
      return self._fallback(x)
    K = x.shape[-2]
    if not isinstance(K, int) or K != 1:  # batched (verify/prefill)
      if not isinstance(K, int) or K > 32: return self._fallback(x)
      from extra.q6_k_gemv_primitive import q6k_gemm_kernel, parse_opt
      x_batch = x[0].cast(dtypes.float16).contiguous()  # [K, in_features]
      partials = Tensor.empty(self.out_features, K, self.parts, dtype=dtypes.float32, device=x.device)
      gemm_opts = self.opts + (parse_opt(f"UPCAST:1:{min(K, 16)}"),)  # hoist the dequant across the K columns
      out = partials.custom_kernel(self.q6k_storage.halfs.to(x.device), x_batch.reshape(K*self.in_features),
        fxn=q6k_gemm_kernel(self.out_features, self.in_features, K, self.parts, gemm_opts))[0]
      return out.sum(axis=2).transpose(0, 1).reshape(1, K, self.out_features)
    x_vec = x[:, 0, :].reshape(self.in_features).cast(dtypes.float16).contiguous()
    # Cooperative-K Q6_K decode GEMV (MMVQ_COOP family): the within-block pos becomes a LOCAL lane axis ->
    # coalesced packed-weight loads (the default one-row-per-thread path runs Q6_K roles at ~10-14% HBM peak;
    # coop reaches ~40-51%). fp-reassoc-tol exact, byte-identical greedy. Gated per role-class:
    #   lm_head (out>=100000): Q6K_LM_HEAD_COOP default on (+19% decode, isolated 5x).
    #   ffn_down (4096x12288): Q6K_FFN_DOWN_COOP default on (isolated 2.77x). See docs/qk-mmvq-q6k-*.
    # FFN-down Q6_K work-decomposition warp (same lossless lever). SEPARATE research flag Q6K_GEMV_WARP_DOWN (NOT the
    # promoted Q4K_GEMV_WARP_DOWN): the Q6_K down is ALREADY coop-routed (~51% peak) so warp is only ~1.09x and does NOT
    # improve W==D (correct, byte-identical, but not worth promoting). down shape only (NOT lm_head). default-off.
    if getenv("Q6K_GEMV_WARP_DOWN") and self.parts == 1 and self.out_features == 4096 and self.in_features == 12288 \
       and (self.in_features // 256) % 2 == 0 and DECODE_ATTN_AMDGCN_ARCH_OK:
      try:
        from extra.q6_k_gemv_primitive import q6k_gemv_warp_kernel
        out = Tensor.empty(self.out_features, dtype=dtypes.float32, device=x.device)
        got = out.custom_kernel(self.q6k_storage.halfs.to(x.device), x_vec,
                                fxn=q6k_gemv_warp_kernel(self.out_features, self.in_features))[0]
        return got.reshape(1, 1, self.out_features)
      except Exception as e:
        if getenv("DEBUG", 0): print(f"Q6K_GEMV_WARP down fallback: {e}")
    # Q6K-2/3 direct/warp lm_head route (research flag Q6K_DIRECT_ROUTE, default-off). lm_head currently uses
    # Q6K_LM_HEAD_COOP = coop_partial + external .sum (the r_32_4_1187 reduce). The half-warp 2-row partition route
    # (q6k_halfwarp_partition_kernel: 2 independent rows per 32-lane wave as two 16-lane partitions, in-warp
    # warp_reduce_sum(width=16)) writes out[row] DIRECTLY -> no partials buffer, no external r_* reduce. Dequant
    # (_q6k_weight) is byte-identical to coop. lm_head shape only (out>=100000, even rows). Flag-off => coop unchanged.
    if getenv("Q6K_DIRECT_ROUTE") and self.parts == 1 and self.out_features >= 100000 \
       and self.out_features % 2 == 0 and DECODE_ATTN_AMDGCN_ARCH_OK:
      try:
        from extra.q6_k_gemv_primitive import q6k_halfwarp_partition_kernel
        out = Tensor.empty(self.out_features, dtype=dtypes.float32, device=x.device)
        got = out.custom_kernel(self.q6k_storage.halfs.to(x.device), x_vec,
                                fxn=q6k_halfwarp_partition_kernel(self.out_features, self.in_features))[0]
        return got.reshape(1, 1, self.out_features)
      except Exception as e:
        if getenv("DEBUG", 0): print(f"Q6K_DIRECT_ROUTE lm_head fallback: {e}")
    rt = getenv("Q6K_COOP_RT", 4)
    use_coop = self.parts == 1 and self.out_features % rt == 0 and (
      (getenv("Q6K_LM_HEAD_COOP", 1) and self.out_features >= 100000) or
      (getenv("Q6K_FFN_DOWN_COOP", 1) and self.out_features == 4096 and self.in_features == 12288) or
      # L3 (rollback = DECODE_Q6K_FFN_DOWN_LONGK=0): route LARGE-IN Q6_K ffn_down (14B 17408->5120,
      # 32B 25600->5120) through the same coop-partial route the 8B ffn_down already uses. The shipped gate
      # above hardcodes the 8B dims (4096/12288), so 14B/32B Q6_K ffn_down falls to the slower generic partial
      # path (~253 GB/s). Structural class (long in-features, moderate out, not lm_head), not a model-dim hardcode.
      (getenv("DECODE_Q6K_FFN_DOWN_LONGK", 1) and self.in_features >= 8192 and self.out_features < 100000))
    if use_coop:
      from extra.q6_k_gemv_primitive import q6k_coop_partial_kernel
      partials = Tensor.empty(self.out_features, 16, dtype=dtypes.float32, device=x.device)
      partial = partials.custom_kernel(self.q6k_storage.halfs.to(x.device), x_vec,
                                       fxn=q6k_coop_partial_kernel(self.out_features, self.in_features, rt))[0]
      return partial.sum(axis=1).reshape(1, 1, self.out_features)
    from extra.q6_k_gemv_primitive import q6k_gemv_partial_kernel
    partials = Tensor.empty(self.out_features, self.parts, dtype=dtypes.float32, device=x.device)
    partial = partials.custom_kernel(self.q6k_storage.halfs.to(x.device), x_vec,
                                     fxn=q6k_gemv_partial_kernel(self.out_features, self.in_features, self.parts, self.opts))[0]
    return partial.sum(axis=1).reshape(1, 1, self.out_features)

def should_use_flash_decode(start_pos, T, use_flash:bool=False) -> bool:
  """Centralized flash-decode selection policy (decode attention). Invariants: single-token decode (T==1) with a
  symbolic start_pos (the flash-decode kernel needs the symbolic KV length). `FLASH_DECODE` env: "0"/off, "1"/on,
  "auto" (default). In auto, enable when the trace-time context (the decode-start position, read from the bound
  start_pos) >= FLASH_DECODE_THRESHOLD (default 512) -- short-context decode <512 stays SDPA, and if the context
  can't be read we stay SDPA. flash-decode is exact-vs-SDPA up to fp reassociation and measured neutral-or-better
  at/above the threshold; the crossover is ~ctx384 (flash REGRESSES below ~256: 0.93x @128, 0.95x @256), and it
  wins above: +12.8% real-generate @ctx520 (byte-identical greedy), 1.05x @512, 1.23x @1024, 1.73x @4096. 512 is
  the measured safe cutover (Arc 1, docs/qk-8b-attention-fusion-result-20260617.md)."""
  if not (isinstance(start_pos, UOp) and isinstance(T, int) and T == 1): return False  # decode-only invariant
  mode = str(getenv("FLASH_DECODE", "auto")).lower()
  if mode in ("0", "false", "off"): return False                 # force off
  if use_flash or mode in ("1", "true", "on"): return True       # force on (programmatic or env), invariants hold
  if mode != "auto": return False                                # unknown value -> conservative SDPA
  try: ctx = start_pos.unbind()[1] + T                           # decode-start context known at trace/capture time
  except Exception: return False                                 # can't read context -> conservative SDPA
  return ctx >= getenv("FLASH_DECODE_THRESHOLD", 512)

def apply_rope(x:Tensor, freqs_cis:Tensor) -> Tensor:
  assert x.shape[-1] % 2 == 0
  cos, sin = freqs_cis.reshape(1, 1, x.shape[2], -1).chunk(2, dim=-1)
  x1, x2 = x.chunk(2, dim=-1)
  return (x1 * cos - x2 * sin).cat(x2 * cos + x1 * sin, dim=-1)

def _q4k_policy(name:str) -> tuple[int, tuple[str, ...]]|None:
  if ".ffn_gate.weight" in name or ".ffn_up.weight" in name: return 1, ("LOCAL:0:64",)
  if ".ffn_down.weight" in name: return 4, ("LOCAL:0:32",)
  if ".attn_q.weight" in name or ".attn_output.weight" in name: return 1, ("LOCAL:0:64",)
  # PROMOTED default-ON 2026-06-30 (rollback DECODE_ROUTE_ATTN_K=0): attn_k is Q4_K (same as attn_q) but was
  # omitted here, so it fell to a plain nn.Linear -> the slow generic lazy-dequant GEMV (kernel r_8_32_4_20_4_2_32),
  # measured 38% of 14B decode. Cover it so it takes the same primitive/generated route as attn_q. Byte-identical;
  # W==D 14B 27.8->44.5 (+60%), 32B 11.8->21.0 (+78%), 8B 103.5->107.6 (+4%). See
  # docs/qwen-14b-32b-attn-k-route-miss-result-20260630.md.
  if ".attn_k.weight" in name and getenv("DECODE_ROUTE_ATTN_K", 1): return 1, ("LOCAL:0:64",)
  return None

def _q6k_policy(name:str) -> tuple[int, tuple[str, ...]]|None:
  # ffn_down wins decisively (the dominant Q6_K decode cost). attn_v/output were historically left to the
  # fused graph; re-measured 2026-06-15 on RX 7900 XTX at full clock they now also win (+5%, 50.8->53.4 tok/s,
  # byte-identical output). The older "lose to fused graph" claim was likely a clock-ramp-confounded bench.
  # Default-on (exact dequant, no accuracy risk); set Q6K_COVER_MORE=0 to disable if a model regresses.
  if ".ffn_down.weight" in name: return 1, ("LOCAL:0:64",)
  if getenv("Q6K_COVER_MORE", 1):
    if ".attn_v.weight" in name: return 4, ("LOCAL:0:32",)
    if name == "output.weight": return 1, ("LOCAL:0:64",)
  return None

def _qk_policy_value(entry:dict) -> dict:
  cand = entry.get("candidate") or {}
  return {
    "winner": entry.get("winner"), "parts": int(cand.get("parts", 0)),
    "opts": tuple(cand.get("opts", ())), "family": cand.get("family", ""),
    "reduction": cand.get("reduction", ""),
    "policy_reason": entry.get("policy_reason", ""), "storage": entry.get("storage", {}),
  }

def _load_qk_generated_policy(path:str) -> dict:
  policy_path = pathlib.Path(path).expanduser()
  data = json.loads(policy_path.read_text())
  if data.get("kind") != "qk_generated_policy": raise ValueError(f"{policy_path} is not a QK generated policy cache")
  if data.get("generator_version") not in (0, 1):
    raise ValueError(f"{policy_path} has unsupported generator_version={data.get('generator_version')}")
  by_shape: dict[tuple[int, int, int], dict] = {}
  by_tensor: dict[tuple[str, int, int, int], dict] = {}
  for entry in data.get("entries", []):
    desc, cand = entry.get("descriptor", {}), entry.get("candidate") or {}
    key = (int(desc["ggml_type"]), int(desc["rows"]), int(desc["cols"]))
    value = _qk_policy_value(entry)
    if entry.get("scope") == "tensor":
      tensor = str(desc.get("tensor", ""))
      if not tensor: raise ValueError(f"{policy_path} has tensor-scoped entry without descriptor.tensor")
      tensor_key = (tensor, *key)
      if tensor_key in by_tensor and by_tensor[tensor_key] != value:
        raise ValueError(f"{policy_path} has conflicting tensor generated policy entries for key={tensor_key}: "
                         f"{by_tensor[tensor_key]} vs {value}")
      by_tensor[tensor_key] = value
    else:
      if key in by_shape and by_shape[key] != value:
        raise ValueError(f"{policy_path} has conflicting generated policy entries for key={key}: {by_shape[key]} vs {value}")
      by_shape[key] = value
  if not by_shape and not by_tensor: raise ValueError(f"{policy_path} contains no generated policy entries")
  return {"by_shape": by_shape, "by_tensor": by_tensor}

def _qk_generated_policy_len(policy:dict|None) -> int:
  if policy is None: return 0
  return len(policy.get("by_shape", {})) + len(policy.get("by_tensor", {}))

def _qk_generated_policy_entry(policy:dict|None, typ:int, rows:int, cols:int, name:str|None=None) -> dict|None:
  if policy is None: return None
  if name is not None and (entry:=policy.get("by_tensor", {}).get((name, typ, rows, cols))) is not None: return entry
  return policy.get("by_shape", {}).get((typ, rows, cols))

def _qk_storage_cap_from_env() -> int|None:
  raw = getenv("QK_PRIMITIVE_MAX_STORAGE_MB", "")
  if raw == "": return None
  cap = int(float(raw) * 1024 * 1024)
  if cap < 0: raise ValueError(f"QK_PRIMITIVE_MAX_STORAGE_MB must be non-negative, got {raw!r}")
  return cap

def _qk_storage_mode_from_env(default:str="sidecar") -> str:
  mode = getenv("QK_PRIMITIVE_STORAGE", default)
  if mode not in ("sidecar", "q4_ondemand", "shared"):
    raise ValueError(f"QK_PRIMITIVE_STORAGE must be sidecar, q4_ondemand, or shared, got {mode!r}")
  return mode

def _q6k_effective_storage_mode(requested_mode:str) -> str:
  # q4_ondemand is a Q4_K-only experiment. Q6_K stays persistent unless storage is shared.
  if requested_mode not in ("sidecar", "q4_ondemand", "shared"):
    raise ValueError(f"unsupported QK primitive storage mode {requested_mode!r}")
  return "shared" if requested_mode == "shared" else "sidecar"

@dataclass(frozen=True)
class QKConfig:
  """Single authority for the QK primitive *install* config read from the environment
  inside `Transformer.from_gguf` once primitives are active (i.e. the flags consumed
  under `if q4k_meta is not None`). Centralizes the reads + validation that were
  scattered as `getenv` calls so invalid QK runtime config is rejected in one place.

  Scope is deliberately the install config, not everything QK-shaped:
  - Activation gating (`Q4K_PRIMITIVE`/`Q6K_PRIMITIVE`/`QK_GENERATED_POLICY`) stays at
    the from_gguf gate -- its auto-default is coupled to the gguf source + device, a
    separate (runtime, not env-only) concern.
  - Forward-pass probe flags (`Q4K_VDOT`/`Q4K_VDOT_AMORT`/`Q6K_COVER_MORE`/`Q4K_UNFUSE`/
    `FLASH_DECODE`/`FLASH_L`) are read per-call at their own sites; folding them here
    would change when they are read.

  Built via `from_env(storage_default=...)` at the top of the active-primitive block,
  so every field is read exactly when the original scattered reads were (when the block
  is entered) -- a behaviour-preserving centralization."""
  generated_policy_strict: bool
  max_storage_bytes: int | None
  storage_mode: str
  q6_storage_mode: str
  policy_debug: bool
  storage_debug: bool
  demote_q6k_ffndown: bool
  demote_targets: tuple[str, ...]
  fuse_q4k: bool

  @staticmethod
  def from_env(*, storage_default:str) -> "QKConfig":
    # Read order mirrors the original sites: cap + strict (QKPrimitiveBudget), then the
    # validated storage mode, then the debug/probe flags -- so a first-raise on invalid
    # input lands on the same variable as before.
    max_storage_bytes = _qk_storage_cap_from_env()
    generated_policy_strict = bool(getenv("QK_GENERATED_POLICY_STRICT", 0))
    storage_mode = _qk_storage_mode_from_env(storage_default)
    # B3: per-tensor Q6->Q4 demotion. QK_DEMOTE_TENSORS (comma-sep name substrings) generalizes the
    # single-tensor Q6K_DEMOTE_FFNDOWN flag; the flag stays as the ffn_down shortcut for back-compat.
    demote_q6k_ffndown = bool(getenv("Q6K_DEMOTE_FFNDOWN"))
    explicit = tuple(t for t in getenv("QK_DEMOTE_TENSORS", "").replace(" ", "").split(",") if t)
    demote_targets = explicit or (("ffn_down",) if demote_q6k_ffndown else ())
    return QKConfig(
      generated_policy_strict=generated_policy_strict,
      max_storage_bytes=max_storage_bytes,
      storage_mode=storage_mode,
      q6_storage_mode=_q6k_effective_storage_mode(storage_mode),
      policy_debug=bool(getenv("QK_GENERATED_POLICY_DEBUG", 0)),
      storage_debug=bool(getenv("QK_GENERATED_POLICY_DEBUG", getenv("Q4K_PRIMITIVE_DEBUG", getenv("Q6K_PRIMITIVE_DEBUG", 0)))),
      demote_q6k_ffndown=demote_q6k_ffndown,
      demote_targets=demote_targets,
      fuse_q4k=bool(getenv("Q4K_FUSE")))

def _qk_storage_summary(linears:list[Q4KPrimitiveLinear|Q6KPrimitiveLinear]) -> dict:
  by_kind: collections.Counter[str] = collections.Counter()
  by_mode: collections.Counter[str] = collections.Counter()
  source_bytes = persistent_bytes = shared_bytes = nonpersistent_bytes = 0
  for linear in linears:
    storage = linear.q4k_storage if isinstance(linear, Q4KPrimitiveLinear) else linear.q6k_storage
    kind = "Q4K" if isinstance(linear, Q4KPrimitiveLinear) else "Q6K"
    by_kind[kind] += storage.persistent_bytes
    by_mode[storage.mode] += 1
    source_bytes += storage.source_bytes
    persistent_bytes += storage.persistent_bytes
    shared_bytes += getattr(storage, "shared_bytes", 0)
    nonpersistent_bytes += getattr(storage, "nonpersistent_bytes", 0)
  return {
    "source_bytes": source_bytes, "persistent_bytes": persistent_bytes,
    "shared_bytes": shared_bytes, "nonpersistent_bytes": nonpersistent_bytes,
    "by_kind": dict(sorted(by_kind.items())), "by_mode": dict(sorted(by_mode.items())),
  }

def _shared_packed_view(meta:dict, byte_start:int, nbytes:int, dtype) -> Tensor:
  raw = meta.get("raw_tensor")
  if raw is None: raise ValueError("shared QK primitive storage requires gguf_load_with_metadata raw_tensor")
  if byte_start % dtype.itemsize != 0 or nbytes % dtype.itemsize != 0:
    raise ValueError(f"shared QK primitive storage requires uint{dtype.itemsize*8}-aligned range: "
                     f"byte_start={byte_start} nbytes={nbytes}")
  raw_view = raw[byte_start:byte_start+nbytes].uop.buffer
  return Tensor(UOp.from_buffer(raw_view.view(nbytes//dtype.itemsize, dtype, 0), raw.device))

def _module_at(root, path:str):
  obj = root
  for part in path.split("."):
    obj = obj[int(part)] if isinstance(obj, list) and part.isdigit() else getattr(obj, part)
  return obj

def _set_module_at(root, path:str, value) -> None:
  if "." not in path: setattr(root, path, value); return  # top-level module (e.g. output)
  parent_path, attr = path.rsplit(".", 1)
  parent = _module_at(root, parent_path)
  if isinstance(parent, list) and attr.isdigit(): parent[int(attr)] = value
  else: setattr(parent, attr, value)

def _install_q4k_primitives(model, gguf:pathlib.Path, meta:dict, generated_policy:dict|None=None,
                            budget:QKPrimitiveBudget|None=None, storage_mode:str="sidecar") -> list[Q4KPrimitiveLinear]:
  from extra.q4_k_gemv_primitive import parse_opt
  supported_generated_families = {"q4_k_packed_u32", "q4_k_packed_u32_direct"}
  raw_words = Tensor(gguf, dtype=dtypes.uint32)
  installed: list[Q4KPrimitiveLinear] = []
  skipped: collections.Counter[str] = collections.Counter()
  budget = budget or QKPrimitiveBudget()
  debug = bool(getenv("Q4K_PRIMITIVE_DEBUG", getenv("QK_GENERATED_POLICY_DEBUG", 0)))
  for name, dims, typ, off in meta["tensor_infos"]:
    if typ != 12:
      skipped["not_q4_k"] += 1
      continue
    if len(dims) != 2:
      skipped["not_2d"] += 1
      continue
    if not name.endswith(".weight"):
      skipped["not_weight"] += 1
      continue
    rows, cols = tuple(reversed(dims))
    if generated_policy is None:
      if (policy := _q4k_policy(name)) is None:
        skipped["policy_fallback"] += 1
        continue
      parts, opt_specs = policy
      kernel_mode = "partial"
    else:
      if (policy_entry := _qk_generated_policy_entry(generated_policy, typ, rows, cols, name)) is None:
        skipped["policy_missing"] += 1
        continue
      if policy_entry["winner"] == "fused_graph":
        skipped["policy_fused"] += 1
        continue
      if policy_entry["family"] not in supported_generated_families:
        skipped["policy_unsupported"] += 1
        continue
      parts, opt_specs = policy_entry["parts"], policy_entry["opts"]
      kernel_mode = "direct_out" if policy_entry["family"] == "q4_k_packed_u32_direct" or policy_entry.get("reduction") == "direct_out" else "partial"
      if kernel_mode == "direct_out" and parts != 1:
        skipped["policy_invalid_direct_parts"] += 1
        continue
    byte_start = meta["data_start"] + off
    if byte_start % 4 != 0:
      skipped["misaligned"] += 1
      continue
    module_path = name[:-len(".weight")]
    try: module = _module_at(model, module_path)
    except (AttributeError, IndexError, ValueError):
      skipped["missing_module"] += 1
      continue
    if not hasattr(module, "weight"):
      skipped["missing_weight"] += 1
      continue
    if getattr(module, "bias", None) is not None:
      skipped["bias"] += 1
      continue
    q4_bytes = prod(dims) // 256 * 144
    persistent_bytes = q4_bytes if storage_mode == "sidecar" else 0
    if not budget.reserve(name, persistent_bytes, "Q4_K"):
      skipped["runtime_storage_cap"] += 1
      continue
    source = raw_words[byte_start//4:byte_start//4+q4_bytes//4]
    if storage_mode == "shared":
      words, shared_bytes, nonpersistent_bytes = _shared_packed_view(meta, byte_start, q4_bytes, dtypes.uint32), q4_bytes, 0
    else:
      words = source.contiguous() if storage_mode == "q4_ondemand" else source.to(None).contiguous().realize()
      shared_bytes, nonpersistent_bytes = 0, q4_bytes if storage_mode == "q4_ondemand" else 0
    q4k_linear = Q4KPrimitiveLinear(module.weight, module.bias, words, rows, cols, parts, tuple(parse_opt(x) for x in opt_specs), name,
                                    q4_bytes, persistent_bytes, storage_mode, shared_bytes, nonpersistent_bytes, kernel_mode=kernel_mode)
    _set_module_at(model, module_path, q4k_linear)
    installed.append(q4k_linear)
  if debug:
    skipped_s = " ".join(f"{k}={v}" for k, v in sorted(skipped.items()))
    installed_s = " ".join(f"{x.name}:mode={x.kernel_mode}:parts={x.parts}:opts={[str(o) for o in x.opts]}" for x in installed[:8])
    more_s = f" ...+{len(installed)-8}" if len(installed) > 8 else ""
    summary = _qk_storage_summary(installed)
    cap = -1 if budget.cap_bytes is None else budget.cap_bytes
    print(f"Q4K_PRIMITIVE_DEBUG installed={len(installed)} skipped_total={sum(skipped.values())} {skipped_s} "
          f"source_bytes={summary['source_bytes']} storage_bytes={summary['persistent_bytes']} "
          f"shared_bytes={summary['shared_bytes']} nonpersistent_bytes={summary['nonpersistent_bytes']} "
          f"runtime_cap_bytes={cap} runtime_cap_used_bytes={budget.used_bytes} storage_mode={storage_mode}")
    if installed: print(f"Q4K_PRIMITIVE_DEBUG installed_linears {installed_s}{more_s}")
  return installed

def _install_q6k_primitives(model, gguf:pathlib.Path, meta:dict, generated_policy:dict|None=None,
                            budget:QKPrimitiveBudget|None=None, storage_mode:str="sidecar") -> list[Q6KPrimitiveLinear]:
  from extra.q6_k_gemv_primitive import parse_opt
  raw_halfs = Tensor(gguf, dtype=dtypes.uint16)
  installed: list[Q6KPrimitiveLinear] = []
  skipped: collections.Counter[str] = collections.Counter()
  budget = budget or QKPrimitiveBudget()
  debug = bool(getenv("Q6K_PRIMITIVE_DEBUG", getenv("Q4K_PRIMITIVE_DEBUG", getenv("QK_GENERATED_POLICY_DEBUG", 0))))
  for name, dims, typ, off in meta["tensor_infos"]:
    if typ != 14:
      skipped["not_q6_k"] += 1
      continue
    if len(dims) != 2:
      skipped["not_2d"] += 1
      continue
    if not name.endswith(".weight"):
      skipped["not_weight"] += 1
      continue
    rows, cols = tuple(reversed(dims))
    if generated_policy is None:
      if (policy := _q6k_policy(name)) is None:
        skipped["policy_fallback"] += 1
        continue
      parts, opt_specs = policy
    else:
      if (policy_entry := _qk_generated_policy_entry(generated_policy, typ, rows, cols, name)) is None:
        skipped["policy_missing"] += 1
        continue
      if policy_entry["winner"] == "fused_graph":
        skipped["policy_fused"] += 1
        continue
      if policy_entry["family"] != "q6_k_packed_u16":
        skipped["policy_unsupported"] += 1
        continue
      parts, opt_specs = policy_entry["parts"], policy_entry["opts"]
    byte_start = meta["data_start"] + off
    if byte_start % 2 != 0:
      skipped["misaligned"] += 1
      continue
    module_path = name[:-len(".weight")]
    try: module = _module_at(model, module_path)
    except (AttributeError, IndexError, ValueError):
      skipped["missing_module"] += 1
      continue
    if not hasattr(module, "weight"):
      skipped["missing_weight"] += 1
      continue
    if getattr(module, "bias", None) is not None:
      skipped["bias"] += 1
      continue
    q6_bytes = prod(dims) // 256 * 210
    persistent_bytes = 0 if storage_mode == "shared" else q6_bytes
    if not budget.reserve(name, persistent_bytes, "Q6_K"):
      skipped["runtime_storage_cap"] += 1
      continue
    if storage_mode == "shared":
      halfs, shared_bytes = _shared_packed_view(meta, byte_start, q6_bytes, dtypes.uint16), q6_bytes
    else:
      halfs, shared_bytes = raw_halfs[byte_start//2:byte_start//2+q6_bytes//2].to(None).contiguous().realize(), 0
    q6k_linear = Q6KPrimitiveLinear(module.weight, module.bias, halfs, rows, cols, parts, tuple(parse_opt(x) for x in opt_specs), name,
                                    q6_bytes, persistent_bytes, storage_mode, shared_bytes, 0)
    _set_module_at(model, module_path, q6k_linear)
    installed.append(q6k_linear)
  if debug:
    skipped_s = " ".join(f"{k}={v}" for k, v in sorted(skipped.items()))
    installed_s = " ".join(f"{x.name}:parts={x.parts}:opts={[str(o) for o in x.opts]}" for x in installed[:8])
    more_s = f" ...+{len(installed)-8}" if len(installed) > 8 else ""
    summary = _qk_storage_summary(installed)
    cap = -1 if budget.cap_bytes is None else budget.cap_bytes
    print(f"Q6K_PRIMITIVE_DEBUG installed={len(installed)} skipped_total={sum(skipped.values())} {skipped_s} "
          f"source_bytes={summary['source_bytes']} storage_bytes={summary['persistent_bytes']} "
          f"shared_bytes={summary['shared_bytes']} nonpersistent_bytes={summary['nonpersistent_bytes']} "
          f"runtime_cap_bytes={cap} runtime_cap_used_bytes={budget.used_bytes} storage_mode={storage_mode}")
    if installed: print(f"Q6K_PRIMITIVE_DEBUG installed_linears {installed_s}{more_s}")
  return installed

def _demote_q6k_to_q4(model, linears:list, targets:tuple[str, ...]) -> list:
  # B3: re-quantize over-provisioned Q6_K tensors to Q4_K (offline quantizer; ffn_down measured ~free
  # quality, fewer per-token bytes -> an operating point llama.cpp's fixed Q4_K_M doesn't offer). `targets`
  # is a tuple of tensor-name substrings (e.g. ("ffn_down","attn_v")) selected by the demotion search;
  # each demoted tensor's (parts, opts) reuse _q4k_policy, with a shape-based fallback for roles it omits.
  from extra.qk_quantize import quantize_q4_k
  from extra.q4_k_gemv_primitive import parse_opt
  out = []
  for lin in linears:
    if isinstance(lin, Q6KPrimitiveLinear) and any(t in lin.name for t in targets):
      pol = _q4k_policy(lin.name) or ((4, ("LOCAL:0:32",)) if lin.out_features > 8192 else (1, ("LOCAL:0:64",)))
      parts, opt_strs = pol
      opts = tuple(parse_opt(x) for x in opt_strs)
      words = Tensor(quantize_q4_k(lin.weight.numpy())).to(None).contiguous().realize()
      q4_bytes = lin.out_features * lin.in_features // 256 * 144
      q4 = Q4KPrimitiveLinear(lin.weight, lin.bias, words, lin.out_features, lin.in_features, parts, opts,
                              lin.name, q4_bytes, q4_bytes, "sidecar")
      _set_module_at(model, lin.name[:-len(".weight")], q4)
      out.append(q4)
    else:
      out.append(lin)
  return out

def pairwise_topk(x: Tensor, k: int) -> tuple[Tensor, Tensor]:
  n = x.shape[-1]
  vals = Tensor.arange(n).reshape(1,1,n).cast(x.dtype).expand(x.shape)
  cmp = (x.unsqueeze(-1) > x.unsqueeze(-2)) | ((x.unsqueeze(-1) == x.unsqueeze(-2)) & \
    (Tensor.arange(n).reshape(1,1,n,1) < Tensor.arange(n).reshape(1,1,1,n)))
  sel = x.const_like(0).scatter(-1, cmp.sum(axis=-1).cast('int32'), vals)[:,:,n-k:].cast('int32')
  return x.gather(-1, sel), sel

@dataclass(frozen=True)
class SSMConfig:
  conv_kernel: int
  state_size: int
  group_count: int
  time_step_rank: int
  inner_size: int

@dataclass(frozen=True)
class TransformerConfig:
  num_blocks: int
  dim: int
  hidden_dim: int
  n_heads: int
  n_kv_heads: int
  norm_eps: float
  vocab_size: int
  head_dim: int
  rope_theta: float
  rope_dim: int
  v_head_dim: int
  max_context: int = 0
  qk_norm: int = 0
  num_experts: int = 0
  num_experts_per_tok: int = 0
  norm_topk_prob: bool = False
  q_lora_rank: int = 0
  kv_lora_rank: int = 0
  shared_expert_dim: int = 0
  full_attention_interval: int = 0
  attn_output_gate: bool = False
  ssm: SSMConfig|None = None
  shared_expert_gate: bool = True
  leading_dense_blocks: int = 0
  dense_hidden_dim: int = 0
  routed_scaling_factor: float = 1.0
  qkv_bias: bool = False
  expert_bias: bool = False

class FFNBlock:
  def __init__(self, config:TransformerConfig):
    self.config = config

    # --- RMSNorms --------------------------------------------------------
    self.attn_norm   = nn.RMSNorm(config.dim, config.norm_eps)
    self.ffn_norm    = nn.RMSNorm(config.dim, config.norm_eps)

    # --- feed-forward (MoE or dense) -------------------------------------
    if config.num_experts > 0:
      self.ffn_gate_inp = nn.Linear(config.dim, config.num_experts, bias=False)  # router
      if config.expert_bias: self.exp_probs_b = {"bias": Tensor.zeros(config.num_experts)}
      self.ffn_gate_exps = ExpertWeights(config.num_experts, config.dim, config.hidden_dim)
      self.ffn_up_exps = ExpertWeights(config.num_experts, config.dim, config.hidden_dim)
      self.ffn_down_exps = ExpertWeights(config.num_experts, config.hidden_dim, config.dim)
      if config.shared_expert_dim > 0:
        self.ffn_gate_shexp = nn.Linear(config.dim, config.shared_expert_dim, bias=False)
        self.ffn_up_shexp = nn.Linear(config.dim, config.shared_expert_dim, bias=False)
        self.ffn_down_shexp = nn.Linear(config.shared_expert_dim, config.dim, bias=False)
        if config.shared_expert_gate: self.ffn_gate_inp_shexp = {"weight": Tensor.zeros(config.dim)}
    else:
      self.ffn_gate    = nn.Linear(config.dim, config.hidden_dim, bias=False)
      self.ffn_up      = nn.Linear(config.dim, config.hidden_dim, bias=False)
      self.ffn_down    = nn.Linear(config.hidden_dim, config.dim, bias=False)

  def _feed_forward(self, x:Tensor) -> Tensor:
    if getattr(self, '_prefill_v2', False) and not hasattr(self, 'ffn_gate_exps') and not hasattr(self, 'ffn_gateup'):
      # prefill v2 (dense): fp16 + .contiguous()-isolated matmuls so each is a clean, warmstart-matchable TC
      # kernel (mirrors extra/qk_prefill_gate.py chained_ffn, the gated 37.5%-peak chain). MoE/fused fall through.
      if PREFILL_TENSILE_GEMM:   # research: transpose-free column-layout Tensile FFN (silent fallback if ineligible)
        col = _ffn_tensile_col(self, x)
        if col is not None: return col
      g = _pf16(self.ffn_gate, x).contiguous()
      u = _pf16(self.ffn_up, x).contiguous()
      h = (g.silu() * u).contiguous()
      return _pf16(self.ffn_down, h).contiguous()
    if hasattr(self, 'ffn_gate_exps'):
      h = x.unsqueeze(2)  # (B, T, 1, D) - add expert dim for broadcasting
      logits = self.ffn_gate_inp(x)
      if hasattr(self, 'exp_probs_b'):
        probs = logits.sigmoid()
        _, sel = pairwise_topk(probs + self.exp_probs_b["bias"], self.config.num_experts_per_tok)
        probs = probs.gather(-1, sel)
        if self.config.norm_topk_prob: probs = probs / probs.sum(axis=-1, keepdim=True)
      else:
        vals, sel = pairwise_topk(logits, self.config.num_experts_per_tok)
        probs = vals.softmax(-1) if self.config.norm_topk_prob else logits.softmax(-1).gather(-1, sel)
      probs = probs * self.config.routed_scaling_factor
      x_down = self.ffn_down_exps(sel, (self.ffn_gate_exps(sel, h).silu() * self.ffn_up_exps(sel, h)).contiguous())  # (B, T, k, D)
      out = (x_down * probs.unsqueeze(-1)).sum(axis=2)  # (B, T, D)
      if hasattr(self, 'ffn_gate_shexp'):
        shexp = self.ffn_down_shexp(self.ffn_gate_shexp(x).silu().contiguous() * self.ffn_up_shexp(x))
        if hasattr(self, 'ffn_gate_inp_shexp'): shexp = shexp * (x * self.ffn_gate_inp_shexp["weight"]).sum(axis=-1, keepdim=True).sigmoid()
        out = out + shexp
      return out
    # TODO: remove the need for this contiguous
    if hasattr(self, "ffn_gateup"):  # B1 fused gate/up
      gate, up = self.ffn_gateup(x)
      return self.ffn_down(gate.silu().contiguous() * up)
    if getenv("Q4K_UNFUSE"):  # run FFN matmuls in fp16 so RDNA3 WMMA tensor cores can apply (minimal-overhead)
      xh = x.cast(dtypes.float16)
      return self.ffn_down((self.ffn_gate(xh).silu().contiguous() * self.ffn_up(xh)).cast(dtypes.float16))
    if getenv("DECODE_FUSE_SILU_GATE", 0):
      return self.ffn_down(self.ffn_gate(x).silu() * self.ffn_up(x))
    return self.ffn_down(self.ffn_gate(x).silu().contiguous() * self.ffn_up(x))

  # given the token-prefix match, return how much cached state this block can still reuse
  def _reusable_prefix_len(self, prefix_len:int, cached_len:int) -> int: return prefix_len
  # return writes that reset this block's state after a cache mismatch
  def _state_reset_ops(self) -> list[Tensor]: return []
  def _init_state(self, x:Tensor): raise NotImplementedError
  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor: raise NotImplementedError

  def __call__(self, x: Tensor, start_pos: int|UOp):
    self._init_state(x)
    # we pass in the weights implicitly so we unpack the GGUF on the fly
    @function(precompile=True, allow_implicit=True)
    def _run(x:Tensor, start_pos:int|UOp):
      h =     x + self._attention(self.attn_norm(x), start_pos)
      if Q8_FFN_HANDWRITTEN and not hasattr(self, 'ffn_gate_exps'):
        from extra.q8_ffn_graph_route import route_q8_ffn
        if (routed := route_q8_ffn(self, h)) is not None: return (h + routed).contiguous()
      return (h + self._feed_forward(self.ffn_norm(h))).contiguous()
    return _run(x, start_pos)

class TransformerBlock(FFNBlock):
  def __init__(self, config:TransformerConfig):
    super().__init__(config)
    assert config.v_head_dim == config.head_dim, "TransformerBlock requires v_head_dim == head_dim"

    # --- attention projections (all linear, bias-free) ------------------
    q_proj_out       = config.head_dim * config.n_heads * (2 if config.attn_output_gate else 1)
    kv_proj_out      = config.head_dim * config.n_kv_heads
    self.attn_q      = nn.Linear(config.dim, q_proj_out,  bias=config.qkv_bias)
    self.attn_k      = nn.Linear(config.dim, kv_proj_out, bias=config.qkv_bias)
    self.attn_v      = nn.Linear(config.dim, kv_proj_out, bias=config.qkv_bias)
    self.attn_output = nn.Linear(config.head_dim * config.n_heads, config.dim, bias=False)
    if config.qk_norm: self.attn_q_norm, self.attn_k_norm = nn.RMSNorm(config.qk_norm, config.norm_eps), nn.RMSNorm(config.qk_norm, config.norm_eps)

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    if getattr(self, '_prefill_v2', False) and not hasattr(self, "attn_qkv"):  # prefill v2: fp16 isolated q/k/v
      q, k, v = _pf16(self.attn_q, x).contiguous(), _pf16(self.attn_k, x).contiguous(), _pf16(self.attn_v, x).contiguous()
    elif hasattr(self, "attn_qkv"): q, k, v = self.attn_qkv(x)  # B1 fused q/k/v
    else: q, k, v = self.attn_q(x), self.attn_k(x), self.attn_v(x)
    if self.config.qk_norm and self.config.qk_norm != self.config.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)

    B, T, _ = x.shape
    if self.config.attn_output_gate:
      qg = q.reshape(B, T, self.config.n_heads, 2, self.config.head_dim)
      q, gate = qg[:, :, :, 0, :], qg[:, :, :, 1, :].reshape(B, T, self.config.n_heads * self.config.head_dim)
    q = q.reshape(B, T, self.config.n_heads,    self.config.head_dim).transpose(1, 2)  # (B,H,T,Hd)
    k = k.reshape(B, T, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    v = v.reshape(B, T, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    if self.config.qk_norm == self.config.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)

    q = apply_rope(q[..., :self.config.rope_dim], self.freqs_cis[start_pos:start_pos+T]).cat(q[..., self.config.rope_dim:], dim=-1)
    k = apply_rope(k[..., :self.config.rope_dim], self.freqs_cis[start_pos:start_pos+T]).cat(k[..., self.config.rope_dim:], dim=-1)

    # NOTE: we don't want to change self.cache_kv, the function API doesn't support this well
    assigned_kv = Tensor(self.cache_kv.uop.after(self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.store(Tensor.stack(k, v).uop)))
    k = assigned_kv[0, :, :, 0:start_pos+T, :]
    v = assigned_kv[1, :, :, 0:start_pos+T, :]

    #self.cache_kv[:, :, :, start_pos:start_pos+T, :].assign(Tensor.stack(k, v))
    #k = self.cache_kv[0, :, :, 0:start_pos+T, :]
    #v = self.cache_kv[1, :, :, 0:start_pos+T, :]

    # NOTE: this mask is causal_lower_right, not the causal_upper_left generated by is_casual = True
    # TODO: this if statement should be removed and it shouldn't generate extra kernels
    mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, buffer=False).triu(start_pos+1) \
      if resolve(T != 1) else None
    if should_use_flash_decode(start_pos, T, getattr(self, "_use_flash", False)):
      # P2: Flash-Decoding for batch-1 GQA decode. Splits the symbolic-length KV cache into S chunks
      # -> Hq*S workgroups (saturates the GPU at batch 1, vs SDPA's <1% occupancy at long context).
      # Exact vs SDPA up to fp reassociation. Decode-only (T==1, symbolic start_pos).
      from extra.qk_flash_decode import flash_decode_attention
      Hq, Hkv, Hd = self.config.n_heads, self.config.n_kv_heads, self.config.head_dim
      MAXC, L = self.config.max_context, getenv("FLASH_L", 128)  # Track-3 search: L=128 >= L=256 at every ctx
      vsp = UOp.variable("start_pos", 0, MAXC - 1)  # unbound twin of start_pos (for kernel ranges)
      # variant 'gqa_coop_vec' (default): cooperative GQA V-reuse (kv-head global axis, V read once/group) PLUS
      # the output-dim d mapped to LOCAL workgroup threads, so V loads coalesce within a wavefront (gqa_coop ran
      # as 1-thread workgroups -> scalar uncoalesced loads). Byte-identical greedy; in-model vs gqa_coop
      # +6.5/+13.3/+25.5/+48.8% @ctx 512/1024/2048/4096 -- flattens the decode slope to ~llama-flat (-8%).
      # See docs/qk-gqa-coop-vector-load-result-*.
      # Route B B4 (default-off, owner-gated): replay the OWNED hand-AMDGCN flash-decode tile (extra/
      # qk_owned_flash_decode.hip) as external precompiled Ops.PROGRAM JIT graph nodes via Tensor.custom_kernel
      # (the B3 kernel; NO repack -- reads tinygrad's native [Hkv,MAXC,Hd] layout). Strictly shape/device-guarded to
      # the validated Qwen3-8B/gfx1100 decode shape; ANY mismatch or failure falls back to gqa_coop_vec.
      # ctx-gate: the owned tile only wins at long context (its KV-split combine over-splits short KV); below the
      # threshold the route falls back to gqa_coop_vec. Threshold read from the bound start_pos at trace time.
      try: _amdgcn_ctx = start_pos.unbind()[1] + T if isinstance(start_pos, UOp) else -1
      except Exception: _amdgcn_ctx = -1
      out = None
      # Hybrid route-binding guard (fail-loud, default-inert): DECODE_ATTN_BLOCK_TILE only binds in-model via the
      # whole-cache fused-xlane route. Set WITHOUT its enabling flags, the selection silently falls back to
      # owned/gqa_coop_vec -> phantom W==D (docs/decode-attention-block-tile-route-binding-scope-20260627.md). Catch
      # the partial stack HERE, before any W==D run, instead of after via the route_bound precheck. BLOCK_TILE off
      # => guard inert => byte-identical owned default. DECODE_ATTN_BLOCK_TILE_STRICT=1 (default) raises; =0 warns.
      if getenv("DECODE_ATTN_BLOCK_TILE", 0) and B == 1 and Hd == 128 and Hq == 32 and Hkv == 8 \
         and not (getenv("DECODE_ATTN_GENERATED_WHOLECACHE", 0) and getenv("DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE", 0)):
        _bt_msg = ("DECODE_ATTN_BLOCK_TILE=1 does not bind in-model without DECODE_ATTN_GENERATED_WHOLECACHE=1 and "
                   "DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE=1 -- it would silently fall back to owned/gqa_coop_vec "
                   "(phantom W==D). Set the full stack (also DECODE_ATTN_AMDGCN_TILE=0) or unset DECODE_ATTN_BLOCK_TILE.")
        if getenv("DECODE_ATTN_BLOCK_TILE_STRICT", 1): raise RuntimeError(_bt_msg)
        if getenv("DEBUG", 0): print("WARN:", _bt_msg)
      if getenv("DECODE_ATTN_GENERATED_WHOLECACHE", 0) and B == 1 and Hd == 128 and Hq == 32 and Hkv == 8 \
         and (Hq // Hkv) == 4:
        # A2 pure-search skeleton (default-off): generated flash-decode route that reads the whole assigned_kv cache
        # buffer directly. This targets lifecycle cleanliness: generated route + no owned tile + no E_49152.
        # NOTE: relaxing this guard to 14B (Hq=40,G=5) was tried — token-correct but REGRESSES ctx512 -10%
        # (the score-broadcast variant is 8B-tuned); kept 8B-scoped. See attention-combine result doc.
        from extra.qk_flash_decode import flash_decode_attention_whole_cache
        out = flash_decode_attention_whole_cache(q.reshape(Hq, Hd), assigned_kv, start_pos + T, vsp + T,
                                                 Hd, Hq, Hkv, MAXC, L)
      if out is None and getenv("DECODE_ATTN_GENERATED_SKELETON", 0) and B == 1 and Hd == 128 and Hq == 32 and Hkv == 8 \
         and (Hq // Hkv) == 4:
        # A1 pure-search skeleton (default-off): force the scheduler-generated flash-decode route and bypass the
        # owned AMDGCN tile. This is an attribution/correctness candidate, not a speed candidate. It intentionally
        # reuses the generated UOp flash kernels so the gate can prove whether a generated route can stay route-clean
        # without reintroducing full-KV materialization.
        out = flash_decode_attention(q.reshape(Hq, Hd), assigned_kv[0, 0], assigned_kv[1, 0],
                                     start_pos + T, vsp + T, Hd, Hq, Hkv, MAXC, L,
                                     variant=str(getenv("DECODE_ATTN_GENERATED_SKELETON_VARIANT",
                                                        getenv("FLASH_VARIANT", "gqa_coop_vec"))))
      # Attention-combine parity lever (rollback = DECODE_ATTN_FUSED_COMBINE=0, default-off): a GENERATED fused
      # flash-decode kernel that puts the S splits as WAVES in one workgroup per head and does the online-softmax
      # LSE combine IN LDS -> out[h,:], removing the 3 external flash_gmax/den/combine reduce kernels (the ~12-24%
      # attention_combine bucket). Structural shape class (Hkv==8, Hq%Hkv==0, Hd==128), not a model-dim hardcode.
      if out is None and getenv("DECODE_ATTN_FUSED_COMBINE", 0) and B == 1 and Hd == 128 and Hkv == 8 and Hq % Hkv == 0:
        from extra.qk_flash_decode_fused_combine import flash_decode_fused_combine
        out = flash_decode_fused_combine(q.reshape(Hq, Hd), assigned_kv, start_pos + T, vsp + T,
                                         Hd, Hq, Hkv, MAXC, getenv("FLASH_COMBINE_L", 256))
      # DEFAULT-ON (2026-06-23) for the validated gfx1100 / Qwen3-8B / B=1 / T=1 / Hq=32 / Hkv=8 / Hd=128 / ctx>=512
      # shape; strict guards keep every other shape/device on gqa. DECODE_ATTN_AMDGCN_TILE=0 disables.
      if out is None and getenv("DECODE_ATTN_AMDGCN_TILE", 1) and DECODE_ATTN_AMDGCN_ARCH_OK and B == 1 and Hd == 128 and Hq == 32 \
         and Hkv == 8 and (Hq // Hkv) == 4 and _amdgcn_ctx >= getenv("DECODE_ATTN_AMDGCN_MIN_CTX", 512):
        try:
          from extra.qk_owned_flash_decode_graph_node import amdgcn_flash_decode
          # DTYPE CONTRACT (mandatory): the owned tile kernel reads __half K/V/Q, but the canonical cache_kv is fp32.
          # Without this cast the tile reads fp32 bytes as fp16 -> NaN K -> garbage real-decode tokens (the route was
          # silently broken for real cache; W==D was only validated with a degenerate zero cache). fp16->fp16 is a no-op.
          # Validated 2026-06-23: byte-identical to gqa for 64 tokens; W==D +11.5%@ctx2048 / +16%@ctx4096.
          _Qt = q.reshape(Hq, Hd).cast(dtypes.float16)
          if getenv("DECODE_ATTN_KV_IDENTITY", 1):
            # buffer-identity read (DEFAULT-ON 2026-06-23, owner-authorized; DECODE_ATTN_KV_IDENTITY=0 disables): pass
            # the WHOLE cache_kv buffer (assigned_kv = cache_kv.after(store), no slice/reshape) so callify reads it
            # directly (no full-MAXC slice materialization E_49152); the whole-cache tile offsets K/V halves. Byte-
            # identical to the slice route; W==D +13-19% (ctx512..4096), tinygrad decode now 102-105% of llama.cpp.
            out = amdgcn_flash_decode(_Qt, assigned_kv, assigned_kv, vsp,
                                      getenv("DECODE_ATTN_AMDGCN_S", 48), MAXC,
                                      getenv("DECODE_ATTN_AMDGCN_COMBINE", "base"), whole_cache=True,
                                      # Mode-B tile-constant knobs (default 16/1/1 = shipped kernel, byte-identical):
                                      tk=getenv("DECODE_ATTN_AMDGCN_TK", 16), vec=getenv("DECODE_ATTN_AMDGCN_VEC", 1),
                                      unroll=getenv("DECODE_ATTN_AMDGCN_UNROLL", 1))
          else:
            _Kt, _Vt = assigned_kv[0, 0].cast(dtypes.float16), assigned_kv[1, 0].cast(dtypes.float16)
            out = amdgcn_flash_decode(_Qt, _Kt, _Vt, vsp,
                                      getenv("DECODE_ATTN_AMDGCN_S", 48), MAXC,
                                      getenv("DECODE_ATTN_AMDGCN_COMBINE", "base"))  # B5: 'base' or 'hd64' cheaper combine
        except Exception as e:
          if getenv("DEBUG", 0): print(f"DECODE_ATTN_AMDGCN_TILE fallback to gqa_coop_vec: {e}")
          out = None
      if out is None:
        out = flash_decode_attention(q.reshape(Hq, Hd), assigned_kv[0, 0], assigned_kv[1, 0],
                                     start_pos + T, vsp + T, Hd, Hq, Hkv, MAXC, L,
                                     variant=str(getenv("FLASH_VARIANT", "gqa_coop_vec")))
      attn = out.reshape(B, Hq, T, Hd).cast(q.dtype)
    elif PREFILL_TC_ATTN and getattr(self, '_prefill_v2', False) and isinstance(start_pos, int) and resolve(T != 1):
      # P2: Option-B explicit TC attention on CONCRETE KV. Q@Kᵀ (TC) -> fp16 scores -> softmax -> P@V (TC),
      # GQA via broadcast (K/V per kv-head expanded over the G group dim). Concrete KV=start_pos+T -> TC fires.
      Hkv, Hd, KV = self.config.n_kv_heads, self.config.head_dim, start_pos + T
      G = self.config.n_heads // Hkv; scale = Hd ** -0.5
      qg = q.reshape(B, Hkv, G, T, Hd).cast(dtypes.float16)
      kg = k.reshape(B, Hkv, 1, KV, Hd).cast(dtypes.float16)
      vg = v.reshape(B, Hkv, 1, KV, Hd).cast(dtypes.float16)
      s = ((qg @ kg.transpose(-1, -2)).float() * scale + mask.reshape(1, 1, 1, T, KV)).softmax(-1)
      attn = (s.cast(dtypes.float16) @ vg).reshape(B, self.config.n_heads, T, Hd).cast(q.dtype)  # (B,H,T,Hd)
    else:
      attn = q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)   # (B,H,T,Hd)
    attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
    out_in = attn if not self.config.attn_output_gate else (attn * gate.sigmoid())
    if getattr(self, '_prefill_v2', False): return _pf16(self.attn_output, out_in).contiguous()  # prefill v2
    if DECODE_MMVQ_IMPORT_Q4 and resolve(T == 1) and out_in.shape == (1, 1, self.config.dim) and hasattr(self.attn_output, "q4k_storage"):
      from extra.qk_decode_mmvq_graph_route import Q8_BYTES, route_imported_q4_mmvq
      if not hasattr(self, "_decode_mmvq_import_q4_q8"):
        self._decode_mmvq_import_q4_q8 = Tensor.empty(Q8_BYTES, dtype=dtypes.uint8, device=out_in.device).contiguous().realize()
        self._decode_mmvq_import_q4_out = Tensor.empty(self.config.dim, dtype=dtypes.float32, device=out_in.device).contiguous().realize()
      routed = route_imported_q4_mmvq(self.attn_output, out_in.cast(dtypes.float32).contiguous(),
                                      self._decode_mmvq_import_q4_q8, self._decode_mmvq_import_q4_out)
      if routed is not None: return routed
    return self.attn_output(out_in)

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_kv"):
      # TODO: how is the dtype of this determined?
      # DEFAULT-ON (2026-06-23) for the validated shape/device: the owned AMDGCN decode-attention route uses a native
      # fp16 cache (the tile reads __half; fp16 cache makes the cast a no-op, dropping the fp32->fp16 copy). Confirmed
      # byte-identical to gqa across the whole decode range (short-ctx SDPA + mid/long-ctx owned tile) and W==D
      # +12.7/+15.4/+18.7/+22.4% @ctx512/1024/2048/4096 (canonical harness). GATED to the supported shape so other
      # models keep fp32; DECODE_ATTN_AMDGCN_TILE=0 fully disables (back to fp32 gqa).
      _owned_supported = DECODE_ATTN_AMDGCN_ARCH_OK and x.shape[0] == 1 and self.config.n_heads == 32 \
        and self.config.n_kv_heads == 8 and self.config.head_dim == 128
      _kv_dtype = dtypes.float16 if ((getenv("DECODE_ATTN_AMDGCN_FP16CACHE") or getenv("DECODE_ATTN_AMDGCN_TILE", 1)) and _owned_supported) else None
      self.cache_kv = Tensor.empty(2, x.shape[0], self.config.n_kv_heads, self.config.max_context, self.config.head_dim, dtype=_kv_dtype, device=x.device)
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta, device=x.device)

class MLATransformerBlock(FFNBlock):
  def __init__(self, config:TransformerConfig):
    super().__init__(config)
    qk_nope_head_dim = config.head_dim - config.rope_dim
    if config.q_lora_rank > 0:
      self.attn_q_a = nn.Linear(config.dim, config.q_lora_rank, bias=False)
      self.attn_q_a_norm = nn.RMSNorm(config.q_lora_rank, config.norm_eps)
      self.attn_q_b = nn.Linear(config.q_lora_rank, config.n_heads * config.head_dim, bias=False)
    else:
      self.attn_q = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
    self.attn_kv_a_mqa = nn.Linear(config.dim, config.kv_lora_rank + config.rope_dim, bias=False)
    self.attn_kv_a_norm = nn.RMSNorm(config.kv_lora_rank, config.norm_eps)
    self.attn_k_b = {"weight": Tensor.zeros(config.n_heads, config.kv_lora_rank, qk_nope_head_dim)}
    self.attn_v_b = {"weight": Tensor.zeros(config.n_heads, config.v_head_dim, config.kv_lora_rank)}
    self.attn_output = nn.Linear(config.n_heads * config.v_head_dim, config.dim, bias=False)

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    B, T, _ = x.shape
    q_nope_head_dim = self.config.head_dim - self.config.rope_dim
    q_proj = self.attn_q_b(self.attn_q_a_norm(self.attn_q_a(x))) if self.config.q_lora_rank > 0 else self.attn_q(x)
    q = q_proj.reshape(B, T, self.config.n_heads, self.config.head_dim).transpose(1, 2)
    q_nope, q_rope = q[..., :q_nope_head_dim], q[..., q_nope_head_dim:]
    q = (q_nope @ self.attn_k_b["weight"].transpose(-1, -2)).cat(apply_rope(q_rope, self.freqs_cis[start_pos:start_pos+T]), dim=-1)

    kv_a = self.attn_kv_a_mqa(x)
    c_kv = self.attn_kv_a_norm(kv_a[..., :self.config.kv_lora_rank])
    k_rope = apply_rope(
      kv_a[..., self.config.kv_lora_rank:].reshape(B, T, 1, self.config.rope_dim).transpose(1, 2),
      self.freqs_cis[start_pos:start_pos+T])

    k_store = c_kv.reshape(B, 1, T, self.config.kv_lora_rank).cat(k_rope.reshape(B, 1, T, self.config.rope_dim), dim=-1)
    k = Tensor(self.cache_k.uop.after(self.cache_k[:, :, start_pos:start_pos+T, :].uop.store(k_store.uop)))[:, :, 0:start_pos+T, :]
    v = k[..., :self.config.kv_lora_rank]

    mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, buffer=False).triu(start_pos+1) \
      if resolve(T != 1) else None
    attn = q @ k.transpose(-1, -2) * (1.0 / self.config.head_dim ** 0.5)
    if mask is not None: attn = attn + mask
    attn = attn.softmax(-1)
    attn = ((attn @ v) @ self.attn_v_b["weight"].transpose(-1, -2)).transpose(1, 2).reshape(B, T, -1)
    return self.attn_output(attn)

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_k"):
      self.cache_k = Tensor.empty(x.shape[0], 1, self.config.max_context, self.config.kv_lora_rank + self.config.rope_dim, device=x.device)
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta, device=x.device)

class GatedDeltaNetBlock(FFNBlock):
  def __init__(self, config:TransformerConfig, ssm:SSMConfig):
    super().__init__(config)
    self.head_k_dim, self.num_k_heads, self.num_v_heads = ssm.state_size, ssm.group_count, ssm.time_step_rank
    assert self.num_v_heads % self.num_k_heads == 0
    self.head_v_dim, self.ssm_conv_kernel = ssm.inner_size // ssm.time_step_rank, ssm.conv_kernel
    self.conv_channels, self.q_dim = ssm.inner_size + 2*ssm.group_count*ssm.state_size, ssm.state_size*ssm.group_count
    self.attn_qkv, self.attn_gate = nn.Linear(config.dim, self.conv_channels, bias=False), nn.Linear(config.dim, ssm.inner_size, bias=False)
    self.ssm_alpha, self.ssm_beta = nn.Linear(config.dim, self.num_v_heads, bias=False), nn.Linear(config.dim, self.num_v_heads, bias=False)
    self.ssm_conv1d = {"weight": Tensor.zeros(self.conv_channels, self.ssm_conv_kernel)}
    self.ssm_dt = {"bias": Tensor.zeros(self.num_v_heads)}
    self.ssm_a = Tensor.zeros(self.num_v_heads)
    self.ssm_norm, self.ssm_out = nn.RMSNorm(self.head_v_dim, config.norm_eps), nn.Linear(ssm.inner_size, config.dim, bias=False)

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    B, T, _ = x.shape
    assert T == 1, "GatedDeltaNetBlock currently only supports T=1"

    # input processing
    x = x.half()
    out_gate = self.attn_gate(x).reshape(B, 1, self.num_v_heads, self.head_v_dim)
    beta = self.ssm_beta(x).sigmoid().reshape(B, self.num_v_heads, 1, 1)
    alpha = ((self.ssm_alpha(x).float() + self.ssm_dt["bias"]).softplus() * self.ssm_a).reshape(B, self.num_v_heads, 1, 1).exp()

    # qkv conv
    conv_window = self.conv_state.cat(self.attn_qkv(x), dim=1)
    conv_out = (conv_window * self.ssm_conv1d["weight"].T.unsqueeze(0)).sum(1).silu()
    q, k, v = conv_out.split([self.q_dim, self.q_dim, self.conv_channels - 2*self.q_dim], dim=-1)
    q = q.reshape(B, self.num_k_heads, self.head_k_dim).normalize(dim=-1).repeat(1, self.num_v_heads//self.num_k_heads, 1)
    k = k.reshape(B, self.num_k_heads, self.head_k_dim).normalize(dim=-1).repeat(1, self.num_v_heads//self.num_k_heads, 1)
    v = v.reshape(B, self.num_v_heads, self.head_v_dim)
    q, k, v = q.mul(self.head_k_dim**-0.5).unsqueeze(-1), k.unsqueeze(-1), v.unsqueeze(-1)

    # recurrent
    recurrent_state = self.recurrent_state * alpha
    recurrent_state = recurrent_state + ((v - recurrent_state@k) * beta)@k.transpose(-1, -2)

    # store the updated state
    conv_state_store = self.conv_state.uop.store(conv_window[:, 1:, :].cast(self.conv_state.dtype).uop)
    recurrent_state_store = self.recurrent_state.uop.store(recurrent_state.cast(self.recurrent_state.dtype).uop)
    recurrent_state = Tensor(self.recurrent_state.uop.after(recurrent_state_store, conv_state_store))

    # output
    core_attn_out = self.ssm_norm((recurrent_state@q).squeeze(-1).reshape(B, 1, self.num_v_heads, self.head_v_dim))
    return self.ssm_out((core_attn_out * out_gate.silu()).reshape(B, 1, -1).cast(x.dtype))

  # recurrent state can't be partially reused after divergence, force a full rebuild
  def _state_reset_ops(self):
    return [self.conv_state.assign(self.conv_state.const_like(0)),
            self.recurrent_state.assign(self.recurrent_state.const_like(0))] if hasattr(self, "conv_state") else []
  def _reusable_prefix_len(self, prefix_len:int, cached_len:int) -> int: return 0 if prefix_len != cached_len else prefix_len

  def _init_state(self, x):
    if not hasattr(self, "conv_state"):
      self.conv_state = Tensor.zeros(x.shape[0], self.ssm_conv_kernel-1, self.conv_channels, device=x.device).clone()
      self.recurrent_state = Tensor.zeros(x.shape[0], self.num_v_heads, self.head_v_dim, self.head_v_dim, device=x.device).clone()

class Transformer:
  def __init__(self, config:TransformerConfig):
    dense_config = replace(config, num_experts=0, num_experts_per_tok=0, shared_expert_dim=0, hidden_dim=config.dense_hidden_dim or config.hidden_dim)
    if config.ssm: config = replace(config, qk_norm=config.head_dim)
    block_cls = MLATransformerBlock if config.kv_lora_rank > 0 else TransformerBlock
    self.blk:list[FFNBlock] = [GatedDeltaNetBlock(config, config.ssm) if config.ssm and (i+1) % config.full_attention_interval != 0 else
                               block_cls(dense_config if i < config.leading_dense_blocks else config) for i in range(config.num_blocks)]
    self.token_embd  = nn.Embedding(config.vocab_size, config.dim)
    self.output_norm = nn.RMSNorm(config.dim, config.norm_eps)
    self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
    self.max_context = config.max_context
    self.has_recurrent_block = any(isinstance(b, GatedDeltaNetBlock) for b in self.blk)
    self._cached_tokens: list[int] = []
    self._q4k_linears = Q4KPrimitiveRegistry()
    # we specialize the JIT for prefill and rollout; rollout_jit_flash is the context-aware flash-decode
    # decode graph (captured lazily when ctx crosses FLASH_DECODE_THRESHOLD), see __call__/generate.
    self.prefill_jit = TinyJit(self.forward)
    self.rollout_jit = TinyJit(self.forward)
    self.rollout_jit_flash = TinyJit(self.forward)
    # prefill v2 (opt-in): a SEPARATE jit captured with a CONCRETE token batch (T=PREFILL_UBATCH) so tensor
    # cores apply, distinct from the symbolic-batch prefill_jit. Only ever called when PREFILL_V2.
    self.prefill_v2_jit = TinyJit(self.forward)
    self.prefill_v2_jits: dict = {}   # concrete-KV: one prefill jit per concrete start_pos (PREFILL_CONCRETE_KV)
    # prefill v2 warmstart table is built here but installed into the global codegen knob ONLY for the duration
    # of the prefill-v2 forward (see __call__), to contain that ambient power rather than leave it set process-wide.
    self._pf16_warmstart:dict|None = None
    if PREFILL_V2:
      _prefill_v2_validate_ubatch(PREFILL_UBATCH)
      self._pf16_warmstart = self._build_prefill_v2_warmstart()
      if PREFILL_TENSILE_GEMM:   # research-only: build Tensile runners + install routing EAGERLY (outside the prefill trace)
        from extra.qk_tensile_inmodel import install
        install()
    if Q8_FFN_HANDWRITTEN:        # research-only: install q8 decode artifacts before block function tracing
      from extra.q8_ffn_graph_route import install_q8_ffn_artifacts
      install_q8_ffn_artifacts()

  # the dense FFN + attn projection linears prefill-v2 accelerates (per block)
  _PREFILL_V2_LINEARS = ("ffn_gate", "ffn_up", "ffn_down", "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp",
                         "attn_q", "attn_k", "attn_v", "attn_output")
  def _prefill_v2_dims(self, lin):
    out_f = getattr(lin, "out_features", None) or lin.weight.shape[0]
    in_f  = getattr(lin, "in_features", None)  or lin.weight.shape[1]
    return (out_f, in_f) if isinstance(out_f, int) and isinstance(in_f, int) else (None, None)

  def _prefill_v2_covered(self):
    # (linear, out_f, in_f) for each covered linear with a known concrete shape -- single source for the
    # warmstart table, the VRAM estimate, and the realization, so they can't drift apart.
    for block in self.blk:
      for n in self._PREFILL_V2_LINEARS:
        lin = getattr(block, n, None)
        if lin is None or getattr(lin, "weight", None) is None: continue
        out_f, in_f = self._prefill_v2_dims(lin)
        if out_f is not None: yield lin, out_f, in_f

  def _build_prefill_v2_warmstart(self) -> dict:
    # The loop-found per-shape TC schedule for the prefill-v2 fp16 FFN/attn matmuls, keyed by the in-model
    # kernel signature (verified): (frozenset({out_features, PREFILL_UBATCH}), in_features). Shapes are
    # config-fixed so this is computable at init from the (pre-primitive) nn.Linears. A kernel whose key is
    # absent (e.g. a silu-fused gate) keeps the heuristic; a key that applies-then-errors falls back too
    # (postrange.py). NOT set into the global here -- installed only around the prefill-v2 forward (__call__).
    return {(frozenset({out_f, PREFILL_UBATCH}), in_f): _prefill_v2_opts(out_f, in_f)
            for _, out_f, in_f in self._prefill_v2_covered()}

  def realize_prefill_v2_weights(self) -> int:
    # Realize a clean fp16 weight per covered linear (cached as `_pf16_w`, read by _pf16). The primitives' lazy
    # Q4_K/Q6_K->fp16 dequant graph, used raw, fuses into the matmul -> ~3% peak (no TC win); a realized fp16
    # buffer makes the prefill-v2 matmul a real TC GEMM (~13x prefill on 8B). COST: ~fp16-model-size extra VRAM
    # (it coexists with the Q4_K decode storage). Gated/opt-in; called at the end of from_gguf when PREFILL_V2.
    covered = list(self._prefill_v2_covered())
    # preflight: realizing ~fp16-model-size on top of Q4_K OOMs for 14B/32B -> fail fast with the estimate.
    est_gb = _prefill_v2_realize_bytes([(o, i) for _, o, i in covered]) / 1e9
    budget_gb = getenv("PREFILL_V2_MAX_REALIZE_GB", 18)
    if est_gb > budget_gb and not getenv("PREFILL_V2_FORCE_REALIZE", 0):
      raise RuntimeError(f"PREFILL_V2 would realize ~{est_gb:.1f} GB of fp16 weights (on top of Q4_K decode "
                         f"storage), over the ~{budget_gb} GB budget -- likely OOM. This is 8B-sized work; for "
                         f"larger models raise PREFILL_V2_MAX_REALIZE_GB or set PREFILL_V2_FORCE_REALIZE=1 to "
                         f"override (a VRAM-frugal per-layer realize is future work).")
    n = 0
    for lin, _, _ in covered:
      lin._pf16_w = lin.weight.cast(dtypes.float16).contiguous().realize(); n += 1
    return n

  def precompile_concrete_prefill_jits(self) -> int:
    # Increment 0 ship: with PREFILL_CONCRETE_KV, every prefill chunk runs through a per-start_pos CONCRETE jit
    # (-> the fusion attention path, 1.7-4.4x/chunk faster than the symbolic chunk). Those jits compile on first
    # use (~5s each), so a COLD long prompt pays the tax inline. Precompiling them ONCE at load (here) moves the
    # tax to load time, so every generation -- including the first -- is warm. Bounded: ceil(max_context/UBATCH)
    # jits. Safe to leave the dummy KV behind: a fresh model's first generation starts at start_pos=0 and
    # overwrites the cache in chunk order before any position is read. gfx1100/PREFILL_V2/PREFILL_CONCRETE_KV only.
    if not (PREFILL_V2 and PREFILL_CONCRETE_KV): return 0
    temp = Tensor([0.0])
    dummy = Tensor.zeros(1, PREFILL_UBATCH, dtype="int32").contiguous().realize()
    n = 0
    for sp in range(0, self.max_context - PREFILL_UBATCH + 1, PREFILL_UBATCH):
      self(dummy, sp, temp, use_flash=False).realize(); n += 1   # populates self.prefill_v2_jits[sp]
    return n

  def logits(self, tokens:Tensor, start_pos:int|UOp) -> Tensor:
    x = self.token_embd(tokens).float()                   # (B, T, D)
    for block in self.blk: x = block(x, start_pos)
    return self.output(self.output_norm(x))

  def forward(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    logits = self.logits(tokens, start_pos)[:, -1, :]
    # Gumbel-max trick: argmax(logits/temp - log(-log(uniform))) is equivalent to sampling from softmax(logits/temp)
    return (logits / temperature.maximum(1e-12) - (Tensor.rand_like(logits).maximum(1e-12).log().neg()).log()).argmax(-1, keepdim=True)

  def __call__(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor, use_flash:bool=False) -> Tensor:
    is_prefill = resolve(tokens.shape[1] != 1)
    # prefill v2: only when opt-in AND this is a CONCRETE-batch prefill chunk. Normal prefill passes a symbolic
    # v_toks (tokens.shape[1] is a UOp -> not int), so the two paths never collide; decode is T==1.
    is_prefill_v2 = PREFILL_V2 and is_prefill and isinstance(tokens.shape[1], int)
    if getenv("Q4K_VDOT_AMORT"): _VDOT_QUANT_CACHE.clear()  # E0: fresh quant cache per forward/trace
    for q4k_linear in self._q4k_linears.linears:
      q4k_linear.decode_enabled = not is_prefill
    # context-aware flash: each block reads _use_flash at trace time; rollout_jit (SDPA) and
    # rollout_jit_flash bake distinct attention -- each is only ever called with its own use_flash, so
    # capture is consistent. The decode-only T==1 guard in _attention ignores it during prefill.
    for block in self.blk: block._use_flash, block._prefill_v2 = use_flash, is_prefill_v2
    # concrete-KV: a CONCRETE int start_pos (KV concrete -> attention TC fires) gets a per-start_pos jit.
    if is_prefill_v2 and isinstance(start_pos, int):
      jit = self.prefill_v2_jits.setdefault(start_pos, TinyJit(self.forward))
    else:
      jit = (self.prefill_v2_jit if is_prefill_v2 else self.prefill_jit) if is_prefill else \
            (self.rollout_jit_flash if use_flash else self.rollout_jit)
    if not is_prefill_v2: return jit(tokens.contiguous(), start_pos, temperature)
    # contain the ambient codegen power: install the warmstart table ONLY around the prefill-v2 forward (it's
    # consulted at kernel-compile time, i.e. this jit's first call), then restore -- decode/other paths never
    # see a populated _WARMSTART_OPTS even within this process.
    import tinygrad.codegen.opt.postrange as pr
    saved = pr._WARMSTART_OPTS
    pr._WARMSTART_OPTS = self._pf16_warmstart
    try: return jit(tokens.contiguous(), start_pos, temperature)
    finally: pr._WARMSTART_OPTS = saved

  @staticmethod
  def from_gguf(gguf:Tensor|str|pathlib.Path, max_context:int|None=None,
                realize=bool(getenv("REALIZE", 0))) -> tuple[Transformer, dict]:
    # TODO: remove the need for copy to default device
    # Q4K_PRIMITIVE defaults ON for a GGUF path ON AMD (the exact ~2.2x decode win, validated on AMD), and
    # OFF for a preloaded Tensor (no GGUF storage to view; primitive paths require a path) or a non-AMD
    # default device (the kernels are AMD-targeted). Set Q4K_PRIMITIVE=0/1 to override.
    q4k_auto = "Q4K_PRIMITIVE" not in os.environ and not isinstance(gguf, Tensor) and Device.DEFAULT == "AMD"
    use_q4k_primitive = bool(getenv("Q4K_PRIMITIVE", 1 if q4k_auto else 0))
    # Q6_K (ffn_down etc. in mixed-quant Q4_K_M) defaults ON with Q4K_PRIMITIVE: it's the decode bottleneck
    # otherwise (the slow fp-dequant fallback ~= 59% of GPU work), Q6_K dequant is exact (identical output),
    # and enabling it is a ~2.2x decode win. Set Q6K_PRIMITIVE=0 to opt out.
    use_q6k_primitive = bool(getenv("Q6K_PRIMITIVE", 1 if use_q4k_primitive else 0))
    qk_generated_policy_path = getenv("QK_GENERATED_POLICY", "")
    use_qk_generated_policy = bool(qk_generated_policy_path)
    if (use_q4k_primitive or use_q6k_primitive or use_qk_generated_policy) and isinstance(gguf, Tensor):
      raise ValueError("quant primitive paths require a GGUF path, not a preloaded Tensor")
    # QK primitive/generated linears are backed by AMD-targeted custom kernels. Auto-enable is already
    # AMD-only (q4k_auto), so this only catches an *explicit* Q4K_PRIMITIVE/Q6K_PRIMITIVE/QK_GENERATED_POLICY
    # on another backend -- fail fast with a clear message instead of an obscure later kernel failure.
    if (use_q4k_primitive or use_q6k_primitive or use_qk_generated_policy) and Device.DEFAULT != "AMD":
      raise ValueError(f"QK quant primitive paths (Q4K_PRIMITIVE/Q6K_PRIMITIVE/QK_GENERATED_POLICY) require "
                       f"DEV=AMD; the kernels are AMD-targeted. Got Device.DEFAULT={Device.DEFAULT!r}.")
    if use_q4k_primitive or use_q6k_primitive or use_qk_generated_policy:
      kv, state_dict, q4k_meta = gguf_load_with_metadata(gguf)
    else:
      kv, state_dict = gguf_load(gguf.to(None).realize() if isinstance(gguf, Tensor) else gguf)
      q4k_meta = None

    # all state items should be float16, not float32
    state_dict = {k:v.cast('float16') if getenv("HALF", 1) else v for k,v in state_dict.items()}

    # some models like Llama 3.2 don't have an output.weight, they just tie to the token_embd.weight
    if 'output.weight' not in state_dict: state_dict['output.weight'] = state_dict['token_embd.weight']

    arch = kv['general.architecture']
    max_context = min(max_context, kv[f'{arch}.context_length']) if max_context is not None else kv[f'{arch}.context_length']
    n_heads, n_kv_heads = kv[f'{arch}.attention.head_count'], kv[f'{arch}.attention.head_count_kv']

    ssm = None
    if arch in ('qwen35', 'qwen35moe'):
      ssm = SSMConfig(**{k: kv[f'{arch}.ssm.{k}'] for k in ('conv_kernel','state_size','group_count','time_step_rank','inner_size')})
    if arch in ('qwen35', 'qwen35moe', 'glm4moe'):
      state_dict = {k.replace('post_attention_norm', 'ffn_norm'):v for k,v in state_dict.items()}

    kv_lora_rank = kv.get(f'{arch}.attention.kv_lora_rank', 0)
    head_dim = kv.get(f'{arch}.attention.key_length_mla', kv.get(f'{arch}.attention.key_length', kv[f'{arch}.embedding_length'] // n_heads))
    rope_dim = kv.get(f'{arch}.rope.dimension_count', head_dim)

    # Permute RoPE weights from interleaved to half-split layout.
    for name in state_dict:
      if ('attn_q.weight' in name or 'attn_q_b.weight' in name) and (arch == 'llama' or kv_lora_rank):
        w = state_dict[name].reshape(n_heads, state_dict[name].shape[0]//n_heads, -1)
        prefix = head_dim-rope_dim
        state_dict[name] = w[:, :prefix].cat(w[:, prefix:].rearrange("n (h two) d -> n (two h) d", two=2), dim=1).reshape(-1, w.shape[-1])
      elif arch == 'llama' and 'attn_k.weight' in name:
        w = state_dict[name].reshape(n_kv_heads, state_dict[name].shape[0]//n_kv_heads, -1)
        state_dict[name] = w.rearrange("n (h two) d -> n (two h) d", two=2).reshape(-1, w.shape[-1])
      elif kv_lora_rank and 'attn_kv_a_mqa.weight' in name:
        state_dict[name] = state_dict[name][:kv_lora_rank].cat(state_dict[name][kv_lora_rank:].rearrange("(h two) d -> (two h) d", two=2), dim=0)
    config = TransformerConfig(
      num_blocks=kv[f'{arch}.block_count'] - kv.get(f'{arch}.nextn_predict_layers', 0), dim=kv[f'{arch}.embedding_length'],
      hidden_dim=kv.get(f'{arch}.expert_feed_forward_length', kv.get(f'{arch}.feed_forward_length', 0)),
      n_heads=n_heads, n_kv_heads=n_kv_heads, norm_eps=kv[f'{arch}.attention.layer_norm_rms_epsilon'],
      vocab_size=len(kv['tokenizer.ggml.tokens']),
      head_dim=head_dim,
      rope_theta=kv[f'{arch}.rope.freq_base'],
      rope_dim=rope_dim,
      v_head_dim=kv.get(f'{arch}.attention.value_length_mla', kv.get(f'{arch}.attention.value_length', head_dim)),
      max_context=max_context,
      qk_norm=int(state_dict['blk.0.attn_q_norm.weight'].shape[0]) if 'blk.0.attn_q_norm.weight' in state_dict else 0,
      num_experts=kv.get(f'{arch}.expert_count', 0), num_experts_per_tok=kv.get(f'{arch}.expert_used_count', 0),
      norm_topk_prob=kv.get(f'{arch}.expert_weights_norm', arch in ('qwen3moe', 'qwen35moe')),
      kv_lora_rank=kv_lora_rank, q_lora_rank=kv.get(f'{arch}.attention.q_lora_rank', 0),
      leading_dense_blocks=kv.get(f'{arch}.leading_dense_block_count', 0),
      shared_expert_dim=kv.get(
        f'{arch}.expert_shared_feed_forward_length',
        kv.get(f'{arch}.expert_shared_count', 0) * kv.get(f'{arch}.expert_feed_forward_length', 0)),
      shared_expert_gate=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.ffn_gate_inp_shexp.weight" in state_dict,
      dense_hidden_dim=kv.get(f'{arch}.feed_forward_length', 0) if kv.get(f'{arch}.leading_dense_block_count', 0) else 0,
      routed_scaling_factor=kv.get(f'{arch}.expert_weights_scale', 1.0), attn_output_gate=arch in ('qwen35', 'qwen35moe'), ssm=ssm,
      full_attention_interval=kv.get(f'{arch}.full_attention_interval', 0),
      qkv_bias='blk.0.attn_q.bias' in state_dict,
      expert_bias=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.exp_probs_b.bias" in state_dict)
    # Prefill policy auto-resolution, BEFORE Transformer() (its __init__ reads PREFILL_V2 to build the warmstart,
    # and the concrete-KV precompile + generate() read PREFILL_CONCRETE_KV). Explicit 0/1 skip these.
    # PREFILL_SERVER_PROFILE=1 implies PREFILL_V2=auto (when V2 unset) + concrete-KV on (when V2 ends up on).
    _v2_auto = PREFILL_V2_AUTO or (PREFILL_SERVER_PROFILE and "PREFILL_V2" not in os.environ)
    if _v2_auto:
      _cov = tuple(f"{n}.weight" for n in Transformer._PREFILL_V2_LINEARS)
      est_fp16 = sum(t.numel() * 2 for k, t in state_dict.items() if any(k.endswith(s) for s in _cov))
      q4_bytes = pathlib.Path(gguf).stat().st_size if not isinstance(gguf, Tensor) else 0
      kv_bytes = 2 * config.n_kv_heads * config.max_context * config.head_dim * 2 * config.num_blocks
      enabled, reason = prefill_v2_auto_decision(_detect_total_vram_bytes(), est_fp16, q4_bytes, kv_bytes)
      _set_prefill_v2(enabled)
      print(f"PREFILL_V2=auto -> {'ON' if enabled else 'OFF'}: {reason} "
            f"(fp16 covered {est_fp16/1e9:.1f}GB, Q4 {q4_bytes/1e9:.1f}GB, KV {kv_bytes/1e9:.1f}GB @ctx{config.max_context})")
    if PREFILL_CONCRETE_KV_AUTO or PREFILL_SERVER_PROFILE:
      ckv_on, ckv_reason = prefill_concrete_kv_auto_decision(PREFILL_SERVER_PROFILE, PREFILL_V2)
      _set_prefill_concrete_kv(ckv_on)
      print(f"PREFILL_CONCRETE_KV=auto -> {'ON' if ckv_on else 'OFF'}: {ckv_reason}")
    model = Transformer(config)
    nn.state.load_state_dict(model, state_dict, verbose=False, consume=True, realize=False)  # NOTE: rope_freqs.weight (32,) is unused
    if q4k_meta is not None:
      # auto-enabled primitives default to `shared` storage (view the GGUF in place, storage_bytes=0) so
      # large models (e.g. 32B) stay within VRAM; explicit Q4K_PRIMITIVE keeps `sidecar`; env always wins.
      qk_cfg = QKConfig.from_env(storage_default="shared" if q4k_auto else "sidecar")
      primitive_linears = []
      primitive_budget = QKPrimitiveBudget(qk_cfg.max_storage_bytes, qk_cfg.generated_policy_strict)
      q4_storage_mode, q6_storage_mode = qk_cfg.storage_mode, qk_cfg.q6_storage_mode
      generated_policy = _load_qk_generated_policy(qk_generated_policy_path) if use_qk_generated_policy else None
      if generated_policy is not None:
        if qk_cfg.policy_debug:
          print(f"QK_GENERATED_POLICY_DEBUG loaded={qk_generated_policy_path} entries={_qk_generated_policy_len(generated_policy)}")
        primitive_linears += _install_q4k_primitives(model, pathlib.Path(gguf), q4k_meta, generated_policy, primitive_budget, q4_storage_mode)
        primitive_linears += _install_q6k_primitives(model, pathlib.Path(gguf), q4k_meta, generated_policy, primitive_budget, q6_storage_mode)
      else:
        if use_q4k_primitive: primitive_linears += _install_q4k_primitives(model, pathlib.Path(gguf), q4k_meta, None, primitive_budget, q4_storage_mode)
        if use_q6k_primitive: primitive_linears += _install_q6k_primitives(model, pathlib.Path(gguf), q4k_meta, None, primitive_budget, q6_storage_mode)
      if qk_cfg.storage_debug:
        summary = _qk_storage_summary(primitive_linears)
        cap = -1 if primitive_budget.cap_bytes is None else primitive_budget.cap_bytes
        by_kind_s = ",".join(f"{k}:{v}" for k, v in summary["by_kind"].items()) or "none"
        by_mode_s = ",".join(f"{k}:{v}" for k, v in summary["by_mode"].items()) or "none"
        print(f"QK_PRIMITIVE_STORAGE_DEBUG installed={len(primitive_linears)} source_bytes={summary['source_bytes']} "
              f"storage_bytes={summary['persistent_bytes']} shared_bytes={summary['shared_bytes']} "
              f"nonpersistent_bytes={summary['nonpersistent_bytes']} runtime_cap_bytes={cap} "
              f"runtime_cap_used_bytes={primitive_budget.used_bytes} by_kind={by_kind_s} by_mode={by_mode_s} "
              f"requested_storage_mode={q4_storage_mode} q4_effective_storage_mode={q4_storage_mode} "
              f"q6_effective_storage_mode={q6_storage_mode}")
      if primitive_linears and qk_cfg.demote_targets:  # B3: requant over-provisioned Q6 tensors -> Q4 (searched set)
        primitive_linears = _demote_q6k_to_q4(model, primitive_linears, qk_cfg.demote_targets)
      if primitive_linears: model._q4k_linears = Q4KPrimitiveRegistry(primitive_linears)
      if primitive_linears and qk_cfg.fuse_q4k:  # B1 horizontal-fusion probe -- REFUTED, experimental only
        import sys as _sys
        print("WARNING: Q4K_FUSE is REFUTED (qk-gemv-final-mile-20260617.md): ~-18% decode + prefill fallback "
              "broken on T>32. Enabled only for experiments; do NOT use for shipped benchmarks.", file=_sys.stderr)
        _install_q4k_fusions(model)
    # NOTE: without this contiguous, it unpacks the weights from the model every time. we shouldn't need this, but for now it's faster
    if realize:
      for s in (params:=nn.state.get_parameters(model)): s.replace(s.contiguous())
      Tensor.realize(*params)
    # prefill v2 (opt-in): realize fp16 weights now that primitives are installed (shapes/dequant graphs ready)
    if PREFILL_V2: model.realize_prefill_v2_weights()
    # Increment 0 ship: with PREFILL_CONCRETE_KV, precompile the per-start_pos concrete prefill jits at load so the
    # ~5s/jit compile tax is paid once here, not inline on a cold prompt -> every generation is warm.
    if PREFILL_V2 and PREFILL_CONCRETE_KV: model.precompile_concrete_prefill_jits()
    return model, kv

  def get_start_pos(self, tokens:list[int]) -> int:
    prefix_len = sum(1 for _ in itertools.takewhile(lambda ab: ab[0] == ab[1], zip(tokens[:-1], self._cached_tokens)))
    return min(block._reusable_prefix_len(prefix_len, len(self._cached_tokens)) for block in self.blk)

  def warmup_flash_decode(self):
    # Pre-capture rollout_jit_flash so the in-generation crossover at ctx>=FLASH_DECODE_THRESHOLD does not pay the
    # one-time jit-compile stall inline (the first long generation would otherwise pause ~once at the boundary).
    # Dummy single-token decode at a high symbolic start_pos; cache contents are irrelevant for graph capture.
    if self.has_recurrent_block: return
    ctx = min(max(self.max_context // 2, getenv("FLASH_DECODE_THRESHOLD", 512)), self.max_context - 1)
    if ctx < 1: return
    v_sp = UOp.variable("start_pos", 0, self.max_context - 1)
    dummy, temp = Tensor([[0]], dtype="int32"), Tensor([0.0])
    for _ in range(3):
      try: self(dummy, v_sp.bind(ctx), temp, use_flash=True).realize()
      except Exception: return

  def generate(self, tokens:list[int], chunk_size:int=32, temperature:float=0.0):
    if self.has_recurrent_block: chunk_size = 1
    v_start_pos = UOp.variable("start_pos", 0, self.max_context-1)
    v_toks = UOp.variable("toks", 1, chunk_size)
    # TODO: use UOp.variable for temperature once float variables are supported
    temp = Tensor([temperature])
    # assign all input tokens once, then slice from start_pos for the model call
    t = Tensor(tokens + [0] * (self.max_context - len(tokens)), dtype="int32").reshape(1, self.max_context)
    # recompute start_pos from what's currently valid in the caches
    start_pos = self.get_start_pos(tokens)
    if start_pos < len(self._cached_tokens) and (resets := [r for b in self.blk for r in b._state_reset_ops()]): Tensor.realize(*resets)
    # flash-decode selection is centralized in should_use_flash_decode (default FLASH_DECODE=auto, threshold
    # 512): generate passes no use_flash override and lets that single authority decide per captured graph.
    out, prompt_len = None, len(tokens)
    while len(tokens) < self.max_context:
      if PREFILL_V2 and (prompt_len - start_pos) >= PREFILL_UBATCH:
        # prefill v2: a CONCRETE-T chunk of all-real prompt tokens (start_pos still symbolic; only the token
        # dim must be concrete for tensor cores). remaining>=UBATCH => start_pos<prompt_len so we slice from t.
        # concrete start_pos -> KV=start_pos+T concrete -> attention TC fires (the validated 1.24x, byte-identical).
        # Default ON for the FIRST chunk (start_pos==0): one cached concrete jit, no multi-chunk compile cost.
        # PREFILL_CONCRETE_KV=1 forces it for ALL chunks (K jits, pays off only when cached / for prompt<=512).
        use_concrete = (start_pos == 0) or PREFILL_CONCRETE_KV
        sp, ntv = (start_pos if use_concrete else v_start_pos.bind(start_pos)), PREFILL_UBATCH
        out = self(t[:, sp:sp+PREFILL_UBATCH], sp, temp, use_flash=False).realize()
      elif PREFILL_REMAINDER_FIX and PREFILL_V2 and start_pos < prompt_len and prompt_len >= PREFILL_UBATCH:
        # Phase-3 fix: a sub-UBATCH PROMPT remainder would otherwise fall to many slow 32-token symbolic calls
        # (the fallback trap). Instead process the LAST PREFILL_UBATCH tokens as ONE prefill-v2 chunk by shifting
        # the window back so it ENDS exactly at prompt_len -> all-real tokens (no padding), last position is
        # prompt_len-1 so out.item() is the next token. Re-processes the small overlap with the prior chunk (same
        # tokens -> same KV) -> correct. Symbolic start_pos reuses the one prefill_v2_jit (no per-remainder compile).
        sp = v_start_pos.bind(prompt_len - PREFILL_UBATCH)   # symbolic offset -> matches the prefill_v2_jit signature
        out = self(t[:, sp:sp+PREFILL_UBATCH], sp, temp, use_flash=False).realize()
        ntv = prompt_len - start_pos                      # advance straight to end of prompt
      else:
        sp, nt = v_start_pos.bind(start_pos), v_toks.bind(min(chunk_size, len(tokens) - start_pos))
        ntv = nt.val
        # Select the flash-decode graph (rollout_jit_flash) vs SDPA graph (rollout_jit) per-token by context, so a
        # generation that STARTS short still crosses over to flash once ctx reaches the threshold. Without this the
        # decode graph is baked SDPA at the start ctx and never switches -> short-prompt decode SDPA-degrades the
        # whole way (e.g. 85->54 tok/s by ctx512). should_use_flash_decode returns False for ntv!=1 (prefill chunks).
        out = self(t[:, sp:sp+nt] if start_pos < prompt_len or out is None else out, sp, temp,
                   use_flash=should_use_flash_decode(sp, ntv)).realize()
      start_pos += ntv
      # chunked prefill: keep processing until all prompt tokens are consumed
      if start_pos < len(tokens): continue
      tokens.append(int(out.item()))
      self._cached_tokens = tokens[:-1]
      yield tokens[-1]
