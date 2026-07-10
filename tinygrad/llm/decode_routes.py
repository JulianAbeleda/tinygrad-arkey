from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any

from tinygrad import Tensor, UOp, dtypes, getenv
from tinygrad.llm import route_ops as qk_ops
from tinygrad.llm.route_policy import (
  _qk_route_policy_selected, _qk_route_policy_selects_q4k_g3,
  has_qk_route_policy, qk_route_policy_strict,
)

def _linear_role(linear:Any) -> str:
  for attr in ("route_role", "role"):
    role = str(getattr(linear, attr, ""))
    if role: return role
  name = str(getattr(linear, "name", ""))
  if "ffn_down" in name: return "ffn_down"
  if "lm_head" in name: return "lm_head"
  if "attn_kv" in name or "attn_k" in name or "attn_v" in name or ".wk" in name or ".wv" in name: return "attn_kv"
  return "unknown"

def _q4k_decode_facts(linear:Any, x:Tensor, arch_ok:bool) -> dict[str, Any]:
  shape = tuple(getattr(x, "shape", ()))
  return {
    "quant": "Q4_K" if hasattr(linear, "q4k_storage") else "unknown",
    "phase": "decode",
    "role": _linear_role(linear),
    "B": shape[0] if len(shape) == 3 else None,
    "T": shape[-2] if len(shape) == 3 else None,
    "K": shape[-1] if len(shape) == 3 else None,
    "N": getattr(linear, "out_features", None),
    "bias": getattr(linear, "bias", None) is not None,
    "decode_enabled": bool(getattr(linear, "decode_enabled", False)),
    "arch_ok": bool(arch_ok),
  }

def _q4k_g3_structural_shape(linear:Any) -> bool:
  return (linear.in_features // 256) % 4 == 0 and linear.out_features % 32 == 0

def _q4k_g3_default_manifest_shape(linear:Any) -> bool:
  return _q4k_g3_structural_shape(linear) and qk_ops.q4k_g3_manifest_shape(linear.out_features, linear.in_features)

@dataclass(frozen=True)
class _Q4KDecodeCandidate:
  candidate_id: str = "quant_linear_decode.q4k_generated_g3"
  route_id: str = "decode_q4k_g3_generated"

  def bind(self, linear:Any, x:Tensor, arch_ok:bool) -> dict[str, Any] | None:
    facts = _q4k_decode_facts(linear, x, arch_ok)
    if facts["quant"] != "Q4_K" or not facts["decode_enabled"] or facts["bias"]: return None
    if facts["B"] != 1 or facts["K"] != getattr(linear, "in_features", None): return None
    if not isinstance(facts["T"], int) or facts["T"] != 1: return None
    return facts

  def execute(self, linear:Any, x:Tensor, facts:dict[str, Any], fallback:Callable[[Tensor], Tensor]) -> Tensor:
    bubblebeam_futuresight = getenv("BUBBLEBEAM_FUTURESIGHT", 1)
    g3_policy_selected = _qk_route_policy_selects_q4k_g3(linear.out_features, linear.in_features)
    g3_bubblebeam_shape = facts["arch_ok"] and _q4k_g3_default_manifest_shape(linear)
    # Generated G3 lanemap is the default Q4_K decode route when structurally eligible.
    g3_anyshape = (bool(getenv("DECODE_Q4K_G3_ANYSHAPE", 1)) or g3_policy_selected) and facts["arch_ok"] \
      and _q4k_g3_structural_shape(linear)
    if bubblebeam_futuresight and not getenv("Q4K_GEMV_SCHEDULER") and (g3_bubblebeam_shape or g3_anyshape):
      if g3_anyshape or qk_ops.should_route_q4k_lane_partition(linear.out_features, linear.in_features):
        _w = linear.q4k_storage.words.to(x.device).contiguous() if linear.q4k_storage.mode == "q4_ondemand" else linear.q4k_storage.words.to(x.device)
        _xv = x[:, 0, :].reshape(linear.in_features).cast(dtypes.float16).contiguous()
        if getenv("DECODE_Q4K_INKERNEL_COMBINE_KV", 0) and linear.out_features <= getenv("DECODE_SPLIT_K_MAX_ROWS", 2048):
          _bpg = (linear.in_features // 256) // 4
          _cap = getenv("DECODE_SPLIT_K_TARGET_WG", 8192)
          _parts = max((p for p in range(1, _bpg + 1) if _bpg % p == 0 and linear.out_features * p <= _cap), default=1)
          if _parts > 1:
            _out = Tensor.empty(linear.out_features, dtype=dtypes.float32, device=x.device)
            return _out.custom_kernel(_w, _xv,
              fxn=qk_ops.q4k_g3_lanemap_gemv_inkernel_combine_kernel(linear.out_features, linear.in_features, _parts))[0].reshape(1, 1, linear.out_features)
        if getenv("DECODE_Q4K_SPLIT_K_KV", 0) and linear.out_features <= getenv("DECODE_SPLIT_K_MAX_ROWS", 2048):
          _bpg = (linear.in_features // 256) // 4
          _cap = getenv("DECODE_SPLIT_K_TARGET_WG", 8192)
          _parts = max((p for p in range(1, _bpg + 1) if _bpg % p == 0 and linear.out_features * p <= _cap), default=1)
          if _parts > 1:
            _p = Tensor.empty(linear.out_features * _parts, dtype=dtypes.float32, device=x.device)
            _p = _p.custom_kernel(_w, _xv, fxn=qk_ops.q4k_g3_lanemap_gemv_splitk_kernel(linear.out_features, linear.in_features, _parts))[0]
            return _p.reshape(linear.out_features, _parts).sum(axis=1).reshape(1, 1, linear.out_features)
        _out = Tensor.empty(linear.out_features, dtype=dtypes.float32, device=x.device)
        return _out.custom_kernel(_w, _xv, fxn=qk_ops.q4k_g3_lanemap_gemv_kernel(linear.out_features, linear.in_features))[0].reshape(1, 1, linear.out_features)

    if g3_policy_selected and qk_route_policy_strict() and bubblebeam_futuresight and not getenv("Q4K_GEMV_SCHEDULER"):
      raise ValueError(f"TG_P2_BLOCKED_HIDDEN_FALLBACK: QK_ROUTE_POLICY selects decode_q4k_g3_generated for Q4_K "
                       f"tensor {linear.name!r} (out={linear.out_features}, in={linear.in_features}) but it did not bind to "
                       f"the G3 route (structural eligibility (in//256)%4==0 and out%32==0 not met, or arch unsupported)")

    # No backups: when the generated G3 route is not selected (BUBBLEBEAM_FUTURESIGHT=0 or a non-G3-eligible shape),
    # decode falls through to the ordinary tinygrad graph. The handwritten warp/coop/direct/vdot rollback kernels
    # that used to live here have been retired.
    return fallback(x)

Q4K_DECODE_CANDIDATE = _Q4KDecodeCandidate()

def q4k_primitive_linear_call(linear:Any, x:Tensor, fallback:Callable[[Tensor], Tensor], arch_ok:bool) -> Tensor:
  # Decode GEMV (1 token) or batched verify/prefill GEMM (K tokens). Unsupported bias/shape -> normal graph.
  facts = Q4K_DECODE_CANDIDATE.bind(linear, x, arch_ok)
  if facts is None: return fallback(x)
  return Q4K_DECODE_CANDIDATE.execute(linear, x, facts, fallback)

def _q6k_decode_facts(linear:Any, x:Tensor, arch_ok:bool) -> dict[str, Any]:
  shape = tuple(getattr(x, "shape", ()))
  return {
    "quant": "Q6_K" if hasattr(linear, "q6k_storage") else "unknown",
    "phase": "decode",
    "role": _linear_role(linear),
    "B": shape[0] if len(shape) == 3 else None,
    "T": shape[-2] if len(shape) == 3 else None,
    "K": shape[-1] if len(shape) == 3 else None,
    "N": getattr(linear, "out_features", None),
    "bias": getattr(linear, "bias", None) is not None,
    "decode_enabled": bool(getattr(linear, "decode_enabled", False)),
    "arch_ok": bool(arch_ok),
  }

def _q6k_generated_coop_enabled(linear:Any, row_tile:int) -> bool:
  if linear.parts != 1 or linear.out_features % row_tile != 0: return False
  lm_head_selected = getenv("Q6K_LM_HEAD_COOP", 1) and linear.out_features >= 100000
  ffn_down_selected = getenv("Q6K_FFN_DOWN_COOP", 1) and _linear_role(linear) == "ffn_down"
  long_k_selected = getenv("DECODE_Q6K_FFN_DOWN_LONGK", 1) and linear.in_features >= 8192 and linear.out_features < 100000
  return bool(lm_head_selected or ffn_down_selected or long_k_selected)

@dataclass(frozen=True)
class _Q6KDecodeCandidate:
  candidate_id: str = "quant_linear_decode.q6k_generated_coop"
  route_id: str = "decode_q6k_coop_generated"

  def bind(self, linear:Any, x:Tensor, arch_ok:bool) -> dict[str, Any] | None:
    facts = _q6k_decode_facts(linear, x, arch_ok)
    if facts["quant"] != "Q6_K" or not facts["decode_enabled"] or facts["bias"]: return None
    if facts["B"] != 1 or facts["K"] != getattr(linear, "in_features", None): return None
    if not isinstance(facts["T"], int) or facts["T"] != 1: return None
    return facts

  def execute(self, linear:Any, x:Tensor, facts:dict[str, Any]) -> Tensor:
    x_vec = x[:, 0, :].reshape(facts["K"]).cast(dtypes.float16).contiguous()

    # No backups: the Q6K_GEMV_WARP_DOWN handwritten warp rollback was deleted 2026-07-06. Generated Q6_K decode
    # (q6k_spec_for_role + emit_q6k_gemv_kernel) is unconditional below.
    rt = getenv("Q6K_COOP_RT", 4)
    use_coop = _q6k_generated_coop_enabled(linear, rt)
    # Generated Q6_K decode is unconditional (no-backups: the DECODE_Q6K_GENERATED=0 shipped hand rollback was deleted).
    spec = qk_ops.q6k_spec_for_role(linear.out_features, linear.in_features, role=linear.name, parts=linear.parts,
                                    row_tile=rt, use_coop=use_coop, opts=linear.opts)
    partials = Tensor.empty(linear.out_features, spec.partial_axis_extent, dtype=dtypes.float32, device=x.device)
    partial = partials.custom_kernel(linear.q6k_storage.halfs.to(x.device), x_vec, fxn=qk_ops.emit_q6k_gemv_kernel(spec))[0]
    return partial.sum(axis=1).reshape(1, 1, linear.out_features)

Q6K_DECODE_CANDIDATE = _Q6KDecodeCandidate()

def q6k_primitive_linear_call(linear:Any, x:Tensor, fallback:Callable[[Tensor], Tensor], arch_ok:bool) -> Tensor:
  # Q6_K decode GEMV (1 token) or batched verify/prefill GEMM (K tokens).
  facts = Q6K_DECODE_CANDIDATE.bind(linear, x, arch_ok)
  if facts is None: return fallback(x)
  return Q6K_DECODE_CANDIDATE.execute(linear, x, facts)

def flash_decode_attention_route(q:Tensor, assigned_kv:Tensor, start_pos:int|UOp, T:int|UOp, B:int,
                                 Hq:int, Hkv:int, Hd:int, max_context:int, kv_scale:Tensor|None=None,
                                 freqs:Tensor|None=None, ring_full:bool=False) -> Tensor:
  MAXC, L = max_context, getenv("FLASH_L", 128)
  vsp = UOp.variable("start_pos", 0, MAXC - 1)  # unbound twin of start_pos (for kernel ranges)
  # full-ring (ctx>=N): the ring buffer is full and start_pos is the wrapped WRITE slot, so the live read length is the
  # whole buffer (all MAXC slots valid) -- a CONCRETE Tc, not vsp+T. Keeps the graph's read extent constant across wrap.
  _tc = MAXC if ring_full else (vsp + T)
  out = None
  # KV-quant (assigned_kv int8 + kv_scale) and rope-at-read (assigned_kv holds UN-roped K, rotated in-kernel from
  # `freqs`) are BOTH only supported on the live-split route -- every other route here reads fp16 pre-roped KV and would
  # silently misread. Fail loud rather than emit a phantom result. The live-split path below threads kv_scale + freqs.
  _ls_only = B == 1 and Hd == 128 and Hkv == 8 and Hq % Hkv == 0 and bool(getenv("DECODE_LIVE_SPLIT", 1))
  if (kv_scale is not None or freqs is not None) and not _ls_only:
    raise RuntimeError(f"KV-quant/rope-at-read (kv_scale={kv_scale is not None}, freqs={freqs is not None}) is only "
                       "supported on the live-split decode route (B=1,Hd=128,Hkv=8,Hq%Hkv==0, DECODE_LIVE_SPLIT=1).")
  # Scorched earth: all handwritten research attention routes (whole-cache / generated-skeleton / fused-combine /
  # bypass-kv / generic flash) deleted 2026-07-06. Generated live-split is the ONLY attention kernel route.
  # Promoted generated live-split flash-decode. Structural shape class: B==1, Hd==128, Hkv==8, Hq%Hkv==0.
  # KV_BOTH is the default because K_ONLY assumes the old g5 V layout and was verified to produce bad logits on 8B.
  _ls_shape = B == 1 and Hd == 128 and Hkv == 8 and Hq % Hkv == 0
  if has_qk_route_policy():
    _ls_enabled = _ls_shape and (
      _qk_route_policy_selected("decode_flash_block_tile_g5_konly", {"B": 1, "Hq": Hq, "Hkv": Hkv, "Hd": Hd}) or
      _qk_route_policy_selected("decode_flash_live_split_g4_8b_kvboth", {"B": 1, "Hq": Hq, "Hkv": Hkv, "Hd": Hd}))
  else:
    _ls_enabled = _ls_shape and bool(getenv("DECODE_LIVE_SPLIT", 1))
  if out is None and _ls_enabled:
    out = qk_ops.flash_decode_live_split_block_tile(q.reshape(Hq, Hd), assigned_kv, _tc, Hd, Hq, Hkv, MAXC,
                                                    getenv("DECODE_LIVE_SPLIT_S", 48),
                                                    staging=str(getenv("DECODE_LIVE_SPLIT_STAGING", "KV_BOTH")),
                                                    fused_combine=True, kv_scale=kv_scale, freqs=freqs)
  if out is None:
    # No backups: the generated live-split route is the only attention kernel path. Unsupported shapes fail loud
    # (rather than silently emitting a deleted handwritten flash kernel); model.py gates flash vs SDPA upstream.
    raise RuntimeError(f"flash_decode_attention_route: shape B={B} Hd={Hd} Hkv={Hkv} Hq={Hq} "
                       f"(DECODE_LIVE_SPLIT={getenv('DECODE_LIVE_SPLIT', 1)}) is not served by the generated "
                       "live-split route, and all handwritten fallback flash routes were deleted.")
  return out
