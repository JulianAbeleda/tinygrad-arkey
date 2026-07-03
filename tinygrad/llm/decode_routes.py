from __future__ import annotations

from typing import Callable, Any

from tinygrad import Tensor, UOp, dtypes, getenv
from tinygrad.llm import route_ops as qk_ops
from tinygrad.llm.route_policy import (
  _qk_route_policy_selected, _qk_route_policy_selects_q4k_g3, _qk_route_policy_selects_q6k_generated,
  has_qk_route_policy, qk_route_policy_strict,
)

_VDOT_QUANT_CACHE: dict = {}  # per-token q8 quant cache keyed by x.uop.key

def clear_vdot_quant_cache() -> None:
  _VDOT_QUANT_CACHE.clear()

def q4k_primitive_linear_call(linear:Any, x:Tensor, fallback:Callable[[Tensor], Tensor], arch_ok:bool) -> Tensor:
  # Decode GEMV (1 token) or batched verify/prefill GEMM (K tokens). Unsupported bias/shape -> normal graph.
  if not linear.decode_enabled or linear.bias is not None or len(x.shape) != 3 or x.shape[0] != 1 or x.shape[-1] != linear.in_features:
    return fallback(x)
  K = x.shape[-2]
  if not isinstance(K, int) or K != 1:  # batched (verify/prefill); fall back for large or symbolic K
    if not isinstance(K, int) or K > 32 or linear.kernel_mode == "direct_out": return fallback(x)
    x_batch = x[0].cast(dtypes.float16).contiguous()  # [K, in_features]
    words = linear.q4k_storage.words.to(x.device).contiguous() if linear.q4k_storage.mode == "q4_ondemand" else linear.q4k_storage.words.to(x.device)
    partials = Tensor.empty(linear.out_features, K, linear.parts, dtype=dtypes.float32, device=x.device)
    gemm_opts = linear.opts + (qk_ops.q4k_parse_opt(f"UPCAST:1:{min(K, 16)}"),)
    out = partials.custom_kernel(words, x_batch.reshape(K*linear.in_features),
      fxn=qk_ops.q4k_gemm_kernel(linear.out_features, linear.in_features, K, linear.parts, "none", gemm_opts))[0]
    return out.sum(axis=2).transpose(0, 1).reshape(1, K, linear.out_features)

  bubblebeam_futuresight = getenv("BUBBLEBEAM_FUTURESIGHT", 1) or getenv("BEAM_COALESCE")
  g3_policy_selected = _qk_route_policy_selects_q4k_g3(linear.out_features, linear.in_features)
  g3_bubblebeam_shape = (linear.in_features // 256) % 4 == 0 and arch_ok and (
    (linear.in_features == 4096 and linear.out_features in (4096, 12288)) or
    (linear.in_features == 12288 and linear.out_features == 4096))
  # Generated G3 lanemap is the default Q4_K decode route when structurally eligible.
  g3_anyshape = (bool(getenv("DECODE_Q4K_G3_ANYSHAPE", 1)) or g3_policy_selected) and arch_ok \
    and (linear.in_features // 256) % 4 == 0 and linear.out_features % 32 == 0
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

  if (getenv("Q4K_GEMV_SCHEDULER") or bubblebeam_futuresight) and linear.in_features == 4096 and linear.out_features == 12288:
    if bubblebeam_futuresight and not getenv("Q4K_GEMV_SCHEDULER"):
      if not (qk_ops.should_route_q4k_lane_partition(linear.out_features, linear.in_features) or g3_bubblebeam_shape): return fallback(x)
      _w = linear.q4k_storage.words.to(x.device).contiguous() if linear.q4k_storage.mode == "q4_ondemand" else linear.q4k_storage.words.to(x.device)
      _xv = x[:, 0, :].reshape(linear.in_features).cast(dtypes.float16).contiguous()
      _out = Tensor.empty(linear.out_features, dtype=dtypes.float32, device=x.device)
      return _out.custom_kernel(_w, _xv, fxn=qk_ops.q4k_g3_lanemap_gemv_kernel(linear.out_features, linear.in_features))[0].reshape(1, 1, linear.out_features)
    if getenv("Q4K_GEMV_SCHEDULER") == 4:
      _w = linear.q4k_storage.words.to(x.device).contiguous() if linear.q4k_storage.mode == "q4_ondemand" else linear.q4k_storage.words.to(x.device)
      _xv = x[:, 0, :].reshape(linear.in_features).cast(dtypes.float16).contiguous()
      _out = Tensor.empty(linear.out_features, dtype=dtypes.float32, device=x.device)
      return _out.custom_kernel(_w, _xv, fxn=qk_ops.q4k_lane_partition_gemv_kernel(linear.out_features, linear.in_features))[0].reshape(1, 1, linear.out_features)
    if getenv("Q4K_GEMV_SCHEDULER") == 6:
      _w = linear.q4k_storage.words.to(x.device).contiguous() if linear.q4k_storage.mode == "q4_ondemand" else linear.q4k_storage.words.to(x.device)
      _xv = x[:, 0, :].reshape(linear.in_features).cast(dtypes.float16).contiguous()
      _out = Tensor.empty(linear.out_features, dtype=dtypes.float32, device=x.device)
      return _out.custom_kernel(_w, _xv, fxn=qk_ops.q4k_g3_lanemap_gemv_kernel(linear.out_features, linear.in_features))[0].reshape(1, 1, linear.out_features)
    if getenv("Q4K_GEMV_SCHEDULER") in (2, 3, 5):
      _w = linear.q4k_storage.words.to(x.device)
      _xv = x[:, 0, :].reshape(linear.in_features).cast(dtypes.float32)
      _fn = qk_ops.q4k_scheduler_matvec_lanemap if getenv("Q4K_GEMV_SCHEDULER") == 5 else qk_ops.q4k_scheduler_matvec_wordlane if getenv("Q4K_GEMV_SCHEDULER") == 3 else qk_ops.q4k_scheduler_matvec
      return _fn(_w, _xv, linear.out_features, linear.in_features).reshape(1, 1, linear.out_features)
    return fallback(x)

  x_vec = x[:, 0, :].reshape(linear.in_features).cast(dtypes.float16).contiguous()
  words = linear.q4k_storage.words.to(x.device).contiguous() if linear.q4k_storage.mode == "q4_ondemand" else linear.q4k_storage.words.to(x.device)
  if getenv("Q4K_GEMV_WARP_PROJ", 1) and linear.parts == 1 and linear.out_features == 4096 and linear.in_features == 4096 \
     and (linear.in_features // 256) % 4 == 0 and arch_ok:
    try:
      out = Tensor.empty(linear.out_features, dtype=dtypes.float32, device=x.device)
      got = out.custom_kernel(words, x_vec, fxn=qk_ops.q4k_gemv_warp_kernel(linear.out_features, linear.in_features))[0]
      return got.reshape(1, 1, linear.out_features)
    except Exception as e:
      if getenv("DEBUG", 0): print(f"Q4K_GEMV_WARP_PROJ fallback: {e}")
  rt4 = getenv("Q4K_COOP_RT", 16)
  if getenv("Q4K_ATTN_QO_COOP", 1) and linear.parts == 1 and linear.out_features == 4096 and linear.in_features == 4096 \
      and linear.out_features % rt4 == 0:
    partials = Tensor.empty(linear.out_features, 8, dtype=dtypes.float32, device=x.device)
    partial = partials.custom_kernel(words, x_vec, fxn=qk_ops.q4k_coop_partial_kernel(linear.out_features, linear.in_features, rt4))[0]
    return partial.sum(axis=1).reshape(1, 1, linear.out_features)
  if linear.kernel_mode == "direct_out":
    out = Tensor.empty(linear.out_features, dtype=dtypes.float32, device=x.device)
    got = out.custom_kernel(words, x_vec, fxn=qk_ops.q4k_gemv_kernel(linear.out_features, linear.in_features, "none", linear.opts))[0]
    return got.reshape(1, 1, linear.out_features)
  partials = Tensor.empty(linear.out_features, linear.parts, dtype=dtypes.float32, device=x.device)
  if getenv("Q4K_VDOT") and linear.parts == 1:
    amort = bool(getenv("Q4K_VDOT_AMORT"))
    ck = x.uop.key if amort else None
    cached = _VDOT_QUANT_CACHE.get(ck) if amort else None
    if cached is None:
      q, scales = qk_ops.q8_1_quantize(x_vec.cast(dtypes.float32))
      q_bias_words = Tensor.empty(linear.in_features // 4, dtype=dtypes.uint32, device=x.device).custom_kernel(
        q, fxn=qk_ops.q8_1_bias_pack_u32_kernel(linear.in_features))[0]
      if amort: _VDOT_QUANT_CACHE[ck] = (q_bias_words, scales); _VDOT_QUANT_CACHE["m"] = _VDOT_QUANT_CACHE.get("m", 0)+1
    else:
      q_bias_words, scales = cached; _VDOT_QUANT_CACHE["h"] = _VDOT_QUANT_CACHE.get("h", 0)+1
    partial = partials.custom_kernel(words, q_bias_words, scales,
      fxn=qk_ops.q4k_q8_1_vdot_builtin_partial_kernel(linear.out_features, linear.in_features, 1, "none", ()))[0]
    return partial.sum(axis=1).reshape(1, 1, linear.out_features)
  if getenv("Q4K_GEMV_WARP", 1) and (linear.in_features // 256) % 4 == 0 and arch_ok \
     and ((linear.in_features == 4096 and linear.out_features == 12288 and linear.parts == 1)
          or (getenv("Q4K_GEMV_WARP_DOWN", 1) and linear.in_features == 12288 and linear.out_features == 4096)):
    try:
      out = Tensor.empty(linear.out_features, dtype=dtypes.float32, device=x.device)
      got = out.custom_kernel(words, x_vec, fxn=qk_ops.q4k_gemv_warp_kernel(linear.out_features, linear.in_features))[0]
      return got.reshape(1, 1, linear.out_features)
    except Exception as e:
      if getenv("DEBUG", 0): print(f"Q4K_GEMV_WARP fallback: {e}")
  partial = partials.custom_kernel(words, x_vec,
    fxn=qk_ops.q4k_gemv_partial_kernel(linear.out_features, linear.in_features, linear.parts, "none", linear.opts))[0]
  return partial.sum(axis=1).reshape(1, 1, linear.out_features)

def q6k_primitive_linear_call(linear:Any, x:Tensor, fallback:Callable[[Tensor], Tensor], arch_ok:bool) -> Tensor:
  # Q6_K decode GEMV (1 token) or batched verify/prefill GEMM (K tokens).
  if not linear.decode_enabled or linear.bias is not None or len(x.shape) != 3 or x.shape[0] != 1 or x.shape[-1] != linear.in_features:
    return fallback(x)
  K = x.shape[-2]
  if not isinstance(K, int) or K != 1:  # batched (verify/prefill)
    if not isinstance(K, int) or K > 32: return fallback(x)
    x_batch = x[0].cast(dtypes.float16).contiguous()  # [K, in_features]
    partials = Tensor.empty(linear.out_features, K, linear.parts, dtype=dtypes.float32, device=x.device)
    gemm_opts = linear.opts + (qk_ops.q6k_parse_opt(f"UPCAST:1:{min(K, 16)}"),)
    out = partials.custom_kernel(linear.q6k_storage.halfs.to(x.device), x_batch.reshape(K*linear.in_features),
      fxn=qk_ops.q6k_gemm_kernel(linear.out_features, linear.in_features, K, linear.parts, gemm_opts))[0]
    return out.sum(axis=2).transpose(0, 1).reshape(1, K, linear.out_features)
  x_vec = x[:, 0, :].reshape(linear.in_features).cast(dtypes.float16).contiguous()

  if getenv("Q6K_GEMV_WARP_DOWN") and linear.parts == 1 and linear.out_features == 4096 and linear.in_features == 12288 \
     and (linear.in_features // 256) % 2 == 0 and arch_ok:
    try:
      out = Tensor.empty(linear.out_features, dtype=dtypes.float32, device=x.device)
      got = out.custom_kernel(linear.q6k_storage.halfs.to(x.device), x_vec,
                              fxn=qk_ops.q6k_gemv_warp_kernel(linear.out_features, linear.in_features))[0]
      return got.reshape(1, 1, linear.out_features)
    except Exception as e:
      if getenv("DEBUG", 0): print(f"Q6K_GEMV_WARP down fallback: {e}")
  if getenv("Q6K_DIRECT_ROUTE") and linear.parts == 1 and linear.out_features >= 100000 \
     and linear.out_features % 2 == 0 and arch_ok:
    try:
      out = Tensor.empty(linear.out_features, dtype=dtypes.float32, device=x.device)
      got = out.custom_kernel(linear.q6k_storage.halfs.to(x.device), x_vec,
                              fxn=qk_ops.q6k_halfwarp_partition_kernel(linear.out_features, linear.in_features))[0]
      return got.reshape(1, 1, linear.out_features)
    except Exception as e:
      if getenv("DEBUG", 0): print(f"Q6K_DIRECT_ROUTE lm_head fallback: {e}")
  rt = getenv("Q6K_COOP_RT", 4)
  use_coop = linear.parts == 1 and linear.out_features % rt == 0 and (
    (getenv("Q6K_LM_HEAD_COOP", 1) and linear.out_features >= 100000) or
    (getenv("Q6K_FFN_DOWN_COOP", 1) and linear.out_features == 4096 and linear.in_features == 12288) or
    (getenv("DECODE_Q6K_FFN_DOWN_LONGK", 1) and linear.in_features >= 8192 and linear.out_features < 100000))
  q6k_gen_selected = _qk_route_policy_selects_q6k_generated(linear.out_features, linear.in_features)
  q6k_generated = bool(getenv("DECODE_Q6K_GENERATED", 1))
  if q6k_generated:
    spec = qk_ops.q6k_spec_for_role(linear.out_features, linear.in_features, role=linear.name, parts=linear.parts,
                                    row_tile=rt, use_coop=use_coop, opts=linear.opts)
    partials = Tensor.empty(linear.out_features, spec.partial_axis_extent, dtype=dtypes.float32, device=x.device)
    partial = partials.custom_kernel(linear.q6k_storage.halfs.to(x.device), x_vec, fxn=qk_ops.emit_q6k_gemv_kernel(spec))[0]
    return partial.sum(axis=1).reshape(1, 1, linear.out_features)
  if q6k_gen_selected and qk_route_policy_strict():
    raise ValueError(f"TG_P3_BLOCKED_HIDDEN_FALLBACK: QK_ROUTE_POLICY selects decode_q6k_coop_generated for Q6_K "
                     f"tensor {linear.name!r} (out={linear.out_features}, in={linear.in_features}) but DECODE_Q6K_GENERATED "
                     f"is off -> it fell back to the shipped hand template")
  if use_coop:
    partials = Tensor.empty(linear.out_features, 16, dtype=dtypes.float32, device=x.device)
    partial = partials.custom_kernel(linear.q6k_storage.halfs.to(x.device), x_vec,
                                     fxn=qk_ops.q6k_coop_partial_kernel(linear.out_features, linear.in_features, rt))[0]
    return partial.sum(axis=1).reshape(1, 1, linear.out_features)
  partials = Tensor.empty(linear.out_features, linear.parts, dtype=dtypes.float32, device=x.device)
  partial = partials.custom_kernel(linear.q6k_storage.halfs.to(x.device), x_vec,
                                   fxn=qk_ops.q6k_gemv_partial_kernel(linear.out_features, linear.in_features, linear.parts, linear.opts))[0]
  return partial.sum(axis=1).reshape(1, 1, linear.out_features)

def flash_decode_attention_route(q:Tensor, assigned_kv:Tensor, start_pos:int|UOp, T:int|UOp, B:int,
                                 Hq:int, Hkv:int, Hd:int, max_context:int, kv_scale:Tensor|None=None) -> Tensor:
  MAXC, L = max_context, getenv("FLASH_L", 128)
  vsp = UOp.variable("start_pos", 0, MAXC - 1)  # unbound twin of start_pos (for kernel ranges)
  out = None
  # KV-quant tier: assigned_kv is int8 + a per-(K|V,head,token) fp16 kv_scale. ONLY the live-split route dequants
  # in-register; every other route here reads KV as fp16 and would silently misread int8. Fail loud rather than
  # feed int8 to an fp16-only route (no phantom output). The live-split path below threads kv_scale.
  if kv_scale is not None and not (B == 1 and Hd == 128 and Hkv == 8 and Hq % Hkv == 0 and bool(getenv("DECODE_LIVE_SPLIT", 1))):
    raise RuntimeError("KV-quant (int8 KV) is only supported on the live-split decode route (structural class "
                       "B=1,Hd=128,Hkv=8,Hq%Hkv==0 with DECODE_LIVE_SPLIT=1). Disable KV-quant for this shape/route.")
  if getenv("DECODE_ATTN_BLOCK_TILE", 0) and B == 1 and Hd == 128 and Hq == 32 and Hkv == 8 \
     and not (getenv("DECODE_ATTN_GENERATED_WHOLECACHE", 0) and getenv("DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE", 0)):
    _bt_msg = ("DECODE_ATTN_BLOCK_TILE=1 does not bind in-model without DECODE_ATTN_GENERATED_WHOLECACHE=1 and "
               "DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE=1 -- it would silently fall back to the generic generated "
               "flash route (phantom W==D). Set the full generated whole-cache stack or unset DECODE_ATTN_BLOCK_TILE.")
    if getenv("DECODE_ATTN_BLOCK_TILE_STRICT", 1): raise RuntimeError(_bt_msg)
    if getenv("DEBUG", 0): print("WARN:", _bt_msg)
  if getenv("DECODE_ATTN_GENERATED_WHOLECACHE", 0) and B == 1 and Hd == 128 and Hq == 32 and Hkv == 8 \
     and (Hq // Hkv) == 4:
    out = qk_ops.flash_decode_attention_whole_cache(q.reshape(Hq, Hd), assigned_kv, start_pos + T, vsp + T,
                                                    Hd, Hq, Hkv, MAXC, L)

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
    out = qk_ops.flash_decode_live_split_block_tile(q.reshape(Hq, Hd), assigned_kv, vsp + T, Hd, Hq, Hkv, MAXC,
                                                    getenv("DECODE_LIVE_SPLIT_S", 48),
                                                    staging=str(getenv("DECODE_LIVE_SPLIT_STAGING", "KV_BOTH")),
                                                    fused_combine=True, kv_scale=kv_scale)
  if out is None and getenv("DECODE_ATTN_GENERATED_SKELETON", 0) and B == 1 and Hd == 128 and Hq == 32 and Hkv == 8 \
     and (Hq // Hkv) == 4:
    out = qk_ops.flash_decode_attention(q.reshape(Hq, Hd), assigned_kv[0, 0], assigned_kv[1, 0],
                                        start_pos + T, vsp + T, Hd, Hq, Hkv, MAXC, L,
                                        variant=str(getenv("DECODE_ATTN_GENERATED_SKELETON_VARIANT",
                                                           getenv("FLASH_VARIANT", "gqa_coop_vec"))))
  if out is None and getenv("DECODE_ATTN_FUSED_COMBINE", 0) and B == 1 and Hd == 128 and Hkv == 8 and Hq % Hkv == 0:
    out = qk_ops.flash_decode_fused_combine(q.reshape(Hq, Hd), assigned_kv, start_pos + T, vsp + T,
                                            Hd, Hq, Hkv, MAXC, getenv("FLASH_COMBINE_L", 256))
  if out is None and getenv("DECODE_BYPASS_KV_SLICE", 0) and B == 1 and (getenv("FLASH_VARIANT", "gqa_coop_vec") == "gqa_coop_vec"):
    kv_flat = assigned_kv.reshape(2 * Hkv * MAXC * Hd)
    out = qk_ops.flash_decode_attention_kv_flat(q.reshape(Hq, Hd), assigned_kv[0, 0], kv_flat,
                                                start_pos + T, vsp + T, Hd, Hq, Hkv, MAXC, L)
  if out is None:
    out = qk_ops.flash_decode_attention(q.reshape(Hq, Hd), assigned_kv[0, 0], assigned_kv[1, 0],
                                        start_pos + T, vsp + T, Hd, Hq, Hkv, MAXC, L,
                                        variant=str(getenv("FLASH_VARIANT", "gqa_coop_vec")))
  return out
