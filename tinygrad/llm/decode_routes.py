from __future__ import annotations

from typing import Callable, Any

from tinygrad import Tensor, UOp, dtypes, getenv
from tinygrad.llm import route_ops as qk_ops
from tinygrad.llm.route_policy import (
  _qk_route_policy_selected, _qk_route_policy_selects_q4k_g3,
  has_qk_route_policy, qk_route_policy_strict,
)

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

  bubblebeam_futuresight = getenv("BUBBLEBEAM_FUTURESIGHT", 1)
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

  # No backups: when the generated G3 route is not selected (BUBBLEBEAM_FUTURESIGHT=0 or a non-G3-eligible shape),
  # decode falls through to the ordinary tinygrad graph. The handwritten warp/coop/direct/vdot rollback kernels
  # that used to live here have been retired.
  return fallback(x)

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
  rt = getenv("Q6K_COOP_RT", 4)
  use_coop = linear.parts == 1 and linear.out_features % rt == 0 and (
    (getenv("Q6K_LM_HEAD_COOP", 1) and linear.out_features >= 100000) or
    (getenv("Q6K_FFN_DOWN_COOP", 1) and linear.out_features == 4096 and linear.in_features == 12288) or
    (getenv("DECODE_Q6K_FFN_DOWN_LONGK", 1) and linear.in_features >= 8192 and linear.out_features < 100000))
  # Generated Q6_K decode is unconditional (no-backups: the DECODE_Q6K_GENERATED=0 shipped hand rollback was deleted).
  spec = qk_ops.q6k_spec_for_role(linear.out_features, linear.in_features, role=linear.name, parts=linear.parts,
                                  row_tile=rt, use_coop=use_coop, opts=linear.opts)
  partials = Tensor.empty(linear.out_features, spec.partial_axis_extent, dtype=dtypes.float32, device=x.device)
  partial = partials.custom_kernel(linear.q6k_storage.halfs.to(x.device), x_vec, fxn=qk_ops.emit_q6k_gemv_kernel(spec))[0]
  return partial.sum(axis=1).reshape(1, 1, linear.out_features)

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
    out = qk_ops.flash_decode_live_split_block_tile(q.reshape(Hq, Hd), assigned_kv, _tc, Hd, Hq, Hkv, MAXC,
                                                    getenv("DECODE_LIVE_SPLIT_S", 48),
                                                    staging=str(getenv("DECODE_LIVE_SPLIT_STAGING", "KV_BOTH")),
                                                    fused_combine=True, kv_scale=kv_scale, freqs=freqs)
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
