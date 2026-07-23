from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any

from tinygrad import Tensor, UOp, dtypes
from tinygrad.llm import route_ops as qk_ops

def _decode_shape(x:Tensor) -> tuple[Any, Any, Any]:
  shape = tuple(getattr(x, "shape", ()))
  return (shape[0], shape[1], shape[2]) if len(shape) == 3 else (None, None, None)

@dataclass(frozen=True)
class _LinearDecodeBinding:
  candidate_id: str
  route_id: str
  quant: str
  target: str
  B: int
  T: int
  K: int
  N: int
  parts: int = 1
  row_tile: int = 1
  use_coop: bool = False

@dataclass(frozen=True)
class _Q4KDecodeCandidate:
  candidate_id: str = "quant_linear_decode.q4k_generated_g3"
  route_id: str = "decode_q4k_g3_generated"
  quant: str = "Q4_K"
  target: str = "amd_gfx1100"
  batch: int = 1
  tokens: int = 1
  k_multiple: int = 1024
  n_multiple: int = 32

  def bind(self, linear:Any, x:Tensor, arch_ok:bool) -> _LinearDecodeBinding | None:
    B, T, K = _decode_shape(x)
    if not hasattr(linear, "q4k_storage") or not getattr(linear, "decode_enabled", False): return None
    if getattr(linear, "bias", None) is not None or not arch_ok: return None
    if B != self.batch or T != self.tokens or not isinstance(T, int) or K != getattr(linear, "in_features", None): return None
    if not isinstance(K, int) or K <= 0 or K % self.k_multiple != 0: return None
    if not isinstance(linear.out_features, int) or linear.out_features <= 0 or linear.out_features % self.n_multiple != 0: return None
    return _LinearDecodeBinding(self.candidate_id, self.route_id, self.quant, self.target, self.batch, self.tokens,
                                linear.in_features, linear.out_features)

  def execute(self, linear:Any, x:Tensor, binding:_LinearDecodeBinding) -> Tensor:
    _w = linear.q4k_storage.words.to(x.device).contiguous() if linear.q4k_storage.mode == "q4_ondemand" else linear.q4k_storage.words.to(x.device)
    _xv = x[:, 0, :].reshape(binding.K).cast(dtypes.float16).contiguous()
    _out = Tensor.empty(binding.N, dtype=dtypes.float32, device=x.device)
    return _out.custom_kernel(_w, _xv, fxn=qk_ops.q4k_g3_lanemap_gemv_kernel(binding.N, binding.K))[0].reshape(1, 1, binding.N)

Q4K_DECODE_CANDIDATE = _Q4KDecodeCandidate()

def q4k_primitive_linear_call(linear:Any, x:Tensor, fallback:Callable[[Tensor], Tensor], arch_ok:bool) -> Tensor:
  # Decode GEMV (1 token) or batched verify/prefill GEMM (K tokens). Unsupported bias/shape -> normal graph.
  binding = Q4K_DECODE_CANDIDATE.bind(linear, x, arch_ok)
  if binding is None: return fallback(x)
  return Q4K_DECODE_CANDIDATE.execute(linear, x, binding)

@dataclass(frozen=True)
class _Q6KDecodeCandidate:
  candidate_id: str = "quant_linear_decode.q6k_generated_coop"
  route_id: str = "decode_q6k_coop_generated"
  quant: str = "Q6_K"
  target: str = "amd_gfx1100"
  batch: int = 1
  tokens: int = 1
  k_multiple: int = 256
  row_tile: int = 4

  def bind(self, linear:Any, x:Tensor, arch_ok:bool) -> _LinearDecodeBinding | None:
    B, T, K = _decode_shape(x)
    if not hasattr(linear, "q6k_storage") or not getattr(linear, "decode_enabled", False): return None
    if getattr(linear, "bias", None) is not None or not arch_ok: return None
    if B != self.batch or T != self.tokens or not isinstance(T, int) or K != getattr(linear, "in_features", None): return None
    if not isinstance(K, int) or K <= 0 or K % self.k_multiple != 0: return None
    if not isinstance(linear.out_features, int) or linear.out_features <= 0: return None
    parts = int(getattr(linear, "parts", 1))
    if parts < 1: return None
    use_coop = parts == 1 and linear.out_features % self.row_tile == 0
    return _LinearDecodeBinding(self.candidate_id, self.route_id, self.quant, self.target, self.batch, self.tokens,
                                linear.in_features, linear.out_features, parts, self.row_tile, use_coop)

  def execute(self, linear:Any, x:Tensor, binding:_LinearDecodeBinding) -> Tensor:
    x_vec = x[:, 0, :].reshape(binding.K).cast(dtypes.float16).contiguous()
    spec = qk_ops.q6k_spec_for_role(binding.N, binding.K, parts=binding.parts, row_tile=binding.row_tile,
                                    use_coop=binding.use_coop, opts=linear.opts)
    partials = Tensor.empty(binding.N, spec.partial_axis_extent, dtype=dtypes.float32, device=x.device)
    partial = partials.custom_kernel(linear.q6k_storage.halfs.to(x.device), x_vec, fxn=qk_ops.emit_q6k_gemv_kernel(spec))[0]
    if qk_ops.q6k_vocab_scalar_reduce_eligible(spec):
      out = Tensor.empty(binding.N, dtype=dtypes.float32, device=x.device)
      return out.custom_kernel(partial, fxn=qk_ops.emit_q6k_vocab_scalar_reduce_kernel(spec))[0].reshape(1, 1, binding.N)
    return partial.sum(axis=1).reshape(1, 1, binding.N)

Q6K_DECODE_CANDIDATE = _Q6KDecodeCandidate()

def q6k_primitive_linear_call(linear:Any, x:Tensor, fallback:Callable[[Tensor], Tensor], arch_ok:bool) -> Tensor:
  # Q6_K decode GEMV (1 token) or batched verify/prefill GEMM (K tokens).
  binding = Q6K_DECODE_CANDIDATE.bind(linear, x, arch_ok)
  if binding is None: return fallback(x)
  return Q6K_DECODE_CANDIDATE.execute(linear, x, binding)

@dataclass(frozen=True)
class _FlashDecodeBinding:
  candidate_id: str
  route_id: str
  target: str
  B: int
  Hq: int
  Hkv: int
  Hd: int
  split_size: int
  staging: str

@dataclass(frozen=True)
class _FlashDecodeCandidate:
  candidate_id: str = "attention_decode.flash_live_split"
  route_id: str = "decode_flash_live_split_g4_kvboth"
  target: str = "AMD"
  split_size: int = 48
  staging: str = "KV_BOTH"

  def bind(self, B:int, Hq:int, Hkv:int, Hd:int, device:str) -> _FlashDecodeBinding | None:
    if not (device == self.target or device.startswith(self.target+":")): return None
    if B != 1 or Hq <= 0 or Hd != 128 or Hkv != 8 or Hq % Hkv != 0: return None
    return _FlashDecodeBinding(self.candidate_id, self.route_id, self.target, B, Hq, Hkv, Hd,
                               self.split_size, self.staging)

FLASH_DECODE_CANDIDATE = _FlashDecodeCandidate()

def flash_decode_attention_route(q:Tensor, assigned_kv:Tensor, start_pos:int|UOp, T:int|UOp, B:int,
                                 Hq:int, Hkv:int, Hd:int, max_context:int, kv_scale:Tensor|None=None,
                                 freqs:Tensor|None=None, ring_full:bool=False) -> Tensor:
  MAXC = max_context
  vsp = UOp.variable("start_pos", 0, MAXC - 1)  # unbound twin of start_pos (for kernel ranges)
  # full-ring (ctx>=N): the ring buffer is full and start_pos is the wrapped WRITE slot, so the live read length is the
  # whole buffer (all MAXC slots valid) -- a CONCRETE Tc, not vsp+T. Keeps the graph's read extent constant across wrap.
  _tc = MAXC if ring_full else (vsp + T)
  binding = FLASH_DECODE_CANDIDATE.bind(B, Hq, Hkv, Hd, str(q.device))
  # KV-quant (assigned_kv int8 + kv_scale) and rope-at-read (assigned_kv holds UN-roped K, rotated in-kernel from
  # `freqs`) are BOTH only supported on the live-split route -- every other route here reads fp16 pre-roped KV and would
  # silently misread. Fail loud rather than emit a phantom result. The live-split path below threads kv_scale + freqs.
  if (kv_scale is not None or freqs is not None) and binding is None:
    raise RuntimeError(f"KV-quant/rope-at-read (kv_scale={kv_scale is not None}, freqs={freqs is not None}) is only "
                       "supported on the live-split decode route (B=1,Hd=128,Hkv=8,Hq%Hkv==0).")
  # Scorched earth: all handwritten research attention routes (whole-cache / generated-skeleton / fused-combine /
  # bypass-kv / generic flash) deleted 2026-07-06. Generated live-split is the ONLY attention kernel route.
  # Promoted generated live-split flash-decode. Structural shape class: B==1, Hd==128, Hkv==8, Hq%Hkv==0.
  # KV_BOTH is the default because K_ONLY assumes the old g5 V layout and was verified to produce bad logits on 8B.
  if binding is None:
    # No backups: the generated live-split route is the only attention kernel path. Unsupported shapes fail loud
    # (rather than silently emitting a deleted handwritten flash kernel); model.py gates flash vs SDPA upstream.
    raise RuntimeError(f"flash_decode_attention_route: shape B={B} Hd={Hd} Hkv={Hkv} Hq={Hq} is not served by "
                       "the generated live-split route, and all handwritten fallback flash routes were deleted.")
  return qk_ops.flash_decode_live_split_block_tile(q.reshape(binding.Hq, binding.Hd), assigned_kv, _tc,
    binding.Hd, binding.Hq, binding.Hkv, MAXC, binding.split_size, staging=binding.staging,
    fused_combine=True, kv_scale=kv_scale, freqs=freqs)
