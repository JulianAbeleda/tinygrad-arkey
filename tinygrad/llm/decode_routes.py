from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any

from tinygrad import Device, Tensor, UOp, dtypes, getenv
from tinygrad.llm.decode_kernels import (emit_q6k_gemv_kernel, emit_q6k_vocab_scalar_reduce_kernel,
  q4k_g3_lanemap_gemv_kernel, q6k_coop_row_tile_for_target, q6k_spec_for_role,
  q6k_vocab_scalar_reduce_eligible)
from tinygrad.llm.flash_decode_attention import (FLASH_DECODE_G4, FLASH_DECODE_G5, FlashDecodeCapability, FlashDecodeRouteConfig,
  flash_decode_capability_from_renderer, flash_decode_live_split_block_tile, flash_decode_target_promoted)
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, OutputSpec, execute_promoted_program
from tinygrad.llm.model_route_plan import decode_epilogue_fusion_promoted
from tinygrad.llm.qk_layout import Q4_K, Q6_K, QuantFormat
from tinygrad.llm.route_selection import parse_route_mode

def decode_route_mode(getenv_fn=getenv) -> str:
  canonical = str(getenv_fn("TINYGRAD_DECODE_ROUTE", "")).strip()
  if canonical:
    return parse_route_mode("TINYGRAD_DECODE_ROUTE", allowed=("auto", "flash", "fp16"), aliases={"sdpa": "fp16", "fallback": "fp16"}, getenv_fn=getenv_fn)
  return parse_route_mode("FLASH_DECODE", allowed=("auto", "flash", "fp16"), aliases={"on": "flash", "1": "flash", "true": "flash", "off": "fp16", "0": "fp16", "false": "fp16"}, getenv_fn=getenv_fn)

def should_use_flash_decode(start_pos, T, use_flash:bool=False, getenv_fn=getenv) -> bool:
  if not (isinstance(start_pos, UOp) and isinstance(T, int) and T == 1): return False
  mode = decode_route_mode(getenv_fn)
  if mode == "fp16": return False
  if use_flash or mode == "flash": return True
  try: ctx = start_pos.unbind()[1] + T
  except Exception: return False
  return ctx >= getenv_fn("FLASH_DECODE_THRESHOLD", 512)

def _decode_shape(x:Tensor) -> tuple[Any, Any, Any]:
  shape = tuple(getattr(x, "shape", ()))
  return (shape[0], shape[1], shape[2]) if len(shape) == 3 else (None, None, None)

@dataclass(frozen=True)
class _LinearDecodeBinding:
  candidate_id: str
  route_id: str
  quant: QuantFormat
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
  quant: QuantFormat = Q4_K
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
    program = KernelProgram(binding.route_id, f"{binding.candidate_id}.gemv",
      KernelProgramProvenance.MACHINE_SEARCH_GENERATED, q4k_g3_lanemap_gemv_kernel(binding.N, binding.K),
      output_spec=OutputSpec((binding.N,), dtypes.float32))
    return execute_promoted_program(None, _w, _xv, program=program).reshape(1, 1, binding.N)

# This is a statically promoted result of offline machine search, not an online
# autotuner. See README.md#why-this-is-machine-search-even-though-the-runtime-is-static.
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
  quant: QuantFormat = Q6_K
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
    # Per-target route value (decode_kernels.py): the target facts ride on the installed
    # primitive's admission record (TG3, resolved at install time) -- never re-opened here.
    capability = getattr(getattr(linear, "route_admission", None), "capability", None)
    backend, architecture = getattr(capability, "backend", None), getattr(capability, "architecture", None)
    row_tile = q6k_coop_row_tile_for_target(backend, architecture)
    use_coop = parts == 1 and linear.out_features % row_tile == 0
    return _LinearDecodeBinding(self.candidate_id, self.route_id, self.quant, self.target, self.batch, self.tokens,
                                linear.in_features, linear.out_features, parts, row_tile, use_coop)

  def execute(self, linear:Any, x:Tensor, binding:_LinearDecodeBinding) -> Tensor:
    x_vec = x[:, 0, :].reshape(binding.K).cast(dtypes.float16).contiguous()
    capability = getattr(getattr(linear, "route_admission", None), "capability", None)
    target = f"{capability.backend}:{capability.architecture}" if capability is not None and \
      getattr(capability, "backend", None) is not None and getattr(capability, "architecture", None) is not None \
      else self.target
    spec = q6k_spec_for_role(binding.N, binding.K, parts=binding.parts, row_tile=binding.row_tile,
                            use_coop=binding.use_coop, opts=linear.opts, target=target)
    gemv_program = KernelProgram(binding.route_id, f"{binding.candidate_id}.gemv",
      KernelProgramProvenance.MACHINE_SEARCH_GENERATED, emit_q6k_gemv_kernel(spec),
      output_spec=OutputSpec((binding.N, spec.partial_axis_extent), dtypes.float32))
    partial = execute_promoted_program(None, linear.q6k_storage.halfs.to(x.device), x_vec, program=gemv_program)
    if q6k_vocab_scalar_reduce_eligible(spec):
      reduce_program = KernelProgram(binding.route_id, f"{binding.candidate_id}.vocab_reduce",
        KernelProgramProvenance.MACHINE_SEARCH_GENERATED, emit_q6k_vocab_scalar_reduce_kernel(spec),
        output_spec=OutputSpec((binding.N,), dtypes.float32))
      return execute_promoted_program(None, partial, program=reduce_program).reshape(1, 1, binding.N)
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
  query_group_size: int | None
  staging: str
  stage_width: int

_RESOLVED_FLASH_DECODE_CAPABILITY: dict[str, tuple[FlashDecodeCapability, tuple[str|None, str|None]]] = {}


def _flash_decode_capability_and_target_for_device(device:str) -> tuple[FlashDecodeCapability, tuple[str|None, str|None]]:
  """Resolve (and cache) TG7 capability + (backend, architecture) for `device`, from its actually-open
  renderer -- never inferred from the device string. `Device[device]` may only be called from an eager
  context (model setup, e.g. model.py's `_flash_decode` precondition check): `tinygrad/function.py` disallows
  it (`ALLOW_DEVICE_USAGE=0`) while a Tensor Function is dispatching, which is exactly where
  flash_decode_attention_route's real per-token call happens. Caching the first (eager) resolution and
  reusing it here means the runtime call never needs to open a device at all -- the same "resolve once at
  load time, read many times" shape as scan_device_facts()/DeviceFacts elsewhere in this scope."""
  if (cached := _RESOLVED_FLASH_DECODE_CAPABILITY.get(device)) is not None: return cached
  try: renderer = Device[device].renderer
  except Exception: renderer = None
  capability = flash_decode_capability_from_renderer(renderer)
  target = (renderer.target.device, renderer.target.arch) if renderer is not None else \
    (device.split(":", 1)[0].upper() if device else None, None)
  if renderer is not None: _RESOLVED_FLASH_DECODE_CAPABILITY[device] = (capability, target)
  return capability, target


@dataclass(frozen=True)
class _FlashDecodeCandidate:
  """Compatibility selection facade over the executor-owned route definition."""
  route: FlashDecodeRouteConfig
  target: str = "AMD"

  @property
  def candidate_id(self): return self.route.candidate_id
  @property
  def route_id(self): return self.route.route_id
  @property
  def query_heads(self): return self.route.query_heads
  @property
  def split_size(self): return self.route.split_size
  @property
  def query_group_size(self): return self.route.query_group_size
  @property
  def staging(self): return self.route.staging
  @property
  def stage_width(self): return self.route.stage_width

  def bind(self, B:int, Hq:int, Hkv:int, Hd:int, device:str, *,
           capability:FlashDecodeCapability|None=None, route_plan:Any=None) -> _FlashDecodeBinding | None:
    """TG7 (docs/task_workflow/input/target-capability-policy-decoupling-scope-20260730.md): the pre-TG7 gate
    collapsed shape, backend identity, codegen capability and promotion into one `device == "AMD"` check.
    Shape stays exactly where it was (FlashDecodeRouteConfig.shape_ok). Capability is read from the resolved
    renderer's declared facts, never inferred from `device`'s string shape: if the caller does not supply one
    explicitly, it is resolved from (and cached against) the target `device`'s own renderer -- see
    _flash_decode_capability_and_target_for_device's docstring for why this must be cached rather than
    resolved fresh on every call: the real per-token call site (flash_decode_attention_route, below) runs
    inside a Tensor Function dispatch, where opening a device is disallowed by tinygrad/function.py, so this
    depends on an earlier eager call (model.py's `_flash_decode` precondition check) having resolved and
    cached it first. Promotion defaults permissively (see flash_decode_target_promoted) until a route_plan is
    threaded through from model.py, matching AMD's current production default (no promotion record loaded)."""
    if capability is None:
      capability, target = _flash_decode_capability_and_target_for_device(device)
    else:
      target = (device.split(":", 1)[0].upper() if device else None, None)
    admission = self.route.evaluate(B, Hq, Hkv, Hd, capability, flash_decode_target_promoted(route_plan, target),
                                    decode_epilogue_fusion_promoted(target))
    if getenv("FLASH_DECODE_ADMISSION_DEBUG"):
      print(f"FLASH_DECODE_ADMISSION_DEBUG candidate={self.candidate_id} device={device} "
            f"admitted={admission.admitted} reason={admission.reason}")
    if not admission.admitted: return None
    return _FlashDecodeBinding(self.candidate_id, self.route_id, self.target, B, Hq, Hkv, Hd,
                               self.split_size, self.query_group_size, self.staging, self.stage_width)

# Public compatibility aliases. Their sole route authority is owned beside the
# flash executor, so selection cannot drift from execution configuration.
FLASH_DECODE_CANDIDATE = _FlashDecodeCandidate(FLASH_DECODE_G4)
FLASH_DECODE_G5_CANDIDATE = _FlashDecodeCandidate(FLASH_DECODE_G5)

def flash_decode_attention_route(q:Tensor, assigned_kv:Tensor, start_pos:int|UOp, T:int|UOp, B:int,
                                 Hq:int, Hkv:int, Hd:int, max_context:int, kv_scale:Tensor|None=None,
                                 freqs:Tensor|None=None, ring_full:bool=False) -> Tensor:
  MAXC = max_context
  vsp = UOp.variable("start_pos", 0, MAXC - 1)  # unbound twin of start_pos (for kernel ranges)
  # full-ring (ctx>=N): the ring buffer is full and start_pos is the wrapped WRITE slot, so the live read length is the
  # whole buffer (all MAXC slots valid) -- a CONCRETE Tc, not vsp+T. Keeps the graph's read extent constant across wrap.
  _tc = MAXC if ring_full else (vsp + T)
  candidate = FLASH_DECODE_G5_CANDIDATE if Hq == FLASH_DECODE_G5_CANDIDATE.query_heads else FLASH_DECODE_CANDIDATE
  binding = candidate.bind(B, Hq, Hkv, Hd, str(q.device))
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
  return flash_decode_live_split_block_tile(q.reshape(binding.Hq, binding.Hd), assigned_kv, _tc,
    binding.Hd, binding.Hq, binding.Hkv, MAXC, binding.split_size, staging=binding.staging,
    fused_combine=True, kv_scale=kv_scale, freqs=freqs, query_group_size=binding.query_group_size,
    stage_width=binding.stage_width)
