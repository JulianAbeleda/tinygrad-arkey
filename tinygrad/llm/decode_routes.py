from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any

from tinygrad import Device, Tensor, UOp, dtypes, getenv
from tinygrad.llm.decode_kernels import (Q6K_POS_EXTENT, decode_kv_rope_store_kernel,
  emit_q6k_gemv_kernel, emit_q6k_vocab_scalar_reduce_kernel, q4k_g3_lanemap_gemv_kernel,
  q6k_coop_row_tile_for_target, q6k_spec_for_role, q6k_vocab_scalar_reduce_eligible)
from tinygrad.llm.flash_decode_attention import (FLASH_DECODE_G4, FLASH_DECODE_G5, FlashDecodeCapability, FlashDecodeRouteConfig,
  flash_decode_capability_from_renderer, flash_decode_live_split_block_tile, flash_decode_target_promoted)
from tinygrad.llm.kernel_program import (KernelProgram, KernelProgramProvenance, OutputSpec, ResidualViewRequest,
                                         TypedViewRequest, execute_promoted_program, execute_research_program)
from tinygrad.llm.model_route_plan import decode_epilogue_fusion_promoted, decode_flash_combine_fusion_promoted
from tinygrad.llm.qk_layout import Q4_K, Q6_K, QuantFormat
from tinygrad.llm.route_selection import parse_route_mode
from tinygrad.uop.ops import Ops

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

  def execute(self, linear:Any, x:Tensor, binding:_LinearDecodeBinding,
              epilogue_inputs:dict[str, Tensor]|None=None) -> Tensor:
    _w = linear.q4k_storage.words.to(x.device).contiguous() if linear.q4k_storage.mode == "q4_ondemand" else linear.q4k_storage.words.to(x.device)
    # Closed-default Q4_K FFN-down MMVQ qualification route. Normal model
    # loads have no admission object, so this returns None before constructing
    # any graph. A research harness may lease exact blocks; the candidate owns
    # its Q8 provider and direct consumer and never changes W1/W3 production.
    if (ffn_down_mmvq_admission := getattr(linear,"_q4k_ffn_down_mmvq_admission",None)) is not None:
      from tinygrad.llm.q4k_ffn_down_mmvq import q4k_ffn_down_mmvq_call
      if (mmvq := q4k_ffn_down_mmvq_call(ffn_down_mmvq_admission,linear,x,binding,epilogue_inputs or {})) is not None:
        return mmvq
    # L1 M4: q4k GEMV epilogue absorption is gated by its own closed record
    # (decode-q4k-epilogue-fusion-route-policy.json, measured non-landing) -- NOT M2's
    # decode_epilogue_fusion record, which stays NV-promoted for the Q6K in-kernel merge only.
    # The o-proj residual_add variant has its OWN per-variant record
    # (decode-q4k-epilogue-resadd-route-policy.json, m4-resadd-landing-scope-20260806.md) so it can
    # promote alone while the combined M4 record (ffn_down prelude, fp16_cast) stays closed.
    q4k_epi_admitted = bool(getattr(getattr(linear, "route_admission", None), "q4k_epilogue_fusion_admitted", False))
    q4k_resadd_admitted = bool(getattr(getattr(linear, "route_admission", None), "q4k_epilogue_resadd_admitted", False))
    route_role = getattr(linear, "route_role", "")
    epi_inputs = epilogue_inputs or {}
    epi_spec = None
    prog_inputs = [_w]

    if (q4k_epi_admitted or q4k_resadd_admitted) and route_role:

      if route_role == "attn_qo" and "residual" in epi_inputs and q4k_resadd_admitted:
        from tinygrad.llm.decode_kernels import Q4KGEMVEpilogue
        epi_spec = Q4KGEMVEpilogue("residual_add")
        _xv = x[:, 0, :].reshape(binding.K).cast(dtypes.float16).contiguous()
        prog_inputs.append(_xv)
        prog_inputs.append(epi_inputs["residual"][:, 0, :].reshape(binding.N).cast(dtypes.float32))
      elif route_role == "ffn_down" and all(k in epi_inputs for k in ("gate_out", "up_out", "normed_h")) and q4k_epi_admitted:
        from tinygrad.llm.decode_kernels import Q4KGEMVEpilogue
        epi_spec = Q4KGEMVEpilogue("ffn_down_fused")
        prog_inputs.append(epi_inputs["gate_out"][:, 0, :].reshape(binding.K).cast(dtypes.float32))
        prog_inputs.append(epi_inputs["up_out"][:, 0, :].reshape(binding.K).cast(dtypes.float32))
        prog_inputs.append(epi_inputs["normed_h"][:, 0, :].reshape(binding.N).cast(dtypes.float32))
      elif route_role == "attn_kv" and not epi_inputs and q4k_epi_admitted:
        from tinygrad.llm.decode_kernels import Q4KGEMVEpilogue
        epi_spec = Q4KGEMVEpilogue("fp16_cast")
        _xv = x[:, 0, :].reshape(binding.K).cast(dtypes.float16).contiguous()
        prog_inputs.append(_xv)

    if epi_spec is None:
      _xv = x[:, 0, :].reshape(binding.K).cast(dtypes.float16).contiguous()
      prog_inputs.append(_xv)

    out_dtype = dtypes.float16 if (epi_spec is not None and epi_spec.kind == "fp16_cast") else dtypes.float32
    # M5 typed boundary (m5-variant-reopen-boundary-p0-scope-20260803.md section 3.2): the o-proj
    # attn_qo Q4K GEMV opts in to the typed input ABI so its activation prelude
    # x[:, 0, :].reshape(K).cast(fp16).contiguous() folds to a view of the fp16 combine AFTER.
    # The opt-in is specific to route_role attn_qo; ffn_down, attn_kv, and every other route keep
    # the generic flat-buffer input ABI. The validator (kernel_program.py) is fail-closed: unless
    # the producer declared an exact-matching layout and both gates are open, the request is
    # rejected and the flat-buffer ABI (with its materializing copy) is used unchanged.
    typed_input_views = (
      (TypedViewRequest(slot=1, dtype=dtypes.float16, flat_shape=(binding.K,), route_role="attn_qo"),)
      if route_role == "attn_qo" else ())
    # M4 residual_add typed input (m4-resadd-landing-scope section 2.2): the residual slot folds to
    # a zero-copy view of the ordinary block-output producer under the extended validator
    # (kernel_program._validated_residual_view). Fail-closed: any mismatch keeps the boundary copy.
    residual_input_views = (
      (ResidualViewRequest(slot=2, dtype=dtypes.float32, flat_shape=(binding.N,), route_role="attn_qo",
                           kind="residual_add"),)
      if (route_role == "attn_qo" and epi_spec is not None and epi_spec.kind == "residual_add") else ())
    program = KernelProgram(binding.route_id, f"{binding.candidate_id}.gemv",
      KernelProgramProvenance.MACHINE_SEARCH_GENERATED,
      q4k_g3_lanemap_gemv_kernel(binding.N, binding.K, epilogue=epi_spec),
      output_spec=OutputSpec((binding.N,), out_dtype),
      typed_input_views=typed_input_views,
      residual_input_views=residual_input_views)
    return execute_promoted_program(None, *prog_inputs, program=program).reshape(1, 1, binding.N)

# This is a statically promoted result of offline machine search, not an online
# autotuner. See README.md#why-this-is-machine-search-even-though-the-runtime-is-static.
Q4K_DECODE_CANDIDATE = _Q4KDecodeCandidate()

def q4k_primitive_linear_call(linear:Any, x:Tensor, fallback:Callable[[Tensor], Tensor], arch_ok:bool,
                              epilogue_inputs:dict[str, Tensor]|None=None) -> Tensor:
  # Decode GEMV (1 token) or batched verify/prefill GEMM (K tokens). Unsupported bias/shape -> normal graph.
  # epilogue_inputs are threaded to the fused variant only when the fusion gate is open; the legacy route ignores them.
  binding = Q4K_DECODE_CANDIDATE.bind(linear, x, arch_ok)
  if binding is None: return fallback(x)
  return Q4K_DECODE_CANDIDATE.execute(linear, x, binding, epilogue_inputs=epilogue_inputs or {})

def q4k_gate_up_primitive_linear_call(gate:Any, up:Any, x:Tensor, fallback:Callable[[], Tensor]) -> Tensor:
  """Fused w1+w3 decode GEMV: ONE kernel computes z = silu(gate(x)) * up(x) from two Q4_K weight buffers
  (q4k-w1w3-fused-qv-implementation-record-20260803.md). Admitted only when BOTH linears bind the legacy
  decode GEMV shape AND both carry `w1w3_fusion_admitted` (their own QKPrimitiveRouteAdmission field,
  resolved from the closed-default decode-q4k-w1w3-fusion-route-policy.json record). Any mismatch (one
  linear not Q4K, bias, K/N inequality, multi-token, off-target, shape outside the quad geometry) falls
  back to the caller's legacy graph -- the fused route never changes what the legacy chain computes."""
  g_bind = Q4K_DECODE_CANDIDATE.bind(gate, x, getattr(getattr(gate, "route_admission", None), "admitted", False))
  u_bind = Q4K_DECODE_CANDIDATE.bind(up, x, getattr(getattr(up, "route_admission", None), "admitted", False))
  if g_bind is None or u_bind is None: return fallback()
  if g_bind.T != 1 or u_bind.T != 1: return fallback()
  if g_bind.K != u_bind.K or g_bind.N != u_bind.N: return fallback()
  if not (getattr(getattr(gate, "route_admission", None), "w1w3_fusion_admitted", False) and
          getattr(getattr(up, "route_admission", None), "w1w3_fusion_admitted", False)): return fallback()
  from tinygrad.llm.decode_kernels import q4k_g3_lanemap_gemv_w1w3_kernel
  gw = gate.q4k_storage.words.to(x.device).contiguous() if gate.q4k_storage.mode == "q4_ondemand" else gate.q4k_storage.words.to(x.device)
  uw = up.q4k_storage.words.to(x.device).contiguous() if up.q4k_storage.mode == "q4_ondemand" else up.q4k_storage.words.to(x.device)
  xv = x[:, 0, :].reshape(g_bind.K).cast(dtypes.float16).contiguous()
  program = KernelProgram(g_bind.route_id, f"{g_bind.candidate_id}.w1w3_fused",
    KernelProgramProvenance.MACHINE_SEARCH_GENERATED,
    q4k_g3_lanemap_gemv_w1w3_kernel(g_bind.N, g_bind.K, load_style="scalar"),
    output_spec=OutputSpec((g_bind.N,), dtypes.float32))
  return execute_promoted_program(None, gw, uw, xv, program=program).reshape(1, 1, g_bind.N)

def q4k_gate_up_rms_affine_qualification_call(gate:Any, up:Any, raw_x:Tensor, norm_weight:Tensor, eps:float,
                                               fallback:Callable[[], Tensor]) -> Tensor:
  """One-block, default-off RMS scale + Q4 gate/up qualification boundary.

  There is deliberately no policy bit here: callers must hold an explicit
  harness-installed lease.  Every shape/type miss is an ordinary fallback.
  """
  g_bind=Q4K_DECODE_CANDIDATE.bind(gate,raw_x,getattr(getattr(gate,"route_admission",None),"admitted",False))
  u_bind=Q4K_DECODE_CANDIDATE.bind(up,raw_x,getattr(getattr(up,"route_admission",None),"admitted",False))
  if g_bind is None or u_bind is None or (g_bind.T,g_bind.K,g_bind.N)!=(1,4096,12288) or (u_bind.T,u_bind.K,u_bind.N)!=(1,4096,12288): return fallback()
  if norm_weight.shape != (4096,) or norm_weight.dtype != dtypes.float16: return fallback()
  from tinygrad.llm.decode_kernels import q4k_g3_lanemap_gemv_w1w3_rms_affine_kernel
  from tinygrad.llm.kernel_program import execute_research_program
  gw=gate.q4k_storage.words.to(raw_x.device).contiguous() if gate.q4k_storage.mode == "q4_ondemand" else gate.q4k_storage.words.to(raw_x.device)
  uw=up.q4k_storage.words.to(raw_x.device).contiguous() if up.q4k_storage.mode == "q4_ondemand" else up.q4k_storage.words.to(raw_x.device)
  xv=raw_x[:,0,:].reshape(4096).cast(dtypes.float16).contiguous()
  scale=((xv.cast(dtypes.float32)*xv.cast(dtypes.float32)).sum()/4096+eps).sqrt().reciprocal().reshape(1)
  program=KernelProgram(g_bind.route_id,f"{g_bind.candidate_id}.rms_affine_qualification",KernelProgramProvenance.RESEARCH_ONLY,
    q4k_g3_lanemap_gemv_w1w3_rms_affine_kernel(12288,4096),output_spec=OutputSpec((12288,),dtypes.float32))
  return execute_research_program(None,gw,uw,xv,norm_weight.to(raw_x.device),scale,program=program).reshape(1,1,12288)


def _kv_store_parts_view(v:Tensor) -> tuple[Tensor, int]:
  """Resolve the decode v capture to its raw GEMV parts view when the graph reduces one.

  The q4k decode GEMV emits VPART fp32 partials per row (NV: 4) and the model's v is their
  `sum(axis=1)`. The fused store kernel absorbs that reduce when it receives the PARTS view
  (shape (rows, VPART), AFTER-marked) instead of the reduced value: the walk skips the
  MEMORY_SEMANTIC/RESHAPE wrappers and, on an ADD-over-axis-1 reduce of an AFTER with int
  extent > 1, returns (Tensor(reduce.src[0]), extent). Any other graph returns (v, 1) and the
  reduce materializes as before (legacy behaviour). When there is NO parts reduce, the walk still
  unwraps a pure MEMORY_SEMANTIC/RESHAPE view chain to its producer AFTER (the q4k GEMV's output
  buffer): `custom_kernel` keeps an AFTER argument as a concrete buffer (no contiguous copy), so
  the fused kernel reads the GEMV output directly instead of materializing a 1024-element copy
  per layer."""
  u = v.uop
  while u.op in (Ops.MEMORY_SEMANTIC, Ops.RESHAPE): u = u.src[0]
  if u.op is Ops.REDUCE and u.arg == (Ops.ADD, (1,)) and u.src[0].op is Ops.AFTER:
    parts = u.src[0].shape[-1]
    if isinstance(parts, int) and parts > 1: return Tensor(u.src[0]), parts
  if u.op is Ops.AFTER: return Tensor(u), 1
  return v, 1


def decode_kv_store_route(cache:Tensor, k:Tensor, v:Tensor, freqs:Tensor, Hkv:int, Hd:int, MAXC:int,
                          vparts:int=1) -> Tensor:
  """Fused decode kv-store (decode-kv-store-chain-fusion-scope-20260803.md, Option A): ONE kernel
  ropes k in-kernel (fp32, the exact `apply_rope` arithmetic), casts k/v to the cache's own dtype
  (fp16 when the target can express fp16 and the validated shape holds, else default fp32), and writes both
  into `cache` at slot `start_pos`. The receiver (slot 0)
  IS the cache; the returned tensor is the cache AFTER the store, which is what the flash route
  reads (same contract as the legacy `cache_kv.uop.after(store)` chain it replaces). Called only when
  the model's gate is open (decode T==1, B==1, fp16/fp32 cache, full-head rope, no rope-at-read,
  promotion record for the target); any miss keeps the legacy chain byte-for-byte. `vparts` is the
  q4k GEMV parts extent (see `_kv_store_parts_view`); with vparts>1 the kernel sums the raw parts
  in-register instead of consuming the reduced v."""
  if vparts > 1 and tuple(v.shape) != (Hkv * Hd, vparts):
    raise ValueError(f"decode_kv_store_route vparts={vparts} requires v shape {(Hkv*Hd, vparts)}, got {tuple(v.shape)}")
  if vparts == 1 and tuple(v.shape) != (Hkv * Hd,): v = v.reshape(Hkv * Hd)
  program = KernelProgram("decode_kv_store_fusion", "decode_kv_rope_store",
    KernelProgramProvenance.MACHINE_SEARCH_GENERATED,
    decode_kv_rope_store_kernel(Hkv, Hd, MAXC, VPART=vparts), output_spec=None)
  return execute_promoted_program(cache, k.reshape(Hkv * Hd), v, freqs, program=program)

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
    # L1 M2 (l1-decode-plumbing-fusion-design-20260802.md section 6, classes 9/10): the in-kernel merge is
    # admitted only through the closed-default epilogue-fusion promotion record; the legacy external_sum
    # route (generic partial.sum(axis=1) merge chain) is untouched and remains the default.
    fusion_admitted = bool(getattr(getattr(linear, "route_admission", None), "fusion_admitted", False))
    # L4 vocab substrate fusion: the vocab head may select the coop in-kernel merge when the single-warp
    # constraint holds (row_tile * pos lanes <= 32, Q6KGEMVRouteSpec.validate). NV sm_120 row_tile=2 is
    # legal (2*16=32); AMD row_tile=4 (4*16=64) and Metal (no fusion admission) stay external_sum, so their
    # vocab scalar-reduce + scatter chain remains the default. The in-kernel merge exists for the coop
    # family only; the partial family's variant was measured and abandoned (M2 non-landing, design section 6).
    reduction = ("in_kernel" if fusion_admitted and binding.use_coop and binding.row_tile * Q6K_POS_EXTENT <= 32
                 else "external_sum")
    spec = q6k_spec_for_role(binding.N, binding.K, parts=binding.parts, row_tile=binding.row_tile,
                            use_coop=binding.use_coop, opts=linear.opts, target=target, reduction=reduction)
    gemv_program = KernelProgram(binding.route_id, f"{binding.candidate_id}.gemv",
      KernelProgramProvenance.MACHINE_SEARCH_GENERATED, emit_q6k_gemv_kernel(spec),
      output_spec=OutputSpec((binding.N,) if reduction == "in_kernel" else (binding.N, spec.partial_axis_extent), dtypes.float32))
    partial = execute_promoted_program(None, linear.q6k_storage.halfs.to(x.device), x_vec, program=gemv_program)
    if reduction == "in_kernel":
      return partial.reshape(1, 1, binding.N)
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
  combine_fusion: bool = False

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
                                    decode_epilogue_fusion_promoted(target),
                                    decode_flash_combine_fusion_promoted(target))
    if getenv("FLASH_DECODE_ADMISSION_DEBUG"):
      print(f"FLASH_DECODE_ADMISSION_DEBUG candidate={self.candidate_id} device={device} "
            f"admitted={admission.admitted} reason={admission.reason}")
    if not admission.admitted: return None
    return _FlashDecodeBinding(self.candidate_id, self.route_id, self.target, B, Hq, Hkv, Hd,
                               self.split_size, self.query_group_size, self.staging, self.stage_width,
                               admission.combine_fusion_admitted)

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
    stage_width=binding.stage_width, combine_fp16=binding.combine_fusion)
