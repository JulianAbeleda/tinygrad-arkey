from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any

from tinygrad import Device, Tensor, UOp, dtypes, getenv
from tinygrad.llm.decode_kernels import (Q6K_POS_EXTENT, decode_kv_rope_store_kernel,
  emit_q6k_gemv_kernel, emit_q6k_vocab_scalar_reduce_kernel, q4k_g3_lanemap_gemv_kernel,
  q6k_coop_row_tile_for_target, q6k_spec_for_role, q6k_vocab_scalar_reduce_eligible)
from tinygrad.llm.flash_decode_attention import (FLASH_DECODE_G4, FLASH_DECODE_G5, FlashDecodeCapability, FlashDecodeRouteConfig,
  flash_decode_capability_from_renderer, flash_decode_coarse_split_override, flash_decode_live_split_block_tile,
  flash_decode_target_promoted, flash_fused_gmax_combine_kernel, flash_vec_llama_score_pv_kernel)
from tinygrad.llm.kernel_program import (ActivationViewRequest, DeclaredTypedOutput, KernelProgram,
                                         KernelProgramProvenance, OutputSpec, ResidualViewRequest, TypedLayout,
                                         TypedViewRequest, execute_promoted_program, execute_research_program)
from tinygrad.llm.kernel_program import execute_research_program_outputs
from tinygrad.llm.model_route_plan import (decode_epilogue_fusion_promoted, decode_flash_combine_fusion_promoted,
                                           decode_flash_llama_vec_wide_promoted)
from tinygrad.llm.packed_argmax import packed_argmax_from_tile_keys
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
class Q4KFFNDownQuadAdmission:
  """Research-only lease marker for the quad-u128-smem Q4 FFN-down load style.

  A census harness attaches one to an exact 4096x12288 ``ffn_down`` linear; the quad
  spelling is then used in place of the installed single-projection kernel for that block's
  ffn_down_resadd GEMV (new ``q4k_g3_lanemap_gemv_quad_epi_ffnresadd_*`` name, legacy
  hashes untouched). Normal model loads never carry this marker, so production keeps
  its target-selected ordinary load style.
  """
  block_index: int
  def __post_init__(self):
    if not isinstance(self.block_index, int) or isinstance(self.block_index, bool) or self.block_index < 0:
      raise ValueError("Q4_K FFN-down quad block index must be a non-negative integer")


def _q4k_single_projection_load_style(linear:Any, getenv_fn=getenv) -> str:
  capability = getattr(getattr(linear, "route_admission", None), "capability", None)
  nv_sm120 = (capability is not None and getattr(capability, "backend", None) == "NV"
              and getattr(capability, "architecture", None) == "sm_120")
  return "vector" if (nv_sm120 and not getenv_fn("TINYGRAD_Q4K_SCALAR_LOAD", 0)) else "scalar"


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
    # Closed-default Q4_K attention-K four-warp exact route. Normal model
    # loads carry no admission object, so this returns None before constructing
    # any graph. A research harness installs the admission on exact Q4_K
    # 1024x4096 attn_kv linears; Q4 V keeps its own shared-Q8 route.
    if (q4k_k_admission := getattr(linear, "_q4k_k_four_warp_admission", None)) is not None:
      from tinygrad.llm.q4k_k_four_warp import q4k_k_four_warp_call
      if (k_four := q4k_k_four_warp_call(q4k_k_admission, linear, x, binding)) is not None:
        return k_four
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

    # M2b ffn_down residual add (nv-epilogue-absorption-route-scope-20260810.md): the ffn_down
    # Q4K GEMV absorbs the standalone h+ffn_out add in-kernel (total + h[row], fp32 store) when
    # the block threads normed_h=h under the harness-installed _ffn_down_resadd_lease. The lease
    # is re-checked HERE on the linear (fail-closed); the loader record's promoted flag
    # (_decode_ffn_down_resadd_promoted, decode-ffn-down-resadd-route-policy.json) admits the same
    # spelling. "gate_out" absent keeps the M4 fused-prelude path closed and distinct. The
    # in-kernel fp32 add is bitwise-identical to the separate add.
    m2b_resadd = (route_role == "ffn_down" and "normed_h" in epi_inputs and "gate_out" not in epi_inputs
                  and (bool(getattr(linear, "_ffn_down_resadd_lease", False))
                       or bool(getattr(linear, "_decode_ffn_down_resadd_promoted", False))))
    if m2b_resadd:
      from tinygrad.llm.decode_kernels import Q4KGEMVEpilogue
      epi_spec = Q4KGEMVEpilogue("ffn_down_resadd")
      _xv = x[:, 0, :].reshape(binding.K).cast(dtypes.float16).contiguous()
      prog_inputs.append(_xv)
      prog_inputs.append(epi_inputs["normed_h"][:, 0, :].reshape(binding.N).cast(dtypes.float32))
    elif (q4k_epi_admitted or q4k_resadd_admitted) and route_role:

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
      if route_role == "attn_qo" else
      (TypedViewRequest(slot=1, dtype=dtypes.float16, flat_shape=(binding.K,), route_role="ffn_down",
                        requires_combine_fusion=False, requires_epilogue_absorption=True),)
      if route_role == "ffn_down" else ())
    # M4 residual_add typed input (m4-resadd-landing-scope section 2.2): the residual slot folds to
    # a zero-copy view of the ordinary block-output producer under the extended validator
    # (kernel_program._validated_residual_view). Fail-closed: any mismatch keeps the boundary copy.
    residual_input_views = (
      (ResidualViewRequest(slot=2, dtype=dtypes.float32, flat_shape=(binding.N,), route_role="attn_qo",
                           kind="residual_add"),)
      if (route_role == "attn_qo" and epi_spec is not None and epi_spec.kind == "residual_add")
      else (ResidualViewRequest(slot=2, dtype=dtypes.float32, flat_shape=(binding.N,), route_role="ffn_down",
                                kind="residual_add"),)
      if (epi_spec is not None and epi_spec.kind == "ffn_down_resadd") else ())
    # M2b block-output typed boundary (nv-epilogue-absorption-route-scope-20260810.md): when the
    # ffn_down GEMV absorbs the h+ffn_out add in-kernel, its fp32 AFTER IS the concrete block
    # output, so it declares the typed layout exactly like the w1w3fused16 producer. The next
    # block's attention residual fold (_residual_producer_identity) then accepts the AFTER
    # directly; without the declaration the generic flat-buffer ABI boundary copy renders after
    # every block. The M4 attn_qo residual_add GEMV declares the same boundary: its fp32 AFTER is
    # the concrete block intermediate h that the M2b ffn_down residual slot consumes, so the
    # ffn_down residual fold must prove the declared layout before binding it zero-copy.
    # Fail-closed: no declaration on any other epilogue/route spelling.
    typed_output = (DeclaredTypedOutput(TypedLayout(dtypes.float32, (binding.N,), (1, 1, binding.N)),
                                        combine_fusion_admitted=False,
                                        epilogue_absorption_admitted=True)
                    if epi_spec is not None and epi_spec.kind in ("ffn_down_resadd", "residual_add") else None)
    # Research-only Q4 FFN-down quad-u128-smem load style (nv-q4-down-quad-re-census-20260813.md):
    # a harness leases exact ffn_down linears with _q4k_ffn_down_quad_admission; the quad spelling is
    # used ONLY when that marker AND the m2b_resadd epilogue are both present. Normal model loads have
    # no marker, so production keeps its ordinary single-projection load style.
    q4k_quad_admission = getattr(linear, "_q4k_ffn_down_quad_admission", None)
    if q4k_quad_admission is not None and epi_spec is not None and epi_spec.kind == "ffn_down_resadd" and route_role == "ffn_down":
      q4k_load_style = "quad"
    elif epi_spec is not None and epi_spec.kind == "ffn_down_fused":
      # The fused prelude reads gate_out/up_out activations (not x), so the vectorized x-load
      # spelling does not apply; keep the installed scalar inner loop for that closed path.
      q4k_load_style = "scalar"
    else:
      # Vectorized global loads (uint4 header, deduplicated qpack, half4 activations) are
      # bit-identical to the scalar spelling. Cold counters show unchanged DRAM bytes but a
      # higher achieved rate, and two production reverse brackets plus a depth-128 bracket pass
      # once the vector residual-output transport is elided. Keep a scalar rollback/control arm.
      q4k_load_style = _q4k_single_projection_load_style(linear)
    program = KernelProgram(binding.route_id, f"{binding.candidate_id}.gemv",
      KernelProgramProvenance.MACHINE_SEARCH_GENERATED,
      q4k_g3_lanemap_gemv_kernel(binding.N, binding.K, epilogue=epi_spec, load_style=q4k_load_style),
      output_spec=OutputSpec((binding.N,), out_dtype, typed_output=typed_output),
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
  if binding is None:
    # M2b fail-closed fallback: the model skips its own h+ffn_out add when it threads normed_h=h
    # under the lease or the promoted record, so a binding miss must reproduce the add to keep the
    # graph unchanged.
    if (epilogue_inputs or {}) and "normed_h" in (epilogue_inputs or {}) and (
        getattr(linear, "_ffn_down_resadd_lease", False) or getattr(linear, "_decode_ffn_down_resadd_promoted", False)):
      return fallback(x) + (epilogue_inputs or {})["normed_h"]
    return fallback(x)
  return Q4K_DECODE_CANDIDATE.execute(linear, x, binding, epilogue_inputs=epilogue_inputs or {})

def q4k_gate_up_primitive_linear_call(gate:Any, up:Any, x:Tensor, fallback:Callable[[], Tensor],
                                      *, store_fp16: bool = False) -> Tensor:
  """Fused w1+w3 decode GEMV: ONE kernel computes z = silu(gate(x)) * up(x) from two Q4_K weight buffers
  (q4k-w1w3-fused-qv-implementation-record-20260803.md). Admitted only when BOTH linears bind the legacy
  decode GEMV shape AND both carry `w1w3_fusion_admitted` (their own QKPrimitiveRouteAdmission field,
  resolved from the closed-default decode-q4k-w1w3-fusion-route-policy.json record). Any mismatch (one
  linear not Q4K, bias, K/N inequality, multi-token, off-target, shape outside the quad geometry) falls
  back to the caller's legacy graph -- the fused route never changes what the legacy chain computes.
  `store_fp16` is a research-only spelling (no loader policy creates it): the fused kernel stores the
  result already cast to fp16 under its own `q4k_g3_lanemap_gemv_w1w3fused16_*` name and the graph's
  fp32->fp16 output cast folds away. Callers must hold an explicit harness-installed lease; the
  legacy fp32 kernel name and output dtype are unchanged when `store_fp16=False`."""
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
  out_dtype = dtypes.float16 if store_fp16 else dtypes.float32
  typed_output = (DeclaredTypedOutput(TypedLayout(out_dtype, (g_bind.N,), (1, 1, g_bind.N)),
                                      combine_fusion_admitted=False,
                                      epilogue_absorption_admitted=True)
                  if store_fp16 else None)
  # Vectorized-load spelling (q4k_g3_lanemap_gemv_w1w3vec16_*): the fused w1+w3 GEMV is NV sm_120
  # only, and the vectorized spelling is bit-identical to the scalar spelling (same per-lane
  # accumulation order and shifts; only the global load widths change from scalar LDG to uint4/half4).
  # TINYGRAD_Q4K_W1W3_SCALAR_LOAD=1 restores the scalar spelling for reverse-bracket control arms.
  load_style = "scalar" if getenv("TINYGRAD_Q4K_W1W3_SCALAR_LOAD", 0) else "vector"
  program = KernelProgram(g_bind.route_id, f"{g_bind.candidate_id}.w1w3_fused",
    KernelProgramProvenance.MACHINE_SEARCH_GENERATED,
    q4k_g3_lanemap_gemv_w1w3_kernel(g_bind.N, g_bind.K, load_style=load_style, store_fp16=store_fp16),
    output_spec=OutputSpec((g_bind.N,), out_dtype, typed_output=typed_output))
  return execute_promoted_program(None, gw, uw, xv, program=program).reshape(1, 1, g_bind.N)

def q4k_gate_up_rms_affine_qualification_call(gate:Any, up:Any, raw_x:Tensor, norm_weight:Tensor, eps:float,
                                               fallback:Callable[[], Tensor], *, store_fp16: bool = False) -> Tensor:
  """One-block, default-off RMS scale + Q4 gate/up qualification boundary.

  There is deliberately no policy bit here: callers must hold an explicit
  harness-installed lease.  Every shape/type miss is an ordinary fallback.

  The fused kernel reads the RAW fp32 hidden state (no fp16 pre-round), computes
  the bitwise-exact control scale
  ``(h.float().square().mean(-1, keepdim=True)+eps).rsqrt()`` on it (the same
  expression the ordinary ffn-norm reduce renders as ``r_16_256``), and applies
  the fp16 norm weight with the control epilogue's single fp16 round at each
  packed-Q4 load.  ``store_fp16=True`` mirrors the landed fused16 spelling: the
  fused z is stored already cast to fp16 under the ``*_rms_affine16_*`` name so
  the graph's fp32->fp16 ffn-activation cast folds away and the ffn_down
  consumer sees the same fp16 ABI as the control graph."""
  g_bind=Q4K_DECODE_CANDIDATE.bind(gate,raw_x,getattr(getattr(gate,"route_admission",None),"admitted",False))
  u_bind=Q4K_DECODE_CANDIDATE.bind(up,raw_x,getattr(getattr(up,"route_admission",None),"admitted",False))
  if g_bind is None or u_bind is None or (g_bind.T,g_bind.K,g_bind.N)!=(1,4096,12288) or (u_bind.T,u_bind.K,u_bind.N)!=(1,4096,12288): return fallback()
  if norm_weight.shape != (4096,) or norm_weight.dtype != dtypes.float16: return fallback()
  from tinygrad.llm.decode_kernels import q4k_g3_lanemap_gemv_w1w3_rms_affine_kernel
  from tinygrad.llm.kernel_program import execute_research_program
  gw=gate.q4k_storage.words.to(raw_x.device).contiguous() if gate.q4k_storage.mode == "q4_ondemand" else gate.q4k_storage.words.to(raw_x.device)
  uw=up.q4k_storage.words.to(raw_x.device).contiguous() if up.q4k_storage.mode == "q4_ondemand" else up.q4k_storage.words.to(raw_x.device)
  xv=raw_x[:,0,:].reshape(4096).contiguous()
  # Bitwise-exact control scale (r_16_256 contract): the ordinary ffn-norm reduce
  # computes rsqrt(mean(h^2)+eps) on the RAW fp32 h.  Rounding h to fp16 first (the
  # old spelling) or using a `.sum()` association changes 2/4096 scale bits and
  # fails the exact-logits gate.
  scale=(raw_x.float().square().mean(-1,keepdim=True)+eps).rsqrt().reshape(1)
  out_dtype=dtypes.float16 if store_fp16 else dtypes.float32
  typed_output=(DeclaredTypedOutput(TypedLayout(out_dtype,(12288,),(1,1,12288)),
                                    combine_fusion_admitted=False, epilogue_absorption_admitted=True)
                if store_fp16 else None)
  # M1 raw-x typed-input opt-in: the raw fp32 h slot (slot 3) binds zero-copy to the
  # block-output epi_resadd AFTER under the fail-closed validator
  # (kernel_program._validated_activation_view).  Without the fold the generic flat-buffer
  # ABI materializes an identity transport copy per block (E_32_32_4_86a23e1a), exactly
  # cancelling the M1 -36 program fold.  The request is issued only here; any mismatch
  # keeps the flat-buffer ABI (byte-identical control).
  # Input tuple is (gate_words, up_words, raw_x, norm_weight, scale): raw_x is slot 2.
  activation_input_views=(ActivationViewRequest(slot=2,dtype=dtypes.float32,flat_shape=(4096,),route_role="ffn_norm"),)
  program=KernelProgram(g_bind.route_id,f"{g_bind.candidate_id}.rms_affine_qualification",KernelProgramProvenance.RESEARCH_ONLY,
    q4k_g3_lanemap_gemv_w1w3_rms_affine_kernel(12288,4096,store_fp16=store_fp16),
    output_spec=OutputSpec((12288,),out_dtype,typed_output=typed_output),
    activation_input_views=activation_input_views)
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

  def execute(self, linear:Any, x:Tensor, binding:_LinearDecodeBinding,
              epilogue_inputs:dict[str, Tensor]|None=None) -> Tensor:
    x_vec = x[:, 0, :].reshape(binding.K).cast(dtypes.float16).contiguous()
    route_role = getattr(linear, "route_role", "")
    epi_inputs = epilogue_inputs or {}
    # Closed-default Q6_K FFN-down four-warp fp16 geometry route. Normal model loads carry no
    # admission object, so this returns None before constructing the installed coop graph. A
    # promoted policy installs the admission on exact Q6_K 4096x12288 ffn_down linears.
    if (q6k_mmvq_admission := getattr(linear, "_q6k_ffn_down_mmvq_admission", None)) is not None:
      from tinygrad.llm.q6k_ffn_down_mmvq import q6k_ffn_down_mmvq_call
      if (mmvq := q6k_ffn_down_mmvq_call(q6k_mmvq_admission, linear, x, binding, epi_inputs)) is not None:
        return mmvq
    # Closed-default Q6_K attention-V four-warp fp16 direct route. Normal model
    # loads carry no admission object, so this returns None before constructing
    # the installed parts graph. A research harness installs the admission on
    # exact Q6_K 1024x4096 attn_kv linears.
    if (q6k_v_admission := getattr(linear, "_q6k_v_four_warp_admission", None)) is not None:
      from tinygrad.llm.q6k_v_mmvq import q6k_v_four_warp_call
      if (v_mmvq := q6k_v_four_warp_call(q6k_v_admission, linear, x, binding)) is not None:
        return v_mmvq
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
    # M2b ffn_down residual add: the shared-block coop GEMV absorbs h+ffn_out in-kernel under the
    # harness-installed _ffn_down_resadd_lease or the promoted record (fail-closed: checked on the
    # linear; "gate_out" absent keeps the M4 fused-prelude path distinct). The in-kernel fp32 add
    # is bitwise-identical to the standalone E_32_32_4_02a9738c add, so the separate add folds
    # away. When the coop reduction is external_sum the epilogue cannot fire in-kernel; the route
    # reproduces the ordinary add over the reduced partials so the block graph stays unchanged.
    m2b_resadd = (route_role == "ffn_down" and "normed_h" in epi_inputs and "gate_out" not in epi_inputs
                  and (bool(getattr(linear, "_ffn_down_resadd_lease", False))
                       or bool(getattr(linear, "_decode_ffn_down_resadd_promoted", False))))
    epilogue = "ffn_down_resadd" if (m2b_resadd and reduction == "in_kernel") else ""
    spec = q6k_spec_for_role(binding.N, binding.K, parts=binding.parts, row_tile=binding.row_tile,
                            use_coop=binding.use_coop, opts=linear.opts, target=target, reduction=reduction,
                            epilogue=epilogue)
    # M2 epilogue-absorption boundary (nv-epilogue-absorption-route-scope-20260810.md): the shared-block
    # ffn_down Q6K GEMV consumes the fused w1+w3 fp16 store directly when the producer declared its
    # fp16 layout under the harness-installed lease. Same fail-closed validator as Q4K: no declaration,
    # no fold, generic flat-buffer ABI unchanged.
    typed_input_views = (
      (TypedViewRequest(slot=1, dtype=dtypes.float16, flat_shape=(binding.K,), route_role="ffn_down",
                        requires_combine_fusion=False, requires_epilogue_absorption=True),)
      if route_role == "ffn_down" else ())
    # M2b block-output typed boundary (same contract as the Q4K ffn_down branch above): the
    # absorbing coop GEMV's fp32 AFTER is the concrete block output; declaring the typed layout
    # lets the next block's attention residual fold bind it in place. Fail-closed: the
    # declaration exists only for the ffn_down_resadd epilogue spelling (in_kernel reduction).
    typed_output = (DeclaredTypedOutput(TypedLayout(dtypes.float32, (binding.N,), (1, 1, binding.N)),
                                        combine_fusion_admitted=False,
                                        epilogue_absorption_admitted=True)
                    if epilogue == "ffn_down_resadd" else None)
    gemv_program = KernelProgram(binding.route_id, f"{binding.candidate_id}.gemv",
      KernelProgramProvenance.MACHINE_SEARCH_GENERATED, emit_q6k_gemv_kernel(spec),
      output_spec=OutputSpec((binding.N,) if reduction == "in_kernel" else (binding.N, spec.partial_axis_extent),
                             dtypes.float32, typed_output=typed_output),
      typed_input_views=typed_input_views,
      residual_input_views=((ResidualViewRequest(slot=2, dtype=dtypes.float32, flat_shape=(binding.N,),
                                                 route_role="ffn_down", kind="residual_add"),)
                           if epilogue == "ffn_down_resadd" else ()))
    if epilogue:
      h_vec = epi_inputs["normed_h"][:, 0, :].reshape(binding.N).cast(dtypes.float32)
      return execute_promoted_program(None, linear.q6k_storage.halfs.to(x.device), x_vec, h_vec,
                                      program=gemv_program).reshape(1, 1, binding.N)
    partial = execute_promoted_program(None, linear.q6k_storage.halfs.to(x.device), x_vec, program=gemv_program)
    if reduction == "in_kernel":
      return partial.reshape(1, 1, binding.N)
    if m2b_resadd:
      # external_sum fallback: reproduce the ordinary h+ffn_out fp32 add over the reduced partials.
      return (partial.sum(axis=1) + epi_inputs["normed_h"][:, 0, :].reshape(binding.N)).reshape(1, 1, binding.N)
    if q6k_vocab_scalar_reduce_eligible(spec):
      reduce_program = KernelProgram(binding.route_id, f"{binding.candidate_id}.vocab_reduce",
        KernelProgramProvenance.MACHINE_SEARCH_GENERATED, emit_q6k_vocab_scalar_reduce_kernel(spec),
        output_spec=OutputSpec((binding.N,), dtypes.float32))
      return execute_promoted_program(None, partial, program=reduce_program).reshape(1, 1, binding.N)
    return partial.sum(axis=1).reshape(1, 1, binding.N)

Q6K_DECODE_CANDIDATE = _Q6KDecodeCandidate()


def q6k_vocab_top1_call(linear:Any, x:Tensor, arch_ok:bool) -> Tensor | None:
  """Research-only fused vocab top-1 route.

  The production Q6K vocab GEMV already emits the coop in-kernel reduction.  This route swaps
  that epilogue for the P1 ``vocab_top1`` spelling (one packed u64 (max,index) key per warp
  tile) and follows it with the tiny cross-tile reduce, so the four sampler tail kernels
  (``E_1187_32_4`` / ``r_32_4_1187`` / ``r_128_16_8_1187`` / ``r_16_8``) disappear.  It is
  closed-default and RESEARCH_ONLY: it is only reachable from the model's forward_greedy path
  when the caller installs ``_decode_vocab_top1_lease``.
  """
  binding = Q6K_DECODE_CANDIDATE.bind(linear, x, arch_ok)
  if binding is None or getattr(linear, "route_role", "") != "lm_head": return None
  if not (binding.use_coop and binding.row_tile * Q6K_POS_EXTENT <= 32): return None
  capability = getattr(getattr(linear, "route_admission", None), "capability", None)
  target = f"{capability.backend}:{capability.architecture}" if capability is not None and \
    getattr(capability, "backend", None) is not None and getattr(capability, "architecture", None) is not None \
    else Q6K_DECODE_CANDIDATE.target
  x_vec = x[:, 0, :].reshape(binding.K).cast(dtypes.float16).contiguous()
  spec = q6k_spec_for_role(binding.N, binding.K, parts=binding.parts, row_tile=binding.row_tile,
                          use_coop=binding.use_coop, opts=linear.opts, target=target,
                          reduction="in_kernel", epilogue="vocab_top1")
  tiles = binding.N // binding.row_tile
  gemv_program = KernelProgram("decode_q6k_vocab_top1_research", f"{binding.candidate_id}.vocab_top1_gemv",
    KernelProgramProvenance.RESEARCH_ONLY, emit_q6k_gemv_kernel(spec),
    output_spec=OutputSpec((tiles,), dtypes.uint64))
  keys = execute_research_program(Tensor.empty((tiles,), dtype=dtypes.uint64, device=x.device),
                                  linear.q6k_storage.halfs.to(x.device), x_vec, program=gemv_program)
  # The cross-tile reduce is an ordinary UOp graph (one u64 MAX + unpack) so the
  # scheduler lowers it with a hierarchical parallel reduction.  The custom
  # emit_q6k_vocab_top1_reduce_kernel path was a single-thread serial loop over
  # rows/row_tile keys and cost ~0.89ms/token in the fused A/B.  The returned
  # view is cloned so the JIT memory plan cannot replay the winner one token
  # behind the eager stream (same firewall the custom reduce route used).
  # The vocab GEMV epilogue streams ~510 MB of weights through L2 and evicts its own
  # 607 KB packed-key write; the single-block 16-thread u64 reduce then reads keys
  # L2-cold and pays DRAM latency (~85 us vs 44 us L2-warm, measured 2026-08-17:
  # nv-vocab-reduce-l2-mechanism-20260817.json).  A tiny copy of the keys re-warms L2
  # before the reduce (the role the legacy E_1187_32_4 copy plays), flipping the fused
  # tail from +25.8 us slower to ~-11 us faster than the legacy chain.
  keys = keys.clone()
  return packed_argmax_from_tile_keys(keys, binding.N, axis=0, keepdim=True).reshape(1, 1).clone()

def q6k_primitive_linear_call(linear:Any, x:Tensor, fallback:Callable[[Tensor], Tensor], arch_ok:bool,
                              epilogue_inputs:dict[str, Tensor]|None=None) -> Tensor:
  # Q6_K decode GEMV (1 token) or batched verify/prefill GEMM (K tokens).
  binding = Q6K_DECODE_CANDIDATE.bind(linear, x, arch_ok)
  if binding is None:
    # M2b fail-closed fallback: reproduce the ordinary h+ffn_out add (see the Q4K route).
    if (epilogue_inputs or {}) and "normed_h" in (epilogue_inputs or {}) and (
        getattr(linear, "_ffn_down_resadd_lease", False) or getattr(linear, "_decode_ffn_down_resadd_promoted", False)):
      return fallback(x) + (epilogue_inputs or {})["normed_h"]
    return fallback(x)
  return Q6K_DECODE_CANDIDATE.execute(linear, x, binding, epilogue_inputs=epilogue_inputs or {})

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
  llama_vec_wide: bool = False

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
    # The binding's target label is derived from the resolved (backend, arch) tuple, never a hardcoded
    # vendor literal: the renderer's target device is "AMD" for HIP and "CUDA" for NV, so AMD keeps its
    # historical label while an NV backend gets its own. `or "AMD"` only guards an unlabeled device string.
    backend_label = target[0] or "AMD"
    return _FlashDecodeBinding(self.candidate_id, self.route_id, backend_label, B, Hq, Hkv, Hd,
                               self.split_size, self.query_group_size, self.staging, self.stage_width,
                               admission.combine_fusion_admitted, decode_flash_llama_vec_wide_promoted(target))

# Public compatibility aliases. Their sole route authority is owned beside the
# flash executor, so selection cannot drift from execution configuration.
FLASH_DECODE_CANDIDATE = _FlashDecodeCandidate(FLASH_DECODE_G4)
FLASH_DECODE_G5_CANDIDATE = _FlashDecodeCandidate(FLASH_DECODE_G5)

def _flash_combine_q8_outputs_emitter(base):
  """Adapt base ``(fp16_out, partial, q8_out)`` to outputs-first program ABI."""
  def outputs_first(fp16_out, q8_out, partial): return base(fp16_out, partial, q8_out)
  return outputs_first

def _flash_llama_vec_wide_research_call(q:Tensor, assigned_kv:Tensor, Tc:UOp, binding:_FlashDecodeBinding,
                                         MAXC:int, S:int, output_fp16:bool, *, promoted:bool=False,
                                         token_bound:int|None=None, wide_q_f32:bool=False,
                                         combine_register_weights:bool=False, query_group_size:int=1,
                                         successor_weights:Tensor|None=None, successor_prefetch_groups:int=0,
                                         output_q8:bool=False, output_q8_fine:bool=False) -> Tensor|tuple[Tensor,Tensor]:
  """Extent-derived wide-KV flash at the approved research/promotion admission boundary."""
  extent_split = MAXC // 128 if MAXC % 128 == 0 else None
  bounded = token_bound is not None and token_bound % 128 == 0 and token_bound <= MAXC and \
    S == (token_bound // 128) * query_group_size
  if (binding.Hq, binding.Hkv, binding.Hd) != (32, 8, 128) or (S != extent_split and not bounded) or \
      assigned_kv.dtype != dtypes.float16:
    raise ValueError(f"llama_vec_wide research route requires Hq32/Hkv8/Hd128, fp16 KV, and either S=MAXC/128 "
                     f"or S=token_bound/128, got Hq={binding.Hq}, Hkv={binding.Hkv}, Hd={binding.Hd}, S={S}, "
                     f"MAXC={MAXC}, token_bound={token_bound}, cache={assigned_kv.dtype}")
  cache_bits = Tensor(assigned_kv.uop.bitcast(dtypes.uint32))
  q_arg = Tensor(q.uop.bitcast(dtypes.uint32)) if wide_q_f32 else q
  provenance = KernelProgramProvenance.MACHINE_SEARCH_GENERATED if promoted else KernelProgramProvenance.RESEARCH_ONLY
  execute = execute_promoted_program if promoted else execute_research_program
  tile_program = KernelProgram(binding.route_id, f"{binding.candidate_id}.llama_vec_wide.tile",
    provenance,
    flash_vec_llama_score_pv_kernel(binding.Hd, binding.Hq, binding.Hkv, MAXC, S, Tc, wide_kv=True, wide_q=False,
                                    wide_q_f32=wide_q_f32, token_bound=token_bound,
                                    query_group_size=query_group_size,
                                    v_pipeline_tail=getenv("NV_FLASH_V_PIPELINE_TAIL", 1)),
    output_spec=OutputSpec((binding.Hq * S * (binding.Hd + 2),), dtypes.float32))
  partial = execute(None, q_arg.reshape(binding.Hq * binding.Hd), cache_bits, program=tile_program)
  combine_emitter=flash_fused_gmax_combine_kernel(binding.Hd, binding.Hq, S, output_fp16=output_fp16, lane_width=128,
                                    register_weights=combine_register_weights,
                                    successor_prefetch_groups=successor_prefetch_groups,output_q8=output_q8,
                                    output_q8_fine=output_q8_fine)
  if output_q8 or output_q8_fine: combine_emitter=_flash_combine_q8_outputs_emitter(combine_emitter)
  combine_program = KernelProgram(binding.route_id, f"{binding.candidate_id}.llama_vec_wide.combine",
    provenance,
    combine_emitter,
    output_spec=OutputSpec((binding.Hq * binding.Hd,), dtypes.float16 if output_fp16 else dtypes.float32))
  if output_q8 or output_q8_fine:
    out=Tensor.empty((binding.Hq*binding.Hd,),dtype=dtypes.float16,device=q.device)
    q8=Tensor.empty((1280 if output_q8_fine else 1152,),dtype=dtypes.uint32,device=q.device)
    results=execute_research_program_outputs(out,q8,partial,program=combine_program)
    return results[0].reshape(binding.Hq,binding.Hd),results[1]
  combine_args=(partial,) if successor_weights is None else (partial,successor_weights)
  return execute(None, *combine_args, program=combine_program).reshape(binding.Hq, binding.Hd)


def _flash_llama_vec_wide_installed_admitted(promoted:bool, geom:dict, max_context:int) -> bool:
  # The installed route is qualified only at the two physical cache extents used by the dense
  # decode endpoint. Explicit geometry always wins, and larger request capacities must not inherit
  # an unmeasured S=MAXC/128 launch merely because the arithmetic divides evenly.
  return promoted and not geom and max_context in (768, 1024)

def _flash_combine_register_weights_admitted(wide_promoted:bool, geom:dict, getenv_fn=getenv) -> bool:
  """Install register-broadcast weights only with the qualified wide route.

  Explicit research geometry remains authoritative.  The environment switch
  is a load-time rollback used by the installed reverse bracket.
  """
  if getenv_fn("TINYGRAD_FLASH_COMBINE_REGISTER_DISABLE", 0): return False
  return bool(geom.get("combine_register_weights", False)) or wide_promoted


def flash_decode_attention_route(q:Tensor, assigned_kv:Tensor, start_pos:int|UOp, T:int|UOp, B:int,
                                 Hq:int, Hkv:int, Hd:int, max_context:int, kv_scale:Tensor|None=None,
                                 freqs:Tensor|None=None, ring_full:bool=False,
                                 combine_fp16:bool|None=None, tile_geometry:dict|None=None,
                                 successor_weights:Tensor|None=None) -> Tensor:
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
  # M2d combine-fp16 lease override (nv-epilogue-absorption-route-scope-20260810.md): the policy
  # record stays closed; a harness-installed `_flash_combine_fp16_lease` on the block forces the
  # fp16 combine variant (flash_fused_gmax_combine_f16_*) so its in-kernel RNE store absorbs the
  # E_32_32_4_0a5eb0ac attention cast. Without the lease, `binding.combine_fusion` is the policy
  # answer (False) and the legacy fp32 combine renders byte-identical.
  geom = dict(tile_geometry or {})
  # Env-gated coarse-split research override (FLASH_DECODE_COARSE_SPLIT): run the production G4
  # decode route with the env-selected split count instead of the promoted S=48. Unset env keeps
  # binding.split_size, so production behavior is byte-identical to today; the admission guard in
  # flash_decode_live_split_block_tile accepts the env value only when it is set.
  split_size = geom.get("split_count", binding.split_size)
  if (coarse_split := flash_decode_coarse_split_override()) and binding.Hq == FLASH_DECODE_G4.query_heads:
    # A graph-local typed geometry lease takes precedence.  The env override
    # remains the admission authority for non-promoted split counts.
    if "split_count" not in geom: split_size = coarse_split
  output_fp16 = bool(binding.combine_fusion or combine_fp16)
  wide_lease = bool(geom.get("llama_vec_wide", False))
  wide_promoted = _flash_llama_vec_wide_installed_admitted(binding.llama_vec_wide, geom, MAXC)
  if wide_promoted: split_size = MAXC // 128
  if wide_lease or wide_promoted:
    if kv_scale is not None or freqs is not None:
      raise ValueError("llama_vec_wide research route requires plain, pre-rotated fp16 KV")
    successor_prefetch_groups=int(geom.get("o_successor_prefetch_groups",0))
    output_q8=bool(geom.get("o_q8_owned",False))
    output_q8_fine=bool(geom.get("o_q8_fine_owned",False))
    if output_q8 and output_q8_fine: raise ValueError("only one combine-owned Q8 representation may be selected")
    if (output_q8 or output_q8_fine) and successor_prefetch_groups: raise ValueError("O prefetch and combine-owned Q8 are exclusive")
    if (successor_weights is not None) != bool(successor_prefetch_groups):
      raise ValueError("O successor weights and prefetch group lease must be supplied together")
    return _flash_llama_vec_wide_research_call(q, assigned_kv, _tc, binding, MAXC, split_size, output_fp16,
                                                promoted=wide_promoted and not successor_prefetch_groups and not output_q8 and not output_q8_fine, token_bound=geom.get("token_bound"),
                                                wide_q_f32=bool(geom.get("wide_q_f32", False)),
                                                combine_register_weights=_flash_combine_register_weights_admitted(
                                                  wide_promoted, geom),
                                                query_group_size=int(geom.get("query_group_size", 1)),
                                                successor_weights=successor_weights,
                                                successor_prefetch_groups=successor_prefetch_groups,output_q8=output_q8,
                                                output_q8_fine=output_q8_fine)
  return flash_decode_live_split_block_tile(q.reshape(binding.Hq, binding.Hd), assigned_kv, _tc,
    binding.Hd, binding.Hq, binding.Hkv, MAXC, split_size, staging=binding.staging,
    fused_combine=True, kv_scale=kv_scale, freqs=freqs, query_group_size=binding.query_group_size,
    stage_width=geom.get("stage_width", binding.stage_width),
    token_block=geom.get("token_block", 16), lane_width=geom.get("lane_width", 32),
    score_group_width=geom.get("score_group_width"), warps=geom.get("warps"),
    reduce_structure=geom.get("reduce_structure", "staged"), dot_pair_width=geom.get("dot_pair_width", 2),
    combine_lane_width=geom.get("combine_lane_width"),
    combine_fp16=output_fp16, split_count_leased="split_count" in geom)
