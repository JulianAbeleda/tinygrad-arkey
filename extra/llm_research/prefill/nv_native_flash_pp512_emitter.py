"""Compiler-owned first Flash prefill emitter (the F1 T512/start0 fixture).

The emitter is intentionally narrow.  It delegates UOp construction to the
existing scheduler-owned attention emitter and never reads CUDA source/cubins.
"""
from __future__ import annotations

import hashlib
import json
import os

from tinygrad import dtypes
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, OutputSpec
from tinygrad.uop.ops import KernelInfo
from tinygrad.schedule.wmma.flash_prefill import FlashPrefillAttentionSpec
from .nv_native_flash_pp512_spec import FlashEmitterRequest, NativeFlashPP512Spec

FIXTURE = NativeFlashPP512Spec()


def _scheduler_spec(spec: NativeFlashPP512Spec) -> FlashPrefillAttentionSpec:
  spec.validate()
  if (spec.tokens, spec.start_pos) != (512, 0):
    raise ValueError("F2 emitter only admits the T512/start_pos=0 fixture")
  return FlashPrefillAttentionSpec(Hq=spec.query_heads, Hkv=spec.kv_heads,
    q_tokens=spec.tokens, kv_tokens=spec.live_tokens, causal=spec.causal,
    # Grid-loop lowering requires the launch scalar to be representable as fp16.
    # This is the exact fp16 rounding of 1/sqrt(128), not a semantic change.
    scale=0.08837890625 if spec.head_dim == 128 else 1.0 / (spec.head_dim ** 0.5), Hd=spec.head_dim,
    valid_kv=spec.live_tokens, query_start=spec.start_pos, target="nv_sm120", warps_per_cta=4)


def build_program(spec: NativeFlashPP512Spec = FIXTURE) -> KernelProgram:
  scheduler = _scheduler_spec(spec)
  raw_emit = scheduler.emit()
  use_nv2c = os.getenv("NV2C_RESEARCH", "0") == "1"
  if use_nv2c:
    from tinygrad.schedule.wmma.kernels import nv_sm120_q16_grid_hd128_cooperative_attention
    from tinygrad.codegen.opt.attention_fragment import attention_fragment_model
    nv_model = attention_fragment_model("nv_sm120")
    def raw_emit(out_ph, q_ph, k_ph, v_ph):
      return nv_sm120_q16_grid_hd128_cooperative_attention(q_ph, k_ph, v_ph, out_ph,
        q_tokens=spec.tokens, q_heads=spec.query_heads, kv_heads=spec.kv_heads,
        kv_tokens=spec.live_tokens, scale=0.08837890625, causal=spec.causal,
        valid_kv=spec.live_tokens, query_start=spec.start_pos, kernel_info=KernelInfo(name="nv2c"),
        warps_per_cta=4, head_dim=spec.head_dim, fragment_model=nv_model)
  def diagnostic_emit(out_ph, q_ph, k_ph, v_ph):
    vals=[]
    for name,u in (("out",out_ph),("q",q_ph),("k",k_ph),("v",v_ph)):
      vals.append({"name":name,"op":str(u.op),"dtype":str(u.dtype),"slot":getattr(u.arg,"slot",None),
                   "base":str(getattr(u,"ptrdtype",None).base) if hasattr(u,"ptrdtype") else None,
                   "size":getattr(getattr(u,"ptrdtype",None),"size",None)})
    print("F2_SLOT_CENSUS " + json.dumps(vals, sort_keys=True))
    root = raw_emit(out_ph,q_ph,k_ph,v_ph)
    seen=set(); specials=[]
    def walk(u):
      if id(u) in seen: return
      seen.add(id(u))
      if str(u.op) == "Ops.SPECIAL":
        specials.append({"identity": id(u), "name": str(u.arg), "dtype": str(u.dtype),
                         "extent": getattr(u.dtype, "size", None), "arg": repr(u.arg)})
      for s in u.src: walk(s)
    walk(root)
    lidx=[s for s in specials if s["name"] == "lidx0"]
    if len(lidx) != 1:
      raise RuntimeError("NV1 SPECIAL census requires exactly one lidx0: " + json.dumps(specials, sort_keys=True))
    return root
  return KernelProgram("nv_native_flash_pp512_nv1_w4_scratch1024_v2", "fixture_t512_start0_nv1_w4_scratch1024_v2",
    KernelProgramProvenance.TINYGRAD_SCHEDULER_GENERATED, diagnostic_emit,
    # The scheduler ABI requires out0 to be a flat PARAM, not a reshaped view.
    output_spec=OutputSpec((spec.batch * spec.query_heads * spec.tokens * spec.head_dim,), dtypes.float16))


def emit(request: FlashEmitterRequest) -> KernelProgram:
  request.validate()
  return build_program(request.spec)


def static_identity() -> str:
  """Stable identity for CI/gates without tensor allocation or compilation."""
  program = build_program()
  payload = {"spec": FIXTURE.to_dict(), "program": program.to_dict(),
             "emitted_kernel_names": ("nv_sm120_q16_grid_hd128_loop_attention",)}
  return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


__all__ = ["FIXTURE", "build_program", "emit", "static_identity"]
