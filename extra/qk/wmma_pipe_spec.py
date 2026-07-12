#!/usr/bin/env python3
"""Register-resident WMMA pipe primitive spec.

This module is intentionally only the extraction seam from the route-level
PrefillGEMMScheduleSpec to a compiler-owned pipe primitive contract. It must not
lower through extra.qk.prefill.wmma or carry route-local instruction lists.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from tinygrad.uop.ops import KernelCandidateContext, Ops, KernelInfo, UOp
from tinygrad.dtype import dtypes


@dataclass(frozen=True)
class WMMAPipeSpec:
  m: int
  n: int
  k: int
  tile_m: int
  tile_n: int
  k_step: int = 16
  stages: int = 2
  pipe_tm: int = 2
  pipe_tn: int = 2
  operand_a: str = "global_row_major_fp16"
  operand_b: str = "global_row_major_bt_fp16"
  wait_policy: str = "targeted_vmcnt"
  target: str = "amd_gfx1100"
  role: str = "unknown"

  def __post_init__(self):
    if any(not isinstance(x, int) or x <= 0 for x in (self.m, self.n, self.k, self.tile_m, self.tile_n, self.k_step, self.pipe_tm, self.pipe_tn)):
      raise ValueError("pipe dimensions and factors must be positive integers")
    if self.m % self.tile_m or self.n % self.tile_n or self.k % self.k_step:
      raise ValueError("pipe shape must be divisible by tile and k_step")
    # Unsupported lifecycle/wait/target values remain constructible so the
    # fail-closed lowerer can report the exact unsupported contract.

  @property
  def loads_per_stage(self) -> int:
    # build_gemm_pipe's targeted wait leaves the opposite stage's A/B b128 loads outstanding.
    return self.pipe_tm * 2 + self.pipe_tn * 2

  def to_json(self) -> dict[str, Any]:
    return {
      "m": self.m, "n": self.n, "k": self.k, "tile_m": self.tile_m, "tile_n": self.tile_n,
      "k_step": self.k_step, "stages": self.stages, "pipe_tm": self.pipe_tm, "pipe_tn": self.pipe_tn,
      "operand_a": self.operand_a, "operand_b": self.operand_b, "wait_policy": self.wait_policy,
      "target": self.target, "role": self.role, "loads_per_stage": self.loads_per_stage,
    }

@dataclass(frozen=True)
class WMMAPipeIR:
  role: str
  shape: tuple[int, int, int]
  stages: int
  loads_per_stage: int
  wait_policy: str
  stores: str = "fp16_global"
  provenance: str = "compiler_owned_typed_pipe_ir"

  def __post_init__(self):
    if len(self.shape) != 3 or any(not isinstance(x, int) or x <= 0 for x in self.shape): raise ValueError("invalid pipe IR shape")
    if self.stores != "fp16_global": raise ValueError("unsupported pipe output dtype/layout")
    if self.stages != 2 or self.loads_per_stage <= 0: raise ValueError("invalid pipe IR lifecycle")
    if self.wait_policy != "targeted_vmcnt": raise ValueError("unsupported pipe wait policy")
    if self.provenance != "compiler_owned_typed_pipe_ir": raise ValueError("invalid pipe provenance")

@dataclass(frozen=True)
class WMMAPipeOp:
  """Design-only typed graph op contract; renderer lowering is intentionally pending."""
  ir: WMMAPipeIR
  input_a: int
  input_b: int
  output: int
  global_size: tuple[int, int, int]
  local_size: tuple[int, int, int]
  wait_scope: str = "per_stage"
  resource_owner: str = "compiler"
  stage_count: int = 2
  wait_vmcnt: int | None = None
  slot_bytes: int = 1
  lifecycle: tuple[tuple[str, int, int], ...] = (("produce", 0, 0), ("ready", 0, 0),
    ("consume", 0, 0), ("produce", 1, 1), ("ready", 1, 1), ("consume", 1, 1))

  def __post_init__(self):
    if self.wait_scope != "per_stage" or self.resource_owner != "compiler" or self.stage_count != self.ir.stages:
      raise ValueError("invalid pipe op contract")
    if len({self.input_a, self.input_b, self.output}) != 3 or min(self.input_a, self.input_b, self.output) < 0:
      raise ValueError("pipe buffer ids must be distinct non-negative integers")
    if self.slot_bytes <= 0: raise ValueError("pipe slot_bytes must be positive")
    if any(len(x) != 3 or x[0] not in ("produce", "ready", "consume", "release") or x[1] < 0 or x[2] < 0 for x in self.lifecycle):
      raise ValueError("invalid pipe lifecycle event")
    produced = {(s, slot) for op, s, slot in self.lifecycle if op == "produce"}
    consumed = {(s, slot) for op, s, slot in self.lifecycle if op == "consume"}
    if not consumed.issubset(produced): raise ValueError("pipe consumes an unproduced stage/slot")
    for stage, slot in produced:
      if (stage, slot) not in consumed: raise ValueError("pipe slot is produced without a consume")
    live:set[int] = set()
    for event, _stage, slot in self.lifecycle:
      if event == "produce":
        if slot in live: raise ValueError("pipe slot is overwritten before consume")
        live.add(slot)
      elif event in ("consume", "release"):
        if slot not in live: raise ValueError("pipe consume/release lacks dominating produce")
        live.remove(slot)
    derived_wait = self.ir.loads_per_stage
    if self.wait_vmcnt is not None and self.wait_vmcnt != derived_wait: raise ValueError("pipe wait does not match staged loads")

  @property
  def derived_wait_vmcnt(self) -> int: return self.ir.loads_per_stage

  def resource_estimate(self) -> dict[str, Any]:
    """Compiler-side launch/resource facts; register counts remain unknown until lowering."""
    if len(self.global_size) != 3 or len(self.local_size) != 3 or any(x <= 0 for x in (*self.global_size, *self.local_size)):
      raise ValueError("pipe launch dimensions must be positive 3-tuples")
    if any(g < l for g, l in zip(self.global_size, self.local_size)):
      raise ValueError("pipe global dimensions must cover local dimensions")
    return {"global_size": self.global_size, "local_size": self.local_size,
            "lds_bytes": self.slot_bytes * self.stage_count,
            "scratch_bytes": 0, "vgpr": None, "sgpr": None,
            "resource_provenance": "typed_host_estimate_registers_unknown_until_lowering"}

def build_wmma_pipe_ir(spec: WMMAPipeSpec) -> WMMAPipeIR:
  return WMMAPipeIR(spec.role, (spec.m, spec.n, spec.k), spec.stages, spec.loads_per_stage, spec.wait_policy)

def attach_pipe_candidate_context(sink: UOp, context: KernelCandidateContext) -> UOp:
  """Attach typed pipe identity to an ordinary compiler sink; no native UOps."""
  if sink.op is not Ops.SINK: raise TypeError("pipe context attaches only to SINK")
  if not isinstance(context, KernelCandidateContext): raise TypeError("invalid candidate context")
  info = sink.arg if isinstance(sink.arg, KernelInfo) else KernelInfo()
  return sink.replace(arg=KernelInfo(name=info.name, axis_types=info.axis_types, dont_use_locals=info.dont_use_locals,
    applied_opts=info.applied_opts, opts_to_apply=info.opts_to_apply, estimates=info.estimates,
    candidate_context=context))


def build_wmma_pipe_barrier_chain(spec: WMMAPipeSpec, context: KernelCandidateContext) -> UOp:
  """Build the smallest compiler-owned LOAD -> barrier -> WMMA -> STORE graph.

  This is a structural slice only: ``Ops.BARRIER`` is a full workgroup fence
  in LLVM AMD lowering, not a targeted vmcnt implementation.  It is restricted
  to the proven attn_qo shape until a general ABI/resource contract exists.
  """
  if not isinstance(spec, WMMAPipeSpec) or not isinstance(context, KernelCandidateContext): raise TypeError("invalid pipe chain inputs")
  if spec.role != "attn_qo" or (spec.m, spec.n, spec.k) != (512, 4096, 4096):
    raise ValueError("barrier pipe slice only supports attn_qo 512x4096x4096")
  a = UOp.param(0, dtypes.half.ptr(spec.tile_m * spec.k))
  b = UOp.param(1, dtypes.half.ptr(spec.tile_n * spec.k))
  out = UOp.param(2, dtypes.float.vec(8).ptr(spec.tile_m * spec.tile_n // 8))
  ia = a.index(UOp.const(dtypes.weakint, 0), ptr=True)
  ib = b.index(UOp.const(dtypes.weakint, 0), ptr=True)
  la = ia.load(dtype=dtypes.half.vec(16))
  lb = ib.load(dtype=dtypes.half.vec(16))
  # BARRIER accepts effectful load sources directly; wrapping plain LOADs in
  # GROUP violates the verifier's effect ordering contract. The subsequent
  # operand loads are attached to the barrier through pointer AFTER edges.
  ready = UOp.barrier(la, lb)
  c = UOp.const(dtypes.float.vec(8), 0.0)
  tc_arg = ("WMMA_16_16_16_half_float", (16, 16, 16), dtypes.half, dtypes.float, "AMD", 32,
            (((101, 2), (102, 2), (103, 2), (104, 2)),) * 2 + (((1, 2),),), ())
  mma = UOp(Ops.WMMA, dtypes.float.vec(8), (la, lb, c), tc_arg)
  store = out.after(ready).index(UOp.const(dtypes.weakint, 0), ptr=True).store(mma)
  return attach_pipe_candidate_context(UOp.sink(store, ready), context)

def pipe_candidate_context(spec: WMMAPipeSpec, canonical_identity: str) -> KernelCandidateContext:
  """Typed compiler context for a generated pipe candidate.

  Geometry is intentionally absent until the backend compiler proves tile/LDS
  resource ownership; the pipeline payload is immutable JSON-shaped data.
  """
  if not isinstance(spec, WMMAPipeSpec): raise TypeError("expected WMMAPipeSpec")
  ir = build_wmma_pipe_ir(spec)
  payload = tuple((k, v) for k, v in (("schema", "wmma_pipe_ir.v1"), ("role", ir.role), ("shape", ir.shape),
             ("stages", ir.stages), ("loads_per_stage", ir.loads_per_stage), ("wait_policy", ir.wait_policy),
             ("stores", ir.stores), ("provenance", ir.provenance)))
  return KernelCandidateContext("boltbeam.full_kernel_candidate.v1", canonical_identity,
                               geometry=None, pipeline=payload)


def extract_wmma_pipe_spec(prefill_spec) -> WMMAPipeSpec | None:
  """Return the compiler primitive spec for register-resident pipe schedules.

  This is the minimal insertion point for replacing the current raw-instruction
  pipe oracle: callers already have a resolved PrefillGEMMScheduleSpec, and only
  `route_family == "pipe"` should be diverted into generated pipe lowering.
  """
  if prefill_spec.route_family != "pipe": return None
  if prefill_spec.pipeline_depth != 2: return None
  if prefill_spec.waitcnt_policy != "targeted_vmcnt": return None
  return WMMAPipeSpec(
    m=prefill_spec.m, n=prefill_spec.n, k=prefill_spec.k, tile_m=prefill_spec.tile_m, tile_n=prefill_spec.tile_n,
    pipe_tm=prefill_spec.pipe_tm, pipe_tn=prefill_spec.pipe_tn, wait_policy=prefill_spec.waitcnt_policy,
    target=prefill_spec.target, role=getattr(prefill_spec, "role", "unknown"),
  )


def pipe_primitive_local_stage_resource_plan(spec: WMMAPipeSpec, *, local_stage_requested: bool,
                                             lds_limit_bytes: int = 65536,
                                             allow_attn_kv_no_local_stage: bool = True) -> dict[str, Any]:
  """Return the pre-COMGR resource gate for generated pipe local staging.

  This is intentionally narrow and evidence-based. The S10 whole-route compile
  capture showed the generated `attn_kv` pipe transport for M=512,N=1024,K=4096
  declaring buf0[2048] half plus buf2[32768] half: 69632 bytes, over gfx1100's
  64 KiB LDS limit. Until generated pipe local staging is retiled, route that
  small-N case away from the generated transport before COMGR.
  """
  gate = "s10_attn_kv_generated_pipe_local_stage_lds"
  role = spec.role
  observed_overflow = local_stage_requested and role == "attn_kv" and spec.m == 512 and spec.n <= 1024 and spec.k >= 4096
  overflow_shared_arrays = []
  if observed_overflow:
    overflow_shared_arrays = [
      {"name": "buf0", "type": "half", "elements": 2048, "bytes": 4096},
      {"name": "buf2", "type": "half", "elements": 32768, "bytes": 65536},
    ]
  overflow_shared_bytes = sum(x["bytes"] for x in overflow_shared_arrays)
  no_local_stage_selected = observed_overflow and allow_attn_kv_no_local_stage
  shared_arrays = [] if no_local_stage_selected else overflow_shared_arrays
  shared_bytes = sum(x["bytes"] for x in shared_arrays)
  safe = no_local_stage_selected or shared_bytes <= lds_limit_bytes
  decision = "generated_pipe_no_local_stage" if no_local_stage_selected else ("allow" if safe else "fallback")
  return {
    "schema": "wmma-pipe-local-stage-resource-plan.v1",
    "gate": gate,
    "role": role,
    "local_stage_requested": local_stage_requested,
    "allow_attn_kv_no_local_stage": allow_attn_kv_no_local_stage,
    "no_local_stage_selected": no_local_stage_selected,
    "lds_limit_bytes": lds_limit_bytes,
    "estimated_shared_bytes": shared_bytes,
    "shared_arrays": shared_arrays,
    "overflow_estimated_shared_bytes": overflow_shared_bytes,
    "overflow_shared_arrays": overflow_shared_arrays,
    "safe": safe,
    "decision": decision,
    "fallback_reason": (
      f"{gate}: role={role} generated pipe local staging for M={spec.m},N={spec.n},K={spec.k} is estimated "
      f"to declare {shared_bytes} bytes LDS > {lds_limit_bytes}; decision={decision}"
      if not safe else None
    ),
  }


def lower_wmma_pipe_spec(spec: WMMAPipeSpec) -> Any:
  """Fail-closed generated pipe lowerer contract.

  This is the opt-in route seam for the future backend-owned WMMA pipe primitive.
  Until that backend lowerer exists, selecting this path must stop clearly rather
  than falling back to extra.qk.prefill.wmma::build_gemm_pipe.
  """
  if not isinstance(spec, WMMAPipeSpec):
    raise TypeError(f"lower_wmma_pipe_spec expected WMMAPipeSpec, got {type(spec).__name__}")
  if spec.stages != 2 or spec.wait_policy != "targeted_vmcnt":
    raise NotImplementedError(
      "WMMA pipe primitive lowering is not implemented for unsupported pipe specs; "
      f"expected stages=2 and wait_policy='targeted_vmcnt', got stages={spec.stages} "
      f"and wait_policy={spec.wait_policy!r}. No fallback to extra.qk.prefill.wmma.build_gemm_pipe was attempted."
    )
  raise NotImplementedError(
    "Generated WMMA pipe primitive lowering is not implemented yet. "
    "This opt-in seam intentionally fails closed and does not call "
    "extra.qk.prefill.wmma.build_gemm_pipe."
  )


def wmma_pipe_postrange_opts(spec: WMMAPipeSpec, *, unr: int = 2):
  from tinygrad.codegen.opt import Opt, OptOps
  return (
    Opt(OptOps.TC, 0, (-1, 2, 1)),
    Opt(OptOps.UPCAST, 0, spec.pipe_tm),
    Opt(OptOps.UPCAST, 1, spec.pipe_tn),
    Opt(OptOps.UNROLL, 0, unr),
  )


def build_wmma_pipe_diagnostic_lowering_report(spec: WMMAPipeSpec, *, loc: int = 0, unr: int = 2,
                                               target: str = "AMD:ISA:gfx1100", resident_ab: bool = True) -> dict[str, Any]:
  """Compile a generated AMD ISA matmul shaped by the pipe spec and report its lifecycle.

  This is a diagnostic compiler-owned lowering proof. It intentionally does not return a route-bound custom-kernel
  instruction list, because the current graph-GEMM route still uses a hand custom-kernel ABI.
  """
  if not isinstance(spec, WMMAPipeSpec):
    raise TypeError(f"build_wmma_pipe_diagnostic_lowering_report expected WMMAPipeSpec, got {type(spec).__name__}")
  if spec.stages != 2 or spec.wait_policy != "targeted_vmcnt":
    raise NotImplementedError("diagnostic WMMA pipe lowering only supports stages=2 and targeted_vmcnt")

  from collections import Counter
  from tinygrad import Tensor, dtypes
  from tinygrad.codegen import to_program, to_program_cache
  from tinygrad.codegen.opt import postrange
  import os
  from tinygrad.helpers import Context, Target, getenv
  from tinygrad.renderer.isa.amd import AMDISARenderer
  from tinygrad.uop.ops import Ops
  from extra.qk.prefill import native_isa_l4_stream_probe as sp

  old_env = {k: os.environ.get(k) for k in (
    "AMD_ISA_WAITCNT_TARGETED", "AMD_ISA_WMMA_B128_FRAG", "AMD_ISA_REG_ACCUM", "PREFILL_WMMA_CHAIN_AB_RESIDENT",
    "PREFILL_TC_LOCAL_STAGE", "PREFILL_TC_LOCAL_STAGE_WITH_LOCAL", "PREFILL_TC_LOCAL_STAGE_B_TILEKEY",
    "PREFILL_LDS_PACK_WITHLOCAL_B128")}
  os.environ["AMD_ISA_WAITCNT_TARGETED"] = "1"
  os.environ["AMD_ISA_WMMA_B128_FRAG"] = "1"
  os.environ["AMD_ISA_REG_ACCUM"] = "1"
  if resident_ab: os.environ["PREFILL_WMMA_CHAIN_AB_RESIDENT"] = "1"
  for k in ("PREFILL_TC_LOCAL_STAGE", "PREFILL_TC_LOCAL_STAGE_WITH_LOCAL", "PREFILL_TC_LOCAL_STAGE_B_TILEKEY",
            "PREFILL_LDS_PACK_WITHLOCAL_B128"):
    os.environ.pop(k, None)
  getenv.cache_clear()
  to_program_cache.clear()
  old_warmstart = postrange._WARMSTART_OPTS
  old_local_stage_keys = getattr(postrange, "_WARMSTART_LOCAL_STAGE_KEYS", None)
  postrange._WARMSTART_OPTS = {(frozenset({spec.m, spec.n}), spec.k): wmma_pipe_postrange_opts(spec, unr=unr)}
  postrange._WARMSTART_LOCAL_STAGE_KEYS = set()
  postrange._warmstart_stats.update({"match": 0, "apply": 0, "error": 0})
  try:
    with Context(DEV=target):
      a = Tensor.empty(spec.m, spec.k, dtype=dtypes.half)
      b = Tensor.empty(spec.n, spec.k, dtype=dtypes.half)
      ast = [u for u in (a @ b.transpose()).schedule_linear().toposort() if u.op is Ops.SINK][0]
      ren = AMDISARenderer(Target.parse(target))
      prg = to_program(ast, ren)
      lin = next(u for u in prg.src if u.op is Ops.LINEAR)
      final = sp._final_stream(ren, lin.src)
      insts = sp._insts_from_uops(final)
  finally:
    for k, v in old_env.items():
      if v is None: os.environ.pop(k, None)
      else: os.environ[k] = v
    getenv.cache_clear()
    to_program_cache.clear()
    postrange._WARMSTART_OPTS = old_warmstart
    postrange._WARMSTART_LOCAL_STAGE_KEYS = old_local_stage_keys

  names = [sp._mn(i) for i in insts if not isinstance(i, tuple)]
  counts = Counter(names)
  waits = []
  for idx, inst in enumerate(insts):
    if isinstance(inst, tuple): continue
    simm16 = sp._waitcnt_simm16(inst)
    if simm16 is not None: waits.append({"idx": idx, **sp._decode_waitcnt(simm16)})
  nonfull_waits = [w for w in waits if w["vmcnt"] < 0x3F or w["lgkmcnt"] < 0x3F]
  vmcnt_target_waits = [w for w in waits if 0 < w["vmcnt"] < 0x3F]
  core_structure_ok = (
    counts.get("global_load_b128", 0) > 0 and
    counts.get(sp.WMMA_NAME, 0) > 0 and
    counts.get("global_store_b16", 0) > 0 and
    len(waits) > 0 and
    counts.get("global_load_u16", 0) == 0 and
    postrange._warmstart_stats["apply"] > 0
  )
  pipe_wait_ok = any(w["vmcnt"] == spec.loads_per_stage for w in vmcnt_target_waits)
  return {
    "schema": "wmma-pipe-diagnostic-lowering.v1",
    "transport": "generated_program_diagnostic",
    "route_bound": False,
    "uses_hand_pipe_oracle": False,
    "uses_route_local_full_ops_ins": False,
    "program": str(prg.arg),
    "target": target,
    "loc": loc,
    "unr": unr,
    "resident_ab": resident_ab,
    "warmstart": dict(postrange._warmstart_stats),
    "spec": spec.to_json(),
    "instruction_total": len(names),
    "instruction_counts": dict(sorted(counts.items())),
    "track_counts": {name: counts.get(name, 0) for name in ("global_load_b128", "global_load_u16",
                                                            "global_store_b16", sp.WMMA_NAME, "s_waitcnt")},
    "waitcnt_summary": {
      "count": len(waits),
      "nonfull_count": len(nonfull_waits),
      "vmcnt_target_count": len(vmcnt_target_waits),
      "expected_pipe_vmcnt": spec.loads_per_stage,
      "has_expected_pipe_vmcnt": pipe_wait_ok,
      "vmcnt_sequence": [w["vmcnt"] for w in waits],
      "lgkmcnt_sequence": [w["lgkmcnt"] for w in waits],
    },
    "mvp_core_structure_ok": core_structure_ok,
    "mvp_pipe_wait_ok": pipe_wait_ok,
    "mvp_structure_ok": core_structure_ok and pipe_wait_ok,
    "next_blocker": "generated pipe wait policy and route transport: preserve future loads, then execute compiler-owned program directly",
  }


def run_wmma_pipe_diagnostic_correctness(spec: WMMAPipeSpec, *, target: str = "AMD:ISA:gfx1100",
                                         resident_ab: bool = True, seed: int = 0) -> dict[str, Any]:
  """Execute the bounded generated pipe diagnostic as an ordinary compiler program and compare to fp32 numpy."""
  if not isinstance(spec, WMMAPipeSpec):
    raise TypeError(f"run_wmma_pipe_diagnostic_correctness expected WMMAPipeSpec, got {type(spec).__name__}")
  import os
  import numpy as np
  from tinygrad import Device, Tensor, dtypes
  from tinygrad.codegen import to_program_cache
  from tinygrad.codegen.opt import postrange
  from tinygrad.helpers import Context, getenv

  old_env = {k: os.environ.get(k) for k in (
    "AMD_ISA_WAITCNT_TARGETED", "AMD_ISA_WMMA_B128_FRAG", "AMD_ISA_REG_ACCUM", "PREFILL_WMMA_CHAIN_AB_RESIDENT",
    "PREFILL_TC_LOCAL_STAGE", "PREFILL_TC_LOCAL_STAGE_WITH_LOCAL", "PREFILL_TC_LOCAL_STAGE_B_TILEKEY",
    "PREFILL_LDS_PACK_WITHLOCAL_B128")}
  os.environ["AMD_ISA_WAITCNT_TARGETED"] = "1"
  os.environ["AMD_ISA_WMMA_B128_FRAG"] = "1"
  os.environ["AMD_ISA_REG_ACCUM"] = "1"
  if resident_ab: os.environ["PREFILL_WMMA_CHAIN_AB_RESIDENT"] = "1"
  for k in ("PREFILL_TC_LOCAL_STAGE", "PREFILL_TC_LOCAL_STAGE_WITH_LOCAL", "PREFILL_TC_LOCAL_STAGE_B_TILEKEY",
            "PREFILL_LDS_PACK_WITHLOCAL_B128"):
    os.environ.pop(k, None)
  getenv.cache_clear()
  to_program_cache.clear()
  rng = np.random.default_rng(seed)
  a_np = (rng.standard_normal((spec.m, spec.k)) * 0.1).astype(np.float16)
  b_np = (rng.standard_normal((spec.n, spec.k)) * 0.1).astype(np.float16)
  ref = a_np.astype(np.float32) @ b_np.astype(np.float32).T
  refn = float(np.sqrt(np.mean(ref ** 2)) + 1e-9)
  old_warmstart = postrange._WARMSTART_OPTS
  old_local_stage_keys = getattr(postrange, "_WARMSTART_LOCAL_STAGE_KEYS", None)
  try:
    with Context(DEV=target):
      postrange._WARMSTART_OPTS = {(frozenset({spec.m, spec.n}), spec.k): wmma_pipe_postrange_opts(spec)}
      postrange._WARMSTART_LOCAL_STAGE_KEYS = set()
      postrange._warmstart_stats.update({"match": 0, "apply": 0, "error": 0})
      a = Tensor(a_np, dtype=dtypes.half)
      b = Tensor(b_np, dtype=dtypes.half)
      out_t = (a @ b.transpose()).realize()
      Device[Device.DEFAULT].synchronize()
      out = out_t.float().numpy()
  finally:
    for k, v in old_env.items():
      if v is None: os.environ.pop(k, None)
      else: os.environ[k] = v
    getenv.cache_clear()
    to_program_cache.clear()
    postrange._WARMSTART_OPTS = old_warmstart
    postrange._WARMSTART_LOCAL_STAGE_KEYS = old_local_stage_keys
  rel_rmse = float(np.sqrt(np.mean((out - ref) ** 2)) / refn)
  max_abs = float(np.max(np.abs(out - ref)))
  return {
    "schema": "wmma-pipe-diagnostic-correctness.v1",
    "target": target,
    "resident_ab": resident_ab,
    "seed": seed,
    "spec": spec.to_json(),
    "finite": bool(np.isfinite(out).all()),
    "rel_rmse": rel_rmse,
    "max_abs_error": max_abs,
    "threshold": 2e-2,
    "passed": bool(np.isfinite(rel_rmse) and rel_rmse <= 2e-2),
    "warmstart": dict(postrange._warmstart_stats),
  }


def wmma_pipe_lowering_insertion_point() -> dict[str, Any]:
  return {
    "route_spec_source": "extra/qk/prefill_schedule_spec.py::describe_prefill_schedule",
    "current_raw_lowering": "extra/qk/prefill_schedule_spec.py::emit_prefill_gemm_from_spec -> "
                            "extra/qk/prefill_graph_gemm_route.py::_emit_schedule -> "
                            "extra/qk/prefill/wmma.py::build_gemm_pipe",
    "first_generated_diversion": "extra/qk/prefill_schedule_spec.py::emit_prefill_gemm_from_spec",
    "diversion_predicate": 'PrefillGEMMScheduleSpec.route_family == "pipe"',
    "primitive_spec": "extra/qk/wmma_pipe_spec.py::WMMAPipeSpec",
    "primitive_lowerer": "extra/qk/wmma_pipe_spec.py::lower_wmma_pipe_spec",
    "diagnostic_lowerer": "extra/qk/wmma_pipe_spec.py::build_wmma_pipe_diagnostic_lowering_report",
    "do_not_copy": ("extra/qk/prefill/wmma.py::build_gemm_pipe instruction list",
                    "route-local UOp(Ops.INS, ...) kernel body"),
  }
