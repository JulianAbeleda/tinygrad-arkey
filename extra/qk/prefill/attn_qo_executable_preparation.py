"""Compile the exact attn_qo transport through tinygrad's normal compiler.

This is compile-only preparation.  It returns the real PROGRAM UOp and gated
evidence; callers must use the separate runtime bridge to create an executable
handle and must call that handle explicitly to dispatch.
"""
from __future__ import annotations

from typing import Any

from tinygrad import Tensor, dtypes
from tinygrad.codegen import to_program_cache
from tinygrad.codegen.opt import postrange
from tinygrad.engine.realize import Estimates, compile_linear
from tinygrad.device import Device
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import Context, getenv
from tinygrad.helpers import colored
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp

from extra.qk.prefill.attn_qo_l2_lds_pair_generator_20260712 import PROFILE, ROLE, SHAPE, TARGET, generate_pair
from extra.qk.prefill.executable_artifact_preparation import compile_evidence, compile_transport_evidence
from extra.qk.prefill_graph_gemm_route import _candidate_schedule_spec, _primitive_warmstart_key
from extra.qk.prefill_graph_gemm_route import _emit_schedule
from extra.qk.prefill_schedule_spec import _spec_to_params, describe_prefill_schedule, register_resident_postrange_opts
from extra.qk.runtime_specs import admit_full_kernel_candidate


def _direct_compile_evidence(admission: Any, record: dict[str, Any]) -> dict[str, Any]:
  roles = sorted({row.get("role") for row in record.get("allocator", {}).get("leases", ()) if isinstance(row, dict)})
  coverage = admission.pipeline_plan.wait_coverage
  pipeline = {"storage_kind": "global_register_resident", "lds_bytes": 0,
              "consumer_identity": "amd.rdna3.wmma.fp16.v1",
              "register_mapping": {"backend": "amd_vgpr", "addressing": "static", "required_roles": roles},
              "wait_required_edges": [list(edge) for edge in coverage.covered]}
  wait = {"typed": True, "kind": "targeted_vmcnt", "coverage": coverage.to_json()}
  abi = {"wave_size": 32, "fragment_carrier": "half.vec(16)", "accumulator_carrier": "float.vec(8)"}
  binding = {"profile": PROFILE, "role": ROLE,
             "shape": {"m": SHAPE[0], "n": SHAPE[1], "k": SHAPE[2]}, "target": dict(TARGET)}
  return compile_evidence(record, pipeline=pipeline, wait=wait, abi_contract=abi,
                          surface={"strict_pure": True, "ops_ins_count": 0}, runtime_binding=binding)


def _compile_lds_program(admission: Any, target: str) -> tuple[UOp, dict[str, Any]]:
  """Compile the exact LDS candidate through the existing raw LDS2 generator.

  This deliberately does not enter the unfinished generated LDS precontract
  lowering.  The instruction generator and custom-kernel ABI are already the
  working route used by the graph GEMM path; this helper only packages that
  route as a compile-only PROGRAM for the shared runtime bridge.
  """
  spec = _candidate_schedule_spec(describe_prefill_schedule(SHAPE[1], SHAPE[2], role=ROLE), admission)
  built = _emit_schedule(_spec_to_params(spec), name=spec.kernel_name)
  if built is None: raise RuntimeError("exact LDS schedule is not tile-divisible")
  insts, lds_bytes, bm, bn, threads, name = built
  grid = (SHAPE[1] // bn, SHAPE[0] // bm, 1)

  def asm_kernel(a, b, c):
    lds = UOp(Ops.DEFINE_LOCAL, dtypes.uint8.ptr(size=lds_bytes, addrspace=AddrSpace.LOCAL), (), "lds")
    g = [UOp.special(grid[0], "gidx0"), UOp.special(grid[1], "gidx1")]
    sink = UOp.sink(a.base, b.base, c.base, lds, *g, UOp.special(threads, "lidx0"),
                    arg=KernelInfo(name=colored(name, "cyan"),
                                   estimates=Estimates(ops=SHAPE[0] * SHAPE[1] * SHAPE[2] * 2,
                                                       mem=(SHAPE[0] * SHAPE[2] + SHAPE[1] * SHAPE[2] + SHAPE[0] * SHAPE[1]) * 2)))
    return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=Device.DEFAULT),
                                 UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=i) for i in insts))))

  with Context(DEV=target):
    a = Tensor.empty(SHAPE[0], SHAPE[2], dtype=dtypes.half)
    b = Tensor.empty(SHAPE[1], SHAPE[2], dtype=dtypes.half)
    c = Tensor.empty(SHAPE[0], SHAPE[1], dtype=dtypes.half)
    compiled = compile_linear(Tensor.custom_kernel(a, b, c, fxn=asm_kernel)[2].schedule_linear())
  programs = [u for u in compiled.toposort() if u.op is Ops.PROGRAM and isinstance(u.arg, ProgramInfo)]
  if len(programs) != 1: raise RuntimeError(f"expected one LDS attn_qo PROGRAM, found {len(programs)}")
  return programs[0], {"tile_m": bm, "tile_n": bn, "tile_k": spec.tile_k, "threads": threads,
                       "lds_bytes": lds_bytes, "dbuf": spec.dbuf, "pipeline_depth": spec.pipeline_depth,
                       "waves_m": spec.waves_m, "waves_n": spec.waves_n}


def compile_attn_qo_program(*, transport: str = "direct_l2", target: str = "AMD:ISA:gfx1100") -> dict[str, Any]:
  if transport not in ("direct_l2", "lds"):
    raise ValueError(f"unsupported attn_qo transport: {transport}")
  pair = generate_pair()
  candidate_row = pair["candidates"][transport]
  admission = admit_full_kernel_candidate(candidate_row["payload"], candidate_row["canonical_identity"],
                                          profile=PROFILE, role=ROLE, shape=SHAPE, target=TARGET)
  if transport == "lds":
    program, schedule = _compile_lds_program(admission, target)
    evidence = compile_transport_evidence(
      program, transport=transport, canonical_identity=candidate_row["canonical_identity"], schedule=schedule,
      surface={"strict_pure": False, "ops_ins_count": 0,
               "generator": "extra.qk.prefill.wmma.build_gemm_lds2", "lds_transport": True},
      runtime_binding={"profile": PROFILE, "role": ROLE,
                       "shape": {"m": SHAPE[0], "n": SHAPE[1], "k": SHAPE[2]}, "target": dict(TARGET)})
    return {"schema": "attn_qo.executable_preparation.v1", "transport": transport,
            "pair_key": pair["pair_key"], "schedule_digest": pair["schedule_digest"],
            "candidate": candidate_row["payload"], "canonical_identity": candidate_row["canonical_identity"],
            "program": program, "compile_evidence": evidence, "dispatch_performed": False}
  spec = describe_prefill_schedule(SHAPE[1], SHAPE[2], role=ROLE)
  candidate_spec = _candidate_schedule_spec(spec, admission)
  key = _primitive_warmstart_key(candidate_spec)
  old_opts, old_contexts = postrange._WARMSTART_OPTS, postrange._WARMSTART_CANDIDATE_CONTEXTS
  try:
    postrange._WARMSTART_OPTS = {**(old_opts or {}), key: register_resident_postrange_opts(candidate_spec)}
    postrange._WARMSTART_CANDIDATE_CONTEXTS = {**(old_contexts or {}), key: admission.context}
    getenv.cache_clear(); to_program_cache.clear()
    with Context(DEV=target):
      a = Tensor.empty(SHAPE[0], SHAPE[2], dtype=dtypes.half)
      b = Tensor.empty(SHAPE[1], SHAPE[2], dtype=dtypes.half)
      compiled = compile_linear((a @ b.transpose()).schedule_linear())
  finally:
    postrange._WARMSTART_OPTS, postrange._WARMSTART_CANDIDATE_CONTEXTS = old_opts, old_contexts
    getenv.cache_clear(); to_program_cache.clear()
  programs = [u for u in compiled.toposort() if u.op is Ops.PROGRAM and isinstance(u.arg, ProgramInfo)
              and getattr(getattr(u.src[0].arg, "candidate_context", None), "canonical_identity", None)
              == candidate_row["canonical_identity"]]
  if len(programs) != 1: raise RuntimeError(f"expected one direct attn_qo PROGRAM, found {len(programs)}")
  program: UOp = programs[0]
  attachments = [x.record for x in program.arg.aux if hasattr(x, "record")]
  if not attachments: raise RuntimeError("direct attn_qo PROGRAM has no compiler-owned final capture")
  evidence = _direct_compile_evidence(admission, attachments[-1])
  return {"schema": "attn_qo.executable_preparation.v1", "transport": transport,
          "pair_key": pair["pair_key"], "schedule_digest": pair["schedule_digest"],
          "candidate": candidate_row["payload"], "canonical_identity": candidate_row["canonical_identity"],
          "program": program, "compile_evidence": evidence, "dispatch_performed": False}


def compile_attn_qo_pair(*, target: str = "AMD:ISA:gfx1100") -> dict[str, Any]:
  """Compile both exact transports while preserving one semantic pair key."""
  pair = generate_pair()
  prepared = {name: compile_attn_qo_program(transport=name, target=target) for name in ("direct_l2", "lds")}
  if any(row["pair_key"] != pair["pair_key"] or row["schedule_digest"] != pair["schedule_digest"]
         for row in prepared.values()):
    raise RuntimeError("compiled transport artifacts do not share the generated semantic pair identity")
  return {"schema": "attn_qo.executable_pair_preparation.v1", "pair_key": pair["pair_key"],
          "schedule_digest": pair["schedule_digest"], "transports": prepared, "dispatch_performed": False}


__all__ = ["compile_attn_qo_program", "compile_attn_qo_pair"]
