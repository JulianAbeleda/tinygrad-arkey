"""Compile the exact attn_qo transport through tinygrad's normal compiler.

This is compile-only preparation.  It returns the real PROGRAM UOp and gated
evidence; callers must use the separate runtime bridge to create an executable
handle and must call that handle explicitly to dispatch.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
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

from extra.qk.prefill.attn_qo_l2_lds_pair_generator_20260712 import generate_pair
from extra.qk.prefill.executable_artifact_preparation import compile_evidence, compile_transport_evidence
from extra.qk.prefill_graph_gemm_route import _candidate_schedule_spec, _primitive_warmstart_key
from extra.qk.prefill_graph_gemm_route import _emit_schedule
from extra.qk.prefill_schedule_spec import _spec_to_params, describe_prefill_schedule, register_resident_postrange_opts
from extra.qk.runtime_specs import (admit_full_kernel_candidate, admit_full_kernel_candidate_set,
                                     full_kernel_candidate_set_from_legacy)


@contextmanager
def _isolated_compile_environment():
  """Prevent unrelated route experiments from changing exact artifact identity."""
  saved = {key: value for key, value in os.environ.items() if key.startswith(("PREFILL_", "AMD_ISA_"))}
  for key in saved: os.environ.pop(key, None)
  getenv.cache_clear(); to_program_cache.clear()
  try:
    yield
  finally:
    for key in tuple(key for key in os.environ if key.startswith(("PREFILL_", "AMD_ISA_"))): os.environ.pop(key, None)
    os.environ.update(saved)
    getenv.cache_clear(); to_program_cache.clear()


def _workload_axes(workload: dict[str, Any]) -> tuple[str, str, tuple[int, int, int], dict[str, Any]]:
  """Unpack the experiment-row workload (P2-3) instead of module constants."""
  shape = workload["shape"]
  return workload["profile"], workload["role"], (shape["m"], shape["n"], shape["k"]), dict(workload["target"])


def _direct_compile_evidence(admission: Any, record: dict[str, Any], *, profile: str, role: str,
                             shape: tuple[int, int, int], target: dict[str, Any]) -> dict[str, Any]:
  roles = sorted({row.get("role") for row in record.get("allocator", {}).get("leases", ()) if isinstance(row, dict)})
  coverage = admission.pipeline_plan.wait_coverage
  pipeline = {"storage_kind": "global_register_resident", "lds_bytes": 0,
              "consumer_identity": "amd.rdna3.wmma.fp16.v1",
              "register_mapping": {"backend": "amd_vgpr", "addressing": "static", "required_roles": roles},
              "wait_required_edges": [list(edge) for edge in coverage.covered]}
  wait = {"typed": True, "kind": "targeted_vmcnt", "coverage": coverage.to_json()}
  abi = {"wave_size": target["wave_size"], "fragment_carrier": "half.vec(16)", "accumulator_carrier": "float.vec(8)"}
  binding = {"profile": profile, "role": role,
             "shape": {"m": shape[0], "n": shape[1], "k": shape[2]}, "target": dict(target)}
  return compile_evidence(record, pipeline=pipeline, wait=wait, abi_contract=abi,
                          surface={"strict_pure": True, "ops_ins_count": 0}, runtime_binding=binding)


def _diagnostic_only_raw_isa_lds_program(admission: Any, target: str, *, role: str,
                                         shape: tuple[int, int, int]) -> tuple[UOp, dict[str, Any]]:
  """DIAGNOSTIC ONLY -- the BROKEN raw-ISA LDS2 oracle (hangs the GPU).

  This is the retired ``_compile_lds_program`` raw ``build_gemm_lds2`` oracle.
  It is QUARANTINED: it is NOT wired into ``_COMPILE_ADAPTERS`` and is unreachable
  as the benchmark ``lds`` transport.  The benchmark ``lds`` candidate compiles
  the proven ``route_pf16_graph_gemm`` WMMA-LDS kernel via ``_lds_compile_adapter``.
  Kept only as an explicitly-named reference for the historical raw oracle; do
  not dispatch its output.
  """
  spec = _candidate_schedule_spec(describe_prefill_schedule(shape[1], shape[2], role=role), admission)
  built = _emit_schedule(_spec_to_params(spec), name=spec.kernel_name)
  if built is None: raise RuntimeError("exact LDS schedule is not tile-divisible")
  insts, lds_bytes, bm, bn, threads, name = built
  grid = (shape[1] // bn, shape[0] // bm, 1)

  def asm_kernel(a, b, c):
    lds = UOp(Ops.DEFINE_LOCAL, dtypes.uint8.ptr(size=lds_bytes, addrspace=AddrSpace.LOCAL), (), "lds")
    g = [UOp.special(grid[0], "gidx0"), UOp.special(grid[1], "gidx1")]
    sink = UOp.sink(a.base, b.base, c.base, lds, *g, UOp.special(threads, "lidx0"),
                    arg=KernelInfo(name=colored(name, "cyan"),
                                   estimates=Estimates(ops=shape[0] * shape[1] * shape[2] * 2,
                                                       mem=(shape[0] * shape[2] + shape[1] * shape[2] + shape[0] * shape[1]) * 2)))
    return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=Device.DEFAULT),
                                 UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=i) for i in insts))))

  with Context(DEV=target):
    a = Tensor.empty(shape[0], shape[2], dtype=dtypes.half)
    b = Tensor.empty(shape[1], shape[2], dtype=dtypes.half)
    c = Tensor.empty(shape[0], shape[1], dtype=dtypes.half)
    compiled = compile_linear(Tensor.custom_kernel(a, b, c, fxn=asm_kernel)[2].schedule_linear())
  programs = [u for u in compiled.toposort() if u.op is Ops.PROGRAM and isinstance(u.arg, ProgramInfo)]
  if len(programs) != 1: raise RuntimeError(f"expected one LDS attn_qo PROGRAM, found {len(programs)}")
  return programs[0], {"tile_m": bm, "tile_n": bn, "tile_k": spec.tile_k, "threads": threads,
                       "lds_bytes": lds_bytes, "dbuf": spec.dbuf, "pipeline_depth": spec.pipeline_depth,
                       "waves_m": spec.waves_m, "waves_n": spec.waves_n}


def _derive_lds_argument_order(program: UOp, shape: tuple[int, int, int]) -> tuple[str, ...]:
  """Derive the kernel ABI order from the compiled PROGRAM's globals.

  The proven WMMA-LDS kernel is the compiler-rendered ``a @ b^T`` matmul, so its
  parameter order is read from the lowered PROGRAM: the STORE-target global is
  ``output=(M,N)``; the two LOAD globals are ``a=(M,K)`` and ``b=(N,K)``,
  distinguished by flat element count (M*K vs N*K, fp16).
  """
  import math
  m, n, k = shape
  info = program.arg
  out_slots, in_slots = set(getattr(info, "outs", ())), set(getattr(info, "ins", ()))
  params = {u.arg.slot: u for u in program.src[0].toposort() if u.op is Ops.PARAM}
  order: list[str] = []
  for slot in getattr(info, "globals", ()):
    size = math.prod(getattr(params[slot], "shape", ())) if slot in params else None
    if slot in out_slots and slot not in in_slots: order.append("output")
    elif size == m * k: order.append("a")
    elif size == n * k: order.append("b")
    else: order.append("output" if slot in out_slots else f"slot{slot}")
  return tuple(order)


def _lds_compile_adapter(candidate_row: dict[str, Any], admission: Any, workload: dict[str, Any],
                         dev_target: str) -> tuple[UOp, dict[str, Any]]:
  """Compile the benchmark ``lds`` candidate as the proven WMMA-LDS kernel.

  Delegates to :func:`compile_candidate_program`, which routes the two-buffer
  candidate through ``route_pf16_graph_gemm`` (the proven WMMA-LDS transport),
  selects its exact compiled PROGRAM, and proves the 40960-byte / two-buffer
  structure.  This NEVER touches the retired raw-ISA oracle.
  """
  from tinygrad.helpers import Context
  from extra.qk.prefill.single_buffer_execution_authority import compile_candidate_program
  profile, role, shape, target = _workload_axes(workload)
  # The proven WMMA-LDS kernel is compiled by the real AMD backend (matching the
  # runtime device the bridge loads), NOT the minimal native-ISA renderer used by
  # the retired raw oracle -- the ISA renderer cannot lower the two-buffer LDS
  # staging.  ``dev_target`` (the ISA variant) is intentionally unused here.
  with Context(DEV=target["backend"]):
    prepared = compile_candidate_program(candidate_row["payload"], candidate_row["canonical_identity"])
  program, structural, surface = prepared.program, prepared.structural_binding, prepared.surface
  if not structural.get("pre_gpu_eligible"):
    raise RuntimeError("proven WMMA-LDS candidate structure is unproven: " + "; ".join(structural.get("errors", ())))
  if not surface.get("strict_pure"):
    raise RuntimeError("proven WMMA-LDS candidate did not select the strict compiler-rendered surface")
  pipeline = getattr(getattr(program.src[0].arg, "candidate_context", None), "pipeline", None)
  buffer_count = pipeline.buffer_count if pipeline is not None else 1
  active_lds = pipeline.active_lds_bytes if pipeline is not None else 0
  argument_order = _derive_lds_argument_order(program, shape)
  source = next(u.arg for u in program.src if u.op is Ops.SOURCE)
  binary = next(u.arg for u in program.src if u.op is Ops.BINARY)
  binary_sha256 = hashlib.sha256(binary).hexdigest()
  geo = admission.geometry
  schedule = {"threads": geo.threads, "tile_m": geo.tile[0], "tile_n": geo.tile[1], "tile_k": geo.tile[2],
              "waves_m": geo.waves[0], "waves_n": geo.waves[1], "buffer_count": buffer_count,
              "dbuf": int(buffer_count == 2), "lds_bytes": active_lds}
  evidence = {"schema": "prefill-transport-compile.v1", "transport": "lds",
              "canonical_identity": candidate_row["canonical_identity"], "binary_sha256": binary_sha256,
              "passed": bool(structural["pre_gpu_eligible"]), "errors": list(structural["errors"]),
              "program": {"name": getattr(program.arg, "name", None),
                          "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                          "binary_sha256": binary_sha256,
                          "global_size": list(getattr(program.arg, "global_size", ()) or ()),
                          "local_size": list(getattr(program.arg, "local_size", ()) or ()),
                          "target": next((u.arg for u in program.src if u.op is Ops.DEVICE), None),
                          "abi": "amdgpu_kernel"},
              "schedule": schedule, "structural": structural, "argument_order": list(argument_order),
              "surface": {"strict_pure": True, "ops_ins_count": surface["ops_ins_count"],
                          "generator": "route_pf16_graph_gemm.generated_lds_matmul_transport", "lds_transport": True},
              "capture": {"mode": "compile_only", "dispatch_permitted": False,
                          "resource_authority": "compiled_program_descriptor"},
              "runtime_binding": {"profile": profile, "role": role,
                                  "shape": {"m": shape[0], "n": shape[1], "k": shape[2]}, "target": dict(target)}}
  return program, evidence


def _direct_compile_adapter(candidate_row: dict[str, Any], admission: Any, workload: dict[str, Any],
                            dev_target: str) -> tuple[UOp, dict[str, Any]]:
  profile, role, shape, target = _workload_axes(workload)
  spec = describe_prefill_schedule(shape[1], shape[2], role=role)
  candidate_spec = _candidate_schedule_spec(spec, admission)
  key = _primitive_warmstart_key(candidate_spec)
  old_opts, old_contexts = postrange._WARMSTART_OPTS, postrange._WARMSTART_CANDIDATE_CONTEXTS
  try:
    postrange._WARMSTART_OPTS = {**(old_opts or {}), key: register_resident_postrange_opts(candidate_spec)}
    postrange._WARMSTART_CANDIDATE_CONTEXTS = {**(old_contexts or {}), key: admission.context}
    getenv.cache_clear(); to_program_cache.clear()
    with Context(DEV=dev_target):
      a = Tensor.empty(shape[0], shape[2], dtype=dtypes.half)
      b = Tensor.empty(shape[1], shape[2], dtype=dtypes.half)
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
  evidence = _direct_compile_evidence(admission, attachments[-1], profile=profile, role=role, shape=shape, target=target)
  return program, evidence


# One explicit transport -> compile-adapter table.  Both compile-only paths
# produce their compile evidence through this single boundary; an unknown
# transport is rejected fail-closed instead of silently defaulting to LDS.
_COMPILE_ADAPTERS = {"direct_l2": _direct_compile_adapter, "lds": _lds_compile_adapter}


def _dev_target(target: str | None, workload_target: dict[str, Any]) -> str:
  return target or f"{workload_target['backend']}:ISA:{workload_target['arch']}"


def compile_attn_qo_program(*, transport: str = "direct_l2", target: str | None = None) -> dict[str, Any]:
  adapter = _COMPILE_ADAPTERS.get(transport)
  if adapter is None:
    raise ValueError(f"unsupported attn_qo transport: {transport!r}; registered adapters: {tuple(sorted(_COMPILE_ADAPTERS))}")
  pair = generate_pair()
  candidate_row = pair["candidates"][transport]
  workload = candidate_row["payload"]["workload"]
  profile, role, shape, target_dict = _workload_axes(workload)
  dev_target = _dev_target(target, target_dict)
  # Route through the candidate-SET admission so the two-buffer LDS candidate
  # admits GFX1100_TWO_BUFFER_STAGE1_CAPABILITY (the singular admit only supports
  # single-buffer stage1).  direct_l2 (register) still resolves the same way.
  admission = admit_full_kernel_candidate_set(
    full_kernel_candidate_set_from_legacy(candidate_row["payload"], candidate_row["canonical_identity"])).admissions[0]
  with _isolated_compile_environment():
    program, evidence = adapter(candidate_row, admission, workload, dev_target)
  return {"schema": "attn_qo.executable_preparation.v1", "transport": transport,
          "pair_key": pair["pair_key"], "schedule_digest": pair["schedule_digest"],
          "candidate": candidate_row["payload"], "canonical_identity": candidate_row["canonical_identity"],
          "program": program, "compile_evidence": evidence, "dispatch_performed": False}


def compile_attn_qo_pair(*, target: str | None = None) -> dict[str, Any]:
  """Compile both exact transports while preserving one semantic pair key."""
  pair = generate_pair()
  prepared = {name: compile_attn_qo_program(transport=name, target=target) for name in ("direct_l2", "lds")}
  if any(row["pair_key"] != pair["pair_key"] or row["schedule_digest"] != pair["schedule_digest"]
         for row in prepared.values()):
    raise RuntimeError("compiled transport artifacts do not share the generated semantic pair identity")
  return {"schema": "attn_qo.executable_pair_preparation.v1", "pair_key": pair["pair_key"],
          "schedule_digest": pair["schedule_digest"], "transports": prepared, "dispatch_performed": False}


__all__ = ["compile_attn_qo_program", "compile_attn_qo_pair"]
