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
from tinygrad.engine.realize import compile_linear
from tinygrad.helpers import Context, getenv
from tinygrad.uop.ops import Ops, ProgramInfo, UOp

from extra.qk.prefill.attn_qo_l2_lds_pair_generator_20260712 import PROFILE, ROLE, SHAPE, TARGET, generate_pair
from extra.qk.prefill.executable_artifact_preparation import compile_evidence
from extra.qk.prefill_graph_gemm_route import _candidate_schedule_spec, _primitive_warmstart_key
from extra.qk.prefill_schedule_spec import describe_prefill_schedule, register_resident_postrange_opts
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


def compile_attn_qo_program(*, transport: str = "direct_l2", target: str = "AMD:ISA:gfx1100") -> dict[str, Any]:
  """Compile the exact direct candidate and return its real PROGRAM/evidence.

  The exact LDS candidate is intentionally not silently substituted here: its
  current payload is not yet bound to the existing generated LDS executable
  route.  Callers receive a clear blocker instead of a mismatched comparison.
  """
  if transport != "direct_l2":
    raise NotImplementedError("exact LDS executable binding is pending generated-route integration")
  pair = generate_pair()
  candidate_row = pair["candidates"][transport]
  admission = admit_full_kernel_candidate(candidate_row["payload"], candidate_row["canonical_identity"],
                                          profile=PROFILE, role=ROLE, shape=SHAPE, target=TARGET)
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
          "candidate": candidate_row["payload"], "canonical_identity": candidate_row["canonical_identity"],
          "program": program, "compile_evidence": evidence, "dispatch_performed": False}


__all__ = ["compile_attn_qo_program"]
