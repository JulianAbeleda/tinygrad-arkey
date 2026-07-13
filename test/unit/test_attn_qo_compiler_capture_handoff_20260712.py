"""CPU-only regression for the compiler-owned exact attn_qo capture handoff."""

import json

from tinygrad import Tensor, dtypes
from tinygrad.codegen import to_program_cache
from tinygrad.codegen.opt import postrange
from tinygrad.engine.realize import compile_linear
from tinygrad.helpers import Context, getenv
from tinygrad.renderer.isa import CompilerCaptureProof
from tinygrad.uop.ops import Ops

from extra.qk.prefill.pure_register_compile_capture import capture_final_program_compile_only
from extra.qk.prefill_graph_gemm_route import _candidate_schedule_spec, _primitive_warmstart_key
from extra.qk.prefill_schedule_spec import describe_prefill_schedule, register_resident_postrange_opts
from extra.qk.runtime_specs import admit_full_kernel_candidate
from test.unit.test_runtime_specs import _single_buffer_anchor_candidate, _strict_full_kernel_candidate


def test_exact_attn_qo_compiler_capture_handoff_is_adapter_compatible_without_dispatch():
  payload = json.loads(json.dumps(_single_buffer_anchor_candidate().full_kernel_candidate))
  payload["workload"].update(role="attn_qo", shape={"m": 512, "n": 4096, "k": 4096})
  payload["applicability"]["roles"] = ["attn_qo"]
  payload["schedule"]["pipeline"].update(buffer_count=1, stage_count=2)
  payload["schedule"]["residency"]["resident"] = ["accumulator", "stage_ab_register"]
  payload["schedule"]["wmma"]["fragment_layout"] = "rdna3_wmma_f32_16x16x16_f16_register_static"
  candidate = _strict_full_kernel_candidate(full_kernel_candidate=payload)
  admission = admit_full_kernel_candidate(payload, candidate.canonical_identity,
    profile="qwen3_8b_q4k_m_gfx1100", role="attn_qo", shape=(512, 4096, 4096),
    target={"backend": "AMD", "arch": "gfx1100", "wave_size": 32})
  spec = describe_prefill_schedule(4096, 4096, role="attn_qo")
  candidate_spec = _candidate_schedule_spec(spec, admission)
  key = _primitive_warmstart_key(candidate_spec)
  old_opts, old_contexts = postrange._WARMSTART_OPTS, postrange._WARMSTART_CANDIDATE_CONTEXTS
  try:
    postrange._WARMSTART_OPTS = {**(old_opts or {}), key: register_resident_postrange_opts(candidate_spec)}
    postrange._WARMSTART_CANDIDATE_CONTEXTS = {**(old_contexts or {}), key: admission.context}
    getenv.cache_clear(); to_program_cache.clear()
    with Context(DEV="AMD:ISA:gfx1100"):
      a = Tensor.empty(512, 4096, dtype=dtypes.half)
      b = Tensor.empty(4096, 4096, dtype=dtypes.half)
      compiled = compile_linear((a @ b.transpose()).schedule_linear())
  finally:
    postrange._WARMSTART_OPTS, postrange._WARMSTART_CANDIDATE_CONTEXTS = old_opts, old_contexts
    getenv.cache_clear(); to_program_cache.clear()

  programs = [u for u in compiled.toposort() if u.op is Ops.PROGRAM and
              getattr(getattr(u.src[0].arg, "candidate_context", None), "canonical_identity", None) == candidate.canonical_identity]
  assert len(programs) == 1
  program = programs[0]
  linear = next(u for u in program.src if u.op is Ops.LINEAR)
  assert isinstance(linear.arg, CompilerCaptureProof) and hash(linear.arg)
  assert {x.logical_role for x in linear.arg.leases} == {"A", "B", "C"}
  record = next(x.record for x in program.arg.aux if hasattr(x, "record"))
  assert record["binary"] == next(u.arg for u in program.src if u.op is Ops.BINARY)
  assert record["descriptor"]["resources"]["scratch_bytes"] == 0
  assert record["descriptor"]["resources"]["vgpr_spills"] == 0
  assert record["descriptor"]["resources"]["sgpr_spills"] == 0
  artifact = capture_final_program_compile_only(record, pipeline={"storage_kind": "global_register_resident"},
    wait={"typed": True}, abi_contract={"wave_size": 32}, surface={"strict_pure": True})
  assert artifact["passed"] is True
  assert artifact["capture"]["dispatch_permitted"] is False
