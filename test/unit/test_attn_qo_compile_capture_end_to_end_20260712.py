"""CPU-only integration boundary for the exact gfx1100 attn_qo program."""

import hashlib

import pytest

from tinygrad import Tensor, dtypes
from tinygrad.codegen import to_program_cache
from tinygrad.codegen.opt import postrange
from tinygrad.engine.realize import compile_linear
from tinygrad.helpers import Context, getenv
from tinygrad.uop.ops import Ops, ProgramInfo

from extra.qk.prefill.pure_register_compile_capture import capture_final_program_compile_only
from extra.qk.prefill.attn_qo_executable_preparation import compile_attn_qo_program
from extra.qk.prefill_graph_gemm_route import _candidate_schedule_spec, _primitive_warmstart_key
from extra.qk.prefill_schedule_spec import describe_prefill_schedule, register_resident_postrange_opts
from extra.qk.runtime_specs import admit_full_kernel_candidate
from test.unit.test_runtime_specs import _single_buffer_anchor_candidate, _strict_full_kernel_candidate


def test_storage_aware_direct_candidate_compiles_wave_owned_resident_fragments():
  compiled = compile_attn_qo_program(transport="direct_l2")
  program, evidence = compiled["program"], compiled["compile_evidence"]
  record = next(x.record for x in program.arg.aux if hasattr(x, "record"))
  resources = evidence["resource_artifact"]["resources"]
  source, disassembly = record["source"], record["disassembly"]
  assert (program.arg.global_size, program.arg.local_size) == ((128, 16, 1), (32, 1, 1))
  assert resources["workgroup_threads"] == resources["wavefront_size"] == 32
  assert resources["lds_bytes"] == resources["scratch_bytes"] == resources["vgpr_spills"] == 0
  assert resources["vgpr"] <= 112
  assert source.count("global_load_b128") == 16 and source.count("v_wmma") == 8
  assert "ds_load" not in source and "ds_store" not in source and "s_barrier" not in source
  assert "s_waitcnt vmcnt(8)" in disassembly and "s_waitcnt vmcnt(0) lgkmcnt(63)" in disassembly
  ab = [x for x in evidence["resource_artifact"]["physical_intervals"] if x["logical_role"] in ("A", "B")]
  assert len(ab) == 8 and all(x["purpose"] == "direct_wmma_fragment" for x in ab)


def test_exact_attn_qo_compile_capture_uses_real_program_info_attachment():
  payload = _single_buffer_anchor_candidate().full_kernel_candidate
  payload = {**payload, "workload": {**payload["workload"], "role": "attn_qo",
    "shape": {"m": 512, "n": 4096, "k": 4096}},
    "applicability": {**payload["applicability"], "roles": ["attn_qo"]},
    "schedule": {**payload["schedule"], "pipeline": {**payload["schedule"]["pipeline"],
      "buffer_count": 1, "stage_count": 2},
      "residency": {**payload["schedule"]["residency"], "resident": ["accumulator", "stage_ab_register"]},
      "cooperative_load": {r: {**payload["schedule"]["cooperative_load"][r], "lane_mapping": "wave_contiguous_b128"}
                           for r in ("a", "b")},
      "wmma": {**payload["schedule"]["wmma"],
        "fragment_layout": "rdna3_wmma_f32_16x16x16_f16_register_static"}}}
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

  programs = [u for u in compiled.toposort() if u.op is Ops.PROGRAM and isinstance(u.arg, ProgramInfo)
              and getattr(getattr(u.src[0].arg, "candidate_context", None), "canonical_identity", None)
              == candidate.canonical_identity]
  assert len(programs) == 1
  info = programs[0].arg
  attachments = [x.record for x in info.aux if hasattr(x, "record")]
  if not attachments:
    with pytest.raises(ValueError, match="final program lacks descriptor mapping"):
      capture_final_program_compile_only(info, pipeline={"storage_kind": "global_register_resident"},
        wait={"typed": True}, abi_contract={"wave_size": 32}, surface={"strict_pure": True})
    return

  artifact = capture_final_program_compile_only(attachments[-1], pipeline={"storage_kind": "global_register_resident"},
    wait={"typed": True}, abi_contract={"wave_size": 32}, surface={"strict_pure": True})
  assert artifact["passed"] is True
  assert artifact["capture"]["dispatch_permitted"] is False
  assert artifact["program"]["binary_sha256"] == hashlib.sha256(next(u.arg for u in programs[0].src if u.op is Ops.BINARY)).hexdigest()
  assert attachments[-1]["source"] and attachments[-1]["disassembly"]
  leases = attachments[-1]["allocator"]["leases"]
  assert {lease["role"] for lease in leases} >= {"A", "B", "C"}
  assert all(lease["fixed"] is True and lease["lifetime"] for lease in leases)
  assert artifact["resource_artifact"]["resources"]["scratch_bytes"] == 0
  assert artifact["resource_artifact"]["resources"]["vgpr_spills"] == 0
  assert artifact["resource_artifact"]["resources"]["sgpr_spills"] == 0
