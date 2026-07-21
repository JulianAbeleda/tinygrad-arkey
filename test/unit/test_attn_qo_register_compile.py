import json

from tinygrad import Tensor, dtypes
from tinygrad.codegen import to_program_cache
from tinygrad.codegen.opt import Opt, OptOps, postrange
from tinygrad.engine.realize import compile_linear
from tinygrad.helpers import Context, getenv
from tinygrad.uop.ops import Ops
from tinygrad.renderer.isa.amd import AMDOps

from test.unit.test_runtime_specs import _single_buffer_anchor_candidate, _strict_full_kernel_candidate
from extra.qk.runtime_specs import admit_full_kernel_candidate


def test_attn_qo_register_prefill_compile_is_cpu_only_and_zero_lds():
  """Compile the production-shaped register/L2 path without ever realizing it."""
  payload = json.loads(json.dumps(_single_buffer_anchor_candidate().full_kernel_candidate))
  payload["workload"].update(role="attn_qo", shape={"m": 512, "n": 4096, "k": 4096})
  payload["applicability"]["roles"] = ["attn_qo"]
  payload["schedule"]["pipeline"].update(buffer_count=1, stage_count=2)
  payload["schedule"]["residency"]["resident"] = ["accumulator", "stage_ab_register"]
  for role in ("a", "b"): payload["schedule"]["cooperative_load"][role]["lane_mapping"] = "wave_contiguous_b128"
  payload["schedule"]["wmma"]["fragment_layout"] = "rdna3_wmma_f32_16x16x16_f16_register_static"
  candidate = _strict_full_kernel_candidate(full_kernel_candidate=payload)
  admission = admit_full_kernel_candidate(payload, candidate.canonical_identity,
    profile="qwen3_8b_q4k_m_gfx1100", role="attn_qo", shape=(512, 4096, 4096),
    target={"backend": "AMD", "arch": "gfx1100", "wave_size": 32})
  assert admission.pipeline_plan.storage.kind == "global_register_resident"
  assert admission.active_lds_bytes == 0

  key = postrange.warmstart_key({512, 4096}, 4096)
  old_opts, old_contexts = postrange._WARMSTART_OPTS, postrange._WARMSTART_CANDIDATE_CONTEXTS
  try:
    postrange._WARMSTART_OPTS = {**(old_opts or {}), key: (Opt(OptOps.TC, 0, (-1, 2, 1)),)}
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
  selected = list(programs[0].src[0].toposort())
  stores = [u for u in selected if u.op is Ops.INS and u.arg is AMDOps.GLOBAL_STORE]
  assert len(stores) == 64
  output_stores = [st for st in stores if st.src[0].arg is AMDOps.V_OFFSET]
  assert len(output_stores) == 64
  assert len([u for u in selected if u.op is Ops.INS and u.arg is AMDOps.V_WMMA]) == 16
  assert not any(u.op is Ops.INS and u.arg in (AMDOps.STAGE_READ, AMDOps.STAGE_WRITE) for u in selected)
  assert not any(u.op is Ops.DEFINE_LOCAL or (u.op is Ops.INS and u.arg in
    (AMDOps.DS_LOAD, AMDOps.DS_STORE, AMDOps.DS_LOAD_B128, AMDOps.DS_STORE_B128)) for u in selected)
  emitted = " ".join(repr(x) for x in programs[0].toposort())
  assert "DEFINE_LOCAL" not in emitted and "DEFINE_ACC" not in emitted
  assert "ds_load" not in emitted.lower() and "ds_store" not in emitted.lower()
  attachments = [x.record for x in programs[0].arg.aux if hasattr(x, "record")]
  assert len(attachments) == 1
  ab = [x for x in attachments[0]["allocator"]["leases"] if x["role"] in ("A", "B")]
  assert [x["purpose"] for x in ab] == ["direct_wmma_fragment", "direct_wmma_fragment"]
  assert all(x["slots"] == 1 for x in ab)
