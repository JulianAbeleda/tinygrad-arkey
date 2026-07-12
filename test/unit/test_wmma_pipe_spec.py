from extra.qk.prefill_schedule_spec import PrefillGEMMScheduleSpec
from extra.qk.wmma_pipe_spec import (
  WMMAPipeSpec, WMMAPipeIR, build_wmma_pipe_diagnostic_lowering_report, extract_wmma_pipe_spec, lower_wmma_pipe_spec,
  pipe_primitive_local_stage_resource_plan, wmma_pipe_lowering_insertion_point)

def test_pipe_spec_rejects_non_divisible_shape_and_ir_output_contract():
  try:
    WMMAPipeSpec(m=513, n=4096, k=4096, tile_m=128, tile_n=128, role="attn_qo")
  except ValueError as exc:
    assert "divisible" in str(exc)
  else:
    raise AssertionError("non-divisible pipe shape must fail closed")
  try:
    WMMAPipeIR("attn_qo", (512, 4096, 4096), 2, 8, "targeted_vmcnt", stores="fp32_global")
  except ValueError as exc:
    assert "output dtype" in str(exc)
  else:
    raise AssertionError("unsupported output contract must fail closed")

def test_pipe_op_lifecycle_derives_wait_and_rejects_unconsumed_slot():
  from extra.qk.wmma_pipe_spec import WMMAPipeOp, build_wmma_pipe_ir
  spec = WMMAPipeSpec(m=512, n=4096, k=4096, tile_m=128, tile_n=128, role="attn_qo")
  op = WMMAPipeOp(build_wmma_pipe_ir(spec), 0, 1, 2, (128, 4, 1), (256, 1, 1))
  assert op.derived_wait_vmcnt == 8
  try:
    WMMAPipeOp(build_wmma_pipe_ir(spec), 0, 1, 2, (128, 4, 1), (256, 1, 1),
               lifecycle=(("produce", 0, 0),))
  except ValueError as exc:
    assert "without a consume" in str(exc)
  else:
    raise AssertionError("unconsumed slot must fail closed")

def test_pipe_op_resource_estimate_is_explicit_about_unknown_registers():
  from extra.qk.wmma_pipe_spec import WMMAPipeOp, build_wmma_pipe_ir
  spec = WMMAPipeSpec(m=512, n=4096, k=4096, tile_m=128, tile_n=128, role="attn_qo")
  op = WMMAPipeOp(build_wmma_pipe_ir(spec), 0, 1, 2, (256, 1, 1), (64, 1, 1), slot_bytes=20480)
  res = op.resource_estimate()
  assert res["lds_bytes"] == 40960 and res["vgpr"] is None and res["scratch_bytes"] == 0

def test_pipe_spec_exposes_common_register_policy():
  spec = WMMAPipeSpec(m=512, n=4096, k=4096, tile_m=128, tile_n=128, role="attn_qo")
  policy = spec.pipeline_policy
  assert policy.storage_kind == "global_register_resident"
  assert policy.logical_stage_count == 2 and policy.resources.lds_bytes == 0

def test_pipe_candidate_context_is_identity_and_abi_complete():
  from extra.qk.wmma_pipe_spec import pipe_candidate_context
  spec = WMMAPipeSpec(m=512, n=4096, k=4096, tile_m=128, tile_n=128, role="attn_qo")
  ctx = pipe_candidate_context(spec, "a" * 64)
  assert ctx.canonical_identity == "a" * 64
  payload = dict(ctx.pipeline)
  assert payload["schema"] == "wmma_pipe_ir.v1"
  assert payload["role"] == "attn_qo" and payload["shape"] == (512, 4096, 4096)
  assert payload["stages"] == 2 and payload["wait_policy"] == "targeted_vmcnt"
  assert payload["stores"] == "fp16_global"

def test_pipe_lifecycle_rejects_overwrite_and_tail_shape():
  from extra.qk.wmma_pipe_spec import WMMAPipeOp, build_wmma_pipe_ir
  spec = WMMAPipeSpec(m=512, n=4096, k=4096, tile_m=128, tile_n=128, role="attn_qo")
  with_overwrite = (("produce", 0, 0), ("produce", 1, 0), ("consume", 0, 0), ("consume", 1, 0))
  try:
    WMMAPipeOp(build_wmma_pipe_ir(spec), 0, 1, 2, (256, 1, 1), (64, 1, 1), lifecycle=with_overwrite)
  except ValueError as exc:
    assert "overwritten" in str(exc)
  else:
    raise AssertionError("slot overwrite before consume must fail closed")
  try:
    WMMAPipeSpec(m=512, n=4096, k=4100, tile_m=128, tile_n=128, role="attn_qo")
  except ValueError as exc:
    assert "divisible" in str(exc)
  else:
    raise AssertionError("non-divisible tail must fail closed")


def _prefill_spec(route_family: str = "pipe", *, n: int = 4096, role: str = "attn_qo",
                  pipeline_depth: int = 2, waitcnt_policy: str = "targeted_vmcnt"):
  return PrefillGEMMScheduleSpec(
    m=512, n=n, k=4096, route_family=route_family, tile_m=128, tile_n=128, tile_k=32,
    waves_m=4, waves_n=2, wm=2, wn=4, pipe_tm=2, pipe_tn=2, pipeline_depth=pipeline_depth,
    threads=256, dbuf=1, plra=0, plrab=1, pad=16, leanaddr=0, role=role,
    waitcnt_policy=waitcnt_policy)


def test_extract_wmma_pipe_spec_from_prefill_pipe_schedule():
  spec = extract_wmma_pipe_spec(_prefill_spec())

  assert spec is not None
  assert spec.m == 512 and spec.n == 4096 and spec.k == 4096
  assert spec.tile_m == 128 and spec.tile_n == 128
  assert spec.stages == 2
  assert spec.loads_per_stage == 8
  assert spec.to_json()["operand_a"] == "global_row_major_fp16"
  assert spec.to_json()["operand_b"] == "global_row_major_bt_fp16"
  assert spec.to_json()["role"] == "attn_qo"


def test_extract_wmma_pipe_spec_rejects_non_pipe_and_unknown_wait_policy():
  assert extract_wmma_pipe_spec(_prefill_spec("lds")) is None
  assert extract_wmma_pipe_spec(_prefill_spec(pipeline_depth=1)) is None
  assert extract_wmma_pipe_spec(_prefill_spec(waitcnt_policy="drain_all")) is None


def test_pipe_primitive_local_stage_resource_plan_allows_safe_roles():
  for role, n in (("attn_qo", 4096), ("ffn_down", 4096)):
    spec = extract_wmma_pipe_spec(_prefill_spec(n=n, role=role))
    assert spec is not None

    plan = pipe_primitive_local_stage_resource_plan(spec, local_stage_requested=True)

    assert plan["schema"] == "wmma-pipe-local-stage-resource-plan.v1"
    assert plan["gate"] == "s10_attn_kv_generated_pipe_local_stage_lds"
    assert plan["role"] == role
    assert plan["local_stage_requested"] is True
    assert plan["estimated_shared_bytes"] == 0
    assert plan["shared_arrays"] == []
    assert plan["safe"] is True
    assert plan["decision"] == "allow"
    assert plan["fallback_reason"] is None


def test_pipe_primitive_local_stage_resource_plan_blocks_s10_attn_kv_over_lds_limit():
  spec = extract_wmma_pipe_spec(_prefill_spec(n=1024, role="attn_kv"))
  assert spec is not None

  plan = pipe_primitive_local_stage_resource_plan(spec, local_stage_requested=True,
                                                  allow_attn_kv_no_local_stage=False)

  assert plan["gate"] == "s10_attn_kv_generated_pipe_local_stage_lds"
  assert plan["role"] == "attn_kv"
  assert plan["lds_limit_bytes"] == 65536
  assert plan["estimated_shared_bytes"] == 69632
  assert plan["shared_arrays"] == [
    {"name": "buf0", "type": "half", "elements": 2048, "bytes": 4096},
    {"name": "buf2", "type": "half", "elements": 32768, "bytes": 65536},
  ]
  assert plan["safe"] is False
  assert plan["decision"] == "fallback"
  assert "s10_attn_kv_generated_pipe_local_stage_lds" in plan["fallback_reason"]
  assert "role=attn_kv" in plan["fallback_reason"]
  assert "69632 bytes LDS > 65536" in plan["fallback_reason"]
  assert "decision=fallback" in plan["fallback_reason"]


def test_pipe_primitive_local_stage_resource_plan_selects_attn_kv_no_local_stage_policy():
  spec = extract_wmma_pipe_spec(_prefill_spec(n=1024, role="attn_kv"))
  assert spec is not None

  plan = pipe_primitive_local_stage_resource_plan(spec, local_stage_requested=True)

  assert plan["role"] == "attn_kv"
  assert plan["local_stage_requested"] is True
  assert plan["allow_attn_kv_no_local_stage"] is True
  assert plan["no_local_stage_selected"] is True
  assert plan["overflow_estimated_shared_bytes"] == 69632
  assert plan["estimated_shared_bytes"] == 0
  assert plan["shared_arrays"] == []
  assert plan["safe"] is True
  assert plan["decision"] == "generated_pipe_no_local_stage"
  assert plan["fallback_reason"] is None


def test_pipe_primitive_local_stage_resource_plan_allows_attn_kv_when_local_stage_disabled():
  spec = extract_wmma_pipe_spec(_prefill_spec(n=1024, role="attn_kv"))
  assert spec is not None

  plan = pipe_primitive_local_stage_resource_plan(spec, local_stage_requested=False)

  assert plan["role"] == "attn_kv"
  assert plan["local_stage_requested"] is False
  assert plan["estimated_shared_bytes"] == 0
  assert plan["safe"] is True
  assert plan["decision"] == "allow"
  assert plan["fallback_reason"] is None


def test_wmma_pipe_insertion_point_targets_spec_lowering_not_route_raw_lists():
  point = wmma_pipe_lowering_insertion_point()

  assert point["first_generated_diversion"] == "extra/qk/prefill_schedule_spec.py::emit_prefill_gemm_from_spec"
  assert point["diversion_predicate"] == 'PrefillGEMMScheduleSpec.route_family == "pipe"'
  assert "build_gemm_pipe" in point["current_raw_lowering"]
  assert point["diagnostic_lowerer"] == "extra/qk/wmma_pipe_spec.py::build_wmma_pipe_diagnostic_lowering_report"
  assert any("build_gemm_pipe instruction list" in item for item in point["do_not_copy"])
  assert any("UOp(Ops.INS" in item for item in point["do_not_copy"])


def test_lower_wmma_pipe_spec_fails_closed_without_calling_build_gemm_pipe(monkeypatch):
  calls = []

  def _build_gemm_pipe_oracle(*args, **kwargs):
    calls.append((args, kwargs))
    raise AssertionError("lower_wmma_pipe_spec must not call build_gemm_pipe")

  from extra.qk.prefill import wmma
  monkeypatch.setattr(wmma, "build_gemm_pipe", _build_gemm_pipe_oracle)

  spec = extract_wmma_pipe_spec(_prefill_spec())
  assert spec is not None

  try:
    lower_wmma_pipe_spec(spec)
  except NotImplementedError as exc:
    message = str(exc)
  else:
    raise AssertionError("lower_wmma_pipe_spec must fail closed until generated backend lowering exists")

  assert calls == []
  assert "Generated WMMA pipe primitive lowering is not implemented yet" in message
  assert "does not call extra.qk.prefill.wmma.build_gemm_pipe" in message


def test_lower_wmma_pipe_spec_rejects_unsupported_spec_without_oracle_fallback(monkeypatch):
  calls = []

  from extra.qk.prefill import wmma
  monkeypatch.setattr(wmma, "build_gemm_pipe", lambda *args, **kwargs: calls.append((args, kwargs)))

  spec = WMMAPipeSpec(
    m=512, n=4096, k=4096, tile_m=128, tile_n=128, stages=1, wait_policy="drain_all")

  try:
    lower_wmma_pipe_spec(spec)
  except NotImplementedError as exc:
    message = str(exc)
  else:
    raise AssertionError("unsupported WMMA pipe specs must fail closed")

  assert calls == []
  assert "unsupported pipe specs" in message
  assert "No fallback to extra.qk.prefill.wmma.build_gemm_pipe was attempted" in message


def test_lower_wmma_pipe_spec_rejects_non_spec_contract():
  try:
    lower_wmma_pipe_spec(object())  # type: ignore[arg-type]
  except TypeError as exc:
    assert "expected WMMAPipeSpec" in str(exc)
  else:
    raise AssertionError("lower_wmma_pipe_spec must only accept WMMAPipeSpec")


def test_build_wmma_pipe_diagnostic_lowering_report_proves_generated_core_structure(monkeypatch):
  calls = []

  from extra.qk.prefill import wmma
  monkeypatch.setattr(wmma, "build_gemm_pipe", lambda *args, **kwargs: calls.append((args, kwargs)))

  spec = WMMAPipeSpec(m=64, n=64, k=64, tile_m=64, tile_n=64, pipe_tm=2, pipe_tn=2)
  report = build_wmma_pipe_diagnostic_lowering_report(spec, unr=2)

  assert calls == []
  assert report["schema"] == "wmma-pipe-diagnostic-lowering.v1"
  assert report["transport"] == "generated_program_diagnostic"
  assert report["route_bound"] is False
  assert report["uses_hand_pipe_oracle"] is False
  assert report["uses_route_local_full_ops_ins"] is False
  assert report["resident_ab"] is True
  assert report["warmstart"]["apply"] > 0
  assert report["track_counts"]["global_load_b128"] > 0
  assert report["track_counts"]["v_wmma_f32_16x16x16_f16"] > 0
  assert report["track_counts"]["global_store_b16"] > 0
  assert report["track_counts"]["global_load_u16"] == 0
  assert report["waitcnt_summary"]["count"] > 0
  assert report["mvp_core_structure_ok"] is True
  assert report["mvp_pipe_wait_ok"] is True
  assert report["mvp_structure_ok"] is True
  assert report["waitcnt_summary"]["has_expected_pipe_vmcnt"] is True
  assert 8 in report["waitcnt_summary"]["vmcnt_sequence"]
  assert "route transport" in report["next_blocker"]
