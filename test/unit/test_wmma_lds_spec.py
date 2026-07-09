from dataclasses import replace
import json

from extra.qk.prefill_schedule_spec import PrefillGEMMScheduleSpec, emit_prefill_gemm_from_spec
from extra.qk.wmma_lds_spec import (
  DBUFEpochPrimitive, LDS2_OWNERSHIP_CLASSIFICATION, WMMALDSSpec, extract_wmma_lds_spec, lower_wmma_lds_spec,
  wmma_lds_generated_env_defaults, wmma_lds_lowering_insertion_point, wmma_lds_postrange_opts,
  wmma_lds_slot_identity_proof)


def _prefill_spec(route_family: str = "lds", *, out_f: int = 12288, in_f: int = 4096) -> PrefillGEMMScheduleSpec:
  return PrefillGEMMScheduleSpec(
    m=512, n=out_f, k=in_f, route_family=route_family, tile_m=128, tile_n=128, tile_k=32,
    waves_m=4, waves_n=2, wm=2, wn=4, pipe_tm=2, pipe_tn=2, pipeline_depth=2, threads=256,
    dbuf=1, plra=0, plrab=1, pad=16, leanaddr=0, role="ffn_gate_up")


def test_extract_wmma_lds_spec_from_ffn_gate_up_schedule():
  spec = extract_wmma_lds_spec(_prefill_spec())

  assert spec is not None
  assert spec.m == 512 and spec.n == 12288 and spec.k == 4096
  assert spec.tile_m == 128 and spec.tile_n == 128 and spec.tile_k == 32
  assert spec.threads == 256
  assert spec.k_substeps == 2
  assert spec.row_stride == 64
  assert spec.loads_a == 2
  assert spec.loads_b == 2
  assert spec.lds_buffers == 2
  assert spec.lds_total_bytes == 40960
  assert spec.plr_mode == "A+B"
  assert spec.legality_errors() == []


def test_wmma_lds_spec_owns_s9_s10_lifecycle_surface():
  spec = WMMALDSSpec.from_prefill_schedule(_prefill_spec())
  assert spec is not None

  assert spec.reg_layout.accumulator == "wmma_accum_wm_x_wn_8_vgprs"
  assert spec.memory_layout.lds_store == "packed_ds_store_b128"
  assert spec.wait.name == "vmem_to_lds_then_lgkm_to_wmma"
  assert spec.cadence.buffers == 2
  assert spec.lifecycle.backend_atom == "asm_backend_atom"
  assert spec.dbuf_epoch_primitive.owner == "hand_coded_backend_primitive"
  assert spec.dbuf_epoch_primitive.nbuf == 2
  assert spec.dbuf_epoch_primitive.slot_expr == "epoch % 2"
  assert spec.selection_label == "S9_COMPLETE_KEEP_OPT_IN"
  assert spec.ownership_classification() == LDS2_OWNERSHIP_CLASSIFICATION
  assert spec.ownership_classification() == "compiler_primitive_spec_owned__asm_backend_atom"


def test_wmma_lds_spec_json_roundtrip_keeps_lifecycle_data():
  spec = WMMALDSSpec.from_prefill_schedule(_prefill_spec())
  assert spec is not None

  payload = json.loads(json.dumps(spec.to_json()))
  restored = WMMALDSSpec.from_json(payload)
  restored_from_text = WMMALDSSpec.from_json(json.dumps(spec.to_json()))

  assert restored == spec
  assert restored_from_text == spec
  assert restored.to_json()["ownership_classification"] == "compiler_primitive_spec_owned__asm_backend_atom"
  assert restored.to_json()["selection_label"] == "S9_COMPLETE_KEEP_OPT_IN"
  assert restored.to_json()["reg_layout"] == spec.reg_layout.to_json()
  assert restored.to_json()["memory_layout"] == spec.memory_layout.to_json()
  assert restored.to_json()["wait"] == spec.wait.to_json()
  assert restored.to_json()["cadence"] == spec.cadence.to_json()
  assert restored.to_json()["lifecycle"] == spec.lifecycle.to_json()
  assert restored.to_json()["dbuf_epoch_primitive"] == spec.dbuf_epoch_primitive.to_json()


def test_dbuf_epoch_primitive_is_narrow_hand_coded_boundary():
  primitive = DBUFEpochPrimitive()
  payload = primitive.to_json()
  restored = DBUFEpochPrimitive.from_json(payload)

  assert restored == primitive
  assert payload["classification"] == "hand_coded_dbuf_epoch_primitive"
  assert payload["owner"] == "hand_coded_backend_primitive"
  assert payload["nbuf"] == 2
  assert payload["slot_expr"] == "epoch % 2"
  assert "fixed registers" not in payload["reusable_contract"]


def test_extract_wmma_lds_spec_rejects_non_lds_or_illegal_schedule():
  assert extract_wmma_lds_spec(_prefill_spec("pipe")) is None
  illegal = replace(_prefill_spec(), tile_k=48, plrab=1)
  assert extract_wmma_lds_spec(illegal) is None


def test_wmma_lds_insertion_point_targets_spec_lowering_not_raw_lists():
  point = wmma_lds_lowering_insertion_point()

  assert point["first_generated_diversion"] == "extra/qk/prefill_schedule_spec.py::emit_prefill_gemm_from_spec"
  assert point["diversion_predicate"] == 'PrefillGEMMScheduleSpec.route_family == "lds"'
  assert point["oracle_role"] == "ffn_gate_up"
  assert "build_gemm_lds2" in point["current_raw_lowering"]
  assert any("postrange.py" in item for item in point["reuse_existing_substrate"])
  assert any("LDSAddr" in item for item in point["reuse_existing_substrate"])
  assert any("native_isa_l4_stream_probe.py" in item for item in point["reuse_existing_substrate"])
  assert point["generated_transport"] == "extra/qk/prefill_graph_gemm_route.py::route_pf16_graph_gemm -> ordinary generated matmul"
  assert point["generated_transport_env"] == "extra/qk/wmma_lds_spec.py::wmma_lds_generated_env_defaults"
  assert point["generated_transport_opts"] == "extra/qk/wmma_lds_spec.py::wmma_lds_postrange_opts"
  assert any("build_gemm_lds2 instruction list" in item for item in point["do_not_copy"])
  assert any("UOp(Ops.INS" in item for item in point["do_not_copy"])
  assert any("parallel LDS lowering" in item for item in point["do_not_copy"])


def test_lower_wmma_lds_spec_fails_closed_without_calling_build_gemm_lds2(monkeypatch):
  calls = []

  from extra.qk.prefill import wmma
  monkeypatch.setattr(wmma, "build_gemm_lds2", lambda *args, **kwargs: calls.append((args, kwargs)))

  spec = extract_wmma_lds_spec(_prefill_spec())
  assert spec is not None

  try:
    lower_wmma_lds_spec(spec)
  except NotImplementedError as exc:
    message = str(exc)
  else:
    raise AssertionError("lower_wmma_lds_spec must fail closed until generated backend lowering exists")

  assert calls == []
  assert "Generated LDS WMMA primitive lowering is not implemented yet" in message
  assert "does not call extra.qk.prefill.wmma.build_gemm_lds2" in message


def test_lower_wmma_lds_spec_rejects_non_spec_contract():
  try:
    lower_wmma_lds_spec(object())  # type: ignore[arg-type]
  except TypeError as exc:
    assert "expected WMMALDSSpec" in str(exc)
  else:
    raise AssertionError("lower_wmma_lds_spec must only accept WMMALDSSpec")


def test_wmma_lds_generated_transport_reuses_existing_single_buffer_substrate():
  spec = extract_wmma_lds_spec(_prefill_spec())
  assert spec is not None

  env = wmma_lds_generated_env_defaults(spec)
  assert env["PREFILL_TC_LOCAL_STAGE"] == "both"
  assert env["PREFILL_TC_LOCAL_STAGE_WITH_LOCAL"] == "1"
  assert env["PREFILL_TC_LOCAL_STAGE_B_TILEKEY"] == "1"
  assert env["PREFILL_LDS_PACK_WITHLOCAL_B128"] == "1"
  assert env["AMD_ISA_WMMA_B128_FRAG"] == "1"
  assert "PREFILL_DBUF" not in env

  opts = wmma_lds_postrange_opts(spec)
  assert [o.op.name for o in opts] == ["TC", "UPCAST", "UPCAST", "UNROLL"]
  assert opts[1].arg == spec.wm
  assert opts[2].arg == spec.wn


def test_wmma_lds_slot_identity_proof_for_generated_single_buffer():
  spec = extract_wmma_lds_spec(_prefill_spec())
  assert spec is not None

  proof = wmma_lds_slot_identity_proof(spec, active_buffers=1)

  assert proof["ok"] is True
  assert proof["active_buffers"] == 1
  assert proof["spec_dbuf"] == 1
  assert proof["materialized_offsets_baseline"] is True
  assert proof["ds_immediate_folding_required"] is False
  assert proof["active_lds_bytes"] == 20480
  assert proof["vectors_by_operand_per_buffer"] == {"A": 640, "B": 640}
  assert proof["expected_stage_vectors_per_buffer"] == 1280
  assert proof["dbuf_cadence_proven"] is False
  assert [(w["operand"], w["buffer"], w["base"], w["end"]) for w in proof["windows"]] == [
    ("A", 0, 0, 10240),
    ("B", 0, 10240, 20480),
  ]


def test_wmma_lds_slot_identity_proof_can_cover_dbuf_but_not_cadence():
  spec = extract_wmma_lds_spec(_prefill_spec())
  assert spec is not None

  proof = wmma_lds_slot_identity_proof(spec, active_buffers=2)

  assert proof["ok"] is True
  assert proof["active_lds_bytes"] == 40960
  assert proof["dbuf_slot_identity_proven"] is True
  assert proof["dbuf_cadence_proven"] is False
  assert [(w["operand"], w["buffer"], w["base"], w["end"]) for w in proof["windows"]] == [
    ("A", 0, 0, 10240),
    ("B", 0, 10240, 20480),
    ("A", 1, 20480, 30720),
    ("B", 1, 30720, 40960),
  ]


def test_emit_prefill_gemm_opt_in_lds_routes_to_lowerer(monkeypatch):
  from extra.qk import prefill_graph_gemm_route as route
  from extra.qk import wmma_lds_spec

  lowered = []

  def fake_lower(lds_spec):
    lowered.append(lds_spec)
    return ("generated_lds_stub", lds_spec.to_json())

  monkeypatch.setenv("PREFILL_WMMA_LDS_PRIMITIVE", "1")
  monkeypatch.delenv("PREFILL_WMMA_PIPE_PRIMITIVE", raising=False)
  monkeypatch.setattr(route, "_emit_schedule", lambda params, name: (_ for _ in ()).throw(AssertionError("_emit_schedule called")))
  monkeypatch.setattr(wmma_lds_spec, "lower_wmma_lds_spec", fake_lower)

  out = emit_prefill_gemm_from_spec(_prefill_spec())
  assert out[0] == "generated_lds_stub"
  assert lowered and lowered[0].n == 12288
  assert lowered[0].plr_mode == "A+B"


def test_emit_prefill_gemm_opt_in_lds_unsupported_falls_back_to_current_oracle(monkeypatch):
  from extra.qk import prefill_graph_gemm_route as route
  from extra.qk import wmma_lds_spec

  calls = []

  def fake_emit(params, name):
    calls.append((params["pipe_mode"], name))
    return ("fallback_emit",)

  monkeypatch.setenv("PREFILL_WMMA_LDS_PRIMITIVE", "1")
  monkeypatch.setattr(route, "_emit_schedule", fake_emit)
  monkeypatch.setattr(wmma_lds_spec, "lower_wmma_lds_spec", lambda spec: (_ for _ in ()).throw(AssertionError("unsupported diverted")))

  assert emit_prefill_gemm_from_spec(replace(_prefill_spec(), tile_k=48, plrab=1)) == ("fallback_emit",)
  assert calls == [(False, "prefill_gen_sched_gemm_512_12288_4096")]


def test_wmma_lds_spec_reports_legality_errors():
  spec = WMMALDSSpec(
    m=512, n=12288, k=4096, tile_m=128, tile_n=128, tile_k=32, waves_m=4, waves_n=2,
    wm=2, wn=4, threads=128, pad=4096, dbuf=1, plrab=1, dshalf=1)
  errors = spec.legality_errors()
  assert any("threads must equal" in x for x in errors)
  assert any("LDS overflow" in x for x in errors)
  assert any("dshalf" in x for x in errors)
