import json, os, subprocess, sys
from pathlib import Path

from extra.qk import pure_kernel_surface_audit as surface_audit
from extra.qk.prefill import kernel_lifecycle_trace as life
from extra.qk.prefill import native_isa_l4_stream_probe as sp
from tinygrad.llm.generated_candidates import select_generated_candidate
from tinygrad.llm.quant_specs import activation_spec, quant_spec
from tinygrad.llm.runtime_specs import RuntimeOpSpec

ROOT = Path(__file__).resolve().parents[2]


def assert_prefill_mvp_structural_provenance_gate(report, surface_row, *, route_id):
  assert report.get("ok", True) is True
  assert report["tail_off"].startswith("generated")
  assert "builder" not in report
  assert report["shared_floor"] == "Inst list -> assemble_linear -> ELF -> AMDProgram/HSA launch -> GPU"

  tc = report["track_counts"]
  wmma_count = tc["v_wmma_f32_16x16x16_f16"]
  assert tc["global_load_b128"] > 0
  assert tc["global_load_u16"] == 0
  assert wmma_count > 0
  assert report["wmma_operand_origin_counts"] == {"global_load_b128/global_load_b128": wmma_count}

  waitcnt = report["waitcnt_summary"]
  assert waitcnt["count"] > 0
  assert waitcnt["nonfull_count"] == waitcnt["count"]
  assert any(
    0 in row["vmcnt_sequence"] and 63 in row["lgkmcnt_sequence"]
    for row in report["waits_per_wmma"][1:]
  )

  assert surface_row["route_id"] == route_id
  assert surface_row["strict_pure"] is True
  assert surface_row["surface_class"] in {
    "ordinary_tinygrad_graph", "descriptor_owned_uop_codegen", "backend_owned_intrinsic_lowering",
  }
  assert surface_row["kernel_authorship"] != "hand_authored_full_kernel_schedule"
  assert surface_row["asm_usage"] != "raw_instruction_or_binary_injection"
  assert surface_row["surface_policy"] == "generated_default_allowed"
  assert surface_row["missing_writer_files"] == []


def test_attn_qo_generated_candidate_lifecycle_smoke_has_b128_wmma_targeted_waitcnt():
  op = RuntimeOpSpec(
    family="QuantizedLinear", phase="prefill", role="attn_qo", shape={"M": 512, "N": 4096, "K": 4096},
    weight=quant_spec("Q4_K").tensor_spec(), activation=activation_spec("fp16").activation_spec(),
  )
  selected = select_generated_candidate(op, preferred=("quant_linear_prefill.prefill_v2_scheduler_matmul_default",))
  assert selected.status == "selected"
  assert selected.candidate is not None
  assert selected.candidate.route_id == "prefill_v2_scheduler_matmul_default"

  env = {**os.environ,
    "DEV": "AMD:ISA",
    "AMD_ISA_SCHED": "1",
    "AMD_ISA_WAITCNT_TARGETED": "1",
    "AMD_ISA_WMMA_B128_FRAG": "1",
    "AMD_ISA_REG_ACCUM": "1",
    "PYTHONPATH": str(ROOT),
  }
  proc = subprocess.run([
    sys.executable, "extra/qk/prefill/kernel_lifecycle_trace.py",
    "--active-generated", "--shapes", "2,2", "--m", "512", "--n", "4096", "--k", "4096",
    "--loc", "0", "--unr", "2", "--json",
  ], cwd=ROOT, env=env, check=True, text=True, capture_output=True)
  report = json.loads(proc.stdout)

  row = surface_audit.route_surface_row(selected.candidate.route_id)
  assert_prefill_mvp_structural_provenance_gate(report, row, route_id="prefill_v2_scheduler_matmul_default")


def test_attn_qo_default_prefill_candidate_has_no_route_local_raw_instruction_list():
  row = surface_audit.route_surface_row("prefill_v2_scheduler_matmul_default")
  assert row["strict_pure"] is True
  assert row["surface_class"] == "ordinary_tinygrad_graph"
  assert row["kernel_authorship"] == "tinygrad_scheduler_generated"
  assert row["asm_usage"] == "backend_emitted_if_needed"
  assert row["surface_policy"] == "generated_default_allowed"


def test_s10_lds_route_trace_cli_writes_artifact_without_raw_oracle_classification(tmp_path):
  artifact = tmp_path / "route-trace.json"
  env = {**os.environ, "PYTHONPATH": str(ROOT), "PREFILL_WMMA_LDS_PRIMITIVE": "1"}
  proc = subprocess.run([
    sys.executable, "extra/qk/prefill/kernel_lifecycle_trace.py",
    "--s10-lds-route-trace", "--json", "--n", "12288", "--k", "4096",
    "--route-trace-out", str(artifact),
  ], cwd=ROOT, env=env, check=True, text=True, capture_output=True)

  stdout_report = json.loads(proc.stdout)
  file_report = json.loads(artifact.read_text())
  assert stdout_report == file_report
  assert file_report["selected_surface"] == "generated_transport"
  assert file_report["classification"] == "compiler_primitive_spec_owned__generated_transport"
  assert file_report["fallback_reason"] is None
  assert file_report["calls_build_gemm_lds2"] is False


def test_dbuf_pipeline_construction_audit_marks_prologue_body_overlap_not_redundancy():
  def row(idx, imm):
    return {"idx": idx, "spans": {"addr": {"kind": "v", "lo": 0, "hi": 0, "n": 1}},
            "text": f"ds_store_b128(v[0], v[1], v[2:5], v[0], {imm})"}

  audit = life._dbuf_pipeline_construction_audit({
    "ds_store_b128": [row(10, 0), row(20, 16), row(130, 0), row(150, 32)],
    "ds_load_b128": [{"idx": 105, "spans": {}, "text": "ds_load_b128(...)"},
                     {"idx": 140, "spans": {}, "text": "ds_load_b128(...)"}],
  }, [100, 120, 160])

  assert audit["verdict"] == "physical_window_overlap_requires_epoch_reaching_def"
  assert audit["store_counts"] == {"prologue": 2, "body": 2, "tail": 0}
  assert audit["prologue_body_physical_window_overlap_count"] == 1
  assert audit["body_loads_before_first_body_store_count"] == 1
  assert audit["warmup_required_overlap_count"] == 0
  assert audit["steady_state_body_produced_overlap_count"] == 1
  assert audit["pipeline_epoch_candidate"] is True
  assert "not a redundancy proof" in audit["note"]


def test_dbuf_pipeline_construction_audit_reports_source_mismatch_for_same_lds_window():
  def store(idx, imm):
    return {"idx": idx, "spans": {"addr": {"kind": "v", "lo": 0, "hi": 0, "n": 1},
                                  "data0": {"kind": "v", "lo": 232, "hi": 235, "n": 4}},
            "text": f"ds_store_b128(v[0], v[0], v[232:235], v[0], {imm})"}
  def gload(idx, sbase):
    return {"idx": idx, "spans": {"vdst": {"kind": "v", "lo": 232, "hi": 235, "n": 4},
                                  "addr": {"kind": "v", "lo": 85, "hi": 85, "n": 1},
                                  "saddr": {"kind": "s", "lo": sbase, "hi": sbase + 1, "n": 2}},
            "text": f"global_load_b128(v[232:235], v[85], v[0], s[{sbase}:{sbase+1}])"}

  audit = life._dbuf_pipeline_construction_audit({
    "global_load_b128": [gload(9, 8), gload(129, 10)],
    "ds_store_b128": [store(10, 48), store(130, 48)],
    "ds_load_b128": [],
  }, [100, 160])

  assert audit["prologue_body_physical_window_overlap_count"] == 1
  assert audit["prologue_body_source_mismatch_count"] == 1
  assert audit["prologue_body_source_mismatch_sample"][0]["prologue_sources"] == ["saddr=s8:9|vaddr=v85:85"]
  assert audit["prologue_body_source_mismatch_sample"][0]["body_sources"] == ["saddr=s10:11|vaddr=v85:85"]


def test_p7_lowered_stream_export_fails_closed_without_metadata():
  out = life._p7_lowered_stream_export({
    "ds_store_b128": [{"idx": 0}],
    "s_barrier": [{"idx": 1}],
    "ds_load_b128": [{"idx": 2}],
  }, {"covered_load_count": 1, "load_count": 1, "key_strength": "synthetic"})

  assert out["status"] == "fail_closed"
  assert "role/epoch/slot" in out["reason"]
  assert out["events"] == []


def test_p7_lowered_stream_export_reports_partial_metadata():
  out = life._p7_lowered_stream_export({
    "ds_store_b128": [{"idx": 0, "dbuf_partial": {"kind": "tc_local_stage_store", "role": "B"}}],
    "s_barrier": [{"idx": 1}],
    "ds_load_b128": [{"idx": 2, "dbuf_partial": {"kind": "wmma_frag_buffer_proof", "role": "B"}}],
  }, {"covered_load_count": 1, "load_count": 1, "key_strength": "synthetic"})

  assert out["status"] == "fail_closed"
  assert out["metadata_rows"] == 0
  assert out["partial_metadata_rows"] == 2
  assert out["partial_metadata_sample"][0]["kind"] == "tc_local_stage_store"


def test_side_channel_lifecycle_events_check_when_complete():
  out = life._side_channel_lifecycle_events([
    {"kind": "dbuf_lifecycle_event", "op": "produce", "role": "B", "epoch": 0, "slot": 0, "window": "B:slot0"},
    {"kind": "dbuf_lifecycle_event", "op": "barrier"},
    {"kind": "dbuf_lifecycle_event", "op": "consume", "role": "B", "epoch": 0, "slot": 0, "window": "B:slot0"},
  ])

  assert out["row_count"] == 3
  assert out["event_count"] == 3
  assert out["errors"] == []
  assert out["check"]["ok"] is True


def test_side_channel_lifecycle_events_reports_incomplete_rows():
  out = life._side_channel_lifecycle_events([
    {"kind": "dbuf_lifecycle_event", "op": "produce", "role": "B", "slot": 0},
  ])

  assert out["row_count"] == 1
  assert out["event_count"] == 0
  assert "missing=['epoch']" in out["errors"][0]["error"]


def test_p7_lowered_stream_export_carries_side_channel_fail_closed_context():
  side = life._side_channel_lifecycle_events([
    {"kind": "dbuf_lifecycle_event", "op": "produce", "role": "B", "epoch": 0, "slot": 0, "window": "B:slot0"},
    {"kind": "dbuf_lifecycle_event", "op": "barrier"},
    {"kind": "dbuf_lifecycle_event", "op": "consume", "role": "B", "epoch": 0, "slot": 0, "window": "B:slot0"},
  ])
  out = life._p7_lowered_stream_export({
    "ds_store_b128": [{"idx": 0}],
    "s_barrier": [{"idx": 1}],
    "ds_load_b128": [{"idx": 2}],
  }, {"covered_load_count": 1, "load_count": 1, "key_strength": "synthetic"}, side)

  assert out["status"] == "fail_closed"
  assert "side-channel records exist" in out["reason"]
  assert out["side_channel"]["check"]["ok"] is True
  assert out["reconciled_side_channel"]["errors"][0]["error"] == "side-channel row has no uop_id/inst_idx anchor"


def test_p7_lowered_stream_export_exports_reconciled_side_channel():
  side = life._side_channel_lifecycle_events([
    {"kind": "dbuf_lifecycle_event", "op": "produce", "role": "B", "epoch": 0, "slot": 0, "window": "B:slot0", "uop_id": 10},
    {"kind": "dbuf_lifecycle_event", "op": "barrier", "uop_id": 11},
    {"kind": "dbuf_lifecycle_event", "op": "consume", "role": "B", "epoch": 0, "slot": 0, "window": "B:slot0", "uop_id": 12},
  ])
  out = life._p7_lowered_stream_export({
    "ds_store_b128": [{"idx": 20, "uop_id": 10}],
    "s_barrier": [{"idx": 21, "uop_id": 11}],
    "ds_load_b128": [{"idx": 22, "uop_id": 12}],
  }, {"covered_load_count": 1, "load_count": 1, "key_strength": "synthetic"}, side)

  assert out["status"] == "exported"
  assert out["check"]["ok"] is True
  assert out["events"] == [
    {"op": "produce", "step": 20, "role": "B", "epoch": 0, "slot": 0, "window": "B:slot0"},
    {"op": "barrier", "step": 21},
    {"op": "consume", "step": 22, "role": "B", "epoch": 0, "slot": 0, "window": "B:slot0"},
  ]


def test_side_channel_reconciliation_rejects_wrong_physical_op():
  side = life._side_channel_lifecycle_events([
    {"kind": "dbuf_lifecycle_event", "op": "produce", "role": "B", "epoch": 0, "slot": 0, "uop_id": 12},
  ])

  out = life._reconcile_side_channel_to_rows({
    "ds_store_b128": [],
    "s_barrier": [],
    "ds_load_b128": [{"idx": 22, "uop_id": 12}],
  }, side)

  assert out["ok"] is False
  assert out["errors"][0]["error"] == "side-channel op 'produce' maps to physical 'consume'"


def test_side_channel_reconciliation_accepts_live_consume_anchor():
  side = life._side_channel_lifecycle_events([
    {"kind": "dbuf_lifecycle_event", "op": "produce", "role": "A", "epoch": "e0", "slot": "s0", "window": "A:slot0", "uop_id": 30},
    {"kind": "dbuf_lifecycle_event", "op": "barrier", "uop_id": 31},
    {"kind": "dbuf_lifecycle_event", "op": "consume", "role": "A", "epoch": "e0", "slot": "s0", "window": "A:slot0", "uop_id": 32},
  ])

  out = life._reconcile_side_channel_to_rows({
    "ds_store_b128": [{"idx": 40, "uop_id": 30}],
    "s_barrier": [{"idx": 41, "uop_id": 31}],
    "ds_load_b128": [{"idx": 42, "uop_id": 32}],
  }, side)

  assert out["ok"] is True
  assert out["events"][-1] == {"op": "consume", "step": 42, "role": "A", "epoch": "e0", "slot": "s0", "window": "A:slot0"}


def test_side_channel_reconciliation_follows_anchor_aliases():
  side = life._side_channel_lifecycle_events([
    {"kind": "dbuf_lifecycle_event", "op": "produce", "role": "A", "epoch": "e0", "slot": "s0", "window": "A:slot0", "uop_id": 30},
    {"kind": "dbuf_lifecycle_event", "op": "barrier", "uop_id": 31},
    {"kind": "dbuf_lifecycle_event", "op": "consume", "role": "A", "epoch": "e0", "slot": "s0", "window": "A:slot0", "uop_id": 32},
    {"kind": "dbuf_lifecycle_anchor_alias", "from_uop_id": 32, "uop_id": 33},
  ])

  out = life._reconcile_side_channel_to_rows({
    "ds_store_b128": [{"idx": 40, "uop_id": 30}],
    "s_barrier": [{"idx": 41, "uop_id": 31}],
    "ds_load_b128": [{"idx": 42, "uop_id": 33}],
  }, side)

  assert out["ok"] is True
  assert out["events"][-1]["step"] == 42


def test_side_channel_reconciliation_checks_value_keys():
  side = life._side_channel_lifecycle_events([
    {"kind": "dbuf_lifecycle_event", "op": "produce", "role": "B", "epoch": 0, "slot": 0, "window": "B:slot0",
     "uop_id": 10, "value_key": {"role": "B", "matrix": "B", "k_tile": 0}},
    {"kind": "dbuf_lifecycle_event", "op": "barrier", "uop_id": 11},
    {"kind": "dbuf_lifecycle_event", "op": "consume", "role": "B", "epoch": 0, "slot": 0, "window": "B:slot0",
     "uop_id": 12, "value_key": {"role": "B", "matrix": "B", "k_tile": 1}},
  ])

  out = life._reconcile_side_channel_to_rows({
    "ds_store_b128": [{"idx": 20, "uop_id": 10}],
    "s_barrier": [{"idx": 21, "uop_id": 11}],
    "ds_load_b128": [{"idx": 22, "uop_id": 12}],
  }, side)

  assert out["ok"] is False
  assert any("consumer value_key does not match producer" in err["error"] for err in out["check"]["errors"])


def test_p7_lowered_stream_export_exports_reconciled_side_channel_with_waits():
  side = life._side_channel_lifecycle_events([
    {"kind": "dbuf_lifecycle_event", "op": "produce", "role": "B", "epoch": 0, "slot": 0, "window": "B:slot0", "uop_id": 10},
    {"kind": "dbuf_lifecycle_event", "op": "wait", "wait_kind": "vm", "count": 0, "uop_id": 9, "phase": "after_coop_load"},
    {"kind": "dbuf_lifecycle_event", "op": "wait", "wait_kind": "lgkm", "count": 0, "uop_id": 11, "phase": "after_coop_store"},
    {"kind": "dbuf_lifecycle_event", "op": "barrier", "uop_id": 12},
    {"kind": "dbuf_lifecycle_event", "op": "wait", "wait_kind": "lgkm", "count": 0, "uop_id": 13, "phase": "after_frag_load"},
    {"kind": "dbuf_lifecycle_event", "op": "consume", "role": "B", "epoch": 0, "slot": 0, "window": "B:slot0", "uop_id": 14},
  ])

  out = life._p7_lowered_stream_export({
    "s_waitcnt": [{"idx": 19, "uop_id": 9}, {"idx": 21, "uop_id": 11}, {"idx": 23, "uop_id": 13}],
    "ds_store_b128": [{"idx": 20, "uop_id": 10}],
    "s_barrier": [{"idx": 22, "uop_id": 12}],
    "ds_load_b128": [{"idx": 24, "uop_id": 14}],
  }, {"covered_load_count": 1, "load_count": 1, "key_strength": "synthetic"}, side)

  assert out["status"] == "exported"
  assert out["check"]["ok"] is True
  assert out["check"]["p5_wait_sync"] == "checked"
  assert [event["op"] for event in out["events"]] == ["wait", "produce", "wait", "barrier", "wait", "consume"]


def test_p7_lowered_stream_export_exports_byte_window_fallback():
  side = life._side_channel_lifecycle_events([
    {"kind": "dbuf_lifecycle_event", "op": "wait", "wait_kind": "vm", "count": 0, "uop_id": 800},
    {"kind": "dbuf_lifecycle_event", "op": "wait", "wait_kind": "lgkm", "count": 0, "uop_id": 801},
    {"kind": "dbuf_lifecycle_event", "op": "consume", "role": "A", "epoch": "e0", "slot": "s0", "window": "A:s0",
     "byte_start": 32, "byte_len": 32, "uop_id": 900},
    {"kind": "dbuf_lifecycle_event", "op": "wait", "wait_kind": "lgkm", "count": 2, "uop_id": 802},
  ])

  out = life._p7_lowered_stream_export({
    "ds_store_b128": [{"idx": 10}, {"idx": 11}],
    "s_barrier": [{"idx": 12}],
    "ds_load_b128": [{"idx": 20}, {"idx": 21}],
    "s_waitcnt": [{"idx": 9, "uop_id": 800}, {"idx": 11, "uop_id": 801}, {"idx": 22, "uop_id": 802}],
    life.sp.WMMA_NAME: [{"idx": 23}],
  }, {"covered_load_count": 1, "load_count": 1, "key_strength": "synthetic"}, side, {
    "stores": [
      {"idx": 10, "op": "ds_store_b128", "normalized_window": {"base": "lds0", "lo": 32, "hi": 48}},
      {"idx": 11, "op": "ds_store_b128", "normalized_window": {"base": "lds0", "lo": 48, "hi": 64}},
    ],
    "loads": [
      {"idx": 20, "op": "ds_load_b128", "normalized_window": {"base": "lds0", "lo": 32, "hi": 48}},
      {"idx": 21, "op": "ds_load_b128", "normalized_window": {"base": "lds0", "lo": 48, "hi": 64}},
    ],
  })

  assert out["status"] == "exported"
  assert out["proof_source"] == "normalized_lds_byte_window_store_cover"
  assert out["check"]["ok"] is True
  assert out["byte_window_reconciled_side_channel"]["p5_check"]["ok"] is True
  assert [event["op"] for event in out["events"]] == ["wait", "produce", "wait", "barrier", "wait", "consume"]


def test_p7_byte_window_fallback_fails_closed_without_store_cover():
  side = life._side_channel_lifecycle_events([
    {"kind": "dbuf_lifecycle_event", "op": "consume", "role": "A", "epoch": "e0", "slot": "s0", "window": "A:s0",
     "byte_start": 32, "byte_len": 32, "uop_id": 900},
  ])

  out = life._p7_lowered_stream_export({
    "ds_store_b128": [{"idx": 10}],
    "s_barrier": [{"idx": 12}],
    "ds_load_b128": [{"idx": 20}],
  }, {"covered_load_count": 1, "load_count": 1, "key_strength": "synthetic"}, side, {
    "stores": [{"idx": 10, "op": "ds_store_b128", "normalized_window": {"base": "lds0", "lo": 32, "hi": 48}}],
    "loads": [{"idx": 20, "op": "ds_load_b128", "normalized_window": {"base": "lds0", "lo": 32, "hi": 48}}],
  })

  assert out["status"] == "fail_closed"
  assert out["byte_window_reconciled_side_channel"]["ok"] is False
  assert "stores do not exactly cover consume byte window" in out["byte_window_reconciled_side_channel"]["errors"][0]["error"]


def test_side_channel_reconciliation_rejects_missing_p5_wait():
  side = life._side_channel_lifecycle_events([
    {"kind": "dbuf_lifecycle_event", "op": "wait", "wait_kind": "vm", "count": 0, "uop_id": 9},
    {"kind": "dbuf_lifecycle_event", "op": "produce", "role": "B", "epoch": 0, "slot": 0, "window": "B:slot0", "uop_id": 10},
    {"kind": "dbuf_lifecycle_event", "op": "barrier", "uop_id": 12},
    {"kind": "dbuf_lifecycle_event", "op": "wait", "wait_kind": "lgkm", "count": 0, "uop_id": 13},
    {"kind": "dbuf_lifecycle_event", "op": "consume", "role": "B", "epoch": 0, "slot": 0, "window": "B:slot0", "uop_id": 14},
  ])

  out = life._reconcile_side_channel_to_rows({
    "s_waitcnt": [{"idx": 19, "uop_id": 9}, {"idx": 23, "uop_id": 13}],
    "ds_store_b128": [{"idx": 20, "uop_id": 10}],
    "s_barrier": [{"idx": 22, "uop_id": 12}],
    "ds_load_b128": [{"idx": 24, "uop_id": 14}],
  }, side)

  assert out["ok"] is False
  assert any("P5 requires LGKM wait after LDS stores before barrier" in err["error"] for err in out["check"]["errors"])


def test_lowered_row_tag_normalizer_exports_complete_lifecycle_metadata():
  dbuf, partial = sp._dbuf_metadata_from_tag((
    "dbuf_lifecycle", ("role", "A"), ("epoch", 3), ("slot", 1), ("window", "A:slot1"),
  ))

  assert partial is None
  assert dbuf == {"role": "A", "epoch": 3, "slot": 1, "window": "A:slot1"}


def test_lowered_row_tag_normalizer_keeps_stage_store_partial():
  dbuf, partial = sp._dbuf_metadata_from_tag(("tc_local_stage_store", "B", 991, 128, 16))

  assert dbuf is None
  assert partial == {"kind": "tc_local_stage_store", "role": "B", "lds_buffer_id": 991, "byte_start": 128, "byte_len": 16}


def test_p7_lowered_stream_export_checks_metadata_when_present():
  out = life._p7_lowered_stream_export({
    "ds_store_b128": [{"idx": 0, "dbuf": {"role": "B", "epoch": 0, "slot": 0, "window": "B:slot0"}}],
    "s_barrier": [{"idx": 1}],
    "ds_load_b128": [{"idx": 2, "dbuf": {"role": "B", "epoch": 0, "slot": 0, "window": "B:slot0"}}],
  }, {"covered_load_count": 1, "load_count": 1, "key_strength": "synthetic"})

  assert out["status"] == "exported"
  assert out["check"]["ok"] is True
  assert out["events"] == [
    {"op": "produce", "step": 0, "role": "B", "epoch": 0, "slot": 0, "window": "B:slot0"},
    {"op": "barrier", "step": 1},
    {"op": "consume", "step": 2, "role": "B", "epoch": 0, "slot": 0, "window": "B:slot0"},
  ]


def test_stage_key_compile_audit_reports_strong_key_collisions():
  out = sp._dbuf_d3a_compile_audit_summary([
    {"kind": "stage_key_audit", "slot": 16, "source": "A0", "strong_key": "K0"},
    {"kind": "stage_key_audit", "slot": 16, "source": "A1", "strong_key": "K1"},
    {"kind": "stage_key_audit", "slot": 24, "source": "B0", "strong_key": "KB"},
    {"kind": "stage_key_suppress_decision", "suppressed": False, "owner_key": "K0"},
    {"kind": "stage_key_suppress_decision", "suppressed": True, "owner_key": "K1"},
  ])

  assert out["stage_key_audit_count"] == 3
  assert out["stage_key_weak_alias_slot_count"] == 1
  assert out["stage_key_strong_collision_count"] == 0
  assert out["stage_key_rejects_weak_aliases"] is True
  assert out["stage_key_suppress_decision_count"] == 2
  assert out["stage_key_suppressed_count"] == 1
