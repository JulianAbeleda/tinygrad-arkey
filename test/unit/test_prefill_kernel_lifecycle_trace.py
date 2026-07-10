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
