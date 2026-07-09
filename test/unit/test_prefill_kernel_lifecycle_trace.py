import json, os, subprocess, sys
from pathlib import Path

from extra.qk import pure_kernel_surface_audit as surface_audit
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
