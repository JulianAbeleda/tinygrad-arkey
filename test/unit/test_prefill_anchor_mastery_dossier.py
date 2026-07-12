import json

import pytest

from extra.qk.prefill.anchor_mastery_dossier import REQUIRED_EVIDENCE, build_dossier, main, validate_dossier


def test_anchor_dossier_reuses_registries_and_fails_closed():
  report = build_dossier()
  validate_dossier(report)
  assert report["anchor"] == {"profile_id": "qwen3_8b_q4k_m_gfx1100", "device_profile": "gfx1100",
                              "role": "ffn_gate_up", "phase": "prefill",
                              "shape": {"M": 512, "N": 12288, "K": 4096}, "quant": "Q4_K_M"}
  routes = report["known_evidence"]["existing_route_ownership"]["value"]
  assert routes["pure_baseline"]["strict_pure"] is True
  assert routes["structural_oracle"]["strict_pure"] is False
  assert set(report["evidence_status"]) == set(REQUIRED_EVIDENCE)
  assert report["mastery_complete"] is False
  assert {row["id"] for row in report["missing_evidence"]} == {
    key for key, value in report["evidence_status"].items() if value["status"] != "complete"
  }


def test_anchor_dossier_names_absent_artifacts_without_fabricating_values(tmp_path):
  report = build_dossier(root=tmp_path)
  assert all(row["present"] is False and row["error"] == "missing" for row in report["artifact_index"].values())
  assert report["known_evidence"]["sample_correctness"]["state"] == "not_available"
  assert report["known_evidence"]["sample_correctness"]["max_abs_error"] is None
  assert report["known_evidence"]["spec_resource_estimates"]["lds_bytes"] is None


def test_anchor_dossier_cli_writes_valid_manifest(tmp_path):
  output = tmp_path / "dossier.json"
  result = main(["--output", str(output)])
  assert json.loads(output.read_text()) == result
  validate_dossier(result)


def test_anchor_dossier_validator_rejects_false_mastery():
  report = build_dossier()
  report["mastery_complete"] = True
  with pytest.raises(ValueError, match="fail closed"):
    validate_dossier(report)


def test_anchor_dossier_accepts_clean_exact_capture_evidence(tmp_path):
  isa = tmp_path / "bench/prefill-pure-full-kernel/anchor-ffn-gate-up/mastery-v1/resources-isa.json"
  timing = tmp_path / "bench/prefill-pure-full-kernel/anchor-ffn-gate-up/mastery-v1/role-timing.json"
  isa.parent.mkdir(parents=True)
  isa.write_text(json.dumps({
    "schema": "prefill-pure-anchor-isa-resource-capture.v1", "git": {"dirty": False}, "binding_complete": True,
    "captures": [{"candidate_id": "pure.default.m512n12288k4096",
      "program": {key: key for key in ("program_key", "source_sha256", "binary_sha256", "isa_sha256")},
      "surface": {"strict_pure": True},
      "resources": {"authority": "metadata", "vgpr": 1, "sgpr": 1, "lds_bytes": 0, "scratch_bytes": 0}}]}))
  timing.write_text(json.dumps({
    "schema": "prefill-anchor-gemm-regime-timing.v1", "complete": True,
    "environment": {"git_dirty": False},
    "rows": [{"regime": name, "binding_pass": True, "measurement": {"tflops": value}}
             for name, value in (("pure_scheduler", 22), ("spec_owned", 15), ("s9_oracle", 74))]}))
  report = build_dossier(root=tmp_path)
  assert report["evidence_status"]["generated_isa_capture"]["status"] == "complete"
  assert report["evidence_status"]["measured_resource_capture"]["status"] == "complete"
  assert report["evidence_status"]["kernel_timing_authority"]["status"] == "complete"
  assert report["evidence_status"]["roofline_attribution"]["status"] == "partial"
