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
  assert {row["id"] for row in report["missing_evidence"]} == set(REQUIRED_EVIDENCE)


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
