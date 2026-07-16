import copy, json
from pathlib import Path

import pytest

from extra.qk.prefill.q4k_q8_five_buffer_role_gate import build_role_gate
from extra.qk.prefill.six_row_policy_artifact import (MissingQualificationEvidence, Q6_EVIDENCE_SCHEMA,
  build_six_row_policy_artifact, load_explicit_evidence, main, missing_qualification_evidence)
from test.unit.test_q4k_q8_five_buffer_role_gate import _fake_compiler


def _inventory():
  return json.loads(Path("bench/prefill-pure-full-kernel/qwen3-14b-mixed-quant-candidate-inventory-v1.json").read_text())


def _q6_evidence(inventory):
  return [{"schema":Q6_EVIDENCE_SCHEMA, "status":"qualified", "route_id":"direct_packed",
           "canonical_identity":binding["canonical_identity"], "qualification_identity":f"test:{index}",
           "workload":{"phase":"prefill", "role":binding["inventory_key"]["role"], "quant_format":"Q6_K",
             "shape":binding["inventory_key"]["shape"], "target":{"backend":"AMD", "arch":"gfx1100", "wave_size":32}}}
          for index, binding in enumerate(inventory["bindings"])
          if binding["inventory_key"]["quant_format"] == "Q6_K"]


def test_current_repository_reports_the_exact_six_missing_evidence_items():
  missing = missing_qualification_evidence(_inventory())
  assert missing == (
    "Q4_K:attn_kv:25a3cb0145bac1306508eed778eec41eb72c9a77729c981f73bf58199d7e64e6:five_buffer_role_gate",
    "Q4_K:attn_qo:513ee40319c03b5e0f4d8a239a115c4a3efca23431ccb26c98ede6c2c63efc4c:five_buffer_role_gate",
    "Q4_K:ffn_down:db5e2872456f952745681815e9f51f766b2ed31f6e6796969de28490fe035721:five_buffer_role_gate",
    "Q4_K:ffn_gate_up:ef78eefac3d436e794868a40877aaf829e12b443ef8ba191608eb2ec7cb636e2:five_buffer_role_gate",
    "Q6_K:attn_kv:63fd66828fe31c4752e916c33baddb3c88d6d42aa34e036d380ddb1ef8c50137:direct_packed_qualification",
    "Q6_K:ffn_down:811e68be02b3ab8a0f3cd710ee69d7bb7710c6ee4c5e93330bd84f190b0ec176:direct_packed_qualification")
  with pytest.raises(MissingQualificationEvidence) as exc: build_six_row_policy_artifact(_inventory())
  assert exc.value.missing == missing


def test_complete_exact_evidence_builds_deterministic_research_only_artifact():
  inventory = _inventory(); q4 = build_role_gate(inventory, compiler=_fake_compiler); q6 = _q6_evidence(inventory)
  first = build_six_row_policy_artifact(inventory, q4_evidence=q4, q6_evidence=q6)
  second = build_six_row_policy_artifact(copy.deepcopy(inventory), q4_evidence=copy.deepcopy(q4), q6_evidence=copy.deepcopy(q6))
  assert first == second and first["status"] == "research_only" and first["production_promotion"] is False
  assert len(first["policy_rows"]) == 6
  assert [row["binding_kind"] for row in first["policy_rows"]].count("candidate") == 4
  assert [row["binding_kind"] for row in first["policy_rows"]].count("fallback") == 2


def test_q6_evidence_must_be_canonical_qualified_direct_packed_evidence():
  inventory = _inventory(); q4 = build_role_gate(inventory, compiler=_fake_compiler); q6 = _q6_evidence(inventory)
  q6[0]["status"] = "measured"
  with pytest.raises(MissingQualificationEvidence) as exc:
    build_six_row_policy_artifact(inventory, q4_evidence=q4, q6_evidence=q6)
  assert len(exc.value.missing) == 1 and exc.value.missing[0].endswith(":direct_packed_qualification")


def test_duplicate_or_unrelated_evidence_cannot_fill_exact_rows():
  inventory = _inventory(); q4 = build_role_gate(inventory, compiler=_fake_compiler); q6 = _q6_evidence(inventory)
  q4["rows"][1] = copy.deepcopy(q4["rows"][0]); q6[1] = copy.deepcopy(q6[0])
  missing = missing_qualification_evidence(inventory, q4_evidence=q4, q6_evidence=q6)
  assert len(missing) == 2 and {item.split(":")[1] for item in missing} == {"attn_qo", "attn_kv"}
  valid_q4 = build_role_gate(inventory, compiler=_fake_compiler)
  assert len(missing_qualification_evidence(inventory, q4_evidence=valid_q4, q6_evidence=_q6_evidence(inventory) * 2)) == 2


def test_explicit_loader_requires_two_q6_paths_and_cli_writes_artifact(tmp_path):
  inventory = _inventory(); q4 = build_role_gate(inventory, compiler=_fake_compiler); q6 = _q6_evidence(inventory)
  paths = [tmp_path / name for name in ("inventory.json", "q4.json", "q6-attn.json", "q6-ffn.json")]
  for path, value in zip(paths, (inventory, q4, *q6)): path.write_text(json.dumps(value))
  with pytest.raises(ValueError, match="exactly two"): load_explicit_evidence(paths[1], paths[2:3])
  output = tmp_path / "policy.json"
  main([str(paths[0]), "--q4-evidence", str(paths[1]), "--q6-evidence", str(paths[3]), str(paths[2]), "--output", str(output)])
  artifact = json.loads(output.read_text())
  assert len(artifact["policy_rows"]) == 6
