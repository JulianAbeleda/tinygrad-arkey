import copy, json
from pathlib import Path

import pytest

from extra.qk.prefill.q4k_q8_five_buffer_role_gate import build_role_gate
from extra.qk.prefill.six_row_policy_artifact import (MissingQualificationEvidence, Q4_JOINED_EVIDENCE_SCHEMA,
  build_six_row_policy_artifact, load_explicit_evidence, main, missing_qualification_evidence)
from test.unit.test_q4k_q8_five_buffer_role_gate import _fake_compiler


ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "bench/prefill-pure-full-kernel/qwen3-14b-mixed-quant-candidate-inventory-v1.json"
Q4_JOINED = ROOT / "docs/qwen3-14b-prefill-mmq-one-role-evidence-20260718.json"
Q6_PATHS = (
  ROOT / "docs/qwen3-14b-prefill-q6-attn-kv-qualification-20260718.json",
  ROOT / "docs/qwen3-14b-prefill-q6-ffn-down-qualification-20260718.json",
)


def _inventory(): return json.loads(INVENTORY.read_text())


def _evidence():
  return json.loads(Q4_JOINED.read_text()), [json.loads(path.read_text()) for path in Q6_PATHS]


def test_current_repository_reports_six_exact_fail_closed_obligations():
  missing = missing_qualification_evidence(_inventory())
  assert missing == (
    "Q4_K:attn_kv:2bdee937a1d23c3ee747ce2a756bafd4c035478839d7eee326710a4fdcba7f76:one_role_joined_direct_packed_negative",
    "Q4_K:attn_qo:771e33f40453c6903ed69ab6b5e8f7b894c5c1d40d3405db8742b546a67b4e58:one_role_joined_direct_packed_negative",
    "Q4_K:ffn_down:931073f16c568dcb2c5435625bb660c947c0c4c08a9f1bc1ccaf5df2bb03f9e4:one_role_joined_direct_packed_negative",
    "Q4_K:ffn_gate_up:3e367ee04b0aa5db66ecaad085ec9f5ea414a56b8161b406cdefb125378356eb:one_role_joined_candidate",
    "Q6_K:attn_kv:63fd66828fe31c4752e916c33baddb3c88d6d42aa34e036d380ddb1ef8c50137:direct_packed_qualification",
    "Q6_K:ffn_down:811e68be02b3ab8a0f3cd710ee69d7bb7710c6ee4c5e93330bd84f190b0ec176:direct_packed_qualification")
  with pytest.raises(MissingQualificationEvidence) as exc: build_six_row_policy_artifact(_inventory())
  assert exc.value.missing == missing


def test_retained_evidence_blocks_all_q4_rows_without_comparable_full_role_win():
  inventory = _inventory(); q4, q6 = _evidence()
  missing = missing_qualification_evidence(inventory, q4_evidence=q4, q6_evidence=q6)
  assert len(missing) == 4 and all(item.startswith("Q4_K:") for item in missing)
  with pytest.raises(MissingQualificationEvidence) as exc:
    build_six_row_policy_artifact(inventory, q4_evidence=q4, q6_evidence=q6)
  assert exc.value.missing == missing


def test_compile_only_q4_role_gate_cannot_qualify_candidate_or_negative_rows():
  inventory = _inventory(); _, q6 = _evidence()
  compile_only = build_role_gate(inventory, compiler=_fake_compiler)
  assert compile_only["passed"] is True
  missing = missing_qualification_evidence(inventory, q4_evidence=compile_only, q6_evidence=q6)
  assert len(missing) == 4 and all(item.startswith("Q4_K:") for item in missing)
  with pytest.raises(MissingQualificationEvidence) as exc:
    build_six_row_policy_artifact(inventory, q4_evidence=compile_only, q6_evidence=q6)
  assert exc.value.missing == missing


def test_joined_q4_evidence_is_revalidated_and_fails_closed_on_drift():
  inventory = _inventory(); q4, q6 = _evidence()
  q4["production_promotion_verdict"] = "PASS"
  missing = missing_qualification_evidence(inventory, q4_evidence=q4, q6_evidence=q6)
  assert len(missing) == 4 and all(item.startswith("Q4_K:") for item in missing)

  q4, _ = _evidence()
  q4["r6_route_gate_status"]["negative_role_fallback_smoke"]["rejected_roles"] = ["attn_qo"]
  assert len(missing_qualification_evidence(inventory, q4_evidence=q4, q6_evidence=q6)) == 4


def test_q6_evidence_requires_exact_retained_qualification_identity():
  inventory = _inventory(); q4, q6 = _evidence()
  q6[0]["qualification_identity"] += "-forged"
  with pytest.raises(MissingQualificationEvidence) as exc:
    build_six_row_policy_artifact(inventory, q4_evidence=q4, q6_evidence=q6)
  assert len(exc.value.missing) == 5
  assert sum(item.endswith(":direct_packed_qualification") for item in exc.value.missing) == 1

  _, valid_q6 = _evidence()
  duplicated = [copy.deepcopy(valid_q6[0]), copy.deepcopy(valid_q6[0])]
  missing = missing_qualification_evidence(inventory, q4_evidence=q4, q6_evidence=duplicated)
  assert len(missing) == 5
  assert sum(item.startswith("Q6_K:") and ":ffn_down:" in item for item in missing) == 1


def test_explicit_loader_requires_two_q6_paths_and_cli_blocks_stale_q4_evidence(tmp_path):
  with pytest.raises(ValueError, match="exactly two"): load_explicit_evidence(Q4_JOINED, Q6_PATHS[:1])
  output = tmp_path / "policy.json"
  with pytest.raises(MissingQualificationEvidence):
    main([str(INVENTORY), "--q4-evidence", str(Q4_JOINED), "--q6-evidence",
          str(Q6_PATHS[1]), str(Q6_PATHS[0]), "--output", str(output)])
  assert not output.exists()
