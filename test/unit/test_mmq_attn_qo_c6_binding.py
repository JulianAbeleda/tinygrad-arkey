from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from extra.qk.mmq_attn_qo_c6_binding import (
  C6_BINDING_SCHEMA, COMPOSITION_SCHEMA, QO_SEEDS,
  compose_attn_qo_c6_binding, rebuild_attn_qo_exact_fixture,
)
from extra.qk.mmq_exact_role_spec import exact_role_spec
from extra.qk.mmq_frozen_staged_family import load_frozen_staged_family_manifest


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "docs/artifacts/qwen3-14b-prefill-attn-qo-staged-951d3615c-20260719"
EVIDENCE = ARTIFACT / "evidence"
FAMILY = EVIDENCE / "qk-attn-qo-staged-951d3615c-final-r1-20260719-family.json"
BUNDLE = ARTIFACT / "bundle"
C6 = {
  "PM4": EVIDENCE / "qk-attn-qo-staged-951d3615c-final-20260719-c6-pm4-full20.json",
  "AQL": EVIDENCE / "qk-attn-qo-staged-951d3615c-final-20260719-c6-aql-full20.json",
}
C7 = EVIDENCE / "qk-attn-qo-staged-c590434d8-20260719-c7-ledger.json"
C7_AUTHORITY = EVIDENCE / "qk-attn-qo-staged-c590434d8-20260719-c7-authority.json"
C7_CAPTURES = {
  "PM4": EVIDENCE / "qk-attn-qo-staged-c590434d8-20260719-c7-pm4.json",
  "AQL": EVIDENCE / "qk-attn-qo-staged-c590434d8-20260719-c7-aql.json",
}


@pytest.fixture(scope="module")
def retained():
  role = exact_role_spec("attn_qo")
  family = load_frozen_staged_family_manifest(
    FAMILY, role_spec=role, frozen_bundle=BUNDLE)
  raw = {queue: json.loads(path.read_text()) for queue, path in C6.items()}
  c7 = json.loads(C7.read_text())
  c7_authority = json.loads(C7_AUTHORITY.read_text())
  captures = {queue: json.loads(path.read_text()) for queue, path in C7_CAPTURES.items()}
  return role, family, raw, c7, c7_authority, captures


def test_rebuilds_exact_retained_qo_fixture_bytes_and_identity(retained):
  role, _family, raw, _c7, _authority, _captures = retained
  fixture = rebuild_attn_qo_exact_fixture(
    role, raw["PM4"]["raw_probe"]["execution_fixture"])
  assert fixture.execution_fixture["seeds"] == QO_SEEDS
  assert fixture.execution_fixture["repack"] == \
    raw["AQL"]["raw_probe"]["execution_fixture"]["repack"]
  assert fixture.words.dtype.name == "uint32"
  assert fixture.source.dtype.name == "float32"
  assert fixture.production_activation.dtype.name == "float16"
  assert fixture.activation_relation["shared_execution_bytes_claimed"] is False
  assert fixture.activation_relation["logical_source_sha256"] == \
    raw["PM4"]["raw_probe"]["execution_fixture"]["source_sha256"]
  assert fixture.words.size == 5120 * 20 * 36
  assert fixture.source.shape == (512, 5120)
  assert fixture.q4_epoch_major.size == fixture.words.size
  assert fixture.fixture_identity.startswith("sha256:")
  assert fixture.input_identity.startswith("sha256:")


def test_composes_strict_dual_queue_c6_with_shared_c7_authority(retained):
  _role, family, raw, c7, authority, captures = retained
  result = compose_attn_qo_c6_binding(
    family=family, raw_c6_by_queue=raw, c7_memory_ledger=c7,
    c7_authority_snapshot=authority, c7_captures_by_queue=captures)
  assert result["schema"] == COMPOSITION_SCHEMA and result["status"] == "PASS"
  c6 = result["c6_correctness_evidence"]
  assert c6["schema"] == C6_BINDING_SCHEMA and c6["status"] == "PASS"
  assert c6["family_identity"] == family.family_identity
  assert c6["candidate_binary_sha256"] == family.binding.binary_sha256
  assert set(c6["queue_correctness"]) == {"PM4", "AQL"}
  assert set(c6["queue_comparators"]) == {"PM4", "AQL"}
  assert c6["device_identity"] == c7["budget"]["authority"]["device_identity"]
  assert c6["software_identity"] == c7["budget"]["authority"]["software_identity"]
  assert result["execution_fixture"] == \
    raw["PM4"]["raw_probe"]["execution_fixture"] == \
    raw["AQL"]["raw_probe"]["execution_fixture"]
  assert result["runtime_canary_by_queue"]["PM4"]["amd_aql_effective"] is False
  assert result["runtime_canary_by_queue"]["AQL"]["amd_aql_effective"] is True
  assert result["promotion_eligible_on_candidate_win"] is False


def test_composer_rejects_queue_fixture_or_family_drift(retained):
  _role, family, raw, c7, authority, captures = retained
  drifted = copy.deepcopy(raw)
  drifted["AQL"]["raw_probe"]["execution_fixture"]["seeds"]["q4"] += 1
  with pytest.raises(ValueError, match="fixture"):
    compose_attn_qo_c6_binding(
      family=family, raw_c6_by_queue=drifted, c7_memory_ledger=c7,
      c7_authority_snapshot=authority, c7_captures_by_queue=captures)

  drifted = copy.deepcopy(raw)
  drifted["PM4"]["family_identity"] = "sha256:" + "0" * 64
  with pytest.raises(ValueError, match="top-level"):
    compose_attn_qo_c6_binding(
      family=family, raw_c6_by_queue=drifted, c7_memory_ledger=c7,
      c7_authority_snapshot=authority, c7_captures_by_queue=captures)


def test_fixture_builder_rejects_hash_drift_before_live_objects(retained):
  role, _family, raw, _c7, _authority, _captures = retained
  fixture = copy.deepcopy(raw["PM4"]["raw_probe"]["execution_fixture"])
  fixture["repack"]["q4_sha256"] = "0" * 64
  with pytest.raises(ValueError, match="fixture bytes differ"):
    rebuild_attn_qo_exact_fixture(role, fixture)
