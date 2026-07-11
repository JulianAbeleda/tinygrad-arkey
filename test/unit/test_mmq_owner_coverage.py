from extra.qk.mmq_owner_coverage import (
  SCHEMA, build_mmq_owner_coverage_artifact, observed_stores_from_oracle, validate_mmq_owner_coverage_artifact,
)
from extra.qk.mmq_q4k_q8_reference import Q8_1_MMQ_DS4_LAYOUT, describe_q4k_q8_1_mmq_tile


def _spec(m=16, n=16, k=256):
  return describe_q4k_q8_1_mmq_tile(role="ffn_gate_up", m=m, n=n, k=k, m_tile=m, n_tile=n,
                                    activation_layout=Q8_1_MMQ_DS4_LAYOUT)


def test_mmq_owner_coverage_passes_perfect_oracle_observed_map_without_dispatch_claim():
  spec = _spec()
  artifact = build_mmq_owner_coverage_artifact(spec, observed_stores_from_oracle(spec))

  assert artifact["schema"] == SCHEMA
  assert artifact["evidence_kind"] == "owner_coverage"
  assert artifact["candidate_id"] == "llama_mmq_r4_store_owner_coverage_probe"
  assert artifact["shape"] == {"M": 16, "N": 16, "K": 256}
  assert artifact["oracle_backend"] == "llama_mmq_q4k_q8_1_coop_tile_oracle"
  assert artifact["expected_stores"]["store_count"] == 256
  assert artifact["observed_stores"]["store_event_count"] == 256
  assert artifact["observed_stores"]["unique_store_count"] == 256
  assert artifact["duplicate_store_summary"]["count"] == 0
  assert artifact["missing_store_summary"]["count"] == 0
  assert artifact["status"] == "PASS"
  assert artifact["exact_blocker"] is None
  assert artifact["research_only"] is True
  assert artifact["production_dispatch_changed"] is False
  assert validate_mmq_owner_coverage_artifact(artifact) == artifact


def test_mmq_owner_coverage_fails_duplicate_store():
  spec = _spec()
  stores = list(observed_stores_from_oracle(spec))
  stores.append(stores[0])

  artifact = build_mmq_owner_coverage_artifact(spec, stores)

  assert artifact["status"] == "FAIL"
  assert artifact["observed_stores"]["store_event_count"] == 257
  assert artifact["duplicate_store_summary"]["count"] == 1
  assert artifact["missing_store_summary"]["count"] == 0
  assert "duplicate_store_count=1" in artifact["exact_blocker"]
  assert artifact["production_dispatch_changed"] is False
  assert validate_mmq_owner_coverage_artifact(artifact)["status"] == "FAIL"


def test_mmq_owner_coverage_fails_missing_store():
  spec = _spec()
  stores = list(observed_stores_from_oracle(spec))[1:]

  artifact = build_mmq_owner_coverage_artifact(spec, stores)

  assert artifact["status"] == "FAIL"
  assert artifact["observed_stores"]["store_event_count"] == 255
  assert artifact["duplicate_store_summary"]["count"] == 0
  assert artifact["missing_store_summary"]["count"] == 1
  assert artifact["missing_store_summary"]["stores"] == [{"m": 0, "n": 0}]
  assert "missing_store_count=1" in artifact["exact_blocker"]
  assert artifact["production_dispatch_changed"] is False
  assert validate_mmq_owner_coverage_artifact(artifact)["status"] == "FAIL"


def test_mmq_owner_coverage_blocks_without_observed_map():
  spec = _spec()

  artifact = build_mmq_owner_coverage_artifact(spec, None)

  assert artifact["status"] == "BLOCKED"
  assert artifact["observed_stores"] is None
  assert artifact["missing_store_summary"]["count"] == 256
  assert artifact["exact_blocker"] == "observed store-only owner map is unavailable"
  assert artifact["production_dispatch_changed"] is False
  assert validate_mmq_owner_coverage_artifact(artifact)["status"] == "BLOCKED"


def test_mmq_owner_coverage_rejects_production_dispatch_claim():
  spec = _spec()
  artifact = build_mmq_owner_coverage_artifact(spec, observed_stores_from_oracle(spec))
  artifact["production_dispatch_changed"] = True

  try:
    validate_mmq_owner_coverage_artifact(artifact)
  except ValueError as exc:
    assert "production_dispatch_changed must be False" in str(exc)
  else:
    raise AssertionError("production dispatch claim should fail")
