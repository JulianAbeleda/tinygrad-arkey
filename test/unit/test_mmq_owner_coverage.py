from extra.qk.mmq_owner_coverage import (
  SCHEMA, build_mmq_owner_coverage_artifact, observed_stores_from_amd_isa_proof_rows,
  structural_static_store_only_owner_map,
  tinygrad_custom_kernel_store_owner_trace_blocker, validate_mmq_owner_coverage_artifact,
)
from extra.qk.mmq_llama_store_probe import lowered_tinygrad_r4_store_owner_trace_rows
from extra.qk.mmq_q4k_q8_reference import Q8_1_MMQ_DS4_LAYOUT, describe_q4k_q8_1_mmq_tile


def _spec(m=16, n=16, k=256):
  return describe_q4k_q8_1_mmq_tile(role="ffn_gate_up", m=m, n=n, k=k, m_tile=m, n_tile=n,
                                    activation_layout=Q8_1_MMQ_DS4_LAYOUT)


def test_mmq_owner_coverage_passes_structural_static_observed_map_without_dispatch_claim():
  spec = _spec()
  stores = structural_static_store_only_owner_map(spec)
  artifact = build_mmq_owner_coverage_artifact(spec, stores)

  assert artifact["schema"] == SCHEMA
  assert artifact["evidence_kind"] == "owner_coverage"
  assert artifact["candidate_id"] == "llama_mmq_r4_store_owner_coverage_probe"
  assert artifact["backend"] == "research_only_structural_static_store_owner_map"
  assert artifact["shape"] == {"M": 16, "N": 16, "K": 256}
  assert artifact["oracle_backend"] == "llama_mmq_q4k_q8_1_coop_tile_oracle"
  assert artifact["expected_stores"]["store_count"] == 256
  assert stores[0].owner["evidence"] == "structural_static_store_only_map"
  assert stores[0].owner["gpu_execution_trace"] is False
  assert stores[0].owner["source"] == "translated_llama_mmq_16x16_c_fragment_writeback_structure"
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
  stores = list(structural_static_store_only_owner_map(spec))
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
  stores = list(structural_static_store_only_owner_map(spec))[1:]

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
  artifact = build_mmq_owner_coverage_artifact(spec, structural_static_store_only_owner_map(spec))
  artifact["production_dispatch_changed"] = True

  try:
    validate_mmq_owner_coverage_artifact(artifact)
  except ValueError as exc:
    assert "production_dispatch_changed must be False" in str(exc)
  else:
    raise AssertionError("production dispatch claim should fail")


def test_mmq_owner_coverage_records_custom_kernel_owner_trace_blocker():
  blocker = tinygrad_custom_kernel_store_owner_trace_blocker()

  assert blocker["status"] == "BLOCKED"
  assert blocker["gpu_execution_trace"] is False
  assert "16x16 R4 map lowers as eight spill-free" in blocker["exact_blocker"]


def test_mmq_owner_coverage_accepts_lowered_amd_isa_store_owner_rows():
  spec = _spec()
  rows = [
    {
      "kind": "global_store",
      "store_owner": store.owner | {"m": store.m, "n": store.n},
      "emitted": "global_store_b32 ...",
      "addr_vgpr": 10,
      "data_vgpr": 11,
    }
    for store in structural_static_store_only_owner_map(spec)
  ]

  observed = observed_stores_from_amd_isa_proof_rows(rows)
  artifact = build_mmq_owner_coverage_artifact(spec, observed, backend="lowered_amd_isa_store_owner_manifest")

  assert len(observed) == 256
  assert observed[0].owner["evidence"] == "lowered_amd_isa_global_store_proof_manifest"
  assert observed[0].owner["gpu_execution_trace"] is False
  assert artifact["status"] == "PASS"
  assert artifact["duplicate_store_summary"]["count"] == 0
  assert artifact["missing_store_summary"]["count"] == 0


def test_mmq_owner_coverage_passes_lowered_tinygrad_r4_store_owner_trace_rows():
  spec = _spec()

  rows = lowered_tinygrad_r4_store_owner_trace_rows(spec)
  observed = observed_stores_from_amd_isa_proof_rows(rows)
  artifact = build_mmq_owner_coverage_artifact(spec, observed, backend="lowered_amd_isa_fragmented_store_owner_manifest")

  assert len(rows) == 256
  assert len(observed) == 256
  assert artifact["status"] == "PASS"
  assert artifact["observed_stores"]["unique_store_count"] == 256
  assert artifact["duplicate_store_summary"]["count"] == 0
  assert artifact["missing_store_summary"]["count"] == 0
  assert all(row["logical_op"] == "GATED_STORE" for row in rows)
  assert all(row["gated"] is True for row in rows)


def test_mmq_owner_coverage_validator_rejects_tampered_counts():
  spec = _spec()
  artifact = build_mmq_owner_coverage_artifact(spec, structural_static_store_only_owner_map(spec))
  artifact["observed_stores"]["unique_store_count"] = 255

  try:
    validate_mmq_owner_coverage_artifact(artifact)
  except ValueError as exc:
    assert "event count cannot be below unique" not in str(exc)
    assert "disagrees" in str(exc)
  else:
    raise AssertionError("tampered observed counts should fail validation")


def test_mmq_owner_coverage_validator_rejects_out_of_tile_pass_claim():
  spec = _spec()
  stores = list(structural_static_store_only_owner_map(spec))
  stores[0] = type(stores[0])(m=99, n=99, owner=stores[0].owner)
  artifact = build_mmq_owner_coverage_artifact(spec, stores)
  assert artifact["status"] == "FAIL"
  artifact["status"] = "PASS"
  artifact["exact_blocker"] = None

  try:
    validate_mmq_owner_coverage_artifact(artifact)
  except ValueError as exc:
    assert "status disagrees" in str(exc)
  else:
    raise AssertionError("out-of-tile PASS claim should fail validation")
