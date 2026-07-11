import pytest

from extra.qk.mmq_staging_evidence import (
  SCHEMA, SUM_SLOT_SCHEMA, build_bounded_16x16_sum_slot_map, build_mmq_staging_evidence_bundle,
  validate_mmq_staging_evidence_bundle,
)


def test_build_mmq_staging_evidence_packages_valid_bounded_16x16x256_metadata():
  bundle = build_mmq_staging_evidence_bundle(
    candidate_id="r4_16x16x256_q4k_q8_1_staging_probe",
    backend="q4k_q8_1_mmq_amd_coop_tile_research_only",
    shape={"M": 16, "N": 16, "K": 256},
    notes="research-only structural evidence; no cooperative atom claim",
  )

  assert bundle["schema"] == SCHEMA
  assert bundle["evidence_kind"] == "staging_sum_slots"
  assert bundle["shape"] == {"M": 16, "N": 16, "K": 256}
  assert bundle["q4k_tile_staging"]["status"] == "present"
  assert bundle["q4k_tile_staging"]["tile_shape"] == {"N": 16, "K": 256}
  assert bundle["q8_1_ds4_staging"]["status"] == "present"
  assert bundle["q8_1_ds4_staging"]["layout"] == "q8_1_mmq_ds4_transposed_blocks"
  assert bundle["q8_1_ds4_staging"]["panels"] == 2
  assert bundle["sum_slot_map"]["schema"] == SUM_SLOT_SCHEMA
  assert bundle["sum_slot_map"]["tile_shape"] == {"M": 16, "N": 16}
  assert bundle["sum_slot_map"]["total_slots"] == 256
  assert bundle["sum_slot_map"]["mapping"][0] == {"m": 0, "n": 0, "slot": 0, "thread": 0, "lane": 0}
  assert bundle["sum_slot_map"]["mapping"][-1] == {"m": 15, "n": 15, "slot": 255, "thread": 255, "lane": 63}
  assert bundle["status"] == "PASS"
  assert bundle["exact_blocker"] is None
  assert bundle["production_dispatch_changed"] is False
  assert validate_mmq_staging_evidence_bundle(bundle) == bundle


@pytest.mark.parametrize(
  ("shape", "message"),
  [
    ({"M": 15, "N": 16, "K": 256}, "shape.M must be 16-aligned"),
    ({"M": 16, "N": 15, "K": 256}, "shape.N must be 16-aligned"),
    ({"M": 16, "N": 16, "K": 128}, "shape.K must be 256-aligned"),
  ],
)
def test_build_mmq_staging_evidence_rejects_invalid_shape_alignment(shape, message):
  with pytest.raises(ValueError, match=message):
    build_mmq_staging_evidence_bundle(candidate_id="c0", backend="research", shape=shape)


def test_build_mmq_staging_evidence_marks_blocked_missing_staging_data():
  bundle = build_mmq_staging_evidence_bundle(
    candidate_id="r4_missing_ds4_panel_probe",
    backend="q4k_q8_1_mmq_amd_coop_tile_research_only",
    shape={"M": 16, "N": 16, "K": 256},
    q8_1_ds4_staging={"status": "missing", "layout": "q8_1_mmq_ds4_transposed_blocks"},
  )

  assert bundle["status"] == "BLOCKED"
  assert bundle["exact_blocker"] == "missing Q8_1 DS4 staging data"
  assert bundle["production_dispatch_changed"] is False
  assert validate_mmq_staging_evidence_bundle(bundle)["status"] == "BLOCKED"


def test_validate_mmq_staging_evidence_rejects_production_dispatch_claim():
  bundle = build_mmq_staging_evidence_bundle(
    candidate_id="r4_16x16x256_q4k_q8_1_staging_probe",
    backend="q4k_q8_1_mmq_amd_coop_tile_research_only",
    shape={"M": 16, "N": 16, "K": 256},
  )
  bundle["production_dispatch_changed"] = True

  with pytest.raises(ValueError, match="production_dispatch_changed must be False"):
    validate_mmq_staging_evidence_bundle(bundle)

  with pytest.raises(ValueError, match="production_dispatch_changed must be False"):
    build_mmq_staging_evidence_bundle(
      candidate_id="r4_prod_claim",
      backend="q4k_q8_1_mmq_amd_coop_tile_research_only",
      shape={"M": 16, "N": 16, "K": 256},
      production_dispatch_changed=True,
    )


def test_sum_slot_map_rejects_duplicate_output_or_slot_claims():
  slots = build_bounded_16x16_sum_slot_map()
  slots["mapping"][1] = dict(slots["mapping"][0])

  with pytest.raises(ValueError, match="duplicates output"):
    build_mmq_staging_evidence_bundle(
      candidate_id="r4_bad_slots",
      backend="research",
      shape={"M": 16, "N": 16, "K": 256},
      sum_slot_map=slots,
    )
