import pytest

from extra.qk.mmq_bounded_harness import BoundedMMQConfig, run_bounded_harness
from extra.qk.mmq_timing_result import (
  SCHEMA, build_mmq_timing_result_bundle, build_mmq_timing_result_from_bounded_harness_report,
  validate_mmq_timing_result_bundle,
)


def test_build_mmq_timing_result_bundle_packages_bounded_candidate_timing_without_route_claim():
  bundle = build_mmq_timing_result_bundle(
    candidate_id="prefill_14b_q4k_q8_1_hybrid_mmq_atom.reference.m4.n4.k256.row_major_q8_1",
    backend="reference",
    shape={"M": 4, "N": 4, "K": 256},
    comparator_id="direct_packed",
    timing_status="measured",
    timings_ms={"candidate_min_ms": 0.25, "comparator_min_ms": 0.5},
    speedup_vs_comparator=2.0,
    blockers=(),
  )

  assert bundle == {
    "schema": SCHEMA,
    "candidate_id": "prefill_14b_q4k_q8_1_hybrid_mmq_atom.reference.m4.n4.k256.row_major_q8_1",
    "backend": "reference",
    "shape": {"M": 4, "N": 4, "K": 256},
    "comparator_id": "direct_packed",
    "production_dispatch_changed": False,
    "timing_status": "measured",
    "timings_ms": {"candidate_min_ms": 0.25, "comparator_min_ms": 0.5},
    "speedup_vs_comparator": 2.0,
    "blockers": [],
  }
  assert validate_mmq_timing_result_bundle(bundle)["production_dispatch_changed"] is False


def test_validate_mmq_timing_result_bundle_rejects_unknown_fields_and_production_binding_claim():
  with pytest.raises(ValueError, match="unknown fields"):
    validate_mmq_timing_result_bundle({
      "schema": SCHEMA,
      "candidate_id": "c0",
      "backend": "reference",
      "shape": {"M": 4, "N": 4, "K": 256},
      "comparator_id": "direct_packed",
      "production_dispatch_changed": False,
      "timing_status": "measured",
      "production_route_bound": True,
    })

  with pytest.raises(ValueError, match="production_dispatch_changed must be False"):
    validate_mmq_timing_result_bundle({
      "schema": SCHEMA,
      "candidate_id": "c0",
      "backend": "reference",
      "shape": {"M": 4, "N": 4, "K": 256},
      "comparator_id": "direct_packed",
      "production_dispatch_changed": True,
      "timing_status": "measured",
    })


@pytest.mark.parametrize(
  ("kwargs", "message"),
  [
    ({"candidate_id": ""}, "candidate_id must be a non-empty string"),
    ({"shape": {"M": 4, "N": 4, "K": 256, "batch": 1}}, "shape contains unknown fields"),
    ({"shape": {"M": 4, "N": 4}}, "shape missing required fields"),
    ({"shape": {"M": 4, "N": 0, "K": 256}}, "shape.N must be a positive integer"),
    ({"timing_status": "promoted"}, "timing_status must be one of"),
    ({"timings_ms": {}}, "timings_ms must not be empty"),
    ({"timings_ms": {"candidate_min_ms": -0.1}}, "timings_ms.candidate_min_ms must be a non-negative number"),
    ({"timings_ms": {"": 1.0}}, "timings_ms key must be a non-empty string"),
    ({"speedup_vs_comparator": 0.0}, "speedup_vs_comparator must be a positive number"),
    ({"blockers": [""]}, r"blockers\[0\] must be a non-empty string"),
  ],
)
def test_build_mmq_timing_result_bundle_rejects_invalid_synthetic_fields(kwargs, message):
  args = {
    "candidate_id": "c0",
    "backend": "reference",
    "shape": {"M": 4, "N": 4, "K": 256},
    "comparator_id": "direct_packed",
    "timing_status": "measured",
    **kwargs,
  }

  with pytest.raises(ValueError, match=message):
    build_mmq_timing_result_bundle(**args)


def test_build_mmq_timing_result_from_bounded_harness_report_marks_measured_with_comparator_speedup():
  cfg = BoundedMMQConfig(m_tile=4, n_tile=4, k_groups=8, rounds=1, measure_direct_packed=True)
  report = run_bounded_harness(cfg)
  bundle = build_mmq_timing_result_from_bounded_harness_report(report)

  assert bundle["schema"] == SCHEMA
  assert bundle["candidate_id"] == (
    "prefill_14b_q4k_q8_1_hybrid_mmq_atom.reference.m4.n4.k256.row_major_q8_1"
  )
  assert bundle["backend"] == "reference"
  assert bundle["shape"] == {"M": 4, "N": 4, "K": 256}
  assert bundle["comparator_id"] == "direct_packed"
  assert bundle["production_dispatch_changed"] is False
  assert bundle["timing_status"] == "measured"
  assert bundle["timings_ms"]["candidate_min_ms"] >= 0
  assert bundle["timings_ms"]["comparator_min_ms"] >= 0
  assert bundle["speedup_vs_comparator"] > 0


def test_build_mmq_timing_result_from_bounded_harness_report_marks_blocked_when_report_has_blockers():
  report = run_bounded_harness(BoundedMMQConfig(m_tile=4, n_tile=4, k_groups=8, rounds=1, backend="atom"))
  bundle = build_mmq_timing_result_from_bounded_harness_report(report)

  assert bundle["timing_status"] == "blocked"
  assert bundle["blockers"] == ["atom backend is reference-backed; AMD GPU atom body is not implemented"]
  assert bundle["production_dispatch_changed"] is False
