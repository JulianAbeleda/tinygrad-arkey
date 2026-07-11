import pytest

from extra.qk.mmq_invocation_v2 import (BASE_TARGETS, FALSE_SITES, PHASES, _interaction_fit, _resolve_steps,
                                        generated_id, run_invocation_v2)


def test_invocation_v2_noncandidate_grid_and_resolved_axis():
  assert all(generated_id(base, sites).startswith("generated_noncandidate.") for base in BASE_TARGETS for sites in FALSE_SITES)
  resolved = [_resolve_steps(target)[1] for target in BASE_TARGETS]
  assert resolved == sorted(resolved) and len(set(resolved)) == 3


def test_invocation_v2_sample_floor():
  with pytest.raises(ValueError, match="rounds >= 30"):
    run_invocation_v2(rounds=29)


def test_invocation_v2_interaction_fit_exact_surface():
  rows = []
  for base in (30, 250, 770):
    for sites in FALSE_SITES:
      rows.append({"base_achieved_uops": base, "false_sites": sites,
                   "phases": {"x": {"overhead_corrected_median_ns": 7 + 2 * base + 3 * sites + 4 * base * sites}}})
  fit = _interaction_fit(rows, "x")
  assert fit["coefficients_ns"] == pytest.approx([7, 2, 3, 4]) and fit["r2"] == pytest.approx(1)


def test_invocation_v2_live_amd_contract():
  result = run_invocation_v2(rounds=30, warmups=1, seed=11, system_snapshot_id="sha256:" + "c" * 64)
  assert result["candidate_ids"] == [] and result["production_dispatch_changed"] is False
  assert len(result["rows"]) == 9 and len(result["protocol"]["randomized_interleaved_order"]) == 270
  assert set(result["interaction_fits"]) == set(PHASES)
  assert result["candidate_fixed_baseline_coverage"]["direct_owner_base_bracketed"] is True
  assert result["candidate_fixed_baseline_coverage"]["gated_matrix_base_bracketed"] is False
  for row in result["rows"]:
    assert row["candidate_id"] is None
    assert all(len(row["phases"][phase]["samples_ns"]) == 30 for phase in PHASES)
