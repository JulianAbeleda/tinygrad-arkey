import pytest

from extra.qk.mmq_invocation_v3 import (BASE_TARGETS, FALSE_SITES, PHASES, _interaction_fit, _resolve_steps,
                                        generated_id, run_invocation_v3)


def test_invocation_v3_exact_grid_and_noncandidate_ids():
  assert len([(b,s) for b in BASE_TARGETS for s in FALSE_SITES]) == 4
  assert all(generated_id(b,s).startswith("generated_noncandidate.") for b in BASE_TARGETS for s in FALSE_SITES)
  assert _resolve_steps(1024)[1] < _resolve_steps(1280)[1]


def test_invocation_v3_requires_exact_sample_contract():
  with pytest.raises(ValueError, match="exactly 30"):
    run_invocation_v3(rounds=31)


def test_invocation_v3_saturated_interaction_fit():
  rows=[]
  for b in (1023,1281):
    for s in FALSE_SITES:
      rows.append({"base_achieved_uops":b,"false_sites":s,"phases":{"x":{"overhead_corrected_median_ns":2+3*b+4*s+5*b*s}}})
  fit=_interaction_fit(rows,"x")
  assert fit["coefficients_ns"] == pytest.approx([2,3,4,5]) and fit["saturated_design"] is True


def test_invocation_v3_live_amd_contract():
  result=run_invocation_v3(rounds=30,warmups=1,seed=13,system_snapshot_id="sha256:"+"d"*64)
  assert result["cells"]==4 and result["candidate_ids"]==result["candidate_binaries"]==result["candidate_timings"]==[]
  assert len(result["protocol"]["randomized_interleaved_order"])==120
  assert set(result["topology_interaction_fits"])==set(PHASES)
  for row in result["rows"]:
    assert row["candidate_id"] is None and row["topology"]["dynamic_false"] is True
    assert all(len(row["phases"][phase]["samples_ns"])==30 for phase in PHASES)
