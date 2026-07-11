import pytest
from extra.qk.mmq_invocation_v4 import PHASES,_admission,_resolve_steps,run_invocation_v4

def test_invocation_v4_exact_topology_admission():
  steps,base=_resolve_steps();admission=_admission(steps)
  assert abs(base-405)<=2 and admission["status"]=="admitted" and admission["failure_audit"]==[]
  assert admission["actual_deltas"]==admission["expected_deltas"]

def test_invocation_v4_sample_floor():
  with pytest.raises(ValueError,match="rounds >= 30"):run_invocation_v4(rounds=29)

def test_invocation_v4_live_amd_contract():
  result=run_invocation_v4(rounds=30,warmups=1,seed=15,system_snapshot_id="sha256:"+"e"*64)
  assert result["topology_admission"]["status"]=="admitted"
  assert result["candidate_ids"]==result["candidate_binaries"]==result["candidate_timings"]==[]
  assert len(result["rows"])==3 and len(result["protocol"]["randomized_interleaved_order"])==90
  assert set(result["topology_contrasts"])==set(PHASES)
  assert all(len(row["phases"][phase]["samples_ns"])==30 for row in result["rows"] for phase in PHASES)
