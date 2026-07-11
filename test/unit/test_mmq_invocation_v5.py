import pytest
from extra.qk.mmq_invocation_v4 import _resolve_steps
from extra.qk.mmq_invocation_v5 import PHASES,run_invocation_v5,topology_admission

def test_invocation_v5_exact_grouped_admission():
  steps,base=_resolve_steps();admission=topology_admission(steps)
  assert abs(base-405)<=2 and admission["status"]=="admitted" and admission["actual_deltas"]==admission["expected_deltas"]

def test_invocation_v5_floor():
  with pytest.raises(ValueError,match="rounds >= 30"):run_invocation_v5(rounds=29)

def test_invocation_v5_live_contract():
  result=run_invocation_v5(rounds=30,warmups=1,seed=17,system_snapshot_id="sha256:"+"f"*64)
  assert result["topology_admission"]["status"]=="admitted"
  assert result["candidate_ids"]==result["candidate_binaries"]==result["candidate_timings"]==[]
  assert len(result["rows"])==3 and len(result["protocol"]["randomized_interleaved_order"])==90
  assert all(len(row["phases"][phase]["samples_ns"])==30 for row in result["rows"] for phase in PHASES)
