import pytest
from extra.qk.mmq_invocation_v5_scaffolding import PHASES,_resolve_steps,run_probe,topology_admission

def test_scaffolding_exact_full_histogram_admission():
  steps,total=_resolve_steps();admission=topology_admission(steps)
  assert abs(total-1246)<=2 and admission["status"]=="admitted" and admission["failure_audit"]==[]
  assert [admission["tracked_deltas"][p]["CMPLT"] for p in (0,128,256)]==[0,128,256]
  assert all(admission["tracked_deltas"][p][op]==value for p in (0,128,256) for op,value in
             (("STORE",255),("INDEX",255),("AND",256),("CMPNE",64)))

def test_scaffolding_floor():
  with pytest.raises(ValueError,match="rounds >= 30"):run_probe(rounds=29)

def test_scaffolding_live_contract():
  result=run_probe(rounds=30,warmups=1,seed=19,system_snapshot_id="sha256:"+"1"*64)
  assert result["topology_admission"]["status"]=="admitted"
  assert result["candidate_ids"]==result["candidate_binaries"]==result["candidate_timings"]==[]
  assert len(result["rows"])==3 and len(result["protocol"]["randomized_interleaved_order"])==90
  assert all(len(row["phases"][phase]["samples_ns"])==30 for row in result["rows"] for phase in PHASES)
