import pytest
from extra.qk.mmq_invocation_v7 import CHANNELS,profile_crosscheck,run_probe,topology_admission

def test_invocation_v7_exact_topology_and_calls():
  admission=topology_admission();assert admission["status"]=="admitted" and admission["failure_audit"]==[]
  assert profile_crosscheck("candidate_shaped",256)["status"]=="PASS"

def test_invocation_v7_floor():
  with pytest.raises(ValueError,match="rounds >= 30"):run_probe(rounds=29)

def test_invocation_v7_live_contract():
  result=run_probe(rounds=30,warmups=1,seed=27,system_snapshot_id="sha256:"+"5"*64)
  assert result["topology_admission"]["status"]=="admitted" and result["canonical_graph_contract"]["uops"]==1246
  assert result["candidate_ids"]==result["candidate_binaries"]==result["candidate_timings"]==[]
  assert len(result["rows"])==4 and len(result["protocol"]["randomized_interleaved_order"])==120
  assert set(result["factorial_contrasts"])==set(CHANNELS)
  assert all(row["profile_crosscheck"]["status"]=="PASS" for row in result["rows"])
