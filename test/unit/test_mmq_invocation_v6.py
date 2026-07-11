import pytest
from extra.qk.mmq_invocation_v6 import BACKBONES,PHASES,backbone_admission,run_probe

def test_invocation_v6_exact_admission():
  admission=backbone_admission();assert admission["status"]=="admitted" and admission["failure_audit"]==[]
  assert all(admission["actual"][kind]["total_uops"]==1246 for kind in BACKBONES)
  assert [admission["actual"][kind]["dependency_depth"] for kind in BACKBONES]==[68,40,31]

def test_invocation_v6_floor():
  with pytest.raises(ValueError,match="rounds >= 30"):run_probe(rounds=29)

def test_invocation_v6_live_contract():
  result=run_probe(rounds=30,warmups=1,seed=21,system_snapshot_id="sha256:"+"2"*64)
  assert result["backbone_admission"]["status"]=="admitted" and result["noncandidate_separation"]["all_generated_identity_hashes_unique"]
  assert result["candidate_ids"]==result["candidate_binaries"]==result["candidate_timings"]==[]
  assert len(result["rows"])==3 and len(result["protocol"]["randomized_interleaved_order"])==90
  assert all(len(row["phases"][phase]["samples_ns"])==30 for row in result["rows"] for phase in PHASES)
