import pytest
from extra.qk.mmq_host_structure_audit import DEPTHS,FANOUTS,run_audit,structure_admission

def test_structure_exact_admission():
  admission=structure_admission();assert admission["status"]=="admitted" and admission["failure_audit"]==[]
  assert all(row["uops"]==1246 and row["edges"]==3120 and row["shared_extra_edges"]==1875 for row in admission["actual"].values())

def test_structure_floor():
  with pytest.raises(ValueError,match="rounds >= 30"):run_audit(rounds=29)

def test_structure_live_contract():
  result=run_audit(rounds=30,warmups=1,seed=23,system_snapshot_id="sha256:"+"3"*64)
  assert result["candidate_ids"]==result["candidate_binaries"]==result["candidate_timings"]==[]
  assert len(result["rows"])==4 and len(result["protocol"]["randomized_interleaved_order"])==120
  assert {(r["depth"],r["fanout"]) for r in result["rows"]}==set((d,f) for d in DEPTHS for f in FANOUTS)
  assert all(len(r["construction"]["samples_ns"])==30 for r in result["rows"])
