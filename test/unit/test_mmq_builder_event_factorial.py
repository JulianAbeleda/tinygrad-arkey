import pytest
from extra.qk.mmq_builder_event_factorial import CHANNELS,profile_crosscheck,run_factorial

def test_builder_profile_exact_calls():
  row=profile_crosscheck(8,1,256);assert row["status"]=="PASS"
  assert row["profiled_calls"]=={"group":8,"quant":8,"reduce":1,"eq":512,"store":256,"canonicalization":1}

def test_builder_floor():
  with pytest.raises(ValueError,match="rounds >= 30"):run_factorial(rounds=29)

def test_builder_live_contract():
  result=run_factorial(rounds=30,warmups=1,seed=25,system_snapshot_id="sha256:"+"4"*64)
  assert result["canonical_sink_contract"]["uops"]==1246
  assert result["candidate_ids"]==result["candidate_binaries"]==result["candidate_timings"]==[]
  assert len(result["rows"])==8 and len(result["protocol"]["randomized_interleaved_order"])==240
  assert set(result["factorial_coefficients"])==set(CHANNELS)
  assert all(row["profile_crosscheck"]["status"]=="PASS" for row in result["rows"])
