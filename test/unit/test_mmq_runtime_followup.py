import pytest
from extra.qk.mmq_runtime_followup import run_runtime_followup


def test_followup_rejects_invalid_protocol():
  with pytest.raises(ValueError, match="repeat counts"): run_runtime_followup("direct_owner_v0", repeats=(0,), rounds=3, warmups=1, system_snapshot_id="s")


def test_real_followup_binds_compile_once_runtime_and_decomposition():
  result=run_runtime_followup("direct_owner_v0", repeats=(1,2), warmups=1, rounds=3, system_snapshot_id="system-1")
  assert len(result["binary_sha256"]) == 64 and result["protocol"]["source_unrolled"] is False
  assert result["correctness"]["status"] == "PASS"
  assert len(result["decomposition"]["points"]) == 2
  assert result["decomposition"]["points"][0]["gpu_timestamp_per_launch_ms"] > 0
  assert result["production_dispatch_changed"] is False
