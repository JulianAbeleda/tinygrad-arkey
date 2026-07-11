import pytest
from extra.qk.mmq_invocation_phases import run_invocation_phase_probe
def test_phase_probe_enforces_sample_floor():
  with pytest.raises(ValueError,match="at least 30"):run_invocation_phase_probe(rounds=3,system_snapshot_id="s")
