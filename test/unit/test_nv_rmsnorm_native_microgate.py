"""Static contracts for the default-off native NV RMSNorm microgate."""
from pathlib import Path

SRC = Path("extra/llm_research/decode/nv_rmsnorm_native_microgate.py").read_text()

def test_microgate_is_realized_input_aba_and_default_off():
  assert "realized_inputs\":True" in SRC
  assert "boundary_materialization_in_scope\":False" in SRC
  assert "a, b, c = timed(baseline), timed(candidate), timed(baseline)" in SRC
  assert "_rmsnorm_native_promoted" in SRC
  assert "if __name__ == \"__main__\"" in SRC
