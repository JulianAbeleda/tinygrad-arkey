import numpy as np
import pytest

from extra.qk.benchmark_shared_attention import _full_output_numeric_gate


def test_full_output_gate_records_every_element_and_identity():
  baseline=np.arange(24,dtype=np.float32).reshape(1,2,3,4)
  got=_full_output_numeric_gate(baseline.copy(),baseline,candidate_id="one-pass-lds")
  assert got["status"] == "PASS" and got["full_output"] is True
  assert got["candidate_id"] == "one-pass-lds"
  assert got["candidate_shape"] == got["baseline_shape"] == [1,2,3,4]
  assert got["compared_elements"] == baseline.size


def test_full_output_gate_rejects_shape_or_value_failure():
  with pytest.raises(RuntimeError,match="full-output shapes differ"):
    _full_output_numeric_gate(np.zeros((2,),dtype=np.float32),np.zeros((1,),dtype=np.float32),candidate_id="x")
  with pytest.raises(RuntimeError,match="full-output numeric gate failed"):
    _full_output_numeric_gate(np.ones((4,),dtype=np.float32),np.zeros((4,),dtype=np.float32),candidate_id="x")
