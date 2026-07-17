import numpy as np
import pytest

from extra.qk.mmq_llama_lds_roundtrip_gpu_harness import (_blocked, _probe_inputs, run_amd_validation,
  BLOCKED, PROTOCOL, SHAPE)
from extra.qk.mmq_llama_lds_roundtrip_probe import (DEBUG_WORDS, compare_llama_lds_roundtrip,
  expected_llama_lds_roundtrip)


def test_adapter_reuses_deterministic_five_buffer_input_conventions():
  words, values, scales, sums = _probe_inputs()
  assert words.dtype == np.uint32 and words.size == 128*36
  assert values.dtype == np.int8 and values.shape == (2, 128, 128)
  assert scales.dtype == sums.dtype == np.float32 and scales.shape == sums.shape == (2, 128, 4)
  assert expected_llama_lds_roundtrip(words, values, scales, sums).size == DEBUG_WORDS


def test_adapter_blocker_and_timeout_are_structured_without_dispatch():
  row = _blocked("test blocker", sample=1)
  assert row == {"protocol": PROTOCOL, "shape": list(SHAPE), "passed": False,
                 "verdict": BLOCKED, "blocker": "test blocker", "evidence": {"sample": 1}}
  timeout = run_amd_validation(timeout_seconds=0)
  assert timeout["verdict"] == BLOCKED and timeout["blocker"] == "timeout_seconds must be positive"


def test_adapter_comparison_preserves_segment_row_word_mismatch_contract():
  words, values, scales, sums = _probe_inputs()
  expected = expected_llama_lds_roundtrip(words, values, scales, sums)
  output = np.zeros(128*128, dtype=np.float32)
  output.view(np.uint32)[:DEBUG_WORDS] = expected
  output.view(np.uint32)[12800 + 4*8 + 2] ^= np.uint32(1)
  result = compare_llama_lds_roundtrip(output, expected)
  assert result["passed"] is False and result["mismatch_count"] == 1
  assert result["first_mismatch"] == {"output_word": 12834, "segment": "q4_persistent_dm",
    "row": 4, "word_in_row": 2, "actual": int(output.view(np.uint32)[12834]),
    "expected": int(expected[12834])}


@pytest.mark.parametrize("timeout", [-1, 0])
def test_adapter_rejects_nonpositive_timeout(timeout):
  assert run_amd_validation(timeout_seconds=timeout)["passed"] is False
