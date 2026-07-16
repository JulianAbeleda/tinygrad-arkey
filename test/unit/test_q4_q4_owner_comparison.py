"""Q4-only evidence contract for the sudot4/iu8-WMMA owner decision."""
import inspect

from extra.qk import mmq_q4k_q8_atom as atom
from extra.qk import prefill_int8_wmma_spec as wmma


def test_q4_candidates_have_distinct_generated_instruction_surfaces():
  sudot_source = inspect.getsource(atom._sudot4)
  assert "__builtin_amdgcn_sudot4" in sudot_source

  # The existing WMMA substrate must remain the comparator; this test is
  # deliberately source/contract-only and does not alter route selection.
  wmma_source = inspect.getsource(wmma.emit_q4k_int8_wmma_prefill_tensor)
  assert "matmul" in wmma_source and "iu8" in wmma_source


def test_q4_sudot4_comparator_is_not_claimed_as_shared_memory_wmma():
  source = inspect.getsource(atom._q4k_q8_1_bounded_ds4_dot4x4_kernel)
  assert "_sudot4" in source
  assert "shared_memory_staging" not in source
