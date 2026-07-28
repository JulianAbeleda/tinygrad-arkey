"""Q4-only evidence contract for the sudot4 owner decision."""
import inspect

from extra.llm_research import mmq_q4k_q8_atom as atom


def test_q4_candidates_have_distinct_generated_instruction_surfaces():
  sudot_source = inspect.getsource(atom._sudot4)
  assert "__builtin_amdgcn_sudot4" in sudot_source


def test_q4_sudot4_comparator_is_not_claimed_as_shared_memory_wmma():
  source = inspect.getsource(atom._q4k_q8_1_bounded_ds4_dot4x4_kernel)
  assert "_sudot4" in source
  assert "shared_memory_staging" not in source
