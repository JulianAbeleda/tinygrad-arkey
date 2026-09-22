from pathlib import Path

from extra.llm_research.prefill.nv_compiler_q6k_wide_double_buffer import (
  bound_compiler_q6k_wide_k256, transform_compiler_q6k_wide_double_buffer)

FIXTURE = Path("docs/task_workflow/evidence/nv-q6-wide-live-publication-20260831/control-artifacts/wide_direct.cu")


def test_q6_wide_double_buffer_closes_phase_lifetimes():
  if not FIXTURE.exists(): return
  control = bound_compiler_q6k_wide_k256(FIXTURE.read_text())
  candidate = transform_compiler_q6k_wide_double_buffer(control)
  assert "Ridx0 < 4" in control and "#pragma unroll 2" in control
  assert "buf1[40960]" in candidate
  assert candidate.count("signed char *q6_stage") == 2
  assert candidate.count("__syncthreads();") == 1
  assert candidate.count("__WMMA_8_16_32_signed_char_int(") == control.count("__WMMA_8_16_32_signed_char_int(")
  assert "q6_cache" not in candidate
