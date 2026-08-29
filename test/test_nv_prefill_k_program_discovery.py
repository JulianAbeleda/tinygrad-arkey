"""Regression for the captured combined-prefill K asset prerequisite."""
import os
import pytest

def test_combined_prefill_k_program_discovery():
  if os.environ.get("RUN_NV_PREFILL_GPU_TESTS") != "1": pytest.skip("set RUN_NV_PREFILL_GPU_TESTS=1")
  from extra.llm_research.prefill.nv_compiler_q4k_k_pp512_binding import binding_for
  binding = binding_for("NV")
  assert binding.producer is not None
  binding.prepare_records(36)
  assert getattr(binding, "records", None) is not None or getattr(binding.asset, "records", None) is not None
