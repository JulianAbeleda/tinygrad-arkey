import numpy as np

from tinygrad import dtypes
from tinygrad.uop.ops import Ops, UOp

from extra.qk.mmq_llama_five_buffer_full_kernel import build_llama_five_buffer_full_kernel
from extra.qk.mmq_llama_five_buffer_gpu_harness import (_bind_sink, _random_q4_words,
  run_amd_validation)


def test_gpu_harness_random_q4_fixture_has_independent_abi_shape():
  words = _random_q4_words(128, 256, 20260717)
  assert words.dtype == np.uint32 and words.shape == (128 * 36,)
  assert np.isfinite(words.view(np.uint8)).all()


def test_gpu_harness_binds_exact_five_buffer_slots_without_reauthoring_graph():
  sink = build_llama_five_buffer_full_kernel(128, 128, 256).sink
  args = tuple(UOp.placeholder((size,), dtype, slot) for slot, (size, dtype) in enumerate(
    ((128 * 128, dtypes.float32), (128 * 36, dtypes.uint32),
     (2 * 128 * 128, dtypes.int8), (2 * 128 * 4, dtypes.float32),
     (2 * 128 * 4, dtypes.float32))))
  bound = _bind_sink(sink, args)
  params = {u.arg.slot for u in bound.toposort() if u.op is Ops.PARAM}
  assert params == set(range(5))
  assert bound.arg.name == "mmq_llama_five_buffer_full_grid"


def test_gpu_harness_timeout_path_fails_closed_without_gpu_access():
  row = run_amd_validation(timeout_seconds=0)
  assert row["passed"] is False
  assert row["verdict"] == "MMQ_LLAMA_FIVE_BUFFER_GPU_BLOCKED"
  assert row["blocker"] == "timeout_seconds must be positive"

