import pytest

from tinygrad import dtypes
from tinygrad.uop.ops import Ops, UOp

from extra.qk.mmq_llama_oracle_epoch import (build_llama_oracle_epoch_stage,
  build_llama_oracle_epoch_stage_five_buffer)
from extra.qk.mmq_llama_oracle_recurrence import build_llama_oracle_recurrence, prove_llama_oracle_recurrence


def _sources():
  return (UOp.param(0, dtypes.uint32.ptr(128*36)), UOp.param(1, dtypes.int8.ptr(2*128*128)),
          UOp.param(2, dtypes.float32.ptr(2*128*4)), UOp.param(3, dtypes.float32.ptr(2*128*4)))


def test_five_buffer_epoch_preserves_geometry_q4_path_and_exact_signed_recurrence():
  q4, values, scales, sums = _sources()
  split = build_llama_oracle_epoch_stage_five_buffer(q4, values, scales, sums)
  interleaved = build_llama_oracle_epoch_stage(q4, UOp.param(4, dtypes.uint8.ptr(2*128*144)))
  assert split.geometry == interleaved.geometry
  assert split.geometry.tile == (128, 128, 256) and split.geometry.lds_bytes == 57856
  assert split.descriptor == interleaved.descriptor and split.contracts == interleaved.contracts
  assert split.templates[0].transform is interleaved.templates[0].transform
  assert split.templates[0].cooperative_schedule is interleaved.templates[0].cooperative_schedule
  assert split.templates[1].transform.produced is interleaved.templates[1].transform.produced
  graph = build_llama_oracle_recurrence(split)
  wmmas = [x for x in graph.consumer_seam.toposort() if x.op is Ops.WMMA]
  assert len(wmmas) == 16 and all(x.src[0].dtype == x.src[1].dtype == dtypes.char.vec(16) for x in wmmas)
  assert prove_llama_oracle_recurrence(graph).passed


@pytest.mark.parametrize("index,bad", [
  (0, UOp.param(8, dtypes.uint32.ptr(36))),
  (1, UOp.param(8, dtypes.int8.ptr(2*128*128-1))),
  (2, UOp.param(8, dtypes.half.ptr(2*128*4))),
  (3, UOp.param(8, dtypes.float32.ptr(2*128*4-1))),
])
def test_five_buffer_epoch_rejects_wrong_or_short_physical_sources(index, bad):
  args = list(_sources())
  args[index] = bad
  with pytest.raises(TypeError): build_llama_oracle_epoch_stage_five_buffer(*args)
