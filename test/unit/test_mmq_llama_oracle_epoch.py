import pytest

from tinygrad import dtypes
from extra.qk.kernel_lds import prove_hierarchical_packed_record_stage
from tinygrad.uop.ops import Ops, UOp

from extra.qk.mmq_llama_oracle_epoch import build_llama_oracle_epoch_stage
from extra.qk.mmq_llama_oracle_recurrence import build_llama_oracle_recurrence, prove_llama_oracle_recurrence


def _stage():
  return build_llama_oracle_epoch_stage(UOp.param(0, dtypes.uint32.ptr(128*36)),
                                        UOp.param(1, dtypes.uint8.ptr(2*128*144)))


def test_exact_record_callbacks_schedules_arena_and_lifecycle_are_one_stage():
  stage = _stage()
  assert stage.geometry.tile == (128, 128, 256) and stage.geometry.waves == (8, 1)
  assert stage.geometry.lds_bytes == 57856
  assert tuple(x.role for x in stage.geometry.lds_windows) == ("B", "A")
  assert [x.cooperative_schedule.name for x in stage.templates] == [
    "llama-load-tiles-q4-k-wave-row-v1", "llama-q8-ds4-linear-256-v1"]
  assert len(stage.phases) == 2 and len(stage.groups) == 8 and len(stage.barriers) == 4
  q4_stores = [x for x in stage.persistent_producer.src if x.op is Ops.STORE]
  assert len(q4_stores) == 36 and sum(x.tag[2] == "qs" for x in q4_stores) == 32
  for phase in stage.phases:
    q8_stores = [x for x in phase.producer.src if x.op is Ops.STORE]
    assert len(q8_stores) == 18
  assert prove_hierarchical_packed_record_stage(stage).passed


def test_exact_record_stage_feeds_connected_oracle_recurrence():
  graph = build_llama_oracle_recurrence(_stage())
  assert [x.k for x in graph.groups] == list(range(0, 256, 32))
  assert len([x for x in graph.consumer_seam.toposort() if x.op is Ops.WMMA]) == 16
  assert graph.phases[0].release in graph.phases[1].producer.backward_slice
  assert graph.phases[1].publish in graph.phases[1].groups[0].wmmas[0].backward_slice
  assert prove_llama_oracle_recurrence(graph).passed


def test_exact_epoch_rejects_split_or_short_physical_sources():
  with pytest.raises(TypeError, match="Q4 source"):
    build_llama_oracle_epoch_stage(UOp.param(0, dtypes.uint32.ptr(36)), UOp.param(1, dtypes.uint8.ptr(2*128*144)))
  with pytest.raises(TypeError, match="Q8 source"):
    build_llama_oracle_epoch_stage(UOp.param(0, dtypes.uint32.ptr(128*36)), UOp.param(1, dtypes.half.ptr(2*128*72)))
