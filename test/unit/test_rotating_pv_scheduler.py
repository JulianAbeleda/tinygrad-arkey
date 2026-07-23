from tinygrad.uop import Ops
from tinygrad.schedule.wmma import amd_gfx1100_rotating_pv_scheduler_probe
from tinygrad.uop.spec import spec_full, type_verify


def test_rotating_pv_scheduler_probe_builds_one_ordered_kv_body_and_drain():
  sink = amd_gfx1100_rotating_pv_scheduler_probe(q_tokens=512, q_heads=32, kv_heads=8, kv_tokens=512)
  topo = sink.toposort()
  end = next(x for x in topo if x.op is Ops.END and x.tag[0] == "rotating_pv_kv_iteration_end_v1")
  writes = end.src[0].src
  assert len(writes) == 8 and all(x.arg[0] == "rotating_pv_state_write_v1" for x in writes)
  assert {x.arg[1].block for x in writes} == set(range(8)) and {x.arg[1].generation for x in writes} == {1}
  assert all(x.src[0].src[0].arg[0] == "rotating_pv_state_read_v1" for x in writes)
  assert all(x.src[-1] is end.src[1] for x in writes)
  drains = [x for x in topo if x.op is Ops.CUSTOMI and x.arg[0] == "rotating_pv_sequential_drain_v1"]
  assert len(drains) == 8 and drains[0].src[2] is end
  assert all(drains[block].src[2] is drains[block-1] for block in range(1, 8))
  assert all(drain.shape == () and drain.tag == ("rotating_pv_sequential_drain_v1", block, 1) for block,drain in enumerate(drains))
  type_verify(sink, spec_full)
