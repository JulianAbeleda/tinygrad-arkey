from tinygrad import dtypes
from tinygrad.uop import Ops
from tinygrad.schedule.wmma import amd_gfx1100_rotating_pv_scheduler_probe
from tinygrad.uop.ops import ParamArg, UOp
from tinygrad.uop.spec import spec_full, type_verify
from extra.qk.rotating_pv_abi import rotating_pv_kernel_probe
import pytest


def test_rotating_pv_scheduler_probe_builds_one_ordered_kv_body_and_drain():
  out = UOp(Ops.PARAM, dtypes.half.ptr(512*32*128), arg=ParamArg(0))
  q = UOp(Ops.PARAM, dtypes.half.ptr(512*32*128), arg=ParamArg(1))
  k = UOp(Ops.PARAM, dtypes.half.ptr(512*8*128), arg=ParamArg(2))
  v = UOp(Ops.PARAM, dtypes.half.ptr(512*8*128), arg=ParamArg(3))
  sink = amd_gfx1100_rotating_pv_scheduler_probe(q, k, v, out, q_tokens=512, q_heads=32, kv_heads=8, kv_tokens=512)
  topo = sink.toposort()
  end = next(x for x in topo if x.op is Ops.END and x.tag[0] == "rotating_pv_kv_iteration_end_v1")
  writes = end.src[0].src
  assert len(writes) == 8 and all(x.arg[0] == "rotating_pv_state_write_v1" for x in writes)
  assert {x.arg[1].block for x in writes} == set(range(8)) and {x.arg[1].generation for x in writes} == {1}
  assert all(x.src[0].op is Ops.WMMA and x.src[0].tag == ("attention_wmma", "PV", block) and
             x.src[0].src[2].src[0].arg[0] == "rotating_pv_loop_read_v1" for block,x in enumerate(writes))
  assert all(x.src[-1] is end.src[1] for x in writes)
  drains = [x for x in topo if x.op is Ops.CUSTOMI and x.arg[0] == "rotating_pv_sequential_drain_v1"]
  assert sum(x.op is Ops.WMMA and x.tag[1] == "QK" for x in topo) == 8
  assert sum(x.op is Ops.WMMA and x.tag[1] == "PV" for x in topo) == 8
  assert len(drains) == 8 and drains[0].src[5] is end and all(drain.src[0] is out for drain in drains)
  assert all(drains[block].src[5] is drains[block-1] for block in range(1, 8))
  assert all(drain.shape == () and drain.tag == ("amd_gfx1100_rotating_pv_sequential_drain_v1", block, 0, block, 1)
             for block,drain in enumerate(drains))
  type_verify(sink, spec_full)


def test_rotating_pv_kernel_probe_wraps_exact_scheduler_sink():
  probe = rotating_pv_kernel_probe()
  assert probe["status"] == "CONSTRUCTED" and not probe["promotion_eligible"]
  assert probe["geometry"] == {"q_tokens": 512, "q_heads": 32, "kv_heads": 8, "kv_tokens": 512, "head_dim": 128}
  assert probe["sink"].arg.name == "rotating_pv_scheduler_probe"


def test_rotating_pv_exact_isa_reaches_resource_gate_after_fragment_lowering():
  from tinygrad.codegen import to_program
  from tinygrad.helpers import Target
  from tinygrad.renderer.isa.amd import AMDISARenderer
  sink = rotating_pv_kernel_probe()["sink"]
  try:
    to_program(sink, AMDISARenderer(Target.parse("AMD:ISA:gfx1100")))
  except NotImplementedError as exc:
    assert "spill-free VGPR/SGPR budget" in str(exc)
  except RuntimeError as exc:
    assert "lazy opaque fragment lowering produced unselected" not in str(exc)
