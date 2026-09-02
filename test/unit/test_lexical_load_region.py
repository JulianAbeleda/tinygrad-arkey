import pytest

from tinygrad import dtypes
from tinygrad.codegen import full_rewrite_to_sink, to_program
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import Target
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import KernelInfo, Ops, PostBarrierRegion, RegionLoad, UOp


TARGET = Target.parse("NV:CUDA:sm_120")


def _ast():
  lid = UOp.special(32, "lidx0")
  source = UOp.placeholder((32,), dtypes.uint, 1)
  scratch = UOp.placeholder((32,), dtypes.uint, 2, addrspace=AddrSpace.LOCAL)
  out = UOp.placeholder((32,), dtypes.uint, 3)
  anchor = UOp(Ops.BARRIER, dtypes.void)
  region = anchor.post_barrier_region(UOp.const(dtypes.bool, True), workgroup_uniform=True)
  loaded = source[lid].load().load_in_region(region)
  stored = scratch[lid].store(loaded)
  ended = region.end_region(stored)
  out[lid].store(loaded)
  return UOp.sink(ended, arg=KernelInfo(name="lexical_load_region", opts_to_apply=()))


def _invalid_ast(case: str):
  lid = UOp.special(32, "lidx0")
  source = UOp.placeholder((32,), dtypes.uint, 11,
    addrspace=AddrSpace.LOCAL if case == "local_source" else AddrSpace.GLOBAL)
  scratch = UOp.placeholder((32,), dtypes.uint, 12, addrspace=AddrSpace.LOCAL)
  out = UOp.placeholder((32,), dtypes.uint, 13)
  region = UOp(Ops.BARRIER, dtypes.void).post_barrier_region(UOp.const(dtypes.bool, True), workgroup_uniform=True)
  load_src = (UOp.const(dtypes.uint, 0),) if case == "masked_load" else ()
  loaded = source[lid].load(*load_src).load_in_region(region)
  destination = out[lid] if case == "global_destination" else scratch[lid]
  stored = destination.store(loaded, gate=(lid < 16) if case == "masked_store" else None)
  roots = [stored]
  suffix = []
  if case == "multiple_consumers": roots.append(scratch[lid+1].store(loaded))
  if case == "mutable_source": suffix.append(source[lid].store(UOp.const(dtypes.uint, 0)))
  owner = (UOp(Ops.BARRIER, dtypes.void).post_barrier_region(UOp.const(dtypes.bool, False), workgroup_uniform=True)
           if case == "wrong_region" else region)
  return UOp.sink(owner.end_region(*roots), *suffix,
                  arg=KernelInfo(name=f"invalid_lexical_load_region_{case}", opts_to_apply=()))


def test_region_load_survives_rewrite_and_is_not_in_index():
  rewritten = full_rewrite_to_sink(_ast(), CUDARenderer(TARGET))
  regions = [u for u in rewritten.toposort() if u.op is Ops.AFTER and isinstance(u.arg, RegionLoad)]
  loads = [u for u in rewritten.toposort() if u.op is Ops.LOAD and any(s in regions for s in u.src[1:])]
  assert len(regions) == 1 and len(loads) == 1
  assert regions[0].src[0].op is Ops.IF and isinstance(regions[0].src[0].arg, PostBarrierRegion)
  assert regions[0] not in loads[0].src[0].backward_slice_with_self


def test_region_load_renders_inside_post_barrier_region():
  program = to_program(_ast(), CUDARenderer(TARGET))
  source = next(x.arg for x in program.src if x.op is Ops.SOURCE)
  lines = source.splitlines()
  barriers = [i for i, line in enumerate(lines) if "__syncthreads" in line]
  loads = [i for i, line in enumerate(lines) if "data1_32" in line and "= *" in line]
  assert barriers[0] < loads[0]
  assert any("if (" in line for line in lines)
  assert not any("val0" in line for line in lines)
  assert any("buf0" in line and "data1_32" in line and "=" in line for line in lines)
  assert "schedule_index" not in source


def test_region_load_fails_closed_on_unsupported_renderer():
  renderer = CUDARenderer(TARGET)
  renderer.supports_region_load = False
  with pytest.raises(RuntimeError, match="cannot preserve lexical load regions"):
    full_rewrite_to_sink(_ast(), renderer)


def test_region_load_api_rejects_non_scalar32_load():
  lid = UOp.special(32, "lidx0")
  source = UOp.placeholder((32,), dtypes.half, 21)
  region = UOp(Ops.BARRIER, dtypes.void).post_barrier_region(UOp.const(dtypes.bool, True), workgroup_uniform=True)
  with pytest.raises(ValueError, match="scalar 32-bit LOAD"):
    source[lid].load().load_in_region(region)


@pytest.mark.parametrize("case,error", (
  ("masked_load", "one unmasked scalar 32-bit LOAD"),
  ("multiple_consumers", "one direct STORE consumer"),
  ("local_source", "GLOBAL source and LOCAL destination"),
  ("global_destination", "GLOBAL source and LOCAL destination"),
  ("masked_store", "one direct STORE consumer"),
  ("mutable_source", "immutable LOAD source"),
  ("wrong_region", "owned by exactly its marker region"),
))
def test_region_load_direct_copy_shape_fails_closed(case, error):
  with pytest.raises(RuntimeError, match=error):
    CUDARenderer(TARGET)._render(_invalid_ast(case).toposort())
