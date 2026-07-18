import re

import pytest

from tinygrad.codegen import line_rewrite
from tinygrad.codegen.late.regalloc import LinearScanRegallocContext, pm_regalloc_rewrite
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.helpers import Target
from tinygrad.renderer.isa import IselContext, PreRegAllocContext, RegisterSpan
from tinygrad.renderer.isa.amd import (AMDISARenderer, AMDOps, SPTR_POOL, VBASE, isel_index, isel_load,
                                      post_regalloc_matcher, pre_regalloc_matcher, _workgroup_sgpr_index)
from tinygrad.uop.ops import Ops, UOp


@pytest.mark.parametrize("dims,expected", (((0,), (2,)), ((1,), (2,)), ((0, 1), (2, 3))))
def test_workgroup_id_system_sgprs_are_packed_by_enabled_dimension(dims, expected):
  specials = tuple(UOp.special(2, f"gidx{d}") for d in dims)
  ctx = IselContext(UOp.sink(*specials))
  assert tuple(_workgroup_sgpr_index(ctx, d) for d in dims) == expected


def _select_global_load(dtype, index:int=0):
  ctx = IselContext(UOp.sink())
  ptr_dtype = dtype.scalar().ptr(size=64)
  ptr = UOp(Ops.INS, ptr_dtype, (UOp.const(dtypes.int32, 0).rtag(),), AMDOps.S_LOAD_PTR, tag=(ctx.vreg(SPTR_POOL),))
  idxc = isel_index(ctx, ptr.index(UOp.const(dtypes.int32, index), ptr=True))
  assert idxc is not None
  return ctx, isel_load(ctx, UOp(Ops.LOAD, dtype, (idxc,)))


def _compile_lane_uses(ctx:IselContext, carrier:UOp):
  uses = tuple(UOp(Ops.INS, dtypes.uint32, (lane, UOp.const(dtypes.int32, 100+i).rtag()), AMDOps.V_OR,
                   tag=(ctx.vreg(VBASE[1:]),)) for i, lane in enumerate(carrier.src))
  ren = AMDISARenderer(Target.parse("AMD:ISA:gfx1100"))
  lst = line_rewrite(list(UOp.sink(*uses).toposort()), pre_regalloc_matcher, PreRegAllocContext())
  regalloc = LinearScanRegallocContext(lst, ren)
  lst = line_rewrite(lst, pm_regalloc_rewrite, regalloc)
  lst = line_rewrite(lst, post_regalloc_matcher, regalloc)
  encoded = [u.arg for u in lst if u.op is Ops.INS and not isinstance(u.arg, tuple)]
  assert all(inst.to_bytes() for inst in encoded)
  return regalloc, [str(inst) for inst in encoded]


@pytest.mark.parametrize("dtype,mnemonic,nregs", (
  (dtypes.uint32.vec(2), "global_load_b64", 2),
  (dtypes.uint32.vec(4), "global_load_b128", 4),
))
def test_uint32_wide_load_owns_consecutive_dword_lanes(dtype, mnemonic, nregs):
  ctx, carrier = _select_global_load(dtype)
  owner = carrier.src[0].src[0]
  assert owner.arg is (AMDOps.GLOBAL_LOAD_B64 if nregs == 2 else AMDOps.GLOBAL_LOAD_B128_GENERIC)
  assert owner.tag[0].span == RegisterSpan(nregs)

  regalloc, asm = _compile_lane_uses(ctx, carrier)
  assert not regalloc.spills and regalloc.stack_size == 0
  loads = [line for line in asm if line.startswith(mnemonic)]
  assert len(loads) == 1
  base, end = map(int, re.search(r"v\[(\d+):(\d+)\]", loads[0]).groups())
  assert end - base + 1 == nregs
  lane_sources = [int(re.search(r", LIT, v\[(\d+)\], 10\d\)$", line).group(1))
                  for line in asm if line.startswith("v_or_b32_e32") and re.search(r", LIT, v\[\d+\], 10\d\)$", line)]
  assert lane_sources == list(range(base, base+nregs))


@pytest.mark.parametrize("dtype,mnemonic,nregs", (
  (dtypes.uint16.vec(4), "global_load_b64", 2),
  (dtypes.uint16.vec(8), "global_load_b128", 4),
))
def test_uint16_geps_extract_low_then_high_from_consecutive_dwords(dtype, mnemonic, nregs):
  ctx, carrier = _select_global_load(dtype)
  owner = carrier.src[0].src[0].src[0]
  assert owner.tag[0].span == RegisterSpan(nregs)

  regalloc, asm = _compile_lane_uses(ctx, carrier)
  assert not regalloc.spills and regalloc.stack_size == 0
  load = next(line for line in asm if line.startswith(mnemonic))
  base, end = map(int, re.search(r"v\[(\d+):(\d+)\]", load).groups())
  assert end - base + 1 == nregs
  extracts = [tuple(map(int, re.search(r"v_bfe_u32\(v\[\d+\], v\[(\d+)\], (\d+), 16\)", line).groups()))
              for line in asm if line.startswith("v_bfe_u32")]
  assert extracts == [(base+word, half*16) for word in range(nregs) for half in range(2)]


def test_unsupported_width_alignment_and_non_integer_keep_scalar_fallback():
  for dtype, index in ((dtypes.uint32, 0), (dtypes.uint32.vec(3), 0), (dtypes.uint32.vec(4), 1), (dtypes.float32.vec(4), 0)):
    _ctx, selected = _select_global_load(dtype, index)
    loads = (selected,) if selected.op is Ops.INS else selected.src
    assert len(loads) == dtype.count
    assert all(load.op is Ops.INS and load.arg is AMDOps.GLOBAL_LOAD for load in loads)


def test_lds_integer_vector_keeps_ds_scalar_fallback():
  ctx = IselContext(UOp.sink())
  addr = UOp(Ops.INS, dtypes.int32, arg=AMDOps.V_MOVK, tag=(ctx.vreg(VBASE[1:]),))
  local = UOp(Ops.NOOP, dtypes.uint32.ptr(addrspace=AddrSpace.LOCAL), (addr, UOp(Ops.NOOP)), arg="lds")
  selected = isel_load(ctx, UOp(Ops.LOAD, dtypes.uint32.vec(2), (local,)))
  assert selected.op is Ops.NOOP and all(load.arg is AMDOps.DS_LOAD for load in selected.src)
