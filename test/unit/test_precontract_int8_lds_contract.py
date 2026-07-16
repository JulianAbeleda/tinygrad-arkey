from types import SimpleNamespace

import pytest

from tinygrad import dtypes
from extra.qk.kernel_lds import wmma_fragment_loads
from tinygrad.codegen.opt.kernel_lds import (PrecontractContractSpec, PrecontractKAxis, PrecontractOperandTemplate,
  PrecontractThreadAxes, build_precontract_lds_stage, derive_precontract_factors, instantiate_precontract_fragments,
  validate_precontract_carriers, validate_precontract_wmma_abi, validate_rdna3_wmma_descriptor)
from tinygrad.codegen.opt.tc import amd_rdna3
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelLDSWindow, KernelTileGeometry, Ops, UOp


def _tc(): return next(tc for tc in amd_rdna3 if tc.dtype_in == dtypes.char and tc.dtype_out == dtypes.int)


def _geometry():
  return KernelTileGeometry((128, 128, 256), (4, 2), 256, 32,
    (KernelLDSWindow("A", 0, 32_768, 256), KernelLDSWindow("B", 32_768, 65_536, 256)))


def _fixture():
  ra, rb = UOp.range(128, 120, AxisType.LOOP), UOp.range(128, 121, AxisType.LOOP)
  ka, kb = UOp.range(1024, 122, AxisType.REDUCE), UOp.range(1024, 123, AxisType.REDUCE)
  a, b = UOp.param(0, dtypes.char.ptr(128*1024)), UOp.param(1, dtypes.char.ptr(128*1024))
  operands = (PrecontractOperandTemplate("A", a.index(ra*1024+ka).load(), ra, ka, UOp.const(dtypes.weakint, 0)),
              PrecontractOperandTemplate("B", b.index(rb*1024+kb).load(), rb, kb, UOp.const(dtypes.weakint, 0)))
  threads = PrecontractThreadAxes(UOp.range(4, 130, AxisType.LOCAL), UOp.range(2, 131, AxisType.LOCAL),
                                  UOp.range(32, -1, AxisType.WARP))
  tile_owner, substep_owner = UOp.range(4, 132, AxisType.REDUCE), UOp.range(16, 133, AxisType.UNROLL)
  k_axis = PrecontractKAxis(tile_owner, substep_owner, tile_owner*256, substep_owner)
  subtile_m, subtile_n = UOp.range(2, 134, AxisType.UPCAST), UOp.range(4, 135, AxisType.UPCAST)
  contracts = []
  for operand_idx, role in enumerate(("A", "B")):
    axes = tuple(UOp.range(2, 140+operand_idx*4+i, AxisType.UPCAST) for i in range(4))
    element = ((axes[0]*2+axes[1])*2+axes[2])*2+axes[3]
    contracts.append(PrecontractContractSpec(role, axes, tuple((x.arg[0], 2) for x in axes), element,
      tuple(_tc().lane_map.remaps()[operand_idx].items())))
  allocation = UOp.placeholder((65_536,), dtypes.char, 994, addrspace=AddrSpace.LOCAL)
  return allocation, operands, threads, k_axis, subtile_m, subtile_n, tuple(contracts)


def _wmma(fragments, contracts, accumulator=None):
  tc = _tc()
  c_axes = ((150, 2), (151, 2), (152, 2))
  arg = (str(tc), tc.dims, tc.dtype_in, tc.dtype_out, "AMD", tc.threads,
         (contracts[0].arg, contracts[1].arg, c_axes), ())
  seed = UOp.const(dtypes.int.vec(8), 0) if accumulator is None else accumulator
  return UOp(Ops.WMMA, dtypes.int.vec(8), (*fragments, seed), arg)


def test_exact_rdna3_int8_descriptor_and_dense_k256_stage_abi():
  tc, geometry = _tc(), _geometry()
  validate_rdna3_wmma_descriptor(tc)
  factors = derive_precontract_factors(geometry, tc)
  assert (factors.k_substeps, factors.vectors_per_row, factors.loads_a, factors.loads_b) == (16, 16, 8, 8)
  allocation, operands, threads, k_axis, sm, sn, contracts = _fixture()
  stage = build_precontract_lds_stage(geometry, tc=tc, allocation=allocation, operands=operands, threads=threads,
                                      k_axis=k_axis, subtile_m=sm, subtile_n=sn, contracts=contracts)
  assert stage.fragment_a.dtype == stage.fragment_b.dtype == dtypes.char.vec(16)
  stores = [x for x in stage.producer.backward_slice_with_self if x.op is Ops.STORE]
  assert len(stores) == 16 and all(x.src[1].dtype == dtypes.char.vec(16) for x in stores)
  node = _wmma((stage.fragment_a, stage.fragment_b), contracts)
  validate_precontract_wmma_abi(node)
  assert node.dtype == node.src[2].dtype == dtypes.int.vec(8)


def test_32_wide_native_outputs_leave_a_float_group_epilogue_boundary():
  tc, geometry = _tc(), _geometry()
  allocation, operands, threads, _, sm, sn, contracts = _fixture()
  ready = UOp.barrier(UOp.group())
  fragments = [instantiate_precontract_fragments(geometry, tc=tc, allocation=allocation, threads=threads,
    k_substep=UOp.const(dtypes.weakint, substep), subtile_m=sm, subtile_n=sn, contracts=contracts,
    epoch=UOp.const(dtypes.weakint, 0), slot=UOp.const(dtypes.weakint, 0), ready=ready).fragments for substep in (0, 1)]
  native = tuple(_wmma(pair, contracts) for pair in fragments)
  for node in native: validate_precontract_wmma_abi(node)
  # Two K=16 native results form a reusable 32-wide boundary. Scaling/correction and the long-lived sum are float-owned.
  group_sum = native[0].cast(dtypes.float.vec(8)) + native[1].cast(dtypes.float.vec(8))
  assert group_sum.dtype == dtypes.float.vec(8)
  assert all(node.src[2].op is not Ops.WMMA for node in native)


def test_int8_fragment_offsets_are_byte_addressed():
  loads = wmma_fragment_loads(_geometry(), "B", tc=_tc())
  first = next(x for x in loads if x.thread == 0 and x.subtile == 0 and x.k_substep == 0 and x.element == 0)
  second = next(x for x in loads if x.thread == 0 and x.subtile == 0 and x.k_substep == 0 and x.element == 1)
  assert first.byte_offset == 32_768 and second.byte_offset == first.byte_offset + 1


def test_int8_abi_rejects_mixed_carriers_and_incorrect_layouts():
  tc = _tc()
  validate_precontract_carriers(dtypes.char.vec(16), dtypes.int.vec(8), tc=tc)
  with pytest.raises(ValueError, match="fragment carrier"):
    validate_precontract_carriers(dtypes.half.vec(16), dtypes.int.vec(8), tc=tc)
  with pytest.raises(ValueError, match="accumulator carrier"):
    validate_precontract_carriers(dtypes.char.vec(16), dtypes.float.vec(8), tc=tc)
  _, _, _, _, _, _, contracts = _fixture()
  char_fragment, half_fragment = UOp.const(dtypes.char.vec(16), 0), UOp.const(dtypes.half.vec(16), 0)
  with pytest.raises(ValueError, match="B fragment"):
    validate_precontract_wmma_abi(_wmma((char_fragment, half_fragment), contracts))
  drift = SimpleNamespace(**{name:getattr(tc, name) for name in
    ("dims", "threads", "elements_per_thread", "dtype_in", "dtype_out", "opts", "lane_map")}, swizzle=(((), (), ()), ((), (), ())))
  with pytest.raises(ValueError, match="swizzle drifted"): validate_rdna3_wmma_descriptor(drift)
  bad_geometry = KernelTileGeometry((128, 128, 256), (4, 2), 256, 32,
    (KernelLDSWindow("A", 0, 30_720, 240), KernelLDSWindow("B", 30_720, 61_440, 240)))
  with pytest.raises(ValueError, match="padded operand rows"): derive_precontract_factors(bad_geometry, tc)
