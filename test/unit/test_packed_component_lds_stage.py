import pytest

from tinygrad import dtypes
from tinygrad.codegen.opt.kernel_lds import (PackedComponentLDSBinding, PackedComponentOperandTemplate,
  PrecontractContractSpec, PrecontractKAxis, PrecontractThreadAxes, build_packed_component_lds_stage)
from tinygrad.codegen.opt.packed_weight import PackedOperandComponent, PackedOperandTransform
from tinygrad.codegen.opt.tc import amd_rdna3
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelLDSComponentWindow, KernelLDSWindow, KernelTileGeometry, Ops, UOp


def _tc(): return next(tc for tc in amd_rdna3 if tc.dtype_in == dtypes.char and tc.dtype_out == dtypes.int)


def _geometry(metadata_dtype=dtypes.half):
  return KernelTileGeometry((128, 128, 32), (2, 2), 128, 32,
    (KernelLDSWindow("A", 0, 6_144, 48), KernelLDSWindow("B", 6_144, 12_288, 48)),
    (KernelLDSComponentWindow("A", "codes", dtypes.char, 0, 4_096, 16, 32),
     KernelLDSComponentWindow("A", "metadata", metadata_dtype, 4_096, 6_144, 16, 16),
     KernelLDSComponentWindow("B", "codes", dtypes.char, 6_144, 10_240, 16, 32),
     KernelLDSComponentWindow("B", "metadata", metadata_dtype, 10_240, 12_288, 16, 16)))


def _transform():
  return PackedOperandTransform("synthetic-int8-components", (
    PackedOperandComponent("codes", dtypes.char, 0, 4_096, "row-major", 32, 16),
    PackedOperandComponent("metadata", dtypes.half, 4_096, 2_048, "row-major", 16, 16)))


def _vector(stride):
  def produce(source, row, k, width):
    return UOp(Ops.STACK, source.ptrdtype.base.vec(width), tuple(source.index(row*stride+k+i).load() for i in range(width)))
  return produce


def _binding(role, component, slot, *, producer=None, source_dtype=None, source_size=None):
  dtype, size, stride = (dtypes.char, 4_096, 32) if component == "codes" else (dtypes.half, 1_024, 8)
  dtype, size = source_dtype or dtype, source_size if source_size is not None else size
  row, k = UOp.range(128, 200+slot*2, AxisType.LOOP), UOp.range(stride, 201+slot*2, AxisType.REDUCE)
  source = UOp.param(slot, dtype.ptr(size))
  return PackedComponentLDSBinding(role, component, source, row, k, UOp.const(dtypes.weakint, 0),
                                   _vector(stride) if producer is None else producer)


def _templates(binding_overrides=None):
  binding_overrides = {} if binding_overrides is None else binding_overrides
  transform = _transform()
  ret = []
  for role, base in (("A", 0), ("B", 2)):
    value = _binding(role, "codes", base, **binding_overrides.get((role, "codes"), {}))
    sidecar = _binding(role, "metadata", base+1, **binding_overrides.get((role, "metadata"), {}))
    ret.append(PackedComponentOperandTemplate(role, transform, value, (sidecar,)))
  return tuple(ret)


def _stage(**overrides):
  tc, geometry, templates = _tc(), _geometry(), _templates()
  threads = PrecontractThreadAxes(UOp.range(2, 220, AxisType.LOCAL), UOp.range(2, 221, AxisType.LOCAL),
                                  UOp.range(32, -1, AxisType.WARP))
  tile_owner, substep_owner = UOp.range(1, 222, AxisType.REDUCE), UOp.range(2, 223, AxisType.UNROLL)
  k_axis = PrecontractKAxis(tile_owner, substep_owner, tile_owner*32, substep_owner)
  sm, sn = UOp.range(4, 224, AxisType.UPCAST), UOp.range(4, 225, AxisType.UPCAST)
  contracts = []
  for operand_idx, role in enumerate(("A", "B")):
    axes = tuple(UOp.range(2, 230+operand_idx*4+i, AxisType.UPCAST) for i in range(4))
    element = ((axes[0]*2+axes[1])*2+axes[2])*2+axes[3]
    contracts.append(PrecontractContractSpec(role, axes, tuple((x.arg[0], 2) for x in axes), element,
      tuple(tc.lane_map.remaps()[operand_idx].items())))
  values = {"geometry":geometry, "tc":tc,
    "allocation":UOp.placeholder((geometry.lds_bytes,), dtypes.uint8, 994, addrspace=AddrSpace.LOCAL),
    "templates":templates, "threads":threads, "k_axis":k_axis, "subtile_m":sm, "subtile_n":sn,
    "contracts":tuple(contracts)} | overrides
  return build_packed_component_lds_stage(**values)


def test_synthetic_int8_components_stage_typed_fragments_and_reachable_half_sidecars():
  stage = _stage()
  assert stage.fragment_a.dtype == stage.fragment_b.dtype == dtypes.char.vec(16)
  stores = [x for x in stage.producer.backward_slice_with_self if x.op is Ops.STORE]
  assert len(stores) == 6
  assert sum(x.src[1].dtype == dtypes.char.vec(16) for x in stores) == 4
  assert sum(x.src[1].dtype == dtypes.half.vec(8) for x in stores) == 2
  assert tuple((x.role, x.component) for x in stage.sidecars) == (("A", "metadata"), ("B", "metadata"))
  assert all(len(x.vectors) == 1 and x.vectors[0].dtype == dtypes.half.vec(8) for x in stage.sidecars)
  graph = UOp.sink(stage.fragment_a, stage.fragment_b, *(v for x in stage.sidecars for v in x.vectors))
  assert all(stage.barrier in vector.backward_slice for x in stage.sidecars for vector in x.vectors)
  assert len([x for x in graph.backward_slice if x.op is Ops.PARAM and x.dtype.base == dtypes.half]) == 2


def test_detached_or_wrong_dtype_vector_producer_is_rejected():
  detached = lambda source, row, k, width: UOp.const(dtypes.char.vec(width), 0)
  with pytest.raises(ValueError, match="detached"):
    _stage(templates=_templates({("A", "codes"):{"producer":detached}}))
  wrong = lambda source, row, k, width: UOp(Ops.STACK, dtypes.half.vec(8),
    tuple(source.index(row*32+k+i).load().cast(dtypes.half) for i in range(8)))
  with pytest.raises(ValueError, match="wrong dtype/vector width"):
    _stage(templates=_templates({("A", "codes"):{"producer":wrong}}))


def test_owned_vector_expression_is_accepted_without_a_callback():
  original = _binding("A", "codes", 0)
  expression = UOp(Ops.STACK, dtypes.char.vec(16), tuple(
    original.source.index(original.row_axis*32+original.k_axis+i).load() for i in range(16)))
  value = PackedComponentLDSBinding("A", "codes", original.source, original.row_axis, original.k_axis,
                                    original.row_tile_base, expression)
  templates = _templates()
  stage = _stage(templates=(PackedComponentOperandTemplate("A", _transform(), value, templates[0].sidecars), templates[1]))
  assert stage.fragment_a.dtype == dtypes.char.vec(16)


def test_binding_rejects_wrong_source_dtype_size_and_component_sum():
  transform = _transform()
  wrong_dtype = _binding("A", "codes", 0, source_dtype=dtypes.uint8)
  with pytest.raises(ValueError, match="source dtype/size"):
    PackedComponentOperandTemplate("A", transform, wrong_dtype, (_binding("A", "metadata", 1),))
  wrong_size = _binding("A", "codes", 0, source_size=4_095)
  with pytest.raises(ValueError, match="source dtype/size"):
    PackedComponentOperandTemplate("A", transform, wrong_size, (_binding("A", "metadata", 1),))
  with pytest.raises(ValueError, match="byte sum"):
    PackedComponentOperandTemplate("A", transform, _binding("A", "codes", 0))


def test_binding_rejects_duplicate_ownership_overlap_and_lds_dtype_mismatch():
  transform, codes = _transform(), _binding("A", "codes", 0)
  with pytest.raises(ValueError, match="unique component ownership"):
    PackedComponentOperandTemplate("A", transform, codes, (codes,))
  with pytest.raises(ValueError, match="overlap"):
    KernelTileGeometry((128, 128, 32), (2, 2), 128, 32,
      (KernelLDSWindow("A", 0, 6_144, 48), KernelLDSWindow("B", 6_144, 12_288, 48)),
      (KernelLDSComponentWindow("A", "codes", dtypes.char, 0, 4_096, 16, 32),
       KernelLDSComponentWindow("A", "metadata", dtypes.half, 4_080, 6_128, 16, 16)))
  with pytest.raises(ValueError, match="LDS dtype"):
    _stage(geometry=_geometry(metadata_dtype=dtypes.float))
