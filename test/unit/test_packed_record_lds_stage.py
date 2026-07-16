import pytest

from tinygrad import dtypes
from tinygrad.codegen.opt.kernel_lds import (PackedRecordFieldProducer, PackedRecordLDSRegionBinding,
  PackedRecordOperandTemplate, PackedRecordSource, PrecontractContractSpec, PrecontractKAxis, PrecontractThreadAxes,
  build_packed_record_lds_stage)
from tinygrad.codegen.opt.packed_weight import PackedOperandComponent, PackedOperandRecordTransform, PackedOperandTransform
from tinygrad.codegen.opt.tc import amd_rdna3
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import (AxisType, KernelLDSArenaRegion, KernelLDSRecordComponent, KernelLDSRecordLayout,
                              KernelLDSWindow, KernelTileGeometry, Ops, UOp)


def _tc(): return next(tc for tc in amd_rdna3 if tc.dtype_in == dtypes.char and tc.dtype_out == dtypes.int)


def _geometry():
  q8 = KernelLDSRecordLayout(128, 144, (
    KernelLDSRecordComponent("ds", dtypes.half, 0, 16, 16),
    KernelLDSRecordComponent("qs", dtypes.int8, 16, 128, 16)))
  q4 = KernelLDSRecordLayout(128, 304, (
    KernelLDSRecordComponent("qs", dtypes.int32, 0, 256, 16),
    KernelLDSRecordComponent("dm", dtypes.half, 256, 32, 16),
    KernelLDSRecordComponent("padding", dtypes.uint8, 288, 16, 16)))
  return KernelTileGeometry((128, 128, 32), (2, 2), 128, 32,
    (KernelLDSWindow("A", 512, 18_944, 144), KernelLDSWindow("B", 18_944, 57_856, 304)), (),
    (KernelLDSArenaRegion("ids", 0, 512), KernelLDSArenaRegion("q8", 512, 18_944, records=q8),
     KernelLDSArenaRegion("q4", 18_944, 57_856, records=q4)))


def _transform(role):
  source = PackedOperandTransform(f"{role}-packed", (PackedOperandComponent("record", dtypes.uint8, 0, 144, alignment=16),))
  produced = PackedOperandTransform(f"{role}-decoded", (
    (PackedOperandComponent("ds", dtypes.half, 0, 16, "record", 144, 16),
     PackedOperandComponent("qs", dtypes.int8, 16, 128, "record", 144, 16)) if role == "A" else
    (PackedOperandComponent("qs", dtypes.int32, 0, 256, "record", 304, 16),
     PackedOperandComponent("dm", dtypes.half, 256, 32, "record", 304, 16),
     PackedOperandComponent("padding", dtypes.uint8, 288, 16, "record", 304, 16))))
  return PackedOperandRecordTransform(f"{role}-record", source, produced)


def _vector(dtype, source_stride=144):
  def produce(sources, row, k, width):
    source = sources[0]
    return UOp(Ops.STACK, dtype.vec(width), tuple(source.index(row*source_stride+k+i).load().cast(dtype) for i in range(width)))
  return produce


def _templates(*, q4_qs_producer=None, q4_reserved=("padding",), q4_fields=None, q4_source=None):
  ret = []
  for role, slot in (("A", 0), ("B", 1)):
    row, k = UOp.range(128, 300+slot*2, AxisType.LOOP), UOp.range(256, 301+slot*2, AxisType.REDUCE)
    source = q4_source if role == "B" and q4_source is not None else UOp.param(slot, dtypes.uint8.ptr(144*128))
    fields = ((PackedRecordFieldProducer("ds", ("record",), _vector(dtypes.half)),
               PackedRecordFieldProducer("qs", ("record",), _vector(dtypes.int8))) if role == "A" else
              (PackedRecordFieldProducer("qs", ("record",), q4_qs_producer or _vector(dtypes.int32)),
               PackedRecordFieldProducer("dm", ("record",), _vector(dtypes.half))))
    if role == "B" and q4_fields is not None: fields = q4_fields
    ret.append(PackedRecordOperandTemplate(role, _transform(role), (PackedRecordSource("record", source),), fields,
      () if role == "A" else q4_reserved, "qs", row, k, UOp.const(dtypes.weakint, 0), dtypes.char))
  return tuple(ret)


def _stage(**overrides):
  tc, geometry = _tc(), _geometry()
  threads = PrecontractThreadAxes(UOp.range(2, 320, AxisType.LOCAL), UOp.range(2, 321, AxisType.LOCAL),
                                  UOp.range(32, -1, AxisType.WARP))
  tile_owner, substep_owner = UOp.range(1, 322, AxisType.REDUCE), UOp.range(2, 323, AxisType.UNROLL)
  sm, sn = UOp.range(4, 324, AxisType.UPCAST), UOp.range(4, 325, AxisType.UPCAST)
  contracts = []
  for operand_idx, role in enumerate(("A", "B")):
    axes = tuple(UOp.range(2, 330+operand_idx*4+i, AxisType.UPCAST) for i in range(4))
    element = ((axes[0]*2+axes[1])*2+axes[2])*2+axes[3]
    contracts.append(PrecontractContractSpec(role, axes, tuple((x.arg[0], 2) for x in axes), element,
      tuple(tc.lane_map.remaps()[operand_idx].items())))
  values = {"geometry":geometry, "tc":tc,
    "allocation":UOp.placeholder((57_856,), dtypes.uint8, 995, addrspace=AddrSpace.LOCAL),
    "templates":_templates(), "regions":(PackedRecordLDSRegionBinding("A", "q8"), PackedRecordLDSRegionBinding("B", "q4")),
    "threads":threads, "k_axis":PrecontractKAxis(tile_owner, substep_owner, tile_owner*32, substep_owner),
    "subtile_m":sm, "subtile_n":sn, "contracts":tuple(contracts)} | overrides
  return build_packed_record_lds_stage(**values)


def test_exact_q8_q4_interleaved_stage_shared_source_typed_views_and_barrier():
  templates = _templates()
  assert templates[1].source("record") is templates[1].source("record")
  stage = _stage(templates=templates)
  assert stage.fragment_a.dtype == stage.fragment_b.dtype == dtypes.int8.vec(16)
  fragment_loads = [x for fragment in (stage.fragment_a, stage.fragment_b) for x in fragment.backward_slice_with_self
                    if x.op is Ops.LOAD and getattr(x.src[0], "tag", None) and x.src[0].tag[0] == "packed_record_fragment_load"]
  assert len(fragment_loads) == 2 and all(x.dtype == dtypes.char for x in fragment_loads)
  stores = [x for x in stage.producer.backward_slice_with_self if x.op is Ops.STORE]
  # Q8 ds=1, qs=8 and Q4 qs=16, dm=2 cooperative rounds.
  assert len(stores) == 27
  assert sum(x.tag[1:3] == ("B", "qs") and x.src[1].dtype == dtypes.int32.vec(4) for x in stores) == 16
  assert all(x.src[1].dtype == dtypes.int8.vec(16) for x in stores if x.tag[1:3] == ("A", "qs"))
  assert tuple((x.role, x.field, len(x.vectors), x.vectors[0].dtype) for x in stage.sidecars) == (
    ("A", "ds", 1, dtypes.half.vec(8)), ("B", "dm", 2, dtypes.half.vec(8)))
  assert all(stage.barrier in vector.backward_slice for x in stage.sidecars for vector in x.vectors)
  assert stage.barrier in stage.fragment_a.backward_slice and stage.barrier in stage.fragment_b.backward_slice
  tags = {x.tag[:3] for x in stores}
  assert tags == {("packed_record_store", "A", "ds"), ("packed_record_store", "A", "qs"),
                  ("packed_record_store", "B", "qs"), ("packed_record_store", "B", "dm")}


def test_field_vector_count_uses_size_not_304_byte_row_stride():
  stage = _stage()
  stores = [x for x in stage.producer.backward_slice_with_self if x.op is Ops.STORE]
  assert sum(x.tag[1:3] == ("B", "dm") for x in stores) == 2
  assert len(next(x for x in stage.sidecars if (x.role, x.field) == ("B", "dm")).vectors) == 2


def test_missing_duplicate_or_reserved_field_producer_fails_closed():
  qs, dm = PackedRecordFieldProducer("qs", ("record",), _vector(dtypes.int32)), PackedRecordFieldProducer("dm", ("record",), _vector(dtypes.half))
  with pytest.raises(ValueError, match="exactly one producer"):
    _templates(q4_fields=(qs,), q4_reserved=("padding",))
  with pytest.raises(ValueError, match="duplicate packed record field producer"):
    _templates(q4_fields=(qs, qs, dm))
  with pytest.raises(ValueError, match="cannot be produced"):
    _templates(q4_fields=(qs, dm, PackedRecordFieldProducer("padding", ("record",), _vector(dtypes.uint8))))


def test_undeclared_detached_and_wrong_typed_producers_fail_closed():
  detached = lambda sources, row, k, width: UOp.const(dtypes.int32.vec(width), 0)
  with pytest.raises(ValueError, match="detached"):
    _stage(templates=_templates(q4_qs_producer=detached))
  wrong = lambda sources, row, k, width: UOp(Ops.STACK, dtypes.half.vec(width), tuple(
    sources[0].index(row*144+k+i).load().cast(dtypes.half) for i in range(width)))
  with pytest.raises(ValueError, match="wrong dtype/vector width"):
    _stage(templates=_templates(q4_qs_producer=wrong))
  foreign = UOp.param(9, dtypes.uint8.ptr(144*128))
  undeclared = lambda sources, row, k, width: UOp(Ops.STACK, dtypes.int32.vec(width), tuple(
    (sources[0].index(row*144+k+i).load() + foreign.index(UOp.const(dtypes.weakint, i)).load()).cast(dtypes.int32) for i in range(width)))
  with pytest.raises(ValueError, match="undeclared source"):
    _stage(templates=_templates(q4_qs_producer=undeclared))


def test_layout_and_region_mismatches_fail_before_address_generation():
  geometry = _geometry()
  bad_q4 = KernelLDSArenaRegion("q4", 18_944, 57_856, records=KernelLDSRecordLayout(128, 304, (
    KernelLDSRecordComponent("qs", dtypes.int32, 0, 240, 16),
    KernelLDSRecordComponent("gapfill", dtypes.uint8, 240, 16, 16),
    KernelLDSRecordComponent("dm", dtypes.half, 256, 32, 16),
    KernelLDSRecordComponent("padding", dtypes.uint8, 288, 16, 16))))
  bad_geometry = KernelTileGeometry(geometry.tile, geometry.waves, geometry.threads, geometry.wave_size, geometry.lds_windows, (),
    (geometry.lds_region("ids"), geometry.lds_region("q8"), bad_q4))
  with pytest.raises(ValueError, match="fields do not exactly match"):
    _stage(geometry=bad_geometry)
  with pytest.raises(ValueError, match="unknown LDS region"):
    _stage(regions=(PackedRecordLDSRegionBinding("A", "q8"), PackedRecordLDSRegionBinding("B", "missing")))
