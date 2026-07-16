from dataclasses import replace

import pytest

from tinygrad import dtypes
from extra.qk.kernel_lds import (HierarchicalPackedRecordStageDescriptor, PackedRecordFieldProducer,
  PackedRecordCooperativeSchedule, PackedRecordCooperativeStore, PackedRecordLDSRegionBinding, PackedRecordOperandTemplate,
  PackedRecordSource, PrecontractContractSpec,
  PrecontractThreadAxes, build_hierarchical_packed_record_stage, contract_symbolic_upcast,
  lower_symbolic_barrier_dependencies, prove_hierarchical_packed_record_stage)
from extra.qk.kernel_pipeline import HierarchicalKernelPipelinePlan, HierarchicalPipelineRole
from tinygrad.codegen.opt.packed_weight import PackedOperandComponent, PackedOperandRecordTransform, PackedOperandTransform
from tinygrad.codegen.opt.tc import amd_rdna3
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, GroupOp, Ops, UOp
from extra.qk.kernel_vocabulary import (KernelLDSArenaRegion, KernelLDSRecordComponent, KernelLDSRecordLayout,
  KernelLDSWindow, KernelTileGeometry)


def _tc(): return next(x for x in amd_rdna3 if x.dtype_in == dtypes.char and x.dtype_out == dtypes.int)


def test_symbolic_upcast_is_contracted_before_effect_barrier():
  axis = UOp.range(8, 1991, AxisType.UPCAST)
  value = (axis.cast(dtypes.float)+1.0).replace(tag=("symbolic_barrier_value",))
  barrier = UOp(Ops.BARRIER, dtypes.void, (value,))
  lowered = lower_symbolic_barrier_dependencies(barrier, axis)
  assert lowered.op is Ops.BARRIER and lowered.src[0] is contract_symbolic_upcast(value, axis)
  assert lowered.src[0].dtype == dtypes.float.vec(8) and lowered.src[0].arg == ((1991, 8),)


def test_symbolic_upcast_contraction_rejects_detached_or_non_upcast_values():
  axis = UOp.range(8, 1992, AxisType.UPCAST)
  with pytest.raises(ValueError, match="does not own"): contract_symbolic_upcast(UOp.const(dtypes.float, 0), axis)
  with pytest.raises(ValueError, match="UPCAST"): contract_symbolic_upcast(axis.cast(dtypes.float), UOp.range(8, 1993, AxisType.LOOP))


def _vector(dtype, stride):
  def produce(sources, row, k, width):
    return UOp(Ops.STACK, dtype.vec(width), tuple(sources[0].index(row*stride+k+i).load().cast(dtype) for i in range(width)))
  return produce


def _one(value, dtype): return UOp(Ops.STACK, dtype.vec(1), (value.cast(dtype),))


def _linear_schedule(template, threads, source_k):
  thread = (threads.wave_m+threads.wave_n)*32+threads.lane
  stores = []
  for binding in template.fields:
    field = template.transform.produced.component(binding.field)
    width, vectors_per_row = binding.vector_bytes//field.dtype.itemsize, field.size_bytes//binding.vector_bytes
    for iteration in range(128*vectors_per_row//256):
      linear = thread+iteration*256
      row, vector = linear//vectors_per_row, linear%vectors_per_row
      logical_row, logical_k = template.row_tile_base+row, UOp.const(dtypes.weakint, source_k)+vector*width
      if binding.sources == ("record",):
        value = UOp(Ops.STACK, field.dtype.vec(width), tuple(
          template.source_load("record", logical_row*field.stride_bytes+logical_k+i,
                               dtype=template.source("record").ptrdtype.base).cast(field.dtype) for i in range(width)))
      else:
        sources = tuple(template.source(x) for x in binding.sources)
        value = binding.producer(sources, logical_row, logical_k, width)
      stores.append(PackedRecordCooperativeStore(binding.field, iteration, logical_row, logical_k, row, vector, value))
  return tuple(stores)


def _q4_schedule(template, threads, source_k):
  source, base_k = template.source("record"), UOp.const(dtypes.weakint, source_k)
  load = lambda index: template.source_load("record", index)
  stores = []
  # load_tiles_q4_K: i = i0 + threadIdx.y and txi = lane. One uint32 fans out to low/high Q4 destinations.
  for iteration in range(16):
    row = threads.wave_m+iteration*8
    word_k, word = base_k+threads.lane, load(row*36+4+base_k//8+threads.lane)
    for high in range(2):
      value = _one((word >> (high*4)) & UOp.const(dtypes.uint32, 0x0f0f0f0f), dtypes.int32)
      stores.append(PackedRecordCooperativeStore("payload", iteration*2+high, row, word_k, row,
                                                 16*(threads.lane//8)+threads.lane%8+high*8, value))
  # Oracle MMA path: rows_per_warp=16, ksc=lane%2, and each lane emits four scale/min half2 values.
  row, ksc = threads.wave_m*16+threads.lane//2, threads.lane%2
  dm = load(row*36+base_k//8)
  sc_lo = load(row*36+1+(ksc%2+(ksc!=0)))
  sc_hi = load(row*36+1+ksc//2)
  m_group = ksc+2
  m_lo = load(row*36+1+(m_group%2+(m_group!=0)))
  m_hi = load(row*36+1+m_group//2)
  packed = dm ^ sc_lo ^ sc_hi ^ m_lo ^ m_hi
  for half in range(4):
      scalar = ((packed >> ((half%2)*16)).cast(dtypes.uint16).bitcast(dtypes.half))
      value = UOp(Ops.STACK, dtypes.half.vec(2), (scalar, scalar))
      stores.append(PackedRecordCooperativeStore("correction", half, row, base_k+ksc, row, 4*ksc+half, value))
  return tuple(stores)


def _fixture(*, outer_k=256, phase_k=128, group_k=32, phase_count=2, threads=None, subtile_m=None, subtile_n=None,
             source_options=None):
  q4_layout = KernelLDSRecordLayout(128, 304, (
    KernelLDSRecordComponent("payload", dtypes.int32, 0, 256, 4),
    KernelLDSRecordComponent("correction", dtypes.half, 256, 32, 4),
    KernelLDSRecordComponent("reserved", dtypes.uint8, 288, 16, 16)))
  q8_layout = KernelLDSRecordLayout(128, 144, (
    KernelLDSRecordComponent("scale", dtypes.half, 0, 16, 4),
    KernelLDSRecordComponent("payload", dtypes.int8, 16, 128, 4)))
  q8_bytes, q4_bytes = q8_layout.size_bytes, q4_layout.size_bytes
  geometry = KernelTileGeometry((128, 128, 32), (8, 1), 256, 32,
    (KernelLDSWindow("A", 0, q8_bytes, 144), KernelLDSWindow("B", q8_bytes, q8_bytes+q4_bytes, 304)), (),
    (KernelLDSArenaRegion("q8", 0, q8_bytes, records=q8_layout),
     KernelLDSArenaRegion("q4", q8_bytes, q8_bytes+q4_bytes, records=q4_layout)))
  templates = []
  for role, layout, slot in (("A", q4_layout, 0), ("B", q8_layout, 1)):
    produced = PackedOperandTransform(role+"-produced", tuple(PackedOperandComponent(
      x.component, x.dtype, x.offset_bytes, x.size_bytes, "record", layout.stride_bytes, x.alignment) for x in layout.components))
    source_dtype = dtypes.uint32 if role == "A" else dtypes.uint8
    source = PackedOperandTransform(role+"-source", (PackedOperandComponent("record", source_dtype, 0, layout.stride_bytes),))
    row, k = UOp.range(128, 800+slot*2, AxisType.LOOP), UOp.range(outer_k, 801+slot*2, AxisType.REDUCE)
    fields = tuple(PackedRecordFieldProducer(x.component, ("record",), _vector(x.dtype, layout.stride_bytes), vector_bytes=4)
                   for x in layout.components if x.component != "reserved")
    pointer = UOp.param(slot, source_dtype.ptr(layout.size_bytes//source_dtype.itemsize))
    schedule = (PackedRecordCooperativeSchedule("oracle-wave-row-fanout", _q4_schedule, ("wave_m", "lane")) if role == "A" else
                PackedRecordCooperativeSchedule("linear-256-copy", _linear_schedule, ("wave_m", "wave_n", "lane")))
    options = (source_options or {}).get(role, {})
    templates.append(PackedRecordOperandTemplate(role, PackedOperandRecordTransform(role+"-record", source, produced),
      (PackedRecordSource("record", pointer, **options),), fields, ("reserved",) if role == "A" else (), "payload", row, k,
      UOp.const(dtypes.weakint, 0), dtypes.char, schedule))
  tc = _tc()
  threads = threads or PrecontractThreadAxes(UOp.range(8, 900, AxisType.LOCAL), UOp.range(1, 901, AxisType.LOCAL),
                                             UOp.range(32, -1, AxisType.WARP))
  subtile_m = subtile_m or UOp.range(1, 902, AxisType.UPCAST)
  subtile_n = subtile_n or UOp.range(8, 903, AxisType.UPCAST)
  contracts = []
  for operand_idx, role in enumerate(("A", "B")):
    axes = tuple(UOp.range(2, 910+operand_idx*4+i, AxisType.UPCAST) for i in range(4))
    element = ((axes[0]*2+axes[1])*2+axes[2])*2+axes[3]
    contracts.append(PrecontractContractSpec(role, axes, tuple((x.arg[0], 2) for x in axes), element,
      tuple(tc.lane_map.remaps()[operand_idx].items())))
  plan = HierarchicalKernelPipelinePlan(HierarchicalPipelineRole("A", "outer_epoch"),
    HierarchicalPipelineRole("B", "inner_phase"), phase_count)
  descriptor = HierarchicalPackedRecordStageDescriptor(plan, outer_k, phase_k, group_k)
  stage = build_hierarchical_packed_record_stage(geometry, tc=tc, contracts=tuple(contracts), threads=threads,
    subtile_m=subtile_m, subtile_n=subtile_n,
    allocation=UOp.placeholder((geometry.lds_bytes,), dtypes.uint8, 899, addrspace=AddrSpace.LOCAL), descriptor=descriptor,
    templates=tuple(templates), regions=(PackedRecordLDSRegionBinding("A", "q4"),
                                         PackedRecordLDSRegionBinding("B", "q8")))
  return geometry, descriptor, tuple(templates), stage


def _stores(producer, role, field):
  return [x for x in producer.src if x.op is Ops.STORE and x.tag[1:3] == (role, field)]


def _at(expr, stage, wave_m, lane):
  values = {stage.threads.wave_m:UOp.const(stage.threads.wave_m.dtype, wave_m),
            stage.threads.wave_n:UOp.const(stage.threads.wave_n.dtype, 0),
            stage.threads.lane:UOp.const(stage.threads.lane.dtype, lane)}
  resolved = expr.substitute(values)
  assert resolved.vmin == resolved.vmax
  return resolved.vmin


def test_exact_full_cooperative_two_phase_record_stage():
  geometry, descriptor, _, stage = _fixture()
  assert geometry.tile == (128, 128, 32) and geometry.waves == (8, 1) and geometry.threads == 256
  assert geometry.lds_region("q4").records.stride_bytes == 256+32+16
  assert descriptor.groups_per_phase == 4 and len(stage.barriers) == 4 and len(stage.groups) == 8
  assert len(_stores(stage.persistent_producer, "A", "payload")) == 32
  assert len(_stores(stage.persistent_producer, "A", "correction")) == 4
  assert all(x.src[1].dtype == dtypes.int32 for x in _stores(stage.persistent_producer, "A", "payload"))
  assert all(x.src[1].dtype == dtypes.half.vec(2) for x in _stores(stage.persistent_producer, "A", "correction"))
  for phase in stage.phases:
    assert len(_stores(phase.producer, "B", "scale")) == 2
    assert len(_stores(phase.producer, "B", "payload")) == 16
    assert all(x.src[1].dtype == dtypes.half.vec(2) for x in _stores(phase.producer, "B", "scale"))
    assert all(x.src[1].dtype == dtypes.char.vec(4) for x in _stores(phase.producer, "B", "payload"))
  assert [x.persistent_k for x in stage.groups] == [0, 32, 64, 96, 128, 160, 192, 224]
  assert [x.overwriteable_k for x in stage.groups] == [0, 32, 64, 96]*2
  assert tuple((x.field, x.value.dtype) for x in stage.groups[0].sidecars) == (
    ("correction", dtypes.half.vec(2)), ("scale", dtypes.half.vec(2)))
  contract_by_role = {x.role:x for x in stage.contracts}
  for group in stage.groups:
    assert group.persistent_fragment.op is group.overwriteable_fragment.op is Ops.CONTRACT
    assert group.persistent_fragment.dtype == group.overwriteable_fragment.dtype == dtypes.char.vec(16)
    assert group.persistent_fragment.arg == contract_by_role["A"].arg
    assert group.overwriteable_fragment.arg == contract_by_role["B"].arg
    assert {stage.threads.wave_m, stage.subtile_m, stage.threads.lane} <= set(group.persistent_row.backward_slice_with_self)
    assert {stage.threads.wave_n, stage.subtile_n, stage.threads.lane} <= set(group.overwriteable_row.backward_slice_with_self)
  assert prove_hierarchical_packed_record_stage(stage).passed


def test_cooperative_producer_orders_each_load_decode_address_store_transaction():
  _, _, _, stage = _fixture()
  for producer in (stage.persistent_producer, *(phase.producer for phase in stage.phases)):
    stores = producer.src
    assert stores and all(store.op is Ops.STORE for store in stores)
    for previous, current in zip(stores, stores[1:]):
      ordered_allocation, address, value = current.src[0].src[0], current.src[0].src[1], current.src[1]
      assert ordered_allocation.op is Ops.AFTER and ordered_allocation.src == (stage.allocation, previous)
      assert address.op is not Ops.AFTER and value.op is not Ops.AFTER
      source_afters = [node for node in value.backward_slice_with_self if node.op is Ops.AFTER and node.src[-1] is previous]
      assert source_afters and all(node.src[0].op in (Ops.PARAM, Ops.BUFFER) for node in source_afters)
      assert previous in current.backward_slice_with_self and previous in value.backward_slice_with_self


def test_cooperative_producer_uses_only_pointer_value_and_address_effect_carriers():
  _, _, _, stage = _fixture()
  effects = (stage.persistent_producer, *(phase.producer for phase in stage.phases))
  afters = {node for effect in effects for node in effect.backward_slice_with_self if node.op is Ops.AFTER}
  assert afters
  assert all(node.src[0].op not in GroupOp.ALU for node in afters)
  value_afters = [node for node in afters if node.src[0] is not stage.allocation]
  assert value_afters and all(node.src[0].op in (Ops.PARAM, Ops.BUFFER) for node in value_afters)


def test_tail_predicate_clamps_invalid_source_addresses_and_stages_zero():
  # Keep 120 of 128 q8 rows and bind a runtime-like base remap. The schedule still covers all LDS rows.
  cutoff, base = 120*144, UOp.variable("q8_base", 0, 7)
  _, _, _, stage = _fixture(source_options={"B": {
    "address_remap": lambda index: index+base,
    "load_validity": lambda index: index < cutoff}})
  source = stage.templates[1].source("record")
  loads = [x for phase in stage.phases for store in phase.producer.src
           for x in store.src[1].backward_slice_with_self
           if x.op is Ops.LOAD and x.src[0].src[0] is source]
  assert loads and all(len(x.src) == 3 and x.src[1].arg == 0 and x.src[2].dtype == dtypes.bool for x in loads)
  # Invalid logical rows select physical element zero before the gated load, so no OOB address is formed.
  assert all(x.src[0].src[1].op is Ops.WHERE and x.src[0].src[1].src[0] is x.src[2] and
             x.src[0].src[1].src[2].arg == 0 and base in x.src[0].src[1].src[1].backward_slice_with_self for x in loads)
  assert prove_hierarchical_packed_record_stage(stage).passed


def test_oracle_schedule_enumerates_every_q4_source_and_destination_address():
  geometry, _, templates, stage = _fixture()
  emissions = templates[0].cooperative_schedule.callback(templates[0], stage.threads, 0)
  payload, correction = emissions[:32], emissions[32:]
  payload_pairs, correction_pairs = set(), set()
  for wave_m in range(8):
    for lane in range(32):
      for emission in payload:
        loads = [x for x in emission.value.backward_slice_with_self if x.op is Ops.LOAD]
        assert len(loads) == 1
        source = _at(loads[0].src[0].src[1], stage, wave_m, lane)
        destination = geometry.lds_region("q4").base + _at(emission.destination_row, stage, wave_m, lane)*304 + \
                      _at(emission.destination_vector, stage, wave_m, lane)*4
        payload_pairs.add((source, destination))
      for emission in correction:
        loads = [x for x in emission.value.backward_slice_with_self if x.op is Ops.LOAD]
        assert len(loads) == 5
        destination = geometry.lds_region("q4").base + _at(emission.destination_row, stage, wave_m, lane)*304 + 256 + \
                      _at(emission.destination_vector, stage, wave_m, lane)*4
        correction_pairs.update((_at(load.src[0].src[1], stage, wave_m, lane), destination) for load in loads)
  assert payload_pairs == {(row*36+4+lane, geometry.lds_region("q4").base+row*304+
                            (16*(lane//8)+lane%8+half*8)*4)
                           for row in range(128) for lane in range(32) for half in range(2)}
  assert correction_pairs == {(row*36+word, geometry.lds_region("q4").base+row*304+256+(4*ksc+l)*4)
                              for row in range(128) for ksc, words in ((0, (0, 1, 2)), (1, (0, 1, 2, 3)))
                              for word in words for l in range(4)}


def test_proof_rejects_flat_q4_schedule_but_retains_linear_q8_schedule():
  _, _, templates, stage = _fixture()
  assert templates[1].cooperative_schedule.name == "linear-256-copy"
  flat = PackedRecordCooperativeSchedule("linear-256-copy", _linear_schedule, ("wave_m", "wave_n", "lane"))
  mutated_templates = (replace(templates[0], cooperative_schedule=flat), templates[1])
  assert not prove_hierarchical_packed_record_stage(replace(stage, templates=mutated_templates)).passed


def test_descriptor_and_independent_role_extent_validation():
  plan = HierarchicalKernelPipelinePlan(HierarchicalPipelineRole("A", "outer_epoch"), HierarchicalPipelineRole("B", "inner_phase"), 2)
  with pytest.raises(ValueError, match="outer_k"):
    HierarchicalPackedRecordStageDescriptor(plan, 255, 128, 32)
  with pytest.raises(ValueError, match="divisible"):
    HierarchicalPackedRecordStageDescriptor(plan, 256, 128, 48)
  with pytest.raises(ValueError, match="K extent mismatch"):
    _fixture(outer_k=128, phase_k=64, group_k=32)


def test_proof_fails_closed_on_lifecycle_address_and_axis_mutations():
  _, _, _, stage = _fixture()
  phase0, phase1 = stage.phases
  early = replace(stage, phases=(phase0, replace(phase1, producer=phase1.producer.replace(src=phase0.producer.src))))
  assert not prove_hierarchical_packed_record_stage(early).passed
  assert not prove_hierarchical_packed_record_stage(replace(stage, persistent_producer=phase0.producer)).passed
  assert not prove_hierarchical_packed_record_stage(replace(stage,
    phases=(replace(phase0, publish=UOp(Ops.NOOP, dtypes.void)), phase1))).passed
  assert not prove_hierarchical_packed_record_stage(replace(stage,
    phases=(replace(phase0, release=UOp(Ops.NOOP, dtypes.void)), phase1))).passed
  side = stage.groups[0].sidecars[0]
  escaped_group = replace(stage.groups[0], sidecars=(replace(side, byte_address=UOp.const(dtypes.weakint, 10_000_000)),)+stage.groups[0].sidecars[1:])
  escaped = replace(stage, phases=(replace(phase0, groups=(escaped_group,)+phase0.groups[1:]), phase1))
  assert not prove_hierarchical_packed_record_stage(escaped).passed
  detached_threads = replace(stage, threads=replace(stage.threads, wave_m=UOp.range(8, 990, AxisType.LOCAL)))
  assert not prove_hierarchical_packed_record_stage(detached_threads).passed
  detached_subtile = replace(stage, subtile_n=UOp.range(8, 991, AxisType.UPCAST))
  assert not prove_hierarchical_packed_record_stage(detached_subtile).passed
  replicated_producer = stage.persistent_producer.replace(src=stage.persistent_producer.src+(stage.persistent_producer.src[0],))
  assert not prove_hierarchical_packed_record_stage(replace(stage, persistent_producer=replicated_producer)).passed
