import pytest

from tinygrad import dtypes
from tinygrad.codegen.opt.compiler_policies import StoragePolicy
from tinygrad.codegen.opt.kernel_lds import PrecontractContractSpec, PrecontractOperandTemplate
from tinygrad.codegen.opt.kernel_pipeline import (KernelStage1PipelinePlan, build_stage1_uop_graph_with_storage,
  prove_stage1_uop_graph)
from tinygrad.codegen.opt.register_pipeline import (RegisterLogicalStagePlan, RegisterPipeTemplate,
  RegisterStorageAdapter, prove_register_graph_no_lds, prove_register_lifecycle, register_geometry)
from tinygrad.codegen.opt.tc import amd_rdna3
from tinygrad.uop.ops import AxisType, Ops, UOp


def _fixture():
  tc = next(x for x in amd_rdna3 if x.dtype_in == dtypes.half and x.dtype_out == dtypes.float)
  ra, rb, ka, kb = (UOp.range(512, 20, AxisType.LOOP), UOp.range(4096, 21, AxisType.LOOP),
                    UOp.range(4096, 22, AxisType.REDUCE), UOp.range(4096, 23, AxisType.REDUCE))
  a, b = UOp.param(0, dtypes.half.ptr(512 * 4096)), UOp.param(1, dtypes.half.ptr(4096 * 4096))
  operands = (PrecontractOperandTemplate("A", a.index(ra * 4096 + ka).load(), ra, ka, UOp.const(dtypes.weakint, 0)),
              PrecontractOperandTemplate("B", b.index(rb * 4096 + kb).load(), rb, kb, UOp.const(dtypes.weakint, 0)))
  contracts = []
  for i, role in enumerate(("A", "B")):
    axes = tuple(UOp.range(2, 30 + i * 4 + j, AxisType.UPCAST) for j in range(4))
    elem = ((axes[0] * 2 + axes[1]) * 2 + axes[2]) * 2 + axes[3]
    contracts.append(PrecontractContractSpec(role, axes, tuple((x.arg[0], 2) for x in axes), elem,
      tuple(tc.lane_map.remaps()[i].items())))
  return RegisterPipeTemplate(tc, register_geometry(), operands, tuple(contracts))


def test_register_template_zero_lds_and_real_descriptor():
  t = _fixture()
  assert t.policy.storage_kind == "global_register_resident" and t.policy.resources.lds_bytes == 0
  adapter = RegisterStorageAdapter.from_template(t)
  assert adapter.policy == StoragePolicy("global_register_resident")
  assert adapter.logical_plan.active_lds_bytes == 0 and adapter.logical_plan.buffer_count == 2


def test_register_stage_buffers_have_independent_role_width_contracts():
  t = _fixture()
  specs = t.stage_buffer_specs
  assert [s.snapshot() for s in specs] == [
    {"role": "A", "slots": 2, "fragments": 2, "lane_width": 16, "role_width": 32, "half_elements": 64,
     "half_bytes": 128, "packed_vgpr_width": 32},
    {"role": "B", "slots": 2, "fragments": 2, "lane_width": 16, "role_width": 32, "half_elements": 64,
     "half_bytes": 128, "packed_vgpr_width": 32},
  ]
  p = t.producer(UOp.const(dtypes.weakint, 0), UOp.const(dtypes.weakint, 0))
  defs = [u for u in UOp.sink(*p.role_nodes).toposort() if u.op is Ops.DEFINE_REG]
  assert sorted((u.ptrdtype.size, u.tag[1]) for u in defs) == [(64, "A"), (64, "B")]


def test_register_stage_dynamic_slot_fails_closed_at_amd_isa_boundary():
  from tinygrad.renderer.isa import IselContext
  from tinygrad.renderer.isa.amd import isel_index
  t = _fixture()
  p = t.producer(UOp.const(dtypes.weakint, 0), UOp.const(dtypes.weakint, 0))
  root = UOp.sink(*p.role_nodes)
  dreg = next(u for u in root.toposort() if u.op is Ops.DEFINE_REG and u.tag[1] == "A")
  dynamic_slot = UOp.range(2, 8801, AxisType.REDUCE)
  idx = dreg.index(dynamic_slot * t.stage_buffer_specs[0].role_width, dtype=dtypes.half.vec(16))
  with pytest.raises(NotImplementedError, match="dynamic VGPR indexing"):
    isel_index(IselContext(root), idx)


def test_register_stage_static_slot_maps_to_pinned_vgpr_carrier():
  from tinygrad.renderer.isa import IselContext
  from tinygrad.renderer.isa.amd import AMDOps, isel_index
  t0 = _fixture()
  t = RegisterPipeTemplate(t0.tc, t0.geometry, t0.operands, t0.contracts, schedule="sequential")
  p = t.producer(UOp.const(dtypes.weakint, 0), UOp.const(dtypes.weakint, 0))
  root = UOp.sink(*p.role_nodes)
  ctx = IselContext(root)
  # A multi-output WMMA has the low C window available for this static
  # stage mapping; force that structural fact for the isolated carrier test.
  ctx._ncruns = 2
  dreg = next(u for u in root.toposort() if u.op is Ops.DEFINE_REG and u.tag[1] == "A")
  carriers = [isel_index(ctx, dreg.index(UOp.const(dtypes.weakint, i), dtype=dtypes.half)) for i in (2, 3)]
  assert all(c is not None and c.arg[:1] == ("stage_reg",) for c in carriers)
  assert carriers[0].arg[1:3] == ("A", 2) and carriers[1].arg[1:3] == ("A", 3)
  assert carriers[0].arg[3] == carriers[1].arg[3], "adjacent fp16 elements share one packed VGPR"


def test_register_stage_carriers_have_real_amd_encoders():
  from tinygrad.renderer.isa import Register
  from tinygrad.renderer.isa.amd import AMDOps, lower_inst
  pin = UOp.const(dtypes.int32, 204)
  src = UOp.const(dtypes.half, 0).replace(tag=(Register("v41", 41),))
  read = UOp(Ops.INS, dtypes.half, (src, pin), AMDOps.STAGE_READ, (Register("v42", 42),))
  write = UOp(Ops.INS, dtypes.void, (src, src, src, src, pin), AMDOps.STAGE_WRITE)
  assert lower_inst(read) is not None
  assert lower_inst(write) is not None


def test_register_template_producer_fragments_have_no_local_or_raw_isa():
  t = _fixture()
  p = t.producer(UOp.const(dtypes.weakint, 0), UOp.const(dtypes.weakint, 0))
  f = t.fragments(p.epoch, p.slot, p.ready)
  root = UOp.sink(*p.role_nodes, *f.fragments)
  assert prove_register_graph_no_lds(root) == ()
  assert len([u for u in root.toposort() if u.op is Ops.LOAD]) == 68
  assert len([u for u in root.toposort() if u.op is Ops.CONTRACT and u.dtype == dtypes.half.vec(16)]) == 4


def test_register_fragment_contracts_remain_flat_through_expander():
  """A full-width vector carrier must not become STACK(vector) during expansion."""
  from tinygrad.codegen import expander, pm_group_for_reduce, pm_pre_expander, sym
  from tinygrad.uop.ops import graph_rewrite
  t = _fixture()
  producer = t.producer(UOp.const(dtypes.weakint, 0), UOp.const(dtypes.weakint, 0))
  fragments = t.fragments(producer.epoch, producer.slot, producer.ready)
  arg = (str(t.tc), t.tc.dims, t.tc.dtype_in, t.tc.dtype_out, "AMD", t.tc.threads,
         (t.contracts[0].arg, t.contracts[1].arg, ((40, 2), (41, 2), (42, 2))), ())
  wmma = UOp(Ops.WMMA, dtypes.float.vec(8),
    (fragments.fragments[0], fragments.fragments[2], UOp.const(dtypes.float.vec(8), 0.0)), arg)
  expanded = graph_rewrite(UOp.sink(*producer.role_nodes, wmma),
    sym+pm_pre_expander+pm_group_for_reduce+expander)
  nested = [u for u in expanded.toposort() if u.op is Ops.STACK and
            any(x.dtype.count > 1 for x in u.src)]
  assert not nested
  assert all(any(isinstance(x.tag, tuple) and x.tag[:1] == ("register_pipe_fragment",)
                 for x in u.src[0].backward_slice_with_self)
             for u in expanded.toposort() if u.op is Ops.WMMA)


def test_register_adapter_does_not_enter_lds_stage1_builder_before_readiness_is_proven():
  adapter = RegisterStorageAdapter.from_template(_fixture())
  with pytest.raises(ValueError, match="zero LDS"):
    build_stage1_uop_graph_with_storage(adapter, KernelStage1PipelinePlan(2, 20480), 2, lambda *_: None)


def test_register_single_epoch_compiles_through_normal_amd_rewrite():
  from tinygrad.codegen import full_rewrite_to_sink
  from tinygrad.helpers import Target
  from tinygrad.renderer.cstyle import HIPRenderer
  t = _fixture()
  producer = t.producer(UOp.const(dtypes.weakint, 0), UOp.const(dtypes.weakint, 0))
  fragments = t.fragments(producer.epoch, producer.slot, producer.ready)
  arg = (str(t.tc), t.tc.dims, t.tc.dtype_in, t.tc.dtype_out, "AMD", t.tc.threads,
         (t.contracts[0].arg, t.contracts[1].arg, ()), ())
  wmma = UOp(Ops.WMMA, dtypes.float.vec(8),
    (fragments.fragments[0], fragments.fragments[2], UOp.const(dtypes.float.vec(8), 0.0)), arg)
  rewritten = full_rewrite_to_sink(UOp.sink(*producer.role_nodes, wmma), HIPRenderer(Target.parse("AMD")), optimize=False)
  topo = rewritten.toposort()
  assert not any(x.op in (Ops.DEFINE_LOCAL, Ops.INS) for x in topo)
  assert len([x for x in topo if x.op is Ops.WMMA]) == 1


def test_register_wmma_abi_rejects_missing_c_axes_before_devectorization():
  t = _fixture()
  producer = t.producer(UOp.const(dtypes.weakint, 0), UOp.const(dtypes.weakint, 0))
  fragments = t.fragments(producer.epoch, producer.slot, producer.ready)
  malformed_arg = (str(t.tc), t.tc.dims, t.tc.dtype_in, t.tc.dtype_out, "AMD", t.tc.threads,
                   (t.contracts[0].arg, t.contracts[1].arg, ()), ())
  wmma = UOp(Ops.WMMA, dtypes.float.vec(8),
    (fragments.fragments[0], fragments.fragments[2], UOp.const(dtypes.float.vec(8), 0.0)), malformed_arg)
  errors = prove_register_graph_no_lds(UOp.sink(*producer.role_nodes, wmma))
  assert any("C WMMA contract requires 3 binary axes" in error for error in errors)


def test_register_fragments_fail_closed_on_unproven_stage_readiness():
  t = _fixture()
  epoch = UOp.const(dtypes.weakint, 0)
  producer = t.producer(epoch, UOp.const(dtypes.weakint, 0))
  with pytest.raises(ValueError, match="readiness epoch/slot mismatch"):
    t.fragments(UOp.const(dtypes.weakint, 1), UOp.const(dtypes.weakint, 1), producer.ready)


def _register_wmma(t):
  def wmma(stage, acc, _subtile):
    arg = (str(t.tc), t.tc.dims, t.tc.dtype_in, t.tc.dtype_out, "AMD", t.tc.threads,
           # WMMA's vec8 result is the 2x2x2 C contract.  Keeping this ABI
           # metadata in sync is required by no_vectorized_wmma when it
           # decomposes the vector result during normal lowering.
           (t.contracts[0].arg, t.contracts[1].arg, ((40, 2), (41, 2), (42, 2))), ())
    return UOp(Ops.WMMA, dtypes.float.vec(8), (stage.fragments[0], stage.fragments[2], acc), arg)
  return wmma


def _register_wmma_chain(t):
  """Build the same two-step fragment chain used by the executable pipeline."""
  def wmma(stage, acc, _subtile):
    arg = (str(t.tc), t.tc.dims, t.tc.dtype_in, t.tc.dtype_out, "AMD", t.tc.threads,
           (t.contracts[0].arg, t.contracts[1].arg, ((40, 2), (41, 2), (42, 2))), ())
    first = UOp(Ops.WMMA, dtypes.float.vec(8), (stage.fragments[0], stage.fragments[1], acc), arg)
    return UOp(Ops.WMMA, dtypes.float.vec(8), (stage.fragments[2], stage.fragments[3], first), arg)
  return wmma


def test_register_matching_readiness_proves_k2_epoch_slot_mapping():
  t = _fixture(); adapter = RegisterStorageAdapter.from_template(t)
  graph = build_stage1_uop_graph_with_storage(adapter, RegisterLogicalStagePlan(), 2, _register_wmma(t),
    subtile_count=1, accumulator_elements=8)
  proof = prove_stage1_uop_graph(graph)
  assert proof.passed, proof.errors
  assert graph.body_readiness == "matching"
  assert graph.body_fragments.epoch.render() == graph.body_range.render()
  assert graph.body_fragments.slot.render() == (graph.body_range % 2).render()
  assert graph.prologue.ready in graph.body_fragments.ready.backward_slice_with_self
  assert len([u for u in graph.sink.toposort() if u.op is Ops.DEFINE_REG]) == 3


def test_register_sequential_schedule_uses_one_static_vgpr_slot():
  base = _fixture()
  t = RegisterPipeTemplate(base.tc, base.geometry, base.operands, base.contracts, schedule="sequential")
  adapter = RegisterStorageAdapter.from_template(t)
  assert adapter.logical_plan.buffer_count == 1
  assert [spec.half_elements for spec in t.stage_buffer_specs] == [32, 32]
  graph = build_stage1_uop_graph_with_storage(adapter, adapter.logical_plan, 3, _register_wmma(t),
    subtile_count=1, accumulator_elements=8)
  proof = prove_stage1_uop_graph(graph)
  assert proof.passed, proof.errors
  assert graph.body_readiness == "sequential"
  assert graph.body_fragments.slot.op is Ops.CONST and graph.body_fragments.slot.arg == 0
  assert graph.body_producer.slot.op is Ops.CONST and graph.body_producer.slot.arg == 0
  # All stage-buffer indexes are compile-time offsets; no RANGE/modulo index
  # can reach the AMD VGPR boundary in this schedule.
  stage_defs = {u for u in graph.sink.toposort() if u.op is Ops.DEFINE_REG and
                isinstance(u.tag, tuple) and u.tag[:1] == ("register_pipe_stage_buffer",)}
  indexes = [u for u in graph.sink.toposort() if u.op is Ops.INDEX and u.src and u.src[0] in stage_defs]
  assert indexes and all(u.src[1].op is Ops.CONST for u in indexes)
  assert prove_register_lifecycle(3, buffer_count=1).passed


def test_register_sequential_producer_carries_accumulator_reuse_dependency():
  base = _fixture()
  t = RegisterPipeTemplate(base.tc, base.geometry, base.operands, base.contracts, schedule="sequential")
  adapter = RegisterStorageAdapter.from_template(t)
  graph = build_stage1_uop_graph_with_storage(adapter, adapter.logical_plan, 2, _register_wmma(t),
    subtile_count=1, accumulator_elements=8)
  proof = prove_stage1_uop_graph(graph)
  assert proof.passed, proof.errors
  updates = tuple(u for u in graph.body_join.backward_slice if u.op is Ops.STORE and graph.accumulator_reg in u.src[0].backward_slice)
  assert updates
  # Each producer role group must retain the explicit update dependency, not
  # merely appear as a sibling in the body barrier.
  assert all(any(update in node.backward_slice for update in updates) for node in graph.body_producer.role_nodes)


def test_register_sequential_full_k_rewrites_through_normal_amd_path():
  from tinygrad.codegen import full_rewrite_to_sink
  from tinygrad.helpers import Target
  from tinygrad.renderer.cstyle import HIPRenderer
  base = _fixture()
  t = RegisterPipeTemplate(base.tc, base.geometry, base.operands, base.contracts, schedule="sequential")
  adapter = RegisterStorageAdapter.from_template(t)
  graph = build_stage1_uop_graph_with_storage(adapter, adapter.logical_plan, 2, _register_wmma(t),
    subtile_count=1, accumulator_elements=8)
  rewritten = full_rewrite_to_sink(graph.sink, HIPRenderer(Target.parse("AMD")), optimize=False)
  assert not any(u.op in (Ops.DEFINE_LOCAL, Ops.INS) for u in rewritten.toposort())
  assert len([u for u in rewritten.toposort() if u.op is Ops.WMMA]) == 2


def test_register_full_k_chain_rewrites_with_valid_wmma_abi():
  from tinygrad.codegen import full_rewrite_to_sink
  from tinygrad.helpers import Target
  from tinygrad.renderer.cstyle import HIPRenderer
  t = _fixture(); adapter = RegisterStorageAdapter.from_template(t)
  graph = build_stage1_uop_graph_with_storage(adapter, RegisterLogicalStagePlan(), 2, _register_wmma_chain(t),
    subtile_count=1, accumulator_elements=8)
  rewritten = full_rewrite_to_sink(graph.sink, HIPRenderer(Target.parse("AMD")), optimize=False)
  assert not any(u.op in (Ops.DEFINE_LOCAL, Ops.INS) for u in rewritten.toposort())
  assert len([u for u in rewritten.toposort() if u.op is Ops.WMMA]) == 4


@pytest.mark.parametrize("k_tiles", (1, 2, 3, 256))
def test_register_matching_readiness_proves_full_k_tail_lifecycle(k_tiles):
  t = _fixture(); adapter = RegisterStorageAdapter.from_template(t)
  graph = build_stage1_uop_graph_with_storage(adapter, RegisterLogicalStagePlan(), k_tiles, _register_wmma(t),
    subtile_count=1, accumulator_elements=8)
  proof = prove_stage1_uop_graph(graph)
  assert proof.passed, proof.errors
  assert not any(u.op is Ops.DEFINE_LOCAL for u in graph.sink.toposort())


def test_register_full_k_rewrite_accepts_real_wmma_output_contract():
  from tinygrad.codegen import full_rewrite_to_sink
  from tinygrad.helpers import Target
  from tinygrad.renderer.cstyle import HIPRenderer
  t = _fixture(); adapter = RegisterStorageAdapter.from_template(t)
  caxes = tuple(UOp.range(2, 50 + i, AxisType.UPCAST) for i in range(3))
  celem = (caxes[0] * 2 + caxes[1]) * 2 + caxes[2]
  arg = (str(t.tc), t.tc.dims, t.tc.dtype_in, t.tc.dtype_out, "AMD", t.tc.threads,
         (t.contracts[0].arg, t.contracts[1].arg, tuple((x.arg[0], 2) for x in caxes)), ())
  def wmma(stage, acc, _subtile):
    first = UOp(Ops.WMMA, dtypes.float.vec(8), (stage.fragments[0], stage.fragments[1], acc), arg)
    return UOp(Ops.WMMA, dtypes.float.vec(8), (stage.fragments[2], stage.fragments[3], first), arg)
  graph = build_stage1_uop_graph_with_storage(adapter, RegisterLogicalStagePlan(), 2, wmma,
    subtile_count=1, accumulator_elements=64, accumulator_offset=UOp.const(dtypes.weakint, 0),
    accumulator_contract=(celem, tuple((x.arg[0], 2) for x in caxes)))
  rewritten = full_rewrite_to_sink(graph.sink, HIPRenderer(Target.parse("AMD")), optimize=False)
  topo = rewritten.toposort()
  assert not any(x.op in (Ops.DEFINE_LOCAL, Ops.INS) for x in topo)
  assert len([x for x in topo if x.op is Ops.WMMA]) == 4

def test_register_native_full_graph_has_flat_stage_carriers():
  """The native AMD rewrite must not receive STACK-of-STACK WMMA inputs."""
  from tinygrad.codegen import full_rewrite_to_sink
  from tinygrad.helpers import Target
  from tinygrad.renderer.isa.amd import AMDISARenderer
  base = _fixture()
  t = RegisterPipeTemplate(base.tc, base.geometry, base.operands, base.contracts, schedule="sequential")
  adapter = RegisterStorageAdapter.from_template(t)
  graph = build_stage1_uop_graph_with_storage(adapter, adapter.logical_plan, 2, _register_wmma(t),
    subtile_count=8, accumulator_elements=64)
  rewritten = full_rewrite_to_sink(graph.sink, AMDISARenderer(Target.parse("AMD:ISA:gfx1100")), optimize=False)
  topo = rewritten.toposort()
  nested = [u for u in topo if u.op is Ops.STACK and u.dtype.count > 1 and any(s.op is Ops.STACK for s in u.src)]
  assert not nested
  assert not any(u.op is Ops.VCAT for u in topo)


def test_register_template_rejects_fake_descriptor_remap():
  t = _fixture()
  bad = list(t.contracts); bad[0] = PrecontractContractSpec("A", bad[0].axes, bad[0].arg, bad[0].element, (("bad", "map"),))
  with pytest.raises(ValueError, match="descriptor"):
    RegisterPipeTemplate(t.tc, t.geometry, t.operands, tuple(bad))


def test_logical_register_stage_plan_has_no_lds_window():
  plan = RegisterLogicalStagePlan()
  assert plan.active_lds_bytes == 0 and plan.slot_for_epoch(3) == 1 and plan.slot_window(0) == (0, 0)


@pytest.mark.parametrize("k_tiles", (1, 2, 3, 256))
def test_register_lifecycle_uses_shared_proof_for_k_tail_cases(k_tiles):
  assert prove_register_lifecycle(k_tiles).passed
