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


def test_register_template_producer_fragments_have_no_local_or_raw_isa():
  t = _fixture()
  p = t.producer(UOp.const(dtypes.weakint, 0), UOp.const(dtypes.weakint, 0))
  f = t.fragments(p.epoch, p.slot, p.ready)
  root = UOp.sink(*p.role_nodes, *f.fragments)
  assert prove_register_graph_no_lds(root) == ()
  assert len([u for u in root.toposort() if u.op is Ops.LOAD]) == 68
  assert len([u for u in root.toposort() if u.op is Ops.CONTRACT and u.dtype == dtypes.half.vec(16)]) == 4


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


def test_register_fragments_fail_closed_on_unproven_stage_readiness():
  t = _fixture()
  epoch = UOp.const(dtypes.weakint, 0)
  producer = t.producer(epoch, UOp.const(dtypes.weakint, 0))
  with pytest.raises(ValueError, match="readiness epoch/slot mismatch"):
    t.fragments(UOp.const(dtypes.weakint, 1), UOp.const(dtypes.weakint, 1), producer.ready)


def _register_wmma(t):
  def wmma(stage, acc, _subtile):
    arg = (str(t.tc), t.tc.dims, t.tc.dtype_in, t.tc.dtype_out, "AMD", t.tc.threads,
           (t.contracts[0].arg, t.contracts[1].arg, ()), ())
    return UOp(Ops.WMMA, dtypes.float.vec(8), (stage.fragments[0], stage.fragments[2], acc), arg)
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


@pytest.mark.parametrize("k_tiles", (1, 2, 3, 256))
def test_register_matching_readiness_proves_full_k_tail_lifecycle(k_tiles):
  t = _fixture(); adapter = RegisterStorageAdapter.from_template(t)
  graph = build_stage1_uop_graph_with_storage(adapter, RegisterLogicalStagePlan(), k_tiles, _register_wmma(t),
    subtile_count=1, accumulator_elements=8)
  proof = prove_stage1_uop_graph(graph)
  assert proof.passed, proof.errors
  assert not any(u.op is Ops.DEFINE_LOCAL for u in graph.sink.toposort())


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
