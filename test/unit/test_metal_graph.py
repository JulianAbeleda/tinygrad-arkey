import unittest
import ctypes
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch
from tinygrad import Device
from tinygrad.uop.ops import Ops, UOp
from tinygrad.dtype import dtypes
from tinygrad.engine.jit import GraphAdmissionReason, GraphException, GraphRunner
from tinygrad.helpers import Context


class TestMetalGraphAdmissionSynthetic(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    from tinygrad.runtime.graph.metal import MetalGraph
    cls.MetalGraph = MetalGraph

  @staticmethod
  def metal_buf(offset, *, size=1):
    buf = MagicMock()
    buf.op, buf.device, buf.dtype, buf.size = Ops.SLICE, "METAL", dtypes.uint8, size
    base = MagicMock(dtype=dtypes.uint8)
    buf.src = (base, UOp.const(dtypes.weakint, offset))
    return buf

  @staticmethod
  def call(*bufs):
    return MagicMock(src=(MagicMock(op=Ops.PROGRAM),) + tuple(bufs))

  def admission(self, *bufs):
    with patch.object(GraphRunner, "_all_devs", return_value=[object()]):
      return self.MetalGraph.admission([], self.call(*bufs))

  def test_uint32_boundary_and_first_invalid_byte_without_large_allocation(self):
    assert self.admission(self.metal_buf(0xFFFFFFFF)).reason is GraphAdmissionReason.ADMITTED
    rejected = self.admission(self.metal_buf(0x100000000, size=64))
    assert rejected.reason is GraphAdmissionReason.BACKEND_BUFFER_OFFSET_WIDTH
    assert rejected.capability == "icb_buffer_offset_bits" and rejected.limit == 0xFFFFFFFF
    assert rejected.observed == 0x100000000
    assert len(rejected.resources) == 1
    resource = rejected.resources[0]
    assert resource.buffer_arg_index == 0 and resource.byte_offset == 0x100000000 and resource.byte_span == 64

  def test_reports_every_offending_argument_and_boolean_delegates(self):
    bufs = (self.metal_buf(0), self.metal_buf(0x100000000), self.metal_buf(0x100000100))
    with patch.object(GraphRunner, "_all_devs", return_value=[object()]):
      admission = self.MetalGraph.admission([], self.call(*bufs))
      assert [resource.buffer_arg_index for resource in admission.resources] == [1, 2]
      assert self.MetalGraph.supports_uop([], self.call(*bufs)) is False

  def test_hybrid_mode_accepts_graph_but_preserves_icb_failure_facts(self):
    with Context(METAL_HYBRID_REPLAY=1): admission = self.admission(self.metal_buf(0x100000000, size=16))
    assert admission.supported is True and admission.reason is GraphAdmissionReason.BACKEND_BUFFER_OFFSET_WIDTH
    assert admission.limit == 0xFFFFFFFF and admission.observed == 0x100000000
    assert admission.resources[0].byte_span == 16


class _FakeBuffer:
  def __init__(self, offset, name):
    self._buf, self.nbytes = SimpleNamespace(buf=name, offset=offset), 4
    self.base = self
  def ensure_allocated(self): return self


class TestMetalHybridReplaySynthetic(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    from tinygrad.runtime.graph import metal as metal_graph
    cls.mod, cls.MetalGraph = metal_graph, metal_graph.MetalGraph

  def replay_graph(self, initial, *, hybrid=True):
    graph = self.MetalGraph.__new__(self.MetalGraph)
    args = SimpleNamespace(
      launch_dims=lambda values: ((values["n"], 1, 1), (1, 1, 1)),
      vals=lambda values: (values["n"],))
    ast, runtime = SimpleNamespace(arg=args), SimpleNamespace(pipeline_state="pipeline", max_total_threads=1)
    graph.current_bufs, graph.updatable, graph.uop_replace = [[initial]], [0], [[(0, 0)]]
    graph.mutable_positions = [(0,)]
    graph.mutable_call_indexes, graph.base_modes = [0], [True]
    graph.immutable_icb_admissions = [self.mod.metal_icb_binding_admission(())]
    graph.calls, graph.runtimes = [(0, ast, [initial], {})], [runtime]
    graph.icb_commands, graph.vars, graph.all_resources = [MagicMock()], [], []
    graph.hybrid_replay = hybrid
    graph.updated_launch_dims = lambda values: iter(((0, (values["n"], 1, 1), (1, 1, 1)),))
    return graph

  def test_icb_ranges_preserve_holes_and_original_indexes(self):
    assert self.mod.metal_icb_ranges([True, True, False, False, True, False, True, True]) == ((0, 2), (4, 1), (6, 2))
    assert self.mod.metal_icb_ranges([]) == ()
    assert self.mod.metal_icb_ranges([False, False]) == ()

  def test_dynamic_binding_transitions_never_write_overflow_to_icb(self):
    initial, limit, overflow, safe_again = (_FakeBuffer(0, "initial"), _FakeBuffer(0xFFFFFFFF, "limit"),
                                            _FakeBuffer(0x100000000, "overflow"), _FakeBuffer(7, "safe_again"))
    graph, command = self.replay_graph(initial), None
    command = graph.icb_commands[0]

    modes, direct, resources = graph._prepare_replay((SimpleNamespace(buffer=limit),), {"n":3})
    assert modes == [True] and direct == {} and "limit" in resources
    assert command.setKernelBuffer_offset_atIndex.call_args.args[1] == 0xFFFFFFFF

    command.reset_mock()
    modes, direct, resources = graph._prepare_replay((SimpleNamespace(buffer=overflow),), {"n":5})
    assert modes == [False] and list(direct) == [0] and "overflow" in resources
    assert command.setKernelBuffer_offset_atIndex.call_count == 0
    assert direct[0][2:] == ((5, 1, 1), (1, 1, 1), (5,))

    modes, direct, _ = graph._prepare_replay((SimpleNamespace(buffer=safe_again),), {"n":9})
    assert modes == [True] and direct == {}
    assert command.setKernelBuffer_offset_atIndex.call_args.args[1] == 7

  def test_partitioned_control_fails_closed_on_dynamic_overflow(self):
    graph = self.replay_graph(_FakeBuffer(0, "initial"), hybrid=False)
    with self.assertRaisesRegex(GraphException, "during partitioned replay"):
      graph._prepare_replay((SimpleNamespace(buffer=_FakeBuffer(0x100000000, "overflow")),), {"n":1})
    assert graph.icb_commands[0].setKernelBuffer_offset_atIndex.call_count == 0

  def test_mixed_encoder_order_and_boundaries(self):
    graph = self.MetalGraph.__new__(self.MetalGraph)
    graph.icb, events = "icb", []
    class Encoder:
      def executeCommandsInBuffer_withRange(self, icb, rng): events.append(("icb", icb, rng))
      def memoryBarrierWithScope(self, scope): events.append(("barrier", scope))
    direct = {2:(SimpleNamespace(pipeline_state=SimpleNamespace(name="d2"), max_total_threads=1), (), (), (), ()),
              3:(SimpleNamespace(pipeline_state=SimpleNamespace(name="d3"), max_total_threads=1), (), (), (), ())}
    with patch.object(self.mod.metal, "NSRange", side_effect=lambda start,length: (start, length)), \
         patch.object(self.mod, "encode_metal_dispatch", side_effect=lambda _enc,rt,*args: events.append(("direct", rt.name))):
      graph._encode_replay(Encoder(), [True, True, False, False, True], direct)
    assert events == [("icb", "icb", (0, 2)), ("barrier", self.mod.metal.MTLBarrierScopeBuffers), ("direct", "d2"),
                      ("barrier", self.mod.metal.MTLBarrierScopeBuffers), ("direct", "d3"),
                      ("barrier", self.mod.metal.MTLBarrierScopeBuffers), ("icb", "icb", (4, 1))]

  def test_pre_apple9_workaround_applies_only_to_active_icb_pipelines(self):
    graph = self.MetalGraph.__new__(self.MetalGraph)
    graph.needs_icb_fix = 1
    graph.runtimes = [SimpleNamespace(pipeline_state="p0"), SimpleNamespace(pipeline_state="direct"), SimpleNamespace(pipeline_state="p2")]
    encoder = MagicMock()
    with patch.object(self.mod, "getenv", return_value=1), patch.object(self.mod.metal, "MTLSize", side_effect=lambda *x: x):
      graph._apply_icb_pipeline_fix(encoder, [True, False, True])
    assert [call.args[0] for call in encoder.setComputePipelineState.call_args_list] == ["p0", "p2"]
    assert encoder.dispatchThreadgroups_threadsPerThreadgroup.call_count == 2

  def test_replay_diagnostics_are_structured_facts_not_label_parsing(self):
    graph = self.MetalGraph.__new__(self.MetalGraph)
    graph.dev, graph.hybrid_replay, graph.calls = SimpleNamespace(), True, [None] * 323
    graph.last_replay_counts = {"icb_calls":256, "direct_encoded_calls":67}
    graph._publish_replay_facts(committed=True)
    with Context(METAL_HYBRID_REPLAY=1): facts = self.mod.metal_replay_facts(graph.dev)
    assert facts["configured_strategy"] == "hybrid_icb_direct" and facts["experimental_ab"] is True
    assert facts["last_graph"] == {"strategy":"hybrid_icb_direct", "graph_calls":323, "icb_calls":256,
                                   "direct_encoded_calls":67, "committed":True}


class TestMetalDirectEncoderSynthetic(unittest.TestCase):
  def test_shared_direct_encoder_binds_buffers_scalars_and_dispatch(self):
    from tinygrad.runtime.ops_metal import encode_metal_dispatch
    encoder, pipeline = MagicMock(), MagicMock()
    pipeline.threadExecutionWidth.return_value, pipeline.staticThreadgroupMemoryLength.return_value = 32, 0
    bufs = (SimpleNamespace(buf="a", offset=7), SimpleNamespace(buf="b", offset=11))
    encode_metal_dispatch(encoder, pipeline, 64, bufs, (3, 2, 1), (4, 1, 1), (17, -2))
    assert [call.args for call in encoder.setBuffer_offset_atIndex.call_args_list] == [("a", 7, 0), ("b", 11, 1)]
    assert [call.args[2] for call in encoder.setBytes_length_atIndex.call_args_list] == [2, 3]
    assert encoder.setBytes_length_atIndex.call_args_list[0].args[0] == bytes(ctypes.c_int(17))
    assert encoder.dispatchThreadgroups_threadsPerThreadgroup.call_count == 1

  def test_shared_direct_encoder_validates_before_encoding(self):
    from tinygrad.runtime.ops_metal import encode_metal_dispatch
    encoder, pipeline = MagicMock(), MagicMock()
    pipeline.threadExecutionWidth.return_value, pipeline.staticThreadgroupMemoryLength.return_value = 32, 8
    with self.assertRaisesRegex(RuntimeError, "bigger than 4"):
      encode_metal_dispatch(encoder, pipeline, 4, (), (1, 1, 1), (5, 1, 1))
    assert encoder.method_calls == []

@unittest.skipUnless(Device.DEFAULT == "METAL", "Metal device required to run")
class TestMetalGraph(unittest.TestCase):
  def setUp(self):
    from tinygrad.runtime.graph.metal import MetalGraph
    self.MetalGraph = MetalGraph
    self.dev = Device[Device.DEFAULT]

  def metal_buf(self, offset):
    buf = MagicMock()
    if offset > 0:
      buf.op = Ops.SLICE
      src = MagicMock()
      src.dtype = dtypes.uint8
      buf.src = (src, UOp.const(dtypes.weakint, offset))
      buf.dtype = dtypes.uint8
    else:
      buf.op = Ops.BUFFER
    buf.device = Device.DEFAULT
    return buf

  def call(self, *bufs):
    c = MagicMock()
    c.src = (MagicMock(op=Ops.PROGRAM),) + tuple(bufs)
    return c

  def test_supports_uop_normal_offset(self):
    assert self.MetalGraph.supports_uop([self.dev], self.call(self.metal_buf(0), self.metal_buf(100), self.metal_buf(0xFFFFFFFF))) is True

  def test_supports_uop_overflow_offset(self):
    assert self.MetalGraph.supports_uop([self.dev], self.call(self.metal_buf(0), self.metal_buf(0x100000000))) is False

  def test_supports_uop_nonmetal_buf(self):
    # non-SLICE ops should not be checked for offset
    buf = MagicMock()
    buf.op = Ops.BUFFER
    buf.device = Device.DEFAULT
    self.MetalGraph.supports_uop([self.dev], self.call(buf))

if __name__ == "__main__":
  unittest.main()
