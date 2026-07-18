from dataclasses import replace
import inspect
import pytest
from types import SimpleNamespace

from tinygrad import Tensor, dtypes
from tinygrad.engine.realize import get_call_arg_uops
from tinygrad.uop.ops import AxisType, Ops, UOp

import extra.qk.mmq_llama_five_buffer_full_kernel as full
from extra.qk.mmq_llama_candidate_plan import llama_mmq_candidate_plan
from extra.qk.mmq_llama_runtime_contract import LLAMA_SOURCE_COMMIT


@pytest.mark.parametrize("shape,grid", [((128, 128, 256), (1, 1, 1)),
  ((256, 384, 256), (3, 2, 1)), ((384, 256, 512), (2, 3, 1))])
def test_full_grid_topology_abi_lds_and_unique_ownership(shape, grid):
  kernel = full.build_llama_five_buffer_full_kernel(*shape)
  assert kernel.topology.grid == grid
  assert kernel.topology.local_size == (256, 1, 1)
  assert (kernel.topology.waves, kernel.topology.wave_size, kernel.topology.lds_bytes) == ((8, 1), 32, 57856)
  assert [(x.slot, x.name) for x in kernel.proof_graph.parameters] == [
    (0, "output"), (1, "q4"), (2, "q8_values"), (3, "q8_scales"), (4, "q8_original_sums")]
  assert len(kernel.owner_coordinates) == shape[0]*shape[1]
  assert kernel.owner_coordinates == frozenset((m, n) for m in range(shape[0]) for n in range(shape[1]))
  specials = [x for x in kernel.sink.toposort() if x.op is Ops.SPECIAL]
  assert {(x.arg, x.src[0].arg) for x in specials} == {("gidx0", grid[0]), ("gidx1", grid[1]), ("lidx0", 256)}
  assert kernel.sink.op is Ops.SINK and not kernel.sink.ranges


@pytest.mark.parametrize("k", [256, 512])
def test_absolute_split_epoch_offsets_and_fp32_dependencies(k):
  kernel = full.build_llama_five_buffer_full_kernel(256, 384, k)
  offsets = kernel.epoch_offsets(1, 2, k//256-1)
  epoch = k//256-1
  assert offsets.q4 == ((2*128)*(k//256)+epoch)*36
  assert offsets.values == (epoch*2*256+128)*128
  assert offsets.scales == offsets.sums == (epoch*2*256+128)*4
  epochs = kernel.proof_graph.body.epochs
  assert all(epochs[0].accumulators[i] in epochs[1].accumulators[i].backward_slice for i in range(8)) if k == 512 else True


def test_epoch_offset_family_uses_full_role_abi_and_one_static_k256_epoch_per_variant():
  family = full.build_llama_five_buffer_epoch_offset_family(128, 128, 512)
  assert tuple(variant.epoch_offset for variant in family.variants) == (0, 1)
  assert all(variant.proof_graph is family.proof_graph for variant in family.variants)
  assert [parameter.size for parameter in family.proof_graph.parameters] == [
    128*128, 128*2*36, 4*128*128, 4*128*4, 4*128*4]
  assert len({variant.sink.key for variant in family.variants}) == 2
  for variant in family.variants:
    nodes = variant.sink.toposort()
    params = sorted({node for node in nodes if node.op is Ops.PARAM}, key=lambda node: node.arg.slot)
    assert [param.max_numel() for param in params] == [parameter.size for parameter in family.proof_graph.parameters]
    assert len([node for node in nodes if node.op is Ops.WMMA]) == 128
    assert {node.tag[1] for node in nodes if node.op is Ops.WMMA} == set(range(8))
  static_bases = []
  for variant in family.variants:
    static_bases.append({
      node.src[0].arg.slot: node.src[1].render()
      for node in variant.sink.toposort()
      if node.op is Ops.INDEX and node.src[0].op is Ops.PARAM and node.src[0].arg.slot in range(1, 5)
    })
  assert static_bases == [
    {1: "0", 2: "0", 3: "0", 4: "0"},
    {1: "36", 2: "32768", 3: "1024", 4: "1024"},
  ]
  assert family.variants[0].epoch_offsets(0, 0, 0) == full.FiveBufferEpochOffsets(0, 0, 0, 0)
  assert family.variants[1].epoch_offsets(0, 0, 1) == full.FiveBufferEpochOffsets(36, 32768, 1024, 1024)


def test_epoch_offset_family_binds_same_five_full_role_buffers_to_every_program(monkeypatch):
  family = full.build_llama_five_buffer_epoch_offset_family(128, 128, 512)
  emitted = []
  def fake_to_program(sink, _renderer):
    program = UOp(Ops.PROGRAM, dtypes.void, arg=f"epoch-program-{len(emitted)}")
    emitted.append((sink, program))
    return program
  monkeypatch.setattr(full, "to_program", fake_to_program)
  family = full.compile_llama_five_buffer_epoch_offset_family(family)
  assert family.emitted and family.programs == tuple(program for _, program in emitted)

  buffers = tuple(Tensor.empty(*parameter.physical_shape, dtype=parameter.dtype, device="AMD")
                  for parameter in family.proof_graph.parameters)
  output = full.bind_llama_five_buffer_epoch_offset_calls(family, buffers, output_is_zeroed=True)
  calls = [node for node in output.uop.toposort() if node.op is Ops.CALL and node.src[0].op is Ops.PROGRAM]
  assert len(calls) == 2 and [call.src[0] for call in calls] == list(family.programs)
  arguments = [get_call_arg_uops(call) for call in calls]
  assert all(len(row) == 5 for row in arguments)
  assert all(arguments[0][slot].buffer is arguments[1][slot].buffer for slot in range(5))
  assert calls[0] in arguments[1][0].toposort()

  bad = list(buffers)
  bad[2] = Tensor.empty(1, dtype=dtypes.int8, device="AMD")
  with pytest.raises(ValueError, match="q8_values"):
    full.bind_llama_five_buffer_epoch_offset_calls(family, tuple(bad), output_is_zeroed=True)
  with pytest.raises(ValueError, match="explicitly zeroed"):
    full.bind_llama_five_buffer_epoch_offset_calls(family, buffers, output_is_zeroed=False)


def test_epoch_offset_binder_keeps_all_twenty_calls_on_the_same_full_role_buffers():
  m, n, k = 128, 128, 5120
  proof = SimpleNamespace(facts=SimpleNamespace(m=m, n=n, k=k), parameters=full.five_buffer_parameters(m, n, k))
  topology = full.FullGridTopology((1, 1, 1))
  owners = frozenset((row, col) for row in range(m) for col in range(n))
  variants = tuple(full.LlamaFiveBufferFullKernel(
    proof, topology, UOp(Ops.SINK, dtypes.void), owners, LLAMA_SOURCE_COMMIT, tuple(),
    epoch_offset=epoch, blocker="", program=UOp(Ops.PROGRAM, dtypes.void, arg=f"epoch-program-{epoch}"), emitted=True)
    for epoch in range(20))
  family = full.LlamaFiveBufferEpochOffsetFamily(proof, topology, variants)
  buffers = tuple(Tensor.empty(*parameter.physical_shape, dtype=parameter.dtype, device="AMD")
                  for parameter in proof.parameters)
  output = full.bind_llama_five_buffer_epoch_offset_calls(family, buffers, output_is_zeroed=True)
  calls = [node for node in output.uop.toposort() if node.op is Ops.CALL and node.src[0].op is Ops.PROGRAM]
  assert len(calls) == 20
  arguments = [get_call_arg_uops(call) for call in calls]
  assert all(arguments[0][slot].buffer is row[slot].buffer for row in arguments for slot in range(5))
  assert all(previous in current[0].toposort() for previous, current in zip(calls, arguments[1:]))


def test_twentieth_k256_variant_encodes_final_full_role_offsets_in_the_uop_graph():
  sink = full._full_grid_sink(128, 256, 5120, accumulate=True, epoch_offset=19)
  bases = {
    node.src[0].arg.slot: node.src[1].render()
    for node in sink.toposort()
    if node.op is Ops.INDEX and node.src[0].op is Ops.PARAM and node.src[0].arg.slot in range(1, 5)
  }
  assert bases == {1: "(gidx0*92160+684)", 2: "622592", 3: "19456", 4: "19456"}
  assert len([node for node in sink.toposort() if node.op is Ops.WMMA]) == 128


def test_source_identity_writeback_vocabulary_and_no_dense_tensor_or_forbidden_tables():
  kernel = full.build_llama_five_buffer_full_kernel(128, 128, 256)
  assert kernel.source_commit == LLAMA_SOURCE_COMMIT
  assert kernel.proof_graph.candidate_identity == llama_mmq_candidate_plan().identity()
  stores = [x for x in kernel.sink.toposort() if x.op is Ops.STORE and isinstance(x.tag, tuple) and
            x.tag[:1] == ("wmma_writeback",)]
  assert len(stores) == 64 and {x.tag[-1] for x in stores} == {"col"}
  # The pointer's INDEX is a real movement value and carries the total store order to the sink.  The value side must
  # not carry a redundant same-dtype BITCAST: codegen folds that no-op away and drops the effect order onto the
  # scalar FP32 update underneath, which spec_program rejects as AFTER(ADD, STORE).
  for previous, current in zip(stores, stores[1:]):
    assert previous in current.src[0].backward_slice
    assert not (current.src[1].op is Ops.BITCAST and current.src[1].dtype == current.src[1].src[0].dtype)
  assert not any(name.startswith(("dense", "dequant")) for name, _, _ in kernel.proof_graph.allocated_shapes)
  source = inspect.getsource(full).lower()
  assert all(word not in source for word in ("model", "profile", "exact_shape", "getenv", "device scan", "autoscan", "route"))


def test_full_grid_orders_each_wmma_behind_the_preceding_lane_drain():
  """The oracle keeps the integer WMMA chain and the eight FP32 lane chains as separate algebraic dependencies, so
  without these edges a legal schedule issues every WMMA before consuming any C lane and retains all 184 drains."""
  kernel = full.build_llama_five_buffer_full_kernel(128, 128, 256)
  nodes = list(kernel.sink.toposort())
  releases = [x for x in nodes if x.op is Ops.BARRIER and isinstance(x.tag, tuple) and
              x.tag[:1] == ("llama_five_buffer_phase_major_group_release",)]
  wmmas = [x for x in nodes if x.op is Ops.WMMA]
  # Each resident phase has 4 groups x 8 subtiles: 3 guarded intra-subtile group heads and 7 guarded cross-subtile
  # phase heads. The phase transition is carried by the global release and the next producer/publish, not a WMMA guard.
  guarded = [x for x in wmmas if any(s.op is Ops.AFTER for s in x.src)]
  assert len(wmmas) == 128 and len(releases) == 48 and len(guarded) == 2*(3*8+7) == 62
  assert all(any(r in x.backward_slice for r in releases) for x in guarded[1:4])


def test_full_grid_stages_each_q8_phase_once_and_releases_all_phase0_subtiles_before_phase1():
  nodes = list(full.build_llama_five_buffer_full_kernel(128, 128, 256).sink.toposort())
  producers = [x for x in nodes if isinstance(x.tag, tuple) and x.tag[:1] == ("hierarchical_record_producer",)]
  publishes = [x for x in nodes if isinstance(x.tag, tuple) and
               x.tag[:1] == ("llama_five_buffer_phase_major_publish",)]
  assert [x.tag for x in producers] == [
    ("hierarchical_record_producer", "A", None),
    ("hierarchical_record_producer", "B", 0),
    ("hierarchical_record_producer", "B", 1)]
  assert [x.tag for x in publishes] == [
    ("llama_five_buffer_phase_major_publish", 0, 0),
    ("llama_five_buffer_phase_major_publish", 0, 1)]
  b_stores = [x for x in nodes if x.op is Ops.STORE and isinstance(x.tag, tuple) and len(x.tag) > 1 and x.tag[1] == "B"]
  assert len(b_stores) == len({x.tag for x in b_stores}) == 36
  release = next(x for x in nodes if x.tag == ("llama_five_buffer_phase_major_collective_release", 0, 0))
  assert len(release.src) == 64 and all(x.tag[:1] == ("llama_oracle_float_update",) for x in release.src)
  phase1_producer = next(x for x in producers if x.tag[-1] == 1)
  phase1_wmmas = [x for x in nodes if x.op is Ops.WMMA and x.tag[1] >= 4]
  assert release in phase1_producer.backward_slice
  assert len(phase1_wmmas) == 64 and all(release in x.backward_slice for x in phase1_wmmas)


def test_full_grid_stage_uses_hardware_local_axes_not_serial_local_or_warp_ranges():
  """The one-workgroup proof must map producer/fragment ownership to lidx0.

  A bare LOCAL/WARP RANGE is a scalar loop in the generated 256-thread
  dispatch.  It duplicates the complete Q4/Q8 stage and gives every lane the
  same WMMA inputs, which is structurally a different (and invalid) kernel.
  """
  kernel = full.build_llama_five_buffer_full_kernel(128, 128, 256)
  nodes = list(kernel.sink.toposort())
  lidx0 = next(x for x in nodes if x.op is Ops.SPECIAL and x.arg == "lidx0")
  assert not [x for x in nodes if x.op is Ops.RANGE and x.arg[-1] in (AxisType.LOCAL, AxisType.WARP)]
  wmmas = [x for x in nodes if x.op is Ops.WMMA]
  assert len(wmmas) == 128 and all(lidx0 in x.backward_slice_with_self for x in wmmas)
  producer_stores = [x for x in nodes if x.op is Ops.STORE and isinstance(x.tag, tuple) and
                     x.tag[:1] == ("hierarchical_record_store",)]
  assert producer_stores and all(lidx0 in x.backward_slice_with_self for x in producer_stores)
  threads = kernel.proof_graph.body.epochs[0].recurrence.stage.threads
  linear_wave = lidx0 // kernel.topology.wave_size
  assert threads.wave_m is (linear_wave // kernel.topology.waves[1])
  assert threads.wave_n is (linear_wave % kernel.topology.waves[1])
  assert threads.lane is (lidx0 % kernel.topology.wave_size)


def test_k512_carries_exact_states_and_orders_next_epoch_staging_after_collective_release():
  nodes = list(full.build_llama_five_buffer_full_kernel(128, 128, 512).sink.toposort())
  wmmas = [x for x in nodes if x.op is Ops.WMMA]
  guarded = [x for x in wmmas if any(s.op is Ops.AFTER for s in x.src)]
  publishes = {x.tag:x for x in nodes if isinstance(x.tag, tuple) and
               x.tag[:1] == ("llama_five_buffer_phase_major_publish",)}
  releases = {x.tag:x for x in nodes if isinstance(x.tag, tuple) and
              x.tag[:1] == ("llama_five_buffer_phase_major_collective_release",)}
  assert len(wmmas) == 256 and len(guarded) == 2*2*(3*8+7) == 124
  assert set(publishes) == {
    ("llama_five_buffer_phase_major_publish", epoch, phase) for epoch in range(2) for phase in range(2)}
  assert set(releases) == {
    ("llama_five_buffer_phase_major_collective_release", 0, 0),
    ("llama_five_buffer_phase_major_collective_release", 0, 1),
    ("llama_five_buffer_phase_major_collective_release", 1, 0)}
  assert all(len(release.src) == 64 for release in releases.values())
  epoch0_final = releases[("llama_five_buffer_phase_major_collective_release", 0, 1)]
  epoch1_phase0 = releases[("llama_five_buffer_phase_major_collective_release", 1, 0)]
  # State carry is lane-for-lane dataflow, not an algebraic epoch delta added after two independent recurrences.
  assert all(prior in current.backward_slice for prior, current in zip(epoch0_final.src, epoch1_phase0.src))
  epoch1_publish0 = publishes[("llama_five_buffer_phase_major_publish", 1, 0)]
  assert epoch1_publish0.src[0].op is Ops.GROUP
  assert len(epoch1_publish0.src[0].src) == 2
  assert all(epoch0_final in producer.backward_slice for producer in epoch1_publish0.src[0].src)


@pytest.mark.parametrize("shape", [(127, 128, 256), (128, 129, 256), (128, 128, 255)])
def test_unaligned_or_non_epoch_shapes_fail_closed(shape):
  with pytest.raises(ValueError): full.build_llama_five_buffer_full_kernel(*shape)


def test_compile_attempt_reaches_known_spill_free_gate_or_claims_only_success(monkeypatch):
  kernel = full.build_llama_five_buffer_full_kernel(128, 128, 256)
  def blocked(*_args): raise NotImplementedError(full.RESOURCE_BLOCKER)
  monkeypatch.setattr(full, "to_program", blocked)
  result = full.compile_llama_five_buffer_full_kernel(kernel)
  assert result is kernel and not result.emitted and result.program is None and result.blocker == full.RESOURCE_BLOCKER

  program = UOp(Ops.PROGRAM, dtypes.void, arg="successful_program_fixture")
  monkeypatch.setattr(full, "to_program", lambda *_args: program)
  emitted = full.compile_llama_five_buffer_full_kernel(kernel)
  assert emitted.emitted and emitted.program is program and emitted.blocker == ""
