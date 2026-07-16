import inspect
import pytest

from tinygrad import dtypes
from tinygrad.codegen import to_program, to_program_cache
from tinygrad.helpers import Target
from tinygrad.renderer.isa.amd import AMDISARenderer
from tinygrad.uop.ops import Ops, UOp

from extra.qk.mmq_llama_candidate_plan import llama_mmq_candidate_plan
from extra.qk.mmq_llama_full_kernel import (BLOCKER, ScannedTargetFacts, bounded_final_release, build_llama_full_kernel_graph,
                                             scalar_writeback_lane, scheduler_valid_callback_sink)
from extra.qk.mmq_llama_runtime_contract import LLAMA_SOURCE_COMMIT
from extra.qk.mmq_llama_record_producers import record_producer_instance_witnesses


TARGET = ScannedTargetFacts("AMD", "gfx1100", 32, 256, 65536, True, 24 << 30, 17 << 30)
def build(m, n, k, target=TARGET): return build_llama_full_kernel_graph(m, n, k, target=target)


def test_multi_epoch_fp32_dependency_and_exact_identity():
  graph = build(129, 257, 512)
  assert len(graph.body.epochs) == 2
  assert graph.body.epochs[1].previous is graph.body.epochs[0].accumulator
  assert graph.body.epochs[0].accumulator.dtype == graph.body.epochs[1].accumulator.dtype == dtypes.float
  assert graph.body.epochs[0].accumulator in graph.body.epochs[1].accumulator.backward_slice
  assert all(len(epoch.accumulators) == len(epoch.previous_accumulators) == 8 for epoch in graph.body.epochs)
  assert len({id(x) for x in graph.body.epochs[0].accumulators}) == 8
  assert all(graph.body.epochs[0].accumulators[slot] in graph.body.epochs[1].accumulators[slot].backward_slice
             for slot in range(8))
  assert graph.source_commit == LLAMA_SOURCE_COMMIT
  assert graph.candidate_identity == llama_mmq_candidate_plan().identity()


def test_logical_grid_tails_ids_and_epoch_addresses():
  graph = build(129, 257, 512)
  assert (graph.grid.x, graph.grid.y, graph.grid.z) == (3, 2, 1)
  assert graph.body.logical_index_axes == ("block_x", "block_y", "block_z")
  assert graph.body.tail_predicated and graph.body.identity_ids
  edge = graph.runtime.conventional_tile(2, 1, 0)
  assert (edge.tails.i_max, edge.tails.j_max) == (0, 0)
  assert edge.destination(0, 0, graph.runtime.strides) == 2*128 + 128*257
  assert [x.kb0 for x in graph.body.epochs] == [0, 1]
  assert graph.body.epochs[1].addresses == graph.body.representative_tile.addresses(
    1, graph.runtime.tile, graph.runtime.extents)
  assert graph.body.epochs[1].binding.q4_word_offset == graph.body.epochs[1].addresses.q4_block*36
  assert graph.body.epochs[1].binding.q8_byte_offset == graph.body.epochs[1].addresses.q8_first_int*4
  assert graph.body.epochs[1].binding.tails == graph.body.representative_tile.tails


def test_bounded_allocations_have_no_hidden_full_dequant_and_single_writeback_owner():
  graph = build(33, 65, 256)
  assert not any(kind.startswith("dequant") or shape == (65, 256) for kind, shape, _ in graph.allocated_shapes)
  assert graph.allocated_shapes[-1] == ("lds_bounded_tile", (57856,), "uint8")
  assert graph.body.output_owner_count == graph.body.writeback_count == 33*65


def test_executable_surface_is_explicitly_blocked_and_never_claimed():
  graph = build(1, 1, 256)
  assert graph.custom_kernel is None and not graph.emitted and not graph.routed
  assert graph.blocker == BLOCKER and "not a launch grid" in graph.blocker
  assert "clears AFTER(CAST(int), STORE)" in graph.blocker
  assert "clears the SPEC=1 UNROLL(float) / STACK(float.vec(8))" in graph.blocker
  assert "AMD:ISA register pressure exceeds the spill-free VGPR/SGPR budget; Inc 0 has no spills" in graph.blocker
  assert "store-to-next-producer progressive drain" in graph.blocker
  assert "progressive-C reuse collapses the 64 logical C runs to one physical drain lease" in graph.blocker
  assert "grid-derived M/N addressing" in graph.blocker and "No genuine full-grid binary emits" in graph.blocker
  with pytest.raises(RuntimeError, match="No genuine full-grid binary emits"):
    graph.program()


def test_bounded_wave_probe_drains_symbolic_subtiles_without_claiming_a_full_grid():
  graph = build(1, 1, 256)
  out = UOp.param(2, dtypes.float.ptr(64))
  store = bounded_final_release(graph, out)
  sink = scheduler_valid_callback_sink(store, name="mmq_llama_full_kernel_compile_probe")
  assert sink.op is Ops.SINK and not sink.ranges
  assert sink.src[0].op is Ops.END and sink.src[0].src[0].op is Ops.STORE
  drains = [x for x in sink.toposort() if x.op is Ops.STORE and isinstance(x.tag, tuple) and
            x.tag[:1] == ("llama_full_kernel_bounded_final_release",)]
  assert len(drains) == 8 and [x.tag[2] for x in drains] == list(range(8))
  assert all(x.src[1].dtype == dtypes.float.vec(8) for x in drains)
  for previous, current in zip(drains, drains[1:]):
    assert current.src[1].op is Ops.AFTER and previous in current.src[1].src[1:]
    assert current.src[1].src[0].op is Ops.BITCAST
  subtile = graph.body.epochs[-1].recurrence.stage.subtile_n
  assert all(subtile not in x.ranges for x in drains)
  assert all(any(x.op is Ops.CONTRACT and x.arg == ((subtile.arg[0], 8),) for x in store.toposort()) for _ in (0,))
  assert len([x for x in store.toposort() if x.op is Ops.WMMA]) == 16
  releases = [x for x in store.toposort() if x.op is Ops.BARRIER and isinstance(x.tag, tuple) and
              x.tag[:1] == ("llama_full_kernel_bounded_epoch_release",)]
  assert len(releases) == 8
  assert [x.tag[2] for x in releases] == list(range(8))
  wmmas = sorted((x for x in store.toposort() if x.op is Ops.WMMA), key=lambda x:x.tag[1:3])
  for i in range(2, len(wmmas), 2):
    previous_ordinal = wmmas[i-1].tag[1]
    assert wmmas[i].src[2].op is Ops.AFTER and wmmas[i].src[2].src[0].op is Ops.BITCAST
    assert sum(y.tag == ("llama_oracle_float_update", previous_ordinal, lane)
               for lane in range(8) for y in wmmas[i].src[2].src[1:]) == 8
  # Every group after the first starts only after the preceding group's
  # accumulator release, so neither its operands nor its C carrier overlap
  # the prior group in the generated schedule.
  for previous, current in zip(wmmas[1::2], wmmas[2::2]):
    assert all(current.src[i].op is Ops.AFTER and current.src[i].src[0].op is Ops.BITCAST for i in range(3))
    assert any(dep.op is Ops.BARRIER and dep.tag[:1] == ("llama_full_kernel_bounded_epoch_release",)
               for dep in current.src[0].backward_slice)
  to_program_cache.clear()
  # This remains a bounded wave probe. Its compiler outcome must not set any full-grid claim.
  with pytest.raises(NotImplementedError, match="AMD:ISA register pressure exceeds the spill-free VGPR/SGPR budget"):
    to_program(sink, AMDISARenderer(Target.parse("AMD:ISA:gfx1100")))
  assert not graph.emitted and not graph.routed and graph.custom_kernel is None


def test_composed_q8_b_stores_keep_one_structural_witness_each():
  graph = build(1, 1, 256)
  store = bounded_final_release(graph, UOp.param(2, dtypes.float.ptr(64)))
  b_stores = [x for x in store.toposort() if x.op is Ops.STORE and isinstance(x.tag, tuple) and
              x.tag[:1] == ("hierarchical_record_store",) and x.tag[1] == "B"]
  assert len(b_stores) == 36
  for actual in b_stores:
    witnesses = record_producer_instance_witnesses(actual.src[1])
    assert len(witnesses) == 1
    witness = witnesses[0]
    assert witness.role == "B" and witness.field == actual.tag[2] and witness.phase == actual.tag[3]
    assert witness.iteration == actual.tag[4]
    assert all(coordinate in actual.src[1].backward_slice_with_self for coordinate in
                (witness.source_row, witness.source_k, witness.destination_row, witness.destination_vector))


def test_q8_b_typed_witness_survives_to_instruction_selection_proof(monkeypatch):
  from extra.qk.amd_isa_renderer_policy import PrefillAMDISARendererPolicy
  original = PrefillAMDISARendererPolicy.wmma_frag_store_epoch_proof
  observed = []
  def audit(self, idx, desc, role, helpers):
    proof = original(self, idx, desc, role, helpers)
    if role == "B": observed.append((desc.const_bytes if desc is not None else None,
                                      record_producer_instance_witnesses(idx), proof))
    return proof
  monkeypatch.setattr(PrefillAMDISARendererPolicy, "wmma_frag_store_epoch_proof", audit)
  graph = build(1, 1, 256)
  sink = scheduler_valid_callback_sink(bounded_final_release(graph, UOp.param(2, dtypes.float.ptr(64))),
                                       name="mmq_llama_b_witness_isel_probe")
  to_program_cache.clear()
  with pytest.raises(NotImplementedError, match="AMD:ISA register pressure exceeds the spill-free VGPR/SGPR budget"):
    to_program(sink, AMDISARenderer(Target.parse("AMD:ISA:gfx1100")))
  assert observed and any(offset == 544 and witnesses and proof is not None for offset, witnesses, proof in observed)


@pytest.mark.parametrize("shape", [(0, 1, 256), (1, 0, 256), (1, 1, 0), (1, 1, 255), (1, 1, 257),
                                    (True, 1, 256), (1, 1.0, 256)])
def test_invalid_shapes_fail_closed(shape):
  with pytest.raises(ValueError): build(*shape)


def test_candidate_capability_matches_scanned_facts_without_semantic_labels_or_vram_tiers():
  low_free = ScannedTargetFacts("AMD", "gfx1100", 32, 256, 65536, True, 24 << 30, 0)
  graph = build(1, 1, 256, low_free)
  assert graph.target.free_vram_bytes == 0
  assert graph.identity() == build(1, 1, 256, ScannedTargetFacts(
    "AMD", "gfx1100", 32, 256, 65536, True, 48 << 30, 17 << 30)).identity()
  for target in (ScannedTargetFacts("AMD", "gfx1101", 32, 256, 65536, True, 24 << 30, 1),
                 ScannedTargetFacts("AMD", "gfx1100", 64, 256, 65536, True, 24 << 30, 1),
                 ScannedTargetFacts("AMD", "gfx1100", 32, 128, 65536, True, 24 << 30, 1),
                 ScannedTargetFacts("AMD", "gfx1100", 32, 256, 32768, True, 24 << 30, 1)):
    with pytest.raises(ValueError, match="scanned target"): build(1, 1, 256, target)
  source = inspect.getsource(build_llama_full_kernel_graph).lower()
  assert all(label not in source for label in ("8b", "14b", "profile", "model"))
