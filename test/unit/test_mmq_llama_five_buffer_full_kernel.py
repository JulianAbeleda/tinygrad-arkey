import inspect
import pytest

from tinygrad import dtypes
from tinygrad.uop.ops import Ops, UOp

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
              x.tag[:1] == ("llama_five_buffer_full_grid_epoch_release",)]
  wmmas = [x for x in nodes if x.op is Ops.WMMA]
  # Intra-subtile: each K32 group's first WMMA takes its A/B/C through a movement carrier ordered on the preceding lane
  # drain.  8 subtiles x 8 groups = 64 group heads, minus the 8 per-subtile chain heads with no prior group == 56.
  # Cross-subtile: each subtile e>0's chain head is ordered behind subtile e-1's drains == 7 more.  56 + 7 == 63.
  guarded = [x for x in wmmas if any(s.op is Ops.AFTER for s in x.src)]
  assert len(wmmas) == 128 and len(releases) == 56 and len(guarded) == 63
  assert all(any(r in x.backward_slice for r in releases) for x in guarded[:4])


def _has_tag(node:UOp, prefix:tuple) -> bool:
  return isinstance(node.tag, tuple) and node.tag[:len(prefix)] == prefix


def test_full_grid_q8_phase1_overwrite_waits_for_every_phase0_fragment_read():
  """The Q8 LDS window is shared by all eight subtiles.  Phase 1 may overwrite it only after every phase-0 fragment
  read, unless that read consumes an explicit phase-0 restage published after the overwrite."""
  nodes = full.build_llama_five_buffer_full_kernel(128, 128, 256).sink.toposort()
  phase0_reads = [x for x in nodes if x.op is Ops.LOAD and x.src and
                  _has_tag(x.src[0], ("llama_oracle_fragment_load", "B", 0))]
  phase1_overwrites = [x for x in nodes if x.op is Ops.GROUP and
                       _has_tag(x, ("hierarchical_record_producer", "B", 1))]
  assert len(phase0_reads) == 8*4*2
  assert phase1_overwrites

  unsafe = 0
  for read in phase0_reads:
    phase0_publishes = [x for x in read.backward_slice if _has_tag(x, ("llama_oracle_publish", 0))]
    for overwrite in phase1_overwrites:
      read_before_overwrite = read in overwrite.backward_slice
      restaged_after_overwrite = any(overwrite in publish.backward_slice for publish in phase0_publishes)
      unsafe += not (read_before_overwrite or restaged_after_overwrite)
  assert unsafe == 0, (
    f"{unsafe} phase-0 Q8 fragment-read/phase-1 overwrite pairs are unordered and have no intervening phase-0 restage")


def test_full_grid_stages_each_shared_q8_phase_once_per_epoch():
  """Subtile expansion consumes one shared Q8 LDS stage; it must not clone either phase's global-to-LDS producer."""
  nodes = full.build_llama_five_buffer_full_kernel(128, 128, 256).sink.toposort()
  producers = [[x for x in nodes if x.op is Ops.GROUP and
                _has_tag(x, ("hierarchical_record_producer", "B", phase))] for phase in range(2)]
  counts = tuple(map(len, producers))
  assert counts == (1, 1), f"shared Q8 stage producer counts are {counts}, expected one producer per phase"


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
