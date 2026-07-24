import pytest

from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.uop import Ops
from tinygrad.uop.ops import UOp, StateRegionSpec, PhaseBoundarySpec, StateHandle
from tinygrad.uop.spec import spec_full, type_verify


def _handle(lanes=1):
  return StateHandle(StateRegionSpec("running_state", dtypes.float, lanes), PhaseBoundarySpec("produce", "consume", 3))


def test_scalar_state_publish_reload_preserves_typed_identity():
  handle = _handle()
  value = UOp.const(dtypes.float, 1.0)
  published, reloaded = handle.publish(value), handle.reload(handle.publish(value))
  assert published.dtype == reloaded.dtype == dtypes.float
  assert reloaded.src == (published,)
  type_verify(UOp.sink(reloaded), spec_full)


def test_vector_state_publish_reload_preserves_lanes_and_phase_boundary():
  handle = _handle(8)
  value = UOp.const(dtypes.float.vec(8), 1.0)
  published = handle.publish(value)
  reloaded = handle.reload(published)
  assert handle.dtype == reloaded.dtype == dtypes.float.vec(8)
  assert reloaded.arg[1].boundary == PhaseBoundarySpec("produce", "consume", 3)
  type_verify(UOp.sink(reloaded), spec_full)


def test_vector_state_reload_is_a_typed_scalar_lane_carrier():
  handle = _handle(8)
  reloaded = handle.reload(handle.publish(UOp.const(dtypes.float.vec(8), 1.0)))
  lane = reloaded.gep(3)
  assert lane.dtype == dtypes.float
  type_verify(UOp.sink(lane), spec_full)


def test_scalar_state_reload_remains_a_raw_scalar_value():
  handle = _handle()
  reloaded = handle.reload(handle.publish(UOp.const(dtypes.float, 1.0)))
  assert reloaded.dtype == dtypes.float
  type_verify(UOp.sink(reloaded), spec_full)


def test_state_handle_rejects_invalid_phase_lifetime_and_foreign_publication():
  with pytest.raises(ValueError): PhaseBoundarySpec("same", "same").validate()
  handle, foreign = _handle(), StateHandle(StateRegionSpec("other", dtypes.float), PhaseBoundarySpec("produce", "consume"))
  with pytest.raises(ValueError): handle.reload(foreign.publish(UOp.const(dtypes.float, 1.0)))
  malformed = UOp(Ops.CUSTOMI, dtypes.float, (UOp.const(dtypes.float, 1.0),), ("state_reload_v1", handle))
  with pytest.raises(RuntimeError): type_verify(UOp.sink(malformed), spec_full)


def test_lane_major_local_state_publish_reload_tracks_storage_and_wait_order():
  storage = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(128, AddrSpace.LOCAL), arg=91)
  lane = UOp.special(8, "state_lane")
  handle = StateHandle(StateRegionSpec("vector_state", dtypes.float, 8), PhaseBoundarySpec("publish", "reload"),
                       storage=storage, lane=lane, lane_stride=16, element_offset=4)
  published = handle.publish(UOp.const(dtypes.float.vec(8), 1.0))
  wait = UOp(Ops.WAIT, dtypes.void, (published,), arg=("state_handle_wait_v1", handle))
  reloaded = handle.reload(published, wait)
  assert published.src[1:] == (storage, lane)
  assert reloaded.op is Ops.CUSTOMI and reloaded.arg == ("state_reload_v1", handle)
  assert reloaded.src[0].op is Ops.STACK and reloaded.src[0].tag == ("state_reload_lanes_v1", handle)
  assert len(reloaded.src[0].src) == 8 and all(source.op is Ops.LOAD for source in reloaded.src[0].src)
  type_verify(UOp.sink(reloaded), spec_full)


def test_storage_backed_vector_reload_stack_tag_allows_scalar_gep():
  storage = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(128, AddrSpace.LOCAL), arg=94)
  handle = StateHandle(StateRegionSpec("vector_state", dtypes.float, 8), PhaseBoundarySpec("publish", "reload"),
                       storage=storage, lane=UOp.special(8, "state_lane"), lane_stride=8)
  reloaded = handle.reload(handle.publish(UOp.const(dtypes.float.vec(8), 1.0)))
  assert reloaded.op is Ops.CUSTOMI and reloaded.arg == ("state_reload_v1", handle)
  lane = reloaded.gep(5)
  assert lane.src == (reloaded,) and lane.dtype == dtypes.float
  type_verify(UOp.sink(lane), spec_full)


def test_one_source_reload_lowers_at_generic_vector_lane_boundary():
  from tinygrad.renderer.isa.amd import lower_state_phase_transfer, native_repack_matcher
  from tinygrad.uop.ops import graph_rewrite
  storage = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(128, AddrSpace.LOCAL), arg=95)
  handle = StateHandle(StateRegionSpec("vector_state", dtypes.float, 8), PhaseBoundarySpec("publish", "reload"),
                       storage=storage, lane=UOp.special(8, "state_lane"), lane_stride=8)
  reloaded = handle.reload(handle.publish(UOp.const(dtypes.float.vec(8), 1.0)))
  assert len(reloaded.src) == 1 and lower_state_phase_transfer(reloaded) is None
  lowered = graph_rewrite(UOp.sink(reloaded.gep(2)), native_repack_matcher, bottom_up=True)
  assert not any(u.op is Ops.CUSTOMI and u.arg == ("state_reload_v1", handle) for u in lowered.toposort())
  assert any(u.op is Ops.LOAD for u in lowered.toposort())


def test_handle_owned_loop_state_lane_lowers_without_register_placeholder():
  from tinygrad.renderer.isa.amd import lower_state_phase_transfer
  storage = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(128, AddrSpace.LOCAL), arg=96)
  handle = StateHandle(StateRegionSpec("loop_state", dtypes.float, 8), PhaseBoundarySpec("loop_write", "loop_read"),
                       storage=storage, lane=UOp.special(8, "state_lane"), lane_stride=8)
  lane = handle.loop_read(3)
  type_verify(UOp.sink(lane), spec_full)
  lowered = lower_state_phase_transfer(lane)
  assert lowered is not None and lowered.op is Ops.LOAD
  assert not any(u.op is Ops.DEFINE_REG for u in lowered.toposort())


def test_handle_owned_loop_read_preserves_after_ordering():
  from tinygrad.renderer.isa.amd import lower_state_phase_transfer
  storage = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(128, AddrSpace.LOCAL), arg=98)
  handle = StateHandle(StateRegionSpec("loop_state", dtypes.float, 8), PhaseBoundarySpec("write", "read"),
                       storage=storage, lane=UOp.special(8, "state_lane"), lane_stride=8)
  dependency = UOp.const(dtypes.float, 2.0)
  read = handle.loop_read(2, after=dependency)
  type_verify(UOp.sink(read), spec_full)
  lowered = lower_state_phase_transfer(read)
  assert lowered.op is Ops.LOAD and lowered.src[0].op is Ops.INDEX
  assert any(u is dependency for u in lowered.toposort())


def test_handle_owned_loop_state_init_iteration_final_write_read_ownership():
  from tinygrad.renderer.isa.amd import lower_state_phase_transfer
  storage = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(128, AddrSpace.LOCAL), arg=97)
  handle = StateHandle(StateRegionSpec("loop_state", dtypes.float, 8), PhaseBoundarySpec("init", "final"),
                       storage=storage, lane=UOp.special(8, "state_lane"), lane_stride=8)
  init = handle.loop_write(UOp.const(dtypes.float, 0.0), 0)
  iteration = handle.loop_write(UOp.const(dtypes.float, 1.0), 0, after=handle.loop_read(0))
  final = handle.loop_read(0)
  type_verify(UOp.sink(init, iteration, final), spec_full)
  assert lower_state_phase_transfer(init).op is Ops.STORE
  assert lower_state_phase_transfer(iteration).op is Ops.STORE
  assert lower_state_phase_transfer(final).op is Ops.LOAD


def test_storage_backed_state_rejects_invalid_storage_lane_and_offset():
  storage = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(64, AddrSpace.LOCAL), arg=92)
  lane = UOp.special(8, "state_lane")
  region = StateRegionSpec("vector_state", dtypes.float, 8)
  boundary = PhaseBoundarySpec("publish", "reload")
  with pytest.raises(TypeError): StateHandle(region, boundary, storage=UOp(Ops.DEFINE_REG, dtypes.float.ptr(64, AddrSpace.REG), arg=93), lane=lane, lane_stride=8).validate()
  with pytest.raises(TypeError): StateHandle(region, boundary, storage=storage, lane=UOp.const(dtypes.float, 0.0), lane_stride=8).validate()
  with pytest.raises(ValueError): StateHandle(region, boundary, storage=storage, lane=lane, lane_stride=8, element_offset=1).validate()


# ---- Single-authority contracts for the native AMD attention ABI -------------------------------
# These are the machine-enforced half of rules that were previously only human-facing. The drain
# lane convention used to exist three times (a decorative string in the spec, UOp arithmetic in
# renderer/cstyle.py, register-level shift/immediate arithmetic in renderer/isa/amd.py) with nothing
# tying them together, so the two renderers could drift from each other and from the declaration.

def test_drain_lane_coeffs_are_the_only_statement_of_the_convention():
  from tinygrad.uop.ops import AMDAttentionOutputDrainSpec
  for head_dim in (64, 128, 256):
    spec = AMDAttentionOutputDrainSpec(head_dim=head_dim, blocks=head_dim//16)
    c_e, c_half, c_j, c_col = spec.drain_lane_coeffs
    # the declared human-facing string is a rendering of the coefficients, not an independent copy
    assert spec.address_expr_text == f"e*{c_e}+halfwave*{c_half}+j*{c_j}+col"
    # the relations both renderers rely on when they factor / shift
    assert c_e == 2*c_half and c_col == 1 and c_j == 16
  # the shipped default declaration must still agree with its own derivation
  AMDAttentionOutputDrainSpec().validate()
  assert AMDAttentionOutputDrainSpec().address_expr == AMDAttentionOutputDrainSpec().address_expr_text


def test_isa_drain_encoding_agrees_with_the_declared_address_expression():
  from tinygrad.uop.ops import AMDAttentionOutputDrainSpec
  from tinygrad.renderer.isa.amd_attention_abi import drain_lane_encoding
  head_dim = 128
  c_e, c_half, c_j, c_col = AMDAttentionOutputDrainSpec(head_dim=head_dim).drain_lane_coeffs
  for output_block_base in (0, 4):
    half_shift, group_row_stride, _ = drain_lane_encoding(head_dim, 0, 0, output_block_base)
    assert 1 << half_shift == c_half and group_row_stride == 16*c_half
    for e in range(8):
      for j in range(4):
        # the encoder splits the address into a runtime VGPR part and a store byte immediate; their
        # sum must reproduce the declared element formula exactly, in bytes.
        byte_imm = drain_lane_encoding(head_dim, e, j, output_block_base)[2]
        for halfwave in range(2):
          for col in (0, 7, 15):
            runtime_bytes = ((halfwave << half_shift) + col*c_col) * 2
            declared = e*c_e + halfwave*c_half + (j+output_block_base)*c_j + col*c_col
            assert runtime_bytes + byte_imm == declared*2


def test_hip_drain_expansion_addresses_match_the_declared_convention():
  from tinygrad.uop.ops import AMDAttentionOutputDrainSpec
  spec = AMDAttentionOutputDrainSpec(head_dim=128, blocks=8)
  c_e, c_half, c_j, c_col = spec.drain_lane_coeffs
  # cstyle.py emits `(2e+halfwave)*c_half + (j+base)*c_j + col`; that factoring is only equal to the
  # declared `e*c_e + halfwave*c_half + ...` while c_e == 2*c_half, which is what it checks.
  for e in range(8):
    for j in range(spec.blocks):
      for halfwave in range(2):
        for col in (0, 15):
          factored = (2*e + halfwave)*c_half + (j + spec.output_block_base)*c_j + col
          declared = e*c_e + halfwave*c_half + (j + spec.output_block_base)*c_j + col*c_col
          assert factored == declared


def test_attention_loop_state_register_map_spans_are_disjoint_and_reserved():
  """Defends amd_register_contracts.AMD_ATTENTION_LOOP_STATE (the ex-`{"m":72,"l":80,"acc":8}` dict).

  The cost of this map having been an undocumented, undefended bare dict is recorded in
  docs/shared-attention-phase-lds-negative-result-20260724.md: publishing the state through a typed
  StateHandle LDS region created a register mirror instead of transferring ownership, because the
  hard alias had no owner to transfer from.
  """
  from tinygrad.renderer.isa.amd_register_contracts import AMD_ATTENTION_LOOP_STATE as M, FRAG_BASE, WMMA_ACC_BASE
  M.validate()
  spans = M.spans()
  assert len(spans) == M.blocks + 4                      # eight acc blocks plus m, l, qk_c, alpha
  occupied = [i for _, start, end in spans for i in range(start, end)]
  assert len(occupied) == len(set(occupied))             # no two roles alias
  assert min(occupied) == WMMA_ACC_BASE == M.acc         # starts at the reserved low accumulator base
  assert max(occupied) < FRAG_BASE                       # stays clear of the high fragment window
  assert sorted(occupied) == list(range(M.acc, M.top()))  # contiguous, so runs() covers it exactly
  assert all(start % M.lanes == 0 for _, start, _ in spans)
  # runs() is what _n_c_runs reports, which is what makes _vpool exclude the window.
  assert M.runs() * M.lanes == M.top() - M.acc


def test_attention_loop_state_register_map_rejects_aliasing_and_bad_roles():
  import pytest as _pytest
  from tinygrad.renderer.isa.amd_register_contracts import AMDAttentionLoopStateMap, AMD_ATTENTION_LOOP_STATE as M
  with _pytest.raises(ValueError): AMDAttentionLoopStateMap(m=80).validate()            # m aliases l
  with _pytest.raises(ValueError): AMDAttentionLoopStateMap(m=73).validate()            # misaligned span
  with _pytest.raises(ValueError): AMDAttentionLoopStateMap(alpha=192).validate()       # collides with fragments
  with _pytest.raises(ValueError): M.base("acc", M.blocks)                              # block out of range
  with _pytest.raises(ValueError): M.base("m", 1)                                       # m has no block dimension
  with _pytest.raises(ValueError): M.base("lanes")                                      # not a role
  with _pytest.raises(ValueError): M.base("nope")


def test_attention_loop_state_registers_are_not_restated_in_the_renderer():
  """No consumer may re-spell the physical map; the boundary is the only source."""
  import pathlib, re
  root = pathlib.Path(__file__).resolve().parents[2] / "tinygrad" / "renderer" / "isa"
  offenders = []
  for path in (root/"amd.py", root/"amd_attention_abi.py", root/"amd_wmma_residency.py"):
    for n, line in enumerate(path.read_text().splitlines(), 1):
      code = line.split("#", 1)[0]
      if re.search(r'\{\s*"(m|l|acc|alpha)"\s*:', code): offenders.append(f"{path.name}:{n}")
  assert not offenders, f"attention loop-state register map restated at {offenders}"
