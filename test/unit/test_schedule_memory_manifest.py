from dataclasses import FrozenInstanceError

import pytest

from tinygrad.dtype import dtypes
from tinygrad.schedule import memory as memory_mod
from tinygrad.schedule.memory import collect_memory_plan_manifests, memory_plan_manifest, memory_plan_rewrite
from tinygrad.callify import AllocCtx, replace_input_buffer
from tinygrad.schedule.memory import _call_bound_owners
from tinygrad.llm.memory_semantics import PREFILL_ACTIVATION, PREFILL_SCRATCH, mark_memory_semantic, memory_semantic_owner
from tinygrad.uop.ops import Ops, UOp


def _buf(size:int, num:int, *, device="CPU", owner=None) -> UOp:
  tag = None if owner is None else ("semantic_owner", owner)
  return UOp.new_buffer(device, size, dtypes.uint8, num=num).replace(tag=tag)


def _schedule(*steps:tuple[str, tuple[UOp, ...]]) -> UOp:
  items = []
  for lane, bufs in steps:
    ast = UOp(Ops.COPY if lane == "copy" else Ops.NOOP)
    items.append(UOp(Ops.SINK, src=(ast,)+bufs))
  return UOp(Ops.SINK, src=tuple(items))


def test_disjoint_lifetimes_reuse_one_exact_range_and_manifest_is_frozen():
  a, b = _buf(17, 1001), _buf(129, 1002)
  manifest = memory_plan_manifest(_schedule(("compute", (a,)), ("compute", (b,))))
  assert manifest.buffers[0].byte_range == manifest.buffers[1].byte_range == (0, 256)
  assert manifest.buffers[0].arena_identity == manifest.buffers[1].arena_identity
  assert [x.physical_byte_union for x in manifest.indices] == [256, 256]
  assert manifest.peak_physical_bytes == 256
  with pytest.raises(FrozenInstanceError): manifest.buffers[0].offset = 1  # type: ignore[misc]

def test_disjoint_semantic_owners_can_alias_one_arena_range():
  a, b = _buf(32, 1081), _buf(32, 1082)
  a = mark_memory_semantic(a, PREFILL_ACTIVATION)
  b = mark_memory_semantic(b, PREFILL_SCRATCH)
  manifest = memory_plan_manifest(_schedule(("compute", (a,)), ("compute", (b,))))
  assert manifest.buffers[0].byte_range == manifest.buffers[1].byte_range
  assert manifest.buffers[0].arena_identity == manifest.buffers[1].arena_identity
  assert [row.semantic_owner for row in manifest.buffers] == [PREFILL_ACTIVATION, PREFILL_SCRATCH]


def test_callified_parameter_keeps_exact_marked_buffer_identity():
  source = _buf(32, 1083)
  source = mark_memory_semantic(source, PREFILL_ACTIVATION)
  parameter = replace_input_buffer(AllocCtx(), source)
  assert memory_semantic_owner(parameter) is None
  assert memory_semantic_owner(source) == PREFILL_ACTIVATION

def test_call_parameter_owner_binding_is_per_invocation():
  parameter = UOp.param(0, dtypes.uint8, (32,), "CPU")
  body = UOp(Ops.SINK, dtypes.void, (UOp(Ops.NOOP), parameter))
  first, second = _buf(32, 1084), _buf(32, 1085)
  first = mark_memory_semantic(first, PREFILL_ACTIVATION)
  second = mark_memory_semantic(second, PREFILL_SCRATCH)
  first_call, second_call = body.call(first), body.call(second)
  assert _call_bound_owners(first_call)[first.src[0]] == PREFILL_ACTIVATION
  assert _call_bound_owners(second_call)[second.src[0]] == PREFILL_SCRATCH
  assert _call_bound_owners(first_call).get(second) is None


def test_overlapping_lifetimes_count_physical_union_not_logical_sum():
  a, b = _buf(257, 1011), _buf(17, 1012)
  manifest = memory_plan_manifest(_schedule(("compute", (a,)), ("compute", (a, b)), ("compute", (b,))))
  assert [x.rounded_bytes for x in manifest.buffers] == [512, 256]
  assert [x.physical_byte_union for x in manifest.indices] == [512, 768, 256]
  assert manifest.peak_physical_bytes == 768


def test_held_output_is_a_dedicated_physical_allocation():
  temporary, output = _buf(32, 1021), _buf(65, 1022, owner="output")
  manifest = memory_plan_manifest(_schedule(("compute", (temporary, output)),), {output})
  held = next(x for x in manifest.buffers if x.semantic_owner == "output")
  assert held.arena_identity.startswith("dedicated:buffer:")
  assert held.arena_size == held.rounded_bytes == 256
  assert manifest.peak_physical_bytes == 512


def test_structurally_marked_held_output_keeps_owner_on_dedicated_allocation():
  output = mark_memory_semantic(_buf(65, 1023), PREFILL_ACTIVATION)
  manifest = memory_plan_manifest(_schedule(("compute", (output,)),), {output.src[0]})
  held = next(x for x in manifest.buffers if x.semantic_owner == PREFILL_ACTIVATION)
  assert held.arena_identity.startswith("dedicated:buffer:")


def test_semantic_categories_may_reuse_at_different_times_and_unknown_is_explicit():
  lhs, rhs, unknown = _buf(32, 1031, owner="lhs"), _buf(32, 1032, owner="rhs"), _buf(32, 1033)
  manifest = memory_plan_manifest(_schedule(("compute", (lhs,)), ("compute", (rhs,)), ("compute", (unknown,))))
  assert len({x.byte_range for x in manifest.buffers}) == 1
  assert [x.semantic_owner for x in manifest.buffers] == ["lhs", "rhs", "unknown"]


def test_conflicting_simultaneous_ownership_of_same_physical_bytes_fails(monkeypatch):
  # A deliberately invalid allocator result proves the evidence layer independently rejects aliasing.
  monkeypatch.setattr(memory_mod.TLSFAllocator, "alloc", lambda self, size: 0)
  monkeypatch.setattr(memory_mod.TLSFAllocator, "free", lambda self, offset: None)
  a, b = _buf(32, 1041, owner="lhs"), _buf(32, 1042, owner="rhs")
  with pytest.raises(ValueError, match="conflicting semantic ownership"):
    memory_plan_manifest(_schedule(("compute", (a, b)),))


def test_copy_compute_lanes_are_separate_and_union_is_summed():
  compute, copy = _buf(32, 1051), _buf(32, 1052)
  manifest = memory_plan_manifest(_schedule(("compute", (compute,)), ("copy", (copy,))))
  assert {(x.lane, x.identity.rsplit(":", 1)[-1]) for x in manifest.arenas} == {(0, "compute"), (1, "copy")}
  assert [x.physical_byte_union for x in manifest.indices] == [256, 256]


def test_copy_lane_manifest_uses_the_planners_extended_lifetime():
  first, second = _buf(32, 1053), _buf(32, 1054)
  manifest = memory_plan_manifest(_schedule(("copy", (first,)), ("copy", (first,)), ("copy", (second,))))
  rows = {x.identity: x for x in manifest.buffers}
  assert rows[f"buffer:{first.key.hex()}"].last_index == 3
  assert rows[f"buffer:{second.key.hex()}"].last_index == 3
  # The planner keeps the first copy alive when the second begins, so their physical ranges cannot be counted as reuse.
  assert [x.physical_byte_union for x in manifest.indices] == [256, 256, 512]


def test_requesting_manifest_does_not_change_rewritten_uops():
  a, b = _buf(32, 1061), _buf(32, 1062)
  linear = _schedule(("compute", (a,)), ("compute", (a, b)))
  linear_key = linear.key
  rewritten = memory_plan_rewrite(linear)
  rewritten_key, rewritten_src = rewritten.key, rewritten.src
  memory_plan_manifest(linear)
  assert linear.key == linear_key
  assert rewritten.key == rewritten_key
  assert rewritten.src == rewritten_src


def test_rewrite_collector_is_opt_in_nested_and_references_exact_arena_uop(monkeypatch):
  a, b = _buf(32, 1063), _buf(32, 1064)
  linear = _schedule(("compute", (a,)), ("compute", (a, b)))
  with collect_memory_plan_manifests() as outer:
    rewritten = memory_plan_rewrite(linear)
    with collect_memory_plan_manifests() as inner:
      memory_plan_rewrite(linear)
  assert len(outer) == 2 and len(inner) == 1
  shared = next(arena for arena in outer[0].arenas if arena.identity == "arena:CPU:compute")
  assert shared.backing_uop is not None
  assert any(u is shared.backing_uop for u in rewritten.toposort())

  # Inactive rewriting does not even attempt manifest construction.
  monkeypatch.setattr(memory_mod, "_memory_plan_manifest", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("collected")))
  memory_plan_rewrite(linear)


def test_requested_collection_fails_closed(monkeypatch):
  linear = _schedule(("compute", (_buf(32, 1065),)),)
  monkeypatch.setattr(memory_mod, "_memory_plan_manifest", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("manifest failed")))
  with collect_memory_plan_manifests(), pytest.raises(ValueError, match="manifest failed"):
    memory_plan_rewrite(linear)


def test_collection_callback_runs_synchronously_and_can_fail_closed():
  linear = _schedule(("compute", (_buf(32, 1066),)),)
  observed = []
  with collect_memory_plan_manifests(lambda manifest: observed.append(manifest.peak_physical_bytes)) as manifests:
    memory_plan_rewrite(linear)
  assert observed == [256] and len(manifests) == 1
  with collect_memory_plan_manifests(lambda manifest: (_ for _ in ()).throw(ValueError("binding failed"))):
    with pytest.raises(ValueError, match="binding failed"): memory_plan_rewrite(linear)


def test_symbolic_size_fails_manifest_completeness():
  size = UOp.variable("manifest_size", 1, 32)
  symbolic = UOp(Ops.BUFFER, dtypes.uint8, (UOp.unique(1071), UOp(Ops.DEVICE, arg="CPU")), size)
  with pytest.raises(ValueError, match="manifest incomplete: unresolved size"):
    memory_plan_manifest(_schedule(("compute", (symbolic,)),))
