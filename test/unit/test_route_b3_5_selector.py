"""Hermetic CPU-only tests for Route B3.5 fail-closed held-buffer selector.

Covers the required cases: (a) absent selector leaves lowering byte-identical
to stock, (b) an enabled candidate present in the linear leaves the arena while
the remaining placements equal natively holding that buffer, (c) unknown or
mismatched ids are refused, (d) applied vs skipped reporting, (e) synthetic
CLI smoke. No GPU is touched.
"""
import json
import os
import subprocess
import sys

import pytest

from extra.llm_research.decode.route_b3_5_selector import (
  HeldBufferSelector, HeldBufferSelectorError, SCHEMA, canonical_linear,
  enable_held_buffers, linear_buffers, parse_candidate_id,
)


def _multi_buffer_linear():
  """Linear whose planner view has several plan-able non-input buffers."""
  from tinygrad.dtype import dtypes
  from tinygrad.tensor import Tensor
  a = Tensor.empty(16, 16, dtype=dtypes.float32, device="CPU")
  b = Tensor.empty(16, 16, dtype=dtypes.float32, device="CPU")
  c = Tensor.empty(16, 16, dtype=dtypes.float32, device="CPU")
  d = Tensor.empty(16, 16, dtype=dtypes.float32, device="CPU")
  t1 = (a @ b).realize()
  t2 = (c @ d).realize()
  linear = (t1 + t2).schedule_linear()
  input_uops = [a.uop.base, b.uop.base, c.uop.base, d.uop.base]
  return linear, input_uops


def _plan_and_lower(linear, held, input_uops):
  """Run jit_lower with a placement collector installed; return (manifest, linear)."""
  from extra.llm_research.decode.route_b3_dag_attribution import PlannerManifestCollector
  from tinygrad.engine.jit import jit_lower
  from tinygrad.schedule import memory as tmem
  collector = PlannerManifestCollector()
  token = tmem._memory_manifest_collectors.set((collector,))
  try:
    lowered = jit_lower(linear, set(held), input_uops)
  finally:
    tmem._memory_manifest_collectors.reset(token)
  return collector.manifest, lowered


def test_absent_selector_byte_identical_and_restore():
  """(a) No active candidate set: lowering is byte-identical to stock, and the
  jit_lower seam is restored to the original object on exit."""
  from tinygrad.engine import jit as tjit
  from tinygrad.helpers import Context
  linear, input_uops = _multi_buffer_linear()
  original = tjit.jit_lower
  assert tjit.jit_lower is original  # nothing patched before any enable
  with Context(NO_MEMORY_PLANNER=0):
    stock = tjit.jit_lower(linear, set(), input_uops)
    with enable_held_buffers([]) as empty_sel:
      assert tjit.jit_lower is not original  # wrapper installed
      wrapped = tjit.jit_lower(linear, set(), input_uops)
    assert tjit.jit_lower is original  # wrapper restored
  assert canonical_linear(stock) == canonical_linear(wrapped)
  assert len(wrapped.src) == len(stock.src)
  assert empty_sel.applied == [] and empty_sel.skipped == []


def test_enabled_holds_buffer_out_of_arena_surgically():
  """(b) An enabled candidate present in the linear leaves arena placement,
  and the remaining arena assignments are exactly those of natively holding
  the same buffer (no collateral effects on other buffers)."""
  from tinygrad.helpers import Context
  linear, input_uops = _multi_buffer_linear()
  buffers = linear_buffers(linear, input_uops)
  target = buffers[0]
  target_id = parse_candidate_id("buf:CPU:dtypes.float:256:1")
  assert (target_id.device, target_id.dtype, target_id.arg) == (
    target.device, str(target.dtype), int(target.arg))
  with Context(NO_MEMORY_PLANNER=0):
    manifest0, linear0 = _plan_and_lower(linear, set(), input_uops)
    assert any(int(k.rsplit(":", 1)[-1]) >= 1 for k in manifest0)
    with enable_held_buffers([target_id.id]) as selector:
      manifest_sel, linear_sel = _plan_and_lower(linear, set(), input_uops)
      assert selector.applied == [target_id.id] and selector.skipped == []
    manifest_native, linear_native = _plan_and_lower(linear, {target}, input_uops)
  # The held buffer is out of the arena: one fewer placement, identical to
  # natively holding it, and the lowered linear is byte-identical to native.
  assert len(manifest_sel) == len(manifest0) - 1
  assert manifest_sel == manifest_native
  assert canonical_linear(linear_sel) == canonical_linear(linear_native)
  # Enabling changes placements relative to stock but touches nothing else.
  assert canonical_linear(linear0) != canonical_linear(linear_sel)
  assert len(manifest_sel) == len(manifest_native)


def test_unknown_id_fails_closed():
  """(c) A well-formed candidate absent from the linear is refused at
  jit_lower time; malformed ids and contradicting expected identities are
  refused at enable time."""
  from tinygrad.helpers import Context
  linear, input_uops = _multi_buffer_linear()
  with Context(NO_MEMORY_PLANNER=0):
    with pytest.raises(HeldBufferSelectorError):
      with enable_held_buffers(["buf:CPU:dtypes.float:999999:1"]):
        _plan_and_lower(linear, set(), input_uops)
    with pytest.raises(HeldBufferSelectorError, match="must be a canonical"):
      with enable_held_buffers(["not-a-candidate-id"]):
        pass
    with pytest.raises(HeldBufferSelectorError, match="ordinal"):
      with enable_held_buffers(["buf:CPU:dtypes.float:256:0"]):
        pass
    with pytest.raises(HeldBufferSelectorError, match="contradicts"):
      with enable_held_buffers(["buf:CPU:dtypes.float:256:1"],
                               expected={"buf:CPU:dtypes.float:256:1": {"dtype": "dtypes.half"}}):
        pass
    with pytest.raises(HeldBufferSelectorError, match="unknown candidate"):
      with enable_held_buffers(["buf:CPU:dtypes.float:256:1"],
                               expected={"buf:CPU:dtypes.float:256:1": {},
                                         "buf:CPU:dtypes.float:128:1": {}}):
        pass


def test_reporting_applied_vs_skipped():
  """(d) Applied and skipped candidate ids are reported; an already-held buffer
  is a benign skip, and a non-plan-able (DISK) buffer is a not_planable skip."""
  from tinygrad.dtype import dtypes
  from tinygrad.helpers import Context
  from tinygrad.uop.ops import Ops, UOp
  linear, input_uops = _multi_buffer_linear()
  buffers = linear_buffers(linear, input_uops)
  id1 = "buf:CPU:dtypes.float:256:1"
  id2 = "buf:CPU:dtypes.float:256:2"
  with Context(NO_MEMORY_PLANNER=0):
    with enable_held_buffers([id1, id2]) as selector:
      _plan_and_lower(linear, {buffers[0]}, input_uops)  # buffers[0] already held
  assert selector.applied == [id2]
  assert any(s["id"] == id1 and s["reason"] == "already_held" for s in selector.skipped)
  assert selector.report()["schema"] == SCHEMA
  assert set(selector.report()) == {"schema", "candidate_count", "applied", "skipped", "errors", "jit_lower_calls"}

  # Pure-logic skip: a DISK buffer cannot be arena-planned and is skipped.
  disk = UOp.new_buffer("DISK", 100, dtypes.float)
  disk_linear = UOp(Ops.LINEAR, src=(UOp(Ops.CALL, src=(UOp(Ops.SINK), disk)),))
  disk_sel = HeldBufferSelector((parse_candidate_id("buf:DISK:dtypes.float:100:1"),))
  new_held = disk_sel.apply(disk_linear, set(), [])
  assert new_held == set()
  assert disk_sel.applied == []
  assert disk_sel.skipped[0]["reason"] == "not_planable"


def test_synthetic_cli_smoke():
  """(e) The --synthetic CLI exits 0 and reports byte-identical absent,
  changed placements enabled, and fail-closed rejection."""
  root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  cli = os.path.join(root, "extra", "llm_research", "decode", "route_b3_5_selector.py")
  env = dict(os.environ, PYTHONPATH=root)
  proc = subprocess.run(
    [sys.executable, cli, "--synthetic", "--json"],
    capture_output=True, text=True, env=env, cwd=root)
  assert proc.returncode == 0, proc.stderr
  report = json.loads(proc.stdout)
  assert report["schema"] == SCHEMA
  assert report["cpu_allocator_has_offset"] is True
  assert report["planner_plans_on_cpu"] is True
  assert report["byte_identical"]["stock_rerun"] is True
  assert report["byte_identical"]["empty_enable"] is True
  assert report["byte_identical"]["tinyjit_capture"] is True
  assert report["enabled"]["applied"]
  assert report["enabled"]["held_buffer_removed_from_arena"] is True
  assert report["enabled"]["matches_native_hold_manifest"] is True
  assert report["enabled"]["matches_native_hold_linear"] is True
  assert report["enabled"]["linear_changed"] is True
  assert report["fail_closed"]["absent_id"]["refused"] is True
  assert report["fail_closed"]["malformed_id"]["refused"] is True
  assert report["fail_closed"]["expected_mismatch"]["refused"] is True
