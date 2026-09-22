#!/usr/bin/env python3
"""Route B3.5: process-local fail-closed held-buffer selector (CPU only).

Implements a selector that lets a candidate set of logical buffers be treated
as held by the JIT memory planner, without touching any tinygrad runtime file.
This is the process-local plumbing for the B3.6 wall-clock A/B experiment: the
same JIT run executes once with the selector disabled and once with a candidate
set enabled, and only the planner's held set differs.

Semantics
---------

* ``enable_held_buffers(candidate_ids)`` is a context manager. While active it
  installs a wrapper around ``tinygrad.engine.jit.jit_lower`` that injects the
  matched buffers into ``held_bufs`` before the original lowering runs, so
  ``memory_plan_rewrite`` does not arena-place them. The wrapper is installed
  and restored by this module alone; no runtime file is modified.

* Candidate ids address buffers in the planner's view: non-input buffers that
  ``memory_plan_rewrite`` would consider, exactly the buffers the B3.1
  placement manifest enumerates. The canonical id format is
  ``buf:DEVICE:dtype:arg[:ordinal]`` where ``arg`` is the buffer element count
  (``UOp.arg``) and ``ordinal`` is the 1-based occurrence index of that
  ``(DEVICE, dtype, arg)`` key in planner scan order. Omitting ``ordinal``
  defaults to 1. These ids match the ``buf:DEV:dtype:arg:ordinal`` labels of
  the B3.1 manifest one-to-one.

* Fail closed means the selector never silently does nothing. It refuses with
  ``HeldBufferSelectorError`` when a candidate id is malformed or unknown at
  enable time, when an explicit ``expected`` identity contradicts the id, when
  a candidate names no plan-able buffer in the linear, or when the resolved
  buffer's identity does not match the id. Reported "skipped" candidates are
  the only non-error outcomes and are always benign: the buffer was already in
  ``held_bufs`` (``already_held``) or cannot be arena-planned on its device
  (``not_planable``).

* Byte-identical when absent: with no candidate set enabled the runtime path
  is stock. The wrapper reads a per-context ContextVar; when no selector is
  active it forwards to the original ``jit_lower`` untouched, and an empty
  candidate set applies nothing. ``canonical_linear`` normalizes the
  ``Ops.UNIQUE`` counter so pre/post lowering can be compared byte-for-byte.

Usage:
  python3 extra/llm_research/decode/route_b3_5_selector.py --synthetic [--json]
"""
from __future__ import annotations

import argparse, contextlib, contextvars, json, re, sys
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

SCHEMA = "tinygrad.route_b3_5_selector.v1"


class HeldBufferSelectorError(ValueError):
  pass


# ---------------------------------------------------------------------------
# Candidate ids
# ---------------------------------------------------------------------------

_CANDIDATE_RE = re.compile(r"^buf:(?P<device>[^:]+):(?P<dtype>[^:]+):(?P<arg>\d+)(?::(?P<ordinal>\d+))?$")
_UNIQUE_RE = re.compile(r"UOp\(Ops\.UNIQUE, dtypes\.void, arg=\d+")


@dataclass(frozen=True)
class HeldBufferCandidate:
  """One parsed candidate id with its expected buffer identity."""
  id: str
  device: str
  dtype: str  # canonical str(dtype), e.g. "dtypes.float"
  arg: int    # element count of the buffer (UOp.arg)
  ordinal: int  # 1-based occurrence index for the (device, dtype, arg) key


def _canonical_dtype(dtype: Any) -> str:
  from tinygrad.dtype import dtypes
  if isinstance(dtype, str):
    name = dtype[len("dtypes."):] if dtype.startswith("dtypes.") else dtype
    resolved = getattr(dtypes, name, None)
    if resolved is None:
      raise HeldBufferSelectorError(f"unknown dtype {dtype!r} in candidate id")
    return str(resolved)
  return str(dtype)


def parse_candidate_id(candidate_id: str) -> HeldBufferCandidate:
  """Strictly parse a canonical 'buf:DEVICE:dtype:arg[:ordinal]' id."""
  if not isinstance(candidate_id, str) or not candidate_id.startswith("buf:"):
    raise HeldBufferSelectorError(
      f"candidate id {candidate_id!r} must be a canonical 'buf:DEVICE:dtype:arg[:ordinal]' id")
  match = _CANDIDATE_RE.match(candidate_id)
  if match is None:
    raise HeldBufferSelectorError(
      f"candidate id {candidate_id!r} is malformed (expected 'buf:DEVICE:dtype:arg[:ordinal]')")
  device, dtype_s, arg_s, ordinal_s = (match.group("device"), match.group("dtype"),
                                       match.group("arg"), match.group("ordinal"))
  if ":" in device or "," in device:
    raise HeldBufferSelectorError(
      f"candidate id {candidate_id!r}: multi-device buffers are not addressable by canonical id")
  dtype = _canonical_dtype(dtype_s)
  arg = int(arg_s)
  if arg <= 0:
    raise HeldBufferSelectorError(f"candidate id {candidate_id!r}: arg must be a positive element count")
  ordinal = int(ordinal_s) if ordinal_s is not None else 1
  if ordinal < 1:
    raise HeldBufferSelectorError(f"candidate id {candidate_id!r}: ordinal must be >= 1")
  return HeldBufferCandidate(candidate_id, device, dtype, arg, ordinal)


def candidate_id(device: Any, dtype: Any, arg: int, ordinal: int = 1) -> str:
  """Build the canonical candidate id for a buffer identity (mirrors B3.1 ids)."""
  dev = device if isinstance(device, str) else ",".join(sorted(device))
  if ":" in dev:
    raise HeldBufferSelectorError("multi-device buffers are not addressable by canonical id")
  return "buf:%s:%s:%d:%d" % (dev, _canonical_dtype(dtype), int(arg), int(ordinal))


# ---------------------------------------------------------------------------
# Linear inspection (planner view)
# ---------------------------------------------------------------------------

def _key_of(buf: Any) -> tuple[str, str, int]:
  device = buf.device if isinstance(buf.device, str) else ",".join(sorted(buf.device))
  return (device, str(buf.dtype), int(buf.arg))


def linear_buffers(linear: Any, input_uops: Any = ()) -> list[Any]:
  """Non-input BUFFER uops of a linear in planner scan (first-occurrence) order.

  Mirrors exactly the buffers ``memory_plan_rewrite`` sees after ``jit_lower``
  substitutes the input uops with PARAM: every BUFFER reachable from a call's
  arguments, deduplicated by object, minus the input uops. Plan-ability is not
  filtered here; candidates for non-plan-able devices are reported as skipped
  (``not_planable``) rather than silently held.
  """
  from tinygrad.schedule.memory import _collect_bufs
  excluded = set(input_uops)
  seen: set[int] = set()
  order: list[Any] = []
  for call in linear.src:
    for buf in [b for src in call.src[1:] for b in _collect_bufs(src)]:
      if id(buf) in seen or buf in excluded:
        continue
      seen.add(id(buf))
      order.append(buf)
  return order


def _index_buffers(linear: Any, input_uops: Any) -> dict[tuple[str, str, int], list[Any]]:
  by_key: dict[tuple[str, str, int], list[Any]] = {}
  for buf in linear_buffers(linear, input_uops):
    by_key.setdefault(_key_of(buf), []).append(buf)
  return by_key


def canonical_linear(linear: Any) -> str:
  """Deterministic lowering text: str(linear) with Ops.UNIQUE counters normalized."""
  return _UNIQUE_RE.sub("UOp(Ops.UNIQUE, dtypes.void, arg=#", str(linear))


# ---------------------------------------------------------------------------
# Selector state
# ---------------------------------------------------------------------------

@dataclass
class HeldBufferSelector:
  """Tracks which candidate ids were applied vs skipped across jit_lower calls."""
  candidates: tuple[HeldBufferCandidate, ...]
  applied: list[str] = field(default_factory=list)
  skipped: list[dict[str, Any]] = field(default_factory=list)
  errors: list[str] = field(default_factory=list)
  jit_lower_calls: int = 0

  def apply(self, linear: Any, held_bufs: Any, input_uops: Any) -> set[Any]:
    """Return held_bufs plus every matched candidate buffer (fail closed)."""
    from tinygrad.schedule.memory import _can_plan
    by_key = _index_buffers(linear, input_uops)
    held = set(held_bufs)
    added: set[Any] = set()
    applied_now: list[str] = []
    skipped_now: list[dict[str, Any]] = []
    for cand in self.candidates:
      key = (cand.device, cand.dtype, cand.arg)
      bufs = by_key.get(key, [])
      if len(bufs) < cand.ordinal:
        msg = (f"candidate {cand.id!r}: no plan-able buffer ({cand.device}, {cand.dtype}, "
               f"{cand.arg}) at ordinal {cand.ordinal} in this linear; refusing to proceed silently")
        self.errors.append(msg)
        raise HeldBufferSelectorError(msg)
      buf = bufs[cand.ordinal - 1]
      if _key_of(buf) != key:
        msg = (f"candidate {cand.id!r}: buffer identity mismatch, found {_key_of(buf)!r} "
               f"expected {key!r}; refusing to proceed silently")
        self.errors.append(msg)
        raise HeldBufferSelectorError(msg)
      if buf in held:
        skipped_now.append({"id": cand.id, "reason": "already_held",
                            "detail": "buffer already in jit_lower held_bufs"})
      elif not _can_plan(buf, set()):
        skipped_now.append({"id": cand.id, "reason": "not_planable",
                            "detail": "buffer device cannot be arena-planned"})
      else:
        applied_now.append(cand.id)
        added.add(buf)
    self.applied.extend(applied_now)
    self.skipped.extend(skipped_now)
    self.jit_lower_calls += 1
    return held | added

  def report(self) -> dict[str, Any]:
    return {
      "schema": SCHEMA,
      "candidate_count": len(self.candidates),
      "applied": list(self.applied),
      "skipped": [dict(s) for s in self.skipped],
      "errors": list(self.errors),
      "jit_lower_calls": self.jit_lower_calls,
    }


# ---------------------------------------------------------------------------
# Process-local enable/disable
# ---------------------------------------------------------------------------

_active_selector: contextvars.ContextVar[Optional[HeldBufferSelector]] = contextvars.ContextVar(
  "route_b3_5_active_selector", default=None)
_orig_jit_lower: Any = None
_patch_count = 0


def _wrapped_jit_lower(linear: Any, held_bufs: Any, input_uops: Any) -> Any:
  selector = _active_selector.get()
  if selector is None:
    return _orig_jit_lower(linear, held_bufs, input_uops)
  return _orig_jit_lower(linear, selector.apply(linear, held_bufs, input_uops), input_uops)


def _validate_expected(candidates: tuple[HeldBufferCandidate, ...], expected: Any) -> None:
  if expected is None:
    return
  if not isinstance(expected, dict):
    raise HeldBufferSelectorError("expected must be a dict mapping candidate id to expected identity")
  by_id = {c.id: c for c in candidates}
  for cid, exp in expected.items():
    cand = by_id.get(cid)
    if cand is None:
      raise HeldBufferSelectorError(f"expected identity given for unknown candidate id {cid!r}")
    if not isinstance(exp, dict):
      raise HeldBufferSelectorError(f"expected identity for {cid!r} must be a dict")
    for field_name in ("device", "dtype", "arg", "ordinal"):
      if field_name not in exp:
        continue
      value = exp[field_name]
      normalized = _canonical_dtype(value) if field_name == "dtype" else value
      if normalized != getattr(cand, field_name):
        raise HeldBufferSelectorError(
          f"candidate {cid!r}: expected {field_name}={value!r} contradicts canonical id "
          f"({getattr(cand, field_name)!r}); refusing")


@contextlib.contextmanager
def enable_held_buffers(candidate_ids: list[str], expected: dict[str, dict[str, Any]] | None = None,
                        ) -> Iterator[HeldBufferSelector]:
  """Hold the given candidate buffers during jit_lower (fail closed, CPU only).

  Validates every candidate id and the optional explicit ``expected`` identity
  map before installing the jit_lower wrapper. Yields the selector; its
  ``report()`` (and ``applied``/``skipped`` lists) tell which candidate ids
  were actually injected vs benignly skipped. The wrapper is always restored
  on exit, so the runtime path is byte-identical to stock outside the context.
  """
  global _patch_count, _orig_jit_lower
  if _active_selector.get() is not None:
    raise HeldBufferSelectorError(
      "a held-buffer selector is already active in this context; nested enable is not supported")
  candidates = tuple(parse_candidate_id(cid) for cid in candidate_ids)
  _validate_expected(candidates, expected)
  selector = HeldBufferSelector(candidates)
  if _patch_count == 0:
    from tinygrad.engine import jit as tjit
    _orig_jit_lower = tjit.jit_lower
    tjit.jit_lower = _wrapped_jit_lower
  _patch_count += 1
  token = _active_selector.set(selector)
  try:
    yield selector
  finally:
    _active_selector.reset(token)
    _patch_count -= 1
    if _patch_count == 0:
      from tinygrad.engine import jit as tjit
      tjit.jit_lower = _orig_jit_lower
      _orig_jit_lower = None


# ---------------------------------------------------------------------------
# Synthetic CLI (hermetic, CPU only)
# ---------------------------------------------------------------------------

def _cpu_allocator_has_offset() -> bool:
  from tinygrad.device import Device
  return hasattr(Device["CPU"].allocator, "_offset")


def _multi_buffer_linear() -> tuple[Any, list[Any]]:
  """A linear whose planner view has several plan-able non-input buffers."""
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


def _plan_and_lower(linear: Any, held: Any, input_uops: Any) -> tuple[dict[str, Any], Any]:
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


def _capture_via_tinyjit() -> Any:
  """Capture a real TinyJit and return the lowered (planned) linear."""
  from tinygrad.dtype import dtypes
  from tinygrad.engine.jit import TinyJit
  from tinygrad.helpers import Context
  from tinygrad.tensor import Tensor

  @TinyJit
  def fn(x: Tensor, y: Tensor) -> Tensor:
    return (x @ y).relu()

  a = Tensor.empty(8, 8, dtype=dtypes.float32, device="CPU")
  b = Tensor.empty(8, 8, dtype=dtypes.float32, device="CPU")
  with Context(JIT=1, NO_MEMORY_PLANNER=0):
    fn(a, b)  # cnt 0: eager execution
    fn(a, b)  # cnt 1: capture; jit_lower fires here
  return fn.captured.linear


def run_synthetic() -> dict[str, Any]:
  """Self-checking CLI body: byte-identical absent, changed placements enabled,
  fail-closed rejection, all on DEV=CPU with no GPU involvement."""
  from tinygrad.helpers import Context

  report: dict[str, Any] = {
    "schema": SCHEMA,
    "device": "CPU",
    "cpu_allocator_has_offset": _cpu_allocator_has_offset(),
    "planner_plans_on_cpu": None,
    "linear_calls": None,
    "plan_able_buffer_count": None,
    "byte_identical": {},
    "enabled": {},
    "fail_closed": {},
  }

  def check(name: str, ok: bool, detail: str) -> None:
    if not ok:
      raise HeldBufferSelectorError(f"synthetic self-test failed: {name} ({detail})")

  with Context(NO_MEMORY_PLANNER=0):
    linear, input_uops = _multi_buffer_linear()
    report["linear_calls"] = len(linear.src)
    buffers = linear_buffers(linear, input_uops)
    report["plan_able_buffer_count"] = len(buffers)
    check("cpu_allocator_offset", report["cpu_allocator_has_offset"],
          "CPU allocator must implement _offset for memory_plan_rewrite to plan on CPU")
    check("planner_plans_on_cpu", len(buffers) >= 2, "expected several plan-able CPU buffers")

    # Stock baseline plus the byte-identical proofs.
    manifest0, linear0 = _plan_and_lower(linear, set(), input_uops)
    report["planner_plans_on_cpu"] = len(manifest0) > 0
    _, linear0_rerun = _plan_and_lower(linear, set(), input_uops)
    _, linear0_empty = (None, None)
    with enable_held_buffers([]) as empty_sel:
      _, linear0_empty = _plan_and_lower(linear, set(), input_uops)
      empty_report = empty_sel.report()
    captured_stock = _capture_via_tinyjit()
    with enable_held_buffers([]):
      captured_empty = _capture_via_tinyjit()
    report["byte_identical"] = {
      "stock_rerun": canonical_linear(linear0) == canonical_linear(linear0_rerun),
      "empty_enable": canonical_linear(linear0) == canonical_linear(linear0_empty),
      "tinyjit_capture": canonical_linear(captured_stock) == canonical_linear(captured_empty),
      "empty_selector_report": empty_report,
    }
    check("byte_identical.stock_rerun", report["byte_identical"]["stock_rerun"],
          "jit_lower must be deterministic modulo Ops.UNIQUE")
    check("byte_identical.empty_enable", report["byte_identical"]["empty_enable"],
          "empty candidate set must leave lowering byte-identical")
    check("byte_identical.tinyjit_capture", report["byte_identical"]["tinyjit_capture"],
          "real TinyJit capture must be byte-identical with an empty selector")

    # Enabled: hold the first plan-able buffer; must equal natively holding it.
    target_id = candidate_id(*_key_of(buffers[0]), 1)
    with enable_held_buffers([target_id]) as selector:
      manifest_sel, linear_sel = _plan_and_lower(linear, set(), input_uops)
      sel_report = selector.report()
    manifest_native, linear_native = _plan_and_lower(linear, {buffers[0]}, input_uops)
    report["enabled"] = {
      "candidate": target_id,
      "applied": sel_report["applied"],
      "skipped": sel_report["skipped"],
      "held_buffer_removed_from_arena": len(manifest_sel) == len(manifest0) - 1,
      "matches_native_hold_manifest": manifest_sel == manifest_native,
      "matches_native_hold_linear": canonical_linear(linear_sel) == canonical_linear(linear_native),
      "linear_changed": canonical_linear(linear0) != canonical_linear(linear_sel),
      "other_buffers_still_placed": len(manifest_sel) == len(manifest_native),
    }
    check("enabled.applied", report["enabled"]["applied"] == [target_id],
          "candidate present in the linear must be applied")
    check("enabled.removed_from_arena", report["enabled"]["held_buffer_removed_from_arena"],
          "held buffer must not receive an arena offset")
    check("enabled.matches_native_manifest", report["enabled"]["matches_native_hold_manifest"],
          "selector injection must be indistinguishable from natively holding the buffer")
    check("enabled.matches_native_linear", report["enabled"]["matches_native_hold_linear"],
          "selector injection must be indistinguishable from natively holding the buffer")
    check("enabled.linear_changed", report["enabled"]["linear_changed"],
          "holding a planned buffer must change the lowered linear")

    # Fail closed: unknown id, malformed id, and contradicting expected identity.
    def refuse(label: str, thunk: Any) -> dict[str, Any]:
      try:
        thunk()
      except HeldBufferSelectorError as exc:
        return {"refused": True, "error": str(exc)}
      raise HeldBufferSelectorError(f"fail-closed {label} was not refused")

    def with_candidates(ids: list[str], expected: dict[str, Any] | None = None) -> None:
      with enable_held_buffers(ids, expected):
        _plan_and_lower(linear, set(), input_uops)

    report["fail_closed"] = {
      "absent_id": refuse("absent_id", lambda: with_candidates(["buf:CPU:dtypes.float:999999:1"])),
      "malformed_id": refuse("malformed_id", lambda: with_candidates(["not-a-candidate-id"])),
      "expected_mismatch": refuse("expected_mismatch", lambda: with_candidates(
        [target_id], {target_id: {"dtype": "dtypes.half"}})),
    }
    check("fail_closed.absent_id", report["fail_closed"]["absent_id"]["refused"],
          "a candidate absent from the linear must be refused")
    check("fail_closed.malformed_id", report["fail_closed"]["malformed_id"]["refused"],
          "a malformed candidate id must be refused")
    check("fail_closed.expected_mismatch", report["fail_closed"]["expected_mismatch"]["refused"],
          "a contradicting expected identity must be refused")
  return report


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--synthetic", action="store_true", help="run the hermetic CPU self-check")
  ap.add_argument("--json", action="store_true", help="emit the report as JSON")
  args = ap.parse_args()
  if not args.synthetic:
    ap.error("no mode selected (--synthetic)")
    return 2
  report = run_synthetic()
  if args.json:
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0
  print("route B3.5 selector synthetic self-check (CPU):")
  print("  cpu_allocator_has_offset: %s" % report["cpu_allocator_has_offset"])
  print("  planner_plans_on_cpu: %s (%d plan-able buffers)" %
        (report["planner_plans_on_cpu"], report["plan_able_buffer_count"]))
  print("  byte_identical: %s" % report["byte_identical"])
  print("  enabled: %s" % report["enabled"])
  print("  fail_closed: %s" % report["fail_closed"])
  return 0


if __name__ == "__main__":
  sys.exit(main())
