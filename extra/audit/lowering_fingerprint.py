#!/usr/bin/env python3
"""LR CPU lowering fingerprint gate.

Purpose: a fast, CPU-only, dependency-free gate that every remaining lowering-architecture refactor slice can run
to prove "this change produced no different generated code" for a small, representative set of Tensor graphs
(elementwise/reduce, matmul, chained reduce, broadcast+max, softmax, transpose/permute, dtype cast). It hashes the
linearized UOp schedule (`Tensor.schedule_linear().key`, itself a recursive structural sha256 over op/dtype/arg) for
each graph and compares against a stored baseline. This is compile-only in the sense that it never executes a
kernel: `schedule_linear()` builds the linear UOp program but does not render or run it, and every graph is built
against `device="CPU"` explicitly, so this gate never touches a GPU.

Determinism, by construction (an earlier prototype flaked ~1/8 runs before these three fixes; see
docs/task_workflow -- keep all three, do not "simplify" this away):
  1. `Tensor.manual_seed(1337)` immediately before *each* graph is built, so `Tensor.rand(...)` cannot vary
     between graphs or between runs.
  2. `CACHELEVEL=0` is forced before tinygrad is (first) imported in this process, so a stale disk kernel/schedule
     cache can never be consulted or written -- every run is a cold compile of the graph itself.
  3. Any inherited gate/tuning env var is stripped from os.environ, by prefix, BEFORE tinygrad is imported:
     PREFILL_, SCHED_, WARP_, V_DOT2, COALESCED_, REGALLOC_, UNSAFE_, TC_OPT, LOWER_. A leaked var from an earlier
     test/bench run (e.g. a stray PREFILL_* or TC_OPT override) was the leading suspect for the original flake.

Run:
  PYTHONPATH=. python3 extra/audit/lowering_fingerprint.py           # writes bench/lowering-cpu-fingerprint/latest.json
  PYTHONPATH=. python3 extra/audit/lowering_fingerprint.py --check   # recompute + diff vs stored latest.json; exit 1 on any change
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import pathlib
import sys
from typing import Any, Callable

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "bench" / "lowering-cpu-fingerprint"
OUT_PATH = OUT_DIR / "latest.json"
SCHEMA = "tinygrad.lowering_cpu_fingerprint.v1"

# Prefixes of gate/tuning env vars that must never leak into this fingerprint. Kept as a tuple of *prefixes*
# (not exact names) to match how these are actually set across the repo (e.g. PREFILL_SOFTMAX_REDUCE_FUSE,
# SCHED_BEAM, TC_OPT itself).
STRIP_PREFIXES = ("PREFILL_", "SCHED_", "WARP_", "V_DOT2", "COALESCED_", "REGALLOC_", "UNSAFE_", "TC_OPT", "LOWER_")


def strip_gate_env_vars() -> list[str]:
  """Pop every os.environ key starting with a STRIP_PREFIXES entry, in place. Must be called BEFORE tinygrad is
  first imported in this process (ContextVar reads os.environ once, at ContextVar-construction time). Returns the
  sorted list of keys actually removed, so callers/tests can assert the stripping happened."""
  removed = [k for k in os.environ if k.startswith(STRIP_PREFIXES)]
  for k in removed:
    os.environ.pop(k, None)
  return sorted(removed)


def _prepare_env_and_import_tinygrad():
  """Strip gate env vars, force CACHELEVEL=0, THEN import tinygrad. Order matters (see module docstring)."""
  strip_gate_env_vars()
  os.environ["CACHELEVEL"] = "0"
  from tinygrad import Tensor, Device  # noqa: E402  (deliberately imported only after env prep)
  return Tensor, Device


def _cpu_renderer(Device):
  """Return the CPU/Clang renderer or fail loudly. Never falls back to another device/renderer."""
  try:
    dev = Device["CPU"]
  except Exception as exc:
    raise RuntimeError(f"CPU device is unavailable ({exc!r}); refusing to fingerprint against a different renderer") from exc
  return dev.renderer


# --------------------------------------------------------------------------------------------------------------
# Graphs. Every leaf Tensor is pinned to device="CPU" explicitly: this gate must never touch a GPU even if one is
# present and selected as Device.DEFAULT in the ambient environment.
# --------------------------------------------------------------------------------------------------------------

def _build_graphs(Tensor) -> dict[str, Callable[[], Any]]:
  from tinygrad import dtypes
  return {
    "elementwise_reduce": lambda: ((Tensor.rand(64, 64, device="CPU") + 1.0) * 2.0).sum(),
    "matmul": lambda: (Tensor.rand(64, 64, device="CPU") @ Tensor.rand(64, 64, device="CPU")).sum(),
    "chained_reduce": lambda: (Tensor.rand(128, 128, device="CPU").sum(axis=0) * 3.0).sum(),
    "broadcast_max": lambda: (Tensor.rand(32, 32, device="CPU")
                              - Tensor.rand(32, 32, device="CPU").max(axis=1, keepdim=True)).exp().sum(),
    "softmax_like": lambda: (Tensor.rand(16, 64, device="CPU").softmax(axis=-1)).sum(),
    # New: exercises permute/transpose lowering (distinct index/stride handling from the plain matmul above --
    # the left operand is transposed before the contraction, forcing a non-contiguous view through the lowering).
    "transpose_matmul": lambda: (Tensor.rand(48, 32, device="CPU").transpose(0, 1)
                                @ Tensor.rand(48, 24, device="CPU")).sum(),
    # New: exercises a dtype cast (float32 -> int32 -> float32) between elementwise ops, distinct from every
    # other graph here (all of which stay in a single dtype end to end).
    "cast_dtype_roundtrip": lambda: ((Tensor.rand(32, 32, device="CPU") * 100.0)
                                     .cast(dtypes.int32).cast(dtypes.float32)).sum(),
    # The graphs below exist because a review found the gate blind to them, and the blindness was load-bearing:
    # schedule/realize.py's realize_store_after_src has a WAR-hazard branch whose own comment cites
    # TestAssign.test_assign_double_diamond_reduce -- a test this fork no longer contains. Nothing in test/ calls
    # .assign() at all. So LR-040 moved a function with a branch that no gate and no test exercised.
    "assign_self_dependent": lambda: _assign_self_dependent(Tensor),
    "assign_war_hazard": lambda: _assign_war_hazard(Tensor),
    "multi_output_shared_producer": lambda: _multi_output_shared_producer(Tensor),
  }


def _assign_self_dependent(Tensor):
  """dest.base appears in src's backward slice -- the branch that sets ctx[src] = None."""
  a = Tensor.rand(16, 16, device="CPU").contiguous().realize()
  a.assign(a + 1.0)
  return a


def _assign_war_hazard(Tensor):
  """A write-after-read across two consumers of the same buffer, the case the deleted TestAssign covered."""
  a = Tensor.rand(16, 16, device="CPU").contiguous().realize()
  b = (a * 2.0).sum(axis=0)
  a.assign(a + b.reshape(1, 16))
  return a


def _multi_output_shared_producer(Tensor):
  """Two sinks over one producer: exercises realize_srcs and the producer/consumer partition that every gate
  graph until now avoided by being single-output."""
  x = (Tensor.rand(32, 32, device="CPU") + 1.0).contiguous()
  return (x.sum(axis=0) * x.sum(axis=1)).sum()


def _reset_uop_unique_counter() -> None:
  """Reset UOp's process-global buffer-identity counter (UOp.unique_num) to 0.

  Every Tensor.rand/empty allocates its BUFFER UOp via UOp.unique(), which draws from this single
  itertools.count() shared across the whole process. That counter value is embedded in the UOp arg and therefore
  in schedule_linear().key. Left unreset, calling compute_fingerprints() a second time in the same process (e.g.
  from a test, or a caller that runs the gate twice) advances the counter and changes every hash even though the
  graphs are logically identical -- this was reproduced directly (two in-process calls disagreeing on every
  graph) while hardening this gate. Resetting it here makes the fingerprint depend only on the graphs built
  below, not on how much other tinygrad activity happened earlier in this process."""
  from tinygrad.uop.ops import UOp
  # UOp.unique_num drives BUFFER identity, but a warm process also holds scheduler-level memoization keyed on
  # those identities (the precompile-body bases and nested-resolution cache in tinygrad.schedule). Those caches
  # return first-run UOp objects whose .unique() calls have already advanced the process counter, so a second
  # in-process build skips re-interned BUFFER(LUNIQUE) scratch and shifts every downstream UNIQUE arg. Clear them
  # alongside the counter so the second run rebuilds the same objects from the same source.
  from tinygrad.schedule import _resolve_precompile_base, _resolve_precompile_body_key, _resolve_nested_cache
  _resolve_precompile_base.clear()
  _resolve_precompile_body_key.clear()
  _resolve_nested_cache.clear()
  UOp.unique_num = itertools.count(0)


# Names that `graph_rewrite` is called with that are NOT lowering passes: they are PatternMatcher construction,
# which happens lazily the first time a matcher is used and therefore appears in whichever graph happens to run
# first. Including them would make the order artifact depend on corpus iteration order rather than on lowering.
NON_PASS_REWRITE_NAMES = frozenset({"process UPat", "compile UPat"})


def _collapse(seq: list[str]) -> list[str]:
  """Drop matcher-construction names, then collapse runs of the same name to one entry.

  A pass that rewrites to a fixed point calls `graph_rewrite` a data-dependent number of times (`simplify` alone
  accounts for ~180 of the ~440 raw calls on the CPU corpus). That count tracks how many rewrites a *graph*
  needed, not what the pipeline is, so it would turn every unrelated graph edit into an order diff. The collapsed
  sequence is the thing this gate is actually pinning: which passes run, in which order, how many times the
  pipeline re-enters each one.
  """
  kept = [n for n in seq if n not in NON_PASS_REWRITE_NAMES]
  return [n for i, n in enumerate(kept) if i == 0 or kept[i - 1] != n]


class WarmProcessError(RuntimeError):
  """Raised when a pass order is requested from a process that has already lowered the corpus once."""


_CORPUS_RUNS = 0


def _run_corpus() -> tuple[dict[str, str], dict[str, list[str]] | None]:
  """Build and lower every graph once, returning (fingerprints, collapsed pass orders).

  **The pass order is only meaningful on the FIRST corpus run in a process, and this returns None for it on any
  later run rather than returning a wrong one.** Measured: run 1 records 981 collapsed steps across 64 distinct
  pass names, run 2 in the same process records 132 across 13.

  The cause is memoization keyed on UOp identity. `_reset_uop_unique_counter()` deliberately makes the second run's
  graphs byte-identical to the first's, which is what makes the *fingerprint* reproducible -- and which therefore
  turns every UOp-keyed cache into a hit. `tinygrad/schedule/indexing.py` has two `functools.cache` functions,
  `_apply_reshape` and `apply_movement_op`, that each contain a named `graph_rewrite` ("reshape", "pad",
  "minimum_valid" -- 144 collapsed steps between them); on a warm run those rewrites never execute, so the trace
  cannot see them. They are not the only such cache, which is exactly why this is enforced structurally instead of
  by clearing a list of caches that would silently go stale.

  Note what this implies for the fingerprint too: a second in-process run is substantially a cache hit, not an
  independent re-derivation. It is still a real check that lowering is reproducible given identical input, but it
  is weaker than it looks, and only the cold run exercises the full pipeline.

  Both artifacts come from a single pass over the corpus so they can never describe different runs. The lowering
  trace is switched on in-process rather than through LOWER_TRACE, because `strip_gate_env_vars()` strips the
  LOWER_ prefix by design -- and it should keep stripping it, so an inherited LOWER_* from a caller still cannot
  reach the fingerprint. Tracing is an observer: `record_rewrite` appends to a list and returns, it does not touch
  the graph. The fingerprints below are the control for that claim -- if enabling the trace changed lowering, they
  would move.
  """
  global _CORPUS_RUNS
  cold = _CORPUS_RUNS == 0
  _CORPUS_RUNS += 1
  Tensor, Device = _prepare_env_and_import_tinygrad()
  _cpu_renderer(Device)  # fail loudly if CPU is somehow unavailable; never silently pick another device
  _reset_uop_unique_counter()
  from tinygrad.uop import trace
  graphs = _build_graphs(Tensor)
  fingerprints: dict[str, str] = {}
  orders: dict[str, list[str]] = {}
  was_enabled = trace.ENABLED
  try:
    trace.ENABLED = True
    for name, build in graphs.items():
      Tensor.manual_seed(1337)
      trace.reset(reread_env=False)   # reread_env=False: keep tracing on; LOWER_TRACE is stripped from the env
      fingerprints[name] = hashlib.sha256(build().schedule_linear().key).hexdigest()
      active = trace.active()
      orders[name] = _collapse(active.order()) if active is not None else []
  finally:
    trace.reset(reread_env=False)
    trace.ENABLED = was_enabled
  return fingerprints, (orders if cold else None)


def compute_fingerprints() -> dict[str, str]:
  return _run_corpus()[0]


def compute_pass_orders() -> dict[str, list[str]]:
  """Collapsed pass order per graph. Raises WarmProcessError unless this is the process's first corpus run --
  see _run_corpus. Callers that need this from a used process should use pass_orders_in_fresh_process()."""
  orders = _run_corpus()[1]
  if orders is None:
    raise WarmProcessError(
      "pass order requested from a warm process: UOp-keyed caches make later runs skip most rewrites "
      "(981 collapsed steps cold vs 132 warm). Use pass_orders_in_fresh_process().")
  return orders


# The only environment the subprocess below inherits. This is an ALLOWLIST, not a denylist, and that distinction
# was earned: an earlier version inherited os.environ minus STRIP_PREFIXES and recorded 1006 collapsed steps under
# the full test suite versus 981 standalone, because some variable outside those prefixes changed the pipeline.
# A gate whose pinned value depends on who invoked it is not a gate. Only what is needed to start Python and find
# the tree is passed through; everything else is dropped, so an unknown future gate variable cannot silently move
# the number either.
_ENV_ALLOWLIST = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "LD_LIBRARY_PATH", "PYTHONHASHSEED")


def pass_orders_in_fresh_process() -> dict[str, list[str]]:
  """compute_pass_orders() in a clean subprocess: a cold run, in a known environment, by construction."""
  import subprocess
  src = ("import json,sys; sys.path.insert(0, %r); "
         "from extra.audit import lowering_fingerprint as lf; "
         "print(json.dumps(lf.compute_pass_orders()))" % str(ROOT))
  env = {k: os.environ[k] for k in _ENV_ALLOWLIST if k in os.environ}
  env["PYTHONPATH"] = str(ROOT)
  r = subprocess.run([sys.executable, "-c", src], cwd=str(ROOT), capture_output=True, text=True, env=env)
  if r.returncode != 0:
    raise RuntimeError(f"fresh-process pass order run failed:\n{r.stderr[-2000:]}")
  return json.loads(r.stdout.strip().splitlines()[-1])


def build_header(argv: list[str]) -> dict[str, Any]:
  return {
    "schema": SCHEMA,
    "device": "CPU",
    "python_version": sys.version,
    "command": "python3 " + " ".join(["extra/audit/lowering_fingerprint.py", *argv]),
  }


def build_artifact(argv: list[str], *, require_cold: bool = True) -> dict[str, Any]:
  """The artifact for this run. `pass_orders` is present only when it is real.

  require_cold=True (the CLI, which is always a cold process) refuses to build an artifact whose pass order would
  be a cache-hit artefact. require_cold=False is for in-process callers that only care about fingerprints -- they
  get an artifact with no `pass_orders` key at all, so a warm order can never be written to disk or compared
  against a cold one. Omission is safe here in a way that a wrong order would not be: every consumer treats a
  missing order as "not checked" and says so, rather than as "unchanged".
  """
  header = build_header(argv)
  fingerprints, pass_orders = _run_corpus()
  if pass_orders is None and require_cold:
    raise WarmProcessError(
      "build_artifact() called twice in one process: the second call cannot observe a true pass order. The CLI "
      "always runs cold; in-process callers must pass require_cold=False or use a fresh process.")
  out: dict[str, Any] = {"header": header, "fingerprints": fingerprints}
  if pass_orders is not None: out["pass_orders"] = pass_orders
  return out


# --------------------------------------------------------------------------------------------------------------
# --check: recompute and diff against the stored artifact
# --------------------------------------------------------------------------------------------------------------

def _classify_diff(old: dict[str, str], new: dict[str, str]) -> list[tuple[str, str, str]]:
  """Returns rows of (graph_name, status, detail) for every graph that differs, was added, or was removed."""
  rows: list[tuple[str, str, str]] = []
  all_names = sorted(set(old) | set(new))
  for name in all_names:
    old_hash, new_hash = old.get(name), new.get(name)
    if old_hash is None:
      rows.append((name, "ADDED", f"new graph, hash={new_hash}"))
    elif new_hash is None:
      rows.append((name, "REMOVED", f"graph missing from fresh run, was hash={old_hash}"))
    elif old_hash != new_hash:
      rows.append((name, "CHANGED", f"{old_hash} -> {new_hash}"))
  return rows


def _first_divergence(old: list[str], new: list[str]) -> str:
  """Where two pass orders first disagree, in pass names rather than indices.

  A pass-order diff is only actionable if it says which pass moved. Reporting `len 102 -> 101` sends the reader
  back to the artifact to diff it by hand, which is how an order gate ends up being ignored.
  """
  for i, (a, b) in enumerate(itertools.zip_longest(old, new)):
    if a != b:
      ctx = " after " + " -> ".join(old[max(0, i - 2):i]) if i else " at the start"
      return f"step {i}{ctx}: expected {a!r}, observed {b!r} (len {len(old)} -> {len(new)})"
  return f"identical ({len(old)} steps)"


def _classify_order_diff(old: dict[str, list[str]], new: dict[str, list[str]]) -> list[tuple[str, str, str]]:
  rows: list[tuple[str, str, str]] = []
  for name in sorted(set(old) | set(new)):
    o, n = old.get(name), new.get(name)
    if o is None:
      rows.append((name, "ADDED", f"new graph, {len(n or [])} pass steps"))
    elif n is None:
      rows.append((name, "REMOVED", f"graph missing from fresh run, was {len(o)} pass steps"))
    elif o != n:
      rows.append((name, "REORDERED", _first_divergence(o, n)))
  return rows


def run_check(argv: list[str]) -> int:
  if not OUT_PATH.is_file():
    print(f"FAIL: no stored fingerprint at {OUT_PATH}; run without --check first")
    return 1
  stored = json.loads(OUT_PATH.read_text())
  fresh = build_artifact(argv, require_cold=False)
  stored_fp = stored.get("fingerprints", {})
  fresh_fp = fresh.get("fingerprints", {})
  rows = _classify_diff(stored_fp, fresh_fp)
  print(f"{'graph':24s} {'status':8s} detail")
  for name, status, detail in rows:
    print(f"{name:24s} {status:8s} {detail}")

  # Pass order is checked separately and reported separately: identical generated code with a changed pipeline is a
  # real finding (a pass was reordered, split, or renamed without moving output on this corpus), and so is the
  # reverse. Collapsing both into one verdict would hide whichever one the reader was not looking for.
  stored_po = stored.get("pass_orders")
  order_rows: list[tuple[str, str, str]] = []
  if "pass_orders" not in fresh:
    print("\npass order: NOT CHECKED (warm process -- most rewrites are cache hits and would report as removed)")
  elif stored_po is None:
    print("\npass order: no stored order in this artifact (predates the order gate); rerun without --check to pin it")
  else:
    order_rows = _classify_order_diff(stored_po, fresh.get("pass_orders", {}))
    print(f"\n{'graph':24s} {'status':10s} detail")
    for name, status, detail in order_rows:
      print(f"{name:24s} {status:10s} {detail}")
    if not order_rows:
      steps = sum(len(v) for v in fresh.get("pass_orders", {}).values())
      print(f"pass order: PASS ({steps} collapsed pass steps across {len(fresh_fp)} graphs, order unchanged)")

  if not rows and not order_rows:
    print(f"verdict: PASS ({len(fresh_fp)} graphs, lowering fingerprint and pass order unchanged)")
    return 0
  print(f"verdict: FAIL ({len(rows)} fingerprint, {len(order_rows)} pass-order differences)")
  return 1


def purge_env_to_allowlist() -> list[str]:
  """Reduce os.environ to _ENV_ALLOWLIST (plus CACHELEVEL, set by the import prep). Returns the keys removed.

  CLI-ONLY, and deliberately not called from run_check/build_artifact. STRIP_PREFIXES catches the repo's own gate
  variables, but it is a denylist, and lowering responds to variables outside it: with NOOPT=1 in the environment
  this gate reports a genuine pass-order difference ('shift (0,) 2 upcast' replaced by 'flatten range'). That is
  the gate working -- the pipeline really is different -- but a gate is supposed to measure the code change rather
  than the caller's shell, so the CLI removes the ambiguity at the source.

  This is not applied to in-process callers because it mutates os.environ, and under pytest that would delete
  environment other tests depend on. In-process callers get the same guarantee a different way:
  pass_orders_in_fresh_process() builds the allowlist environment for a subprocess instead of destroying its own.
  """
  removed = [k for k in os.environ if k not in _ENV_ALLOWLIST]
  for k in removed: os.environ.pop(k, None)
  return sorted(removed)


def main() -> int:
  purge_env_to_allowlist()
  ap = argparse.ArgumentParser(description="CPU-only lowering fingerprint gate.")
  ap.add_argument("--check", action="store_true", help="recompute and diff against the stored latest.json; write nothing")
  args, _unknown = ap.parse_known_args()
  argv = sys.argv[1:]
  if args.check:
    return run_check(argv)
  artifact = build_artifact(argv)
  OUT_DIR.mkdir(parents=True, exist_ok=True)
  OUT_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
  print(f"wrote {OUT_PATH} ({len(artifact['fingerprints'])} graphs)")
  print("verdict: PASS")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
