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
  UOp.unique_num = itertools.count(0)


def compute_fingerprints() -> dict[str, str]:
  Tensor, Device = _prepare_env_and_import_tinygrad()
  _cpu_renderer(Device)  # fail loudly if CPU is somehow unavailable; never silently pick another device
  _reset_uop_unique_counter()
  graphs = _build_graphs(Tensor)
  out: dict[str, str] = {}
  for name, build in graphs.items():
    Tensor.manual_seed(1337)
    out[name] = hashlib.sha256(build().schedule_linear().key).hexdigest()
  return out


def build_header(argv: list[str]) -> dict[str, Any]:
  return {
    "schema": SCHEMA,
    "device": "CPU",
    "python_version": sys.version,
    "command": "python3 " + " ".join(["extra/audit/lowering_fingerprint.py", *argv]),
  }


def build_artifact(argv: list[str]) -> dict[str, Any]:
  header = build_header(argv)
  fingerprints = compute_fingerprints()
  return {"header": header, "fingerprints": fingerprints}


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


def run_check(argv: list[str]) -> int:
  if not OUT_PATH.is_file():
    print(f"FAIL: no stored fingerprint at {OUT_PATH}; run without --check first")
    return 1
  stored = json.loads(OUT_PATH.read_text())
  fresh = build_artifact(argv)
  stored_fp = stored.get("fingerprints", {})
  fresh_fp = fresh.get("fingerprints", {})
  rows = _classify_diff(stored_fp, fresh_fp)
  print(f"{'graph':24s} {'status':8s} detail")
  for name, status, detail in rows:
    print(f"{name:24s} {status:8s} {detail}")
  if not rows:
    print(f"verdict: PASS ({len(fresh_fp)} graphs, lowering fingerprint unchanged)")
    return 0
  print(f"verdict: FAIL ({len(rows)}/{len(set(stored_fp) | set(fresh_fp))} graphs differ)")
  return 1


def main() -> int:
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
