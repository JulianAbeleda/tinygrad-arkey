"""Callback-only paired direct-L2/LDS decision runner.

The callbacks are supplied by an external, explicitly gated harness.  This
module only joins captured evidence and calls the pure decision authority; it
never imports a device runtime or dispatches a kernel.
"""
from __future__ import annotations

import random
from typing import Any, Callable

from extra.qk.prefill.pure_register_direct_l2_decision import decide

SCHEMA = "paired-direct-l2-benchmark-runner.v1"
STORAGES = ("direct_l2", "lds")


def run_paired_direct_l2_benchmark(*, role: str, shape: dict[str, int],
    canonical_identity: str, environment: dict[str, Any], pair_key: str | None = None,
    artifact: Callable[[str], dict[str, Any]],
    route_binding: Callable[[str], dict[str, Any]],
    correctness: Callable[[str], dict[str, Any]],
    benchmark: Callable[[str, str, int], dict[str, Any]],
    rounds: int = 12, warmups: int = 2, seed: int = 0,
    thresholds: dict[str, float] | None = None) -> dict[str, Any]:
  """Collect a paired report from callbacks and return promote/retain/blocked.

  ``benchmark(storage, phase, round_index)`` is called in randomized order,
  where phase is ``warmup`` or ``timed``.  Its timed result must contain
  ``samples_ms`` (the callback may return one sample per call or accumulated
  samples) and ``counters``.  Callbacks are intentionally opaque: callers own
  gating, compilation, dispatch, synchronization, and counter collection.
  """
  protocol: dict[str, Any] = {"name": "paired-random-interleave-v1", "seed": seed,
    "warmups": warmups, "rounds": rounds, "dispatch": "external-callback-only",
    "randomized_interleaved_order": []}
  result: dict[str, Any] = {"schema": SCHEMA, "status": "blocked", "decision": "blocked",
    "protocol": protocol}
  if rounds <= 0 or warmups < 0:
    result["blockers"] = ["rounds must be positive and warmups non-negative"]
    return result
  rng = random.Random(seed)
  rows: dict[str, dict[str, Any]] = {}
  for storage in STORAGES:
    a, b, c = artifact(storage), route_binding(storage), correctness(storage)
    rows[storage] = {"role": role, "shape": shape, "canonical_identity": canonical_identity,
                     "pair_key": pair_key or canonical_identity,
                     "environment": environment, "storage": storage, **a, **b, **c,
                     "samples_ms": [], "counters": {}}
  for phase, count in (("warmup", warmups), ("timed", rounds)):
    for index in range(count):
      order = list(STORAGES); rng.shuffle(order)
      protocol["randomized_interleaved_order"].append({"phase": phase, "round": index, "order": order})
      for selected in order:
        observed = benchmark(selected, phase, index)
        if phase == "timed":
          rows[selected]["samples_ms"].extend(observed.get("samples_ms", []))
          if observed.get("counters") is not None: rows[selected]["counters"] = observed["counters"]
  pair = {"direct_l2": rows["direct_l2"], "lds": rows["lds"]}
  report = decide(pair, thresholds=thresholds)
  binding_blockers = [f"{storage} route binding prerequisite is missing or failed"
                      for storage in STORAGES
                      if rows[storage].get("route_binding", {}).get("status") != "pass"]
  if binding_blockers:
    report = {**report, "status": "blocked", "decision": "blocked",
              "blockers": list(report.get("blockers", [])) + binding_blockers}
  result.update(report)
  result["rows"] = rows
  return result
