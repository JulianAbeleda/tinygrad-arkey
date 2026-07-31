#!/usr/bin/env python3
"""MB2: sweep bc=2 geometries through the committed lane
(`extra/llm_research/prefill/metal_precontract_lane.py`) after parameterizing the buffer2
accumulator-contract binary-axis literals in `tinygrad/codegen/opt/postrange.py` (were hardcoded
to RDNA3's three-binary-axes/8-elements-per-lane; now derived from `tc.elements_per_thread[2]`
via `binary_axis_count`/`fold_binary_axes`, exactly as PG0/PG1a/PG1 did for the sibling A/B
operand folds).

This is NOT a new compile/admit/execute driver -- every config below is dispatched through
`run_precontract_probe`, the lane's one entry point. This script only enumerates configs and
prints/saves the results, exactly the way `scratchpad/m1e_metal_precontract_sweep.py` and
`scratchpad/m1g_k_tile_iteration_sweep.py` do.

Fixed: Q4_K, ffn_gate_up, (512, 12288, 4096) -- the same route/shape M1b/M1c/M1d/M1e/M1f/M1g
measured BUG A's failure signature at. Only geometry varies, restricted to the five bc=2 tuples
M1a's population proved legal under `bc*(tm+tn)*80 <= 32768` (all `tk=32`, `(wm,wn)` picked from
M1a's own recorded splits, docs/task_workflow/output/m1a-readiness-and-geometry-population-result-
20260730.md, "Full population" table, the five `bc=2` rows):

  (64,32,32,8,1,2), (64,64,32,8,1,2), (64,128,32,8,1,2), (128,32,32,16,1,2), (128,64,32,16,1,2)

Single GPU lane discipline: run sequentially, one config's isolated child fully exits before the
next config's child spawns (no concurrent GPU work); another agent is running compile-only work
concurrently in this same tree, so this script is the only GPU consumer.
"""
from __future__ import annotations
import sys, json, time
sys.path.insert(0, "/Users/julianabeleda/env/tinygrad-arkey-exp")

from extra.llm_research.prefill.metal_precontract_lane import ProbeConfig, admit_probe_config, run_precontract_probe

QUANT, ROLE = "Q4_K", "ffn_gate_up"
SHAPE = (512, 12288, 4096)
DEVICE = "METAL"
ROUNDS = 3

# (tm, tn, tk, wm, wn, bc) -- M1a's five legal bc=2 tuples, tk=32 throughout.
BC2_GEOMETRIES = [
  (64, 32, 32, 8, 1, 2),
  (64, 64, 32, 8, 1, 2),
  (64, 128, 32, 8, 1, 2),
  (128, 32, 32, 16, 1, 2),
  (128, 64, 32, 16, 1, 2),
]


def main() -> None:
  results = []
  for geometry in BC2_GEOMETRIES:
    config = ProbeConfig(QUANT, ROLE, SHAPE, geometry, device=DEVICE, rounds=ROUNDS)
    print(f"\n=== geometry={geometry} shape={config.shape} device={config.device} rounds={config.rounds} ===",
          flush=True)

    # Admission check first, pure Python, no GPU -- report rejections with their reason, exactly
    # as the task requires, before ever spawning a GPU child for this config.
    try:
      admit_probe_config(config)
      admitted, admit_reason = True, None
      print(f"admission: OK", flush=True)
    except Exception as exc:
      admitted, admit_reason = False, f"{type(exc).__name__}: {exc}"
      print(f"admission: REJECTED -- {admit_reason}", flush=True)

    if not admitted:
      results.append({"geometry": geometry, "admitted": False, "admit_reason": admit_reason, "elapsed_seconds": 0.0})
      continue

    t0 = time.monotonic()
    result = run_precontract_probe(config)
    elapsed = time.monotonic() - t0
    row = {"geometry": geometry, "admitted": True, "admit_reason": None, "elapsed_seconds": elapsed, **result.to_json()}
    results.append(row)
    print(json.dumps(row, sort_keys=True, default=str), flush=True)
    print(f"--- elapsed {elapsed:.1f}s ---\n", flush=True)

  out_path = "/tmp/mb2_bc2_geometry_sweep_results.json"
  with open(out_path, "w") as f:
    json.dump(results, f, indent=2, sort_keys=True, default=str)
  print("wrote", out_path)


if __name__ == "__main__":
  main()
