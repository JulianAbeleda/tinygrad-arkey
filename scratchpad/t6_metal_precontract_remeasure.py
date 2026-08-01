#!/usr/bin/env python3
"""T6 -- re-measure M1e's sweep through the (now device-aware) admission fix.

Not a new compile/admit/execute driver: every config below is dispatched through
`run_precontract_probe`, the same committed lane M1e built. This script only enumerates configs
(re-running M1e's own wave_count group verbatim, plus every one of M1a's 23 Metal-legal tuples that
now admits under the device-aware tc resolution -- t6_admission_census.py already established which
5 of 23 admit) and prints/saves the results.

Single GPU lane discipline: run sequentially, one config's isolated child fully exits before the next
config's child spawns (no concurrent GPU work) -- unchanged from M1e.
"""
from __future__ import annotations
import sys, json, time
sys.path.insert(0, "/Users/julianabeleda/env/tinygrad-arkey-exp")

from extra.llm_research.prefill.precontract_probe_lane import ProbeConfig, run_precontract_probe

QUANT, ROLE = "Q4_K", "ffn_gate_up"
SHAPE = (512, 12288, 4096)
DEVICE = "METAL"

# --- M1e's wave_count group, verbatim (must still admit and reproduce the same failure signature,
# not a new numeric result -- this is the non-regression half of the re-measurement). --------------
WAVE_COUNT_CONFIGS = [
  ("wave_count", ProbeConfig(QUANT, ROLE, SHAPE, (256, 64, 32, wm, 1, 1), device=DEVICE))
  for wm in (8, 4, 2)
]

# --- The 5 of M1a's 23 tuples that t6_admission_census.py found now admit under the device-aware fix
# (all 23 were rejected under the old AMD-tc-hardcoded gate; the other 18 now fail a different,
# pre-existing, unrelated gate -- "operand vectors must divide evenly across cooperative threads" --
# not tc-subtile divisibility, so they are not re-attempted here). ----------------------------------
M1A_NOW_ADMITTED_CONFIGS = [
  ("m1a_tuple", ProbeConfig(QUANT, ROLE, SHAPE, (64, 64, 32, 8, 1, 1), device=DEVICE)),
  ("m1a_tuple", ProbeConfig(QUANT, ROLE, SHAPE, (64, 128, 32, 8, 1, 1), device=DEVICE)),
  ("m1a_tuple", ProbeConfig(QUANT, ROLE, SHAPE, (128, 128, 32, 16, 1, 1), device=DEVICE)),
  ("m1a_tuple", ProbeConfig(QUANT, ROLE, SHAPE, (64, 64, 32, 8, 1, 2), device=DEVICE)),
  ("m1a_tuple", ProbeConfig(QUANT, ROLE, SHAPE, (64, 128, 32, 8, 1, 2), device=DEVICE)),
]

ALL_CONFIGS = WAVE_COUNT_CONFIGS + M1A_NOW_ADMITTED_CONFIGS


def main() -> None:
  results = []
  for group, config in ALL_CONFIGS:
    print(f"=== {group}: shape={config.shape} geometry={config.geometry} device={config.device} ===", flush=True)
    t0 = time.monotonic()
    result = run_precontract_probe(config)
    elapsed = time.monotonic() - t0
    row = {"group": group, "elapsed_seconds": elapsed, **result.to_json()}
    results.append(row)
    print(json.dumps(row, sort_keys=True, default=str), flush=True)
    print(f"--- elapsed {elapsed:.1f}s ---\n", flush=True)

  out_path = "/tmp/t6_metal_precontract_remeasure_results.json"
  with open(out_path, "w") as f:
    json.dump(results, f, indent=2, sort_keys=True, default=str)
  print("wrote", out_path)


if __name__ == "__main__":
  main()
