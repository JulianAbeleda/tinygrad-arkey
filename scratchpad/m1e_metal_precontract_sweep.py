#!/usr/bin/env python3
"""M1e Part 2: sweep configurations through the committed lane
(`extra/llm_research/prefill/precontract_probe_lane.py`) to find the correct/incorrect boundary
around the M1b/M1c/M1d Metal numeric failure.

This is NOT a new compile/admit/execute driver -- every config below is dispatched through
`run_precontract_probe`, the lane's one entry point. This script only enumerates configs and
prints/saves the results.

Ordered by discriminating power, per the task brief:
  1. Wave count: wm in {8,4,2,1} at fixed (tm,tn,tk)=(256,64,32), bc=1, same shape M1c used.
  2. Scale: a substantially smaller shape at the same geometry family (256,64,32,8,1,1).
  3. Tile shape: three other tuples from M1a's 20-identity Metal-legal geometry population
     (docs/task_workflow/output/m1a-readiness-and-geometry-population-result-20260730.md),
     same shape as M1c (Q4_K, ffn_gate_up, (512,12288,4096)).

Single GPU lane discipline: run sequentially, one config's isolated child fully exits before the
next config's child spawns (no concurrent GPU work).
"""
from __future__ import annotations
import sys, json, time
sys.path.insert(0, "/Users/julianabeleda/env/tinygrad-arkey-exp")

from extra.llm_research.prefill.precontract_probe_lane import ProbeConfig, run_precontract_probe

QUANT, ROLE = "Q4_K", "ffn_gate_up"
M1C_SHAPE = (512, 12288, 4096)
M1C_GEOMETRY = (256, 64, 32, 8, 1, 1)
DEVICE = "METAL"

# --- Part 2.1: wave count, wm in {8,4,2,1}, (tm,tn,tk)=(256,64,32), bc=1 ------------------------
WAVE_COUNT_CONFIGS = [
  ("wave_count", ProbeConfig(QUANT, ROLE, M1C_SHAPE, (256, 64, 32, wm, 1, 1), device=DEVICE))
  for wm in (8, 4, 2, 1)
]

# --- Part 2.2: scale -- substantially smaller shape, same geometry family ----------------------
# (256,64,32,8,1,1) family: tm=256 forces m>=256; tn=64 divides n; tk=32 divides k, and the
# Q4_K block granularity (256 K-elements/block) additionally requires k a multiple of 256 for the
# fixture's per-row block indexing to stay correct (extra/llm_research/prefill/
# packed_wmma_correctness_canary.py:_decode_selected_q4's `blocks = n*(k//256) + k_position//256`
# assumes k//256 is exact). (256,256,256) is the smallest shape satisfying all three: elements
# 256*256=65,536 output vs the M1c shape's 6,291,456 (~1% of the size), k=256 vs 4096 (16x smaller).
SCALE_SHAPE = (256, 256, 256)
SCALE_CONFIGS = [
  ("scale", ProbeConfig(QUANT, ROLE, SCALE_SHAPE, M1C_GEOMETRY, device=DEVICE)),
]

# --- Part 2.3: tile shape, 3 tuples from M1a's 20-identity population, same M1c shape -----------
# Rows verbatim from docs/task_workflow/output/m1a-readiness-and-geometry-population-result-20260730.md:
#   (64,32,8,1,1)   -- smallest legal tile in the population
#   (128,128,16,1,1) -- a mid-size square tile
#   (64,128,8,1,2)   -- smaller tile, bc=2 (double-buffered), distinct from the wave-count sweep
TILE_SHAPE_CONFIGS = [
  ("tile_shape", ProbeConfig(QUANT, ROLE, M1C_SHAPE, (64, 32, 32, 8, 1, 1), device=DEVICE)),
  ("tile_shape", ProbeConfig(QUANT, ROLE, M1C_SHAPE, (128, 128, 32, 16, 1, 1), device=DEVICE)),
  ("tile_shape", ProbeConfig(QUANT, ROLE, M1C_SHAPE, (64, 128, 32, 8, 1, 2), device=DEVICE)),
]

ALL_CONFIGS = WAVE_COUNT_CONFIGS + SCALE_CONFIGS + TILE_SHAPE_CONFIGS


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

  out_path = "/tmp/m1e_metal_precontract_sweep_results.json"
  with open(out_path, "w") as f:
    json.dump(results, f, indent=2, sort_keys=True, default=str)
  print("wrote", out_path)


if __name__ == "__main__":
  main()
