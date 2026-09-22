#!/usr/bin/env python3
"""M1g: sweep K (== K-tile loop-iteration count) through the committed lane
(`extra/llm_research/prefill/precontract_probe_lane.py`) to test the loop-carried
write-after-read hazard theory from `build_precontract_lds_stage`'s single-buffer
(`bc=1`, `slot_base=0` every iteration) LDS reuse.

This is NOT a new compile/admit/execute driver -- every config below is dispatched through
`run_precontract_probe`, the lane's one entry point. This script only enumerates configs and
prints/saves the results, exactly the way `scratchpad/m1e_metal_precontract_sweep.py` does.

Fixed: geometry (256,64,32,8,1,1), Q4_K, ffn_gate_up, m=512, n=12288 -- identical to every prior
M1b/M1c/M1d/M1e/M1f measurement at this route. Only k (hence iterations = k/tk = k/32) varies.

True legal minimum k established beforehand, pure-Python admission only (no GPU, no Device[...]):
  k=32,64,128,192,224 all REJECTED -- "k must be Q4_K block aligned (256), got <k>"
  k=256 ADMITTED
So the floor is k=256 (8 iterations), not k=32 (1 iteration) -- Q4_K's 256-wide superblock is
enforced by admission itself, not merely by the correctness-canary fixture's own indexing.

Single GPU lane discipline: run sequentially, one config's isolated child fully exits before the
next config's child spawns (no concurrent GPU work).
"""
from __future__ import annotations
import sys, json, time
sys.path.insert(0, "/Users/julianabeleda/env/tinygrad-arkey-exp")

from extra.llm_research.prefill.precontract_probe_lane import ProbeConfig, admit_probe_config, run_precontract_probe

QUANT, ROLE = "Q4_K", "ffn_gate_up"
M, N = 512, 12288
GEOMETRY = (256, 64, 32, 8, 1, 1)
TK = GEOMETRY[2]
DEVICE = "METAL"
ROUNDS = 3

# Below-floor probes, recorded for the report but never touching the GPU (admit_probe_config is
# pure Python -- no Device[...] -- so these do not require or consume the GPU lane).
BELOW_FLOOR_K = (32, 64, 128, 192, 224)

# The sweep itself: K = 256 (8 iters) .. 4096 (128 iters), the known-bad M1c/M1f reference point.
SWEEP_K = (256, 512, 1024, 2048, 4096)


def check_admission_only(k: int) -> dict:
  cfg = ProbeConfig(QUANT, ROLE, (M, N, k), GEOMETRY, device=DEVICE)
  try:
    admit_probe_config(cfg)
    return {"k": k, "iterations": k // TK, "admitted": True, "reason": None}
  except Exception as exc:
    return {"k": k, "iterations": k // TK, "admitted": False, "reason": f"{type(exc).__name__}: {exc}"}


def main() -> None:
  below_floor_results = [check_admission_only(k) for k in BELOW_FLOOR_K]
  print("=== below-floor admission checks (pure Python, no GPU) ===", flush=True)
  for r in below_floor_results:
    print(json.dumps(r), flush=True)

  results = []
  for k in SWEEP_K:
    iterations = k // TK
    config = ProbeConfig(QUANT, ROLE, (M, N, k), GEOMETRY, device=DEVICE, rounds=ROUNDS)
    print(f"\n=== K={k} iterations={iterations} shape={config.shape} geometry={config.geometry} "
          f"device={config.device} rounds={config.rounds} ===", flush=True)
    t0 = time.monotonic()
    result = run_precontract_probe(config)
    elapsed = time.monotonic() - t0
    row = {"k": k, "iterations": iterations, "elapsed_seconds": elapsed, **result.to_json()}
    results.append(row)
    print(json.dumps(row, sort_keys=True, default=str), flush=True)
    print(f"--- elapsed {elapsed:.1f}s ---\n", flush=True)

  out_path = "/tmp/m1g_k_tile_iteration_sweep_results.json"
  with open(out_path, "w") as f:
    json.dump({"below_floor": below_floor_results, "sweep": results}, f, indent=2, sort_keys=True, default=str)
  print("wrote", out_path)


if __name__ == "__main__":
  main()
