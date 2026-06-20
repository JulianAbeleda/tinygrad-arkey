#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-broad-backend-roadmap"


def read_json(rel: str, default: Any = None) -> Any:
  path = ROOT / rel
  if not path.exists(): return default
  return json.loads(path.read_text())


def write_json(name: str, data: Any) -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main() -> int:
  p7e = read_json("bench/amd-broad-backend-roadmap/bb5a10_p7e_p8_handoff_result.json", {})
  p8 = read_json("bench/amd-broad-backend-roadmap/bb5a10_p8_performance_result.json", {})
  authority = ((p7e.get("handoff") or {}).get("correctness") or {}).get("authority_contract") or {"M": 512, "N": 12288, "K": 4096}
  m, n, k = int(authority["M"]), int(authority["N"]), int(authority["K"])
  tile_m = tile_n = 16
  simple_grid = [n // tile_n, m // tile_m, 1]
  selected_macro = {"macro_m": 128, "macro_n": 128, "grid": [n // 128, m // 128, 1]}
  phases = [
    {
      "id": "TTA0",
      "name": "freeze tile-to-authority contract",
      "minimum_pass": "record full authority M/N/K, row-major A, row-major Bt, fp16 C, and exact-grid divisibility",
      "artifact": "bench/amd-broad-backend-roadmap/bb5a10_p8_tta_scope_result.json",
      "status": "complete",
      "if_blocked": "do not edit kernels; authority contract mismatch means P7e handoff must be corrected first",
    },
    {
      "id": "TTA1",
      "name": "single-wave full-grid correctness bridge",
      "minimum_pass": "extend P7d to gidx0/gidx1: grid=(768,32,1), one 16x16 tile per workgroup, full K=4096, RMSE <=1e-3 on deterministic subset",
      "artifact": "bench/amd-broad-backend-roadmap/bb5a10_p8_tta1_full_grid_correctness_result.json",
      "status": "planned",
      "if_blocked": "debug output base, A row base, Bt column base, and K-loop address increments before performance work",
    },
    {
      "id": "TTA2",
      "name": "authority-shape smoke without full readback cost",
      "minimum_pass": "run full M=512,N=12288,K=4096 launch and verify deterministic sampled output tiles against numpy/reference slices",
      "artifact": "bench/amd-broad-backend-roadmap/bb5a10_p8_tta2_authority_sample_correctness_result.json",
      "status": "planned",
      "if_blocked": "keep P8 blocked; full launch correctness is required before timing",
    },
    {
      "id": "TTA3",
      "name": "selected macro-tile performance candidate",
      "minimum_pass": "promote from 16x16 correctness bridge to selected-compatible 128x128 macro tile with scratch/private 0 and resource metadata",
      "artifact": "bench/amd-broad-backend-roadmap/bb5a10_p8_tta3_macro_candidate_result.json",
      "status": "planned",
      "if_blocked": "classify whether blocker is accumulator VGPR pressure, LDS layout, launch mapping, or output epilogue; do not time toy tile",
    },
    {
      "id": "TTA4",
      "name": "P8 timing gate",
      "minimum_pass": "same candidate that passed TTA2/TTA3 reaches >=60 TFLOPS on authority shape without scratch/private spill",
      "artifact": "bench/amd-broad-backend-roadmap/bb5a10_p8_performance_result.json",
      "status": "blocked_on_TTA1_TTA2_TTA3",
      "if_blocked": "q8 transfer stays blocked; record measured bottleneck and do not reopen P9",
    },
  ]
  missing = [
    {
      "item": "gidx0/gidx1 tile offsets",
      "why": "P7d computes only output tile (0,0); full launch needs column tile and row tile in global addresses",
      "required_for": ["TTA1", "TTA2"],
    },
    {
      "item": "multi-tile output address mapping",
      "why": "C store currently writes a local 16x16 tile with no global row/column base",
      "required_for": ["TTA1", "TTA2"],
    },
    {
      "item": "A/Bt global base formulas",
      "why": "A base must include output row tile, Bt base must include output column tile, while K-loop increments stay +32 bytes",
      "required_for": ["TTA1", "TTA2"],
    },
    {
      "item": "sampled full-authority correctness harness",
      "why": "full C is large but still must be checked without hiding launch bugs",
      "required_for": ["TTA2"],
    },
    {
      "item": "128x128 macro-tile candidate",
      "why": "16x16 one-wave grid can prove mapping but is not a credible >=60 TFLOPS performance candidate",
      "required_for": ["TTA3", "TTA4"],
    },
    {
      "item": "resource and spill proof for timed candidate",
      "why": "P8 acceptance requires no scratch/private spill on the actual timed kernel",
      "required_for": ["TTA3", "TTA4"],
    },
  ]
  gate = {
    "input_p7e_pass": p7e.get("verdict") == "PASS_BB5A10_P7E_P8_HANDOFF_PACKAGE" and bool(p7e.get("gate_pass")),
    "input_p8_blocked_on_launch_mapping": p8.get("verdict") == "BLOCKED_BB5A10_P8_FULL_AUTHORITY_LAUNCH_MAPPING_REQUIRED",
    "authority_shape_divisible_by_16": m % tile_m == 0 and n % tile_n == 0 and k % 16 == 0,
    "simple_grid_declared": simple_grid == [768, 32, 1],
    "selected_macro_grid_declared": selected_macro["grid"] == [96, 4, 1],
    "phases_have_blocked_continuations": all(bool(p["if_blocked"]) for p in phases),
    "performance_separated_from_correctness_bridge": phases[1]["id"] == "TTA1" and phases[3]["id"] == "TTA3",
  }
  gate_pass = all(gate.values())
  result = {
    "date": "2026-06-19",
    "phase": "BB-5a.10_P8_TTA_scope",
    "schema": "amd_bb5a10_p8_tta_scope_v1",
    "verdict": "PASS_BB5A10_P8_TTA_SCOPE_READY" if gate_pass else "BLOCKED_BB5A10_P8_TTA_SCOPE_INPUTS",
    "gate_pass": gate_pass,
    "default_behavior_changed": False,
    "performance_claim": False,
    "definition": {"TTA": "tile-to-authority launch mapping from the proven P7d 16x16x4096 tile to full authority shape"},
    "authority_contract": authority,
    "simple_correctness_bridge": {"tile": [16, 16, 4096], "grid": simple_grid, "workgroups": simple_grid[0] * simple_grid[1]},
    "selected_performance_target": {"tile": [128, 128, 4096], **selected_macro},
    "phases": phases,
    "missing_items": missing,
    "gate": gate,
    "decision": "P8 TTA is scoped. Start TTA1 full-grid correctness bridge; do not time until TTA2/TTA3 pass." if gate_pass else
                "P8 TTA scope blocked; P7e/P8 inputs or authority divisibility are missing.",
    "next_action": "Implement TTA1: gidx0/gidx1 full-grid correctness bridge for the proven P7d K-loop." if gate_pass else "Fix scope inputs before implementation.",
    "input_artifacts": [
      "bench/amd-broad-backend-roadmap/bb5a10_p7e_p8_handoff_result.json",
      "bench/amd-broad-backend-roadmap/bb5a10_p8_performance_result.json",
    ],
  }
  write_json("bb5a10_p8_tta_scope_result.json", result)
  print(json.dumps({
    "out": "bench/amd-broad-backend-roadmap/bb5a10_p8_tta_scope_result.json",
    "verdict": result["verdict"],
    "gate_pass": gate_pass,
    "next": result["next_action"],
  }, indent=2))
  return 0 if gate_pass else 1


if __name__ == "__main__":
  raise SystemExit(main())
