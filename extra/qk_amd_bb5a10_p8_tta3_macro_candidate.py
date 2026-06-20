#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib, sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extra.gemm.rdna3_wmma_matmul import build_gemm_lds2

OUT = ROOT / "bench/amd-broad-backend-roadmap"
M, N, K = 512, 12288, 4096
WAVES_M, WAVES_N, WM, WN, BK, PAD, DBUF = 2, 2, 4, 4, 16, 0, 0


def read_json(rel: str, default: Any = None) -> Any:
  path = ROOT / rel
  if not path.exists(): return default
  return json.loads(path.read_text())


def write_json(name: str, data: Any) -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def macro_candidate() -> dict[str, Any]:
  tta3a = read_json("bench/amd-broad-backend-roadmap/bb5a10_p8_tta3a_ds64_macro_conversion_result.json", {})
  if tta3a.get("verdict") == "PASS_BB5A10_P8_TTA3A_DS64_MACRO_CONVERSION" and tta3a.get("gate_pass"):
    return dict((tta3a.get("converted_candidate") or {}), source="tta3a_converted_ds64_macro")
  insts = build_gemm_lds2(M, N, K, WAVES_M, WAVES_N, WM, WN, BK, PAD, DBUF)
  names = [getattr(i, "op_name", type(i).__name__) for i in insts]
  threads = WAVES_M * WAVES_N * 32
  bm, bn = WAVES_M * WM * 16, WAVES_N * WN * 16
  lds_bytes = (BK * 2 + PAD) * (bm + bn) * (2 if DBUF else 1)
  return {
    "shape": [M, N, K],
    "macro_tile": [bm, bn, K],
    "grid": [N // bn, M // bm, 1],
    "local_size": [threads, 1, 1],
    "parameters": {"WAVES_M": WAVES_M, "WAVES_N": WAVES_N, "WM": WM, "WN": WN, "BK": BK, "PAD": PAD, "DBUF": DBUF},
    "resource_summary": {
      "lds_bytes": lds_bytes,
      "scratch_bytes": 0,
      "private_segment_fixed_size": 0,
      "instruction_count": len(insts),
      "vgpr_static_upper_bound": 256,
      "spill_risk": "unknown_until_allocator",
    },
    "instruction_counts": {
      "ds_store_b64": names.count("DS_STORE_B64"),
      "ds_store_b128": names.count("DS_STORE_B128"),
      "ds_load_b128": names.count("DS_LOAD_B128"),
      "v_wmma": sum("WMMA" in n for n in names),
      "global_load_b128": names.count("GLOBAL_LOAD_B128"),
      "global_store_b16": names.count("GLOBAL_STORE_B16"),
      "s_barrier": names.count("S_BARRIER"),
      "s_cbranch_scc1": names.count("S_CBRANCH_SCC1"),
    },
    "source": "original_build_gemm_lds2",
  }


def main() -> int:
  tta2 = read_json("bench/amd-broad-backend-roadmap/bb5a10_p8_tta2_authority_sample_correctness_result.json", {})
  try:
    candidate = macro_candidate()
  except Exception as e:
    candidate = {"error": repr(e)}
  counts = candidate.get("instruction_counts") or {}
  resources = candidate.get("resource_summary") or {}
  gate = {
    "input_tta2_pass": tta2.get("verdict") == "PASS_BB5A10_P8_TTA2_AUTHORITY_SAMPLE_CORRECTNESS" and bool(tta2.get("gate_pass")),
    "macro_tile_128x128": candidate.get("macro_tile") == [128, 128, K],
    "authority_grid_96x4": candidate.get("grid") == [96, 4, 1],
    "local_size_128": candidate.get("local_size") == [128, 1, 1],
    "uses_wmma": (counts.get("v_wmma") or 0) > 0,
    "uses_ds_load_b128": (counts.get("ds_load_b128") or 0) > 0,
    "uses_selected_compatible_ds_store_b64": (counts.get("ds_store_b64") or 0) > 0,
    "does_not_use_unselected_ds_store_b128": (counts.get("ds_store_b128") or 0) == 0,
    "scratch_private_zero": resources.get("scratch_bytes") == 0 and resources.get("private_segment_fixed_size") == 0,
    "resource_metadata_present": bool(resources),
  }
  gate_pass = all(gate.values())
  blockers = []
  if not gate["uses_selected_compatible_ds_store_b64"]:
    blockers.append("macro helper uses no selected-compatible ds_store_b64 stores")
  if not gate["does_not_use_unselected_ds_store_b128"]:
    blockers.append("macro helper still uses ds_store_b128; convert cooperative LDS stores to ds_store_b64 before P8 timing")
  result = {
    "date": "2026-06-20",
    "phase": "BB-5a.10_P8_TTA3_macro_candidate",
    "schema": "amd_bb5a10_p8_tta3_macro_candidate_v1",
    "verdict": "PASS_BB5A10_P8_TTA3_MACRO_CANDIDATE" if gate_pass else "BLOCKED_BB5A10_P8_TTA3_SELECTED_COMPATIBLE_MACRO_CANDIDATE",
    "gate_pass": gate_pass,
    "default_behavior_changed": False,
    "performance_claim": False,
    "candidate": candidate,
    "blockers": blockers,
    "gate": gate,
    "decision": "TTA3 passes: selected-compatible 128x128 macro candidate is ready for P8 timing." if gate_pass else
                "TTA3 blocked: the existing 128x128 macro helper has the right launch shape but not the selected-compatible ds_store_b64 LDS store contract.",
    "next_action": "Run P8 timing gate." if gate_pass else
                   "Implement TTA3a: convert the 128x128 macro candidate cooperative LDS stores from ds_store_b128 to selected-compatible ds_store_b64, then rerun TTA3.",
    "input_artifacts": ["bench/amd-broad-backend-roadmap/bb5a10_p8_tta2_authority_sample_correctness_result.json"],
  }
  write_json("bb5a10_p8_tta3_macro_candidate_result.json", result)
  print(json.dumps({
    "out": "bench/amd-broad-backend-roadmap/bb5a10_p8_tta3_macro_candidate_result.json",
    "verdict": result["verdict"],
    "gate_pass": gate_pass,
    "blockers": blockers,
    "next": result["next_action"],
  }, indent=2))
  return 0 if gate_pass else 1


if __name__ == "__main__":
  raise SystemExit(main())
