#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib, sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extra.qk_amd_bb5a10_p8_tta1_full_grid_correctness import K, M, N, run_tta1

OUT = ROOT / "bench/amd-broad-backend-roadmap"


def read_json(rel: str, default: Any = None) -> Any:
  path = ROOT / rel
  if not path.exists(): return default
  return json.loads(path.read_text())


def write_json(name: str, data: Any) -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def coverage(samples: list[dict[str, Any]]) -> dict[str, bool]:
  rows = {tuple(s.get("row_col", [None, None]))[0] for s in samples}
  cols = {tuple(s.get("row_col", [None, None]))[1] for s in samples}
  return {
    "first_row": 0 in rows,
    "middle_row": 256 in rows,
    "last_row": 496 in rows,
    "first_col": 0 in cols,
    "middle_col": 6144 in cols,
    "last_col": 12272 in cols,
  }


def main() -> int:
  tta1_prev = read_json("bench/amd-broad-backend-roadmap/bb5a10_p8_tta1_full_grid_correctness_result.json", {})
  try:
    authority = run_tta1()
  except Exception as e:
    authority = {"correct": False, "error": repr(e)}
  samples = authority.get("samples") or []
  cov = coverage(samples)
  gate = {
    "input_tta1_pass": tta1_prev.get("verdict") == "PASS_BB5A10_P8_TTA1_FULL_GRID_CORRECTNESS" and bool(tta1_prev.get("gate_pass")),
    "full_authority_shape": authority.get("authority_shape") == [M, N, K],
    "full_grid_launch": authority.get("grid") == [768, 32, 1],
    "sampled_first_middle_last_rows": cov["first_row"] and cov["middle_row"] and cov["last_row"],
    "sampled_first_middle_last_cols": cov["first_col"] and cov["middle_col"] and cov["last_col"],
    "sample_count_at_least_5": (authority.get("sample_count") or 0) >= 5,
    "max_rmse_le_1e_3": authority.get("max_relative_rmse") is not None and float(authority.get("max_relative_rmse")) <= 0.001,
    "all_samples_correct": bool(authority.get("correct")),
    "no_narrow_grid_shortcut": authority.get("grid") == [N // 16, M // 16, 1],
  }
  gate_pass = all(gate.values())
  result = {
    "date": "2026-06-20",
    "phase": "BB-5a.10_P8_TTA2_authority_sample_correctness",
    "schema": "amd_bb5a10_p8_tta2_authority_sample_correctness_v1",
    "verdict": "PASS_BB5A10_P8_TTA2_AUTHORITY_SAMPLE_CORRECTNESS" if gate_pass else "BLOCKED_BB5A10_P8_TTA2_AUTHORITY_SAMPLE_CORRECTNESS",
    "gate_pass": gate_pass,
    "default_behavior_changed": False,
    "performance_claim": False,
    "authority": authority,
    "coverage": cov,
    "gate": gate,
    "decision": "TTA2 passes: full authority launch sampled correctness passes with no narrow-grid shortcut. Next is TTA3 macro-tile performance candidate." if gate_pass else
                "TTA2 blocked; full launch, sample coverage, or sampled correctness is insufficient.",
    "next_action": "Implement TTA3 selected-compatible 128x128 macro candidate." if gate_pass else "Fix TTA2 before TTA3/P8.",
    "input_artifacts": ["bench/amd-broad-backend-roadmap/bb5a10_p8_tta1_full_grid_correctness_result.json"],
  }
  write_json("bb5a10_p8_tta2_authority_sample_correctness_result.json", result)
  print(json.dumps({
    "out": "bench/amd-broad-backend-roadmap/bb5a10_p8_tta2_authority_sample_correctness_result.json",
    "verdict": result["verdict"],
    "gate_pass": gate_pass,
    "max_relative_rmse": authority.get("max_relative_rmse"),
    "next": result["next_action"],
  }, indent=2))
  return 0 if gate_pass else 1


if __name__ == "__main__":
  raise SystemExit(main())
