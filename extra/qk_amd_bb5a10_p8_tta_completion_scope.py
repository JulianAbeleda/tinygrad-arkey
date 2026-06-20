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
  tta = read_json("bench/amd-broad-backend-roadmap/bb5a10_p8_tta_scope_result.json", {})
  p7d = read_json("bench/amd-broad-backend-roadmap/bb5a10_p7d_authority_correctness_result.json", {})
  contract = tta.get("authority_contract") or {"M": 512, "N": 12288, "K": 4096}
  m, n, k = int(contract["M"]), int(contract["N"]), int(contract["K"])
  formulas = {
    "tile_dims": {"TTA1": [16, 16, k], "TTA3": [128, 128, k]},
    "grid": {
      "TTA1": {"gidx0_cols": n // 16, "gidx1_rows": m // 16, "global_size": [n // 16, m // 16, 1], "local_size": [32, 1, 1]},
      "TTA3": {"gidx0_cols": n // 128, "gidx1_rows": m // 128, "global_size": [n // 128, m // 128, 1], "local_size": [128, 1, 1]},
    },
    "row_col_bases": {
      "TTA1": {"row_base": "gidx1 * 16", "col_base": "gidx0 * 16"},
      "TTA3": {"row_base": "gidx1 * 128 + wave_m * 64 + mi * 16", "col_base": "gidx0 * 128 + wave_n * 64 + ni * 16"},
    },
    "global_offsets_bytes": {
      "A": "(row_base + lane_or_row_fragment) * K * 2 + k_iter * 32",
      "Bt": "(col_base + lane_or_col_fragment) * K * 2 + k_iter * 32",
      "C": "((row_base + output_row) * N + (col_base + output_col)) * 2",
    },
    "k_loop": {"depth": k // 16, "step_bytes": 32, "wmma_k": 16},
  }
  artifacts = [
    {
      "id": "TTA1",
      "script": "extra/qk_amd_bb5a10_p8_tta1_full_grid_correctness.py",
      "result": "bench/amd-broad-backend-roadmap/bb5a10_p8_tta1_full_grid_correctness_result.json",
      "doc": "docs/amd-broad-backend-bb5a10-p8-tta1-full-grid-correctness-result-20260620.md",
      "must_prove": [
        "gidx0/gidx1 drive C tile placement over grid=(768,32,1)",
        "A and Bt bases include row_base/col_base while K-loop still increments by 32 bytes",
        "deterministic sampled tiles match fp32 reference with relative RMSE <= 1e-3",
      ],
      "allowed_to_time": False,
      "if_blocked": "debug address formulas against P7d tile (0,0); do not change LDS layout or start macro-tile work",
    },
    {
      "id": "TTA2",
      "script": "extra/qk_amd_bb5a10_p8_tta2_authority_sample_correctness.py",
      "result": "bench/amd-broad-backend-roadmap/bb5a10_p8_tta2_authority_sample_correctness_result.json",
      "doc": "docs/amd-broad-backend-bb5a10-p8-tta2-authority-sample-correctness-result-20260620.md",
      "must_prove": [
        "full M=512,N=12288,K=4096 launch completes",
        "sampled first/middle/last row and column tiles match reference",
        "no hidden narrow-grid shortcut remains in the kernel wrapper",
      ],
      "allowed_to_time": False,
      "if_blocked": "keep P8 blocked and reduce only the sampled verifier cost, not the launch shape",
    },
    {
      "id": "TTA3",
      "script": "extra/qk_amd_bb5a10_p8_tta3_macro_candidate.py",
      "result": "bench/amd-broad-backend-roadmap/bb5a10_p8_tta3_macro_candidate_result.json",
      "doc": "docs/amd-broad-backend-bb5a10-p8-tta3-macro-candidate-result-20260620.md",
      "must_prove": [
        "selected-compatible 128x128 macro tile maps grid=(96,4,1)",
        "candidate uses LDS staging and WMMA accumulation beyond the TTA1 toy tile",
        "resource summary reports LDS/VGPR/SGPR/scratch/private and rejects spill candidates",
      ],
      "allowed_to_time": "only smoke timing, not P8 acceptance timing",
      "if_blocked": "classify as launch mapping, accumulator/VGPR pressure, LDS layout, or epilogue; do not reopen q8",
    },
    {
      "id": "TTA4",
      "script": "extra/qk_amd_bb5a10_p8_performance.py",
      "result": "bench/amd-broad-backend-roadmap/bb5a10_p8_performance_result.json",
      "doc": "docs/amd-broad-backend-bb5a10-p8-performance-result-20260620.md",
      "must_prove": [
        "same candidate that passed TTA2/TTA3 is timed",
        "best/median TFLOPS are reported with CNT and command",
        ">=60 TFLOPS and scratch/private 0 for P8 pass",
      ],
      "allowed_to_time": True,
      "if_blocked": "record measured bottleneck and keep P9/q8 transfer blocked",
    },
    {
      "id": "P9",
      "script": "extra/qk_amd_bb5a10_p9_q8_reopen.py",
      "result": "bench/amd-broad-backend-roadmap/bb5a10_p9_q8_reopen_result.json",
      "doc": "docs/amd-broad-backend-bb5a10-p9-q8-reopen-result-20260620.md",
      "must_prove": [
        "P8 passed before q8 transfer is considered",
        "q8 continuation gate <=75us and strong pass <=60us are preserved",
        "fallback/default behavior remains unchanged",
      ],
      "allowed_to_time": True,
      "if_blocked": "do not start q8 transfer; finish or close P8 first",
    },
  ]
  checklist = [
    {"check": "TTA1 correctness bridge cannot satisfy P8 performance", "required": True},
    {"check": "TTA2 full-launch correctness must precede TTA3/P8 timing", "required": True},
    {"check": "TTA3 resource policy must reject scratch/private spill before P8", "required": True},
    {"check": "P8 must time the exact candidate that passed TTA2/TTA3", "required": True},
    {"check": "P9 remains blocked until P8 passes", "required": True},
  ]
  gate = {
    "input_tta_scope_pass": tta.get("verdict") == "PASS_BB5A10_P8_TTA_SCOPE_READY" and bool(tta.get("gate_pass")),
    "input_p7d_pass": p7d.get("verdict") == "PASS_BB5A10_P7D_AUTHORITY_SUBSET_CORRECTNESS" and bool(p7d.get("gate_pass")),
    "all_artifacts_named": all(a["script"] and a["result"] and a["doc"] for a in artifacts),
    "all_blocked_continuations_present": all(bool(a["if_blocked"]) for a in artifacts),
    "address_formulas_present": all(bool(v) for v in formulas["global_offsets_bytes"].values()),
    "p8_before_p9": [a["id"] for a in artifacts][-2:] == ["TTA4", "P9"],
    "q8_transfer_blocked_until_p8": artifacts[-1]["must_prove"][0].startswith("P8 passed"),
    "all_required_checks_present": all(row["required"] for row in checklist),
  }
  gate_pass = all(gate.values())
  result = {
    "date": "2026-06-20",
    "phase": "BB-5a.10_P8_TTA_completion_scope",
    "schema": "amd_bb5a10_p8_tta_completion_scope_v1",
    "verdict": "PASS_BB5A10_P8_TTA_COMPLETION_SCOPE_READY" if gate_pass else "BLOCKED_BB5A10_P8_TTA_COMPLETION_SCOPE_INPUTS",
    "gate_pass": gate_pass,
    "default_behavior_changed": False,
    "performance_claim": False,
    "authority_contract": contract,
    "formulas": formulas,
    "artifacts": artifacts,
    "checklist": checklist,
    "gate": gate,
    "decision": "TTA is scoped through completion. Implement TTA1 first; P8/P9 remain blocked until their prerequisites pass." if gate_pass else
                "TTA completion scope blocked; missing input scope, formulas, artifacts, or blocked continuations.",
    "next_action": "Implement TTA1 full-grid correctness bridge.",
    "input_artifacts": [
      "bench/amd-broad-backend-roadmap/bb5a10_p8_tta_scope_result.json",
      "bench/amd-broad-backend-roadmap/bb5a10_p7d_authority_correctness_result.json",
    ],
  }
  write_json("bb5a10_p8_tta_completion_scope_result.json", result)
  print(json.dumps({
    "out": "bench/amd-broad-backend-roadmap/bb5a10_p8_tta_completion_scope_result.json",
    "verdict": result["verdict"],
    "gate_pass": gate_pass,
    "next": result["next_action"],
  }, indent=2))
  return 0 if gate_pass else 1


if __name__ == "__main__":
  raise SystemExit(main())
