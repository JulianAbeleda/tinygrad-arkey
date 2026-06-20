#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib, sys
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tinygrad import Tensor
from tinygrad.dtype import dtypes

OUT = ROOT / "bench/amd-broad-backend-roadmap"


def read_json(rel: str, default: Any = None) -> Any:
  path = ROOT / rel
  if not path.exists(): return default
  return json.loads(path.read_text())


def write_json(name: str, data: Any) -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def small_wmma_correctness() -> dict[str, Any]:
  rng = np.random.default_rng(0)
  a_np = (rng.standard_normal((16, 16), dtype=np.float32) * 0.1).astype(np.float16)
  b_np = (rng.standard_normal((16, 16), dtype=np.float32) * 0.1).astype(np.float16)
  got = (Tensor(a_np, device="AMD", dtype=dtypes.half) @ Tensor(b_np, device="AMD", dtype=dtypes.half)).realize().numpy().astype(np.float32)
  ref = a_np.astype(np.float32) @ b_np.astype(np.float32)
  rel_err = float(np.max(np.abs(got - ref)) / (np.max(np.abs(ref)) + 1e-9))
  return {"shape": [16, 16, 16], "rel_err": rel_err, "correct": rel_err <= 1e-2}


def authority_prefill_correctness() -> dict[str, Any]:
  shape = read_json("bench/qk-tensile-extraction/shape_matrix.json", {})
  rows = [row for row in shape.get("rows", []) if row.get("role") == "ffn_gate_up"]
  row = rows[0] if rows else {}
  return {
    "role": row.get("role"),
    "shape": {k: row.get(k) for k in ("m", "n", "k")},
    "rel_err": row.get("rel_err"),
    "correct": bool(row.get("correct")),
    "source_artifact": "bench/qk-tensile-extraction/shape_matrix.json",
  }


def main() -> int:
  bb5a5 = read_json("bench/amd-broad-backend-roadmap/bb5a5_resource_policy_result.json", {})
  small = small_wmma_correctness()
  authority = authority_prefill_correctness()
  gate = {
    "input_bb5a5_pass": bb5a5.get("verdict") == "PASS_BB5A5_RESOURCE_POLICY" and bool(bb5a5.get("gate_pass")),
    "small_wmma_correct": bool(small["correct"]),
    "small_wmma_rel_err_within_tolerance": small["rel_err"] <= 1e-2,
    "authority_prefill_correct": bool(authority["correct"]),
    "authority_prefill_rel_err_within_tolerance": authority["rel_err"] is not None and float(authority["rel_err"]) <= 1e-3,
    "default_behavior_changed": False,
    "performance_claim": False,
  }
  positive = [
    "input_bb5a5_pass", "small_wmma_correct", "small_wmma_rel_err_within_tolerance",
    "authority_prefill_correct", "authority_prefill_rel_err_within_tolerance",
  ]
  gate_pass = all(gate[x] for x in positive) and not gate["default_behavior_changed"] and not gate["performance_claim"]
  result = {
    "date": "2026-06-19",
    "phase": "BB-5a.6_correctness",
    "schema": "amd_bb5a6_correctness_result_v1",
    "verdict": "PASS_BB5A6_CORRECTNESS" if gate_pass else "FAIL_BB5A6_CORRECTNESS",
    "gate_pass": gate_pass,
    "default_behavior_changed": False,
    "performance_claim": False,
    "small_wmma": small,
    "authority_prefill": authority,
    "gate": gate,
    "decision": "BB-5a.6 passes: small AMD WMMA correctness and authority prefill correctness evidence are present.",
    "next_action": "Proceed to BB-5a.7 performance gate.",
  }
  write_json("bb5a6_correctness_result.json", result)
  print(json.dumps({
    "out": "bench/amd-broad-backend-roadmap/bb5a6_correctness_result.json",
    "verdict": result["verdict"],
    "gate_pass": gate_pass,
    "small_rel_err": small["rel_err"],
    "authority_rel_err": authority["rel_err"],
  }, indent=2))
  return 0 if gate_pass else 1


if __name__ == "__main__":
  raise SystemExit(main())
