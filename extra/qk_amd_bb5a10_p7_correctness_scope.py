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
  p6 = read_json("bench/amd-broad-backend-roadmap/bb5a10_p6_structural_candidate_result.json", {})
  p2 = read_json("bench/amd-broad-backend-roadmap/bb5a10_p2_rendered_lds_result.json", {})
  candidate = p6.get("candidate") or {}
  inst_names = candidate.get("instruction_names") or []
  scope = [
    {
      "id": "P7a",
      "name": "known-good LDS WMMA hardware smoke",
      "minimum_pass": "existing in-repo RDNA3 LDS tile path runs and reports relative RMSE <= 0.05",
      "route": "reuse extra/gemm/rdna3_wmma_matmul.py LDS tile harness as environment/hardware sanity check",
      "if_blocked": "stop P7; hardware/runtime correctness harness is not trustworthy",
    },
    {
      "id": "P7b",
      "name": "structural candidate executable wrapper",
      "minimum_pass": "P6 instruction stream is wrapped as a Tensor.custom_kernel PROGRAM with real kernargs, LDS allocation, lidx/gidx, and at least one output store",
      "route": "extend the structural candidate; P6 currently ends after WMMA and is not numerically checkable",
      "if_blocked": "split wrapper construction from numeric correctness; do not proceed to P8",
    },
    {
      "id": "P7c",
      "name": "small deterministic numeric correctness",
      "minimum_pass": "16x16x16 or two-tile variant using staged LDS returns fp16 output with relative RMSE <= 0.05 versus numpy fp32 reference",
      "route": "complete input load -> selected-compatible LDS store -> ds_load_b128 -> WMMA -> output-store path",
      "if_blocked": "debug LDS address mapping and WMMA fragment layout before authority shape",
    },
    {
      "id": "P7d",
      "name": "authority-shape correctness smoke",
      "minimum_pass": "authority-shape or tiled authority-subset correctness passes rel_err <= 1e-3 on deterministic fp16 inputs",
      "route": "scale from small tile to selected authority contract without claiming performance",
      "if_blocked": "debug launch mapping, edge predicates, and K-loop coverage; do not time",
    },
    {
      "id": "P7e",
      "name": "P8 handoff package",
      "minimum_pass": "correct executable candidate artifact records source/ISA/resource metadata and exact command to run P8 timing",
      "route": "package correctness output plus structural evidence for performance gate",
      "if_blocked": "P8 remains blocked until a reproducible correctness artifact exists",
    },
  ]
  blockers = [
    {
      "blocker": "P6 structural stream has no output store",
      "why": "it proves LDS/WMMA structure but cannot be compared numerically",
      "resolution": "P7b must add output-store code and Tensor.custom_kernel wrapper",
    },
    {
      "blocker": "P6 structural stream is not a complete K-loop matmul",
      "why": "it is a minimal representative instruction package, not all input fragments or output lanes",
      "resolution": "P7c must complete a small deterministic tile before authority-shape correctness",
    },
    {
      "blocker": "performance gate would be meaningless before correctness",
      "why": "P8 TFLOPS can only time a numerically valid candidate",
      "resolution": "keep P8 and q8 transfer blocked until P7d/P7e pass",
    },
  ]
  gate = {
    "input_p6_pass": p6.get("verdict") == "PASS_BB5A10_P6_STRUCTURAL_CANDIDATE" and bool(p6.get("gate_pass")),
    "p6_has_lds": (candidate.get("lds_bytes") or 0) > 0,
    "p6_has_ds_load_b128": "DS_LOAD_B128" in inst_names,
    "p6_has_wmma": any("WMMA" in n for n in inst_names),
    "p6_has_no_output_store": not any("STORE" in n and not n.startswith("DS_") for n in inst_names),
    "p7_subphase_count": len(scope),
    "blocked_continuations_present": all(bool(row["if_blocked"]) for row in scope),
    "p8_blocked_until_correctness": scope[-1]["id"] == "P7e",
  }
  gate_pass = all(v for k, v in gate.items() if k != "p6_has_no_output_store") and gate["p6_has_no_output_store"]
  result = {
    "date": "2026-06-19",
    "phase": "BB-5a.10_P7_correctness_scope",
    "schema": "amd_bb5a10_p7_correctness_scope_v1",
    "verdict": "PASS_BB5A10_P7_CORRECTNESS_SCOPE_READY" if gate_pass else "BLOCKED_BB5A10_P7_CORRECTNESS_SCOPE_INPUTS",
    "gate_pass": gate_pass,
    "default_behavior_changed": False,
    "performance_claim": False,
    "scope": scope,
    "known_blockers": blockers,
    "current_structural_candidate": {
      "lds_bytes": candidate.get("lds_bytes"),
      "instruction_names": inst_names,
      "p2_instruction_sha256": p2.get("instruction_sha256"),
    },
    "gate": gate,
    "decision": "P7 is scoped. First implement P7a/P7b: prove the hardware harness and wrap the structural candidate into an executable kernel with output.",
    "next_action": "Implement P7a known-good LDS WMMA smoke, then P7b executable structural wrapper. P8 remains blocked.",
    "input_artifacts": [
      "bench/amd-broad-backend-roadmap/bb5a10_p6_structural_candidate_result.json",
      "bench/amd-broad-backend-roadmap/bb5a10_p2_rendered_lds_result.json",
    ],
  }
  write_json("bb5a10_p7_correctness_scope_result.json", result)
  print(json.dumps({
    "out": "bench/amd-broad-backend-roadmap/bb5a10_p7_correctness_scope_result.json",
    "verdict": result["verdict"],
    "gate_pass": gate_pass,
    "subphases": [row["id"] for row in scope],
    "next": result["next_action"],
  }, indent=2))
  return 0 if gate_pass else 1


if __name__ == "__main__":
  raise SystemExit(main())
