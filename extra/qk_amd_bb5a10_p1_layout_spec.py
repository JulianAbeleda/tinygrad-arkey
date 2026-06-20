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
  audit = read_json("bench/amd-broad-backend-roadmap/bb5a10_tensile_layout_audit_result.json", {})
  offsets = ((audit.get("disasm_summary") or {}).get("offset_summary") or {})
  store_offsets = offsets.get("ds_store_b64", {}).get("first_64_sorted") or []
  load_offsets = offsets.get("ds_load_b128", {}).get("first_64_sorted") or []
  contract = audit.get("oracle_contract") or {}
  lds_bytes = int(contract.get("lds_bytes") or 0)
  # Non-bitexact regions derived from the selected-function offset families.
  layout = {
    "strategy": "selected_kernel_compatible_non_bitexact",
    "authority_contract": {
      "macro_tile": (contract.get("schedule") or {}).get("macro_tile_MxNxK"),
      "wmma": (contract.get("schedule") or {}).get("wmma_MI"),
      "depthU": (contract.get("schedule") or {}).get("depthU"),
      "lds_bytes": lds_bytes,
      "vgpr": contract.get("vgpr"),
      "scratch": contract.get("scratch"),
    },
    "lds_regions": [
      {
        "name": "operand_A_or_low_region",
        "base": 0,
        "observed_store_offsets": [x for x in store_offsets if x < 16384],
        "observed_load_b128_offsets": [x for x in load_offsets if x < 16384],
      },
      {
        "name": "operand_B_or_high_stage_region",
        "base": 16384,
        "observed_store_offsets": [x for x in store_offsets if x >= 16384],
        "observed_load_b128_offsets": [x for x in load_offsets if x >= 16384],
      },
    ],
    "required_lowering_features": [
      "nonzero DEFINE_LOCAL/ELF LDS allocation",
      "selected-kernel-compatible LDS stores; selected authority uses ds_store_b64",
      "ds_load_b128 LDS reads",
      "WMMA source operands overlap ds_load_b128 destination VGPRs",
      "dependency metadata for vmcnt/lgkmcnt waits and barriers",
      "resource policy rejects scratch/private spill before timing",
    ],
    "not_required": [
      "bit-identical Tensile LDS byte layout",
      "ds_store_b128 for the first selected rocBLAS authority candidate",
      "q8 transfer before >=60 TFLOPS pure tinygrad prefill",
    ],
  }
  gate = {
    "input_audit_pass": audit.get("verdict") == "PASS_TENSILE_LAYOUT_AUDIT_CANDIDATE_SPEC_READY_NOT_BITEXACT" and bool(audit.get("gate_pass")),
    "has_nonzero_lds_budget": lds_bytes > 0,
    "has_store_offsets": bool(store_offsets),
    "has_load_b128_offsets": bool(load_offsets),
    "has_two_logical_regions": len(layout["lds_regions"]) == 2 and all(region["observed_load_b128_offsets"] for region in layout["lds_regions"]),
    "store_path_matches_selected_authority": ((audit.get("disasm_summary") or {}).get("instruction_counts") or {}).get("ds_store_b64", 0) > 0,
    "does_not_require_bitexact_layout": "bit-identical Tensile LDS byte layout" in layout["not_required"],
    "does_not_require_ds_store_b128": "ds_store_b128 for the first selected rocBLAS authority candidate" in layout["not_required"],
  }
  gate_pass = all(gate.values())
  result = {
    "date": "2026-06-19",
    "phase": "BB-5a.10_P1_selected_layout_lowering_spec",
    "schema": "amd_bb5a10_p1_layout_spec_v1",
    "verdict": "PASS_BB5A10_P1_LAYOUT_SPEC_READY" if gate_pass else "BLOCKED_BB5A10_P1_LAYOUT_SPEC_INPUTS",
    "gate_pass": gate_pass,
    "default_behavior_changed": False,
    "performance_claim": False,
    "layout_spec": layout,
    "gate": gate,
    "decision": "P1 complete: implement against a selected-kernel-compatible non-bitexact LDS layout spec, with ds_store_b64 stores and ds_load_b128 WMMA operand reads.",
    "next_action": "Run P2/P3/P4/P5 implementation batch: renderer LDS lowering, K-loop staging, semantic waits/barriers, and resource policy.",
    "input_artifacts": ["bench/amd-broad-backend-roadmap/bb5a10_tensile_layout_audit_result.json"],
  }
  write_json("bb5a10_p1_layout_spec_result.json", result)
  print(json.dumps({
    "out": "bench/amd-broad-backend-roadmap/bb5a10_p1_layout_spec_result.json",
    "verdict": result["verdict"],
    "gate_pass": gate_pass,
    "regions": [r["name"] for r in layout["lds_regions"]],
  }, indent=2))
  return 0 if gate_pass else 1


if __name__ == "__main__":
  raise SystemExit(main())
