#!/usr/bin/env python3
from __future__ import annotations

import hashlib, json, pathlib, sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.helpers import Target
from tinygrad.renderer.llvmir import AMDLLVMRenderer
from tinygrad.uop.ops import KernelInfo, Ops, UOp

OUT = ROOT / "bench/amd-broad-backend-roadmap"


def read_json(rel: str, default: Any = None) -> Any:
  path = ROOT / rel
  if not path.exists(): return default
  return json.loads(path.read_text())


def write_json(name: str, data: Any) -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def make_source(name: str, lowered_rows: list[dict[str, Any]]) -> str:
  locals_ = tuple(
    UOp.placeholder((int(row["element_count"]),), dtypes.half, int(row["define_local_slot"]), AddrSpace.LOCAL)
    for row in lowered_rows
  )
  sink = UOp.sink(*locals_, arg=KernelInfo(name=name))
  return AMDLLVMRenderer(Target("AMD", "gfx1100")).render(list(sink.toposort()))


def source_summary(source: str) -> dict[str, Any]:
  lines = source.splitlines()
  return {
    "sha256": hashlib.sha256(source.encode()).hexdigest(),
    "line_count": len(lines),
    "addrspace3_global_count": sum("addrspace(3) global" in line for line in lines),
    "addrspacecast_count": sum("addrspacecast" in line for line in lines),
    "local_9000_present": "local_9000" in source,
    "local_9001_present": "local_9001" in source,
    "contains_wmma": "amdgcn.wmma" in source or "amdgcn.mfma" in source,
    "contains_lds_store_or_load": any(x in source for x in (" store ", " load ")),
  }


def main() -> int:
  layer3 = read_json("bench/amd-broad-backend-roadmap/bb5a2_render_isa_evidence_result.json", {})
  layer2 = read_json("bench/amd-broad-backend-roadmap/bb5a2_lowering_hook_result.json", {})
  rows = ((layer2.get("lds_lowering") or {}).get("lowered_slots") or [])
  candidate_source = make_source("bb5a2_two_slot_render_source", rows)
  baseline_source = make_source("bb5a2_single_slot_render_source", rows[:1])
  candidate = source_summary(candidate_source)
  baseline = source_summary(baseline_source)
  gate = {
    "input_render_elf_evidence_pass": layer3.get("verdict") == "PASS_RENDER_ELF_LDS_EVIDENCE" and bool(layer3.get("gate_pass")),
    "candidate_has_two_addrspace3_globals": candidate["addrspace3_global_count"] == 2,
    "candidate_has_two_addrspacecasts": candidate["addrspacecast_count"] == 2,
    "candidate_names_both_slots": candidate["local_9000_present"] and candidate["local_9001_present"],
    "baseline_names_single_slot": baseline["local_9000_present"] and not baseline["local_9001_present"],
    "source_hash_non_identical": candidate["sha256"] != baseline["sha256"],
    "pipelined_dataflow_present": candidate["contains_wmma"] and candidate["contains_lds_store_or_load"],
    "default_behavior_changed": False,
    "performance_claim": False,
  }
  positive = [
    "input_render_elf_evidence_pass", "candidate_has_two_addrspace3_globals", "candidate_has_two_addrspacecasts",
    "candidate_names_both_slots", "baseline_names_single_slot", "source_hash_non_identical",
  ]
  gate_pass = all(gate[x] for x in positive) and not gate["default_behavior_changed"] and not gate["performance_claim"]
  result = {
    "date": "2026-06-19",
    "phase": "BB-5a.2_real_lowering_integration_probe",
    "schema": "amd_bb5a2_real_lowering_integration_result_v1",
    "verdict": "PASS_RENDER_SOURCE_LDS_INTEGRATION" if gate_pass else "FAIL_RENDER_SOURCE_LDS_INTEGRATION",
    "gate_pass": gate_pass,
    "default_behavior_changed": False,
    "performance_claim": False,
    "input_artifacts": [
      "bench/amd-broad-backend-roadmap/bb5a2_lowering_hook_result.json",
      "bench/amd-broad-backend-roadmap/bb5a2_render_isa_evidence_result.json",
    ],
    "candidate": candidate,
    "baseline": baseline,
    "gate": gate,
    "decision": (
      "Renderer source integration passes: AMD LLVM rendering sees both lowered LDS slots and emits non-identical "
      "source relative to the single-slot baseline. Full BB-5a.2 remains blocked because no pipelined LDS store/load "
      "plus WMMA dataflow is integrated yet."
    ),
    "next_action": "Build a gated pipelined dataflow skeleton that stores/loads both LDS slots and reaches WMMA-shaped source/ISA.",
  }
  write_json("bb5a2_real_lowering_integration_result.json", result)
  print(json.dumps({
    "out": "bench/amd-broad-backend-roadmap/bb5a2_real_lowering_integration_result.json",
    "verdict": result["verdict"],
    "gate_pass": gate_pass,
    "candidate_addrspace3_globals": candidate["addrspace3_global_count"],
    "pipelined_dataflow_present": gate["pipelined_dataflow_present"],
  }, indent=2))
  return 0 if gate_pass else 1


if __name__ == "__main__":
  raise SystemExit(main())
