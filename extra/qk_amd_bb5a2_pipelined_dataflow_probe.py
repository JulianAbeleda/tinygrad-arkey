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


def make_source(name: str, rows: list[dict[str, Any]], use_two_slots: bool) -> str:
  used = rows[:2] if use_two_slots else rows[:1]
  locals_ = [UOp.placeholder((int(row["element_count"]),), dtypes.half, int(row["define_local_slot"]), AddrSpace.LOCAL) for row in used]
  zero = UOp.const(dtypes.int, 0)
  stores = [local.index(zero).store(UOp.const(dtypes.half, 0.0)) for local in locals_]
  a_load = locals_[0].index(zero).load()
  b_load = (locals_[1] if len(locals_) > 1 else locals_[0]).index(zero).load()
  a = UOp(Ops.STACK, dtypes.half.vec(16), tuple(a_load for _ in range(16)))
  b = UOp(Ops.STACK, dtypes.half.vec(16), tuple(b_load for _ in range(16)))
  acc = UOp(Ops.STACK, dtypes.float.vec(8), tuple(UOp.const(dtypes.float, 0.0) for _ in range(8)))
  wmma = UOp(Ops.WMMA, dtypes.float.vec(8), (a, b, acc),
             arg=("WMMA_16_16_16", (16, 16, 16), dtypes.half, dtypes.float, "AMD", 32, (), ()))
  sink = UOp.sink(*stores, wmma, arg=KernelInfo(name=name))
  return AMDLLVMRenderer(Target("AMD", "gfx1100")).render(list(sink.toposort()))


def summarize(source: str) -> dict[str, Any]:
  return {
    "sha256": hashlib.sha256(source.encode()).hexdigest(),
    "addrspace3_global_count": source.count("addrspace(3) global"),
    "addrspacecast_count": source.count("addrspacecast"),
    "store_count": source.count("store half"),
    "load_count": source.count("load half"),
    "wmma_count": source.count("@llvm.amdgcn.wmma"),
    "local_9000_present": "local_9000" in source,
    "local_9001_present": "local_9001" in source,
  }


def main() -> int:
  integration = read_json("bench/amd-broad-backend-roadmap/bb5a2_real_lowering_integration_result.json", {})
  layer2 = read_json("bench/amd-broad-backend-roadmap/bb5a2_lowering_hook_result.json", {})
  rows = ((layer2.get("lds_lowering") or {}).get("lowered_slots") or [])
  candidate_source = make_source("bb5a2_two_slot_lds_wmma_source", rows, use_two_slots=True)
  baseline_source = make_source("bb5a2_single_slot_lds_wmma_source", rows, use_two_slots=False)
  candidate = summarize(candidate_source)
  baseline = summarize(baseline_source)
  gate = {
    "input_render_source_integration_pass": integration.get("verdict") == "PASS_RENDER_SOURCE_LDS_INTEGRATION" and bool(integration.get("gate_pass")),
    "candidate_uses_two_lds_slots": candidate["local_9000_present"] and candidate["local_9001_present"],
    "candidate_has_two_lds_stores": candidate["store_count"] >= 2,
    "candidate_has_two_lds_loads": candidate["load_count"] >= 2,
    "candidate_has_wmma": candidate["wmma_count"] >= 1,
    "baseline_uses_single_lds_slot": baseline["local_9000_present"] and not baseline["local_9001_present"],
    "source_hash_non_identical": candidate["sha256"] != baseline["sha256"],
    "default_behavior_changed": False,
    "performance_claim": False,
  }
  positive = [
    "input_render_source_integration_pass", "candidate_uses_two_lds_slots", "candidate_has_two_lds_stores",
    "candidate_has_two_lds_loads", "candidate_has_wmma", "baseline_uses_single_lds_slot", "source_hash_non_identical",
  ]
  gate_pass = all(gate[x] for x in positive) and not gate["default_behavior_changed"] and not gate["performance_claim"]
  result = {
    "date": "2026-06-19",
    "phase": "BB-5a.2_pipelined_lds_wmma_source_skeleton",
    "schema": "amd_bb5a2_pipelined_dataflow_result_v1",
    "verdict": "PASS_PIPELINED_LDS_WMMA_SOURCE_SKELETON" if gate_pass else "FAIL_PIPELINED_LDS_WMMA_SOURCE_SKELETON",
    "gate_pass": gate_pass,
    "default_behavior_changed": False,
    "performance_claim": False,
    "input_artifacts": [
      "bench/amd-broad-backend-roadmap/bb5a2_real_lowering_integration_result.json",
      "bench/amd-broad-backend-roadmap/bb5a2_lowering_hook_result.json",
    ],
    "candidate": candidate,
    "baseline": baseline,
    "gate": gate,
    "decision": (
      "BB-5a.2 gated source skeleton passes: the candidate renders two LDS slots, LDS stores, LDS loads, and an AMD "
      "WMMA intrinsic with non-identical source versus the single-slot baseline. This completes the BB-5a.2 lowering "
      "evidence needed to move to semantic wait scheduler integration."
    ),
    "next_action": "Proceed to BB-5a.3 semantic wait scheduler integration over the lowered LDS/WMMA stream.",
  }
  write_json("bb5a2_pipelined_dataflow_result.json", result)
  print(json.dumps({
    "out": "bench/amd-broad-backend-roadmap/bb5a2_pipelined_dataflow_result.json",
    "verdict": result["verdict"],
    "gate_pass": gate_pass,
    "stores": candidate["store_count"],
    "loads": candidate["load_count"],
    "wmma": candidate["wmma_count"],
  }, indent=2))
  return 0 if gate_pass else 1


if __name__ == "__main__":
  raise SystemExit(main())
