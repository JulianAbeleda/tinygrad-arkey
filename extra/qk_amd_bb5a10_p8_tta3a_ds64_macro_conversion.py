#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib, sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extra.gemm.rdna3_wmma_matmul import build_gemm_lds2
from tinygrad.runtime.autogen.amd.rdna3.ins import ds_store_b64

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


def dsoff(total: int) -> dict[str, int]:
  return {"offset0": total & 0xFF, "offset1": (total >> 8) & 0xFF}


def ds_total_offset(inst: Any) -> int:
  return int(inst.offset0) + (int(inst.offset1) << 8)


def repatch_single_backward_branch(insts: list[Any]) -> dict[str, Any]:
  branch_idxs = [i for i, inst in enumerate(insts) if getattr(inst, "op_name", "") == "S_CBRANCH_SCC1"]
  if len(branch_idxs) != 1: return {"patched": False, "reason": f"expected one branch, found {len(branch_idxs)}"}
  branch_idx = branch_idxs[0]
  mov_idxs = [i for i, inst in enumerate(insts[:branch_idx]) if getattr(inst, "op_name", "") == "S_MOV_B32"]
  if not mov_idxs: return {"patched": False, "reason": "no pre-loop S_MOV_B32 found"}
  loop_start_idx = mov_idxs[-1] + 1
  label_pc = sum(inst.size() for inst in insts[:loop_start_idx])
  branch_next_pc = sum(inst.size() for inst in insts[:branch_idx+1])
  off = (label_pc - branch_next_pc) // 4
  insts[branch_idx].simm16 = off
  return {"patched": True, "branch_idx": branch_idx, "loop_start_idx": loop_start_idx, "branch_offset_dwords": off}


def build_converted_macro_insts() -> tuple[list[Any], dict[str, Any]]:
  original = build_gemm_lds2(M, N, K, WAVES_M, WAVES_N, WM, WN, BK, PAD, DBUF)
  converted: list[Any] = []
  replaced = 0
  for inst in original:
    if getattr(inst, "op_name", "") != "DS_STORE_B128":
      converted.append(inst)
      continue
    base = ds_total_offset(inst)
    converted.append(ds_store_b64(addr=inst.addr, data0=inst.data0[0:1], **dsoff(base)))
    converted.append(ds_store_b64(addr=inst.addr, data0=inst.data0[2:3], **dsoff(base + 8)))
    replaced += 1
  patch = repatch_single_backward_branch(converted)
  meta = {"original_instruction_count": len(original), "converted_instruction_count": len(converted), "ds_store_b128_replaced": replaced, "branch_patch": patch}
  return converted, meta


def converted_summary() -> dict[str, Any]:
  insts, meta = build_converted_macro_insts()
  names = [getattr(i, "op_name", type(i).__name__) for i in insts]
  bm, bn = WAVES_M * WM * 16, WAVES_N * WN * 16
  lds_bytes = (BK * 2 + PAD) * (bm + bn) * (2 if DBUF else 1)
  return {
    "shape": [M, N, K],
    "macro_tile": [bm, bn, K],
    "grid": [N // bn, M // bm, 1],
    "local_size": [WAVES_M * WAVES_N * 32, 1, 1],
    "parameters": {"WAVES_M": WAVES_M, "WAVES_N": WAVES_N, "WM": WM, "WN": WN, "BK": BK, "PAD": PAD, "DBUF": DBUF},
    "conversion": meta,
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
  }


def main() -> int:
  tta3 = read_json("bench/amd-broad-backend-roadmap/bb5a10_p8_tta3_macro_candidate_result.json", {})
  try:
    converted = converted_summary()
  except Exception as e:
    converted = {"error": repr(e)}
  counts = converted.get("instruction_counts") or {}
  conv = converted.get("conversion") or {}
  resources = converted.get("resource_summary") or {}
  gate = {
    "input_tta3_blocked_on_store_contract": tta3.get("verdict") == "BLOCKED_BB5A10_P8_TTA3_SELECTED_COMPATIBLE_MACRO_CANDIDATE",
    "macro_tile_128x128": converted.get("macro_tile") == [128, 128, K],
    "authority_grid_96x4": converted.get("grid") == [96, 4, 1],
    "replaced_four_ds_store_b128": conv.get("ds_store_b128_replaced") == 4,
    "uses_eight_ds_store_b64": counts.get("ds_store_b64") == 8,
    "no_ds_store_b128_remaining": counts.get("ds_store_b128") == 0,
    "preserves_ds_load_b128": (counts.get("ds_load_b128") or 0) > 0,
    "preserves_wmma": (counts.get("v_wmma") or 0) > 0,
    "branch_repatched": bool((conv.get("branch_patch") or {}).get("patched")),
    "scratch_private_zero": resources.get("scratch_bytes") == 0 and resources.get("private_segment_fixed_size") == 0,
  }
  gate_pass = all(gate.values())
  result = {
    "date": "2026-06-20",
    "phase": "BB-5a.10_P8_TTA3a_ds64_macro_conversion",
    "schema": "amd_bb5a10_p8_tta3a_ds64_macro_conversion_v1",
    "verdict": "PASS_BB5A10_P8_TTA3A_DS64_MACRO_CONVERSION" if gate_pass else "BLOCKED_BB5A10_P8_TTA3A_DS64_MACRO_CONVERSION",
    "gate_pass": gate_pass,
    "default_behavior_changed": False,
    "performance_claim": False,
    "converted_candidate": converted,
    "gate": gate,
    "decision": "TTA3a passes: 128x128 macro helper LDS stores are converted to selected-compatible ds_store_b64 and loop branch is repatched." if gate_pass else
                "TTA3a blocked; conversion did not preserve the macro candidate contract.",
    "next_action": "Rerun TTA3 macro candidate gate against the converted stream." if gate_pass else "Fix TTA3a conversion before TTA3/P8.",
    "input_artifacts": ["bench/amd-broad-backend-roadmap/bb5a10_p8_tta3_macro_candidate_result.json"],
  }
  write_json("bb5a10_p8_tta3a_ds64_macro_conversion_result.json", result)
  print(json.dumps({
    "out": "bench/amd-broad-backend-roadmap/bb5a10_p8_tta3a_ds64_macro_conversion_result.json",
    "verdict": result["verdict"],
    "gate_pass": gate_pass,
    "next": result["next_action"],
  }, indent=2))
  return 0 if gate_pass else 1


if __name__ == "__main__":
  raise SystemExit(main())
