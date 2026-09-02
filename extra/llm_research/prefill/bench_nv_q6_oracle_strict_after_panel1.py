#!/usr/bin/env python3
"""Non-GPU, fail-closed preflight for future strict-after Q8 panel-1 integration."""
from __future__ import annotations

import argparse, importlib, inspect, json, os
from pathlib import Path

ANCHOR_MAIN_SHA = "6eb663b3a3fd628e3394a0ce8f8780e108e47f40b887b0a75a0756dcf33e9137"
FIXUP_SHA = "483de2ee3eed3597932a8632f9892377ce054e77bfe34c2420fe5a5d54ff5514"

LDG_PCS = (0x1e80, 0x1eb0, 0x1ee0, 0x1f10, 0x1f40, 0x1f70, 0x1fa0, 0x1fd0,
           0x2020, 0x2050, 0x20b0, 0x20e0, 0x2110, 0x2170, 0x21b0, 0x21f0, 0x2220, 0x2280)
LDG_REGS = (192, 191, 190, 189, 188, 187, 186, 185, 184, 183, 119, 182, 178, 177, 176, 175, 174, 173)
REMAINING_IMMA_PCS = (0xa060, 0xa180, 0xa230, 0xa280, 0xa2b0, 0xa340, 0xa4d0, 0xa670, 0xa760, 0xa810)

DEPENDENCY_CONTRACT = {
  "source": "admitted anchor SASS, not CUDA source order",
  "phase": "phase0 tail after the IMMA at 0x9f70 and before ten remaining phase0 IMMA instructions",
  "anchor_token": {"pc": "0x9f80", "ordinal": 0x9f80 // 16, "instruction": "FADD R167, R53, R36"},
  "first_legal_panel1_ldg_ordinal": 0x9f80 // 16 + 1,
  "initial_combined_barrier": {"pc": "0x3060", "ordinal": 0x3060 // 16},
  "overwrite_barrier": {"pc": "0xa930", "ordinal": 0xa930 // 16},
  "panel1_publish_barrier": {"pc": "0xaab0", "ordinal": 0xaab0 // 16},
  "anchor_first_panel1_sts": {"pc": "0xa990", "ordinal": 0xa990 // 16},
  "maximum_first_ldg_to_first_sts_span": 160,
  "remaining_imma_pcs": [f"0x{x:x}" for x in REMAINING_IMMA_PCS],
}

PANEL1_WORDS = [{
  "logical_word": i,
  "anchor_global_byte_offset": f"0x{0x4800 + i*0x400:x}",
  "anchor_ldg_pc": f"0x{LDG_PCS[i]:x}",
  "anchor_ldg_register": f"R{LDG_REGS[i]}",
  "anchor_shared_byte_offset": f"0x{0x9800 + i*0x400:x}",
  "anchor_sts_pc": f"0x{0xa990 + i*0x10:x}",
} for i in range(18)]

GATES = {
  "frozen": {
    "anchor_main_cubin_sha256": ANCHOR_MAIN_SHA,
    "all_partials_fixup_cubin_sha256": FIXUP_SHA,
    "anchor_default_builder_must_remain_byte_identical": True,
    "candidate_cubin_must_differ_from_anchor": True,
  },
  "correctness": {
    "trusted_reference_exact": True,
    "candidate_vs_anchor_partial_uint32_identity": True,
    "candidate_vs_anchor_final_uint32_identity": True,
    "active_outputs_finite": True,
    "unused_outputs_nan": True,
  },
  "sass": {
    "panel1_ldg": 18, "panel1_sts": 18,
    "first_ldg_after_dependency_token": True,
    "all_ldg_before_overwrite_barrier": True,
    "all_sts_after_overwrite_and_before_publish_barrier": True,
    "first_ldg_to_first_sts_span_le": 160,
    "families": {"IMMA": 256, "LDSM": 32, "LDS": 176, "LDG": 109, "STS": 73, "STG": 64, "BAR": 4},
    "arithmetic": {"I2FP": 1024, "FMUL": 1544, "FADD": 1024, "FFMA": 0},
    "instruction_total_le": 5144,
  },
  "resources": {"registers_le": 255, "stack_bytes": 0, "local_static_bytes": 0, "LDL": 0, "STL": 0},
  "timing": {"mode": "same-process alternating locked R31", "warmups": 3, "rounds": 31,
             "main_paired_median_delta_us_le": -3.0, "total_paired_median_delta_us_le": -3.0,
             "main_wins_ge": 24, "total_wins_ge": 24},
}


def resolve(spec: str):
  if not spec or ":" not in spec: raise ValueError("expected an explicitly supplied MODULE:SYMBOL")
  module_name, symbol_name = spec.rsplit(":", 1)
  value = getattr(importlib.import_module(module_name), symbol_name)
  if not callable(value): raise TypeError(f"{spec} is not callable")
  return value


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--strict-helper", default=os.getenv("NV_Q6_STRICT_AFTER_HELPER", ""),
                  help="Future core helper as MODULE:SYMBOL; deliberately has no repository default")
  ap.add_argument("--candidate-factory", default=os.getenv("NV_Q6_STRICT_AFTER_CANDIDATE_FACTORY", ""),
                  help="Future isolated Q8 candidate factory as MODULE:SYMBOL; deliberately unbound")
  ap.add_argument("--out", default="docs/task_workflow/evidence/nv-q6-strict-after-panel1-integration-20260831/preflight.json")
  args = ap.parse_args()
  result = {
    "schema": "tinygrad.nv_q6_strict_after_panel1_preflight.v1",
    "mode": "non_gpu_fail_closed_scaffold",
    "dependency_contract": DEPENDENCY_CONTRACT,
    "panel1_words": PANEL1_WORDS,
    "gates": GATES,
    "binding": {"strict_helper": args.strict_helper or None, "candidate_factory": args.candidate_factory or None},
    "compile_started": False, "device_runtime_imported": False, "gpu_work_started": False,
  }
  missing, errors = [], []
  helper = factory = None
  if not args.strict_helper: missing.append("strict_helper ABI is intentionally unbound until core substrate release")
  else:
    try: helper = resolve(args.strict_helper)
    except Exception as exc: errors.append(f"strict_helper: {type(exc).__name__}: {exc}")
  if not args.candidate_factory: missing.append("Q8 candidate factory is intentionally unbound until helper ABI release")
  else:
    try: factory = resolve(args.candidate_factory)
    except Exception as exc: errors.append(f"candidate_factory: {type(exc).__name__}: {exc}")
  if missing or errors:
    result |= {"verdict": "BLOCKED_CORE_SUBSTRATE_UNAVAILABLE", "promotion_eligible": False,
               "missing": missing, "errors": errors}
    exit_code = 2
  else:
    result |= {"verdict": "READY_FOR_NON_GPU_Q8_INTEGRATION", "promotion_eligible": False,
               "resolved": {"strict_helper_signature": str(inspect.signature(helper)),
                            "candidate_factory_signature": str(inspect.signature(factory))},
               "next_required_action": "bind the released ABI in an isolated candidate arm, then run compile/SASS gates before any GPU work"}
    exit_code = 0
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({k: result[k] for k in ("verdict", "compile_started", "device_runtime_imported", "gpu_work_started")}, sort_keys=True))
  return exit_code


if __name__ == "__main__": raise SystemExit(main())
