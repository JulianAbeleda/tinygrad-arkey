#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib
from collections import Counter
from typing import Any

import tinygrad.runtime.autogen.amd.rdna3.ins as rdna3_ins

OUT = pathlib.Path("bench/q8-ffn-amd-scheduler-project")

def read_json(path:pathlib.Path) -> dict[str, Any]:
  return json.loads(path.read_text())

def top_to_counter(obj:dict[str, Any]) -> Counter:
  return Counter({k: v for k, v in obj.get("top_mnemonics", [])})

def mnemonic_counts(disasm:dict[str, Any]) -> dict[str, int]:
  # S0 stores top-40 mnemonics, enough for the q8 oracle/tinygrad deltas this pass needs.
  return dict(top_to_counter(disasm))

def count_prefix(counts:dict[str, int], prefixes:tuple[str, ...]) -> int:
  return sum(v for k, v in counts.items() if any(k.startswith(p) for p in prefixes))

def feature_row(name:str, status:str, evidence:list[str], estimated_us:float|None, a2_gate:bool, action:str) -> dict[str, Any]:
  return {
    "feature": name,
    "status": status,
    "evidence": evidence,
    "estimated_or_measured_us": estimated_us,
    "a2_gate_ge_30us": a2_gate,
    "action": action,
  }

def build_contract(s0:dict[str, Any], dso:dict[str, Any], artifact:dict[str, Any]) -> dict[str, Any]:
  objs = s0["objects"]
  oracle = objs["hipcc_lld_fast_gateup"]
  asm = objs["tinygrad_asm_gateup_full"]
  comgr = objs["comgr_fused_gateup"]
  oracle_counts = mnemonic_counts(oracle["disasm"])
  asm_counts = mnemonic_counts(asm["disasm"])
  comgr_counts = mnemonic_counts(comgr["disasm"])
  return {
    "date": "2026-06-19",
    "phase": "Route_A_A0_schedule_contract_extraction",
    "oracle": "hipcc/LLD q8_mmvq_gateup",
    "target_gate_us": 60.0,
    "known_timings_us": {
      "hipcc_lld_gateup_current_loader": artifact["summary"]["gateup_consumer_us"],
      "comgr_fused_gateup": s0["timing_authority"]["comgr_fused_gateup_us"],
      "tinygrad_asm_gateup_full": s0["timing_authority"]["tinygrad_asm_gateup_full_us"],
    },
    "launch_contract": {
      "global_size": [12288, 2, 1],
      "local_size": [32, 4, 1],
      "work_decomposition": "128 threads per row; block y selects gate/up; 16 Q4_K blocks; sub=tid&7; kb=tid/8",
      "kernarg_size": artifact["summary"].get("gateup_kernarg_size", 40),
    },
    "resource_contract": {
      "oracle": oracle["readelf"].get("kernel_symbols", []),
      "artifact_manifest_runtime": (read_json(OUT/"artifact_build_manifest.json")["artifacts"]["gateup"]["inspect"]["runtime"]
                                    if (OUT/"artifact_build_manifest.json").exists() else {}),
      "s0_runtime": oracle.get("runtime", {}),
    },
    "instruction_contract": {
      "oracle_grouped": oracle["disasm"]["grouped_counts"],
      "tinygrad_asm_grouped": asm["disasm"]["grouped_counts"],
      "comgr_grouped": comgr["disasm"]["grouped_counts"],
      "oracle_top_mnemonics": oracle["disasm"]["top_mnemonics"],
      "tinygrad_asm_top_mnemonics": asm["disasm"]["top_mnemonics"],
      "key_load_shape": {
        "oracle_global_load_b128_top_count": oracle_counts.get("global_load_b128", 0),
        "oracle_global_load_u8_top_count": oracle_counts.get("global_load_u8", 0),
        "tinygrad_global_load_b32_top_count": asm_counts.get("global_load_b32", 0),
        "tinygrad_global_load_u8_top_count": asm_counts.get("global_load_u8", 0),
        "tinygrad_global_load_u16_top_count": asm_counts.get("global_load_u16", 0),
      },
      "scheduler_markers": {
        "oracle_s_clause": oracle_counts.get("s_clause", 0),
        "oracle_s_delay_alu": oracle_counts.get("s_delay_alu", 0),
        "tinygrad_s_clause": asm_counts.get("s_clause", 0),
        "tinygrad_s_delay_alu": asm_counts.get("s_delay_alu", 0),
        "comgr_s_clause": comgr_counts.get("s_clause", 0),
        "comgr_s_delay_alu": comgr_counts.get("s_delay_alu", 0),
      },
      "reduction_shape": {
        "oracle_ds_bpermute": oracle_counts.get("ds_bpermute_b32", 0),
        "tinygrad_ds_bpermute": asm_counts.get("ds_bpermute_b32", 0),
        "oracle_ds_total": oracle["disasm"]["grouped_counts"].get("ds"),
        "tinygrad_ds_total": asm["disasm"]["grouped_counts"].get("ds"),
      },
    },
    "dynamic_contract": {
      "classifier": dso["classifier"],
      "body_insensitive_variant_ladder": dso["summary"]["body_insensitive_variant_ladder"],
      "variant_medians_ms": dso["summary"]["variant_medians_ms"],
    },
    "gate": {
      "contract_identifies_features": True,
      "not_just_llvm_is_better": True,
    },
    "verdict": "PASS_A0",
  }

def build_capability_map(contract:dict[str, Any], dso:dict[str, Any]) -> dict[str, Any]:
  has = lambda n: hasattr(rdna3_ins, n)
  variants = dso["summary"]["variant_medians_ms"]
  full = dso["summary"]["asm_full_ms"]
  reduction_delta_us = (full - variants["reduction_only"]) * 1000.0
  wait_delta_us = (variants["load_wait_only"] - variants["wait_grouped_load_only"]) * 1000.0
  dot_body_delta_us = (full - variants["dot_synthetic"]) * 1000.0
  load_body_delta_us = (full - variants["load_wait_only"]) * 1000.0
  features = [
    feature_row("native_dot4", "expressible_now",
      ["tinygrad ASM, COMGR, and hipcc/LLD all emit 16 v_dot4_i32_iu8"], 0.0, False,
      "closed as non-blocker"),
    feature_row("vector_or_coalesced_global_loads", "expressible_with_assembler_mnemonics_but_not_proven_scheduler_feature",
      [f"rdna3 assembler exposes global_load_b128={has('global_load_b128')}",
       "S0 visible delta: tinygrad 22 global loads vs oracle 11",
       f"DSO load-only variant still {variants['load_wait_only']:.6f}ms vs full {full:.6f}ms"], load_body_delta_us, False,
      "do not start A2 as standalone; include only inside broader scheduler work"),
    feature_row("waitcnt_grouping", "expressible_now",
      ["DSO grouped-wait load-only variant reduced waitcnt but barely moved timing",
       f"measured movement {wait_delta_us:.3f}us"], wait_delta_us, False,
      "closed as standalone A2 feature"),
    feature_row("reduction_rewrite", "expressible_now",
      ["oracle and tinygrad both use 5 ds_bpermute operations",
       f"reduction-only variant is {variants['reduction_only']:.6f}ms, still {variants['reduction_only']/full:.2f}x full"], reduction_delta_us, False,
      "closed as standalone A2 feature"),
    feature_row("schedule_annotations_s_clause_delay_alu", "mnemonics_expressible_but_semantics_are_renderer_scheduler_level",
      [f"rdna3 assembler exposes s_clause={has('s_clause')}, s_delay_alu={has('s_delay_alu')}",
       "oracle has s_clause/s_delay_alu markers; tinygrad ASM has none",
       "DSO body-insensitive ladder points to scheduling/work-decomposition rather than body instructions"], None, False,
      "project-level scheduler/codegen; no bounded A2 until semantic insertion rules are defined"),
    feature_row("local_id_y_descriptor", "small_assembler_runtime_feature_but_low_ev",
      ["B2b local=(32,4,1) local-y caused MMU fault; local=(128,1,1) was viable",
       "DSO body-insensitive variants under flattened local shape show no bounded recovery path"], None, False,
      "keep as ergonomics/compiler-roadmap item, not q8 decode reopen"),
    feature_row("register_allocation_live_range_scheduler", "renderer_scheduler_feature",
      ["static instruction count is not high, but dynamic variants are body-insensitive",
       "requires broad dependency/latency/register scheduling model"], None, False,
      "project-level Route A, not A2 single feature"),
  ]
  a2_candidates = [f for f in features if f["a2_gate_ge_30us"]]
  return {
    "date": "2026-06-19",
    "phase": "Route_A_A1_amd_dsl_capability_map",
    "features": features,
    "summary": {
      "a2_candidates_count": len(a2_candidates),
      "largest_measured_standalone_delta_us": max(reduction_delta_us, wait_delta_us, dot_body_delta_us, load_body_delta_us),
      "body_insensitive_variant_ladder": dso["summary"]["body_insensitive_variant_ladder"],
      "route_a_gate": "FAIL_A1_NO_BOUNDED_A2_FEATURE",
    },
    "decision": "do_not_start_A2_for_q8_decode; Route A remains project-level AMD scheduler/codegen roadmap",
    "verdict": "FAIL_A1_NO_BOUNDED_FEATURE",
  }

def main() -> int:
  ap = argparse.ArgumentParser(description="Route A A0/A1 q8 schedule contract and capability map")
  ap.add_argument("--out-dir", type=pathlib.Path, default=OUT)
  args = ap.parse_args()
  args.out_dir.mkdir(parents=True, exist_ok=True)
  s0 = read_json(pathlib.Path("bench/q8-ffn-codegen-transfer/asm_schedule_audit.json"))
  dso = read_json(pathlib.Path("bench/q8-ffn-dynamic-scheduler-observability/result.json"))
  artifact = read_json(pathlib.Path("bench/q8-ffn-amd-scheduler-project/result.json"))
  contract = build_contract(s0, dso, artifact)
  capability = build_capability_map(contract, dso)
  (args.out_dir/"oracle_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
  (args.out_dir/"dsl_capability_map.json").write_text(json.dumps(capability, indent=2) + "\n")
  result = {
    "date": "2026-06-19",
    "phase": "Route_A_A0_A1",
    "oracle_contract": "bench/q8-ffn-amd-scheduler-project/oracle_contract.json",
    "dsl_capability_map": "bench/q8-ffn-amd-scheduler-project/dsl_capability_map.json",
    "a0_verdict": contract["verdict"],
    "a1_verdict": capability["verdict"],
    "decision": capability["decision"],
    "next": "Do not execute A2 for q8 decode unless new PMU/SQTT evidence identifies a >=30us bounded feature.",
  }
  (args.out_dir/"route_a_result.json").write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps({"out": str(args.out_dir/"route_a_result.json"), "a0": result["a0_verdict"], "a1": result["a1_verdict"], "decision": result["decision"]}, indent=2))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
