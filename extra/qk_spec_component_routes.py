#!/usr/bin/env python3
"""Read-only SCR-0..4 audit for spec-decode component routes.

This script does not run kernels or route SPEC_DECODE. It converts the
TBF/SDB evidence into a component-route ledger so implementation cannot start
without a candidate that changes the measured component ratios.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench" / "qk-spec-component-routes"


def read_json(rel: str) -> dict[str, Any]:
  with open(ROOT / rel) as f:
    return json.load(f)


def write_json(name: str, obj: dict[str, Any]) -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  with open(OUT / name, "w") as f:
    json.dump(obj, f, indent=2)
    f.write("\n")


def row_by_component(component_audit: dict[str, Any]) -> dict[str, dict[str, Any]]:
  return {r["component"]: r for r in component_audit["rows"]}


def candidate_inventory(rows: dict[str, dict[str, Any]], sdb2: dict[str, Any]) -> dict[str, Any]:
  return {
    "schema": "spec_component_route_candidates_v1",
    "phase": "SCR-0",
    "status": "PASS_INVENTORY_BUILT",
    "current_gate_state": {
      "verify_T5_x_one_pass": sdb2["current_T5_ms"] / sdb2["one_pass_ms"],
      "required_T5_x_one_pass": "<=1.3-1.5",
      "q4k_T5_over_T1": rows["q4k_gemm"]["T5_over_T1"],
      "q6k_lm_head_T5_over_T1": rows["q6k_lm_head"]["T5_over_T1"],
      "attention_reduces_T5_over_T1": rows["attention_reduces"]["T5_over_T1"],
      "linears_group_T5_over_T1": rows["linears_group"]["T5_over_T1"],
    },
    "candidates": [
      {
        "id": "L1_native_short_t_mmvq_q4_q6",
        "class": "grouped_short_block_quantized_linears",
        "status": "PROJECT_LEVEL",
        "gate": "Q4_K and Q6_K/lm_head samples both <=1.5x T1-equivalent at T=5",
        "evidence": [
          "Q4_K T>1 batched path already exists and is 2.916x T1 at T=5",
          "Q6_K/lm_head T>1 batched path already exists and is 5.831x T1 at T=5",
          "linears group is 3.523x T1 at T=5",
        ],
        "risk": "same bounded schedule family across Q4_K and Q6_K is not present",
      },
      {
        "id": "L2_artifact_import_quantized_small_t",
        "class": "grouped_short_block_quantized_linears",
        "status": "NO_KNOWN_ARTIFACT",
        "gate": "standalone exact Q4_K/Q6_K T=5 kernels clear <=1.5x component gates",
        "evidence": [
          "Tensile-like mature artifacts exist for fp16 GEMM, not GGUF Q4_K/Q6_K MMVQ",
          "no local extracted artifact family covers quantized GGUF small-T verify",
        ],
        "risk": "would be an external kernel-family project, not a local route candidate",
      },
      {
        "id": "L3_renderer_grouped_short_t",
        "class": "grouped_short_block_quantized_linears",
        "status": "PROJECT_LEVEL",
        "gate": "native renderer emits a grouped short-T schedule for Q4_K and Q6_K",
        "evidence": [
          "current generated batched kernels fail gates despite existing dequant reuse",
          "native route needs scheduler/register ownership beyond a single UOp knob",
        ],
        "risk": "AMD scheduler/codegen project",
      },
      {
        "id": "A1_flash_decode_short_t_generalization",
        "class": "short_block_causal_verify_attention",
        "status": "PROJECT_LEVEL",
        "gate": "T=5 attention/reduces <=1.5x T1-equivalent, exact vs current verify",
        "evidence": [
          "current flash-decode path is selected only for T==1",
          "existing flash kernels assume one query vector and output one query result",
          "T>1 verify needs proposed-block KV plus lower-right causal masking",
        ],
        "risk": "new short-block attention kernel family",
      },
      {
        "id": "A2_mini_flash_short_block",
        "class": "short_block_causal_verify_attention",
        "status": "PROJECT_LEVEL",
        "gate": "standalone T=5 short-block attention/reduces clears <=1.5x",
        "evidence": [
          "attention/reduces is 3.061x T1 at T=5",
          "attention alone is not enough for full verify, but it is required",
        ],
        "risk": "new semantics and temporary KV overlay",
      },
      {
        "id": "A3_sdpa_layout_fix",
        "class": "short_block_causal_verify_attention",
        "status": "REFUTED_OR_INSUFFICIENT",
        "gate": "existing SDPA/reduce path drops from 3.061x to <=1.5x",
        "evidence": [
          "current path already falls off T==1 flash-decode specialization",
          "layout/graph fixes do not create the missing short-block flash primitive",
        ],
        "risk": "likely preserves the global reread/reduce pattern",
      },
      {
        "id": "C_combined_shortblock_verify",
        "class": "combined_linears_attention_verify",
        "status": "BLOCKED",
        "gate": "projected T=5 verify <=1.5x one pass using measured component timings",
        "evidence": [
          "Candidate L has no bounded proof surface",
          "Candidate A has no bounded proof surface",
          "single-component fixes are already insufficient by SDB-2",
        ],
        "risk": "requires both component families to stop being T-linear",
      },
    ],
  }


def attention_audit(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
  return {
    "schema": "spec_component_route_attention_audit_v1",
    "phase": "SCR-1",
    "status": "NO_BOUNDED_ATTENTION_PROOF_SURFACE",
    "component": {
      "name": "attention_reduces",
      "T5_over_T1": rows["attention_reduces"]["T5_over_T1"],
      "gate": rows["attention_reduces"]["target_gate"],
      "passes_now": rows["attention_reduces"]["passes_gate_now"],
    },
    "current_flash_decode_assumptions": [
      "normal model selection uses flash decode only when T==1",
      "score/probability/reduce kernels are organized around one query position",
      "outputs are one-query attention results, not [T, heads, dim]",
      "no proposed-block KV overlay or lower-right causal mask is represented",
    ],
    "required_new_semantics": [
      "process T=K+1 target queries as a short block",
      "attend over prefix KV plus proposed K/V produced inside verify",
      "apply intra-block causal lower-right mask",
      "preserve GQA and exact greedy target predictions",
      "avoid changing normal T==1 decode attention",
    ],
    "candidate_verdicts": {
      "A1_flash_decode_short_t_generalization": "PROJECT_LEVEL",
      "A2_mini_flash_short_block": "PROJECT_LEVEL",
      "A3_sdpa_layout_fix": "REFUTED_OR_INSUFFICIENT",
    },
    "gate_result": "FAIL",
    "next_action": "Do not implement attention TBF-3 unless a standalone T=5 attention-only proof is introduced.",
  }


def linears_audit(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
  return {
    "schema": "spec_component_route_linears_audit_v1",
    "phase": "SCR-2",
    "status": "NO_BOUNDED_GROUPED_LINEAR_PROOF_SURFACE",
    "components": {
      "q4k_gemm": {
        "T5_over_T1": rows["q4k_gemm"]["T5_over_T1"],
        "gate": rows["q4k_gemm"]["target_gate"],
        "passes_now": rows["q4k_gemm"]["passes_gate_now"],
      },
      "q6k_lm_head": {
        "T5_over_T1": rows["q6k_lm_head"]["T5_over_T1"],
        "gate": rows["q6k_lm_head"]["target_gate"],
        "passes_now": rows["q6k_lm_head"]["passes_gate_now"],
      },
      "linears_group": {
        "T5_over_T1": rows["linears_group"]["T5_over_T1"],
        "gate": rows["linears_group"]["target_gate"],
        "passes_now": rows["linears_group"]["passes_gate_now"],
      },
    },
    "existing_paths": [
      "Q4_K has a T>1 q4k_gemm_kernel path",
      "Q6_K has a T>1 q6k_gemm_kernel path",
      "both paths already use a short-T column axis/upcast form to expose dequant reuse",
    ],
    "why_not_bounded": [
      "Q4_K-only improvement is insufficient by scope kill rule",
      "Q6_K/lm_head is the most T-linear measured component at 5.831x",
      "no single existing schedule knob covers both Q4_K and Q6_K roles",
      "q8 activation lifecycle is explicitly out of scope for this T-cheap verify proof",
    ],
    "candidate_verdicts": {
      "L1_native_short_t_mmvq_q4_q6": "PROJECT_LEVEL",
      "L2_artifact_import_quantized_small_t": "NO_KNOWN_ARTIFACT",
      "L3_renderer_grouped_short_t": "PROJECT_LEVEL",
    },
    "gate_result": "FAIL",
    "next_action": "Do not implement grouped-linears TBF-3 unless both Q4_K and Q6_K samples clear the component gate.",
  }


def projection(rows: dict[str, dict[str, Any]], sdb2: dict[str, Any], attention: dict[str, Any], linears: dict[str, Any]) -> dict[str, Any]:
  component_ms = {c["component"]: c["real_ms_at_T5_directional"] for c in sdb2["components"]}
  single_component = {c["component"]: c["single_component_T_independent_would_meet_1p5x"] for c in sdb2["components"]}
  needed_cut_ms = sdb2["needed_cut_ms"]
  available_single_cut_ms = {
    "q4k_gemm": component_ms["q4k_gemm"],
    "q6k_gemm_lm_head": component_ms["q6k_gemm_lm_head"],
    "attention_reduces": component_ms["attention_reduces"],
    "elementwise_norm": component_ms["elementwise_norm"],
  }
  return {
    "schema": "spec_component_route_projection_v1",
    "phase": "SCR-3",
    "status": "FAIL_NO_PASSING_COMPONENT_CEILINGS",
    "full_verify_gate": {
      "current_T5_ms": sdb2["current_T5_ms"],
      "one_pass_ms": sdb2["one_pass_ms"],
      "current_T5_x_one_pass": sdb2["current_T5_ms"] / sdb2["one_pass_ms"],
      "target_T5_ms_for_1p5x_one_pass": sdb2["target_T5_ms_for_1p5x_one_pass"],
      "needed_cut_ms": needed_cut_ms,
      "needed_cut_fraction": sdb2["needed_cut_fraction"],
    },
    "component_gates": {
      "q4k_gemm_T5_over_T1": rows["q4k_gemm"]["T5_over_T1"],
      "q6k_lm_head_T5_over_T1": rows["q6k_lm_head"]["T5_over_T1"],
      "attention_reduces_T5_over_T1": rows["attention_reduces"]["T5_over_T1"],
      "linears_group_T5_over_T1": rows["linears_group"]["T5_over_T1"],
      "required": "<=1.5x T1-equivalent for each component before implementation",
    },
    "single_component_sdb2_check": single_component,
    "directional_T5_ms_by_component": component_ms,
    "max_possible_single_component_cut_ms_if_free": available_single_cut_ms,
    "combined_candidate_inputs": {
      "attention_status": attention["status"],
      "linears_status": linears["status"],
    },
    "projection_result": "BLOCKED_NO_CANDIDATE_CEILINGS",
    "why": [
      "SCR-1 did not produce a bounded attention ceiling",
      "SCR-2 did not produce a bounded grouped-linear ceiling",
      "SDB-2 already showed no single component is sufficient",
      "therefore Candidate C cannot be projected below the <=1.5x full-verify gate",
    ],
  }


def result(inv: dict[str, Any], attention: dict[str, Any], linears: dict[str, Any], proj: dict[str, Any]) -> dict[str, Any]:
  return {
    "schema": "spec_component_route_scr0_scr4_result_v1",
    "scope": "docs/spec-decode-component-route-candidates-scope-20260619.md",
    "SCR_0": {"status": inv["status"], "artifact": "bench/qk-spec-component-routes/candidates.json"},
    "SCR_1": {"status": attention["status"], "artifact": "bench/qk-spec-component-routes/attention_audit.json"},
    "SCR_2": {"status": linears["status"], "artifact": "bench/qk-spec-component-routes/linears_audit.json"},
    "SCR_3": {"status": proj["status"], "artifact": "bench/qk-spec-component-routes/projection.json"},
    "SCR_4": {
      "decision": "PROJECT_LEVEL_CLOSE",
      "implementation": "DO_NOT_BUILD_TBF_3",
      "reason": "both candidate components lack a bounded proof surface, and no measured component ceiling can be projected to the full verify gate",
    },
    "final_verdict": {
      "status": "PROJECT_LEVEL_CLOSE",
      "normal_decode_changed": False,
      "prefill_touched": False,
      "spec_decode_routed": False,
      "next_reopen_condition": "new concrete component route with measured T=5 attention or Q4_K+Q6_K samples <=1.5x T1-equivalent",
    },
  }


def write_summary(res: dict[str, Any], proj: dict[str, Any]) -> None:
  lines = [
    "# Spec component routes SCR-0..4",
    "",
    f"Final verdict: `{res['final_verdict']['status']}`.",
    "",
    "No TBF-3 implementation is earned. Attention generalization and grouped short-T linears both require new kernel families/project-level scheduler work.",
    "",
    "## Gate state",
    "",
    f"- current T=5 verify: `{proj['full_verify_gate']['current_T5_x_one_pass']:.3f}x` one pass",
    "- required: `<=1.3-1.5x` one pass",
    f"- needed cut: `{proj['full_verify_gate']['needed_cut_fraction']:.3f}`",
    "- Q4_K: `2.916x`; Q6_K/lm_head: `5.831x`; attention/reduces: `3.061x`; linears group: `3.523x`",
    "",
    "## Decision",
    "",
    "`PROJECT_LEVEL_CLOSE`: reopen only with a measured component candidate, not by starting implementation from the current baseline.",
    "",
  ]
  with open(OUT / "summary.md", "w") as f:
    f.write("\n".join(lines))


def main() -> None:
  component_audit = read_json("bench/qk-spec-tcheap-forward/component_audit.json")
  sdb_model = read_json("bench/qk-spec-decode-bandwidth-amortization/model.json")
  sdb2 = sdb_model["sdb2_verify_design_audit"]
  rows = row_by_component(component_audit)

  inv = candidate_inventory(rows, sdb2)
  attn = attention_audit(rows)
  lin = linears_audit(rows)
  proj = projection(rows, sdb2, attn, lin)
  res = result(inv, attn, lin, proj)

  write_json("candidates.json", inv)
  write_json("attention_audit.json", attn)
  write_json("linears_audit.json", lin)
  write_json("projection.json", proj)
  write_json("result.json", res)
  write_summary(res, proj)
  print(json.dumps({"status": res["final_verdict"]["status"], "out": str(OUT)}, indent=2))


if __name__ == "__main__":
  main()
