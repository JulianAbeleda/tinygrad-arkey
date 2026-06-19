#!/usr/bin/env python3
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-integration-diagnostic"


def load(rel):
  with open(ROOT / rel) as f:
    return json.load(f)


def speedup_for_bw(current_pct, target_pct, weight_share):
  """E2E decode speedup if only the weight-GEMV bucket changes."""
  return 1.0 / (weight_share * (current_pct / target_pct) + (1.0 - weight_share))


def main():
  OUT.mkdir(parents=True, exist_ok=True)

  fmi = load("bench/qk-decode-fused-mmvq-integration/result.json")
  launch = load("bench/qk-decode-fused-mmvq-integration/launch_contract_diff.json")
  b1 = load("bench/qk-decode-fused-mmvq-integration/fmi4_b1_knob_probe.json")
  coop = load("bench/qk-mmvq-fused-coop-row/baseline.json")
  q8 = load("bench/qk-q8-lifecycle/reuse_map.json")
  pmu = load("bench/qk-primitive-pmu-atlas/result.json")

  current_hbm_pct = 44.0
  llama_inmodel_hbm_pct = 54.0
  standalone_hbm_pct = 76.0
  weight_share = 0.85

  stage2 = coop["stage2_decomposition"]
  stage2_tax = {
    "name": "global partials plus stage2 reduce",
    "status": "MEASURED_BUT_INSUFFICIENT",
    "evidence": {
      "role": coop["role"],
      "partial_alone_us": stage2["partial_kernel_alone_us"],
      "partial_plus_stage2_us": stage2["partial_plus_stage2_us"],
      "stage2_sum_cost_us": stage2["stage2_sum_cost_us"],
      "stage2_pct_of_total": stage2["stage2_pct_of_total"],
      "pct_peak_partial_plus_stage2": stage2["partial_plus_stage2_pct"],
      "pct_peak_partial_alone": stage2["partial_kernel_alone_pct"],
      "best_case_if_removed_pct_peak": "~53-54",
    },
    "interpretation": "This is the closest decode analogue to prefill's layout tax, but it only explains about 10% on one Q4_K role and still misses llama-class bandwidth.",
    "decision": "closed_as_standalone_route",
  }

  q8_tax = {
    "name": "activation q8 lifecycle not amortized like llama",
    "status": "SECONDARY_LOSSY_CODEGEN_WALLED",
    "evidence": {
      "max_q4k_activation_reuse": q8["max_amortization_for_q4k_int_dot"],
      "note": q8["note"],
      "best_activation": "FFN norm/input -> gate+up",
      "expected_decode_ev": "~3-6% from prior q8 lifecycle scopes",
    },
    "interpretation": "llama pays activation quant once and reuses it across input-sharing MMVQs; tinygrad's native producer remains blocked and the proven route is lossy/research-only.",
    "decision": "keep_as_secondary_research_flag_unless_inmodel_q8_replay_passes",
  }

  env_knobs = {
    "name": "existing tinygrad launch-shape env knobs",
    "status": b1["status"],
    "evidence": {
      "best_by_role": b1["best_by_role"],
      "passing_rows": len(b1["passing_rows"]),
      "gate": b1["gate"],
    },
    "interpretation": "The available Q4K/Q6K row-tile and coop toggles do not preserve the standalone fast contract in-model.",
    "decision": "closed_for_b1",
  }

  launch_tax = {
    "name": "MMVQ launch/occupancy contract not preserved in-model",
    "status": "LIVE_PROJECT_LEVEL",
    "evidence": {
      "tinygrad_standalone_hbm_pct": standalone_hbm_pct,
      "tinygrad_inmodel_weight_gemv_hbm_pct": current_hbm_pct,
      "llama_standalone_hbm_pct": 57.0,
      "llama_inmodel_weight_gemv_hbm_pct": llama_inmodel_hbm_pct,
      "llama_dominant_contract": launch["llama_contract"]["summary"]["dominant_contract"],
      "diff_axes": launch["diff"],
    },
    "interpretation": "This is the large remaining transfer loss. It is not exposed by current env knobs and is larger than the measured stage2 tax.",
    "decision": "requires_runtime_cache_identity_diagnostic_then_renderer_scheduler_or_artifact_import",
  }

  potential = {
    "schema": "decode_integration_potential_v1",
    "assumptions": {
      "weight_gemv_gpu_share": weight_share,
      "current_tinygrad_inmodel_weight_hbm_pct": current_hbm_pct,
      "formula": "1 / (weight_share * current_pct / target_pct + (1 - weight_share))",
      "classification": "estimates_from_measured_bw_authorities",
    },
    "targets": [
      {
        "target": "llama-like in-model retention",
        "target_hbm_pct": llama_inmodel_hbm_pct,
        "decode_speedup": round(speedup_for_bw(current_hbm_pct, llama_inmodel_hbm_pct, weight_share), 3),
        "meaning": "close tinygrad 44% -> llama ~54% over the weight-GEMV bucket",
      },
      {
        "target": "moderate above-llama retention",
        "target_hbm_pct": 60.0,
        "decode_speedup": round(speedup_for_bw(current_hbm_pct, 60.0, weight_share), 3),
        "meaning": "requires better-than-llama in-model bandwidth, still below tinygrad standalone",
      },
      {
        "target": "full standalone transfer",
        "target_hbm_pct": standalone_hbm_pct,
        "decode_speedup": round(speedup_for_bw(current_hbm_pct, standalone_hbm_pct, weight_share), 3),
        "meaning": "theoretical upper bound if the 76% standalone GEMV behavior survived the whole model",
        "realism": "not earned by current evidence",
      },
    ],
    "small_levers": [
      {
        "lever": "remove stage2 reduce on Q4_K coop roles",
        "isolated_effect": "48% -> ~53-54% on the measured Q4_K ffn_gate/up micro-surface",
        "e2e_interpretation": "low single digits; already closed as below the route gate",
      },
      {
        "lever": "q8 gate/up lifecycle",
        "expected_decode_effect": "~3-6%",
        "caveat": "lossy plus native producer/codegen wall",
      },
    ],
  }

  tax_ledger = {
    "schema": "decode_integration_tax_ledger_v1",
    "source_docs": [
      "docs/inference-perf-measured-map-20260619.md",
      "docs/decode-fused-mmvq-integration-fmi1-fmi2-result-20260619.md",
      "docs/decode-fused-mmvq-integration-fmi4-b1-result-20260619.md",
      "docs/qk-mmvq-fused-coop-row-verdict-20260618.md",
      "docs/qk-q8-activation-lifecycle-verdict-20260618.md",
    ],
    "pmu_ctx_summary": {ctx: {"bw_bound_gpu%": row["bw_bound_gpu%"], "cache_served_gpu%": row["cache_served_gpu%"]} for ctx, row in pmu["by_ctx"].items()},
    "taxes": [stage2_tax, q8_tax, env_knobs, launch_tax],
    "decode_analog_to_prefill": {
      "prefill": "one dominant layout-conversion tax cancels the fast Tensile kernel",
      "decode": "no single clean conversion tax; the measured local stage2 tax is insufficient, q8 lifecycle is capped/lossy, and the large loss is MMVQ contract preservation in-model",
    },
  }

  result = {
    "schema": "decode_integration_diagnostic_result_v1",
    "status": "LOCALIZED_NO_SINGLE_TAX",
    "FMI_status": fmi["decision"]["status"],
    "B1_status": b1["status"],
    "verdict": "Decode does not have a prefill-style one-line transpose tax. The only measured local cancellation tax is the stage2 partial reduction, and it is too small. The large remaining loss is preserving the llama-like MMVQ lifecycle and launch/occupancy contract inside the model.",
    "next": [
      "B2 runtime/cache identity diagnostic: prove whether in-model compiled program and metadata match the intended standalone/role kernel.",
      "If B2 does not find a bounded wiring bug, choose between B3 renderer/scheduler project or B4 artifact/import.",
      "Keep q8 replay secondary because it is lossy and capped by activation reuse count 2.",
    ],
    "artifacts": {
      "tax_ledger": "bench/qk-decode-integration-diagnostic/tax_ledger.json",
      "potential": "bench/qk-decode-integration-diagnostic/potential.json",
    },
  }

  (OUT / "tax_ledger.json").write_text(json.dumps(tax_ledger, indent=2) + "\n")
  (OUT / "potential.json").write_text(json.dumps(potential, indent=2) + "\n")
  (OUT / "result.json").write_text(json.dumps(result, indent=2) + "\n")

  summary = f"""# Decode integration diagnostic summary

Status: `{result['status']}`

Decode does not mirror the prefill Tensile finding as one clean layout tax. The ledger has four parts:

- stage2 partial reduce: measured `{stage2['stage2_sum_cost_us']}us` / `{stage2['stage2_pct_of_total']}%` on the Q4_K ffn_gate/up surface; removing it only reaches `~53-54%` peak on that micro-surface.
- q8 activation lifecycle: max Q4_K activation reuse is `{q8['max_amortization_for_q4k_int_dot']}`; useful but lossy and native-producer-walled.
- existing env knobs: `{b1['status']}`, no passing rows.
- MMVQ contract preservation: tinygrad `76%` standalone -> `44%` in-model, while llama `57%` -> `54%`.

Potential model:

- `44 -> 54%` over the weight-GEMV bucket: `{potential['targets'][0]['decode_speedup']}x` decode.
- `44 -> 60%`: `{potential['targets'][1]['decode_speedup']}x`.
- `44 -> 76%`: `{potential['targets'][2]['decode_speedup']}x` theoretical, not earned by current evidence.
"""
  (OUT / "summary.md").write_text(summary)


if __name__ == "__main__":
  main()
