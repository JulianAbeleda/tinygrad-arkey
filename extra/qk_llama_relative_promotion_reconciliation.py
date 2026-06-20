#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-llama-promotion"


def read_json(rel: str, default: Any = None) -> Any:
  path = ROOT / rel
  if not path.exists(): return default
  return json.loads(path.read_text())


def write_json(name: str, data: Any) -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main() -> int:
  ptm_scope = read_json("bench/qk-tensile-primitive-transfer/scope.json", {})
  tensile_shape = read_json("bench/qk-tensile-extraction/shape_matrix.json", {})
  p8_timing = read_json("bench/amd-broad-backend-roadmap/bb5a10_p8_timing_authority_reconciliation_result.json", {})
  decode_ready = read_json("bench/qk-decode-native-tooling/readiness.json", {})
  coverage = read_json("bench/qk-primitive-coverage/rows.json", {})
  runtime = read_json("bench/qk-decode-runtime-overhead/result.json", {})
  q8_promotion = read_json("bench/q8-ffn-artifact-promotion/promotion_result.json", {})

  ffn_gate_up = next((r for r in tensile_shape.get("rows", []) if r.get("role") == "ffn_gate_up"), {})
  ffn_down = next((r for r in tensile_shape.get("rows", []) if r.get("role") == "ffn_down"), {})
  attn_qo = next((r for r in tensile_shape.get("rows", []) if r.get("role") == "attn_q_o"), {})
  coverage_by_id = {r.get("id"): r for r in coverage.get("rows", [])}

  promotion_rows = [
    {
      "phase": "prefill",
      "candidate": "concrete_kv_wmma_dependency_free_baseline",
      "status": "PROMOTED_BASELINE",
      "llama_relative": "47-49% llama pp512, byte-identical",
      "evidence": "concrete-KV WMMA 1449-1515 tok/s vs llama ~3070 tok/s",
      "promotion_action": "Keep as dependency-free default/baseline; do not claim llama parity.",
      "blocker_or_policy": None,
      "sources": [
        "docs/prefill-tensile-DEFINITIVE-source-of-truth-20260619.md",
        "docs/prefill-clock-controlled-benchmark-result-20260619.md",
      ],
    },
    {
      "phase": "prefill",
      "candidate": "external_tensile_ffn_plus_qo_route",
      "status": "PROMOTE_IF_DEPENDENCY_POLICY_ACCEPTED",
      "llama_relative": "86-87% llama pp512, byte-identical",
      "evidence": {
        "model_tok_s": "2636-2673 in source docs",
        "gate_up_tflops": ffn_gate_up.get("median_tflops"),
        "down_tflops": ffn_down.get("median_tflops"),
        "attn_qo_tflops": attn_qo.get("median_tflops"),
      },
      "promotion_action": "Decision is policy, not measurement: accept vendored rocBLAS Tensile .co with fallback/provenance, or keep off.",
      "blocker_or_policy": "external code-object dependency and shape/route coverage policy",
      "sources": [
        "bench/qk-tensile-extraction/shape_matrix.json",
        "docs/prefill-tensile-DEFINITIVE-source-of-truth-20260619.md",
      ],
    },
    {
      "phase": "prefill",
      "candidate": "native_tensile_class_codegen_transfer",
      "status": "BLOCKED_NATIVE_PROJECT",
      "llama_relative": "target would be >=60 TFLOPS / toward 86-87% llama, but current P8 candidates are ~18-21 TFLOPS",
      "evidence": p8_timing.get("performance_summary", {}),
      "promotion_action": "Do not promote; first run PTM-1 same-harness bridge, then choose exactly one native row.",
      "blocker_or_policy": "same-harness authority bridge, then K-loop/resource/timing decision",
      "sources": [
        "bench/amd-broad-backend-roadmap/bb5a10_p8_timing_authority_reconciliation_result.json",
        "bench/qk-tensile-primitive-transfer/scope.json",
      ],
    },
    {
      "phase": "decode",
      "candidate": "banked_default_decode_stack",
      "status": "PROMOTED_DEFAULT",
      "llama_relative": "68.2/66.4/60.7 tok/s at ctx512/1024/4096, ~67% llama, W==D, GPU-bound",
      "evidence": {
        "decode_runtime_artifact": bool(runtime),
        "host_sync": "0.0% in reproduced W==D doc",
      },
      "promotion_action": "Keep as the llama-relative default decode baseline and measurement authority.",
      "blocker_or_policy": None,
      "sources": [
        "docs/qk-decode-banked-reproduce-20260618.md",
        "docs/qk-8b-decode-banked-20260617.md",
      ],
    },
    {
      "phase": "decode",
      "candidate": "q8_ffn_handwritten_artifact_route",
      "status": "HARDENED_OPT_IN_CANDIDATE" if q8_promotion.get("verdict") == "PASS_Q8_FFN_ARTIFACT_PROMOTION_TO_HARDENED_OPT_IN" else "RESEARCH_FLAG_ONLY",
      "llama_relative": "~3-6% decode upside, W==D +5.1-6.3%, dNLL +0.002887",
      "evidence": {"coverage": coverage_by_id.get("decode_q8_artifact_lifecycle", {}), "promotion": q8_promotion},
      "promotion_action": "Promote to hardened opt-in behind Q8_FFN_HANDWRITTEN=1; keep default-off unless a maintainer/user explicitly accepts lossy external-artifact default behavior."
                          if q8_promotion.get("verdict") == "PASS_Q8_FFN_ARTIFACT_PROMOTION_TO_HARDENED_OPT_IN" else
                          "Can remain/default as research oracle; do not promote to default without broader quality/policy acceptance.",
      "blocker_or_policy": "default-on remains a policy decision because the route is lossy and externally owned" if q8_promotion.get("verdict") == "PASS_Q8_FFN_ARTIFACT_PROMOTION_TO_HARDENED_OPT_IN" else
                           "lossy q8 policy and limited reuse_count=2 Amdahl",
      "sources": ["bench/qk-primitive-coverage/rows.json", "bench/q8-ffn-artifact-promotion/promotion_result.json"],
    },
    {
      "phase": "decode",
      "candidate": "native_q8_scheduler_renderer",
      "status": "BLOCKED_ROADMAP_ONLY",
      "llama_relative": "unknown; no >=30us attributed feature, max movement remains below gate",
      "evidence": {
        "readiness_verdict": decode_ready.get("verdict"),
        "start_gate": decode_ready.get("start_gate"),
      },
      "promotion_action": "Do not start N2/native q8 work from Tensile or prefill evidence.",
      "blocker_or_policy": "needs >=30us q8 same-binary timing movement and W==D projection",
      "sources": [
        "bench/qk-decode-native-tooling/readiness.json",
        "docs/decode-native-tooling-pass-result-20260619.md",
      ],
    },
    {
      "phase": "decode",
      "candidate": "mmvq_contract_preservation_or_source_import",
      "status": "PROJECT_LEVEL_OPTION",
      "llama_relative": "large path targets tinygrad ~44% -> llama-like ~54% over weight-GEMV bucket, ~1.187x decode potential",
      "evidence": coverage_by_id.get("decode_mmvq_contract_preservation", {}),
      "promotion_action": "Promote only as funded project-level path, not bounded diagnostic work.",
      "blocker_or_policy": "renderer/scheduler or source-contract import acceptance",
      "sources": ["bench/qk-primitive-coverage/rows.json"],
    },
    {
      "phase": "attention_decode",
      "candidate": "q4k_attn_qo_coop_and_flash_decode_stack",
      "status": "PROMOTED_DEFAULT",
      "llama_relative": "part of reproduced ~67% llama decode line; attention gains are already in default stack",
      "evidence": "Q4_K attn_q/o coop and FLASH_VARIANT=gqa_coop_vec/threshold 512 banked in decode docs",
      "promotion_action": "Keep promoted; use W==D decode harness for any regression/promotion.",
      "blocker_or_policy": None,
      "sources": [
        "docs/qk-8b-decode-banked-20260617.md",
        "docs/qk-decode-banked-reproduce-20260618.md",
      ],
    },
    {
      "phase": "attention_decode",
      "candidate": "decode_attention_v3_wmma_gqa_v_reuse",
      "status": "DEEP_CODEGEN_CANDIDATE",
      "llama_relative": "projected +4-10% at ctx<=1024 and +12-36% at ctx4096, not yet implemented",
      "evidence": "selected in decode-bank doc as hard target after flash variant arc",
      "promotion_action": "Do not promote; scope only if taking deep codegen risk.",
      "blocker_or_policy": "WMMA convention/codegen wall and in-model byte-identical validation",
      "sources": ["docs/qk-8b-decode-banked-20260617.md"],
    },
    {
      "phase": "attention_prefill",
      "candidate": "score_free_flash_prefill_or_reuse_free_attention",
      "status": "CLOSED_NOT_PROMOTE",
      "llama_relative": "correct but slower than SDPA / no llama-relative promotion",
      "evidence": coverage_by_id.get("long_context_kv_attention_lifecycle", {}),
      "promotion_action": "Do not reopen without a new tiled/locality primitive and accepted long-context target.",
      "blocker_or_policy": "reuse-free memory path; long-context target not current baseline",
      "sources": [
        "bench/qk-flash-prefill-phase5/result.json",
        "bench/qk-primitive-coverage/rows.json",
      ],
    },
    {
      "phase": "spec_decode",
      "candidate": "tcheap_verify_forward",
      "status": "CLOSED_NOT_PROMOTE",
      "llama_relative": "current T=5 verify loses; no llama-relative promotion",
      "evidence": coverage_by_id.get("spec_decode_tcheap_batched_forward", {}),
      "promotion_action": "Do not reopen without a concrete component route for grouped linears or short-block attention.",
      "blocker_or_policy": "verify <=1.5x one-pass gate not met",
      "sources": ["bench/qk-primitive-coverage/rows.json"],
    },
  ]

  promote_now = [r["candidate"] for r in promotion_rows if r["status"] in {"PROMOTED_BASELINE", "PROMOTED_DEFAULT"}]
  policy_ready = [r["candidate"] for r in promotion_rows if r["status"] in {"PROMOTE_IF_DEPENDENCY_POLICY_ACCEPTED", "HARDENED_OPT_IN_CANDIDATE"}]
  blocked = [r["candidate"] for r in promotion_rows if "BLOCKED" in r["status"] or r["status"].startswith("CLOSED")]
  gates = {
    "ptm_scope_pass": ptm_scope.get("verdict") == "PASS_TENSILE_PRIMITIVE_TRANSFER_MATRIX_SCOPED",
    "p8_timing_reconciled": p8_timing.get("verdict") == "PASS_BB5A10_P8_TIMING_AUTHORITY_RECONCILED_SAME_HARNESS_REQUIRED",
    "primitive_coverage_present": bool(coverage.get("rows")),
    "promotion_rows_have_sources": all(bool(r.get("sources")) for r in promotion_rows),
    "has_prefill_decode_attention_rows": {"prefill", "decode", "attention_decode", "attention_prefill"}.issubset({r["phase"] for r in promotion_rows}),
  }
  gate_pass = all(gates.values())
  result = {
    "date": "2026-06-20",
    "schema": "qk_llama_relative_promotion_reconciliation_v1",
    "phase": "llama_promotion_reconciliation",
    "verdict": "PASS_LLAMA_RELATIVE_PROMOTION_RECONCILIATION" if gate_pass else "BLOCKED_LLAMA_RELATIVE_PROMOTION_RECONCILIATION",
    "gate_pass": gate_pass,
    "default_behavior_changed": False,
    "performance_claim": False,
    "promotion_rows": promotion_rows,
    "summary": {
      "promote_or_keep_promoted_now": promote_now,
      "policy_ready": policy_ready,
      "blocked_or_closed": blocked,
      "next_required_gate_for_native_tensile": "PTM-1 same-harness authority bridge",
      "next_required_gate_for_decode_transfer": ">=30us q8 same-binary timing movement plus W==D quality",
    },
    "gates": gates,
    "next_action": "Use this reconciliation to choose policy-ready external Tensile prefill, project-level decode MMVQ, or PTM-1 native prefill bridge; do not conflate them.",
  }
  write_json("reconciliation.json", result)
  print(json.dumps({
    "out": "bench/qk-llama-promotion/reconciliation.json",
    "verdict": result["verdict"],
    "gate_pass": gate_pass,
    "rows": len(promotion_rows),
    "policy_ready": policy_ready,
    "next": result["next_action"],
  }, indent=2))
  return 0 if gate_pass else 1


if __name__ == "__main__":
  raise SystemExit(main())
