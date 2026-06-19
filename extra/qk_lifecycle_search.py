#!/usr/bin/env python3
"""Read-only primitive lifecycle search ledger.

This sits above kernel search. It does not generate kernels, run hardware, or
route model defaults. It records lifecycle candidates: producer format,
consumer primitive, routing boundary, quality policy, and refutations.

Run:
  PYTHONPATH=. python3 extra/qk_lifecycle_search.py
"""
from __future__ import annotations

import datetime, json, pathlib, subprocess
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench"
OUT = BENCH / "qk-lifecycle-search"

VALID_STATES = {
  "proposed", "diagnostic", "open", "pass_research", "pass_strong_policy_gated",
  "shipped", "refuted", "closed", "deferred", "policy_bound", "project_level",
}


def _read_json(rel:str) -> dict[str, Any]:
  try:
    obj = json.loads((ROOT / rel).read_text())
    return obj if isinstance(obj, dict) else {}
  except Exception:
    return {}


def _git_commit() -> str:
  try:
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                          check=False, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL, timeout=10).stdout.strip() or "unknown"
  except Exception:
    return "unknown"


def _speedup_by_ctx() -> dict[str, float]:
  base = _read_json("bench/q8-ffn-handwritten-oracle/decode_wd_baseline.json")
  route = _read_json("bench/q8-ffn-handwritten-oracle/decode_wd_q8_route.json")
  out: dict[str, float] = {}
  brows = {int(r["ctx"]): float(r["tok_s_W"]) for r in base.get("rows", []) if "ctx" in r and "tok_s_W" in r}
  for row in route.get("rows", []):
    ctx = int(row.get("ctx", -1))
    if ctx in brows and brows[ctx] > 0:
      out[str(ctx)] = round(float(row["tok_s_W"]) / brows[ctx], 4)
  return out


def _dnll_delta() -> float | None:
  base = _read_json("bench/q8-ffn-handwritten-oracle/nll_baseline.json")
  route = _read_json("bench/q8-ffn-handwritten-oracle/nll_q8_route.json")
  if isinstance(base.get("nll"), (int, float)) and isinstance(route.get("nll"), (int, float)):
    return round(float(route["nll"]) - float(base["nll"]), 6)
  return None


def _candidate(**kw: Any) -> dict[str, Any]:
  if kw["state"] not in VALID_STATES:
    raise ValueError(f"bad lifecycle state {kw['state']!r}")
  return {
    "id": kw["id"],
    "phase": kw["phase"],
    "state": kw["state"],
    "role": kw["role"],
    "producer": kw["producer"],
    "format": kw["format"],
    "consumer": kw["consumer"],
    "routing": kw["routing"],
    "quality": kw["quality"],
    "score": kw["score"],
    "evidence": kw["evidence"],
    "blocked_by_refutations": kw.get("blocked_by_refutations", []),
    "next_action": kw["next_action"],
  }


REFUTATION_MEMORY: list[dict[str, Any]] = [
  {
    "id": "separate_q8_pack_wall",
    "applies_to": ["decode_q8_separate_pack"],
    "prunes_patterns": ["q8 activation produced in a separate pack kernel", "dot4-only decode win"],
    "reason": "The activation producer cost erases the native dot4/MMVQ consumer gain at reuse_count=2.",
    "evidence": ["docs/qk-q8-activation-lifecycle-verdict-20260618.md",
                 "docs/q8-mmvq-lifecycle-deep-result-20260619.md"],
  },
  {
    "id": "native_q8_bounded_feature_absent",
    "applies_to": ["decode_q8_native_codegen"],
    "prunes_patterns": ["single bounded AMD DSL tweak for q8 decode", "COMGR source reshuffle"],
    "reason": "The passing artifact route is concrete, but tinygrad-owned COMGR/ASM variants miss the schedule and no bounded A2 feature clears the gate.",
    "evidence": ["docs/q8-ffn-route-a-scheduler-codegen-result-20260619.md",
                 "docs/q8-ffn-route-a-pmu-sqtt-evidence-result-20260619.md"],
  },
  {
    "id": "spec_verify_distributed_t_scaling",
    "applies_to": ["decode_spec_verify_shortcut"],
    "prunes_patterns": ["single q4k verify kernel", "spec decode as one-kernel shortcut"],
    "reason": "Verify T-scaling is distributed across attention, Q4_K, Q6_K, and fast-path loss.",
    "evidence": ["docs/qk-spec-verify-component-breakdown-20260618.md"],
  },
  {
    "id": "spec_current_verify_not_amortized",
    "applies_to": ["decode_spec_weight_amortization_lifecycle"],
    "prunes_patterns": ["current T>1 verify path", "spec decode without T-cheap target verify"],
    "reason": "The current T=5 verify costs 4.66x one T=1 pass, so spec only reopens if the verify lifecycle changes.",
    "evidence": ["docs/qk-spec-verify-component-breakdown-20260618.md",
                 "docs/spec-decode-bandwidth-amortization-scope-20260619.md"],
  },
  {
    "id": "bounded_wmma_plateau",
    "applies_to": ["prefill_tensile_codegen_transfer"],
    "prunes_patterns": ["bounded WMMA knob sweep", "LDS-only prefill matmul row"],
    "reason": "Pure tinygrad bounded WMMA/LDS sweeps stay near the ~42 TFLOPS plateau; Tensile-class transfer is project-level.",
    "evidence": ["docs/prefill-own-wmma-kernel-result-20260619.md",
                 "docs/amd-schedule-codegen-exhaustion-result-20260619.md"],
  },
  {
    "id": "hip_runtime_hcq_exclusion",
    "applies_to": ["prefill_tensile_artifact_full"],
    "prunes_patterns": ["in-process rocBLAS on tinygrad pointers", "HIP runtime bridge lane A"],
    "reason": "HIP runtime and tinygrad HCQ/KFD are mutually exclusive in-process; extracted HSACO through HCQ is the viable artifact boundary.",
    "evidence": ["docs/prefill-external-bridge-ebt1-result-20260619.md",
                 "docs/prefill-tensile-tpe4-perf-result-20260619.md"],
  },
  {
    "id": "reuse_free_attention_not_a_primitive",
    "applies_to": ["prefill_attention_flash_lifecycle"],
    "prunes_patterns": ["reuse-free flash-prefill", "global-reread attention rewrite"],
    "reason": "Correct attention math without K/V locality is not a performance primitive.",
    "evidence": ["docs/qk-machine-search-primitive-rows-20260618.md"],
  },
]


RUNNER_BINDINGS: dict[str, dict[str, Any]] = {
  "decode_q8_artifact_lifecycle": {
    "schema": "primitive_lifecycle_runner_binding_v1",
    "phase": "decode",
    "authority": "W==D ctx sweep + dNLL",
    "existing_artifacts": ["bench/q8-ffn-handwritten-oracle/decode_wd_baseline.json",
                           "bench/q8-ffn-handwritten-oracle/decode_wd_q8_route.json",
                           "bench/q8-ffn-handwritten-oracle/nll_baseline.json",
                           "bench/q8-ffn-handwritten-oracle/nll_q8_route.json"],
    "runner_commands": [
      "Q8_FFN_HANDWRITTEN=1 <decode W==D harness for ctx 128/512/1024/4096>",
      "Q8_FFN_HANDWRITTEN=1 <dNLL harness over fixed token windows>",
    ],
    "promotion_gate": ">=3% sustained W==D decode speedup and dNLL <= 0.01",
    "runs_hardware": False,
  },
  "decode_q8_native_codegen": {
    "schema": "primitive_lifecycle_runner_binding_v1",
    "phase": "decode",
    "authority": "compare native route to q8 artifact authority before W==D",
    "existing_artifacts": ["bench/q8-ffn-amd-scheduler-project/route_a_result.json",
                           "bench/q8-ffn-amd-scheduler-project/pmu_sqtt_evidence.json"],
    "runner_commands": ["<native q8 lifecycle micro + W==D only after lifecycle <= artifact budget>"],
    "promotion_gate": "native lifecycle close to artifact route, then same W==D/dNLL gate",
    "runs_hardware": False,
  },
  "decode_spec_weight_amortization_lifecycle": {
    "schema": "primitive_lifecycle_runner_binding_v1",
    "phase": "decode",
    "authority": "verify-cost ladder + acceptance + W==D greedy-exact route",
    "existing_artifacts": ["bench/qk-spec-decode-acceptance/result.json",
                           "bench/qk-spec-decode-production/baseline.json",
                           "bench/qk-spec-verify-component-breakdown/result.json",
                           "bench/qk-primitive-pmu-atlas/result.json"],
    "runner_commands": [
      "<SDB-1 analytic speed model over K=2/3/4/8>",
      "<SDB-2 verify component map for T=K+1>",
      "SPEC_DECODE=1 <W==D greedy-exact decode only if verify <=1.5x one pass>",
    ],
    "promotion_gate": "T=K+1 verify <=1.5x one T==1 pass, greedy byte-exact, W==D >=1.2x",
    "runs_hardware": False,
  },
  "prefill_tensile_artifact_full": {
    "schema": "primitive_lifecycle_runner_binding_v1",
    "phase": "prefill",
    "authority": "warm pp512/pp1024 + dNLL + fallback",
    "existing_artifacts": ["bench/qk-tensile-extraction/inmodel_measurement.json",
                           "bench/qk-tensile-extraction/shape_matrix.json"],
    "runner_commands": [
      "PREFILL_TENSILE_GEMM=1 <warm pp512 harness>",
      "PREFILL_TENSILE_GEMM=1 <prefill dNLL harness>",
      "PREFILL_TENSILE_GEMM=0 <fallback parity harness>",
    ],
    "promotion_gate": "pp512 >=1.25x, dNLL <=0.01, fallback clean, decode untouched",
    "runs_hardware": False,
  },
  "prefill_tensile_codegen_transfer": {
    "schema": "primitive_lifecycle_runner_binding_v1",
    "phase": "prefill",
    "authority": "Tensile artifact is oracle; native route must match isolated and in-model gates",
    "existing_artifacts": ["bench/amd-schedule-codegen-exhaustion/oracle_matrix.json"],
    "runner_commands": ["<native renderer GEMM oracle ladder>", "<warm pp512 after native route>"],
    "promotion_gate": "native matmul approaches Tensile-class TFLOPS, then pp/dNLL gates",
    "runs_hardware": False,
  },
  "prefill_route_a_asm_lds": {
    "schema": "primitive_lifecycle_runner_binding_v1",
    "phase": "prefill",
    "authority": "P2/P3 isolated TFLOPS first; no model route while below A2/PREFILL_V2",
    "existing_artifacts": ["bench/qk-codegen-wmma/route_a_a3_lds_multiwave.json"],
    "runner_commands": ["<Route A P2 double-buffer isolated GEMM>", "<warm pp only if isolated clears gate>"],
    "promotion_gate": "beat A2/PREFILL_V2 isolated before any in-model route",
    "runs_hardware": False,
  },
  "prefill_attention_flash_lifecycle": {
    "schema": "primitive_lifecycle_runner_binding_v1",
    "phase": "prefill",
    "authority": "long-prompt pp and attention share",
    "existing_artifacts": ["bench/qk-flash-prefill-phase5/result.json"],
    "runner_commands": ["<long-prompt attention component map>", "<long-prompt warm pp after locality kernel>"],
    "promotion_gate": "long-prompt +10% warm pp and exact/dNLL gate",
    "runs_hardware": False,
  },
}


def attach_refutations(candidates:list[dict[str, Any]]) -> dict[str, Any]:
  by_id = {c["id"]: c for c in candidates}
  report: dict[str, Any] = {
    "schema": "primitive_lifecycle_refutation_memory_v1",
    "entries": REFUTATION_MEMORY,
    "candidate_pruning": {},
    "validation": {"live_candidates_with_refutations": True, "errors": []},
  }
  for cand in candidates:
    entries = [r for r in REFUTATION_MEMORY if cand["id"] in r["applies_to"]]
    report["candidate_pruning"][cand["id"]] = {
      "state": cand["state"],
      "refutations": [r["id"] for r in entries],
      "blocked_by_refutations": cand.get("blocked_by_refutations", []),
    }
    if cand["state"] not in {"refuted", "closed", "deferred"} and not cand.get("blocked_by_refutations"):
      report["validation"]["live_candidates_with_refutations"] = False
      report["validation"]["errors"].append(f"{cand['id']} has no blocked_by_refutations")
    if cand["id"] not in by_id:
      report["validation"]["errors"].append(f"unknown candidate {cand['id']}")
  return report


def build_runner_bindings(candidates:list[dict[str, Any]]) -> dict[str, Any]:
  rows = []
  errors = []
  for cand in candidates:
    binding = RUNNER_BINDINGS.get(cand["id"])
    if cand["state"] in {"diagnostic", "open", "pass_research", "pass_strong_policy_gated", "project_level", "deferred"} and binding is None:
      errors.append(f"missing runner binding for {cand['id']}")
    if binding is not None:
      rows.append({"candidate_id": cand["id"], "candidate_state": cand["state"], **binding})
  return {
    "schema": "primitive_lifecycle_runner_bindings_v1",
    "rows": rows,
    "validation": {"all_live_candidates_bound": not errors, "errors": errors},
  }


def build_policy_exports(candidates:list[dict[str, Any]]) -> dict[str, Any]:
  by_id = {c["id"]: c for c in candidates}
  policies = []
  for cid, flag, policy_state in [
    ("decode_q8_artifact_lifecycle", "Q8_FFN_HANDWRITTEN=1", "research_candidate_default_off"),
    ("prefill_tensile_artifact_full", "PREFILL_TENSILE_GEMM=1", "research_candidate_default_off"),
  ]:
    cand = by_id[cid]
    policies.append({
      "schema": "primitive_lifecycle_policy_candidate_v1",
      "candidate_id": cid,
      "policy_state": policy_state,
      "flag": flag,
      "default": "off",
      "fallback": "flag off uses current tinygrad path; unsupported shapes fall through",
      "artifact_dependency": cand["routing"].get("artifact_dependency", False),
      "quality_gate": cand["quality"].get("gate"),
      "speed_gate": cand["score"].get("speed_gate"),
      "evidence": cand["evidence"],
      "decision_required": "accept external/research artifact policy before any maintained opt-in",
    })
  return {
    "schema": "primitive_lifecycle_policy_exports_v1",
    "note": "Policy candidates only. No model default or runtime route is changed by this artifact.",
    "policies": policies,
  }


def generate_candidates(candidates:list[dict[str, Any]], refutations:dict[str, Any]) -> dict[str, Any]:
  existing = {c["id"] for c in candidates}
  generated = [
    {
      "id": "decode_q8_sidechannel_native_after_codegen_capability",
      "derived_from": "decode_q8_native_codegen",
      "phase": "decode",
      "state": "proposed",
      "legal": True,
      "requires": ["fused multi-output RMSNorm/q8 producer", "hipcc-quality schedule or imported equivalent"],
      "would_be_pruned_by": [],
      "promotion_gate": "same W==D/dNLL gate as decode_q8_artifact_lifecycle",
    },
    {
      "id": "decode_spec_tcheap_verify_forward",
      "derived_from": "decode_spec_weight_amortization_lifecycle",
      "phase": "decode",
      "state": "proposed",
      "legal": True,
      "requires": ["project-level T-cheap batched-forward route", "T=K+1 target verify <=1.3-1.5x one pass", "low-sync accept/commit", "greedy byte-exact KV protocol"],
      "would_be_pruned_by": ["spec_current_verify_not_amortized"],
      "promotion_gate": "Only after project route exists: verify <=1.3-1.5x one pass, then W==D >=1.2x",
    },
    {
      "id": "decode_q8_separate_pack_dot4_retry",
      "derived_from": "decode_q8_separate_pack",
      "phase": "decode",
      "state": "pruned",
      "legal": False,
      "requires": ["separate activation pack"],
      "would_be_pruned_by": ["separate_q8_pack_wall"],
      "promotion_gate": "none; refuted lifecycle",
    },
    {
      "id": "decode_spec_current_verify_retry",
      "derived_from": "decode_spec_verify_shortcut",
      "phase": "decode",
      "state": "pruned",
      "legal": False,
      "requires": ["current T>1 verify path"],
      "would_be_pruned_by": ["spec_verify_distributed_t_scaling", "spec_current_verify_not_amortized"],
      "promotion_gate": "none; current verify is measured 4.66x one pass",
    },
    {
      "id": "prefill_tensile_artifact_hardened_shapes",
      "derived_from": "prefill_tensile_artifact_full",
      "phase": "prefill",
      "state": "proposed",
      "legal": True,
      "requires": ["artifact policy yes", "shape/fallback matrix", "versioned HSACO contract"],
      "would_be_pruned_by": [],
      "promotion_gate": "pp512/pp1024 + dNLL + fallback",
    },
    {
      "id": "prefill_rocblas_runtime_bridge_retry",
      "derived_from": "prefill_tensile_artifact_full",
      "phase": "prefill",
      "state": "pruned",
      "legal": False,
      "requires": ["HIP runtime inside tinygrad process"],
      "would_be_pruned_by": ["hip_runtime_hcq_exclusion"],
      "promotion_gate": "none; runtime boundary refuted",
    },
    {
      "id": "prefill_tensile_native_renderer_transfer",
      "derived_from": "prefill_tensile_codegen_transfer",
      "phase": "prefill",
      "state": "proposed",
      "legal": True,
      "requires": ["software-pipelined K-loop", "spill-free accumulators", "renderer/scheduler capability"],
      "would_be_pruned_by": ["bounded_wmma_plateau"],
      "promotion_gate": "approach Tensile isolated TFLOPS, then pp/dNLL",
    },
    {
      "id": "prefill_attention_reuse_free_retry",
      "derived_from": "prefill_attention_flash_lifecycle",
      "phase": "prefill",
      "state": "pruned",
      "legal": False,
      "requires": ["reuse-free K/V reread"],
      "would_be_pruned_by": ["reuse_free_attention_not_a_primitive"],
      "promotion_gate": "none; locality missing",
    },
  ]
  for row in generated:
    if row["id"] in existing:
      raise ValueError(f"generated candidate collides with seed candidate {row['id']}")
  return {
    "schema": "primitive_lifecycle_generated_candidates_v1",
    "method": "manual table-driven producer x format x consumer x routing enumeration; no hardware execution",
    "rows": generated,
    "summary": {
      "generated": len(generated),
      "legal_proposed": sum(1 for r in generated if r["legal"]),
      "pruned": sum(1 for r in generated if not r["legal"]),
      "refutation_entries_used": sorted({rid for r in generated for rid in r["would_be_pruned_by"]}),
    },
  }


def build_candidates() -> list[dict[str, Any]]:
  tensile = _read_json("bench/qk-tensile-extraction/inmodel_measurement.json")
  shape = _read_json("bench/qk-tensile-extraction/shape_matrix.json")
  q8_artifact = _read_json("bench/q8-ffn-amd-scheduler-project/result.json")
  codegen = _read_json("bench/amd-schedule-codegen-exhaustion/oracle_matrix.json")
  spec_model = _read_json("bench/qk-spec-decode-bandwidth-amortization/model.json")
  q8_speedup = _speedup_by_ctx()
  q8_dnll = _dnll_delta()

  candidates = [
    _candidate(
      id="decode_q8_separate_pack",
      phase="decode",
      state="refuted",
      role="Q4_K ffn_gate/up",
      producer={"placement": "separate post-RMSNorm q8 pack", "reuse_count": 2},
      format={"activation": "q8_1 side buffer", "weights": "Q4_K", "lossy": True},
      consumer={"primitive": "native dot4 MMVQ", "owner": "tinygrad/hand artifact"},
      routing={"mode": "not routed", "default_safe": True, "artifact_dependency": False},
      quality={"gate": "dNLL <= 0.01 if speed passes", "observed": "not worth routing"},
      score={"speed_gate": "FAIL", "expected_e2e": "<=0 or sub-gate", "rank": 90},
      evidence=["docs/qk-q8-activation-lifecycle-verdict-20260618.md",
                "docs/q8-mmvq-lifecycle-deep-result-20260619.md"],
      blocked_by_refutations=["separate q8 pack cost erases dot4 gain"],
      next_action="Do not reopen without a fused producer.",
    ),
    _candidate(
      id="decode_q8_artifact_lifecycle",
      phase="decode",
      state="pass_research",
      role="Q4_K ffn_gate/up",
      producer={"placement": "hipcc/LLD fused RMSNorm/q8 producer artifact", "reuse_count": 2},
      format={"activation": "q8_1", "weights": "Q4_K", "lossy": True},
      consumer={"primitive": "hipcc/LLD fused gate+up dot4 artifact", "owner": "external artifact"},
      routing={"mode": "Q8_FFN_HANDWRITTEN=1 research flag", "default_safe": True,
                 "artifact_dependency": True, "no_hip_runtime_in_process": True},
      quality={"gate": "dNLL <= 0.01", "observed_delta": q8_dnll},
      score={"speed_gate": "PASS_RESEARCH", "ctx_speedup": q8_speedup,
               "isolated_lifecycle_us": q8_artifact.get("summary", {}).get("lifecycle_us"),
               "rank": 20},
      evidence=["docs/q8-ffn-handwritten-a4-decode-result-20260619.md",
                "docs/q8-ffn-artifact-import-route-result-20260619.md",
                "bench/q8-ffn-handwritten-oracle/decode_wd_q8_route.json",
                "bench/q8-ffn-amd-scheduler-project/result.json"],
      blocked_by_refutations=["native tinygrad ASM/COMGR variants fail perf"],
      next_action="Policy decision: accept research-only artifact route or keep default off.",
    ),
    _candidate(
      id="decode_q8_native_codegen",
      phase="decode",
      state="project_level",
      role="Q4_K ffn_gate/up",
      producer={"placement": "tinygrad-owned fused RMSNorm/apply side-channel", "reuse_count": 2},
      format={"activation": "q8_1", "weights": "Q4_K", "lossy": True},
      consumer={"primitive": "tinygrad-owned dot4 MMVQ", "owner": "renderer/AMD scheduler"},
      routing={"mode": "future native route", "default_safe": True, "artifact_dependency": False},
      quality={"gate": "same as artifact route", "observed_delta_authority": q8_dnll},
      score={"speed_gate": "UNKNOWN_NATIVE", "oracle_ctx_speedup": q8_speedup,
               "rank": 30},
      evidence=["docs/q8-ffn-route-a-scheduler-codegen-result-20260619.md",
                "docs/q8-ffn-route-a-pmu-sqtt-evidence-result-20260619.md",
                "bench/amd-schedule-codegen-exhaustion/oracle_matrix.json"],
      blocked_by_refutations=["no bounded AMD DSL feature closes artifact gap",
                              "producer multi-granularity reduction not UOp-expressible"],
      next_action="Only fund as AMD scheduler/codegen project, not primitive search.",
    ),
    _candidate(
      id="decode_spec_verify_shortcut",
      phase="decode",
      state="closed",
      role="speculative decode verify",
      producer={"placement": "draft tokens then batched verify", "reuse_count": "K+1"},
      format={"activation": "batched T", "weights": "existing GGUF", "lossy": False},
      consumer={"primitive": "batched forward verify", "owner": "tinygrad"},
      routing={"mode": "not routed", "default_safe": True, "artifact_dependency": False},
      quality={"gate": "exact", "observed": "correct but too slow"},
      score={"speed_gate": "FAIL", "rank": 95},
      evidence=["docs/qk-spec-verify-component-breakdown-20260618.md"],
      blocked_by_refutations=["verify cost distributed across attention + Q4_K + Q6_K"],
      next_action="Do not treat spec verify as a single lifecycle candidate.",
    ),
    _candidate(
      id="decode_spec_weight_amortization_lifecycle",
      phase="decode",
      state="project_level",
      role="speculative decode target verify",
      producer={"placement": "0.6B draft low-sync proposal graph", "reuse_count": "accepted tokens/pass"},
      format={"activation": "short target verify block T=K+1", "weights": "target GGUF", "lossy": False},
      consumer={"primitive": "T-cheap target verify forward", "owner": "future tinygrad batched-forward/runtime route"},
      routing={"mode": "future SPEC_DECODE=1 research flag", "default_safe": True, "artifact_dependency": False},
      quality={"gate": "greedy byte-exact target output", "observed": "acceptance excellent; naive route exact but slow"},
      score={"speed_gate": "NO_BOUNDED_SHARED_PRIMITIVE",
             "accepted_per_pass_K4_0p6B": 2.844,
             "current_verify_T5_x_one_pass": 4.66,
             "required_verify_x_one_pass": "1.0-1.5",
             "sdb2_classification": spec_model.get("sdb2_verify_design_audit", {}).get("classification"),
             "rank": 32},
      evidence=["docs/spec-decode-bandwidth-amortization-scope-20260619.md",
                "docs/spec-decode-bandwidth-amortization-sdb1-sdb2-result-20260619.md",
                "docs/spec-decode-tcheap-batched-forward-project-scope-20260619.md",
                "docs/spec-decode-tcheap-batched-forward-tbf0-tbf2-result-20260619.md",
                "docs/spec-decode-component-route-candidates-scope-20260619.md",
                "docs/spec-decode-component-route-candidates-result-20260619.md",
                "bench/qk-spec-decode-bandwidth-amortization/model.json",
                "bench/qk-spec-tcheap-forward/result.json",
                "bench/qk-spec-component-routes/result.json",
                "bench/qk-spec-decode-acceptance/result.json",
                "bench/qk-spec-decode-production/baseline.json",
                "bench/qk-spec-verify-component-breakdown/result.json",
                "bench/qk-primitive-pmu-atlas/result.json"],
      blocked_by_refutations=["current T>1 verify path does not amortize weights",
                              "naive production route is host/sync-bound",
                              "SCR-0..4 found no bounded attention or grouped-linears component route"],
      next_action="Closed at the bounded-kernel level; reopen only with a measured <=1.5x component candidate or as a project-level T-cheap forward effort.",
    ),
    _candidate(
      id="prefill_tensile_artifact_full",
      phase="prefill",
      state="pass_strong_policy_gated",
      role="ffn_gate/up + ffn_down + attn_q/o",
      producer={"placement": "PREFILL_V2 fp16 realized weights", "reuse_count": "T=512"},
      format={"activation": "fp16/fp32 accum", "weights": "fp16 realized", "lossy": False},
      consumer={"primitive": "extracted rocBLAS/Tensile GEMM through HCQ", "owner": "external artifact"},
      routing={"mode": "PREFILL_TENSILE_GEMM=1 research flag", "default_safe": True,
                 "artifact_dependency": True, "no_hip_runtime_in_process": True},
      quality={"gate": "dNLL <= 0.01", "observed": tensile.get("quality_dNLL")},
      score={"speed_gate": "PASS_STRONG_POLICY_GATED",
               "pp512_speedup_ffn_only": tensile.get("warm_pp512_speedup"),
               "pp512_speedup_all_roles": tensile.get("A5_attn_qo_routed", {}).get("warm_pp512_speedup"),
               "shape_matrix_speedup": shape.get("full_pp_speedup"),
               "rank": 1},
      evidence=["docs/prefill-tensile-research-measurement-result-20260619.md",
                "docs/prefill-tensile-tpe5-shape-matrix-result-20260619.md",
                "bench/qk-tensile-extraction/inmodel_measurement.json",
                "bench/qk-tensile-extraction/shape_matrix.json"],
      blocked_by_refutations=["in-process HIP runtime bridge killed; HCQ artifact route required"],
      next_action="Decide external artifact policy; if accepted, harden shape/fallback matrix.",
    ),
    _candidate(
      id="prefill_tensile_codegen_transfer",
      phase="prefill",
      state="project_level",
      role="fp16 GEMM matmul bucket",
      producer={"placement": "PREFILL_V2 fp16 realized weights", "reuse_count": "T"},
      format={"activation": "fp16", "weights": "fp16 realized", "lossy": False},
      consumer={"primitive": "tinygrad renderer schedule matching Tensile", "owner": "tinygrad"},
      routing={"mode": "future native prefill route", "default_safe": True, "artifact_dependency": False},
      quality={"gate": "same as Tensile artifact", "observed": "artifact oracle quality passes"},
      score={"speed_gate": "UNKNOWN_NATIVE", "oracle_pp512_speedup": tensile.get("A5_attn_qo_routed", {}).get("warm_pp512_speedup"),
               "rank": 35},
      evidence=["docs/amd-schedule-codegen-exhaustion-result-20260619.md",
                "bench/amd-schedule-codegen-exhaustion/oracle_matrix.json"],
      blocked_by_refutations=["bounded WMMA/LDS knob sweeps plateau near 42 TFLOPS"],
      next_action="Treat as reusable AMD renderer/scheduler project, using Tensile as oracle.",
    ),
    _candidate(
      id="prefill_route_a_asm_lds",
      phase="prefill",
      state="diagnostic",
      role="dependency-free RDNA3 WMMA GEMM",
      producer={"placement": "hand ASM LDS-staged GEMM", "reuse_count": "T"},
      format={"activation": "fp16", "weights": "fp16 realized", "lossy": False},
      consumer={"primitive": "multi-wave LDS WMMA hand schedule", "owner": "tinygrad research ASM"},
      routing={"mode": "not routed", "default_safe": True, "artifact_dependency": False},
      quality={"gate": "fp16 oracle + pp/dNLL once fast", "observed": "correct P0/P1, perf below A2"},
      score={"speed_gate": "DIAGNOSTIC", "rank": 45},
      evidence=["docs/route-a-a3-p0-p1-result-20260619.md",
                "docs/route-a-a3-lds-multiwave-result-20260619.md"],
      blocked_by_refutations=["naive multi-wave LDS occupancy/barrier tax; not yet double-buffered"],
      next_action="Wait for P2 double-buffer/occupancy result before promoting.",
    ),
    _candidate(
      id="prefill_attention_flash_lifecycle",
      phase="prefill",
      state="deferred",
      role="long-prompt attention",
      producer={"placement": "K/V tile in LDS/registers", "reuse_count": "prompt length"},
      format={"activation": "fp16 attention state", "weights": "n/a", "lossy": False},
      consumer={"primitive": "flash-prefill attention", "owner": "future tinygrad kernel"},
      routing={"mode": "not routed", "default_safe": True, "artifact_dependency": False},
      quality={"gate": "exact/dNLL if dtype changes", "observed": "reuse-free variants refuted"},
      score={"speed_gate": "DEFERRED_LONG_CONTEXT", "rank": 60},
      evidence=["docs/qk-machine-search-primitive-rows-20260618.md",
                "docs/performance-primitive-external-research-audit-20260619.md"],
      blocked_by_refutations=["reuse-free attention is not a performance primitive"],
      next_action="Only reopen with an actual locality design and long-prompt Amdahl.",
    ),
  ]

  for row in codegen.get("rows", []):
    if row.get("primitive") == "prefill_tensile_fp16_gemm":
      candidates[5]["score"]["codegen_oracle_state"] = row.get("state")
    if row.get("primitive") == "q8_decode_mmvq_lifecycle":
      candidates[2]["score"]["codegen_oracle_state"] = row.get("state")
  return candidates


def summarize(candidates:list[dict[str, Any]]) -> dict[str, Any]:
  counts: dict[str, int] = {}
  by_phase: dict[str, dict[str, int]] = {}
  for cand in candidates:
    counts[cand["state"]] = counts.get(cand["state"], 0) + 1
    phase = cand["phase"]
    by_phase.setdefault(phase, {})
    by_phase[phase][cand["state"]] = by_phase[phase].get(cand["state"], 0) + 1
  ranked = sorted(candidates, key=lambda c: c["score"].get("rank", 999))
  return {
    "state_counts": counts,
    "state_counts_by_phase": by_phase,
    "top_ranked": [c["id"] for c in ranked[:5]],
    "live_questions": [
      "Is external artifact policy acceptable for research routes?",
      "Does Claude's Route A/P2 dependency-free LDS work beat the current diagnostic state?",
      "Should q8 decode artifact route remain research-only or become a maintained opt-in?",
      "Is a reusable AMD renderer/scheduler or T-cheap batched-forward project funded, or are native codegen/spec rows closed for now?",
    ],
  }


def write_summary_md(payload:dict[str, Any], refutations:dict[str, Any], runners:dict[str, Any],
                     policies:dict[str, Any], generated:dict[str, Any]) -> None:
  lines = [
    "# Primitive lifecycle search - 2026-06-19",
    "",
    "Read-only seed ledger. It does not run hardware or route a model path.",
    "",
    "## State counts",
    "",
  ]
  for state, count in sorted(payload["summary"]["state_counts"].items()):
    lines.append(f"- `{state}`: {count}")
  lines += ["", "## Ranked candidates", ""]
  by_id = {c["id"]: c for c in payload["candidates"]}
  for cid in payload["summary"]["top_ranked"]:
    cand = by_id[cid]
    lines.append(f"- `{cid}`: `{cand['state']}`; next: {cand['next_action']}")
  lines += ["", "## Live questions", ""]
  lines += [f"- {q}" for q in payload["summary"]["live_questions"]]
  lines += [
    "",
    "## PLS completion",
    "",
    f"- `PLS-1 refutation memory`: {len(refutations['entries'])} entries; validation `{refutations['validation']['live_candidates_with_refutations']}`",
    f"- `PLS-2 runner bindings`: {len(runners['rows'])} bindings; validation `{runners['validation']['all_live_candidates_bound']}`",
    f"- `PLS-3 policy exports`: {len(policies['policies'])} research policy candidates; defaults remain off",
    f"- `PLS-4 generator`: {generated['summary']['generated']} generated rows, {generated['summary']['pruned']} pruned by refutations",
    "",
    "## Generated legal rows",
    "",
  ]
  for row in generated["rows"]:
    if row["legal"]:
      lines.append(f"- `{row['id']}`: requires {', '.join(row['requires'])}")
  lines.append("")
  (OUT / "summary.md").write_text("\n".join(lines))


def main() -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  candidates = build_candidates()
  refutations = attach_refutations(candidates)
  runners = build_runner_bindings(candidates)
  policies = build_policy_exports(candidates)
  generated = generate_candidates(candidates, refutations)
  payload = {
    "schema": "primitive_lifecycle_search_v1",
    "date": datetime.date.today().isoformat(),
    "commit": _git_commit(),
    "scope_doc": "docs/primitive-lifecycle-search-scope-20260619.md",
    "candidates": candidates,
    "refutation_memory": refutations,
    "runner_bindings": runners,
    "policy_exports": policies,
    "generated_candidates": generated,
    "pls_status": {"PLS-0": "done", "PLS-1": "done", "PLS-2": "done", "PLS-3": "done", "PLS-4": "done"},
    "summary": summarize(candidates),
  }
  (OUT / "candidates.json").write_text(json.dumps(payload, indent=2) + "\n")
  (OUT / "refutations.json").write_text(json.dumps(refutations, indent=2) + "\n")
  (OUT / "runner_bindings.json").write_text(json.dumps(runners, indent=2) + "\n")
  (OUT / "policy_exports.json").write_text(json.dumps(policies, indent=2) + "\n")
  (OUT / "generated_candidates.json").write_text(json.dumps(generated, indent=2) + "\n")
  write_summary_md(payload, refutations, runners, policies, generated)
  print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
  main()
