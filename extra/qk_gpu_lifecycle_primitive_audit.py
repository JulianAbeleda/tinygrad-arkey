#!/usr/bin/env python3
"""Exhaustive-ish GPU lifecycle primitive audit synthesis.

Read-only artifact synthesis. This intentionally separates:
- exploration gap
- time/correctness confidence
"""
from __future__ import annotations

import json
import pathlib
import subprocess
from datetime import datetime, timezone
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-gpu-lifecycle-primitive-audit-20260624"

DECODE_AUDIT = ROOT / "bench/qk-decode-ctx-slope-lifecycle-primitive-audit-20260624"
PREFILL_AUDIT = ROOT / "bench/qk-prefill-long-context-integration-20260624"
TRACKER = ROOT / "docs/gpu-lifecycle-primitive-coverage-tracker-20260624.md"
SCOPE = ROOT / "docs/exhaustive-gpu-lifecycle-primitive-audit-scope-20260624.md"
PROJECT_LEDGER = ROOT / "bench/qk-project-search-ledger/ledger.jsonl"
PRIM_OBS = ROOT / "bench/qk-primitive-observability/summary.md"


def read_json(path: pathlib.Path) -> dict[str, Any]:
  try:
    obj = json.loads(path.read_text())
    return obj if isinstance(obj, dict) else {}
  except Exception:
    return {}


def git(*args: str) -> str:
  try:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
  except Exception:
    return "unknown"


def rel(path: pathlib.Path) -> str:
  try:
    return str(path.relative_to(ROOT))
  except Exception:
    return str(path)


def effective(gap: float, conf: float) -> int:
  return round((100.0 - gap) * conf / 100.0)


def score(category: str, gap: int, conf: int, reason: str, evidence: list[str], missing: list[str]) -> dict[str, Any]:
  return {
    "category": category,
    "exploration_gap_percent": gap,
    "time_correctness_confidence_percent": conf,
    "effective_explored_percent": effective(gap, conf),
    "reason": reason,
    "evidence": evidence,
    "missing": missing,
  }


def main() -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
  decode_decision = read_json(DECODE_AUDIT / "decision.json")
  decode_cov = read_json(DECODE_AUDIT / "coverage_score_update.json")
  decode_unknown = read_json(DECODE_AUDIT / "unknown_primitive_candidates.json")
  prefill_decision = read_json(PREFILL_AUDIT / "decision.json")

  authority = {
    "date": date,
    "phase": "GPU_LIFECYCLE_PRIMITIVE_AUDIT",
    "branch": git("branch", "--show-current"),
    "commit": git("rev-parse", "HEAD"),
    "dirty": bool(git("status", "--short")),
    "mode": "read_only_artifact_synthesis",
    "scope": rel(SCOPE),
  }

  taxonomy = {
    "schema": "gpu_lifecycle_primitive_taxonomy_v1",
    "provisional": True,
    "categories": [
      {"id": "weight_gemv_matvec", "boundary": "quantized weight read, dequant, activation lifecycle, dot/reduction"},
      {"id": "decode_attention_tile", "boundary": "QK, mask, softmax, PV, split policy, combine, GQA reuse"},
      {"id": "kv_cache_read_lifecycle", "boundary": "cache layout, valid-prefix read, slice/view materialization, buffer identity"},
      {"id": "kv_append_write_lifecycle", "boundary": "append semantics, per-token write, persistence, AFTER/read ordering"},
      {"id": "smallop_lifecycle", "boundary": "RMSNorm, RoPE, residual add, SiLU/mul, casts, copies"},
      {"id": "launch_graph_lifecycle", "boundary": "programs/token, syncs, graph reuse, dispatch overhead"},
      {"id": "memory_bandwidth_layout", "boundary": "effective bytes, coalescing, striding, LDS/vector loads, packed loads"},
      {"id": "codegen_isa_control", "boundary": "v_dot2, cross-lane, waitcnt, LDS, vector loads, renderer/scheduler expressibility"},
      {"id": "prefill_gemm_lifecycle", "boundary": "graph-GEMM/Tensile/WMMA route, LDS pipeline, occupancy, integration"},
      {"id": "prefill_non_gemm_lifecycle", "boundary": "prefill attention, copy/layout, prompt chunk integration"},
      {"id": "harness_authority_lifecycle", "boundary": "W==D, whole-prefill, llama comparator, route flags, clock/dirty metadata"},
      {"id": "unknown_primitive_discovery", "boundary": "unclassified timing, memory, launch, compiler, runtime, or lifecycle effects"},
    ],
  }

  artifacts = {
    "schema": "gpu_lifecycle_artifact_inventory_v1",
    "artifacts": [
      {"path": rel(DECODE_AUDIT / "decision.json"), "role": "decode ctx-slope lifecycle audit", "exists": (DECODE_AUDIT / "decision.json").exists()},
      {"path": rel(PREFILL_AUDIT / "decision.json"), "role": "prefill long-context hardening", "exists": (PREFILL_AUDIT / "decision.json").exists()},
      {"path": rel(PROJECT_LEDGER), "role": "project machine-search ledger", "exists": PROJECT_LEDGER.exists()},
      {"path": rel(PRIM_OBS), "role": "primitive observability summary", "exists": PRIM_OBS.exists()},
      {"path": rel(TRACKER), "role": "manual current tracker", "exists": TRACKER.exists()},
    ],
  }

  # Seed with current tracker, then reflect decode audit updates.
  category_scores = [
    score("weight_gemv_matvec", 15, 82,
          "Q4_K/Q6_K GEMV routes are benchmarked and decode is above llama; remaining gap is route proof/codegen portability.",
          [rel(DECODE_AUDIT / "q4k_route_coverage_by_role.json"), rel(ROOT / "bench/qk-machine-code-translation/primitive_inventory.json")],
          ["fresh per-role route proof with current kernel names"]),
    score("decode_attention_tile", 22, 82,
          "Owned tile is default-on and ctx-slope residual is known; sub-split inside tile remains missing.",
          [rel(DECODE_AUDIT / "attention_qk_pv_softmax_split_by_ctx.json"), rel(DECODE_AUDIT / "decision.json")],
          ["QK/PV/softmax/KV-read byte split"]),
    score("kv_cache_read_lifecycle", 18, 88,
          "Buffer identity removed the major materialization tax; current audit keeps it as pass with residual strided-read slope.",
          [rel(DECODE_AUDIT / "kv_identity_materialization_by_ctx.json"), rel(ROOT / "bench/qk-decode-oracle-explanation/primitive_decomposition.json")],
          ["standing materialization guard rerun"]),
    score("kv_append_write_lifecycle", 55, 50,
          "Correctness feasibility exists, but serving/runtime persistence is not productized and not current speed-critical.",
          ["docs/runtime-kv-core-engine-result-v2-20260623.md"],
          ["serving workload", "append/persistence benchmark"]),
    score("smallop_lifecycle", 60, 45,
          "Current audit emits small-op bucket but it remains coarse; prior docs refute genuine norm/RoPE as primary lever.",
          [rel(DECODE_AUDIT / "smallop_residual_census.json"), "docs/small-ops-time-tax-sub-audit-result-20260622.md"],
          ["fresh current-stack sub-census"]),
    score("launch_graph_lifecycle", 25, 85,
          "Current decode benchmark has 6 programs/token and host sync near zero; keep as guard.",
          [rel(DECODE_AUDIT / "programs_and_syncs_by_ctx.json")],
          ["CI-style regression guard"]),
    score("memory_bandwidth_layout", 35, 68,
          "Whole-cache strided read slope is named, but effective byte and coalescing counters are not measured.",
          [rel(DECODE_AUDIT / "unknown_primitive_candidates.json")],
          ["effective GB/s per owned-tile subrole"]),
    score("codegen_isa_control", 45, 60,
          "Owned kernels and microsearch identify v_dot2/cross-lane/LDS targets; native generalization remains open.",
          [rel(ROOT / "bench/qk-machine-code-translation/primitive_inventory.json")],
          ["native-codegen lowering for owned tile primitives"]),
    score("prefill_gemm_lifecycle", 10, 88,
          "Corrected prefill is flat through ctx8192 and ahead of recorded llama reference; broad prefill GEMM search is low value.",
          [rel(PREFILL_AUDIT / "decision.json"), rel(PREFILL_AUDIT / "whole_prefill_by_ctx_raw.json")],
          ["fresh llama prefill ctx ladder", "cross-model generality"]),
    score("prefill_non_gemm_lifecycle", 35, 72,
          "Corrected long-context run shows no growth; attention/copy split attribution remains weaker than GEMM.",
          [rel(PREFILL_AUDIT / "kv_attention_split_timeseries.json"), rel(PREFILL_AUDIT / "per_role_time_tax_timeseries_by_ctx.json")],
          ["prefill attention/copy sub-split with stronger role classifier"]),
    score("harness_authority_lifecycle", 25, 82,
          "Decode has current ctx ladder against llama; prefill has current tinygrad ladder but llama side is pp512 only.",
          [rel(DECODE_AUDIT / "llama_vs_tinygrad_decode_by_ctx.json"), rel(PREFILL_AUDIT / "whole_prefill_by_ctx_raw.json")],
          ["fresh llama prefill ctx ladder"]),
    score("unknown_primitive_discovery", 70, 40,
          "Decode audit emits unknown candidates; broader automated unknown-source mapping is still absent.",
          [rel(DECODE_AUDIT / "unknown_primitive_candidates.json")],
          ["automated kernel-source mapping", "all-mode unknown bucket sweep"]),
  ]

  target_scores = [
    score("8b_decode_speed_vs_llama", 15, 88,
          "Current decode is above llama at all measured ctx; remaining issue is slope residual, not parity.",
          [rel(DECODE_AUDIT / "llama_vs_tinygrad_decode_by_ctx.json")],
          ["fresh attribution rerun"]),
    score("8b_decode_ctx_slope", 35, 78,
          "Residual is known: whole-cache strided K/V read slope; below action bar within MAXC.",
          [rel(DECODE_AUDIT / "decision.json"), rel(DECODE_AUDIT / "unknown_primitive_candidates.json")],
          ["owned-tile sub-split counters"]),
    score("8b_prefill_speed_vs_llama", 15, 78,
          "Tinygrad prefill is ahead of recorded llama pp512 and flat; comparator is not a fresh llama ctx ladder.",
          [rel(PREFILL_AUDIT / "whole_prefill_by_ctx_raw.json")],
          ["fresh llama pp512/1024/2048/4096/8192"]),
    score("8b_prefill_long_context_stability", 12, 88,
          "Corrected hardening run reaches ctx8192 with flat per-chunk time and corrected launch accounting.",
          [rel(PREFILL_AUDIT / "decision.json"), rel(PREFILL_AUDIT / "runtime_overlap_by_ctx.json")],
          ["stronger attention/copy split"]),
    score("machine_search_readiness_current_8b", 42, 65,
          "Runners/gates exist, but search should wait for bounded lifecycle specs because current speed is above llama.",
          [rel(PROJECT_LEDGER)],
          ["generated exhaustive scoring tool", "unknown-discovery integration"]),
    score("native_codegen_portability", 50, 60,
          "Owned/hand paths prove targets, not compiler generality.",
          [rel(ROOT / "bench/qk-machine-code-translation/primitive_inventory.json")],
          ["v_dot2/cross-lane lowering route"]),
    score("serving_runtime_kv_lifecycle", 55, 50,
          "Runtime-KV lane is understood enough to defer, but serving lifecycle is not explored.",
          ["docs/runtime-kv-core-engine-result-v2-20260623.md"],
          ["serving workload definition"]),
  ]

  unknown_rows = decode_unknown.get("rows", [])
  unknown_discovery = {
    "schema": "unknown_primitive_discovery_v1",
    "thresholds": {"unclassified_wall_share": 0.02, "ctx_slope_above_noise": True},
    "rows": unknown_rows,
    "verdict": "UNKNOWN_CANDIDATES_PRESENT" if unknown_rows else "NO_UNKNOWN_CANDIDATES_UNDER_THRESHOLDS",
  }

  unexplored = {
    "schema": "unexplored_space_v1",
    "rows": [
      {"area": "decode owned tile sub-split", "why": "ctx-slope residual exists but QK/PV/softmax/KV-read bytes are not separated", "priority": "medium"},
      {"area": "small-op current-stack census", "why": "coarse bucket remains visible and unknown bucket is >2% in prior attribution", "priority": "medium"},
      {"area": "native-codegen portability", "why": "owned kernels solve speed but native compiler cannot express all primitives", "priority": "medium"},
      {"area": "serving/runtime KV lifecycle", "why": "append/persistence not productized; not current speed-critical", "priority": "low"},
      {"area": "fresh llama prefill ctx ladder", "why": "prefill comparator is pp512 only", "priority": "low_medium"},
    ],
  }

  evidence = {
    "schema": "evidence_matrix_v1",
    "category_scores": category_scores,
    "target_scores": target_scores,
  }
  refutations = {
    "schema": "refutation_map_v1",
    "rows": [
      {"id": "broad_decode_search_now", "status": "refuted/deferred", "reason": "current decode is above llama and residual is < action bar"},
      {"id": "prefill_gemm_search_now", "status": "refuted/deferred", "reason": "corrected prefill is flat and ahead of reference"},
      {"id": "rmsnorm_rope_primary_lever", "status": "refuted", "reason": "prior small-op audit says genuine norm/rope near parity"},
      {"id": "fixed_E49152_only_ctx_slope", "status": "refined", "reason": "fixed tax exists, but whole-cache tile slope also erodes savings"},
    ],
  }
  next_queue = {
    "schema": "next_audit_queue_v1",
    "rows": [
      {"rank": 1, "audit": "owned_tile_subsplit_kv_read_qk_pv_softmax", "reason": "only named decode ctx-slope residual", "expected_value": "medium"},
      {"rank": 2, "audit": "smallop_unknown_bucket_source_map", "reason": "unknown bucket appears above threshold", "expected_value": "medium"},
      {"rank": 3, "audit": "native_codegen_translation_targets", "reason": "portability/codegen learning, not current 8B speed", "expected_value": "medium"},
      {"rank": 4, "audit": "fresh_llama_prefill_ctx_ladder", "reason": "tightens prefill comparator only", "expected_value": "low_medium"},
    ],
  }

  decision = {
    "date": date,
    "phase": "GPU_LIFECYCLE_PRIMITIVE_AUDIT_DECISION",
    "label": "GPU_LIFECYCLE_PRIMITIVE_AUDIT_COMPLETE_WITH_UNKNOWN_CANDIDATES",
    "decode_input_decision": decode_decision.get("label"),
    "prefill_input_decision": prefill_decision.get("label"),
    "highest_exploration_gaps": sorted(
      [{"category": r["category"], "gap": r["exploration_gap_percent"]} for r in category_scores],
      key=lambda x: -x["gap"],
    )[:5],
    "lowest_effective_explored": sorted(
      [{"category": r["category"], "effective": r["effective_explored_percent"]} for r in category_scores],
      key=lambda x: x["effective"],
    )[:5],
    "next_step": "owned_tile_subsplit_or_smallop_unknown_source_map",
  }

  summary = [
    "# GPU Lifecycle Primitive Audit Result (2026-06-24)",
    "",
    f"Decision: `{decision['label']}`.",
    "",
    "The audit is not claiming the primitive taxonomy is complete. It emits unknown candidates and keeps separate scores for exploration gap and time/correctness confidence.",
    "",
    "## Category Scores",
    "",
    "| category | exploration gap | time/correctness confidence | effective explored |",
    "|---|---:|---:|---:|",
  ]
  for r in category_scores:
    summary.append(f"| `{r['category']}` | {r['exploration_gap_percent']}% | {r['time_correctness_confidence_percent']}% | {r['effective_explored_percent']}% |")
  summary += [
    "",
    "## Target Scores",
    "",
    "| target | exploration gap | time/correctness confidence | effective explored |",
    "|---|---:|---:|---:|",
  ]
  for r in target_scores:
    summary.append(f"| `{r['category']}` | {r['exploration_gap_percent']}% | {r['time_correctness_confidence_percent']}% | {r['effective_explored_percent']}% |")
  summary += [
    "",
    "## Next Queue",
    "",
  ]
  for row in next_queue["rows"]:
    summary.append(f"- {row['rank']}. `{row['audit']}`: {row['reason']} ({row['expected_value']})")

  outputs = {
    "authority.json": authority,
    "primitive_taxonomy.json": taxonomy,
    "unknown_primitive_discovery.json": unknown_discovery,
    "artifact_inventory.json": artifacts,
    "coverage_scores_by_category.json": {"rows": category_scores},
    "coverage_scores_by_benchmark_target.json": {"rows": target_scores},
    "exploration_gap_by_category.json": {"rows": [{"category": r["category"], "exploration_gap_percent": r["exploration_gap_percent"], "missing": r["missing"]} for r in category_scores]},
    "time_correctness_confidence_by_category.json": {"rows": [{"category": r["category"], "time_correctness_confidence_percent": r["time_correctness_confidence_percent"], "evidence": r["evidence"]} for r in category_scores]},
    "evidence_matrix.json": evidence,
    "unexplored_space.json": unexplored,
    "refutation_map.json": refutations,
    "next_audit_queue.json": next_queue,
    "decision.json": decision,
  }
  for name, obj in outputs.items():
    (OUT / name).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
  (OUT / "summary.md").write_text("\n".join(summary) + "\n")
  print(json.dumps({"ok": True, "out": rel(OUT), "label": decision["label"], "unknown_candidates": len(unknown_rows)}, sort_keys=True))


if __name__ == "__main__":
  main()
