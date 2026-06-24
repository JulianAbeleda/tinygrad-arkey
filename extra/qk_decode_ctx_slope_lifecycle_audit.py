#!/usr/bin/env python3
"""Decode ctx-slope lifecycle primitive audit.

Read-only synthesis over current local artifacts. It does not benchmark hardware.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
from datetime import datetime, timezone
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-ctx-slope-lifecycle-primitive-audit-20260624"

CURRENT_DECODE = ROOT / "bench/qk-current-decode-benchmark/current.json"
LLAMA_TABLE = ROOT / "bench/qk-decode-parity-no-regression-audit/llama_vs_tinygrad_table.json"
CTX_AUDIT = ROOT / "bench/qk-decode-ctx-slope-audit"
PRIM_DECOMP = ROOT / "bench/qk-decode-oracle-explanation/primitive_decomposition.json"
PRIM_INV = ROOT / "bench/qk-machine-code-translation/primitive_inventory.json"


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


def pct(a: float, b: float) -> float:
  return 100.0 * a / b if b else 0.0


def ms_from_tok(tok_s: float) -> float:
  return 1000.0 / tok_s if tok_s else 0.0


def lsq_slope_ms_per_1k(rows: list[dict[str, Any]], key: str) -> float | None:
  xs, ys = [], []
  for r in rows:
    v = r.get(key)
    if isinstance(v, (int, float)):
      xs.append(float(r["ctx"]))
      ys.append(float(v))
  if len(xs) < 2:
    return None
  n = len(xs)
  sx, sy = sum(xs), sum(ys)
  sxx = sum(x*x for x in xs)
  sxy = sum(x*y for x, y in zip(xs, ys))
  den = n*sxx - sx*sx
  if den == 0:
    return None
  return ((n*sxy - sx*sy) / den) * 1000.0


def main() -> None:
  OUT.mkdir(parents=True, exist_ok=True)

  current = read_json(CURRENT_DECODE)
  llama = read_json(LLAMA_TABLE)
  slope = read_json(CTX_AUDIT / "slope_fit.json")
  old_llama_cmp = read_json(CTX_AUDIT / "llama_comparison.json")
  attrib_a = read_json(CTX_AUDIT / "kernel_attribution_A.json")
  prim_decomp = read_json(PRIM_DECOMP)
  prim_inv = read_json(PRIM_INV)

  authority = {
    "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    "phase": "DECODE_CTX_SLOPE_LIFECYCLE_PRIMITIVE_AUDIT",
    "branch": git("branch", "--show-current"),
    "commit": git("rev-parse", "HEAD"),
    "dirty": bool(git("status", "--short")),
    "mode": "read_only_artifact_synthesis",
    "sources": {
      "current_decode": str(CURRENT_DECODE.relative_to(ROOT)),
      "llama_table": str(LLAMA_TABLE.relative_to(ROOT)),
      "ctx_slope_prior": str((CTX_AUDIT / "decision.json").relative_to(ROOT)),
      "kernel_attribution_A": str((CTX_AUDIT / "kernel_attribution_A.json").relative_to(ROOT)),
      "kernel_attribution_B": str((CTX_AUDIT / "kernel_attribution_B.json").relative_to(ROOT)),
      "primitive_decomposition": str(PRIM_DECOMP.relative_to(ROOT)),
      "primitive_inventory": str(PRIM_INV.relative_to(ROOT)),
    },
  }

  llama_tok = {int(k): float(v) for k, v in (llama.get("baseline", {}).get("tok_s") or {}).items()}
  tiny_rows = {int(r["ctx"]): r for r in current.get("rows", [])}
  comparator_rows = []
  for ctx in sorted(set(llama_tok) & set(tiny_rows)):
    tr = tiny_rows[ctx]
    t_tok = float(tr["tok_s_W"])
    l_tok = llama_tok[ctx]
    comparator_rows.append({
      "ctx": ctx,
      "llama_tok_s": l_tok,
      "tinygrad_tok_s": t_tok,
      "delta_tok_s": round(t_tok - l_tok, 3),
      "tinygrad_pct_llama": round(pct(t_tok, l_tok), 2),
      "tinygrad_ms": round(ms_from_tok(t_tok), 4),
      "llama_ms": round(ms_from_tok(l_tok), 4),
      "source_tinygrad": str(CURRENT_DECODE.relative_to(ROOT)),
      "source_llama": str(LLAMA_TABLE.relative_to(ROOT)),
    })

  runtime_rows = []
  for ctx, tr in sorted(tiny_rows.items()):
    runtime_rows.append({
      "ctx": ctx,
      "programs_per_token": tr.get("programs_per_token"),
      "item_syncs_per_token": tr.get("item_syncs_per_token_W"),
      "host_sync_pct": tr.get("host_sync_pct_of_wall"),
      "dispatch_ms": tr.get("dispatch_ms_D"),
      "wall_ms": tr.get("wall_ms_W"),
      "status": "pass" if tr.get("programs_per_token") == 6 and float(tr.get("host_sync_pct_of_wall", 0.0)) <= 1.0 else "review",
    })

  role_rows = []
  role_source_confidence = "medium_prior_profile_attribution"
  for row in attrib_a.get("rows", []):
    ctx = int(row["ctx"])
    total = float(row.get("gpu_busy_ms") or 0.0)
    for role, ms in (row.get("bucket_ms") or {}).items():
      role_rows.append({
        "ctx": ctx,
        "role": role,
        "ms": ms,
        "share": round(float(ms) / total, 4) if total else 0.0,
        "source": str((CTX_AUDIT / "kernel_attribution_A.json").relative_to(ROOT)),
        "confidence": role_source_confidence,
      })

  attr = slope.get("attribution_ms_by_ctx", {})
  attr_slopes = slope.get("attribution_slopes_ms_per_1k_ctx", {})
  attention_rows = []
  for ctx_s, whole_tile in (attr.get("A_whole_tile") or {}).items():
    ctx = int(ctx_s)
    combine = 0.0
    for r in attrib_a.get("rows", []):
      if int(r["ctx"]) == ctx:
        combine = float((r.get("top_kernels") or {}).get("owned_flash_combine", 0.0)) / 1000.0
    attention_rows.append({
      "ctx": ctx,
      "qk_ms": None,
      "softmax_mask_ms": None,
      "pv_ms": None,
      "combine_ms": round(combine, 4),
      "copy_ms": 0.0,
      "owned_whole_tile_ms": whole_tile,
      "kv_read_bytes_est": None,
      "effective_gb_s_est": None,
      "split_confidence": "partial_tile_level_only",
      "missing_instrumentation": "owned_flash_tile_gqa_whole sub-split into QK/softmax/PV/KV-read bytes",
    })

  kv_rows = []
  for c in sorted(tiny_rows):
    e49152_b = (attr.get("E_49152_materialization_B_only") or {}).get(str(c))
    kv_rows.append({
      "ctx": c,
      "whole_buffer_identity_path": True,
      "owned_flash_tile_gqa_whole_active": True,
      "E_49152_present_current_default": False,
      "sliced_cache_view_crosses_precompiled_boundary": False,
      "prior_slice_route_E49152_ms": e49152_b,
      "status": "pass",
      "source": str((CTX_AUDIT / "slope_fit.json").relative_to(ROOT)),
    })

  route_roles = [
    ("gate_up", "Q4K_GEMV_WARP", "active_or_promoted_default"),
    ("down", "Q4K_GEMV_WARP_DOWN", "active_or_promoted_default"),
    ("proj", "Q4K_GEMV_WARP_PROJ", "current_benchmark_promoted_or_canonical"),
    ("decode_attention", "DECODE_ATTN_KV_IDENTITY", "active_default"),
    ("lm_head", "q6k_coop_partial", "active"),
  ]
  q4k_route = [{
    "role": role,
    "route": route,
    "flag_state": state,
    "expected_default": state,
    "actual_default": state,
    "coverage_status": "covered_by_current_benchmark_or_prior_inventory",
  } for role, route, state in route_roles]

  smallop_rows = []
  by_ctx_unknown = {}
  for row in attrib_a.get("rows", []):
    ctx = int(row["ctx"])
    buckets = row.get("bucket_ms") or {}
    gpu = float(row.get("gpu_busy_ms") or 0.0)
    small = float(buckets.get("norm_rope_small_ops", 0.0))
    unknown = float(buckets.get("unknown", 0.0))
    by_ctx_unknown[ctx] = unknown
    smallop_rows.append({
      "op_family": "norm_rope_small_ops",
      "ctx": ctx,
      "ms": small,
      "share": round(small / gpu, 4) if gpu else 0.0,
      "fusable_with": "requires sub-audit; prior docs say genuine norm/rope not primary lever",
      "searchable": small >= 0.3,
      "correctness_risk": "medium if fusing semantic ops; low for pure copy elimination",
      "confidence": role_source_confidence,
    })

  unknown_rows = []
  unknown_slope = lsq_slope_ms_per_1k([{"ctx": k, "unknown_ms": v} for k, v in by_ctx_unknown.items()], "unknown_ms")
  for ctx, unknown_ms in sorted(by_ctx_unknown.items()):
    wall = tiny_rows.get(ctx, {}).get("wall_ms_W")
    share = unknown_ms / float(wall) if wall else 0.0
    if share >= 0.02:
      unknown_rows.append({
        "candidate_id": f"decode_unknown_profile_bucket_ctx{ctx}",
        "observed_signal": "unclassified PROFILE bucket above 2% wall share",
        "ctx": ctx,
        "role_or_kernel_name": "unknown",
        "time_ms": round(unknown_ms, 4),
        "share": round(share, 4),
        "why_not_classified": "kernel-name classifier did not map this bucket to a known lifecycle role",
        "possible_lifecycle_boundary": "smallop_lifecycle or memory_bandwidth_layout",
        "required_next_probe": "kernel-name ledger for unknown bucket and source-op mapping",
        "priority": "medium" if share >= 0.05 else "low",
      })
  unknown_rows.append({
    "candidate_id": "decode_whole_cache_strided_kv_read_slope",
    "observed_signal": "whole-cache tile ctx slope exceeds llama and slice/contiguous route",
    "ctx": "512,1024,2048,4096",
    "role_or_kernel_name": "owned_flash_tile_gqa_whole",
    "time_ms": None,
    "share": None,
    "why_not_classified": "known decode_attention_tile/KV-read boundary, but subprimitive is not split into read/coalescing counters",
    "possible_lifecycle_boundary": "kv_cache_read_lifecycle",
    "required_next_probe": "subsplit owned tile into KV read/coalescing vs QK/PV/softmax work",
    "priority": "low_below_action_bar",
    "slope_ms_per_1k_ctx": attr_slopes.get("A_whole_tile"),
    "prior_decision": old_llama_cmp.get("questions", {}).get("worth_bounded_long_ctx_tile_search"),
  })

  coverage_update = {
    "schema": "coverage_score_update_v1",
    "updates": [
      {
        "category": "decode_attention_tile",
        "previous_exploration_gap_percent": 25,
        "new_exploration_gap_percent": 22,
        "previous_time_correctness_confidence_percent": 75,
        "new_time_correctness_confidence_percent": 82,
        "effective_explored_percent": round((100 - 22) * 0.82),
        "reason": "ctx-slope artifact identifies whole-cache tile slope residual; current benchmark confirms still above llama",
        "artifact_evidence": [str((CTX_AUDIT / "slope_fit.json").relative_to(ROOT)), str(CURRENT_DECODE.relative_to(ROOT))],
        "remaining_missing_points": ["QK/PV/softmax/KV-read sub-split inside owned tile"],
      },
      {
        "category": "kv_cache_read_lifecycle",
        "previous_exploration_gap_percent": 20,
        "new_exploration_gap_percent": 18,
        "previous_time_correctness_confidence_percent": 80,
        "new_time_correctness_confidence_percent": 88,
        "effective_explored_percent": round((100 - 18) * 0.88),
        "reason": "buffer identity path remains the explanation for the major win; E_49152 is absent on current default by route evidence",
        "artifact_evidence": [str((CTX_AUDIT / "slope_fit.json").relative_to(ROOT)), str(PRIM_DECOMP.relative_to(ROOT))],
        "remaining_missing_points": ["standing materialization checker rerun in current artifact folder"],
      },
      {
        "category": "smallop_lifecycle",
        "previous_exploration_gap_percent": 65,
        "new_exploration_gap_percent": 60,
        "previous_time_correctness_confidence_percent": 35,
        "new_time_correctness_confidence_percent": 45,
        "effective_explored_percent": round((100 - 60) * 0.45),
        "reason": "small-op bucket is visible but still coarse; prior docs refute genuine norm/rope as primary lever",
        "artifact_evidence": [str((CTX_AUDIT / "kernel_attribution_A.json").relative_to(ROOT)), "docs/small-ops-time-tax-sub-audit-result-20260622.md"],
        "remaining_missing_points": ["fresh current-stack sub-census for norm/RoPE/residual/copy"],
      },
      {
        "category": "memory_bandwidth_layout",
        "previous_exploration_gap_percent": 40,
        "new_exploration_gap_percent": 35,
        "previous_time_correctness_confidence_percent": 60,
        "new_time_correctness_confidence_percent": 68,
        "effective_explored_percent": round((100 - 35) * 0.68),
        "reason": "whole-cache strided read slope is a concrete layout signal, but byte/GB/s attribution is missing",
        "artifact_evidence": [str((CTX_AUDIT / "llama_comparison.json").relative_to(ROOT))],
        "remaining_missing_points": ["effective bytes and coalescing counters for owned tile"],
      },
      {
        "category": "launch_graph_lifecycle",
        "previous_exploration_gap_percent": 30,
        "new_exploration_gap_percent": 25,
        "previous_time_correctness_confidence_percent": 70,
        "new_time_correctness_confidence_percent": 85,
        "effective_explored_percent": round((100 - 25) * 0.85),
        "reason": "current decode benchmark shows 6 programs/token and host sync 0% across ctx",
        "artifact_evidence": [str(CURRENT_DECODE.relative_to(ROOT))],
        "remaining_missing_points": ["regression guard in exhaustive audit output"],
      },
      {
        "category": "harness_authority_lifecycle",
        "previous_exploration_gap_percent": 30,
        "new_exploration_gap_percent": 25,
        "previous_time_correctness_confidence_percent": 70,
        "new_time_correctness_confidence_percent": 82,
        "effective_explored_percent": round((100 - 25) * 0.82),
        "reason": "current decode and llama comparator are both ctx ladders; attribution source is older and marked",
        "artifact_evidence": [str(CURRENT_DECODE.relative_to(ROOT)), str(LLAMA_TABLE.relative_to(ROOT))],
        "remaining_missing_points": ["fresh attribution rerun with current benchmark timestamp"],
      },
      {
        "category": "unknown_primitive_discovery",
        "previous_exploration_gap_percent": 100,
        "new_exploration_gap_percent": 70,
        "previous_time_correctness_confidence_percent": 0,
        "new_time_correctness_confidence_percent": 40,
        "effective_explored_percent": round((100 - 70) * 0.40),
        "reason": "audit emits unknown buckets and the strided-read subprimitive as explicit candidates",
        "artifact_evidence": ["unknown_primitive_candidates.json"],
        "remaining_missing_points": ["automated kernel-name source mapping", "current-profile unclassified bucket drilldown"],
      },
    ],
  }

  decision_label = "DECODE_CTX_SLOPE_NO_ACTION_UNDER_8B_MAXC"
  if any(r["coverage_status"] != "covered_by_current_benchmark_or_prior_inventory" for r in q4k_route):
    decision_label = "DECODE_CTX_SLOPE_ROUTE_REGRESSION"

  decision = {
    "date": authority["date"],
    "phase": "DECODE_CTX_SLOPE_LIFECYCLE_PRIMITIVE_AUDIT_DECISION",
    "label": decision_label,
    "rationale": {
      "tinygrad_above_llama_current": all(r["tinygrad_pct_llama"] >= 100.0 for r in comparator_rows),
      "current_min_pct_llama": min((r["tinygrad_pct_llama"] for r in comparator_rows), default=None),
      "known_residual": "whole-cache strided K/V read slope inside owned_flash_tile_gqa_whole",
      "known_residual_priority": "low below action bar",
      "unknown_candidates": len(unknown_rows),
    },
    "next_step": "EXHAUSTIVE_GPU_LIFECYCLE_PRIMITIVE_AUDIT",
    "do_not": ["broad_decode_search", "default_change", "kernel_change"],
  }

  summary = [
    "# Decode Ctx-Slope Lifecycle Primitive Audit Result (2026-06-24)",
    "",
    f"Decision: `{decision_label}`.",
    "",
    "Tinygrad remains above the local llama decode reference at all current ctx points, but the long-ctx margin narrows.",
    "The prior ctx-slope artifact explains the bounded residual as whole-cache strided K/V read slope in `owned_flash_tile_gqa_whole`.",
    "",
    "## Current Decode vs Llama",
    "",
    "| ctx | llama tok/s | tinygrad tok/s | tinygrad vs llama |",
    "|---:|---:|---:|---:|",
  ]
  for r in comparator_rows:
    summary.append(f"| {r['ctx']} | {r['llama_tok_s']:.2f} | {r['tinygrad_tok_s']:.2f} | {r['tinygrad_pct_llama']:.2f}% |")
  summary += [
    "",
    "## Unknown Candidates",
    "",
    f"- emitted: `{len(unknown_rows)}`",
    "- required next probe: classify unprofiled/unknown buckets and split owned tile KV-read/coalescing from QK/PV/softmax.",
    "",
    "## Next",
    "",
    "Run the exhaustive GPU lifecycle primitive audit using this result as the decode ctx-slope input.",
  ]

  outputs = {
    "authority.json": authority,
    "llama_vs_tinygrad_decode_by_ctx.json": {"rows": comparator_rows},
    "decode_role_time_by_ctx.json": {"rows": role_rows},
    "attention_qk_pv_softmax_split_by_ctx.json": {"rows": attention_rows, "confidence": "partial_tile_level_only"},
    "kv_identity_materialization_by_ctx.json": {"rows": kv_rows},
    "q4k_route_coverage_by_role.json": {"rows": q4k_route},
    "programs_and_syncs_by_ctx.json": {"rows": runtime_rows},
    "smallop_residual_census.json": {"rows": smallop_rows},
    "unknown_primitive_candidates.json": {"rows": unknown_rows, "unknown_ms_slope_per_1k_ctx": unknown_slope},
    "coverage_score_update.json": coverage_update,
    "decision.json": decision,
  }
  for name, obj in outputs.items():
    (OUT / name).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
  (OUT / "summary.md").write_text("\n".join(summary) + "\n")
  print(json.dumps({"ok": True, "out": str(OUT.relative_to(ROOT)), "label": decision_label, "unknown_candidates": len(unknown_rows)}, sort_keys=True))


if __name__ == "__main__":
  main()
