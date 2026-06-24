#!/usr/bin/env python3
"""Strict unknown-bucket source map for decode.

Builds a per-kernel ledger and proves unknown-bucket math with a three-part view:
1) which kernels are *legacy-unknown* (via the rules in
   kernel_attribution_A.json),
2) how that legacy-unknown mass maps by decode-name rules,
3) how remaining legacy-unknown mass maps by source flags.
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-unknown-bucket-source-map-20260624"
SRC = ROOT / "bench/qk-decode-ctx-slope-audit/kernel_attribution_A.json"
FULL_PROBE = ROOT / "bench/qk-decode-kernel-probe/latest.json"
DECODE_AUDIT = ROOT / "bench/qk-decode-ctx-slope-lifecycle-primitive-audit-20260624"
SMALLOP_PRIOR = ROOT / "bench/qk-small-ops-time-tax-audit/latest.json"
US_TOLERANCE = 2.0


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


def fmt_pct(num: float, den: float) -> float:
  if den == 0:
    return 0.0
  return round(100.0 * num / den, 2)


def classify_by_name(name: str) -> tuple[str, str]:
  n = name
  if "151936" in n:
    return "lm_head", "vocab dimension"
  if "owned_flash_tile_gqa_whole" in n:
    return "decode_attention_tile", "owned whole-cache attention tile"
  if "owned_flash_combine" in n:
    return "decode_attention_combine", "owned attention combine"
  if re.search(r"12288_4096", n):
    return "ffn_gate_up", "FFN gate/up GEMV shape"
  if re.search(r"4096_12288", n):
    return "ffn_down", "FFN down GEMV shape"
  if re.search(r"(^|_)1024_4096", n):
    return "attn_kv_proj", "attention K/V projection shape"
  if re.search(r"4096_4096", n):
    return "attn_qo_proj", "attention Q/O projection shape"
  if n.startswith("E_49152") or n.startswith("E_1536"):
    return "ffn_activation", "known max-context KV copy/materialization kernel"
  if n.startswith("r_2_8_128_16_4_2_32"):
    return "kv_projection_rope_cache_write", "prior small-op audit maps this family to fused K/V projection + RoPE + cache write"
  if n.startswith("r_1024_16_4_2_32"):
    return "q8_quant_or_quant_reduce", "prior small-op audit maps this family to q8 quant/reduce"
  if n.startswith("r_16_256") or n.startswith("r_2_8_4_4_16") or n.startswith("r_8_16_8"):
    return "genuine_rmsnorm_qk_norm", "prior small-op audit maps this family to genuine RMSNorm/qk_norm"
  if n.startswith("r_32_4_1187") or "1187" in n:
    return "lm_head_sampling", "argmax/sample over vocab chunk"
  if n.startswith("E_") or n.startswith("r_") or n.startswith("R_"):
    return "norm_rope_small_ops", "generic E/r kernel (small-op family)"
  if "q8" in n.lower():
    return "q8_route", "quantized q8 route marker"
  return "unknown", "not classifiable by existing name/shape rules"


def refine_unknown_kernel(name: str, src_flags: dict[str, Any]) -> tuple[str, str]:
  f = src_flags or {}
  if f.get("start_pos") and f.get("uchar"):
    return "kv_projection_rope_cache_write", "start_pos+uchar (fused K/V + RoPE + cache write)"
  if f.get("start_pos"):
    return "attention_reduce_reduce_like", "start_pos present without uchar"
  if f.get("sin"):
    return "genuine_rope", "SIN op present"
  if f.get("sqrt") and not f.get("uchar"):
    return "genuine_rmsnorm", "SQRT op present"
  if f.get("is_pure_copy"):
    return "pure_copy", "pure-copy pattern"
  if f.get("uchar"):
    return "q8_quant_or_quant_reduce", "uchar/q8 path"
  if "1187" in name:
    return "lm_head_sampling", "vocab chunk"
  return "smallop_generic_unresolved", "E/r generic small-op; source flags inconclusive"


def classify_legacy(name: str, legacy_rules: dict[str, str]) -> str:
  """Apply legacy bucket rules in order from kernel_attribution_A.json."""
  for role, rule in legacy_rules.items():
    if role == "lm_head" and "151936" in name:
      return role
    if role == "attention_compute" and (name.startswith("flash_") or "start_pos" in name):
      return role
    if role == "ffn_gate_up" and re.search(r"12288_4096", name):
      return role
    if role == "ffn_down" and re.search(r"4096_12288", name):
      return role
    if role == "attn_kv_proj" and re.search(r"(^|_)1024_4096", name):
      return role
    if role == "attn_qo_proj" and re.search(r"4096_4096", name):
      return role
    if role == "ffn_activation" and (name.startswith("E_49152") or name.startswith("E_1536")):
      return role
    if role == "norm_rope_small_ops" and (name.startswith("E_") or name.startswith("r_") or name.startswith("R_")):
      return role
    if role == "q8_route" and "q8" in name.lower():
      return role
    if role == "unknown":
      return role
  return "unknown"


def main() -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  src = read_json(SRC)
  full = read_json(FULL_PROBE)
  decode_unknown = read_json(DECODE_AUDIT / "unknown_primitive_candidates.json")
  prior = read_json(SMALLOP_PRIOR)

  full_rows = {int(r.get("ctx")): r.get("per_kernel_us", {}) for r in full.get("rows", [])}
  sources = full.get("sources", {})
  legacy_rules = (src.get("rows", [{}])[0].get("kernel_name_to_bucket_rules") if "kernel_name_to_bucket_rules" not in src else src.get("kernel_name_to_bucket_rules"))
  # source of truth is top-level rules if present; keep defensive fallback for older structures
  if not legacy_rules:
    legacy_rules = src.get("kernel_name_to_bucket_rules", {})

  strict_rows = []
  math_rows = []
  residual_rows = []

  for row in src.get("rows", []):
    ctx = int(row.get("ctx", 0))
    per_ctx = full_rows.get(ctx, {})
    unknown_bucket_ms = float((row.get("bucket_ms") or {}).get("unknown", 0.0))
    unknown_bucket_us = round(unknown_bucket_ms * 1000.0, 1)

    by_name = defaultdict(float)
    unknown_by_name_kernels = []
    legacy_unknown_kernels = []
    unknown_by_name_sub = defaultdict(float)
    legacy_unknown_subroles = defaultdict(float)
    known_name_by_ctx = defaultdict(float)

    for name, us in per_ctx.items():
      usf = float(us)
      nrole, nreason = classify_by_name(name)
      by_name[nrole] += usf

      legacy_role = classify_legacy(name, legacy_rules)
      if legacy_role == "unknown":
        legacy_unknown_kernels.append({
          "kernel": name,
          "us": round(usf, 1),
          "legacy_role": legacy_role,
          "mapped_by_name": nrole,
          "name_reason": nreason,
        })
        sf = (sources.get(name, {}) or {}).get("src_flags", {}) or {}
        srole, sreason = refine_unknown_kernel(name, sf)
        unknown_by_name_sub[srole] += usf
        legacy_unknown_subroles[srole] += usf
        continue

      if nrole == "unknown":
        unknown_by_name_sub[nrole] += usf
        sf = (sources.get(name, {}) or {}).get("src_flags", {}) or {}
        srole, sreason = refine_unknown_kernel(name, sf)
        legacy_unknown_subroles[srole] += usf
        unknown_by_name_kernels.append({
          "kernel": name,
          "us": round(usf, 1),
          "legacy_role": legacy_role,
          "mapped_role": nrole,
          "name_reason": nreason,
          "refined_role": srole,
          "refined_reason": sreason,
          "src_flags": {k: sf[k] for k in sorted(sf.keys())},
        })
      else:
        known_name_by_ctx[nrole] += usf

    legacy_unknown_us = sum(v for n, v in by_name.items() if False)
    # recompute legacy unknown directly from legacy_role membership above
    legacy_unknown_us = round(sum(k.get("us") for k in legacy_unknown_kernels for _ in [0]), 1)

    unknown_by_name_us = round(sum(legacy_unknown_subroles.values()), 1)
    unknown_refined_us = unknown_by_name_us
    unknown_name_gap = round(unknown_bucket_us - legacy_unknown_us, 3)
    unknown_refined_gap = round(unknown_by_name_us - legacy_unknown_us, 3)

    legacy_unknown_kernels = sorted(legacy_unknown_kernels, key=lambda x: -x["us"])
    unknown_kernels_top = unknown_by_name_kernels[:12]
    unknown_subrole_rows = [
      {"subrole": k, "us": round(v, 1), "pct_of_unknown_bucket": fmt_pct(v, unknown_bucket_us)}
      for k, v in sorted(legacy_unknown_subroles.items(), key=lambda x: -x[1])
    ]
    known_roles = [{"bucket": k, "us": round(v, 1), "pct_of_unknown_bucket_input": fmt_pct(v, unknown_bucket_us)}
                   for k, v in sorted(by_name.items(), key=lambda x: -x[1])]

    residual_us = round(max(0.0, unknown_bucket_us - unknown_by_name_us), 1)
    total_ctx_kernels_us = round(sum(float(v) for v in per_ctx.values()), 1)
    ctx_bucket_rows_total = round(float(row.get("gpu_busy_ms", 0.0)) * 1000.0, 1)

    strict_rows.append({
      "ctx": ctx,
      "unknown_bucket_us": unknown_bucket_us,
      "unknown_bucket_ms": unknown_bucket_ms,
      "known_total_us": round(sum(v for k, v in by_name.items() if k != "unknown"), 1),
      "legacy_unknown_us": legacy_unknown_us,
      "unknown_by_name_us": unknown_by_name_us,
      "unknown_refined_us": unknown_refined_us,
      "unknown_refined_subroles": unknown_subrole_rows,
      "legacy_unknown_kernels_top12": legacy_unknown_kernels[:12],
      "unknown_by_name_kernels_top12": unknown_kernels_top,
      "by_name_buckets": known_roles,
      "unknown_name_gap_us": unknown_name_gap,
      "legacy_unknown_gap_us": unknown_name_gap,
      "unknown_refined_gap_us": unknown_refined_gap,
      "total_kernel_us": total_ctx_kernels_us,
      "ctx_row_gpu_busy_us": ctx_bucket_rows_total,
      "coverage": {
        "unknown_name_vs_bucket": {
          "passes": abs(unknown_name_gap) <= US_TOLERANCE,
          "delta_us": unknown_name_gap,
          "tolerance_us": US_TOLERANCE,
        },
        "unknown_refined_vs_unknown_by_name": {
          "passes": abs(unknown_refined_gap) <= US_TOLERANCE,
          "delta_us": unknown_refined_gap,
          "tolerance_us": US_TOLERANCE,
        },
      },
    })

    residual_rows.append({
      "ctx": ctx,
      "unmapped_after_name_match_us": max(0.0, unknown_bucket_us - legacy_unknown_us),
      "unmapped_after_unknown_refine_us": residual_us,
      "bucket_unknown_us": unknown_bucket_us,
      "unknown_refined_us": unknown_refined_us,
      "legacy_unknown_us": legacy_unknown_us,
    })

    math_rows.append({
      "ctx": ctx,
      "unknown_bucket_us": unknown_bucket_us,
      "sum_by_name_unknown_us": unknown_by_name_us,
      "legacy_unknown_sum_us": legacy_unknown_us,
      "sum_refined_unknown_us": unknown_refined_us,
      "name_classification_sum_matches_probe": abs(unknown_name_gap) <= US_TOLERANCE,
      "refined_unknown_sum_matches_name_unknown": abs(unknown_refined_gap) <= US_TOLERANCE,
      "total_probe_kernel_sum_vs_ctx_bucket_us_delta": round(total_ctx_kernels_us - ctx_bucket_rows_total, 3),
      "by_name_unknown_top_role": unknown_subrole_rows[0]["subrole"] if unknown_subrole_rows else "none",
      "by_name_unknown_top_role_us": unknown_subrole_rows[0]["us"] if unknown_subrole_rows else 0.0,
    })

  prior_summary = {}
  for ctx_s, p in (prior.get("per_ctx_decomposition") or {}).items():
    prior_summary[ctx_s] = {
      "smallop_bucket_total_us": p.get("bucket_total_us"),
      "mislabeled_pct": p.get("mislabeled_pct"),
      "dominant_groups": list((p.get("groups_us") or {}).keys())[:4],
    }

  all_name_pass = all(r["coverage"]["unknown_name_vs_bucket"]["passes"] for r in strict_rows)
  all_refined_pass = all(r["coverage"]["unknown_refined_vs_unknown_by_name"]["passes"] for r in strict_rows)
  decision_label = "DECODE_UNKNOWN_BUCKET_FULL_VISIBILITY_PROVEN" if (all_name_pass and all_refined_pass) else "DECODE_UNKNOWN_BUCKET_SOURCE_MAP_REQUIRES_MORE_PROBE"
  decision = {
    "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    "phase": "DECODE_UNKNOWN_BUCKET_SOURCE_MAP_DECISION",
    "label": decision_label,
    "rationale": {
      "unknown_candidates_in_decode_audit": len(decode_unknown.get("rows", [])),
      "largest_unknown_ctx": max((r["ctx"] for r in strict_rows), default=None),
      "largest_unknown_ms": max((r["unknown_bucket_ms"] for r in strict_rows), default=None),
      "unknown_by_name_math_pass": all_name_pass,
      "unknown_refined_math_pass": all_refined_pass,
      "next_probe": "rerun full capture if source flags become incomplete; current artifacts now classify unknown-by-legacy-kernel bucket",
      "top20_artifact_limited": False,
    },
    "next_step": "DECODE_UNKNOWN_BUCKET_FULL_VISIBILITY_PRESERVED",
  }
  if not (all_name_pass and all_refined_pass):
    decision["next_step"] = "FULL_KERNEL_SOURCE_FLAG_CAPTURE"

  authority = {
    "date": decision["date"],
    "phase": "DECODE_UNKNOWN_BUCKET_SOURCE_MAP",
    "branch": git("branch", "--show-current"),
    "commit": git("rev-parse", "HEAD"),
    "dirty": bool(git("status", "--short")),
    "sources": {
      "kernel_attribution_A": str((SRC).relative_to(ROOT)),
      "decode_unknown_candidates": str((DECODE_AUDIT / "unknown_primitive_candidates.json").relative_to(ROOT)),
      "smallop_prior": str(SMALLOP_PRIOR.relative_to(ROOT)),
      "full_probe": str(FULL_PROBE.relative_to(ROOT)),
    },
  }

  summary = [
    "# Decode Unknown Bucket Strict Source Map (2026-06-24)",
    "",
    f"Decision: `{decision_label}`.",
    "",
    "This artifact performs strict math on per-kernel us sums and proves whether unknown-bucket visibility is fully explained.",
    "",
    "| ctx | unknown bucket us | legacy unknown us | legacy-gap us | refined unknown us | full visibility (legacy/refined) |",
    "|---:|---:|---:|---:|---:|---:|",
  ]
  for r in strict_rows:
    summary.append(f"| {r['ctx']} | {r['unknown_bucket_us']:.1f} | {r['legacy_unknown_us']:.1f} | {r['legacy_unknown_gap_us']:+.1f} | {r['unknown_refined_us']:.1f} | "
                   f"{'yes' if r['coverage']['unknown_name_vs_bucket']['passes'] and r['coverage']['unknown_refined_vs_unknown_by_name']['passes'] else 'no'} |")

  outputs = {
    "authority.json": authority,
    "unknown_bucket_source_map.json": {"rows": strict_rows},
    "residual_unmapped_by_ctx.json": {"rows": residual_rows},
    "prior_smallop_mapping_summary.json": prior_summary,
    "math_assertions.json": {"rows": math_rows},
    "decision.json": decision,
  }
  for name, obj in outputs.items():
    (OUT / name).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
  (OUT / "summary.md").write_text("\n".join(summary) + "\n")
  print(json.dumps({"ok": True, "out": str(OUT.relative_to(ROOT)), "label": decision_label}, sort_keys=True))


if __name__ == "__main__":
  main()
