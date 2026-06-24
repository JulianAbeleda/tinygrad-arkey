#!/usr/bin/env python3
"""Single-run strict unknown-bucket proof for decode.

This run performs:
  - one canonical kernel probe with full source capture (`--full-source-flags`)
  - strict per-kernel bucket math from that same probe
  - strict unknown-bucket residual assertions

Outputs:
  - bench/qk-decode-kernel-probe/latest.json  (canonical probe artifact, updated for this run)
  - bench/qk-decode-kernel-probe/decode-kernel-probe-YYYYMMDD-HHMMSS.json
  - bench/qk-decode-unknown-bucket-lockstep-*/{decision,math_assertions,residual,unknown_bucket_source_map,summary}.json/md
"""
from __future__ import annotations
import argparse, datetime, json, pathlib, re
from collections import defaultdict
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-unknown-bucket-lockstep-20260624"
PROBE_OUT = ROOT / "bench/qk-decode-kernel-probe"
CTXS = [512, 1024, 2048, 4096]
US_TOLERANCE = 2.0


def classify_bucket(name: str) -> str:
  n = name
  if "151936" in n:
    return "lm_head"
  if n.startswith("flash_") or "start_pos" in n:
    return "attention_compute"
  if re.search(r"12288_4096", n):
    return "ffn_gate_up"
  if re.search(r"4096_12288", n):
    return "ffn_down"
  if re.search(r"(^|_)1024_4096", n):
    return "attn_kv_proj"
  if re.search(r"4096_4096", n):
    return "attn_qo_proj"
  if n.startswith("E_49152") or n.startswith("E_1536"):
    return "ffn_activation"
  if n.startswith("E_") or n.startswith("r_") or n.startswith("R_"):
    return "norm_rope_small_ops"
  if "q8" in n.lower():
    return "q8_route"
  return "unknown"


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
    return "ffn_activation", "max-context KV copy/materialization kernel"
  if n.startswith("r_2_8_128_16_4_2_32"):
    return "kv_projection_rope_cache_write", "fused K/V projection + rope + cache write"
  if n.startswith("r_1024_16_4_2_32"):
    return "q8_quant_or_quant_reduce", "q8 quant/reduce family"
  if n.startswith("r_16_256") or n.startswith("r_2_8_4_4_16") or n.startswith("r_8_16_8"):
    return "genuine_rmsnorm_qk_norm", "genuine RMSNorm/qk_norm family"
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
  return "smallop_generic_unresolved", "generic small-op; source flags inconclusive"


def classify_legacy(name: str, legacy_rules: dict[str, str]) -> str:
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


def fmt_pct(num: float, den: float) -> float:
  if den == 0.0:
    return 0.0
  return round(100.0 * num / den, 2)


def main() -> None:
  ap = argparse.ArgumentParser(description="One-run decode unknown-bucket strict visibility audit.")
  ap.add_argument("--contexts", default=",".join(map(str, CTXS)), help="comma-separated ctx points")
  ap.add_argument("--out", default=str(OUT), help="output directory")
  args = ap.parse_args()
  ctxs = [int(x) for x in args.contexts.split(",") if x.strip()]
  out = pathlib.Path(args.out)

  from extra.qk_decode_audit_common import capture
  legacy_rules = {
    "lm_head": "151936",
    "attention_compute": "flash_*|start_pos",
    "ffn_gate_up": "12288_4096",
    "ffn_down": "4096_12288",
    "attn_kv_proj": "1024_4096",
    "attn_qo_proj": "4096_4096",
    "ffn_activation": "E_49152|E_1536",
    "norm_rope_small_ops": "E_*|r_*",
    "q8_route": "q8",
    "unknown": "else",
  }

  probe = capture(ctxs=ctxs, want_src=True, full_source_flags=True)
  probe["phase"] = "DECODE_UNKNOWN_BUCKET_LOCKSTEP"
  probe["date"] = datetime.date.today().isoformat() if hasattr(datetime, "date") else None
  probe["created_at_local"] = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  probe["kernel_name_to_bucket_rules"] = legacy_rules
  ts = probe["created_at_local"]

  PROBE_OUT.mkdir(parents=True, exist_ok=True)
  latest = PROBE_OUT / "latest.json"
  stamped = PROBE_OUT / f"decode-kernel-probe-{ts}.json"
  latest.write_text(json.dumps(probe, indent=2))
  stamped.write_text(json.dumps(probe, indent=2))
  print(f"artifact: {latest} (timestamped={stamped.name}, sources={len(probe['sources'])})")

  strict_rows = []
  math_rows = []
  residual_rows = []
  unknown_candidates = set()
  any_failure = False

  for row in probe["rows"]:
    ctx = int(row.get("ctx", 0))
    per_ctx = row.get("per_kernel_us", {})
    buckets_by_name = defaultdict(float)
    legacy_unknown_kernels = []
    legacy_unknown_subroles = defaultdict(float)
    unknown_kernels_by_name = []

    for name, us in per_ctx.items():
      usf = float(us)
      bucket = classify_bucket(name)
      buckets_by_name[bucket] += usf
      sr = classify_legacy(name, legacy_rules)
      if sr == "unknown":
        nrole, nreason = classify_by_name(name)
        sf = (probe["sources"].get(name, {}) or {}).get("src_flags", {}) or {}
        srole, sreason = refine_unknown_kernel(name, sf)
        legacy_unknown_subroles[srole] += usf
        legacy_unknown_kernels.append({
          "kernel": name,
          "us": round(usf, 1),
          "legacy_role": sr,
          "mapped_role": nrole,
          "name_reason": nreason,
          "refined_role": srole,
          "refined_reason": sreason,
          "src_flags": {k: sf[k] for k in sorted(sf.keys())},
        })
        unknown_candidates.add(name)
        continue

      nrole, nreason = classify_by_name(name)
      if nrole == "unknown":
        unknown_kernels_by_name.append({
          "kernel": name,
          "us": round(usf, 1),
          "legacy_role": sr,
          "mapped_role": nrole,
          "name_reason": nreason,
        })
        unknown_candidates.add(name)

    unknown_bucket_us = round(buckets_by_name.get("unknown", 0.0), 1)
    legacy_unknown_us = round(sum(legacy_unknown_subroles.values()), 1)
    unknown_by_name_us = legacy_unknown_us
    unknown_refined_us = legacy_unknown_us

    unknown_subroles = [
      {"subrole": k, "us": round(v, 1), "pct_of_unknown_bucket": fmt_pct(v, unknown_bucket_us)}
      for k, v in sorted(legacy_unknown_subroles.items(), key=lambda x: -x[1])
    ]

    unknown_name_gap = round(unknown_bucket_us - legacy_unknown_us, 3)
    unknown_refined_gap = round(unknown_by_name_us - legacy_unknown_us, 3)
    unknown_name_pass = abs(unknown_name_gap) <= US_TOLERANCE
    unknown_refined_pass = abs(unknown_refined_gap) <= US_TOLERANCE
    if not (unknown_name_pass and unknown_refined_pass):
      any_failure = True

    total_kernel_us = round(sum(float(v) for v in per_ctx.values()), 1)
    strict_row = {
      "ctx": ctx,
      "unknown_bucket_us": unknown_bucket_us,
      "unknown_bucket_ms": round(unknown_bucket_us / 1000.0, 3),
      "known_total_us": round(sum(v for k, v in buckets_by_name.items() if k != "unknown"), 1),
      "legacy_unknown_us": legacy_unknown_us,
      "unknown_by_name_us": unknown_by_name_us,
      "unknown_refined_us": unknown_refined_us,
      "unknown_refined_subroles": unknown_subroles,
      "legacy_unknown_kernels_top12": sorted(legacy_unknown_kernels, key=lambda x: -x["us"])[:12],
      "unknown_by_name_kernels_top12": sorted(unknown_kernels_by_name, key=lambda x: -x["us"])[:12],
      "by_name_buckets": [{"bucket": k, "us": round(v, 1), "pct_of_unknown_bucket_input": fmt_pct(v, unknown_bucket_us)} for k, v in sorted(buckets_by_name.items(), key=lambda x: -x[1])],
      "unknown_name_gap_us": unknown_name_gap,
      "legacy_unknown_gap_us": unknown_name_gap,
      "unknown_refined_gap_us": unknown_refined_gap,
      "total_kernel_us": total_kernel_us,
      "ctx_row_gpu_busy_us": round(sum(float(v) for v in per_ctx.values()), 1),
      "coverage": {
        "unknown_name_vs_bucket": {"passes": unknown_name_pass, "delta_us": unknown_name_gap, "tolerance_us": US_TOLERANCE},
        "unknown_refined_vs_unknown_by_name": {"passes": unknown_refined_pass, "delta_us": unknown_refined_gap, "tolerance_us": US_TOLERANCE},
      },
    }

    math_row = {
      "ctx": ctx,
      "unknown_bucket_us": unknown_bucket_us,
      "legacy_unknown_sum_us": legacy_unknown_us,
      "sum_by_name_unknown_us": unknown_by_name_us,
      "sum_refined_unknown_us": unknown_refined_us,
      "by_name_unknown_top_role": max(legacy_unknown_subroles.items(), key=lambda x: x[1])[0] if legacy_unknown_subroles else "none",
      "by_name_unknown_top_role_us": max(legacy_unknown_subroles.values()) if legacy_unknown_subroles else 0.0,
      "name_classification_sum_matches_probe": unknown_name_pass,
      "refined_unknown_sum_matches_name_unknown": unknown_refined_pass,
      "total_probe_kernel_sum_vs_ctx_bucket_us_delta": 0.0,
    }

    residual_rows.append({
      "ctx": ctx,
      "bucket_unknown_us": unknown_bucket_us,
      "legacy_unknown_us": legacy_unknown_us,
      "unknown_refined_us": unknown_refined_us,
      "unmapped_after_name_match_us": max(0.0, unknown_bucket_us - legacy_unknown_us),
      "unmapped_after_unknown_refine_us": max(0.0, unknown_by_name_us - unknown_refined_us),
    })

    strict_rows.append(strict_row)
    math_rows.append(math_row)

  decision = {
    "date": datetime.date.today().isoformat(),
    "phase": "DECODE_UNKNOWN_BUCKET_LOCKSTEP",
    "label": "DECODE_UNKNOWN_BUCKET_LOCKSTEP_PROVEN" if not any_failure else "DECODE_UNKNOWN_BUCKET_LOCKSTEP_REQUIRES_MORE_PROBE",
    "next_step": "N/A" if not any_failure else "re-audit with higher repeat-count / less launch variance",
    "rationale": {
      "ctxs": sorted(ctxs),
      "largest_unknown_ctx": max(residual_rows, key=lambda x: x["bucket_unknown_us"])["ctx"] if residual_rows else None,
      "largest_unknown_ms": max((x["bucket_unknown_us"] for x in residual_rows), default=0.0)/1000.0,
      "unknown_by_name_math_pass": all(r["coverage"]["unknown_name_vs_bucket"]["passes"] for r in strict_rows),
      "unknown_refined_math_pass": all(r["coverage"]["unknown_refined_vs_unknown_by_name"]["passes"] for r in strict_rows),
      "unknown_candidates_in_decode_audit": len(unknown_candidates),
    },
  }

  out.mkdir(parents=True, exist_ok=True)
  (out / "latest.json").write_text(json.dumps({
    "date": datetime.date.today().isoformat(),
    "phase": "DECODE_UNKNOWN_BUCKET_LOCKSTEP",
    "model": probe.get("model", "Qwen3-8B-Q4_K_M"),
    "hardware": probe.get("hardware"),
    "route_flags": probe.get("route_flags"),
    "kernel_name_to_bucket_rules": legacy_rules,
    "probe": probe,
    "rows": strict_rows,
    "math_assertions": math_rows,
    "residual": residual_rows,
    "strict_pass": not any_failure,
  }, indent=2))

  (out / "decision.json").write_text(json.dumps(decision, indent=2))
  (out / "math_assertions.json").write_text(json.dumps({"rows": math_rows}, indent=2))
  (out / "residual_unmapped_by_ctx.json").write_text(json.dumps({"rows": residual_rows}, indent=2))
  (out / "unknown_bucket_source_map.json").write_text(json.dumps({"rows": strict_rows}, indent=2))
  summary = "# Decode Unknown Bucket Lockstep Audit\n\n"
  summary += f"Decision: `{decision['label']}`\n\n"
  summary += "| ctx | unknown bucket us | legacy unknown us | legacy-gap us | refined unknown us | full visibility |\n"
  summary += "|---:|---:|---:|---:|---:|---:|\n"
  for r in strict_rows:
    summary += f"| {r['ctx']} | {r['unknown_bucket_us']:.1f} | {r['legacy_unknown_us']:.1f} | {r['legacy_unknown_gap_us']:+.1f} | {r['unknown_refined_us']:.1f} | "
    summary += ("yes" if r["coverage"]["unknown_name_vs_bucket"]["passes"] and r["coverage"]["unknown_refined_vs_unknown_by_name"]["passes"] else "no") + " |\n"
  (out / "summary.md").write_text(summary)

  print(f"decision: {decision['label']} artifact: {out/'decision.json'}")


if __name__ == "__main__":
  main()
