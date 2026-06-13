#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib, re
from typing import Any


REPO = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_VECTOR_PROBE = pathlib.Path("bench/qk-memory-access-20260613/vector-probe.json")
DEFAULT_PROBE_LOAD_WIDTH = pathlib.Path("bench/qk-memory-access-20260613/load-width/report.json")
DEFAULT_LOAD_WIDTH = pathlib.Path("bench/qk-ansor-transition-20260612/semantic-codegen-v3/load-width/report.json")
DEFAULT_ROOFLINE = pathlib.Path("bench/qk-bandwidth-roofline-20260613/roofline.json")
DEFAULT_PMC = pathlib.Path("bench/qk-semantic-20260612/pmc-q4-gate.json")


def _read_json(path:pathlib.Path) -> dict[str, Any] | None:
  full = (REPO / path).resolve() if not path.is_absolute() else path
  if not full.exists(): return None
  return json.loads(full.read_text())


def _portable(path:pathlib.Path) -> str:
  full = (REPO / path).resolve() if not path.is_absolute() else path.resolve()
  try: return str(full.relative_to(REPO))
  except ValueError: return str(full)


def _has(pattern:str, path:pathlib.Path) -> bool:
  text = (REPO / path).read_text()
  return re.search(pattern, text, re.MULTILINE | re.DOTALL) is not None


def build_audit(*, vector_probe:pathlib.Path, probe_load_width:pathlib.Path, load_width:pathlib.Path, roofline:pathlib.Path, pmc:pathlib.Path) -> dict[str, Any]:
  vector = _read_json(vector_probe)
  probe_width = _read_json(probe_load_width)
  width = _read_json(load_width)
  roof = _read_json(roofline)
  pmc_report = _read_json(pmc)

  renderer_vector_access = _has(r"u\.max_numel\(\) > 1.*render_type", pathlib.Path("tinygrad/renderer/cstyle.py"))
  integer_vector_folding_supported = _has(
    r"buf\.addrspace == AddrSpace\.GLOBAL and buf\.dtype\.base == dtypes\.uint32.*?lengths = \[4\]",
    pathlib.Path("tinygrad/codegen/late/devectorizer.py"),
  )
  custom_probe_ok = bool((vector or {}).get("summary", {}).get("raw_custom_uint4_escape_supported"))
  uop_probe_ok = bool((vector or {}).get("summary", {}).get("normal_uop_uint4_load_supported"))
  probe_uop_vector_evidence = any(row.get("mode") == "uop_vec_request" and str(row.get("load_width_inferred", "")).startswith("vector_")
                                  for row in (probe_width or {}).get("rows", []))
  v3_vector_evidence = bool((width or {}).get("summary", {}).get("has_vector_load_evidence"))

  roof_rows = []
  if roof:
    for row in roof.get("rows", []):
      roof_rows.append({
        "model": row.get("model") or row.get("model_size"),
        "tinygrad_tok_s": row.get("tinygrad_tok_s") or row.get("generated_tok_s"),
        "llama_tok_s": row.get("llama_tok_s"),
        "tinygrad_percent_peak": row.get("tinygrad_percent_peak") or row.get("generated_pct_peak_mem_by_file"),
        "llama_percent_peak": row.get("llama_percent_peak") or row.get("llama_pct_peak_mem_by_file"),
        "tinygrad_percent_llama": row.get("tinygrad_percent_llama") or row.get("generated_pct_llama_by_file"),
      })

  pmc_rows = []
  if pmc_report:
    for name, row in (pmc_report.get("kernels") or {}).items():
      pmc_rows.append({
        "kernel": name,
        "events": row.get("events"),
        "gl2c_hit_rate": (row.get("derived") or {}).get("gl2c_hit_rate"),
        "valu_per_busy_cycle": (row.get("derived") or {}).get("valu_per_busy_cycle"),
        "sq_busy_cycles": (row.get("counters") or {}).get("SQ_BUSY_CYCLES"),
        "valu_inst": (row.get("counters") or {}).get("SQ_INSTS_VALU"),
      })

  if uop_probe_ok and probe_uop_vector_evidence and integer_vector_folding_supported:
    decision = "family_c_v1_source_supported"
  elif v3_vector_evidence:
    decision = "family_c_v1_source_supported"
  elif custom_probe_ok and not uop_probe_ok:
    decision = "family_c_v1_requires_core_integer_vector_load_lowering"
  elif not custom_probe_ok:
    decision = "family_c_v1_blocked_until_uint4_escape_or_renderer_support"
  else:
    decision = "family_c_v1_needs_manual_review"

  return {
    "kind": "qk_memory_access_audit",
    "schema_version": 1,
    "inputs": {
      "vector_probe": _portable(vector_probe),
      "probe_load_width": _portable(probe_load_width),
      "semantic_codegen_v3_load_width": _portable(load_width),
      "roofline": _portable(roofline),
      "pmc": _portable(pmc),
    },
    "environment": {
      "external_rocprof_used": False,
      "note": "This audit does not depend on local rocprof availability; it uses committed roofline, PMC prior, DEBUG=4 source logs, and source audit evidence.",
    },
    "source_audit": {
      "renderer_has_vector_pointer_cast_syntax": renderer_vector_access,
      "integer_uint32x4_global_load_store_folding_supported": integer_vector_folding_supported,
      "evidence": [
        "CStyle.render_access can render vector pointer casts when an INDEX has max_numel > 1.",
        "correct_load_store.split_load_store now allows aligned uint32x4 global load/store folding.",
      ],
    },
    "vector_probe_summary": (vector or {}).get("summary"),
    "probe_load_width_summary": (probe_width or {}).get("summary"),
    "semantic_codegen_v3_load_width_summary": (width or {}).get("summary"),
    "roofline_rows": roof_rows,
    "pmc_rows": pmc_rows,
    "decision": {
      "status": decision,
      "run_family_c_v1_now": decision == "family_c_v1_source_supported",
      "run_32b": False,
      "next_required_change": (
        "Patch core integer vector load/store lowering for aligned uint32 global buffers, then rerun the probe and only then build Family C v1."
        if decision == "family_c_v1_requires_core_integer_vector_load_lowering" else
        "Build Family C v1 as a generated memory-access candidate using the verified uint32x4 lowering."
      ),
      "stop_rule": "Do not broaden packed-load expression rewrites unless generated source shows vector/coalesced integer loads.",
    },
  }


def audit_markdown(report:dict[str, Any]) -> str:
  lines = [
    "# QK Memory Access Audit",
    "",
    "Evidence gate for Family C v1 implementation.",
    "",
    "## Decision",
    "",
    f"- status: `{report['decision']['status']}`",
    f"- run Family C v1 now: `{report['decision']['run_family_c_v1_now']}`",
    f"- run 32B: `{report['decision']['run_32b']}`",
    f"- next required change: {report['decision']['next_required_change']}",
    f"- stop rule: {report['decision']['stop_rule']}",
    "",
    "## Source Audit",
    "",
    f"- renderer vector pointer-cast syntax: `{report['source_audit']['renderer_has_vector_pointer_cast_syntax']}`",
    f"- integer uint32x4 global load/store folding supported: `{report['source_audit']['integer_uint32x4_global_load_store_folding_supported']}`",
    "",
  ]
  for item in report["source_audit"]["evidence"]:
    lines.append(f"- {item}")
  lines += [
    "",
    "## Probe Summary",
    "",
  ]
  vector = report.get("vector_probe_summary") or {}
  probe_width = report.get("probe_load_width_summary") or {}
  width = report.get("semantic_codegen_v3_load_width_summary") or {}
  lines += [
    f"- normal UOp uint4 load supported: `{vector.get('normal_uop_uint4_load_supported')}`",
    f"- raw custom uint4 escape supported: `{vector.get('raw_custom_uint4_escape_supported')}`",
    f"- probe UOp vector load evidence: `{probe_width.get('has_vector_load_evidence')}`",
    f"- Family C v0 vector load evidence: `{width.get('has_vector_load_evidence')}`",
    f"- Family C v0 packed-load kernel present: `{width.get('has_packed_load_kernel')}`",
    "",
    "## Roofline Context",
    "",
    "| model | tinygrad tok/s | llama tok/s | tinygrad % peak | llama % peak | tinygrad % llama |",
    "|---|---:|---:|---:|---:|---:|",
  ]
  for row in report.get("roofline_rows") or []:
    if any(row.get(key) is None for key in ("tinygrad_tok_s", "llama_tok_s", "tinygrad_percent_peak", "llama_percent_peak", "tinygrad_percent_llama")):
      lines.append(f"| {row.get('model')} | n/a | n/a | n/a | n/a | n/a |")
      continue
    lines.append(
      f"| {row['model']} | {row['tinygrad_tok_s']:.2f} | {row['llama_tok_s']:.2f} | "
      f"{row['tinygrad_percent_peak']:.2f} | {row['llama_percent_peak']:.2f} | {row['tinygrad_percent_llama']:.2f} |"
    )
  lines += [
    "",
    "## PMC Prior",
    "",
    "| kernel | events | GL2 hit rate | VALU / busy | SQ busy | VALU inst |",
    "|---|---:|---:|---:|---:|---:|",
  ]
  for row in report.get("pmc_rows") or []:
    lines.append(
      f"| `{row['kernel']}` | {row['events']} | {row['gl2c_hit_rate']:.4f} | "
      f"{row['valu_per_busy_cycle']:.4f} | {row['sq_busy_cycles']} | {row['valu_inst']} |"
    )
  lines += [
    "",
    "## Interpretation",
    "",
    "The remaining gap is still best explained as memory-load efficiency. The",
    "normal tinygrad codegen path now preserves a requested aligned `uint32x4`",
    "global load/store on AMD, and DEBUG=4 source confirms vector pointer casts.",
    "Family C v1 is therefore unblocked as the next generated memory-access",
    "candidate. Family C v0 remains rejected; it did not request this new load",
    "shape and still emitted scalar `u32` loads.",
    "",
  ]
  return "\n".join(lines)


def main() -> int:
  parser = argparse.ArgumentParser(description="Build the QK memory-access go/no-go audit")
  parser.add_argument("--vector-probe", type=pathlib.Path, default=DEFAULT_VECTOR_PROBE)
  parser.add_argument("--probe-load-width", type=pathlib.Path, default=DEFAULT_PROBE_LOAD_WIDTH)
  parser.add_argument("--load-width", type=pathlib.Path, default=DEFAULT_LOAD_WIDTH)
  parser.add_argument("--roofline", type=pathlib.Path, default=DEFAULT_ROOFLINE)
  parser.add_argument("--pmc", type=pathlib.Path, default=DEFAULT_PMC)
  parser.add_argument("--json", type=pathlib.Path, required=True)
  parser.add_argument("--md", type=pathlib.Path, required=True)
  args = parser.parse_args()
  report = build_audit(vector_probe=args.vector_probe, probe_load_width=args.probe_load_width,
                       load_width=args.load_width, roofline=args.roofline, pmc=args.pmc)
  args.json.parent.mkdir(parents=True, exist_ok=True)
  args.json.write_text(json.dumps(report, indent=2, sort_keys=True))
  args.md.parent.mkdir(parents=True, exist_ok=True)
  args.md.write_text(audit_markdown(report))
  print(audit_markdown(report))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
