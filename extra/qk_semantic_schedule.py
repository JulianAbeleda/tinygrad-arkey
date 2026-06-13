#!/usr/bin/env python3
from __future__ import annotations

import argparse, copy, json, pathlib
from typing import Any

from extra.qk_descriptor_policy import build_policy_from_descriptor, load_json, write_json
from extra.qk_semantic_candidate import (
  correctness_provenance, current_runtime, no_extra_storage_effect, runtime_storage_bytes, slug,
)

RUNTIME_FAMILIES = {"Q4_K": "q4_k_packed_u32", "Q6_K": "q6_k_packed_u16"}
MICROBENCH_FAMILIES = RUNTIME_FAMILIES | {"Q4_K_DIRECT": "q4_k_packed_u32_direct"}
DOMINANT_ROLES = ("ffn_gate", "ffn_down", "attn_q")

def _candidate_requires(fmt:str, family:str, codegen_mode:str) -> list[str]:
  if fmt == "Q4_K" and codegen_mode == "direct_out": return ["q4k_gemv_kernel", "u32_packed_storage"]
  if fmt == "Q4_K": return ["q4k_gemv_partial_kernel", "u32_packed_storage"]
  if fmt == "Q6_K": return ["q6k_gemv_partial_kernel", "u16_packed_storage"]
  return []


def _spec(row:dict[str, Any], name:str, *, family:str, parts:int, opts:list[str], codegen_mode:str,
          row_tile:int|None=None, k_tile_blocks:int|None=None, lane_width:int=1,
          group_unroll:int=1, full_decode_supported:bool=True) -> dict[str, Any]:
  fmt = row["format"]
  storage_effect = no_extra_storage_effect("schedule-only candidate reuses existing packed-weight storage")
  return {
    "name": name,
    "semantic_object": "packed_quant_gemv_schedule",
    "format": fmt,
    "role": row.get("role"),
    "family": family,
    "parts": parts,
    "opts": opts,
    "codegen_mode": codegen_mode,
    "row_tile": row_tile,
    "k_tile_blocks": k_tile_blocks,
    "lane_width": lane_width,
    "group_unroll": group_unroll,
    "activation_cache": "fp16_direct",
    "reduction_mode": "direct_out" if codegen_mode == "direct_out" else "split_k_partial",
    "requires": _candidate_requires(fmt, family, codegen_mode),
    "full_decode_supported": full_decode_supported,
    "storage_effect": storage_effect,
    "correctness_provenance": correctness_provenance(full_decode_supported=full_decode_supported),
  }


def schedule_specs_for_row(row:dict[str, Any]) -> list[dict[str, Any]]:
  fmt, role = row.get("format"), row.get("role")
  current = current_runtime(row)
  if fmt not in RUNTIME_FAMILIES or current["family"] != RUNTIME_FAMILIES[fmt]: return []
  if role not in DOMINANT_ROLES: return []
  local_arg = None
  for opt in current["opts"]:
    parts = str(opt).split(":")
    if len(parts) == 3 and parts[0] == "LOCAL" and parts[1] == "0": local_arg = int(parts[2])
  row_tile = local_arg
  out = []
  if fmt == "Q4_K" and current["parts"] == 1:
    out.append(_spec(row, "direct_out", family="q4_k_packed_u32_direct", parts=1, opts=current["opts"],
                     codegen_mode="direct_out", row_tile=row_tile, full_decode_supported=False))
  if local_arg is not None:
    out.append(_spec(row, "row_upcast2", family=current["family"], parts=current["parts"],
                     opts=current["opts"] + ["UPCAST:0:2"], codegen_mode="partial", row_tile=row_tile, lane_width=2))
    out.append(_spec(row, "reduce_unroll4", family=current["family"], parts=current["parts"],
                     opts=current["opts"] + ["UNROLL:2:4"], codegen_mode="partial", row_tile=row_tile,
                     k_tile_blocks=4, group_unroll=4))
    alt_local = 32 if local_arg != 32 else 64
    out.append(_spec(row, "two_dim_local4", family=current["family"], parts=current["parts"],
                     opts=[f"LOCAL:0:{alt_local}", "LOCAL:1:4"], codegen_mode="partial", row_tile=alt_local))
  return out


def _apply_schedule(entry:dict[str, Any], spec:dict[str, Any]) -> None:
  cand = entry["candidate"]
  cand["family"] = spec["family"]
  cand["name"] = spec["name"]
  cand["opts"] = list(spec["opts"])
  cand["parts"] = int(spec["parts"])
  cand["reduction"] = spec["reduction_mode"]
  cand["requires"] = list(spec["requires"])
  cand["schedule_spec"] = copy.deepcopy(spec)
  entry["winner"] = spec["name"]
  entry["policy_reason"] = "semantic schedule/codegen candidate"
  entry["reason"] = "single-entry semantic schedule candidate"


def _policy_with_schedule(descriptor:dict[str, Any], tensor:str | None=None, spec:dict[str, Any] | None=None) -> dict[str, Any]:
  policy = build_policy_from_descriptor(descriptor)
  policy["kind"] = "qk_generated_policy"
  policy["created_at"] = "generated-by-qk-semantic-schedule"
  if tensor is None: return policy
  assert spec is not None
  for entry in policy["entries"]:
    if entry.get("tensor") == tensor:
      _apply_schedule(entry, spec)
      break
  else:
    raise ValueError(f"descriptor has no tensor {tensor}")
  return policy


def build_schedule_candidate_set(descriptor:dict[str, Any], *, max_candidates:int | None=None) -> dict[str, Any]:
  if descriptor.get("kind") != "qk_semantic_descriptor_set":
    raise ValueError("expected kind=qk_semantic_descriptor_set")
  baseline = _policy_with_schedule(descriptor)
  baseline_storage_bytes = runtime_storage_bytes(baseline)
  candidates = [{
    "id": "current",
    "status": "baseline",
    "description": "current descriptor policy",
    "changes": [],
    "policy": baseline,
    "expected_storage_bytes": baseline_storage_bytes,
    "storage_effect": no_extra_storage_effect("baseline descriptor policy"),
  }]
  count = 0
  for row in descriptor.get("descriptors", []):
    for spec in schedule_specs_for_row(row):
      count += 1
      if max_candidates is not None and count > max_candidates: break
      policy = _policy_with_schedule(descriptor, row["tensor"], spec)
      current = current_runtime(row)
      storage_effect = spec["storage_effect"]
      candidates.append({
        "id": f"{count:03d}-{slug(row.get('role') or row['tensor'])}-{slug(row['tensor'])}-{slug(spec['name'])}",
        "status": "candidate",
        "description": "single-entry semantic schedule/codegen candidate",
        "schedule_spec": spec,
        "changes": [{
          "tensor": row["tensor"],
          "format": row.get("format"),
          "role": row.get("role"),
          "from": current,
          "to": {
            "winner": spec["name"],
            "family": spec["family"],
            "parts": spec["parts"],
            "opts": spec["opts"],
            "reduction": spec["reduction_mode"],
          },
          "schedule_spec": spec,
          "storage_effect": storage_effect,
          "correctness_provenance": spec["correctness_provenance"],
        }],
        "policy": policy,
        "expected_storage_bytes": baseline_storage_bytes + int(storage_effect["persistent_bytes_delta"]),
        "storage_effect": storage_effect,
        "correctness_provenance": spec["correctness_provenance"],
      })
    if max_candidates is not None and count >= max_candidates: break
  return {
    "kind": "qk_semantic_schedule_candidate_set",
    "schema_version": 1,
    "model": descriptor.get("model"),
    "model_size": descriptor.get("model_size"),
    "source_descriptor": descriptor.get("source_policy"),
    "gate_models": ["8B", "14B"],
    "search_space": {
      "scope": "dominant current Q4_K/Q6_K primitive descriptors only",
      "roles": list(DOMINANT_ROLES),
      "axes": ["codegen_mode", "row_tile", "k_tile_blocks", "lane_width", "group_unroll", "reduction_mode"],
      "note": "32B is excluded from the default semantic schedule gate and should only run after 8B/14B evidence.",
    },
    "candidates": candidates,
    "summary": {
      "candidates": len(candidates),
      "single_change_candidates": len(candidates) - 1,
      "current_storage_bytes": baseline_storage_bytes,
    },
  }


def build_static_gate(candidate_set:dict[str, Any]) -> dict[str, Any]:
  rows = []
  for cand in candidate_set.get("candidates", []):
    if cand["id"] == "current":
      rows.append({"id": "current", "status": "baseline", "microbench": False, "full_decode_supported": True, "reasons": []})
      continue
    changes = cand.get("changes") or []
    reasons = []
    if len(changes) != 1: reasons.append("candidate must change exactly one descriptor")
    spec = cand.get("schedule_spec") or (changes[0].get("schedule_spec") if changes else {})
    if spec.get("format") not in RUNTIME_FAMILIES: reasons.append(f"unsupported format {spec.get('format')!r}")
    if spec.get("family") not in set(RUNTIME_FAMILIES.values()) | {"q4_k_packed_u32_direct"}:
      reasons.append(f"unsupported microbench family {spec.get('family')!r}")
    if int(spec.get("parts") or 0) < 1: reasons.append("parts must be positive for primitive schedule candidates")
    opts = list(spec.get("opts") or [])
    for opt in opts:
      op = str(opt).split(":", 1)[0]
      if op not in {"LOCAL", "UPCAST", "UNROLL"}: reasons.append(f"unsupported opt op {op!r}")
    rows.append({
      "id": cand["id"],
      "status": "pass" if not reasons else "fail",
      "microbench": not reasons,
      "full_decode_supported": bool(spec.get("full_decode_supported")) and spec.get("family") in set(RUNTIME_FAMILIES.values()),
      "schedule": spec,
      "changes": changes,
      "expected_storage_bytes": cand.get("expected_storage_bytes"),
      "storage_effect": cand.get("storage_effect") or spec.get("storage_effect"),
      "correctness_provenance": cand.get("correctness_provenance") or spec.get("correctness_provenance"),
      "reasons": reasons,
    })
  return {
    "kind": "qk_semantic_schedule_static_gate",
    "model": candidate_set.get("model"),
    "model_size": candidate_set.get("model_size"),
    "source_candidates": candidate_set.get("source_descriptor"),
    "rows": rows,
    "summary": {
      "candidates": len(rows),
      "passing_microbench": sum(1 for row in rows if row.get("microbench")),
      "full_decode_supported": sum(1 for row in rows if row.get("full_decode_supported")),
      "failing": sum(1 for row in rows if row["status"] == "fail"),
    },
  }


def candidates_markdown(candidate_set:dict[str, Any]) -> str:
  lines = [
    f"# QK Semantic Schedule Candidates: {candidate_set['model_size']}",
    "",
    "Second-stage Ansor-transition surface. These candidates carry semantic",
    "schedule/codegen specs instead of only varying `parts` and `LOCAL`.",
    "",
    "## Summary",
    "",
    f"- candidates: `{candidate_set['summary']['candidates']}`",
    f"- single-change candidates: `{candidate_set['summary']['single_change_candidates']}`",
    f"- current storage bytes: `{candidate_set['summary']['current_storage_bytes']}`",
    "",
    "| id | tensor | schedule | family | parts | opts | persistent delta | metadata sidecar | full decode |",
    "|---|---|---|---|---:|---|---:|---:|---:|",
  ]
  for cand in candidate_set["candidates"]:
    if cand["id"] == "current":
      lines.append("| `current` | n/a | current | n/a | 0 | n/a | 0 | 0 | `True` |")
      continue
    change = cand["changes"][0]
    spec = cand["schedule_spec"]
    storage = spec.get("storage_effect") or {}
    lines.append(f"| `{cand['id']}` | `{change['tensor']}` | `{spec['name']}` | `{spec['family']}` | "
                 f"{spec['parts']} | `{','.join(spec['opts'])}` | {storage.get('persistent_bytes_delta', 0)} | "
                 f"{storage.get('metadata_sidecar_bytes', 0)} | `{spec['full_decode_supported']}` |")
  lines.append("")
  return "\n".join(lines)


def static_gate_markdown(report:dict[str, Any]) -> str:
  lines = [
    f"# QK Semantic Schedule Static Gate: {report['model_size']}",
    "",
    "Fail-closed structural validation before microbench. Passing here does not",
    "mean the schedule compiles or wins; it only means the candidate is shaped",
    "well enough to test in an isolated microbench.",
    "",
    "## Summary",
    "",
    f"- candidates: `{report['summary']['candidates']}`",
    f"- passing microbench: `{report['summary']['passing_microbench']}`",
    f"- full-decode supported: `{report['summary']['full_decode_supported']}`",
    f"- failing: `{report['summary']['failing']}`",
    "",
    "| id | status | microbench | full decode | persistent delta | metadata sidecar | reasons |",
    "|---|---|---:|---:|---:|---:|---|",
  ]
  for row in report["rows"]:
    reasons = "; ".join(row.get("reasons") or []) or "none"
    storage = row.get("storage_effect") or {}
    lines.append(f"| `{row['id']}` | `{row['status']}` | `{row.get('microbench')}` | "
                 f"`{row.get('full_decode_supported')}` | {storage.get('persistent_bytes_delta', 0)} | "
                 f"{storage.get('metadata_sidecar_bytes', 0)} | {reasons} |")
  lines.append("")
  return "\n".join(lines)


def main() -> int:
  parser = argparse.ArgumentParser(description="Generate semantic QK schedule/codegen candidates")
  parser.add_argument("--descriptor", type=pathlib.Path, required=True)
  parser.add_argument("--json", type=pathlib.Path, required=True)
  parser.add_argument("--md", type=pathlib.Path)
  parser.add_argument("--gate-json", type=pathlib.Path)
  parser.add_argument("--gate-md", type=pathlib.Path)
  parser.add_argument("--max-candidates", type=int)
  args = parser.parse_args()
  candidate_set = build_schedule_candidate_set(load_json(args.descriptor.expanduser()), max_candidates=args.max_candidates)
  write_json(args.json, candidate_set)
  if args.md:
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text(candidates_markdown(candidate_set))
  if args.gate_json:
    gate = build_static_gate(candidate_set)
    write_json(args.gate_json, gate)
    if args.gate_md:
      args.gate_md.parent.mkdir(parents=True, exist_ok=True)
      args.gate_md.write_text(static_gate_markdown(gate))
  if not args.md:
    print(candidates_markdown(candidate_set))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
