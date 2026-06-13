#!/usr/bin/env python3
from __future__ import annotations

import argparse, copy, json, pathlib
from typing import Any

from extra.qk_descriptor_policy import build_policy_from_descriptor, load_json, write_json
from extra.qk_semantic_candidate import (
  correctness_provenance, current_runtime, no_extra_storage_effect, runtime_storage_bytes, slug,
)

RUNTIME_Q4_FAMILY = "q4_k_packed_u32"
GROUPED_Q4_FAMILY = "q4_k_packed_u32_grouped"
TARGET_ROLE = "ffn_down"
ROW_GROUPS = (2, 4)


def _grouped_spec(row:dict[str, Any], *, row_group:int) -> dict[str, Any]:
  current = current_runtime(row)
  opts = [opt for opt in current["opts"] if not str(opt).startswith("UPCAST:1:")]
  opts.append(f"UPCAST:1:{row_group}")
  storage_effect = no_extra_storage_effect("row-grouped codegen reuses existing Q4_K packed-weight storage")
  return {
    "name": f"row_group{row_group}",
    "semantic_object": "packed_quant_gemv_codegen",
    "format": row["format"],
    "role": row.get("role"),
    "family": GROUPED_Q4_FAMILY,
    "parts": int(current["parts"]),
    "opts": opts,
    "codegen_mode": "grouped_partial",
    "reduction_mode": "split_k_partial",
    "scope": "tensor",
    "row_group": row_group,
    "k_tile_blocks": 1,
    "activation_cache": "fp16_row_group_shared",
    "requires": ["q4k_gemv_grouped_partial_kernel", "u32_packed_storage"],
    "full_decode_supported": False,
    "storage_effect": storage_effect,
    "correctness_provenance": correctness_provenance(full_decode_supported=False),
  }


def codegen_specs_for_row(row:dict[str, Any]) -> list[dict[str, Any]]:
  current = current_runtime(row)
  if row.get("format") != "Q4_K": return []
  if row.get("role") != TARGET_ROLE: return []
  if current["family"] != RUNTIME_Q4_FAMILY: return []
  rows = int((row.get("shape") or {}).get("rows") or 0)
  if rows <= 0: return []
  return [_grouped_spec(row, row_group=rg) for rg in ROW_GROUPS if rows % rg == 0]


def _apply_codegen(entry:dict[str, Any], spec:dict[str, Any]) -> None:
  cand = entry["candidate"]
  cand["family"] = spec["family"]
  cand["name"] = spec["name"]
  cand["opts"] = list(spec["opts"])
  cand["parts"] = int(spec["parts"])
  cand["reduction"] = spec["reduction_mode"]
  cand["requires"] = list(spec["requires"])
  cand["schedule_spec"] = copy.deepcopy(spec)
  entry["winner"] = spec["name"]
  entry["scope"] = "tensor"
  entry["policy_reason"] = "semantic codegen v2 tensor override"
  entry["reason"] = "single-tensor row-grouped semantic codegen candidate"
  storage = entry.setdefault("storage", {})
  storage["decision"] = "tensor_codegen_v2_override"


def _policy_with_tensor_override(descriptor:dict[str, Any], tensor:str, spec:dict[str, Any]) -> dict[str, Any]:
  policy = build_policy_from_descriptor(descriptor)
  policy["kind"] = "qk_generated_policy"
  policy["created_at"] = "generated-by-qk-semantic-codegen-v2"
  for entry in policy["entries"]:
    if entry.get("tensor") == tensor:
      override = copy.deepcopy(entry)
      _apply_codegen(override, spec)
      policy["entries"].append(override)
      break
  else:
    raise ValueError(f"descriptor has no tensor {tensor}")
  return policy


def _baseline_policy(descriptor:dict[str, Any]) -> dict[str, Any]:
  policy = build_policy_from_descriptor(descriptor)
  policy["kind"] = "qk_generated_policy"
  policy["created_at"] = "generated-by-qk-semantic-codegen-v2"
  return policy


def build_codegen_candidate_set(descriptor:dict[str, Any], *, max_candidates:int | None=None) -> dict[str, Any]:
  if descriptor.get("kind") != "qk_semantic_descriptor_set":
    raise ValueError("expected kind=qk_semantic_descriptor_set")
  baseline = _baseline_policy(descriptor)
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
    for spec in codegen_specs_for_row(row):
      count += 1
      if max_candidates is not None and count > max_candidates: break
      current = current_runtime(row)
      storage_effect = spec["storage_effect"]
      policy = _policy_with_tensor_override(descriptor, row["tensor"], spec)
      candidates.append({
        "id": f"{count:03d}-{slug(row.get('role') or row['tensor'])}-{slug(row['tensor'])}-{slug(spec['name'])}",
        "status": "candidate",
        "description": "single-tensor row-grouped Q4_K codegen candidate",
        "schedule_spec": spec,
        "changes": [{
          "tensor": row["tensor"],
          "format": row.get("format"),
          "role": row.get("role"),
          "scope": "tensor",
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
    "kind": "qk_semantic_codegen_candidate_set",
    "schema_version": 2,
    "model": descriptor.get("model"),
    "model_size": descriptor.get("model_size"),
    "source_descriptor": descriptor.get("source_policy"),
    "gate_models": ["8B", "14B"],
    "search_space": {
      "scope": "single exact-tensor Q4_K ffn_down row-grouped runtime probes",
      "roles": [TARGET_ROLE],
      "axes": ["row_group", "codegen_mode", "scope"],
      "note": "32B is excluded unless the 8B/14B semantic codegen v2 gate accepts.",
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
    if spec.get("format") != "Q4_K": reasons.append(f"unsupported format {spec.get('format')!r}")
    if spec.get("role") != TARGET_ROLE: reasons.append(f"semantic codegen v2 targets only {TARGET_ROLE}")
    if spec.get("family") != GROUPED_Q4_FAMILY: reasons.append(f"unsupported codegen family {spec.get('family')!r}")
    if spec.get("scope") != "tensor": reasons.append("semantic codegen v2 requires tensor-scoped override")
    if spec.get("codegen_mode") != "grouped_partial": reasons.append("semantic codegen v2 only supports grouped_partial")
    if int(spec.get("parts") or 0) < 1: reasons.append("parts must be positive")
    if int(spec.get("row_group") or 0) not in ROW_GROUPS: reasons.append(f"unsupported row_group={spec.get('row_group')!r}")
    opts = list(spec.get("opts") or [])
    for opt in opts:
      op = str(opt).split(":", 1)[0]
      if op not in {"LOCAL", "UPCAST", "UNROLL"}: reasons.append(f"unsupported opt op {op!r}")
    rows.append({
      "id": cand["id"],
      "status": "pass" if not reasons else "fail",
      "microbench": not reasons,
      "full_decode_supported": False,
      "schedule": spec,
      "changes": changes,
      "expected_storage_bytes": cand.get("expected_storage_bytes"),
      "storage_effect": cand.get("storage_effect") or spec.get("storage_effect"),
      "correctness_provenance": cand.get("correctness_provenance") or spec.get("correctness_provenance"),
      "reasons": reasons,
    })
  return {
    "kind": "qk_semantic_codegen_v2_static_gate",
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
    f"# QK Semantic Codegen v2 Candidates: {candidate_set['model_size']}",
    "",
    "This bounded Family B surface tests row-grouped Q4_K `ffn_down` partial",
    "GEMV. It is microbench-supported first; runtime full-decode installation",
    "is intentionally deferred until a strong raw signal exists.",
    "",
    "## Summary",
    "",
    f"- candidates: `{candidate_set['summary']['candidates']}`",
    f"- single-change candidates: `{candidate_set['summary']['single_change_candidates']}`",
    f"- current storage bytes: `{candidate_set['summary']['current_storage_bytes']}`",
    "",
    "| id | tensor | role | family | row group | parts | opts | persistent delta | metadata sidecar | full decode |",
    "|---|---|---|---|---:|---:|---|---:|---:|---:|",
  ]
  for cand in candidate_set["candidates"]:
    if cand["id"] == "current":
      lines.append("| `current` | n/a | n/a | n/a | 0 | 0 | n/a | 0 | 0 | `True` |")
      continue
    change = cand["changes"][0]
    spec = cand["schedule_spec"]
    storage = spec.get("storage_effect") or {}
    lines.append(f"| `{cand['id']}` | `{change['tensor']}` | `{change.get('role')}` | `{spec['family']}` | "
                 f"{spec['row_group']} | {spec['parts']} | `{','.join(spec['opts'])}` | "
                 f"{storage.get('persistent_bytes_delta', 0)} | {storage.get('metadata_sidecar_bytes', 0)} | "
                 f"`{spec['full_decode_supported']}` |")
  lines.append("")
  return "\n".join(lines)


def static_gate_markdown(report:dict[str, Any]) -> str:
  lines = [
    f"# QK Semantic Codegen v2 Static Gate: {report['model_size']}",
    "",
    "Fail-closed structural validation before microbench. Passing here means",
    "the candidate is a Q4_K ffn_down row-grouped probe, not that it is",
    "eligible for full decode.",
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
  parser = argparse.ArgumentParser(description="Generate semantic QK codegen v2 candidates")
  parser.add_argument("--descriptor", type=pathlib.Path, required=True)
  parser.add_argument("--json", type=pathlib.Path, required=True)
  parser.add_argument("--md", type=pathlib.Path)
  parser.add_argument("--gate-json", type=pathlib.Path)
  parser.add_argument("--gate-md", type=pathlib.Path)
  parser.add_argument("--max-candidates", type=int)
  args = parser.parse_args()
  candidate_set = build_codegen_candidate_set(load_json(args.descriptor.expanduser()), max_candidates=args.max_candidates)
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
