#!/usr/bin/env python3
from __future__ import annotations

import argparse, copy, json, pathlib
from typing import Any

from extra.qk_descriptor_policy import build_policy_from_descriptor, load_json, write_json
from extra.qk_semantic_candidate import (
  correctness_provenance, current_runtime, no_extra_storage_effect, runtime_storage_bytes, slug,
)

RUNTIME_Q4_FAMILY = "q4_k_packed_u32"
PACKED_LOAD_Q4_FAMILY = "q4_k_packed_u32_packed_load"
TARGET_ROLES = ("ffn_gate",)


def _packed_load_spec(row:dict[str, Any]) -> dict[str, Any]:
  current = current_runtime(row)
  storage_effect = no_extra_storage_effect("packed-load codegen reuses existing Q4_K uint32 shared storage")
  return {
    "name": "packed_load_u32x4",
    "semantic_object": "packed_quant_gemv_codegen",
    "format": row["format"],
    "role": row.get("role"),
    "family": PACKED_LOAD_Q4_FAMILY,
    "parts": int(current["parts"]),
    "opts": list(current["opts"]),
    "codegen_mode": "packed_load",
    "reduction_mode": "split_k_partial",
    "scope": "tensor",
    "load_mode": "u32_load_once_per_4_nibbles",
    "lane_mapping": "row_part_kblock_lane4",
    "packed_words_per_reduce_step": 1,
    "nibbles_per_packed_load": 4,
    "expected_memory_mechanism": (
      "replace per-position qword indexing with a lane4 reduce axis that loads each Q4_K quant uint32 word once "
      "and unrolls four nibbles from that word"
    ),
    "requires": ["q4k_gemv_packed_load_partial_kernel", "u32_packed_storage"],
    "full_decode_supported": False,
    "storage_effect": storage_effect,
    "correctness_provenance": correctness_provenance(full_decode_supported=False),
  }


def codegen_specs_for_row(row:dict[str, Any]) -> list[dict[str, Any]]:
  current = current_runtime(row)
  if row.get("format") != "Q4_K": return []
  if row.get("role") not in TARGET_ROLES: return []
  if current["family"] != RUNTIME_Q4_FAMILY: return []
  if int(current["parts"]) != 1: return []
  rows = int((row.get("shape") or {}).get("rows") or 0)
  cols = int((row.get("shape") or {}).get("cols") or 0)
  if rows <= 0 or cols <= 0: return []
  return [_packed_load_spec(row)]


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
  entry["policy_reason"] = "semantic codegen v3 tensor override"
  entry["reason"] = "single-tensor packed-load semantic codegen candidate"
  storage = entry.setdefault("storage", {})
  storage["decision"] = "tensor_codegen_v3_override"


def _policy_with_tensor_override(descriptor:dict[str, Any], tensor:str, spec:dict[str, Any]) -> dict[str, Any]:
  policy = build_policy_from_descriptor(descriptor)
  policy["kind"] = "qk_generated_policy"
  policy["created_at"] = "generated-by-qk-semantic-codegen-v3"
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
  policy["created_at"] = "generated-by-qk-semantic-codegen-v3"
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
        "description": "single-tensor Q4_K packed-load codegen candidate",
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
    "schema_version": 3,
    "model": descriptor.get("model"),
    "model_size": descriptor.get("model_size"),
    "source_descriptor": descriptor.get("source_policy"),
    "gate_models": ["8B", "14B"],
    "search_space": {
      "scope": "single exact-tensor Q4_K ffn_gate packed-load runtime probes",
      "roles": list(TARGET_ROLES),
      "axes": ["load_mode", "lane_mapping", "codegen_mode", "scope"],
      "memory_traffic_mechanism": (
        "load each packed Q4_K quant uint32 once per four nibbles and make the packed-word lane explicit"
      ),
      "note": "32B is excluded unless the 8B/14B semantic codegen v3 gate accepts.",
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
    if spec.get("role") not in TARGET_ROLES: reasons.append(f"semantic codegen v3 targets only {TARGET_ROLES}")
    if spec.get("family") != PACKED_LOAD_Q4_FAMILY: reasons.append(f"unsupported codegen family {spec.get('family')!r}")
    if spec.get("scope") != "tensor": reasons.append("semantic codegen v3 requires tensor-scoped override")
    if spec.get("codegen_mode") != "packed_load": reasons.append("semantic codegen v3 only supports packed_load")
    if spec.get("load_mode") != "u32_load_once_per_4_nibbles": reasons.append(f"unsupported load_mode={spec.get('load_mode')!r}")
    if int(spec.get("parts") or 0) != 1: reasons.append("packed-load v0 only targets parts=1")
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
    "kind": "qk_semantic_codegen_v3_static_gate",
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
    f"# QK Semantic Codegen v3 Candidates: {candidate_set['model_size']}",
    "",
    "Family C v0 tests a packed-load Q4_K `ffn_gate` partial GEMV. It is a",
    "memory-access probe: it changes the reduce axis from per-position qword",
    "indexing to explicit packed-word lanes that unroll four nibbles from each",
    "loaded uint32.",
    "",
    "## Summary",
    "",
    f"- candidates: `{candidate_set['summary']['candidates']}`",
    f"- single-change candidates: `{candidate_set['summary']['single_change_candidates']}`",
    f"- current storage bytes: `{candidate_set['summary']['current_storage_bytes']}`",
    "",
    "| id | tensor | role | family | load mode | parts | opts | persistent delta | full decode |",
    "|---|---|---|---|---|---:|---|---:|---:|",
  ]
  for cand in candidate_set["candidates"]:
    if cand["id"] == "current":
      lines.append("| `current` | n/a | n/a | n/a | n/a | 0 | n/a | 0 | `True` |")
      continue
    change = cand["changes"][0]
    spec = cand["schedule_spec"]
    storage = spec.get("storage_effect") or {}
    lines.append(f"| `{cand['id']}` | `{change['tensor']}` | `{change.get('role')}` | `{spec['family']}` | "
                 f"`{spec['load_mode']}` | {spec['parts']} | `{','.join(spec['opts'])}` | "
                 f"{storage.get('persistent_bytes_delta', 0)} | `{spec['full_decode_supported']}` |")
  lines.append("")
  return "\n".join(lines)


def static_gate_markdown(report:dict[str, Any]) -> str:
  lines = [
    f"# QK Semantic Codegen v3 Static Gate: {report['model_size']}",
    "",
    "Fail-closed structural validation before microbench. Passing here means",
    "the candidate is a Q4_K ffn_gate packed-load probe, not that it is",
    "eligible for full decode.",
    "",
    "## Summary",
    "",
    f"- candidates: `{report['summary']['candidates']}`",
    f"- passing microbench: `{report['summary']['passing_microbench']}`",
    f"- full-decode supported: `{report['summary']['full_decode_supported']}`",
    f"- failing: `{report['summary']['failing']}`",
    "",
    "| id | status | microbench | full decode | persistent delta | reasons |",
    "|---|---|---:|---:|---:|---|",
  ]
  for row in report["rows"]:
    reasons = "; ".join(row.get("reasons") or []) or "none"
    storage = row.get("storage_effect") or {}
    lines.append(f"| `{row['id']}` | `{row['status']}` | `{row.get('microbench')}` | "
                 f"`{row.get('full_decode_supported')}` | {storage.get('persistent_bytes_delta', 0)} | {reasons} |")
  lines.append("")
  return "\n".join(lines)


def main() -> int:
  parser = argparse.ArgumentParser(description="Generate semantic QK codegen v3 packed-load candidates")
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
