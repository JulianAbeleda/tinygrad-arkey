#!/usr/bin/env python3
from __future__ import annotations

import argparse, copy, json, pathlib, re
from typing import Any

from extra.qk_descriptor_policy import build_policy_from_descriptor, load_json, write_json

RUNTIME_Q4_FAMILY = "q4_k_packed_u32"
DIRECT_Q4_FAMILY = "q4_k_packed_u32_direct"
DOMINANT_Q4_ROLES = ("ffn_gate", "ffn_down", "attn_q", "attn_k")


def _slug(value:str) -> str:
  return re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()


def _current_runtime(row:dict[str, Any]) -> dict[str, Any]:
  lowering = row["current_lowering"]
  return {
    "winner": lowering.get("winner"),
    "family": lowering.get("family"),
    "parts": int(lowering.get("parts") or 0),
    "opts": list(lowering.get("opts") or []),
    "reduction": lowering.get("reduction"),
    "requires": list(lowering.get("requires") or []),
  }


def _runtime_storage_bytes(policy:dict[str, Any]) -> int:
  total = 0
  for entry in policy.get("entries", []):
    if entry.get("winner") != "fused_graph":
      total += int((entry.get("storage") or {}).get("persistent_bytes") or 0)
  return total


def _direct_spec(row:dict[str, Any]) -> dict[str, Any]:
  current = _current_runtime(row)
  return {
    "name": "direct_out_tensor",
    "semantic_object": "packed_quant_gemv_codegen",
    "format": row["format"],
    "role": row.get("role"),
    "family": DIRECT_Q4_FAMILY,
    "parts": 1,
    "opts": list(current["opts"]),
    "codegen_mode": "direct_out",
    "reduction_mode": "direct_out",
    "scope": "tensor",
    "activation_cache": "fp16_direct",
    "requires": ["q4k_gemv_kernel", "u32_packed_storage"],
    "full_decode_supported": True,
  }


def codegen_specs_for_row(row:dict[str, Any]) -> list[dict[str, Any]]:
  current = _current_runtime(row)
  if row.get("format") != "Q4_K": return []
  if row.get("role") not in DOMINANT_Q4_ROLES: return []
  if current["family"] != RUNTIME_Q4_FAMILY: return []
  if current["parts"] < 1: return []
  return [_direct_spec(row)]


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
  entry["policy_reason"] = "semantic codegen tensor override"
  entry["reason"] = "single-tensor semantic codegen candidate"
  storage = entry.setdefault("storage", {})
  storage["decision"] = "tensor_codegen_override"


def _policy_with_tensor_override(descriptor:dict[str, Any], tensor:str, spec:dict[str, Any]) -> dict[str, Any]:
  policy = build_policy_from_descriptor(descriptor)
  policy["kind"] = "qk_generated_policy"
  policy["created_at"] = "generated-by-qk-semantic-codegen"
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
  policy["created_at"] = "generated-by-qk-semantic-codegen"
  return policy


def build_codegen_candidate_set(descriptor:dict[str, Any], *, max_candidates:int | None=None) -> dict[str, Any]:
  if descriptor.get("kind") != "qk_semantic_descriptor_set":
    raise ValueError("expected kind=qk_semantic_descriptor_set")
  baseline = _baseline_policy(descriptor)
  candidates = [{
    "id": "current",
    "status": "baseline",
    "description": "current descriptor policy",
    "changes": [],
    "policy": baseline,
    "expected_storage_bytes": _runtime_storage_bytes(baseline),
  }]
  count = 0
  for row in descriptor.get("descriptors", []):
    for spec in codegen_specs_for_row(row):
      count += 1
      if max_candidates is not None and count > max_candidates: break
      current = _current_runtime(row)
      policy = _policy_with_tensor_override(descriptor, row["tensor"], spec)
      candidates.append({
        "id": f"{count:03d}-{_slug(row.get('role') or row['tensor'])}-{_slug(row['tensor'])}-{_slug(spec['name'])}",
        "status": "candidate",
        "description": "single-tensor direct-output Q4_K codegen candidate",
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
        }],
        "policy": policy,
        "expected_storage_bytes": _runtime_storage_bytes(policy),
      })
    if max_candidates is not None and count >= max_candidates: break
  return {
    "kind": "qk_semantic_codegen_candidate_set",
    "schema_version": 1,
    "model": descriptor.get("model"),
    "model_size": descriptor.get("model_size"),
    "source_descriptor": descriptor.get("source_policy"),
    "gate_models": ["8B", "14B"],
    "search_space": {
      "scope": "single exact-tensor Q4_K direct-output runtime overrides",
      "roles": list(DOMINANT_Q4_ROLES),
      "axes": ["codegen_mode", "reduction_mode", "scope"],
      "note": "32B is excluded unless the 8B/14B semantic codegen gate accepts.",
    },
    "candidates": candidates,
    "summary": {
      "candidates": len(candidates),
      "single_change_candidates": len(candidates) - 1,
      "current_storage_bytes": _runtime_storage_bytes(baseline),
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
    if spec.get("family") != DIRECT_Q4_FAMILY: reasons.append(f"unsupported codegen family {spec.get('family')!r}")
    if spec.get("scope") != "tensor": reasons.append("semantic codegen v1 requires tensor-scoped override")
    if spec.get("codegen_mode") != "direct_out": reasons.append("semantic codegen v1 only supports direct_out")
    if int(spec.get("parts") or 0) != 1: reasons.append("direct_out requires parts=1")
    opts = list(spec.get("opts") or [])
    for opt in opts:
      op = str(opt).split(":", 1)[0]
      if op not in {"LOCAL", "UPCAST", "UNROLL"}: reasons.append(f"unsupported opt op {op!r}")
    rows.append({
      "id": cand["id"],
      "status": "pass" if not reasons else "fail",
      "microbench": not reasons,
      "full_decode_supported": not reasons,
      "schedule": spec,
      "changes": changes,
      "expected_storage_bytes": cand.get("expected_storage_bytes"),
      "reasons": reasons,
    })
  return {
    "kind": "qk_semantic_codegen_static_gate",
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
    f"# QK Semantic Codegen Candidates: {candidate_set['model_size']}",
    "",
    "This v1 surface promotes one concrete codegen capability into runtime:",
    "`q4_k_packed_u32_direct`, a direct-output Q4_K GEMV that avoids the",
    "separate split-K reduction kernel. Candidates are tensor-scoped so full",
    "decode tests do not change every tensor with the same shape.",
    "",
    "## Summary",
    "",
    f"- candidates: `{candidate_set['summary']['candidates']}`",
    f"- single-change candidates: `{candidate_set['summary']['single_change_candidates']}`",
    f"- current storage bytes: `{candidate_set['summary']['current_storage_bytes']}`",
    "",
    "| id | tensor | role | family | scope | parts | opts | full decode |",
    "|---|---|---|---|---|---:|---|---:|",
  ]
  for cand in candidate_set["candidates"]:
    if cand["id"] == "current":
      lines.append("| `current` | n/a | n/a | n/a | n/a | 0 | n/a | `True` |")
      continue
    change = cand["changes"][0]
    spec = cand["schedule_spec"]
    lines.append(f"| `{cand['id']}` | `{change['tensor']}` | `{change.get('role')}` | `{spec['family']}` | "
                 f"`{spec['scope']}` | {spec['parts']} | `{','.join(spec['opts'])}` | `{spec['full_decode_supported']}` |")
  lines.append("")
  return "\n".join(lines)


def static_gate_markdown(report:dict[str, Any]) -> str:
  lines = [
    f"# QK Semantic Codegen Static Gate: {report['model_size']}",
    "",
    "Fail-closed structural validation before microbench. Passing here means",
    "the candidate is an exact-tensor direct-output Q4_K override that the",
    "runtime can install for full decode.",
    "",
    "## Summary",
    "",
    f"- candidates: `{report['summary']['candidates']}`",
    f"- passing microbench: `{report['summary']['passing_microbench']}`",
    f"- full-decode supported: `{report['summary']['full_decode_supported']}`",
    f"- failing: `{report['summary']['failing']}`",
    "",
    "| id | status | microbench | full decode | reasons |",
    "|---|---|---:|---:|---|",
  ]
  for row in report["rows"]:
    reasons = "; ".join(row.get("reasons") or []) or "none"
    lines.append(f"| `{row['id']}` | `{row['status']}` | `{row.get('microbench')}` | "
                 f"`{row.get('full_decode_supported')}` | {reasons} |")
  lines.append("")
  return "\n".join(lines)


def main() -> int:
  parser = argparse.ArgumentParser(description="Generate semantic QK codegen candidates")
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
