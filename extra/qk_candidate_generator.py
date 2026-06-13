#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib, re
from typing import Any

from extra.qk_descriptor_policy import build_policy_from_descriptor, load_json, write_json

SUPPORTED_FAMILIES = {"Q4_K": "q4_k_packed_u32", "Q6_K": "q6_k_packed_u16"}
FAMILY_REQUIRES = {
  "q4_k_packed_u32": ["q4k_gemv_partial_kernel", "u32_packed_storage"],
  "q6_k_packed_u16": ["q6k_gemv_partial_kernel", "u16_packed_storage"],
}
FAMILY_PREFIX = {"q4_k_packed_u32": "q4", "q6_k_packed_u16": "q6"}
PARTS = {"q4_k_packed_u32": (1, 2, 4), "q6_k_packed_u16": (1, 2)}
LOCAL = (32, 64)


def _slug(value:str) -> str:
  return re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()


def _candidate_name(family:str, parts:int, local:int) -> str:
  prefix = FAMILY_PREFIX[family]
  return f"{prefix}_local{local}_p{parts}"


def _current_runtime(row:dict[str, Any]) -> dict[str, Any]:
  lowering = row["current_lowering"]
  return {
    "winner": lowering.get("winner"),
    "family": lowering.get("family"),
    "parts": int(lowering.get("parts") or 0),
    "opts": list(lowering.get("opts") or []),
  }


def _schedule(runtime:dict[str, Any]) -> tuple[str, int, tuple[str, ...]]:
  return str(runtime["family"]), int(runtime["parts"]), tuple(runtime["opts"])


def _variant_runtime(row:dict[str, Any], *, parts:int, local:int) -> dict[str, Any]:
  family = row["current_lowering"]["family"]
  name = _candidate_name(family, parts, local)
  return {"winner": name, "family": family, "parts": parts, "opts": [f"LOCAL:0:{local}"]}


def _variants(row:dict[str, Any]) -> list[dict[str, Any]]:
  family = row["current_lowering"].get("family")
  if family not in PARTS: return []
  out = []
  for parts in PARTS[family]:
    for local in LOCAL:
      candidate = _variant_runtime(row, parts=parts, local=local)
      if _schedule(candidate) == _schedule(_current_runtime(row)): continue
      out.append(candidate)
  return out


def _apply_runtime(entry:dict[str, Any], runtime:dict[str, Any]) -> None:
  cand = entry["candidate"]
  cand["family"] = runtime["family"]
  cand["name"] = runtime["winner"]
  cand["opts"] = runtime["opts"]
  cand["parts"] = runtime["parts"]
  cand["reduction"] = "split_k_partial"
  cand["requires"] = FAMILY_REQUIRES[runtime["family"]]
  entry["winner"] = runtime["winner"]
  entry["policy_reason"] = "generated Ansor-transition candidate"
  entry["reason"] = "single-entry schedule candidate"


def _policy_with_change(descriptor:dict[str, Any], tensor:str | None=None, runtime:dict[str, Any] | None=None) -> dict[str, Any]:
  policy = build_policy_from_descriptor(descriptor)
  policy["kind"] = "qk_generated_policy"
  policy["created_at"] = "generated-by-qk-candidate-generator"
  if tensor is not None:
    assert runtime is not None
    for entry in policy["entries"]:
      if entry.get("tensor") == tensor:
        _apply_runtime(entry, runtime)
        break
    else:
      raise ValueError(f"descriptor has no tensor {tensor}")
  return policy


def _policy_storage_bytes(policy:dict[str, Any]) -> int:
  total = 0
  for entry in policy.get("entries", []):
    if entry.get("winner") != "fused_graph":
      total += int((entry.get("storage") or {}).get("persistent_bytes") or 0)
  return total


def build_candidate_set(descriptor:dict[str, Any], *, max_single_changes:int | None=None) -> dict[str, Any]:
  if descriptor.get("kind") != "qk_semantic_descriptor_set":
    raise ValueError("expected kind=qk_semantic_descriptor_set")
  baseline = _policy_with_change(descriptor)
  candidates: list[dict[str, Any]] = [{
    "id": "current",
    "description": "current descriptor policy",
    "status": "baseline",
    "changes": [],
    "policy": baseline,
    "expected_storage_bytes": _policy_storage_bytes(baseline),
  }]
  change_count = 0
  for row in descriptor.get("descriptors", []):
    tensor = row["tensor"]
    role = row.get("role") or tensor
    for runtime in _variants(row):
      change_count += 1
      if max_single_changes is not None and change_count > max_single_changes: break
      policy = _policy_with_change(descriptor, tensor, runtime)
      current = _current_runtime(row)
      candidates.append({
        "id": f"{change_count:03d}-{_slug(role)}-{_slug(tensor)}-{runtime['family']}-p{runtime['parts']}-local{runtime['opts'][0].split(':')[-1]}",
        "description": "single-entry parts/local candidate",
        "status": "candidate",
        "changes": [{
          "tensor": tensor,
          "format": row.get("format"),
          "role": role,
          "from": current,
          "to": runtime,
        }],
        "policy": policy,
        "expected_storage_bytes": _policy_storage_bytes(policy),
      })
    if max_single_changes is not None and change_count >= max_single_changes: break
  return {
    "kind": "qk_candidate_set",
    "schema_version": 1,
    "model": descriptor.get("model"),
    "model_size": descriptor.get("model_size"),
    "source_descriptor": descriptor.get("source_policy"),
    "descriptor_summary": descriptor.get("summary"),
    "search_space": {
      "families": SUPPORTED_FAMILIES,
      "parts": {k: list(v) for k, v in PARTS.items()},
      "local": list(LOCAL),
      "scope": "current selected Q4_K/Q6_K primitive descriptors; fused_graph entries stay fused in v0",
    },
    "candidates": candidates,
    "summary": {
      "candidates": len(candidates),
      "single_change_candidates": len(candidates) - 1,
      "current_storage_bytes": _policy_storage_bytes(baseline),
    },
  }


def candidates_markdown(candidate_set:dict[str, Any]) -> str:
  lines = [
    f"# QK Candidates: {candidate_set['model_size']}",
    "",
    "Bounded Ansor-transition v0 search space. These are policy candidates,",
    "not new kernels: v0 only varies supported primitive `parts` and `LOCAL`",
    "settings for one current primitive descriptor at a time.",
    "",
    "## Summary",
    "",
    f"- candidates: `{candidate_set['summary']['candidates']}`",
    f"- single-change candidates: `{candidate_set['summary']['single_change_candidates']}`",
    f"- current storage bytes: `{candidate_set['summary']['current_storage_bytes']}`",
    "",
    "| id | changes | storage bytes |",
    "|---|---:|---:|",
  ]
  for cand in candidate_set["candidates"]:
    lines.append(f"| `{cand['id']}` | {len(cand['changes'])} | {cand['expected_storage_bytes']} |")
  lines.append("")
  return "\n".join(lines)


def main() -> int:
  parser = argparse.ArgumentParser(description="Generate bounded QK policy candidates from a semantic descriptor")
  parser.add_argument("--descriptor", type=pathlib.Path, required=True)
  parser.add_argument("--json", type=pathlib.Path, required=True)
  parser.add_argument("--md", type=pathlib.Path)
  parser.add_argument("--max-single-changes", type=int)
  args = parser.parse_args()
  candidate_set = build_candidate_set(load_json(args.descriptor.expanduser()), max_single_changes=args.max_single_changes)
  write_json(args.json, candidate_set)
  if args.md:
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text(candidates_markdown(candidate_set))
  else:
    print(candidates_markdown(candidate_set))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
