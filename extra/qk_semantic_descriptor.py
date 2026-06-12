#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib, re
from collections import Counter
from typing import Any

MODEL_RE = re.compile(r"Qwen3-(?P<size>[0-9.]+B)-")

def _load_json(path:pathlib.Path) -> Any:
  try:
    return json.loads(path.read_text())
  except json.JSONDecodeError as exc:
    raise ValueError(f"{path}: invalid JSON: {exc}") from exc

def _model_size(model:str, fallback:str) -> str:
  match = MODEL_RE.search(model)
  return match.group("size").upper() if match else fallback.upper()

def _local_opts(opts:list[str]) -> list[dict[str, Any]]:
  out = []
  for opt in opts:
    parts = opt.split(":")
    if len(parts) == 3 and parts[0] == "LOCAL":
      out.append({"op": "LOCAL", "axis": int(parts[1]), "arg": int(parts[2])})
    else:
      out.append({"raw": opt})
  return out

def _role(tensor:str, descriptor:dict[str, Any]) -> str:
  if descriptor.get("role"): return descriptor["role"]
  name = tensor.removesuffix(".weight")
  return name.rsplit(".", 1)[-1]

def entry_descriptor(entry:dict[str, Any], *, storage_mode:str) -> dict[str, Any]:
  desc = entry.get("descriptor") or {}
  cand = entry.get("candidate") or {}
  result = entry.get("result") or {}
  storage = entry.get("storage") or {}
  tensor = entry.get("tensor") or desc.get("tensor")
  if not tensor: raise ValueError("policy entry missing tensor")
  rows = desc.get("rows") or (entry.get("shape") or [None, None])[0]
  cols = desc.get("cols") or (entry.get("shape") or [None, None])[1]
  return {
    "tensor": tensor,
    "role": _role(tensor, desc),
    "format": entry.get("format") or desc.get("format"),
    "ggml_type": desc.get("ggml_type"),
    "shape": {"rows": rows, "cols": cols},
    "layout": {
      "block_bytes": desc.get("block_bytes"),
      "block_elems": desc.get("block_elems"),
      "packed_bytes": desc.get("packed_bytes"),
      "tensor_offset": desc.get("tensor_offset"),
      "byte_start": desc.get("byte_start"),
      "data_start": desc.get("data_start"),
    },
    "activation_dtype": desc.get("dtype_activation") or cand.get("activation"),
    "output_dtype": desc.get("dtype_output"),
    "storage": {
      "mode": storage_mode,
      "policy_persistent_bytes": storage.get("persistent_bytes"),
      "benefit_ms": storage.get("benefit_ms"),
      "benefit_ms_per_mb": storage.get("benefit_ms_per_mb"),
      "scope": entry.get("scope"),
    },
    "current_lowering": {
      "winner": entry.get("winner"),
      "candidate_name": cand.get("name"),
      "family": cand.get("family"),
      "parts": cand.get("parts"),
      "reduction": cand.get("reduction"),
      "opts": cand.get("opts", []),
      "local_opts": _local_opts(cand.get("opts", [])),
      "requires": cand.get("requires", []),
      "metric": entry.get("metric"),
      "metric_value": entry.get("metric_value"),
      "policy_reason": entry.get("policy_reason"),
      "result_status": result.get("status"),
      "quant_gbs": result.get("quant_gbs"),
      "device_ms": result.get("device_ms"),
      "wall_ms": result.get("wall_ms"),
      "gemv_max_abs": result.get("gemv_max_abs"),
      "unpack_max_abs": result.get("unpack_max_abs"),
    },
    "ansor_transition": {
      "semantic_object": "packed_quant_gemv",
      "search_axes": ["format", "tensor_role", "shape", "parts", "local_opts", "reduction", "storage_mode", "fallback_policy"],
      "current_state": "hand_seeded_primitive_selected_by_generated_policy",
    },
  }

def build_descriptor(policy_path:pathlib.Path, *, model_label:str | None=None) -> dict[str, Any]:
  policy = _load_json(policy_path)
  if policy.get("kind") != "qk_generated_policy":
    raise ValueError(f"{policy_path}: expected kind=qk_generated_policy")
  entries = policy.get("entries")
  if not isinstance(entries, list) or not entries: raise ValueError(f"{policy_path}: expected non-empty entries")
  model = policy.get("model", "")
  model_size = model_label.upper() if model_label else _model_size(model, policy_path.parent.name)
  storage_policy = policy.get("storage_policy") or {}
  storage_mode = "shared_runtime_source" if policy_path.parts and "shared-storage" in str(policy_path) else "policy_defined"
  descriptors = [entry_descriptor(entry, storage_mode=storage_mode) for entry in entries]
  by_format = Counter(row["format"] for row in descriptors)
  by_role = Counter(row["role"] for row in descriptors)
  by_family = Counter(row["current_lowering"]["family"] for row in descriptors)
  by_parts = Counter(str(row["current_lowering"]["parts"]) for row in descriptors)
  return {
    "kind": "qk_semantic_descriptor_set",
    "schema_version": 1,
    "source_policy": str(policy_path),
    "policy_commit": policy.get("commit"),
    "generator_version": policy.get("generator_version"),
    "model": model,
    "model_size": model_size,
    "storage_policy": storage_policy,
    "descriptors": descriptors,
    "summary": {
      "entries": len(descriptors),
      "by_format": dict(sorted(by_format.items())),
      "by_role": dict(sorted(by_role.items())),
      "by_family": dict(sorted(by_family.items())),
      "by_parts": dict(sorted(by_parts.items())),
      "selected_primitive_entries": storage_policy.get("selected_primitive_entries"),
      "selected_bytes": storage_policy.get("selected_bytes"),
    },
  }

def descriptor_markdown(descriptor:dict[str, Any]) -> str:
  lines = [
    f"# QK Semantic Descriptor: {descriptor['model_size']}",
    "",
    "Machine-readable bridge from current hand-seeded Q4/Q6 primitive policies",
    "toward an Ansor-style generated search space. This describes selected",
    "packed-quant GEMV shapes and current lowerings as data.",
    "",
    "## Summary",
    "",
    f"- source policy: `{descriptor['source_policy']}`",
    f"- entries: `{descriptor['summary']['entries']}`",
    f"- formats: `{descriptor['summary']['by_format']}`",
    f"- roles: `{descriptor['summary']['by_role']}`",
    f"- families: `{descriptor['summary']['by_family']}`",
    f"- parts: `{descriptor['summary']['by_parts']}`",
    "",
    "| tensor | format | role | shape | family | parts | opts | metric |",
    "|---|---|---|---:|---|---:|---|---:|",
  ]
  for row in descriptor["descriptors"]:
    lowering = row["current_lowering"]
    shape = row["shape"]
    lines.append(
      f"| `{row['tensor']}` | `{row['format']}` | `{row['role']}` | "
      f"`{shape['rows']}x{shape['cols']}` | `{lowering['family']}` | "
      f"{lowering['parts']} | `{','.join(lowering['opts'])}` | {lowering['metric_value']:.2f} |"
    )
  lines.append("")
  return "\n".join(lines)

def write_descriptor(descriptor:dict[str, Any], json_path:pathlib.Path, md_path:pathlib.Path | None=None) -> None:
  json_path.parent.mkdir(parents=True, exist_ok=True)
  json_path.write_text(json.dumps(descriptor, indent=2, sort_keys=True))
  if md_path is not None:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(descriptor_markdown(descriptor))

def main() -> int:
  parser = argparse.ArgumentParser(description="Build semantic QK descriptor set from a generated policy")
  parser.add_argument("--policy", type=pathlib.Path, required=True)
  parser.add_argument("--model-label")
  parser.add_argument("--json", type=pathlib.Path, required=True)
  parser.add_argument("--md", type=pathlib.Path)
  args = parser.parse_args()
  descriptor = build_descriptor(args.policy.expanduser(), model_label=args.model_label)
  write_descriptor(descriptor, args.json, args.md)
  if args.md is None: print(json.dumps(descriptor, indent=2, sort_keys=True))
  else: print(descriptor_markdown(descriptor))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
