#!/usr/bin/env python3
from __future__ import annotations

import argparse, collections, json, pathlib
from dataclasses import asdict, dataclass

from extra.qk_layout import GGML_Q4_K, GGML_Q6_K, GGUFMetadata, format_name, read_metadata, tensor_shape
from tinygrad.llm.model import _load_qk_generated_policy, _q4k_policy, _q6k_policy

SUPPORTED_FAMILIES = {
  GGML_Q4_K: "q4_k_packed_u32",
  GGML_Q6_K: "q6_k_packed_u16",
}

@dataclass(frozen=True)
class PolicyDecision:
  source: str
  raw_winner: str
  effective_winner: str
  parts: int
  opts: tuple[str, ...]
  reason: str
  unsupported: bool = False

  def json(self) -> dict:
    out = asdict(self)
    out["opts"] = list(self.opts)
    return out

@dataclass(frozen=True)
class TensorParity:
  name: str
  format: str
  ggml_type: int
  shape: tuple[int, int]
  explicit: PolicyDecision
  generated: PolicyDecision
  same_effective: bool
  same_raw: bool

  def json(self) -> dict:
    return {
      "name": self.name, "format": self.format, "ggml_type": self.ggml_type, "shape": list(self.shape),
      "explicit": self.explicit.json(), "generated": self.generated.json(),
      "same_effective": self.same_effective, "same_raw": self.same_raw,
    }

def _explicit_decision(name:str, typ:int) -> PolicyDecision:
  if typ == GGML_Q4_K:
    if (policy := _q4k_policy(name)) is None:
      return PolicyDecision("explicit", "fused_graph", "fused_graph", 0, (), "policy_fallback")
    return PolicyDecision("explicit", "v1_q4_packed", "q4_k_packed_u32", policy[0], tuple(policy[1]), "policy_primitive")
  if typ == GGML_Q6_K:
    if (policy := _q6k_policy(name)) is None:
      return PolicyDecision("explicit", "fused_graph", "fused_graph", 0, (), "policy_fallback")
    return PolicyDecision("explicit", "v1_q6_packed", "q6_k_packed_u16", policy[0], tuple(policy[1]), "policy_primitive")
  raise ValueError(f"unsupported ggml_type={typ}")

def _lookup_generated(policy:dict, name:str, typ:int, rows:int, cols:int) -> dict|None:
  if "by_shape" in policy or "by_tensor" in policy:
    if (entry:=policy.get("by_tensor", {}).get((name, typ, rows, cols))) is not None: return entry
    return policy.get("by_shape", {}).get((typ, rows, cols))
  return policy.get((typ, rows, cols))

def _generated_decision(policy:dict, name:str, typ:int, rows:int, cols:int) -> PolicyDecision:
  entry = _lookup_generated(policy, name, typ, rows, cols)
  if entry is None:
    return PolicyDecision("generated", "fused_graph", "fused_graph", 0, (), "policy_missing")
  winner = str(entry["winner"])
  policy_reason = str(entry.get("policy_reason", ""))
  if winner == "fused_graph":
    reason = "policy_memory_cap" if "memory_cap" in policy_reason else "policy_fused"
    return PolicyDecision("generated", "fused_graph", "fused_graph", 0, (), reason)
  family = str(entry.get("family", ""))
  if family != SUPPORTED_FAMILIES[typ]:
    # Unsupported generated winners are skipped, so the model falls back.
    return PolicyDecision("generated", winner, "fused_graph", int(entry.get("parts", 0)),
                          tuple(entry.get("opts", ())), "policy_unsupported", True)
  return PolicyDecision("generated", winner, family, int(entry.get("parts", 0)), tuple(entry.get("opts", ())), "policy_primitive")

def compare_policies(meta:GGUFMetadata, generated_policy:dict) -> list[TensorParity]:
  rows: list[TensorParity] = []
  for info in meta.infos:
    if info.typ not in SUPPORTED_FAMILIES or len(info.dims) != 2 or not info.name.endswith(".weight"): continue
    shape = tensor_shape(info)
    if len(shape) != 2: continue
    explicit = _explicit_decision(info.name, info.typ)
    generated = _generated_decision(generated_policy, info.name, info.typ, int(shape[0]), int(shape[1]))
    same_effective = (
      explicit.effective_winner == generated.effective_winner and explicit.parts == generated.parts and explicit.opts == generated.opts
    )
    same_raw = (
      explicit.raw_winner == generated.raw_winner and explicit.parts == generated.parts and explicit.opts == generated.opts and
      explicit.reason == generated.reason
    )
    rows.append(TensorParity(info.name, format_name(info.typ), info.typ, (int(shape[0]), int(shape[1])),
                             explicit, generated, same_effective, same_raw))
  return rows

def summarize(rows:list[TensorParity]) -> dict:
  by_format: collections.Counter[str] = collections.Counter(r.format for r in rows)
  explicit_reasons: collections.Counter[str] = collections.Counter(r.explicit.reason for r in rows)
  generated_reasons: collections.Counter[str] = collections.Counter(r.generated.reason for r in rows)
  mismatches = [r for r in rows if not r.same_effective]
  raw_diffs = [r for r in rows if not r.same_raw]
  unsupported = [r for r in rows if r.generated.unsupported]
  return {
    "total": len(rows), "same_effective": len(rows) - len(mismatches), "effective_mismatches": len(mismatches),
    "same_raw": len(rows) - len(raw_diffs), "raw_differences": len(raw_diffs), "generated_unsupported": len(unsupported),
    "by_format": dict(sorted(by_format.items())), "explicit_reasons": dict(sorted(explicit_reasons.items())),
    "generated_reasons": dict(sorted(generated_reasons.items())),
    "explicit_installed": sum(1 for r in rows if r.explicit.effective_winner != "fused_graph"),
    "generated_installed": sum(1 for r in rows if r.generated.effective_winner != "fused_graph"),
  }

def make_report(model:pathlib.Path, policy_path:pathlib.Path, rows:list[TensorParity]) -> dict:
  return {
    "kind": "qk_policy_parity", "model": str(model.expanduser()), "policy": str(policy_path.expanduser()),
    "summary": summarize(rows), "rows": [r.json() for r in rows],
  }

def write_markdown(report:dict, path:pathlib.Path) -> None:
  summary = report["summary"]
  lines = [
    "# QK Generated Policy Parity",
    "",
    f"- Model: `{report['model']}`",
    f"- Policy: `{report['policy']}`",
    f"- Total tensors: `{summary['total']}`",
    f"- Effective mismatches: `{summary['effective_mismatches']}`",
    f"- Raw differences: `{summary['raw_differences']}`",
    f"- Generated unsupported winners: `{summary['generated_unsupported']}`",
    f"- Explicit installed: `{summary['explicit_installed']}`",
    f"- Generated installed: `{summary['generated_installed']}`",
    "",
    "## Summary",
    "",
    "```json",
    json.dumps(summary, indent=2, sort_keys=True),
    "```",
    "",
    "## Differences",
    "",
  ]
  diffs = [r for r in report["rows"] if not r["same_effective"] or not r["same_raw"] or r["generated"]["unsupported"]]
  if not diffs:
    lines.append("No parity differences.")
  else:
    lines += ["| tensor | format | shape | explicit | generated | effective match | raw match |", "|---|---|---:|---|---|---:|---:|"]
    for r in diffs:
      explicit = r["explicit"]
      generated = r["generated"]
      exp_s = f"{explicit['raw_winner']} parts={explicit['parts']} opts={explicit['opts']} reason={explicit['reason']}"
      gen_s = f"{generated['raw_winner']} parts={generated['parts']} opts={generated['opts']} reason={generated['reason']}"
      lines.append(f"| `{r['name']}` | {r['format']} | {'x'.join(map(str, r['shape']))} | `{exp_s}` | `{gen_s}` | "
                   f"{r['same_effective']} | {r['same_raw']} |")
  path.write_text("\n".join(lines) + "\n")

def main() -> None:
  parser = argparse.ArgumentParser(description="Compare explicit Q4/Q6 primitive policy against a generated QK policy cache")
  parser.add_argument("--model", type=pathlib.Path, required=True)
  parser.add_argument("--policy", type=pathlib.Path, required=True)
  parser.add_argument("--json", type=pathlib.Path)
  parser.add_argument("--md", type=pathlib.Path)
  parser.add_argument("--strict", action="store_true", help="exit nonzero on effective mismatch or unsupported generated winner")
  args = parser.parse_args()

  model = args.model.expanduser()
  policy_path = args.policy.expanduser()
  rows = compare_policies(read_metadata(model), _load_qk_generated_policy(str(policy_path)))
  report = make_report(model, policy_path, rows)
  if args.json: args.json.write_text(json.dumps(report, indent=2, sort_keys=True))
  if args.md: write_markdown(report, args.md)
  print(json.dumps(report["summary"], indent=2, sort_keys=True))
  if args.strict and (report["summary"]["effective_mismatches"] or report["summary"]["generated_unsupported"]):
    raise SystemExit(1)

if __name__ == "__main__":
  main()
