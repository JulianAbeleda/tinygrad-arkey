#!/usr/bin/env python3
from __future__ import annotations

import argparse, filecmp, json, pathlib
from typing import Any


from extra.llm_eval_common import load_json, write_json


def _confirmations(root:pathlib.Path, model:str, accepted_rows:list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
  confirm_root = root / f"{model}-confirm"
  if not confirm_root.exists(): return {}
  out = {}
  for decision_path in sorted(confirm_root.glob("*/decision.json")):
    decision = load_json(decision_path)
    policy = pathlib.Path(decision["policy"])
    if not policy.is_absolute() and not policy.exists(): policy = decision_path.parent / policy.name
    for row in accepted_rows:
      cand_policy = pathlib.Path(row["policy"])
      if not cand_policy.is_absolute() and not cand_policy.exists(): cand_policy = root / model / row["id"] / "policy.json"
      if policy.exists() and cand_policy.exists() and filecmp.cmp(policy, cand_policy, shallow=False):
        out[row["id"]] = {
          "path": str(decision_path.parent),
          "status": decision.get("status"),
          "gain": decision.get("gain"),
          "reference_tok_s": (decision.get("explicit") or {}).get("avg_tok_s"),
          "candidate_tok_s": (decision.get("generated") or {}).get("avg_tok_s"),
          "ab_match": decision.get("ab_match"),
          "reasons": decision.get("reasons", []),
        }
  return out


def build_verdict(root:pathlib.Path) -> dict[str, Any]:
  rows = []
  order = {"8b": 0, "14b": 1, "32b": 2}
  for matrix_path in sorted(root.glob("*/matrix.json"), key=lambda p: (order.get(p.parent.name, 99), p.parent.name)):
    model = matrix_path.parent.name
    matrix = load_json(matrix_path)
    accepted = [row for row in matrix.get("rows", []) if row.get("status") == "accept"]
    confirmations = _confirmations(root, model, accepted)
    confirmed = [row for row in accepted if (confirmations.get(row["id"]) or {}).get("status") == "accept"]
    if confirmed:
      decision = "promote_confirmed_candidate"
    elif accepted:
      decision = "raw_accept_unconfirmed_or_rejected_by_confirmation"
    else:
      decision = "descriptor_knob_frontier_exhausted"
    rows.append({
      "model": model.upper(),
      "matrix": str(matrix_path),
      "summary": matrix.get("summary", {}),
      "accepted_ids": [row["id"] for row in accepted],
      "confirmations": confirmations,
      "confirmed_accept_ids": [row["id"] for row in confirmed],
      "decision": decision,
    })
  promote = [row for row in rows if row["decision"] == "promote_confirmed_candidate"]
  return {
    "kind": "qk_loop_benchmark_verdict",
    "scope": "policy-vs-policy loop v0 benchmark candidates",
    "rows": rows,
    "summary": {
      "models": len(rows),
      "models_with_raw_accept": sum(1 for row in rows if row["accepted_ids"]),
      "models_with_confirmed_accept": len(promote),
      "overall_decision": "promote_confirmed_candidate" if promote else "descriptor_knob_frontier_exhausted",
    },
  }


def verdict_markdown(verdict:dict[str, Any]) -> str:
  lines = [
    "# QK Loop Benchmark Verdict",
    "",
    "Candidates are compared against the current accepted generated policy, not",
    "against explicit primitive flags. Raw accepts that fail confirmation are not",
    "promoted.",
    "",
    "## Summary",
    "",
    f"- models: `{verdict['summary']['models']}`",
    f"- models with raw accept: `{verdict['summary']['models_with_raw_accept']}`",
    f"- models with confirmed accept: `{verdict['summary']['models_with_confirmed_accept']}`",
    f"- overall decision: `{verdict['summary']['overall_decision']}`",
    "",
    "| model | matrix accepted | confirmed accepted | decision | confirmation |",
    "|---|---|---|---|---|",
  ]
  for row in verdict["rows"]:
    confirmations = []
    for cand_id, confirm in row["confirmations"].items():
      gain = confirm.get("gain")
      gain_s = "n/a" if gain is None else f"{gain*100:.2f}%"
      confirmations.append(f"{cand_id}: {confirm.get('status')} ({gain_s})")
    lines.append(
      f"| `{row['model']}` | `{', '.join(row['accepted_ids']) or 'none'}` | "
      f"`{', '.join(row['confirmed_accept_ids']) or 'none'}` | `{row['decision']}` | "
      f"{'; '.join(confirmations) or 'none'} |"
    )
  lines.append("")
  return "\n".join(lines)


def main() -> int:
  parser = argparse.ArgumentParser(description="Summarize QK loop benchmark matrices and confirmation reruns")
  parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path("bench/qk-ansor-transition-20260612/benchmarks"))
  parser.add_argument("--json", type=pathlib.Path, required=True)
  parser.add_argument("--md", type=pathlib.Path, required=True)
  args = parser.parse_args()
  verdict = build_verdict(args.root)
  write_json(args.json, verdict)
  args.md.parent.mkdir(parents=True, exist_ok=True)
  args.md.write_text(verdict_markdown(verdict))
  print(verdict_markdown(verdict))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
