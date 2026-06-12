#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib
from typing import Any

from extra.qk_experiment_matrix import LLAMA_REFS, make_matrix

DEFAULT_TARGETS = (70.0, 80.0, 100.0)

def _load_json(path:pathlib.Path) -> Any:
  try:
    return json.loads(path.read_text())
  except json.JSONDecodeError as exc:
    raise ValueError(f"{path}: invalid JSON: {exc}") from exc

def _fmt(x:Any) -> str:
  if x is None: return "n/a"
  if isinstance(x, float): return f"{x:.2f}"
  return str(x)

def _rollout_summary(path:pathlib.Path | None) -> dict[str, Any] | None:
  if path is None: return None
  data = _load_json(path)
  if data.get("kind") != "llm_rollout_compare_report":
    raise ValueError(f"{path}: expected kind=llm_rollout_compare_report")
  comparisons = data.get("comparisons") or []
  if not comparisons: raise ValueError(f"{path}: expected at least one comparison")
  regressions = sum(len(c.get("quality", {}).get("regressions", [])) for c in comparisons)
  text_changes = sum(c.get("outputs", {}).get("text_changed", 0) for c in comparisons)
  token_changes = sum(c.get("outputs", {}).get("tokens_changed", 0) for c in comparisons)
  return {
    "path": str(path),
    "baseline": data.get("baseline"),
    "comparisons": len(comparisons),
    "regressions": regressions,
    "text_changes": text_changes,
    "token_changes": token_changes,
    "status": "pass" if regressions == 0 else "fail",
  }

def _row_score(row:dict[str, Any], targets:tuple[float, ...]) -> dict[str, Any]:
  pct = row.get("generated_pct_llama")
  generated = row.get("generated_tok_s")
  model = row.get("model_size")
  llama = LLAMA_REFS.get(model)
  target_gaps = {}
  target_tok_s = {}
  for target in targets:
    target_gaps[f"gap_to_{int(target)}pct"] = None if pct is None else max(0.0, target - pct)
    target_tok_s[f"target_{int(target)}pct_tok_s"] = None if llama is None else llama * target / 100.0
  return {
    "path": row["path"],
    "model_size": model,
    "status": row.get("status"),
    "reference_mode": row.get("reference_mode"),
    "explicit_tok_s": row.get("explicit_tok_s"),
    "generated_tok_s": generated,
    "llama_ref_tok_s": llama,
    "generated_pct_llama": pct,
    "speedup_to_llama_parity": None if generated in (None, 0) or llama is None else llama / generated,
    "gain": row.get("gain"),
    "ab_match": row.get("ab_match"),
    "runtime_storage_bytes": row.get("runtime_storage_bytes"),
    "policy_selected_bytes": row.get("policy_selected_bytes"),
    "policy_selected_primitive_entries": row.get("policy_selected_primitive_entries"),
    **target_gaps,
    **target_tok_s,
  }

def build_scorecard(experiments:list[pathlib.Path], *, rollout_compare:pathlib.Path | None=None,
                    targets:tuple[float, ...]=DEFAULT_TARGETS) -> dict[str, Any]:
  matrix = make_matrix(experiments)
  rows = [_row_score(row, targets) for row in matrix["rows"]]
  accepted = [row for row in rows if row["status"] == "accept"]
  pct_rows = [row["generated_pct_llama"] for row in accepted if isinstance(row.get("generated_pct_llama"), (int, float))]
  correctness_ok = all(row.get("ab_match") is True for row in accepted)
  target70 = all((row.get("generated_pct_llama") or 0.0) >= 70.0 for row in accepted) if accepted else False
  return {
    "kind": "qk_llama_scorecard",
    "goal": "llama.cpp-comparable QK decode",
    "targets_pct_llama": list(targets),
    "source_experiments": [str(p) for p in experiments],
    "rollout_compare": _rollout_summary(rollout_compare),
    "rows": rows,
    "summary": {
      "models": len(rows),
      "accepted": len(accepted),
      "correctness_ok": correctness_ok,
      "rollout_compare_ok": None if rollout_compare is None else _rollout_summary(rollout_compare)["status"] == "pass",
      "min_pct_llama": min(pct_rows) if pct_rows else None,
      "mean_pct_llama": sum(pct_rows) / len(pct_rows) if pct_rows else None,
      "all_models_at_70pct": target70,
      "models_below_70pct": [row["model_size"] for row in accepted if (row.get("generated_pct_llama") or 0.0) < 70.0],
    },
  }

def scorecard_markdown(scorecard:dict[str, Any]) -> str:
  lines = [
    "# QK Llama.cpp Scorecard",
    "",
    "Objective function for the Ansor-transition loop. This is a read-only",
    "summary over committed QK decisions and optional rollout-comparator",
    "artifacts; it does not run benchmarks.",
    "",
    "## Summary",
    "",
  ]
  summary = scorecard["summary"]
  lines += [
    f"- accepted models: `{summary['accepted']}/{summary['models']}`",
    f"- correctness ok: `{summary['correctness_ok']}`",
    f"- rollout compare ok: `{summary['rollout_compare_ok']}`",
    f"- min % llama.cpp: `{_fmt(summary['min_pct_llama'])}`",
    f"- mean % llama.cpp: `{_fmt(summary['mean_pct_llama'])}`",
    f"- all models at 70%: `{summary['all_models_at_70pct']}`",
    f"- below 70%: `{', '.join(summary['models_below_70pct']) or 'none'}`",
    "",
    "## Model Rows",
    "",
    "| model | generated tok/s | llama ref | % llama | gap to 70 | parity speedup needed | A/B | runtime MB |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
  ]
  for row in scorecard["rows"]:
    runtime_mb = None if row["runtime_storage_bytes"] is None else row["runtime_storage_bytes"] / (1024 * 1024)
    lines.append(
      f"| `{row['model_size']}` | {_fmt(row['generated_tok_s'])} | {_fmt(row['llama_ref_tok_s'])} | "
      f"{_fmt(row['generated_pct_llama'])} | {_fmt(row.get('gap_to_70pct'))} | "
      f"{_fmt(row['speedup_to_llama_parity'])}x | `{row['ab_match']}` | {_fmt(runtime_mb)} |"
    )
  if scorecard.get("rollout_compare"):
    rc = scorecard["rollout_compare"]
    lines += [
      "",
      "## Rollout Comparator",
      "",
      f"- path: `{rc['path']}`",
      f"- baseline: `{rc['baseline']}`",
      f"- regressions: `{rc['regressions']}`",
      f"- text changes: `{rc['text_changes']}`",
      f"- token changes: `{rc['token_changes']}`",
    ]
  lines.append("")
  return "\n".join(lines)

def write_scorecard(scorecard:dict[str, Any], json_path:pathlib.Path, md_path:pathlib.Path) -> None:
  json_path.parent.mkdir(parents=True, exist_ok=True)
  md_path.parent.mkdir(parents=True, exist_ok=True)
  json_path.write_text(json.dumps(scorecard, indent=2, sort_keys=True))
  md_path.write_text(scorecard_markdown(scorecard))

def main() -> int:
  parser = argparse.ArgumentParser(description="Build llama.cpp-comparable QK scorecard from committed decisions")
  parser.add_argument("experiments", nargs="+", type=pathlib.Path)
  parser.add_argument("--rollout-compare", type=pathlib.Path)
  parser.add_argument("--json", type=pathlib.Path, required=True)
  parser.add_argument("--md", type=pathlib.Path, required=True)
  args = parser.parse_args()
  scorecard = build_scorecard([p.expanduser() for p in args.experiments],
                              rollout_compare=args.rollout_compare.expanduser() if args.rollout_compare else None)
  write_scorecard(scorecard, args.json, args.md)
  print(scorecard_markdown(scorecard))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
