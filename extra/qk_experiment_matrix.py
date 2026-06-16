#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib, statistics

LLAMA_REFS = {"8B": 101.2, "14B": 65.8, "32B": 30.8}

def _load_json(path:pathlib.Path) -> dict:
  return json.loads(path.read_text())

def _decision_path(path:pathlib.Path) -> pathlib.Path:
  if path.is_dir(): return path / "decision.json"
  if path.name == "decision.json": return path
  raise ValueError(f"{path}: expected experiment directory or decision.json")

def _expand_inputs(paths:list[pathlib.Path]) -> list[pathlib.Path]:
  out: list[pathlib.Path] = []
  for path in paths:
    if path.is_dir() or path.name == "decision.json":
      out.append(_decision_path(path))
      continue
    data = _load_json(path)
    experiments = data.get("experiments")
    if not isinstance(experiments, list): raise ValueError(f"{path}: expected experiments list")
    for item in experiments:
      item_path = pathlib.Path(item["out"] if isinstance(item, dict) else item).expanduser()
      out.append(_decision_path(item_path))
  return out

def _fmt(x, digits:int=2) -> str:
  if x is None: return "n/a"
  if isinstance(x, float): return f"{x:.{digits}f}"
  return str(x)

def _last_runtime_storage(decision:dict, prefix:str) -> dict:
  rows = decision.get("runtime_storage") or {}
  matches = [(k, v) for k, v in rows.items() if k.startswith(prefix)]
  return matches[-1][1] if matches else {}

def row_from_decision(path:pathlib.Path) -> dict:
  decision = _load_json(path)
  explicit = decision.get("explicit") or {}
  generated = decision.get("generated") or {}
  storage = decision.get("storage_policy") or {}
  generated_storage = _last_runtime_storage(decision, "generated")
  model_size = decision.get("model_size", "unknown")
  llama = LLAMA_REFS.get(model_size)
  generated_tok_s = generated.get("avg_tok_s")
  return {
    "path": str(path.parent),
    "model_size": model_size,
    "status": decision.get("status") or "n/a",
    "reference_mode": decision.get("reference_mode") or "n/a",
    "gain": decision.get("gain"),
    "explicit_tok_s": explicit.get("avg_tok_s"),
    "generated_tok_s": generated_tok_s,
    "generated_pct_llama": None if llama is None or generated_tok_s is None else generated_tok_s / llama * 100,
    "ab_match": decision.get("ab_match"),
    "policy_selected_bytes": storage.get("selected_bytes"),
    "policy_cap_bytes": storage.get("cap_bytes"),
    "policy_selected_primitive_entries": storage.get("selected_primitive_entries"),
    "runtime_storage_bytes": generated_storage.get("storage_bytes"),
    "runtime_cap_bytes": generated_storage.get("runtime_cap_bytes"),
    "runtime_cap_used_bytes": generated_storage.get("runtime_cap_used_bytes"),
    "reasons": decision.get("reasons", []),
  }

def make_matrix(paths:list[pathlib.Path]) -> dict:
  decision_paths = _expand_inputs(paths)
  rows = [row_from_decision(path) for path in decision_paths]
  accepted = [r for r in rows if r["status"] == "accept"]
  gains = [r["gain"] for r in accepted if isinstance(r.get("gain"), (int, float))]
  statuses = {}
  for row in rows:
    key = str(row["status"])
    statuses[key] = statuses.get(key, 0) + 1
  return {
    "kind": "qk_experiment_matrix",
    "rows": rows,
    "summary": {
      "experiments": len(rows),
      "accepted": len(accepted),
      "statuses": dict(sorted(statuses.items())),
      "accepted_mean_gain": statistics.mean(gains) if gains else None,
    },
  }

def matrix_markdown(matrix:dict) -> str:
  lines = [
    "# QK Experiment Matrix",
    "",
    "| path | model | status | ref | explicit tok/s | generated tok/s | gain % | % llama | A/B | policy MB | runtime MB |",
    "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
  ]
  for row in matrix["rows"]:
    policy_mb = None if row["policy_selected_bytes"] is None else row["policy_selected_bytes"] / (1024*1024)
    runtime_mb = None if row["runtime_storage_bytes"] is None else row["runtime_storage_bytes"] / (1024*1024)
    gain_pct = None if row["gain"] is None else row["gain"] * 100
    lines.append(
      f"| `{row['path']}` | `{row['model_size']}` | `{row['status']}` | `{row['reference_mode']}` | "
      f"{_fmt(row['explicit_tok_s'])} | {_fmt(row['generated_tok_s'])} | {_fmt(gain_pct)} | "
      f"{_fmt(row['generated_pct_llama'])} | {_fmt(row['ab_match'])} | {_fmt(policy_mb)} | {_fmt(runtime_mb)} |"
    )
  lines += ["", "## Summary", "", "```json", json.dumps(matrix["summary"], indent=2, sort_keys=True), "```", ""]
  return "\n".join(lines)

def main() -> None:
  parser = argparse.ArgumentParser(description="Summarize QK policy pipeline decision directories")
  parser.add_argument("experiments", nargs="+", type=pathlib.Path,
                      help="experiment directory, decision.json, or JSON file with an experiments list")
  parser.add_argument("--json", type=pathlib.Path)
  parser.add_argument("--md", type=pathlib.Path)
  args = parser.parse_args()

  matrix = make_matrix([x.expanduser() for x in args.experiments])
  if args.json:
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(matrix, indent=2, sort_keys=True))
  if args.md:
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text(matrix_markdown(matrix))
  if not args.json and not args.md:
    print(matrix_markdown(matrix))

if __name__ == "__main__":
  main()
