#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib, subprocess, sys
from typing import Any

DEFAULT_MANIFEST = pathlib.Path("bench/qwen-eval-20260612/manifest.json")

def _load_json(path:pathlib.Path) -> dict[str, Any]:
  return json.loads(path.read_text())

def _repo_path(repo:pathlib.Path, value:str | pathlib.Path) -> pathlib.Path:
  path = pathlib.Path(value).expanduser()
  return path if path.is_absolute() else repo / path

def _cmd_path(value:str) -> str:
  path = pathlib.Path(value)
  if value.startswith("~") or path.is_absolute(): return str(path.expanduser())
  return value

def load_manifest(path:pathlib.Path) -> dict[str, Any]:
  data = _load_json(path)
  if data.get("kind") != "qwen_eval_manifest": raise ValueError(f"{path}: expected kind=qwen_eval_manifest")
  if not isinstance(data.get("rows"), list) or not data["rows"]: raise ValueError(f"{path}: expected non-empty rows")
  seen: set[str] = set()
  for idx, row in enumerate(data["rows"]):
    if not isinstance(row, dict): raise ValueError(f"{path}: row {idx} must be an object")
    for key in ("id", "model_size", "model", "policy", "out"):
      if not isinstance(row.get(key), str) or not row[key]: raise ValueError(f"{path}: row {idx} missing string {key}")
    if row["id"] in seen: raise ValueError(f"{path}: duplicate row id {row['id']!r}")
    seen.add(row["id"])
  return data

def _selected_rows(manifest:dict[str, Any], only:set[str], include_disabled:bool) -> list[dict[str, Any]]:
  rows = []
  for row in manifest["rows"]:
    if only and row["id"] not in only and row["model_size"].lower() not in only: continue
    if not include_disabled and row.get("enabled", True) is False: continue
    rows.append(row)
  if only and not rows: raise ValueError(f"no manifest rows matched {sorted(only)}")
  return rows

def run_rows(manifest:dict[str, Any], args:argparse.Namespace) -> None:
  rows = _selected_rows(manifest, set(args.only), args.include_disabled)
  for row in rows:
    out = _repo_path(args.repo, row["out"])
    if args.reuse and (out / "summary.json").exists():
      print(f"reuse {row['id']}: {out / 'summary.json'}")
      continue
    cmd = [
      sys.executable, "extra/qwen_eval_harness.py",
      "--model", _cmd_path(row["model"]),
      "--policy", _cmd_path(row["policy"]),
      "--prompts", _cmd_path(manifest["prompts"]),
      "--out", _cmd_path(row["out"]),
      "--tokens", str(row.get("tokens", manifest.get("tokens", 64))),
      "--max-context", str(row.get("max_context", manifest.get("max_context", 4096))),
      "--temperature", str(row.get("temperature", manifest.get("temperature", 0.0))),
      "--seed", str(row.get("seed", manifest.get("seed", 20260612))),
      "--device", str(row.get("device", manifest.get("device", "AMD"))),
      "--storage", str(row.get("storage", manifest.get("storage", "shared"))),
      "--timeout", str(row.get("timeout", manifest.get("timeout", 1800))),
    ]
    if args.policy_debug or row.get("policy_debug", manifest.get("policy_debug", False)): cmd.append("--policy-debug")
    print(f"run {row['id']}: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=args.repo, text=True)
    if proc.returncode != 0:
      if args.keep_going: print(f"row {row['id']} failed rc={proc.returncode}")
      else: raise RuntimeError(f"row {row['id']} failed rc={proc.returncode}")

def row_from_summary(repo:pathlib.Path, row:dict[str, Any]) -> dict[str, Any]:
  out = _repo_path(repo, row["out"])
  summary_path = out / "summary.json"
  if not summary_path.exists():
    return {"id": row["id"], "model_size": row["model_size"], "enabled": row.get("enabled", True),
            "out": row["out"], "status": "missing", "reason": row.get("reason")}
  summary = _load_json(summary_path)
  explicit, generated = summary["modes"]["explicit"], summary["modes"]["generated"]
  quality = summary.get("quality") or {"status": "unscored", "passed": 0, "scored": 0, "pass_rate": None}
  return {
    "id": row["id"], "model_size": row["model_size"], "enabled": row.get("enabled", True), "out": row["out"],
    "status": summary.get("status"), "tokens_match": summary.get("tokens_match"),
    "quality_status": quality.get("status"), "quality_passed": quality.get("passed"),
    "quality_scored": quality.get("scored"), "quality_pass_rate": quality.get("pass_rate"),
    "prompts": summary.get("prompts"), "policy": summary.get("policy"), "storage": summary.get("storage"),
    "explicit_tok_s": explicit.get("tok_s"), "generated_tok_s": generated.get("tok_s"),
    "explicit_tokens": explicit.get("generated"), "generated_tokens": generated.get("generated"),
    "reason": row.get("reason"),
  }

def make_matrix(manifest:dict[str, Any], repo:pathlib.Path, include_disabled:bool=False) -> dict[str, Any]:
  rows = [row_from_summary(repo, row) for row in _selected_rows(manifest, set(), include_disabled)]
  accepted = [row for row in rows if row.get("status") == "pass"]
  quality_scored = [row for row in rows if row.get("quality_status") != "unscored" and row.get("status") != "missing"]
  return {
    "kind": "qwen_eval_matrix",
    "prompts": manifest.get("prompts"),
    "tokens": manifest.get("tokens"),
    "rows": rows,
    "summary": {
      "rows": len(rows),
      "parity_passed": len(accepted),
      "quality_scored_rows": len(quality_scored),
      "quality_passed_rows": sum(1 for row in quality_scored if row.get("quality_status") == "pass"),
    },
  }

def _fmt(val:Any, digits:int=2) -> str:
  if val is None: return "n/a"
  if isinstance(val, float): return f"{val:.{digits}f}"
  return str(val)

def matrix_markdown(matrix:dict[str, Any]) -> str:
  lines = [
    "# Qwen Eval Matrix",
    "",
    "This matrix is the shared gate for three compounding goals:",
    "fast local inference, future training/rollout/eval loops, and compiler/search",
    "experiments. Exact token parity is the primitive correctness gate; prompt",
    "quality is reported separately.",
    "",
    "| id | model | parity | quality | score | explicit tok/s | generated tok/s | tokens | out |",
    "|---|---|---:|---:|---:|---:|---:|---:|---|",
  ]
  for row in matrix["rows"]:
    score = "n/a" if row.get("quality_scored") in (None, 0) else f"{row.get('quality_passed')}/{row.get('quality_scored')}"
    lines.append(
      f"| `{row['id']}` | `{row['model_size']}` | `{row.get('status')}` | `{row.get('quality_status', 'n/a')}` | "
      f"{score} | {_fmt(row.get('explicit_tok_s'))} | {_fmt(row.get('generated_tok_s'))} | "
      f"{_fmt(row.get('generated_tokens'), 0)} | `{row.get('out')}` |"
    )
  lines += ["", "## Summary", "", "```json", json.dumps(matrix["summary"], indent=2, sort_keys=True), "```", ""]
  return "\n".join(lines)

def main() -> int:
  parser = argparse.ArgumentParser(description="Run and summarize the Qwen generated-policy eval matrix")
  parser.add_argument("--manifest", type=pathlib.Path, default=DEFAULT_MANIFEST)
  parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path.cwd())
  parser.add_argument("--run", action="store_true", help="run enabled manifest rows before writing the matrix")
  parser.add_argument("--reuse", action="store_true", help="skip rows whose summary.json already exists")
  parser.add_argument("--include-disabled", action="store_true")
  parser.add_argument("--only", nargs="*", default=[], help="row ids or model sizes to run")
  parser.add_argument("--keep-going", action="store_true")
  parser.add_argument("--policy-debug", action="store_true")
  parser.add_argument("--json", type=pathlib.Path)
  parser.add_argument("--md", type=pathlib.Path)
  args = parser.parse_args()

  args.repo = args.repo.resolve()
  manifest = load_manifest(_repo_path(args.repo, args.manifest))
  if args.run: run_rows(manifest, args)
  matrix = make_matrix(manifest, args.repo, args.include_disabled)
  if args.json:
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(matrix, indent=2, sort_keys=True))
  if args.md:
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text(matrix_markdown(matrix))
  if not args.json and not args.md: print(matrix_markdown(matrix))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
