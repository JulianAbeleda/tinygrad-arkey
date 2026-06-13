#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib
from typing import Any

from extra.llm_eval_common import md_text

def _read_json(path:pathlib.Path) -> Any:
  try:
    return json.loads(path.read_text())
  except json.JSONDecodeError as exc:
    raise ValueError(f"{path}: invalid JSON: {exc}") from exc

def _read_rows(path:pathlib.Path) -> list[dict[str, Any]]:
  rows = []
  seen: set[str] = set()
  for lineno, raw in enumerate(path.read_text().splitlines(), 1):
    if not raw.strip(): continue
    try:
      row = json.loads(raw)
    except json.JSONDecodeError as exc:
      raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    if not isinstance(row, dict): raise ValueError(f"{path}:{lineno}: expected JSON object")
    row_id = row.get("id")
    if not isinstance(row_id, str) or not row_id: raise ValueError(f"{path}:{lineno}: missing string id")
    if row_id in seen: raise ValueError(f"{path}:{lineno}: duplicate id {row_id!r}")
    seen.add(row_id)
    rows.append(row)
  if not rows: raise ValueError(f"{path}: no rollout rows")
  return rows

def load_artifact(path:pathlib.Path, *, allow_unscored:bool=False) -> dict[str, Any]:
  summary_path, rows_path = path / "summary.json", path / "rollouts.jsonl"
  if not summary_path.exists(): raise ValueError(f"{path}: missing summary.json")
  if not rows_path.exists(): raise ValueError(f"{path}: missing rollouts.jsonl")
  summary = _read_json(summary_path)
  if not isinstance(summary, dict) or summary.get("kind") != "llm_rollout_summary":
    raise ValueError(f"{summary_path}: expected kind=llm_rollout_summary")
  rows = _read_rows(rows_path)
  if summary.get("prompts") != len(rows):
    raise ValueError(f"{path}: summary prompts={summary.get('prompts')} but rollouts has {len(rows)} rows")
  unscored = [row["id"] for row in rows if row.get("score", {}).get("status") not in ("pass", "fail")]
  if unscored and not allow_unscored:
    raise ValueError(f"{path}: unscored rows {unscored[:5]}")
  return {"label": path.name, "path": str(path), "summary": summary, "rows": rows, "by_id": {row["id"]: row for row in rows}}

def artifact_summary(artifact:dict[str, Any]) -> dict[str, Any]:
  summary = artifact["summary"]
  quality = summary["quality"]
  out = {
    "label": artifact["label"],
    "path": artifact["path"],
    "mode": summary["mode"],
    "model": summary["model"],
    "policy": summary["policy"],
    "dataset": summary["dataset"],
    "storage": summary["storage"],
    "prompt_format": summary["prompt_format"],
    "prompts": summary["prompts"],
    "generated": summary["generated"],
    "elapsed_s": summary["elapsed_s"],
    "tok_s": summary["tok_s"],
    "quality": {
      "status": quality["status"],
      "passed": quality["passed"],
      "scored": quality["scored"],
      "pass_rate": quality["pass_rate"],
    },
  }
  if summary.get("adapter") is not None: out["adapter"] = summary["adapter"]
  return out

def _passed(row:dict[str, Any]) -> bool | None:
  return row.get("score", {}).get("passed")

def _status(row:dict[str, Any]) -> str:
  return str(row.get("score", {}).get("status", "missing"))

def compare_pair(base:dict[str, Any], cand:dict[str, Any], *, allow_dataset_mismatch:bool=False) -> dict[str, Any]:
  bsum, csum = base["summary"], cand["summary"]
  if bsum.get("dataset") != csum.get("dataset") and not allow_dataset_mismatch:
    raise ValueError(f"{base['label']} and {cand['label']}: dataset mismatch {bsum.get('dataset')!r} != {csum.get('dataset')!r}")
  base_ids, cand_ids = set(base["by_id"]), set(cand["by_id"])
  if base_ids != cand_ids:
    missing = sorted(base_ids - cand_ids)
    extra = sorted(cand_ids - base_ids)
    raise ValueError(f"{cand['label']}: id mismatch missing={missing[:5]} extra={extra[:5]}")

  changed, regressions, improvements = [], [], []
  text_equal = token_equal = 0
  tag_delta: dict[str, dict[str, int]] = {}
  for row_id in sorted(base_ids):
    brow, crow = base["by_id"][row_id], cand["by_id"][row_id]
    same_text = brow.get("text") == crow.get("text")
    same_tokens = brow.get("tokens") == crow.get("tokens")
    if same_text: text_equal += 1
    if same_tokens: token_equal += 1
    bpass, cpass = _passed(brow), _passed(crow)
    if bpass is True and cpass is False: regressions.append(row_id)
    if bpass is False and cpass is True: improvements.append(row_id)
    for tag in crow.get("tags") or ["untagged"]:
      cur = tag_delta.setdefault(tag, {"baseline_passed": 0, "candidate_passed": 0, "scored": 0})
      cur["scored"] += 1
      if bpass is True: cur["baseline_passed"] += 1
      if cpass is True: cur["candidate_passed"] += 1
    if not same_text or not same_tokens or bpass != cpass:
      changed.append({
        "id": row_id,
        "tags": crow.get("tags", []),
        "baseline_status": _status(brow),
        "candidate_status": _status(crow),
        "baseline_generated": brow.get("generated"),
        "candidate_generated": crow.get("generated"),
        "text_equal": same_text,
        "tokens_equal": same_tokens,
        "baseline_text": brow.get("text", ""),
        "candidate_text": crow.get("text", ""),
      })

  prompts = len(base_ids)
  return {
    "baseline": base["label"],
    "candidate": cand["label"],
    "dataset": bsum.get("dataset"),
    "prompts": prompts,
    "quality": {
      "baseline_passed": bsum["quality"]["passed"],
      "candidate_passed": csum["quality"]["passed"],
      "passed_delta": csum["quality"]["passed"] - bsum["quality"]["passed"],
      "regressions": regressions,
      "improvements": improvements,
    },
    "outputs": {
      "text_equal": text_equal,
      "text_changed": prompts - text_equal,
      "tokens_equal": token_equal,
      "tokens_changed": prompts - token_equal,
      "changed": changed,
    },
    "timing_sanity": {
      "baseline_generated": bsum["generated"],
      "candidate_generated": csum["generated"],
      "generated_delta": csum["generated"] - bsum["generated"],
      "baseline_elapsed_s": bsum["elapsed_s"],
      "candidate_elapsed_s": csum["elapsed_s"],
      "elapsed_delta_s": csum["elapsed_s"] - bsum["elapsed_s"],
      "baseline_tok_s": bsum["tok_s"],
      "candidate_tok_s": csum["tok_s"],
      "tok_s_ratio": None if bsum["tok_s"] == 0 else csum["tok_s"] / bsum["tok_s"],
    },
    "tag_delta": {tag: {**vals, "passed_delta": vals["candidate_passed"] - vals["baseline_passed"]} for tag, vals in sorted(tag_delta.items())},
  }

def build_report(paths:list[pathlib.Path], *, allow_dataset_mismatch:bool=False, allow_unscored:bool=False) -> dict[str, Any]:
  if len(paths) < 2: raise ValueError("expected at least two rollout artifact directories")
  artifacts = [load_artifact(path, allow_unscored=allow_unscored) for path in paths]
  base = artifacts[0]
  return {
    "kind": "llm_rollout_compare_report",
    "baseline": base["label"],
    "artifacts": [artifact_summary(artifact) for artifact in artifacts],
    "comparisons": [compare_pair(base, cand, allow_dataset_mismatch=allow_dataset_mismatch) for cand in artifacts[1:]],
  }

def report_markdown(report:dict[str, Any]) -> str:
  lines = [
    "# LLM Rollout Compare Report",
    "",
    "Deterministic offline comparison for committed rollout artifacts. This checks",
    "quality scores, prompt coverage, output equality, and eval-loop timing sanity.",
    "It is not an LLM-as-judge or broad capability benchmark.",
    "",
    "## Artifacts",
    "",
    "| label | mode | prompts | quality | generated | tok/s | dataset |",
    "|---|---|---:|---:|---:|---:|---|",
  ]
  for artifact in report["artifacts"]:
    q = artifact["quality"]
    lines.append(
      f"| `{artifact['label']}` | `{artifact['mode']}` | {artifact['prompts']} | "
      f"`{q['status']}` {q['passed']}/{q['scored']} | {artifact['generated']} | "
      f"{artifact['tok_s']:.2f} | `{artifact['dataset']}` |"
    )
  lines += ["", "## Comparisons", ""]
  for comp in report["comparisons"]:
    q, out, timing = comp["quality"], comp["outputs"], comp["timing_sanity"]
    lines += [
      f"### `{comp['baseline']}` vs `{comp['candidate']}`",
      "",
      f"- quality delta: `{q['passed_delta']}` ({q['baseline_passed']} -> {q['candidate_passed']})",
      f"- regressions: `{len(q['regressions'])}`",
      f"- improvements: `{len(q['improvements'])}`",
      f"- text changed: `{out['text_changed']}/{comp['prompts']}`",
      f"- tokens changed: `{out['tokens_changed']}/{comp['prompts']}`",
      f"- generated token delta: `{timing['generated_delta']}`",
      f"- eval-loop tok/s ratio: `{timing['tok_s_ratio']:.4f}`" if timing["tok_s_ratio"] is not None else "- eval-loop tok/s ratio: `n/a`",
      "",
      "| tag | baseline passed | candidate passed | delta | scored |",
      "|---|---:|---:|---:|---:|",
    ]
    for tag, row in comp["tag_delta"].items():
      lines.append(f"| `{tag}` | {row['baseline_passed']} | {row['candidate_passed']} | {row['passed_delta']} | {row['scored']} |")
    if out["changed"]:
      lines += ["", "| id | quality | tokens equal | text |", "|---|---|---:|---|"]
      for row in out["changed"][:25]:
        lines.append(
          f"| `{row['id']}` | `{row['baseline_status']} -> {row['candidate_status']}` | "
          f"`{row['tokens_equal']}` | {md_text(row['candidate_text'])} |"
        )
      if len(out["changed"]) > 25:
        lines.append(f"| ... | ... | ... | {len(out['changed']) - 25} more changed rows omitted from markdown |")
    lines.append("")
  return "\n".join(lines)

def write_report(report:dict[str, Any], out:pathlib.Path) -> None:
  out.mkdir(parents=True, exist_ok=True)
  (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
  (out / "report.md").write_text(report_markdown(report))

def main() -> int:
  parser = argparse.ArgumentParser(description="Compare committed LLM rollout artifacts")
  parser.add_argument("artifacts", nargs="+", type=pathlib.Path, help="rollout artifact directories; first is baseline")
  parser.add_argument("--out", type=pathlib.Path, required=True)
  parser.add_argument("--allow-dataset-mismatch", action="store_true")
  parser.add_argument("--allow-unscored", action="store_true")
  args = parser.parse_args()
  report = build_report(args.artifacts, allow_dataset_mismatch=args.allow_dataset_mismatch, allow_unscored=args.allow_unscored)
  write_report(report, args.out)
  print(report_markdown(report))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
