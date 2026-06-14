#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib, re
from collections import Counter, defaultdict
from typing import Any

from extra.llm_sft_smoke_train import load_sft_rows

def read_jsonl(path:pathlib.Path) -> list[dict[str, Any]]:
  if not path.exists(): return []
  rows = []
  for lineno, raw in enumerate(path.read_text().splitlines(), 1):
    if not raw.strip(): continue
    row = json.loads(raw)
    if not isinstance(row, dict): raise ValueError(f"{path}:{lineno}: expected JSON object")
    rows.append(row)
  return rows

def source_maps(sft:pathlib.Path) -> dict[str, dict[str, Any]]:
  rows = load_sft_rows(sft)
  by_id = {row["id"]: row for row in rows}
  if len(by_id) != len(rows): raise ValueError(f"{sft}: duplicate source row ids")
  return by_id

def answer_from_score(row:dict[str, Any]) -> Any:
  actual = row.get("score", {}).get("json_axes", {}).get("actual")
  if isinstance(actual, dict) and set(actual) == {"answer"}: return actual["answer"]
  return actual

def answer_key(value:Any) -> str:
  return json.dumps(value, sort_keys=True, separators=(",", ":"))

def strip_numeric_suffix(value:str) -> str:
  return re.sub(r"_\d+$", "", value)

def classify_value_miss(expected:Any, actual:Any) -> str:
  if type(expected) is not type(actual): return "type_mismatch_value"
  if not isinstance(expected, str) or not isinstance(actual, str): return "wrong_value"
  if actual == "": return "empty_string"
  expected_stem = strip_numeric_suffix(expected)
  if actual == expected_stem: return "stem_without_index"
  if expected.startswith(actual): return "prefix"
  if actual.startswith(expected_stem): return "stem_plus_extra"
  if actual in expected: return "substring"
  return "wrong_string"

def axis_counts(rows:list[dict[str, Any]]) -> dict[str, dict[str, int]]:
  out: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "scored": 0})
  for row in rows:
    for axis, passed in row.get("score", {}).get("json_axes", {}).get("axes", {}).items():
      out[axis]["scored"] += 1
      if passed is True: out[axis]["passed"] += 1
  return {axis: counts for axis, counts in sorted(out.items())}

def top(counter:Counter[str], n:int=12) -> list[dict[str, Any]]:
  return [{"value": key, "count": count} for key, count in counter.most_common(n)]

def artifact_report(path:pathlib.Path, source_by_id:dict[str, dict[str, Any]], category:str) -> dict[str, Any]:
  samples = [row for row in read_jsonl(path / "samples.jsonl") if row.get("category") == category]
  near = [row for row in read_jsonl(path / "near-miss.jsonl") if row.get("category") == category]
  accepted = [row for row in read_jsonl(path / "accepted.jsonl") if row.get("category") == category]
  selected_sources = {row.get("source_id") for row in read_jsonl(path / "sft.jsonl") if row.get("split") == "train" and row.get("category") == category}

  by_template: Counter[str] = Counter()
  by_actual: Counter[str] = Counter()
  by_expected: Counter[str] = Counter()
  by_classification: Counter[str] = Counter()
  by_temperature: Counter[str] = Counter()
  examples = []
  for row in near:
    source = source_by_id.get(row["source_id"], {})
    expected = row.get("expected_json", {}).get("answer")
    actual = answer_from_score(row)
    by_template[source.get("template_id", "unknown")] += 1
    by_actual[answer_key(actual)] += 1
    by_expected[answer_key(expected)] += 1
    by_classification[classify_value_miss(expected, actual)] += 1
    by_temperature[str(row.get("temperature"))] += 1
    if len(examples) < 12:
      examples.append({
        "actual": actual,
        "expected": expected,
        "id": row.get("id"),
        "sample_idx": row.get("sample_idx"),
        "source_id": row.get("source_id"),
        "temperature": row.get("temperature"),
        "template_id": source.get("template_id"),
        "text": row.get("text"),
      })

  return {
    "artifact": str(path),
    "accepted_attempts": len(accepted),
    "attempts": len(samples),
    "axis_counts": axis_counts(samples),
    "classification": top(by_classification),
    "examples": examples,
    "near_miss": len(near),
    "near_miss_actual_top": top(by_actual),
    "near_miss_expected_top": top(by_expected),
    "near_miss_by_template": top(by_template),
    "near_miss_by_temperature": top(by_temperature),
    "selected_train_rows": len(selected_sources),
    "unique_near_miss_sources": len({row.get("source_id") for row in near}),
  }

def rollout_report(path:pathlib.Path, source_by_id:dict[str, dict[str, Any]], category:str) -> dict[str, Any]:
  rows = [row for row in read_jsonl(path / "rollouts.jsonl") if category in (row.get("tags") or [])]
  by_template: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "scored": 0})
  by_actual: Counter[str] = Counter()
  examples = []
  for row in rows:
    source = source_by_id.get(row["id"], {})
    template = source.get("template_id", "unknown")
    by_template[template]["scored"] += 1
    if row.get("score", {}).get("passed") is True: by_template[template]["passed"] += 1
    by_actual[answer_key(answer_from_score(row))] += 1
    if len(examples) < 12:
      examples.append({
        "actual": answer_from_score(row),
        "expected": row.get("score", {}).get("json_axes", {}).get("expected", {}).get("answer"),
        "id": row.get("id"),
        "passed": row.get("score", {}).get("passed"),
        "template_id": template,
        "text": row.get("text"),
      })
  return {
    "artifact": str(path),
    "axis_counts": axis_counts(rows),
    "examples": examples,
    "passed": sum(1 for row in rows if row.get("score", {}).get("passed") is True),
    "rollouts": len(rows),
    "top_actual": top(by_actual),
    "by_template": {template: counts for template, counts in sorted(by_template.items())},
  }

def choose_intervention(artifacts:list[dict[str, Any]]) -> dict[str, Any]:
  accepted = sum(row["accepted_attempts"] for row in artifacts)
  near = sum(row["near_miss"] for row in artifacts)
  classifications = Counter()
  for artifact in artifacts:
    classifications.update({row["value"]: row["count"] for row in artifact["classification"]})
  if accepted == 0 and near > 0 and (classifications["stem_without_index"] or classifications["prefix"] or classifications["empty_string"]):
    return {
      "choice": "prompt_data_fix",
      "rationale": [
        "Compiler failures are mostly valid JSON with wrong values, not form failures.",
        "The generated values often drop the row-specific numeric suffix or collapse to a broad prefix.",
        "Accepting those by normalization would change the task contract, so the data/prompt target should be redesigned before training V7.",
      ],
    }
  return {
    "choice": "manual_review",
    "rationale": ["The near-miss distribution is not dominated by a single obvious contract issue."],
  }

def build_report(sft:pathlib.Path, artifacts:list[pathlib.Path], *, category:str, rollout:pathlib.Path|None=None) -> dict[str, Any]:
  source_by_id = source_maps(sft)
  artifact_reports = [artifact_report(path, source_by_id, category) for path in artifacts]
  report = {
    "kind": "llm_json_nearmiss_audit",
    "category": category,
    "artifacts": artifact_reports,
    "intervention": choose_intervention(artifact_reports),
    "source": str(sft),
  }
  if rollout is not None: report["rollout"] = rollout_report(rollout, source_by_id, category)
  return report

def md_table(rows:list[dict[str, Any]], columns:list[tuple[str, str]]) -> list[str]:
  out = ["| " + " | ".join(label for label, _ in columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
  for row in rows:
    out.append("| " + " | ".join(str(row.get(key, "")) for _, key in columns) + " |")
  return out

def markdown(report:dict[str, Any]) -> str:
  lines = [
    "# Compiler Near-Miss Audit",
    "",
    f"- category: `{report['category']}`",
    f"- intervention: `{report['intervention']['choice']}`",
    "",
    "## Rationale",
    "",
  ]
  lines += [f"- {item}" for item in report["intervention"]["rationale"]]
  for artifact in report["artifacts"]:
    lines += [
      "",
      "## Artifact",
      "",
      f"- path: `{artifact['artifact']}`",
      f"- attempts: `{artifact['attempts']}`",
      f"- accepted attempts: `{artifact['accepted_attempts']}`",
      f"- selected train rows: `{artifact['selected_train_rows']}`",
      f"- near misses: `{artifact['near_miss']}`",
      f"- unique near-miss sources: `{artifact['unique_near_miss_sources']}`",
      "",
      "### Miss Classification",
      "",
    ]
    lines += md_table(artifact["classification"], [("classification", "value"), ("count", "count")])
    lines += ["", "### Top Actual Answers", ""]
    lines += md_table(artifact["near_miss_actual_top"], [("actual", "value"), ("count", "count")])
    lines += ["", "### Near Misses By Template", ""]
    lines += md_table(artifact["near_miss_by_template"], [("template", "value"), ("count", "count")])
  if "rollout" in report:
    rollout = report["rollout"]
    lines += [
      "",
      "## V6 Eval Rollout Reference",
      "",
      f"- path: `{rollout['artifact']}`",
      f"- passed: `{rollout['passed']}/{rollout['rollouts']}`",
      "",
      "| template | passed | scored |",
      "|---|---:|---:|",
    ]
    for template, counts in rollout["by_template"].items():
      lines.append(f"| `{template}` | {counts['passed']} | {counts['scored']} |")
  lines.append("")
  return "\n".join(lines)

def main() -> int:
  parser = argparse.ArgumentParser(description="Audit strict-JSON near misses for a category")
  parser.add_argument("--sft", type=pathlib.Path, required=True)
  parser.add_argument("--artifact", action="append", type=pathlib.Path, required=True)
  parser.add_argument("--category", default="compiler")
  parser.add_argument("--rollout", type=pathlib.Path)
  parser.add_argument("--out", type=pathlib.Path, required=True)
  args = parser.parse_args()
  report = build_report(args.sft, args.artifact, category=args.category, rollout=args.rollout)
  args.out.mkdir(parents=True, exist_ok=True)
  (args.out / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True))
  (args.out / "README.md").write_text(markdown(report))
  print(markdown(report))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
