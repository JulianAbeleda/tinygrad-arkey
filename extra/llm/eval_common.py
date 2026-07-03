from __future__ import annotations

import json, pathlib, re, statistics
from typing import Any

from extra.llm.json_scorer import score_expected_json, summarize_json_axes, wilson_interval

def read_jsonl(path:pathlib.Path) -> list[dict[str, Any]]:
  rows = []
  for lineno, raw in enumerate(path.read_text().splitlines(), 1):
    if not raw.strip(): continue
    try:
      row = json.loads(raw)
    except json.JSONDecodeError as exc:
      raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    if not isinstance(row, dict): raise ValueError(f"{path}:{lineno}: expected JSON object")
    rows.append(row)
  return rows

def read_id_jsonl(path:pathlib.Path) -> list[dict[str, Any]]:
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
  return rows

def write_jsonl(path:pathlib.Path, rows:list[dict[str, Any]]) -> None:
  with path.open("w") as f:
    for row in rows: f.write(json.dumps(row, sort_keys=True) + "\n")

def read_json_object(path:pathlib.Path) -> dict[str, Any]:
  data = json.loads(path.read_text())
  if not isinstance(data, dict): raise ValueError(f"{path}: expected JSON object")
  return data

def load_json(path:pathlib.Path) -> Any:
  try:
    return json.loads(path.read_text())
  except json.JSONDecodeError as exc:
    raise ValueError(f"{path}: invalid JSON: {exc}") from exc

def write_json(path:pathlib.Path, data:Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(data, indent=2, sort_keys=True))

def value_stats(values:list[float]) -> dict[str, float|int]:
  if not values: raise ValueError("cannot summarize empty values")
  return {
    "n": len(values),
    "median": statistics.median(values),
    "min": min(values),
    "max": max(values),
    "mean": statistics.fmean(values),
    "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
  }

def read_prompt_jsonl(path:pathlib.Path) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  seen: set[str] = set()
  for lineno, raw in enumerate(path.read_text().splitlines(), 1):
    line = raw.strip()
    if not line or line.startswith("#"): continue
    try:
      row = json.loads(line)
    except json.JSONDecodeError as exc:
      raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    if not isinstance(row, dict): raise ValueError(f"{path}:{lineno}: expected JSON object")
    prompt_id, prompt = row.get("id"), row.get("prompt")
    if not isinstance(prompt_id, str) or not prompt_id: raise ValueError(f"{path}:{lineno}: missing string id")
    if not isinstance(prompt, str) or not prompt: raise ValueError(f"{path}:{lineno}: missing string prompt")
    if "tags" in row and (not isinstance(row["tags"], list) or not all(isinstance(x, str) for x in row["tags"])):
      raise ValueError(f"{path}:{lineno}: tags must be a list of strings")
    for key in ("expected_contains", "expected_regex"):
      if key not in row: continue
      val = row[key]
      if isinstance(val, str): vals = [val]
      elif isinstance(val, list) and all(isinstance(x, str) for x in val): vals = val
      else: raise ValueError(f"{path}:{lineno}: {key} must be a string or list of strings")
      if key == "expected_regex":
        for pattern in vals:
          try: re.compile(pattern)
          except re.error as exc: raise ValueError(f"{path}:{lineno}: invalid expected_regex {pattern!r}: {exc}") from exc
    if "expected_exact" in row and not isinstance(row["expected_exact"], str):
      raise ValueError(f"{path}:{lineno}: expected_exact must be a string")
    if "expected_json" in row and (not isinstance(row["expected_json"], dict) or not row["expected_json"]):
      raise ValueError(f"{path}:{lineno}: expected_json must be a non-empty object")
    if "case_insensitive" in row and not isinstance(row["case_insensitive"], bool):
      raise ValueError(f"{path}:{lineno}: case_insensitive must be a boolean")
    if "max_tokens" in row and (not isinstance(row["max_tokens"], int) or row["max_tokens"] <= 0):
      raise ValueError(f"{path}:{lineno}: max_tokens must be a positive integer")
    if prompt_id in seen: raise ValueError(f"{path}:{lineno}: duplicate id {prompt_id!r}")
    seen.add(prompt_id)
    rows.append(row)
  if not rows: raise ValueError(f"{path}: no prompts")
  return rows

def as_str_list(val:Any) -> list[str]:
  if val is None: return []
  return [val] if isinstance(val, str) else list(val)

def score_prompt(prompt:dict[str, Any], text:str) -> dict[str, Any]:
  checks = []
  lowered = text.lower()
  for needle in as_str_list(prompt.get("expected_contains")):
    ok = needle.lower() in lowered
    checks.append({"kind": "contains", "value": needle, "passed": ok})
  for pattern in as_str_list(prompt.get("expected_regex")):
    ok = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE) is not None
    checks.append({"kind": "regex", "value": pattern, "passed": ok})
  if "expected_exact" in prompt:
    expected = prompt["expected_exact"]
    checks.append({"kind": "exact", "value": expected, "passed": text.strip() == expected})
  if "expected_json" in prompt:
    expected = prompt["expected_json"]
    json_axes = score_expected_json(text, expected, case_insensitive=bool(prompt.get("case_insensitive", False)))
    detail = json_axes["actual"] if json_axes["axes"]["parse_valid"] else f"parse_error: {json_axes['error']}"
    checks.append({"kind": "json", "value": expected, "actual": detail, "passed": json_axes["passed"]})
  if not checks: return {"status": "unscored", "passed": None, "checks": []}
  passed = all(check["passed"] for check in checks)
  out = {"status": "pass" if passed else "fail", "passed": passed, "checks": checks}
  if "expected_json" in prompt: out["json_axes"] = json_axes
  return out

def quality_summary(rows:list[dict[str, Any]]) -> dict[str, Any]:
  scored = [row for row in rows if row.get("score", {}).get("status") in ("pass", "fail")]
  passed = [row for row in scored if row.get("score", {}).get("passed") is True]
  tag_counts: dict[str, dict[str, int]] = {}
  for row in scored:
    tags = row.get("tags") or ["untagged"]
    for tag in tags:
      cur = tag_counts.setdefault(tag, {"scored": 0, "passed": 0})
      cur["scored"] += 1
      if row.get("score", {}).get("passed") is True: cur["passed"] += 1
  out = {
    "status": "unscored" if not scored else ("pass" if len(scored) == len(passed) else "fail"),
    "scored": len(scored),
    "passed": len(passed),
    "pass_rate": None if not scored else len(passed) / len(scored),
    "tags": {k: {"scored": v["scored"], "passed": v["passed"], "pass_rate": v["passed"] / v["scored"]} for k, v in sorted(tag_counts.items())},
  }
  json_scores = [row.get("score", {}).get("json_axes") for row in scored if isinstance(row.get("score", {}).get("json_axes"), dict)]
  if json_scores:
    out["ci95"] = wilson_interval(len(passed), len(scored))
    out["json_axes"] = summarize_json_axes(json_scores)
  return out

def md_text(text:str) -> str:
  return text.replace("\n", "\\n").replace("|", "\\|")

def build_prompt_ids(tok:Any, text:str, prompt_format:str) -> list[int]:
  if prompt_format == "chat":
    return tok.prefix() + tok.role("user") + tok.encode(text) + tok.end_turn() + tok.role("assistant")
  if prompt_format == "raw":
    return tok.prefix() + tok.encode(text)
  raise ValueError(f"unknown prompt format {prompt_format!r}")
