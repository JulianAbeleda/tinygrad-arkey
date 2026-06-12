from __future__ import annotations

import json, pathlib, re
from typing import Any

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
  if not checks: return {"status": "unscored", "passed": None, "checks": []}
  passed = all(check["passed"] for check in checks)
  return {"status": "pass" if passed else "fail", "passed": passed, "checks": checks}

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
  return {
    "status": "unscored" if not scored else ("pass" if len(scored) == len(passed) else "fail"),
    "scored": len(scored),
    "passed": len(passed),
    "pass_rate": None if not scored else len(passed) / len(scored),
    "tags": {k: {"scored": v["scored"], "passed": v["passed"], "pass_rate": v["passed"] / v["scored"]} for k, v in sorted(tag_counts.items())},
  }

def md_text(text:str) -> str:
  return text.replace("\n", "\\n").replace("|", "\\|")

def build_prompt_ids(tok:Any, text:str, prompt_format:str) -> list[int]:
  if prompt_format == "chat":
    return tok.prefix() + tok.role("user") + tok.encode(text) + tok.end_turn() + tok.role("assistant")
  if prompt_format == "raw":
    return tok.prefix() + tok.encode(text)
  raise ValueError(f"unknown prompt format {prompt_format!r}")
