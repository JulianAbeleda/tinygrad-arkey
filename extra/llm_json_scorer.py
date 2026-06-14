from __future__ import annotations

import json, math
from typing import Any

JSON_AXES = ("parse_valid", "no_extra_text", "schema_ok", "type_ok", "value_correct", "strict_pass")
WILSON_Z_95 = 1.959963984540054

def wilson_interval(passed:int, total:int, *, z:float=WILSON_Z_95) -> dict[str, Any]:
  if not isinstance(passed, int) or not isinstance(total, int):
    raise TypeError("passed and total must be integers")
  if total < 0 or passed < 0 or passed > total:
    raise ValueError(f"invalid Wilson counts passed={passed} total={total}")
  if total == 0: return {"low": None, "high": None, "confidence": 0.95}
  phat = passed / total
  z2 = z * z
  denom = 1.0 + z2 / total
  center = (phat + z2 / (2.0 * total)) / denom
  margin = z * math.sqrt((phat * (1.0 - phat) + z2 / (4.0 * total)) / total) / denom
  return {"low": max(0.0, center - margin), "high": min(1.0, center + margin), "confidence": 0.95}

def _type_name(value:Any) -> str:
  if value is None: return "null"
  if isinstance(value, bool): return "bool"
  if isinstance(value, int): return "int"
  if isinstance(value, float): return "float"
  if isinstance(value, str): return "str"
  if isinstance(value, list): return "list"
  if isinstance(value, dict): return "dict"
  return type(value).__name__

def _type_ok(actual:Any, expected:Any) -> bool:
  if isinstance(expected, bool): return isinstance(actual, bool)
  if isinstance(expected, int) and not isinstance(expected, bool): return isinstance(actual, int) and not isinstance(actual, bool)
  if isinstance(expected, float): return (isinstance(actual, (int, float)) and not isinstance(actual, bool))
  if isinstance(expected, str): return isinstance(actual, str)
  if expected is None: return actual is None
  return type(actual) is type(expected)

def _normalize_value(value:Any, *, case_insensitive:bool=False) -> Any:
  if isinstance(value, str):
    out = value.strip()
    return out.casefold() if case_insensitive else out
  if isinstance(value, list):
    return [_normalize_value(x, case_insensitive=case_insensitive) for x in value]
  if isinstance(value, dict):
    return {k: _normalize_value(v, case_insensitive=case_insensitive) for k, v in value.items()}
  return value

def _parse_json_prefix(text:str) -> tuple[bool, Any, int, str | None, list[str]]:
  duplicates: list[str] = []
  def no_duplicate_object(pairs:list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
      if key in out: duplicates.append(key)
      out[key] = value
    return out
  raw = text.strip()
  if not raw: return False, None, 0, "empty output", duplicates
  decoder = json.JSONDecoder(object_pairs_hook=no_duplicate_object)
  try:
    parsed, idx = decoder.raw_decode(raw)
  except json.JSONDecodeError as exc:
    return False, None, 0, exc.msg, duplicates
  return True, parsed, idx, None, duplicates

def score_expected_json(text:str, expected:dict[str, Any], *, case_insensitive:bool=False) -> dict[str, Any]:
  if not isinstance(expected, dict) or not expected:
    raise ValueError("expected must be a non-empty dict")
  parse_valid, parsed, idx, error, duplicates = _parse_json_prefix(text)
  stripped = text.strip()
  no_extra_text = parse_valid and stripped[idx:].strip() == ""
  expected_keys = set(expected)
  actual_keys = set(parsed) if isinstance(parsed, dict) else set()
  schema_ok = parse_valid and isinstance(parsed, dict) and actual_keys == expected_keys and not duplicates
  type_ok = schema_ok and all(_type_ok(parsed[key], expected[key]) for key in expected)
  value_correct = type_ok and all(
    _normalize_value(parsed[key], case_insensitive=case_insensitive) ==
    _normalize_value(expected[key], case_insensitive=case_insensitive)
    for key in expected
  )
  axes = {
    "parse_valid": bool(parse_valid),
    "no_extra_text": bool(no_extra_text),
    "schema_ok": bool(schema_ok),
    "type_ok": bool(type_ok),
    "value_correct": bool(value_correct),
  }
  axes["strict_pass"] = all(axes.values())
  return {
    "kind": "json_axes",
    "passed": axes["strict_pass"],
    "axes": axes,
    "expected": expected,
    "actual": parsed if parse_valid else None,
    "error": error,
    "details": {
      "actual_type": _type_name(parsed) if parse_valid else None,
      "actual_keys": sorted(actual_keys),
      "expected_keys": sorted(expected_keys),
      "duplicate_keys": sorted(set(duplicates)),
      "case_insensitive": case_insensitive,
    },
  }

def summarize_json_axes(json_scores:list[dict[str, Any]]) -> dict[str, Any]:
  scored = [score for score in json_scores if isinstance(score, dict) and isinstance(score.get("axes"), dict)]
  axis_rows: dict[str, dict[str, Any]] = {}
  for axis in JSON_AXES:
    passed = sum(1 for score in scored if score["axes"].get(axis) is True)
    total = len(scored)
    axis_rows[axis] = {
      "passed": passed,
      "scored": total,
      "pass_rate": None if total == 0 else passed / total,
      "ci95": wilson_interval(passed, total),
    }
  return {"kind": "json_axis_summary", "scored": len(scored), "axes": axis_rows}
