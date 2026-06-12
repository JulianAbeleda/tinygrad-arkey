#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib, re
from collections import Counter
from typing import Any

from extra.qk_descriptor_policy import load_json, runtime_entries, write_json

GGML_FORMAT = {12: "Q4_K", 14: "Q6_K"}
FORMAT_FAMILY = {"Q4_K": "q4_k_packed_u32", "Q6_K": "q6_k_packed_u16"}
ALLOWED_PARTS = {"q4_k_packed_u32": {1, 2, 4}, "q6_k_packed_u16": {1, 2}}
ALLOWED_LOCAL = {32, 64}
LOCAL_RE = re.compile(r"^LOCAL:0:(?P<local>[0-9]+)$")


def _runtime_key(entry:dict[str, Any]) -> str:
  desc = entry.get("descriptor") or {}
  scope = entry.get("scope") or "shape"
  parts = [scope]
  if scope == "tensor": parts.append(str(desc.get("tensor") or entry.get("tensor") or ""))
  parts += [str(int(desc["ggml_type"])), str(int(desc["rows"])), str(int(desc["cols"]))]
  return "|".join(parts)


def _check_entry(entry:dict[str, Any], index:int) -> list[str]:
  reasons: list[str] = []
  desc, cand = entry.get("descriptor") or {}, entry.get("candidate") or {}
  prefix = f"entry[{index}]"
  try:
    ggml_type, rows, cols = int(desc["ggml_type"]), int(desc["rows"]), int(desc["cols"])
  except (KeyError, TypeError, ValueError) as exc:
    return [f"{prefix}: descriptor must contain integer ggml_type/rows/cols ({exc})"]
  if rows <= 0 or cols <= 0: reasons.append(f"{prefix}: rows/cols must be positive, got {rows}x{cols}")
  fmt = entry.get("format") or desc.get("format")
  if GGML_FORMAT.get(ggml_type) != fmt:
    reasons.append(f"{prefix}: format {fmt!r} does not match ggml_type={ggml_type}")
  winner, family = entry.get("winner"), cand.get("family")
  parts, opts = int(cand.get("parts", 0)), list(cand.get("opts") or [])
  if winner == "fused_graph":
    if family != "fused_graph": reasons.append(f"{prefix}: fused_graph winner must use fused_graph family, got {family!r}")
    if parts != 0: reasons.append(f"{prefix}: fused_graph parts must be 0, got {parts}")
    if opts: reasons.append(f"{prefix}: fused_graph opts must be empty, got {opts}")
    return reasons
  expected_family = FORMAT_FAMILY.get(fmt)
  if family != expected_family:
    reasons.append(f"{prefix}: unsupported family {family!r} for {fmt}; expected {expected_family!r}")
    return reasons
  if parts not in ALLOWED_PARTS[family]:
    reasons.append(f"{prefix}: unsupported parts={parts} for {family}")
  if len(opts) != 1:
    reasons.append(f"{prefix}: primitive candidate must have exactly one LOCAL opt, got {opts}")
  else:
    match = LOCAL_RE.match(str(opts[0]))
    if match is None:
      reasons.append(f"{prefix}: malformed LOCAL opt {opts[0]!r}")
    elif int(match.group("local")) not in ALLOWED_LOCAL:
      reasons.append(f"{prefix}: unsupported LOCAL arg {match.group('local')}")
  return reasons


def gate_policy(policy:dict[str, Any]) -> tuple[bool, list[str]]:
  reasons: list[str] = []
  if policy.get("kind") != "qk_generated_policy":
    reasons.append("policy kind must be qk_generated_policy")
    return False, reasons
  entries = policy.get("entries")
  if not isinstance(entries, list) or not entries:
    reasons.append("policy must contain non-empty entries")
    return False, reasons
  keys = []
  for idx, entry in enumerate(entries):
    reasons += _check_entry(entry, idx)
    try:
      keys.append(_runtime_key(entry))
    except (KeyError, TypeError, ValueError) as exc:
      reasons.append(f"entry[{idx}]: invalid runtime key: {exc}")
  dupes = sorted(k for k, count in Counter(keys).items() if count > 1)
  if dupes: reasons.append(f"duplicate runtime keys: {dupes}")
  try:
    runtime_entries(policy)
  except (KeyError, TypeError, ValueError) as exc:
    reasons.append(f"runtime policy normalization failed: {exc}")
  return not reasons, reasons


def build_static_gate(candidate_set:dict[str, Any]) -> dict[str, Any]:
  if candidate_set.get("kind") != "qk_candidate_set":
    raise ValueError("expected kind=qk_candidate_set")
  rows = []
  for cand in candidate_set.get("candidates", []):
    passed, reasons = gate_policy(cand.get("policy") or {})
    rows.append({
      "id": cand.get("id"),
      "status": "pass" if passed else "fail",
      "changes": cand.get("changes", []),
      "expected_storage_bytes": cand.get("expected_storage_bytes"),
      "reasons": reasons,
    })
  passing = [row for row in rows if row["status"] == "pass"]
  failing = [row for row in rows if row["status"] != "pass"]
  return {
    "kind": "qk_candidate_static_gate",
    "model": candidate_set.get("model"),
    "model_size": candidate_set.get("model_size"),
    "source_candidates": candidate_set.get("source_descriptor"),
    "rows": rows,
    "summary": {"candidates": len(rows), "passing": len(passing), "failing": len(failing)},
  }


def static_gate_markdown(report:dict[str, Any]) -> str:
  lines = [
    f"# QK Static Gate: {report['model_size']}",
    "",
    "Fail-closed validation for generated QK policy candidates before any GPU run.",
    "",
    "## Summary",
    "",
    f"- candidates: `{report['summary']['candidates']}`",
    f"- passing: `{report['summary']['passing']}`",
    f"- failing: `{report['summary']['failing']}`",
    "",
    "| id | status | changes | storage bytes | reasons |",
    "|---|---|---:|---:|---|",
  ]
  for row in report["rows"]:
    reasons = "; ".join(row["reasons"]) or "none"
    lines.append(f"| `{row['id']}` | `{row['status']}` | {len(row['changes'])} | {row['expected_storage_bytes']} | {reasons} |")
  lines.append("")
  return "\n".join(lines)


def main() -> int:
  parser = argparse.ArgumentParser(description="Static safety gate for QK candidate policies")
  parser.add_argument("--candidates", type=pathlib.Path, required=True)
  parser.add_argument("--json", type=pathlib.Path, required=True)
  parser.add_argument("--md", type=pathlib.Path)
  parser.add_argument("--fail-on-reject", action="store_true")
  args = parser.parse_args()
  report = build_static_gate(load_json(args.candidates.expanduser()))
  write_json(args.json, report)
  if args.md:
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text(static_gate_markdown(report))
  else:
    print(static_gate_markdown(report))
  if args.fail_on_reject and report["summary"]["failing"]:
    raise SystemExit(f"{args.candidates}: {report['summary']['failing']} candidates failed static gate")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
