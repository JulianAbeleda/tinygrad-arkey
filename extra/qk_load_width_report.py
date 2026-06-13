#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib, re
from typing import Any

KERNEL_RE = re.compile(r"\b(q4k_[a-zA-Z0-9_]*?_\d+_\d+_\d+|q6k_[a-zA-Z0-9_]*?_\d+_\d+_\d+|qk_probe_[a-zA-Z0-9_]*?_\d+)\b")
LOAD_PATTERNS = {
  "uchar_or_u8": re.compile(r"\b(?:uchar|uint8_t|unsigned char|u8)\b"),
  "ushort_or_u16": re.compile(r"\b(?:ushort|uint16_t|unsigned short|u16)\b"),
  "uint_or_u32": re.compile(r"\b(?:uint|uint32_t|unsigned int|u32)\b"),
  "ulong_or_u64": re.compile(r"\b(?:ulong|uint64_t|unsigned long|u64)\b"),
  "vector_u32x2": re.compile(r"\b(?:uint2|u32x2)\b"),
  "vector_u32x4": re.compile(r"\b(?:uint4|tg_uint4|unsigned_int4)\b"),
  "amd_vdot4": re.compile(r"\b(?:v_dot4|sudot4|amdgcn_sudot4)\b"),
}
MODE_PATTERNS = {
  "vector_load": re.compile(r"q4k_gemv_vector_load_partial"),
  "packed_load": re.compile(r"q4k_gemv_packed_load_partial"),
  "baseline_partial": re.compile(r"q4k_gemv_partial"),
  "grouped_partial": re.compile(r"q4k_gemv_grouped_partial"),
  "tile_custom_partial": re.compile(r"q4k_gemv_tile_custom_partial"),
  "q6_partial": re.compile(r"q6k_gemv_partial"),
  "uop_vec_request": re.compile(r"qk_probe_uop_vec_request"),
  "custom_uint4": re.compile(r"qk_probe_custom_uint4"),
  "packed_tile_custom_q4_dot": re.compile(r"qk_probe_tile_custom_q4_dot"),
  "scalar_probe": re.compile(r"qk_probe_scalar"),
}


def _portable(path:pathlib.Path, repo:pathlib.Path) -> str:
  resolved = str(repo.resolve())
  text = str(path)
  if text == resolved: return "."
  return text.replace(resolved + "/", "")


def _mode_for_text(text:str) -> str:
  for mode, pattern in MODE_PATTERNS.items():
    if pattern.search(text): return mode
  return "unknown"


def _generated_source_text(text:str) -> str:
  blocks, cur = [], []
  in_source = False
  for line in text.splitlines():
    if line.startswith("typedef long unsigned int size_t;"):
      if cur: blocks.append("\n".join(cur))
      cur, in_source = [line], True
      continue
    if in_source and line.startswith("*** "):
      blocks.append("\n".join(cur))
      cur, in_source = [], False
      continue
    if in_source: cur.append(line)
  if cur: blocks.append("\n".join(cur))
  return "\n".join(blocks) if blocks else text


def analyze_log(path:pathlib.Path, *, repo:pathlib.Path=pathlib.Path.cwd()) -> dict[str, Any]:
  text = path.read_text(errors="replace")
  source_text = _generated_source_text(text)
  kernels = sorted(set(KERNEL_RE.findall(text)))
  pattern_counts = {name: len(pattern.findall(source_text)) for name, pattern in LOAD_PATTERNS.items()}
  load_width = "unknown"
  if pattern_counts["vector_u32x4"]: load_width = "vector_u32x4"
  elif pattern_counts["vector_u32x2"]: load_width = "vector_u32x2"
  elif pattern_counts["uint_or_u32"]: load_width = "u32_scalar"
  elif pattern_counts["ushort_or_u16"]: load_width = "u16_scalar"
  elif pattern_counts["uchar_or_u8"]: load_width = "u8_scalar"
  return {
    "path": _portable(path, repo),
    "mode": _mode_for_text(text),
    "kernels": kernels,
    "load_width_inferred": load_width,
    "pattern_counts": pattern_counts,
    "contains_packed_dot": bool(pattern_counts["amd_vdot4"]),
    "notes": [
      "This parser infers load width from generated-source text patterns; it is not a hardware-counter report.",
      "Use it to confirm whether a candidate changed the generated source shape before trusting timing.",
    ],
  }


def build_report(logs:list[pathlib.Path], *, repo:pathlib.Path=pathlib.Path.cwd()) -> dict[str, Any]:
  rows = [analyze_log(path, repo=repo) for path in logs]
  changed_modes = sorted(set(row["mode"] for row in rows))
  return {
    "kind": "qk_load_width_report",
    "schema_version": 1,
    "rows": rows,
    "summary": {
      "logs": len(rows),
      "modes": changed_modes,
      "has_vector_load_evidence": any(str(row["load_width_inferred"]).startswith("vector_") for row in rows),
      "has_packed_load_kernel": any(row["mode"] == "packed_load" for row in rows),
      "has_packed_dot": any(row["contains_packed_dot"] for row in rows),
    },
  }


def report_markdown(report:dict[str, Any]) -> str:
  lines = [
    "# QK Load Width Report",
    "",
    "Generated-source parser for QK load-width evidence. This is a source-shape",
    "check, not a hardware-counter measurement.",
    "",
    "## Summary",
    "",
    f"- logs: `{report['summary']['logs']}`",
    f"- modes: `{', '.join(report['summary']['modes'])}`",
    f"- vector load evidence: `{report['summary']['has_vector_load_evidence']}`",
    f"- packed-load kernel present: `{report['summary']['has_packed_load_kernel']}`",
    f"- packed-dot present: `{report['summary']['has_packed_dot']}`",
    "",
    "| log | mode | inferred load width | kernels | packed dot |",
    "|---|---|---|---|---:|",
  ]
  for row in report["rows"]:
    kernels = ", ".join(row["kernels"][:4])
    if len(row["kernels"]) > 4: kernels += f", ...+{len(row['kernels'])-4}"
    lines.append(f"| `{row['path']}` | `{row['mode']}` | `{row['load_width_inferred']}` | `{kernels}` | `{row['contains_packed_dot']}` |")
  lines.append("")
  return "\n".join(lines)


def main() -> int:
  parser = argparse.ArgumentParser(description="Parse DEBUG=4 QK generated-source logs for load-width evidence")
  parser.add_argument("logs", nargs="+", type=pathlib.Path)
  parser.add_argument("--json", type=pathlib.Path, required=True)
  parser.add_argument("--md", type=pathlib.Path, required=True)
  parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path.cwd())
  args = parser.parse_args()
  report = build_report([path.expanduser() for path in args.logs], repo=args.repo.resolve())
  args.json.parent.mkdir(parents=True, exist_ok=True)
  args.json.write_text(json.dumps(report, indent=2, sort_keys=True))
  args.md.parent.mkdir(parents=True, exist_ok=True)
  args.md.write_text(report_markdown(report))
  print(report_markdown(report))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
