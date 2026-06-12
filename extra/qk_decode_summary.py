#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib, re, statistics

TOK_RE = re.compile(r"(?P<ms>[0-9]+\.[0-9]+) ms,\s+(?P<tps>[0-9]+\.[0-9]+) tok/s,\s+(?P<gbs>[0-9]+\.[0-9]+) GB/s")
POLICY_RE = re.compile(r"QK_GENERATED_POLICY_DEBUG loaded=(?P<path>\S+) entries=(?P<entries>\d+)")
INSTALL_RE = re.compile(r"(?P<kind>Q[46]K)_PRIMITIVE_DEBUG installed=(?P<installed>\d+) skipped_total=(?P<skipped>\d+)(?P<rest>.*)")

def _mean(xs:list[float]) -> float|None:
  return statistics.mean(xs) if xs else None

def _stdev(xs:list[float]) -> float|None:
  return statistics.stdev(xs) if len(xs) >= 2 else None

def _fmt(x) -> str:
  if x is None: return "n/a"
  if isinstance(x, float): return f"{x:.2f}"
  return str(x)

def _parse_counts(rest:str) -> dict[str, int]:
  out = {}
  for part in rest.split():
    if "=" not in part: continue
    k, v = part.split("=", 1)
    try: out[k] = int(v)
    except ValueError: pass
  return out

def parse_log(label:str, path:pathlib.Path) -> dict:
  text = path.read_text(errors="replace")
  toks = [float(m.group("tps")) for m in TOK_RE.finditer(text)]
  ms = [float(m.group("ms")) for m in TOK_RE.finditer(text)]
  if not toks: raise ValueError(f"{path}: no benchmark token summaries found")
  policy = [m.groupdict() for m in POLICY_RE.finditer(text)]
  installs = {}
  for m in INSTALL_RE.finditer(text):
    installs[m.group("kind")] = {
      "installed": int(m.group("installed")),
      "skipped_total": int(m.group("skipped")),
      **_parse_counts(m.group("rest")),
    }
  return {
    "label": label,
    "path": str(path),
    "samples": len(toks),
    "avg_tok_s": _mean(toks),
    "median_tok_s": statistics.median(toks),
    "stdev_tok_s": _stdev(toks),
    "avg_drop1_tok_s": _mean(toks[1:]),
    "avg_last64_tok_s": _mean(toks[-64:]),
    "avg_last32_tok_s": _mean(toks[-32:]),
    "avg_last16_tok_s": _mean(toks[-16:]),
    "min_tok_s": min(toks),
    "max_tok_s": max(toks),
    "avg_ms": _mean(ms),
    "max_ms": max(ms),
    "policy": policy[-1] if policy else None,
    "installs": installs,
  }

def _md(rows:list[dict]) -> str:
  lines = [
    "# QK Decode Summary",
    "",
    "| label | samples | avg tok/s | drop1 | last64 | last32 | last16 | stdev | min | max | Q4 install | Q6 install | policy |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
  ]
  for row in rows:
    q4, q6 = row["installs"].get("Q4K", {}), row["installs"].get("Q6K", {})
    policy = row["policy"]["path"] if row["policy"] else ""
    lines.append(
      f"| `{row['label']}` | {row['samples']} | {_fmt(row['avg_tok_s'])} | {_fmt(row['avg_drop1_tok_s'])} | "
      f"{_fmt(row['avg_last64_tok_s'])} | {_fmt(row['avg_last32_tok_s'])} | {_fmt(row['avg_last16_tok_s'])} | "
      f"{_fmt(row['stdev_tok_s'])} | {_fmt(row['min_tok_s'])} | {_fmt(row['max_tok_s'])} | "
      f"{q4.get('installed', '')} | {q6.get('installed', '')} | `{policy}` |"
    )
  return "\n".join(lines) + "\n"

def main() -> None:
  parser = argparse.ArgumentParser(description="Summarize tinygrad LLM --benchmark decode logs")
  parser.add_argument("logs", nargs="+", help="PATH or LABEL=PATH")
  parser.add_argument("--json", type=pathlib.Path)
  parser.add_argument("--md", type=pathlib.Path)
  args = parser.parse_args()

  rows = []
  for item in args.logs:
    if "=" in item:
      label, raw_path = item.split("=", 1)
      path = pathlib.Path(raw_path)
    else:
      path = pathlib.Path(item)
      label = path.stem
    rows.append(parse_log(label, path))
  if args.json:
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(rows, indent=2, sort_keys=True))
  if args.md:
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text(_md(rows))
  if not args.json and not args.md:
    print(_md(rows))

if __name__ == "__main__":
  main()
