#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib

def _best(results:list[dict], prefix:str|None=None) -> dict|None:
  rows = [r for r in results if r.get("status") == "pass" and r.get("quant_gbs") is not None]
  if prefix is not None: rows = [r for r in rows if r.get("candidate", "").startswith(prefix)]
  return max(rows, key=lambda r: r["quant_gbs"]) if rows else None

def _fmt(x) -> str:
  if x is None: return "n/a"
  if isinstance(x, float): return f"{x:.2f}"
  return str(x)

def report_markdown(reports:list[dict], title:str) -> str:
  lines = [
    f"# {title}",
    "",
    "| model | tensor | format | shape | research winner | runtime policy | winner GB/s | best q8 GB/s | stopped vdot | verdict |",
    "|---|---|---|---:|---|---|---:|---:|---:|---|",
  ]
  any_stopped = False
  any_q8_win = False
  for report in reports:
    model = pathlib.Path(report["model"]).name
    for desc_report in report["descriptors"]:
      desc = desc_report["descriptor"]
      results = desc_report["results"]
      winner = desc_report["winner"]
      policy_winner = desc_report.get("policy_winner", winner)
      best_q8 = _best(results, "q8_1_")
      stopped = [r for r in results if r.get("status") == "skipped-stop"]
      any_stopped = any_stopped or bool(stopped)
      any_q8_win = any_q8_win or (best_q8 is not None and best_q8.get("candidate") == winner.get("winner"))
      verdict = "q8 research win; not runtime-supported" if best_q8 is not None and best_q8.get("candidate") == winner.get("winner") else "keep current/fused"
      lines.append(
        f"| `{model}` | `{desc['tensor']}` | {desc['format']} | {desc['rows']}x{desc['cols']} | "
        f"`{winner.get('winner')}` | `{policy_winner.get('winner')}` | {_fmt(winner.get('metric_value'))} | "
        f"{_fmt(None if best_q8 is None else best_q8.get('quant_gbs'))} | {len(stopped)} | {verdict} |"
      )
  lines += [
    "",
    "## Stop-Gate Result",
    "",
  ]
  if any_stopped:
    lines.append("The semantic stop gate fired for isolated packed-dot candidates. They remain available for explicit experiments, but are not default generated-search work.")
  else:
    lines.append("No candidate hit a semantic stop gate.")
  lines.append("")
  if any_q8_win:
    lines.append("At least one q8_1 candidate won a descriptor as a research result. It is intentionally not emitted as the runtime policy because no q8_1 wrapper/full-decode gate exists.")
  else:
    lines.append("No q8_1 candidate won the representative descriptors. The generated search therefore stops at the current v1/fused policy.")
  lines += [
    "",
    "## Interpretation",
    "",
    "This is the machine-readable version of the current hypothesis: packed dot is not rejected as a hardware capability, but isolated packed-dot work is rejected as the next default task. A future candidate has to be a broader semantic layout/schedule/codegen package and must beat these generated gates before touching runtime policy.",
  ]
  return "\n".join(lines) + "\n"

def main() -> None:
  parser = argparse.ArgumentParser(description="Summarize QK semantic generated-search reports")
  parser.add_argument("json", nargs="+", type=pathlib.Path)
  parser.add_argument("--title", default="QK Semantic Generated-Search Report")
  parser.add_argument("--md", type=pathlib.Path, required=True)
  args = parser.parse_args()
  reports = [json.loads(p.read_text()) for p in args.json]
  args.md.write_text(report_markdown(reports, args.title))

if __name__ == "__main__":
  main()
