#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib, pickle, re, struct
from collections import defaultdict

def _u64_values(blob:bytes, off:int, size:int) -> list[int]:
  if size % 8 != 0: raise ValueError(f"PMC sample size is not u64-aligned: off={off} size={size}")
  return list(struct.unpack("<" + "Q" * (size // 8), blob[off:off+size]))

def _summarize_event(event) -> dict:
  counters = {}
  for sample in event.sched:
    vals = _u64_values(event.blob, sample.off, sample.size)
    counters[sample.name] = {
      "sum": int(sum(vals)),
      "max": int(max(vals)) if vals else 0,
      "nonzero": int(sum(1 for v in vals if v)),
      "samples": len(vals),
    }
  hit = counters.get("GL2C_HIT", {}).get("sum", 0)
  miss = counters.get("GL2C_MISS", {}).get("sum", 0)
  busy = counters.get("SQ_BUSY_CYCLES", {}).get("sum", 0)
  valu = counters.get("SQ_INSTS_VALU", {}).get("sum", 0)
  salu = counters.get("SQ_INSTS_SALU", {}).get("sum", 0)
  return {
    "kern": event.kern,
    "exec_tag": event.exec_tag,
    "counters": counters,
    "derived": {
      "gl2c_hit_rate": None if hit + miss == 0 else hit / (hit + miss),
      "valu_per_busy_cycle": None if busy == 0 else valu / busy,
      "salu_per_busy_cycle": None if busy == 0 else salu / busy,
    },
  }

def parse_profile(path:pathlib.Path, kernel_patterns:list[str]) -> dict:
  with path.open("rb") as f:
    events = pickle.load(f)
  programs = {e.tag: e.name for e in events if type(e).__name__ == "ProfileProgramEvent"}
  pmc_events = [e for e in events if type(e).__name__ == "ProfilePMCEvent"]
  if not pmc_events:
    raise ValueError(f"{path} contains no ProfilePMCEvent records; rerun with PROFILE=1 PMC=1")
  patterns = [re.compile(p) for p in kernel_patterns]
  rows, aggregate = [], defaultdict(lambda: defaultdict(int))
  for event in pmc_events:
    name = programs.get(event.kern, f"<unknown:{event.kern}>")
    if patterns and not any(p.search(name) for p in patterns): continue
    row = _summarize_event(event)
    row["name"] = name
    rows.append(row)
    for cname, cdata in row["counters"].items():
      aggregate[name][cname] += cdata["sum"]
  if kernel_patterns and not rows:
    raise ValueError(f"{path} has PMC records, but none matched {kernel_patterns!r}")
  by_kernel = {}
  for name, counters in aggregate.items():
    hit, miss = counters.get("GL2C_HIT", 0), counters.get("GL2C_MISS", 0)
    busy, valu, salu = counters.get("SQ_BUSY_CYCLES", 0), counters.get("SQ_INSTS_VALU", 0), counters.get("SQ_INSTS_SALU", 0)
    by_kernel[name] = {
      "events": sum(1 for row in rows if row["name"] == name),
      "counters": dict(counters),
      "derived": {
        "gl2c_hit_rate": None if hit + miss == 0 else hit / (hit + miss),
        "valu_per_busy_cycle": None if busy == 0 else valu / busy,
        "salu_per_busy_cycle": None if busy == 0 else salu / busy,
      },
    }
  return {
    "profile": str(path),
    "program_count": len(programs),
    "pmc_event_count": len(pmc_events),
    "matched_event_count": len(rows),
    "kernels": by_kernel,
    "events": rows,
  }

def write_markdown(report:dict, path:pathlib.Path) -> None:
  lines = [
    "# QK PMC Profile",
    "",
    f"Profile: `{report['profile']}`",
    "",
    f"Programs: `{report['program_count']}`; PMC events: `{report['pmc_event_count']}`; matched events: `{report['matched_event_count']}`.",
    "",
    "| kernel | events | GL2 hit rate | VALU / busy | SALU / busy | SQ busy | VALU inst | GL2 hit | GL2 miss |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
  ]
  for name, row in sorted(report["kernels"].items()):
    counters, derived = row["counters"], row["derived"]
    def fmt(x):
      return "n/a" if x is None else f"{x:.4f}"
    lines.append(
      f"| `{name}` | {row['events']} | {fmt(derived['gl2c_hit_rate'])} | {fmt(derived['valu_per_busy_cycle'])} | "
      f"{fmt(derived['salu_per_busy_cycle'])} | {counters.get('SQ_BUSY_CYCLES', 0)} | "
      f"{counters.get('SQ_INSTS_VALU', 0)} | {counters.get('GL2C_HIT', 0)} | {counters.get('GL2C_MISS', 0)} |"
    )
  lines += [
    "",
    "Interpretation notes:",
    "",
    "- These are tinygrad AMD PMC aggregates, not normalized hardware occupancy percentages.",
    "- Use them to compare candidate kernels within the same run/profile.",
    "- A low GL2 hit rate or low VALU-per-busy-cycle is a schedule/layout signal, not proof that one instruction is missing.",
  ]
  path.write_text("\n".join(lines) + "\n")

def main() -> None:
  parser = argparse.ArgumentParser(description="Summarize tinygrad AMD PMC profile counters for QK kernels")
  parser.add_argument("--profile", type=pathlib.Path, default=pathlib.Path("/tmp/profile.pkl.ubuntu"))
  parser.add_argument("--kernel", action="append", default=[], help="regex kernel name filter; may be repeated")
  parser.add_argument("--json", type=pathlib.Path)
  parser.add_argument("--md", type=pathlib.Path)
  args = parser.parse_args()
  report = parse_profile(args.profile, args.kernel)
  if args.json: args.json.write_text(json.dumps(report, indent=2, sort_keys=True))
  if args.md: write_markdown(report, args.md)
  if not args.json and not args.md: print(json.dumps(report, indent=2, sort_keys=True))

if __name__ == "__main__":
  main()
