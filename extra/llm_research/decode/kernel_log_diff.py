#!/usr/bin/env python3
"""Parse and diff DEBUG=2 per-kernel timing logs (the section 2.4 measurement basis).

Aggregates launches by kernel identity (name with its trailing content hash stripped) so the
same logical kernel is comparable across trees even when its source hash changes. This is the
promoted form of the diagnostic parser used to validate the 90e93875c devectorizer regression
and every before/after in the target-capability-policy-decoupling scope.

Bandwidth columns (`GFLOPS  achieved|incl_cache GB/s`): the FIRST GB/s column is achieved
traffic on packed bytes -- validated against lm_head (510,504,960 bytes / 63.64 ms = 8.02 GB/s,
matching the reported 8.0). The SECOND column includes cache hits and must not be read as DRAM
bandwidth.
"""
from __future__ import annotations
import argparse, re, sys, pathlib
from collections import defaultdict, namedtuple

ANSI = re.compile(r"\x1b\[[0-9;]*m")
HASH = re.compile(r"_[0-9a-f]{64}\b")
LAUNCH = re.compile(r"\*\*\* \w+\s+(\d+)\s+(.*?)\s+arg\s+\d+\s+mem\s+([\d.]+) GB\s+"
                   r"tm\s+([\d.]+)(us|ms)/.*?\(\s*(\d+) GFLOPS\s+([\d.]+)\|([\d.]+)\s+GB/s\)")

KernelAgg = namedtuple("KernelAgg", "count us achieved_gbs cache_gbs")

def strip_ansi(line:str) -> str: return ANSI.sub("", line)
def canonical_name(name:str) -> str: return HASH.sub("", name).strip()

def iter_launches(text:str):
  """Yield (name, us, achieved_gbs, cache_gbs) for every non-copy kernel launch line."""
  for raw in text.splitlines():
    m = LAUNCH.search(strip_ansi(raw))
    if not m: continue
    name = m.group(2)
    if name.startswith("copy"): continue          # host<->device transfers, not compute
    us = float(m.group(4)) * (1000.0 if m.group(5) == "ms" else 1.0)
    yield name, us, float(m.group(7)), float(m.group(8))

def last_launches(text:str, n:int) -> str:
  """Return the raw text of the last n kernel-launch lines (e.g. one decode token; see the
  graph-admission census in extra/llm_research/decode/decode_runtime_overhead.py)."""
  lines = [raw for raw in text.splitlines() if LAUNCH.search(strip_ansi(raw))]
  return "\n".join(lines[-n:])

def parse_text(text:str) -> tuple[dict[str, KernelAgg], float]:
  """Aggregate a log's text by canonical kernel identity. Returns (per-kernel stats, total us)."""
  agg, total = defaultdict(lambda: [0, 0.0, 0.0, 0.0]), 0.0
  for name, us, gbs, cache_gbs in iter_launches(text):
    row = agg[canonical_name(name)]
    row[0] += 1; row[1] += us; row[2] = max(row[2], gbs); row[3] = max(row[3], cache_gbs)
    total += us
  return {k: KernelAgg(*v) for k, v in agg.items()}, total

def parse(path) -> tuple[dict[str, KernelAgg], float]:
  return parse_text(pathlib.Path(path).read_text(errors="replace"))

def diff(post_path, pre_path):
  """Diff two logs by canonical kernel identity, most-regressed (post - pre) first."""
  post, post_total = parse(post_path)
  pre, pre_total = parse(pre_path)
  zero = KernelAgg(0, 0.0, 0.0, 0.0)
  rows = [(post.get(k, zero).us - pre.get(k, zero).us, k, post.get(k, zero), pre.get(k, zero))
          for k in set(post) | set(pre)]
  rows.sort(reverse=True)
  return rows, post_total, pre_total, post, pre

def _print_summary(agg, total):
  print(f"total kernel time: {total/1000:.1f} ms across {sum(a.count for a in agg.values())} launches "
        f"({len(agg)} distinct kernels)\n")
  print(f"{'ms':>9} {'n':>5} {'achieved_GB/s':>13} {'incl_cache_GB/s':>15}  kernel")
  for key, a in sorted(agg.items(), key=lambda kv: -kv[1].us):
    print(f"{a.us/1000:>9.3f} {a.count:>5} {a.achieved_gbs:>13.1f} {a.cache_gbs:>15.1f}  {key[:64]}")

def _print_diff(rows, post_total, pre_total, post, pre, top:int):
  print(f"POST total kernel time: {post_total/1000:>10.1f} ms across {sum(v.count for v in post.values()):>5} launches")
  print(f"PRE  total kernel time: {pre_total/1000:>10.1f} ms across {sum(v.count for v in pre.values()):>5} launches")
  print(f"ratio post/pre: {post_total/pre_total:.2f}x\n")
  print(f"{'delta_ms':>9} {'post_ms':>9} {'pre_ms':>9} {'post_n':>7} {'pre_n':>6} {'post_GB/s':>10} {'pre_GB/s':>9}  kernel")
  for d, k, p, r in rows[:top] if top else rows:
    print(f"{d/1000:>9.1f} {p.us/1000:>9.1f} {r.us/1000:>9.1f} {p.count:>7} {r.count:>6} "
          f"{p.achieved_gbs:>10.1f} {r.achieved_gbs:>9.1f}  {k[:64]}")
  print("\nkernels only in POST:", sum(1 for k in post if k not in pre))
  print("kernels only in PRE: ", sum(1 for k in pre if k not in post))

def main(argv:list[str] | None=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("log", help="DEBUG=2 kernel log to summarize; the post-arm log when --pre is given")
  ap.add_argument("--pre", metavar="PRE_LOG", default=None,
                  help="pre-arm log; prints a delta-sorted diff against `log` (the post-arm) instead of a summary")
  ap.add_argument("--top", type=int, default=18, help="rows to print in diff mode, 0 for all (default 18)")
  args = ap.parse_args(argv)
  if args.pre is None:
    agg, total = parse(args.log)
    _print_summary(agg, total)
  else:
    _print_diff(*diff(args.log, args.pre), top=args.top)
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
