#!/usr/bin/env python3
"""Independent interval-identity audit over retained H1 wait-exit records."""
from __future__ import annotations

import argparse, hashlib, json, pathlib, statistics, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import llama_weighted_dag as wd
import nv_llama_useful_body_h1 as h1


def _sha(path: pathlib.Path) -> str:
  out = hashlib.sha256()
  with path.open("rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""): out.update(chunk)
  return out.hexdigest()


def _union(intervals: list[tuple[int, int]]) -> int:
  total = 0
  active: list[int] | None = None
  for start, end in sorted(intervals):
    if end <= start: continue
    if active is None: active = [start, end]
    elif start <= active[1]: active[1] = max(active[1], end)
    else: total += active[1] - active[0]; active = [start, end]
  return total + (0 if active is None else active[1] - active[0])


def _metrics(replay: list[dict], rows: list[dict], offset: int, field: str) -> dict:
  for kernel in replay:
    kernel.pop("we_lo", None); kernel.pop("we_hi", None)
  h1._assign_records(rows, replay, offset)
  spans, useful = [], []
  spin_sum = 0
  for kernel in replay:
    start, end = int(kernel["start"]), int(kernel["end"])
    boundary = max(start, min(end, int(kernel.get(field, start))))
    spans.append((start, end)); useful.append((boundary, end)); spin_sum += boundary - start
  node_sum = sum(end - start for start, end in spans)
  resident_union = _union(spans)
  useful_body = node_sum - spin_sum
  useful_union = _union(useful)
  useful_overlap = useful_body - useful_union
  spin_only_union = resident_union - useful_union
  claimed_rhs = useful_body - useful_overlap
  corrected_rhs = claimed_rhs + spin_only_union
  return {k: round(v / 1000.0, 6) for k, v in {
    "node_sum_us": node_sum, "resident_union_us": resident_union,
    "resident_overlap_us": node_sum - resident_union, "spin_sum_us": spin_sum,
    "useful_body_us": useful_body, "useful_union_us": useful_union,
    "useful_overlap_us": useful_overlap, "spin_only_union_us": spin_only_union,
    "claimed_identity_residual_us": resident_union - claimed_rhs,
    "corrected_identity_residual_us": resident_union - corrected_rhs,
    "node_minus_spin_minus_useful_residual_us": node_sum - spin_sum - useful_body,
  }.items()}


def audit(trace: pathlib.Path, prefix: pathlib.Path, graph_id: int, warmup: int) -> dict:
  replays = wd.load_replays(trace, graph_id)
  files = sorted(prefix.parent.glob(prefix.name + "-*.jsonl"),
                 key=lambda path: int(path.stem.rsplit("-", 1)[1]))[-len(replays):]
  rows = []
  for replay, ring in zip(replays[warmup:], files[warmup:]):
    parsed = h1.parse_ring(ring)
    offset, pstd = h1.calibrate(parsed, replay)
    rows.append({"ring": ring.name, "ring_sha256": _sha(ring), "offset_pstd_ns": round(pstd, 3),
                 "earliest_wait_exit": _metrics(replay, parsed, offset, "we_lo"),
                 "latest_wait_exit": _metrics(replay, parsed, offset, "we_hi")})
  means = {}
  for bound in ("earliest_wait_exit", "latest_wait_exit"):
    means[bound] = {key: round(statistics.mean(row[bound][key] for row in rows), 6) for key in rows[0][bound]}
  return {"trace": str(trace), "trace_sha256": _sha(trace), "graph_id": graph_id,
          "warmup_replays_dropped": warmup, "steady_replays": len(rows), "means": means, "per_replay": rows}


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--evidence-dir", type=pathlib.Path, required=True)
  ap.add_argument("--out", type=pathlib.Path, required=True)
  args = ap.parse_args()
  root = args.evidence_dir
  result = {
    "schema": "tinygrad.nv_useful_identity_audit.v1",
    "definition": "resident_union = useful_body - useful_overlap + spin_only_union",
    "captures": {
      "final": audit(root / "h1-final-capture.sqlite", root / "h1-final-capture", 5, 2),
      "full": audit(root / "h1-full-sampling-capture.sqlite", root / "h1-full-sampling-capture", 5, 2),
    },
  }
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({name: row["means"] for name, row in result["captures"].items()}, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
