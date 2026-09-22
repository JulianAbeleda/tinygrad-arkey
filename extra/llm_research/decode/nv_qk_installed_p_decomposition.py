#!/usr/bin/env python3
"""Phase 3 Q/K installed-island P distribution and B/D/R decomposition.

Reads the fresh PROFILE=1 HCQ graph profile and extracts, per complete decode
replay, the 36 Q-norm and 36 K-norm production command intervals in layer
order. It then applies the Phase 0 calibrated constants to split each P into
B (exact body), D (clean chained HCQ minus body), and R (production residual).

R is reported as an unmeasured residual bucket, not assigned to cache,
serialization, or placement without counters.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics


HASH64 = re.compile(r"_[0-9a-f]{64}$")
Q_NORM = "reduce_output_rmsnorm_32_128"
K_NORM = "reduce_output_rmsnorm_8_128"
DECODE_GROUP_SIZES = (32, 64, 128, 256, 116)

# Phase 0 calibrated constants.
BODY = {"q": 1.190, "k": 1.196}
CLEAN_HCQ = 1.698


def canon(name: str) -> str:
  return HASH64.sub("", name).strip()


def percentile(xs: list[float], q: float) -> float:
  xs = sorted(xs)
  if not xs:
    return float("nan")
  idx = min(len(xs) - 1, max(0, round(q * (len(xs) - 1))))
  return xs[idx]


def dist(xs: list[float]) -> dict:
  return {
    "n": len(xs),
    "median_us": round(statistics.median(xs), 3),
    "p10_us": round(percentile(xs, 0.10), 3),
    "p90_us": round(percentile(xs, 0.90), 3),
    "min_us": round(min(xs), 3),
    "max_us": round(max(xs), 3),
  }


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--profile-jsonl", required=True)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()

  lines = [json.loads(l) for l in open(args.profile_jsonl, encoding="utf-8")]
  sizes = [len(d.get("entries", [])) for d in lines]

  replays = []
  i = 0
  while i + len(DECODE_GROUP_SIZES) <= len(lines):
    if tuple(sizes[i:i + len(DECODE_GROUP_SIZES)]) == DECODE_GROUP_SIZES:
      entries = []
      for j in range(len(DECODE_GROUP_SIZES)):
        entries.extend(lines[i + j].get("entries", []))
      replays.append(entries)
      i += len(DECODE_GROUP_SIZES)
    else:
      i += 1

  q_all: list[float] = []
  k_all: list[float] = []
  q_by_layer: list[list[float]] = [[] for _ in range(36)]
  k_by_layer: list[list[float]] = [[] for _ in range(36)]
  for entries in replays:
    q_seen = 0
    k_seen = 0
    for e in entries:
      c = canon(e["name"])
      dur = float(e["duration"])
      if c == Q_NORM:
        q_all.append(dur)
        if q_seen < 36:
          q_by_layer[q_seen].append(dur)
        q_seen += 1
      elif c == K_NORM:
        k_all.append(dur)
        if k_seen < 36:
          k_by_layer[k_seen].append(dur)
        k_seen += 1

  def decompose(p: float, body: float) -> dict:
    d = CLEAN_HCQ - body
    r = p - CLEAN_HCQ
    return {
      "P_us": round(p, 3),
      "B_us": round(body, 3),
      "D_us": round(d, 3),
      "R_us": round(r, 3),
      "identity_residual_us": round(p - body - d - r, 6),
    }

  result = {
    "schema": "tinygrad.nv_qk_installed_p_decomposition.v1",
    "constants": {"body_us": BODY, "clean_chained_hcq_us": CLEAN_HCQ},
    "replay_count": len(replays),
    "q": {
      "all": dist(q_all),
      "per_layer_median_us": [round(statistics.median(v), 3) if v else None for v in q_by_layer],
      "decomposition": decompose(statistics.median(q_all), BODY["q"]),
    },
    "k": {
      "all": dist(k_all),
      "per_layer_median_us": [round(statistics.median(v), 3) if v else None for v in k_by_layer],
      "decomposition": decompose(statistics.median(k_all), BODY["k"]),
    },
    "verdict": {
      "q_body": "BODY_PARITY",
      "k_body": "BODY_PARITY",
      "q_installation_mechanism": "UNMEASURED_RESIDUAL",
      "k_installation_mechanism": "UNMEASURED_RESIDUAL",
      "note": "D is clean front-end; R is production-conditioned residual and is not assigned to cache/serialization/placement without counters.",
    },
  }

  with open(args.out, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, sort_keys=True)
    f.write("\n")
  print(json.dumps(result["q"], indent=2))
  print(json.dumps(result["k"], indent=2))
  print(json.dumps(result["verdict"], indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
