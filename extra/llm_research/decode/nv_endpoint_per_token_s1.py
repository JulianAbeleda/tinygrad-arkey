#!/usr/bin/env python3
"""Per-token S1 extractor and tightened-bracket verdict.

The Stage 4 endpoint measured S1 for a single final token per fresh process.
The retained ``*.profile.jsonl`` files already contain every decode-token
timeline (each token is one 5-group window), so this tool recomputes S1 for
every captured token and turns the bracket into a per-token distribution.

It also drops the first window of each process: that window carries the
prefill/decode transition and graph-construction host gap (~200+ ms) and is
not a steady decode token.  ``--settle`` drops that many leading windows.

This is measurement tooling only.  It reads retained evidence, writes a new
verdict JSON, and makes no performance claim beyond the labeled numbers.
"""
from __future__ import annotations

import argparse, glob, json, os, pathlib, statistics
from collections import defaultdict

SCHEMA = "tinygrad.nv_endpoint_per_token_s1.v1"
DECODE_SIGNATURE = [32, 64, 128, 256, 116]


def anchor_ids(nodes: list[dict]) -> tuple[list[int], list[int]]:
  """Mirror the Q/O half of ``tg_anchor_ids`` from nv_inter_anchor_analysis."""
  q: list[int] = []
  o: list[int] = []
  for n in nodes:
    name = str(n.get("name", ""))
    if name.startswith("q4k_g3_lanemap_gemv_epi_resadd_"):
      o.append(n["id"])
    elif name == "q4k_warp_coop_q8_dp4a_partial_4096_4096":
      q.append(n["id"])
    elif name.startswith("q4k_g3_lanemap_gemv_") and name.endswith("_4096_4096"):
      q.append(n["id"])
  q.sort()
  o.sort()
  return q, o


def windows_from_profile(path: pathlib.Path) -> list[dict]:
  """Return one node-list per decode-token window found in a profile JSONL."""
  records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip()]
  sig = DECODE_SIGNATURE
  n = len(sig)
  out: list[dict] = []
  for i in range(len(records) - n + 1):
    if [len(r.get("entries") or []) for r in records[i:i + n]] != sig:
      continue
    nodes: list[dict] = []
    off = 0
    for gi in range(n):
      for ei, e in enumerate(records[i + gi].get("entries") or []):
        nodes.append({"id": off + ei, "name": str(e.get("name", "")),
                      "start": float(e["start"]), "end": float(e["end"])})
      off += len(records[i + gi].get("entries") or [])
    q, o = anchor_ids(nodes)
    layers = min(len(q), len(o))
    s1 = sum(nodes[oi]["start"] - nodes[qi]["end"] for qi, oi in zip(q[:layers], o[:layers]))
    out.append({"layers": layers, "s1_us": round(s1, 3)})
  return out


def arm_role_from_name(name: str) -> tuple[str, str]:
  # stage4_<arm>_<role>_<index>.profile.jsonl
  stem = name.replace(".profile.jsonl", "")
  parts = stem.split("_")
  if len(parts) >= 3 and parts[-2] in ("control", "candidate"):
    return "_".join(parts[1:-2]), parts[-2]
  return stem, "?"


def labeled_report(values: list[float], settle: int) -> dict:
  vals = sorted(values[settle:])
  if not vals:
    return {"n": 0, "median_us": None, "mad_us": None, "min_us": None, "max_us": None,
            "spread_us": None, "label": "unmeasured"}
  med = statistics.median(vals)
  mad = statistics.median(abs(v - med) for v in vals) if len(vals) > 1 else 0.0
  return {"n": len(vals), "median_us": round(med, 3), "mad_us": round(mad, 3),
          "min_us": round(min(vals), 3), "max_us": round(max(vals), 3),
          "spread_us": round(max(vals) - min(vals), 3), "label": "observed"}


def verdict(controls: dict, candidate: dict, gate_us: float) -> tuple[str, str]:
  if not controls.get("median_us") or not candidate.get("median_us"):
    return "unmeasured", "missing control or candidate per-token S1"
  delta = round(controls["median_us"] - candidate["median_us"], 3)
  # Separability: candidate median must clear the control spread and gate.
  if delta >= gate_us and (controls["spread_us"] or 0) < delta:
    return "supported", (
      f"candidate median S1 {candidate['median_us']}us vs control median "
      f"{controls['median_us']}us = +{delta}us, outside control spread "
      f"{controls['spread_us']}us and >= gate {gate_us}us")
  return "refuted", (
    f"S1 delta +{delta}us; control per-token spread {controls['spread_us']}us "
    f"(gate {gate_us}us) so the arm is not separable from control noise")


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--evidence-dir", type=pathlib.Path, required=True)
  ap.add_argument("--settle", type=int, default=1,
                  help="drop this many leading decode windows per process (default 1)")
  ap.add_argument("--gate-us", type=float, default=150.0)
  ap.add_argument("--out", type=pathlib.Path, default=None)
  args = ap.parse_args()

  buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
  per_file: dict[str, dict] = {}
  for pf in sorted(args.evidence_dir.glob("stage4_*.profile.jsonl")):
    windows = windows_from_profile(pf)
    arm, role = arm_role_from_name(pf.name)
    vals = [w["s1_us"] for w in windows if w["layers"] == 36]
    if not vals:
      continue
    # Settle per file: the leading cold-start windows belong to each fresh
    # process individually, so they must be dropped before pooling across files.
    buckets[(arm, role)].extend(vals[args.settle:])
    per_file[pf.name] = {
      "arm": arm, "role": role, "all_s1_us": vals,
      "settled": labeled_report(vals, args.settle),
      "raw": labeled_report(vals, 0),
    }

  arms = sorted({a for a, _ in buckets})
  brackets: dict[str, dict] = {}
  for arm in arms:
    controls = buckets.get((arm, "control"), [])
    candidate = buckets.get((arm, "candidate"), [])
    # Buckets are already per-file settled above; do not settle a second time.
    ctrl_report = labeled_report(controls, 0)
    cand_report = labeled_report(candidate, 0)
    v, reason = verdict(ctrl_report, cand_report, args.gate_us)
    brackets[arm] = {
      "control": ctrl_report,
      "candidate": cand_report,
      "control_all_us": sorted(controls),
      "candidate_all_us": sorted(candidate),
      "verdict": v,
      "verdict_reason": reason,
    }

  payload = {
    "schema": SCHEMA,
    "evidence_dir": str(args.evidence_dir),
    "settle": args.settle,
    "gate_us": args.gate_us,
    "decode_signature": DECODE_SIGNATURE,
    "per_file": per_file,
    "brackets": brackets,
  }
  out = args.out or (args.evidence_dir / "per_token_s1_verdict.json")
  tmp = out.with_name(f".{out.name}.tmp")
  tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  tmp.replace(out)

  print(f"{'arm':26s} {'role':9s} {'n':>4s} {'median':>9s} {'MAD':>8s} {'min':>9s} {'max':>9s} {'spread':>8s}")
  for arm in arms:
    for role in ("control", "candidate"):
      r = brackets[arm][role]
      if r["n"]:
        print(f"{arm:26s} {role:9s} {r['n']:4d} {r['median_us']:9.2f} {r['mad_us']:8.2f} "
              f"{r['min_us']:9.2f} {r['max_us']:9.2f} {r['spread_us']:8.2f}")
  print()
  for arm in arms:
    b = brackets[arm]
    print(f"{arm}: {b['verdict'].upper()} :: {b['verdict_reason']}")
  print(f"verdict json: {out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
