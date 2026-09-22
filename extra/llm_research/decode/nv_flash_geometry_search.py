#!/usr/bin/env python3
"""Exhaustive NV flash geometry search (P3): enumerate, correctness-check, and CUPTI-measure.

The production score body is pinned at 4.19 us (nv-flash-body-device-timing-20260813.json).
This harness reuses that exact discipline -- DEV=CUDA, isolated score kernel, warmup + back-to-back
launches, nsys --trace=cuda CUPTI_ACTIVITY_KIND_KERNEL duration -- but profiles the whole legal
tile population in one capture and then parses the sqlite by kernel name.  The hard gate is a
candidate median strictly below the control's in-session median and the pinned 4.19 us body.

Modes:
  enumerate          CPU-only population dump (no GPU)
  check              GPU numerical check of the fused tile+combine output vs the production tile
  measure            GPU timing loop; run this process under `nsys profile --trace=cuda`
  parse              CPU-only: attach nsys sqlite durations to the measure metadata
"""
from __future__ import annotations

import argparse, json, sqlite3, subprocess, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tinygrad import Device, Tensor, dtypes
from tinygrad.uop.ops import UOp
from tinygrad.llm.flash_decode_attention import describe_flash_decode_attention
from tinygrad.llm.kernel_program import (KernelProgram, KernelProgramProvenance, OutputSpec,
                                         execute_research_program)
from extra.llm_research.flash_candidate_schema import candidate_hash, tile_fields, to_spec_dict
from extra.llm_research.bubblebeam_futuresight import build_flash_legality, build_flash_static_priority

SCHEMA = "tinygrad.nv_flash_geometry_search.v1"
Hq, Hkv, Hd, MAXC, Tc = 32, 8, 128, 4608, 513
W = Hd + 2
SM120_FACTS = {"subgroup_size": 32, "max_threads_per_threadgroup": 1024,
               "max_threadgroup_memory_bytes": 232448}
CONTROL_NAME = "flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128"


def _ceildiv(a: int, b: int) -> int: return (a + b - 1) // b


def _tile_args(split_count: int):
  return (UOp.placeholder((Hq * split_count * W,), dtypes.float32, 0),
          UOp.placeholder((Hq * Hd,), dtypes.float16, 1),
          UOp.placeholder((2, 1, Hkv, MAXC, Hd), dtypes.float16, 2))


def _inputs(device: str = "CUDA"):
  rng = np.random.default_rng(20260813)
  q = rng.normal(0, .2, Hq * Hd).astype(np.float16)
  cache = rng.normal(0, .2, (2, 1, Hkv, MAXC, Hd)).astype(np.float16)
  return (Tensor(q, device=device).contiguous().realize(),
          Tensor(cache, device=device).contiguous().realize())


def _spec(split_count: int, *, token_block: int = 16, lane_width: int = 32, stage_width: int = 1,
          reduce_structure: str = "staged", dot_pair_width: int = 2, fused_combine: bool = True):
  return describe_flash_decode_attention(Hq, Hd, Hkv, MAXC, split_count, fused_combine=fused_combine,
                                         query_group_size=None, stage_width=stage_width,
                                         token_block=token_block, lane_width=lane_width,
                                         score_group_width=None, warps=None,
                                         reduce_structure=reduce_structure, dot_pair_width=dot_pair_width)


def _tile_program(spec, name: str) -> KernelProgram:
  tc = UOp.const(dtypes.int, Tc)
  return KernelProgram("research.nv_flash_geometry_search", name, KernelProgramProvenance.RESEARCH_ONLY,
                       spec.emit_tile(tc), output_spec=OutputSpec((Hq * spec.tile.split_count * W,), dtypes.float32))


def _combine_program(spec, name: str) -> KernelProgram:
  return KernelProgram("research.nv_flash_geometry_search", name, KernelProgramProvenance.RESEARCH_ONLY,
                       spec.emit_combine(), output_spec=OutputSpec((Hq * Hd,), dtypes.float32))


def build_population(target_facts: dict | None = None, shape: dict | None = None) -> dict:
  target_facts = dict(target_facts or SM120_FACTS)
  shape = dict(shape or {"Hq": Hq, "Hkv": Hkv, "Hd": Hd, "MAXC": MAXC, "Tc": Tc})
  legality = build_flash_legality({}, target_facts)
  priority = build_flash_static_priority(target_facts)
  rows = []
  for split_count in (32, 48, 64):
    for lane_width in (8, 16, 32):
      for token_block in (8, 16, 32):
        for stage_width in (1, 2, 4, 8):
          for reduce_structure in ("staged", "inline"):
            for dot_pair_width in (2, 4):
              tile = tile_fields(Hq=Hq, Hd=Hd, Hkv=Hkv, MAXC=MAXC, split_count=split_count,
                                 staging="KV_BOTH", quant=False, rope=False, token_block=token_block,
                                 lane_width=lane_width, score_group_width=None, warps=None,
                                 query_group_size=None, stage_width=stage_width,
                                 reduce_structure=reduce_structure, dot_pair_width=dot_pair_width)
              descriptor = to_spec_dict(tile=tile)
              envelope = {"schema_version": "flash_decode_candidate.v1", "tile": tile, "combine": None,
                          "candidate_hash": candidate_hash(descriptor)}
              reason = legality(envelope)
              score, why = priority(envelope)
              row = {"candidate_hash": candidate_hash(descriptor), "tile": tile,
                     "legality": reason, "priority_score": score, "priority_reason": why}
              if reason is None:
                try:
                  spec = _spec(split_count, token_block=token_block, lane_width=lane_width,
                               stage_width=stage_width, reduce_structure=reduce_structure,
                               dot_pair_width=dot_pair_width, fused_combine=True)
                  spec.validate()
                  row["kernel_name"] = spec.tile.kernel_name
                  row["combine_name"] = spec.combine.kernel_name
                except Exception as exc:
                  row["legality"] = "emitter_invalid"
                  row["emitter_error"] = str(exc)
              rows.append(row)
  rows.sort(key=lambda r: (-r["priority_score"], r["candidate_hash"]))
  for i, row in enumerate(rows):
    row["deterministic_order"] = i
  return {"schema": SCHEMA + ".population", "target_facts": target_facts, "shape": shape,
          "control_tile_name": CONTROL_NAME, "candidates": rows}


def run_enumerate(args: argparse.Namespace) -> int:
  population = build_population()
  if args.out:
    Path(args.out).write_text(json.dumps(population, indent=2, sort_keys=True) + "\n")
  else:
    print(json.dumps(population, indent=2, sort_keys=True))
  legal = [r for r in population["candidates"] if r["legality"] is None]
  print(f"population={len(population['candidates'])} legal={len(legal)}", file=sys.stderr)
  return 0


def _run_fused(spec, q: Tensor, cache: Tensor) -> np.ndarray:
  tile = _tile_program(spec, spec.tile.kernel_name)
  combine = _combine_program(spec, spec.combine.kernel_name)
  partial = execute_research_program(Tensor.empty(Hq * spec.tile.split_count * W, dtype=dtypes.float32,
                                                   device=q.device), q, cache, program=tile)
  out = execute_research_program(Tensor.empty(Hq * Hd, dtype=dtypes.float32, device=q.device),
                                 partial, program=combine)
  out.realize()
  Device[q.device].synchronize()
  return np.asarray(out.numpy()).astype(np.float32)


def run_check(args: argparse.Namespace) -> int:
  if Device.DEFAULT != "CUDA":
    raise RuntimeError(f"DEV=CUDA required for numerical check, got {Device.DEFAULT}")
  population = json.loads(Path(args.pop).read_text())
  legal = [r for r in population["candidates"] if r["legality"] is None]
  if args.max_candidates:
    legal = legal[:args.max_candidates]
  q, cache = _inputs()
  control_spec = _spec(48, fused_combine=True)
  control = _run_fused(control_spec, q, cache)
  results = {"schema": SCHEMA + ".check", "control_tile_name": control_spec.tile.kernel_name,
             "control_finite": bool(np.isfinite(control).all()), "candidates": []}
  for row in legal:
    try:
      spec = _spec(row["tile"]["split_count"], token_block=row["tile"]["token_block"],
                   lane_width=row["tile"]["lane_width"], stage_width=row["tile"]["stage_width"],
                   reduce_structure=row["tile"]["reduce_structure"], dot_pair_width=row["tile"]["dot_pair_width"],
                   fused_combine=True)
      got = _run_fused(spec, q, cache)
      finite = bool(np.isfinite(got).all())
      max_abs = float(np.max(np.abs(got - control))) if got.shape == control.shape else None
      close = max_abs is not None and bool(np.allclose(got, control, atol=2e-3, rtol=2e-3, equal_nan=True))
      results["candidates"].append({"candidate_hash": row["candidate_hash"],
                                    "kernel_name": spec.tile.kernel_name,
                                    "finite": finite, "max_abs": max_abs, "matches_control": close})
    except Exception as exc:
      results["candidates"].append({"candidate_hash": row["candidate_hash"],
                                    "kernel_name": row.get("kernel_name"),
                                    "check_error": str(exc)})
  if args.out:
    Path(args.out).write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
  else:
    print(json.dumps(results, indent=2, sort_keys=True))
  passing = [r for r in results["candidates"] if r.get("matches_control")]
  print(f"checked={len(results['candidates'])} passing={len(passing)}", file=sys.stderr)
  return 0


def run_measure(args: argparse.Namespace) -> int:
  if Device.DEFAULT != "CUDA":
    raise RuntimeError(f"DEV=CUDA required for measurement, got {Device.DEFAULT}")
  population = json.loads(Path(args.pop).read_text())
  checked = json.loads(Path(args.check).read_text()) if args.check else None
  legal = [r for r in population["candidates"] if r["legality"] is None]
  if checked is not None:
    passing = {r["candidate_hash"] for r in checked["candidates"] if r.get("matches_control")}
    legal = [r for r in legal if r["candidate_hash"] in passing]
  if args.max_candidates:
    legal = legal[:args.max_candidates]
  q, cache = _inputs()
  metadata = {"schema": SCHEMA + ".measure", "replays": args.replays, "warmup": args.warmup,
              "control_tile_name": CONTROL_NAME, "runs": []}
  # Control brackets the sweep so the in-session baseline captures any clock/thermal drift.
  order = [CONTROL_NAME] + [r["kernel_name"] for r in legal] + [CONTROL_NAME]
  specs = {"__control__": _spec(48, fused_combine=False)}
  for r in legal:
    specs[r["candidate_hash"]] = _spec(r["tile"]["split_count"], token_block=r["tile"]["token_block"],
                                       lane_width=r["tile"]["lane_width"], stage_width=r["tile"]["stage_width"],
                                       reduce_structure=r["tile"]["reduce_structure"],
                                       dot_pair_width=r["tile"]["dot_pair_width"], fused_combine=False)
  for name in order:
    if name == CONTROL_NAME:
      spec, key = specs["__control__"], "__control__"
    else:
      key = next(r["candidate_hash"] for r in legal if r["kernel_name"] == name)
      spec = specs[key]
    program = _tile_program(spec, name)
    dst = Tensor.empty(Hq * spec.tile.split_count * W, dtype=dtypes.float32, device="CUDA")
    for _ in range(args.warmup):
      execute_research_program(dst, q, cache, program=program).realize()
    Device["CUDA"].synchronize()
    for _ in range(args.replays):
      execute_research_program(dst, q, cache, program=program).realize()
    Device["CUDA"].synchronize()
    metadata["runs"].append({"kernel_name": name, "order": len(metadata["runs"]),
                             "warmup": args.warmup, "replays": args.replays})
    time.sleep(0.02)
  if args.out:
    Path(args.out).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
  else:
    print(json.dumps(metadata, indent=2, sort_keys=True))
  return 0


def _name_map(con: sqlite3.Connection) -> dict[int, str]:
  return {int(i): str(value) for i, value in con.execute("select id, value from StringIds")}


def parse_trace(trace_path: str, names: set[str]) -> dict[str, dict]:
  con = sqlite3.connect(trace_path)
  names_map = _name_map(con)
  grouped: dict[str, list[float]] = {}
  for start, end, short in con.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL"):
    name = names_map.get(int(short), str(short))
    if name not in names:
      continue
    grouped.setdefault(name, []).append((end - start) / 1000.0)
  out = {}
  for name, vals in grouped.items():
    vals.sort()
    out[name] = {"median_us": float(np.median(vals)), "mean_us": float(np.mean(vals)),
                 "min_us": float(vals[0]), "max_us": float(vals[-1]), "instances": len(vals)}
  return out


def run_parse(args: argparse.Namespace) -> int:
  metadata = json.loads(Path(args.measure).read_text())
  check = json.loads(Path(args.check).read_text()) if args.check else None
  names = {run["kernel_name"] for run in metadata["runs"]}
  durations = parse_trace(args.trace, names)
  control = durations.get(CONTROL_NAME, {}).get("median_us")
  rows = []
  for run in metadata["runs"]:
    d = durations.get(run["kernel_name"], {})
    rows.append({"kernel_name": run["kernel_name"], **d,
                 "beats_control": bool(control is not None and d.get("median_us") is not None
                                       and d["median_us"] < control),
                 "beats_4_19": bool(d.get("median_us") is not None and d["median_us"] < 4.19)})
  result = {"schema": SCHEMA + ".evidence", "trace": args.trace,
            "replays": metadata["replays"], "warmup": metadata["warmup"],
            "control_tile_name": CONTROL_NAME, "control_median_us": control,
            "git_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                                  text=True).strip(),
            "rows": rows,
            "winners_vs_control": [r for r in rows if r.get("beats_control")],
            "winners_vs_4_19": [r for r in rows if r.get("beats_4_19")]}
  if args.out:
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  sub = ap.add_subparsers(dest="mode", required=True)
  e = sub.add_parser("enumerate"); e.add_argument("--out", type=Path)
  c = sub.add_parser("check"); c.add_argument("--pop", required=True); c.add_argument("--out", type=Path)
  c.add_argument("--max-candidates", type=int)
  m = sub.add_parser("measure"); m.add_argument("--pop", required=True); m.add_argument("--check")
  m.add_argument("--out", type=Path); m.add_argument("--replays", type=int, default=400)
  m.add_argument("--warmup", type=int, default=20); m.add_argument("--max-candidates", type=int)
  p = sub.add_parser("parse"); p.add_argument("--trace", required=True); p.add_argument("--measure", required=True)
  p.add_argument("--check"); p.add_argument("--out", type=Path)
  args = ap.parse_args(argv)
  return {"enumerate": run_enumerate, "check": run_check, "measure": run_measure, "parse": run_parse}[args.mode](args)


if __name__ == "__main__":
  sys.exit(main())
