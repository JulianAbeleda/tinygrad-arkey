#!/usr/bin/env python3
"""Offline geometry x split-K search for dense bf16 candidate GEMMs (off the production path).

The candidate space is the precontract tile family the promoted sm_120 schedule belongs to -- tile (m, n, k),
warps (m, n), two LDS buffers -- crossed with a split-K factor S.  A split-K candidate is a batched GEMM over S
K-slices (the batch axis is an ordinary output range the candidate lowering already accepts) followed by an fp32
sum over S.  Each (geometry, S) runs in its own process: compiled programs are cached by the pre-optimization AST,
so two geometries for one shape can never share a process.

  DEV=NV python3 extra/llm_research/prefill/dense_bf16_geometry_search.py --out search.jsonl

Winners are re-validated (oracle error, bit-exactness where S == 1) by dense_bf16_candidate_gate.py before minting.
"""
from __future__ import annotations
import argparse, contextlib, io, json, statistics, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# Nemotron-H 4B BF16 projections: (role, N padded to 128, K)
ROLES = (("ssm_in", 17536, 3136), ("ssm_out", 3200, 7680), ("attn_q", 5120, 3136), ("attn_kv", 1024, 3136),
         ("attn_o", 3200, 5120), ("ffn_up", 12544, 3136), ("ffn_down", 3200, 12544), ("output", 131072, 3136))
ROWS = (16, 32, 64, 128, 256, 512)
# Prefill chunks: hi/lo routes stack two bf16 rows per token, so 8192-token chunks are 16384 GEMM rows.
LARGE_ROWS = (1024, 2048, 4096, 8192, 16384)
# (tile_m, tile_n, tile_k, warps_m, warps_n)
GEOMETRIES = ((128, 128, 32, 4, 2), (128, 64, 32, 4, 2), (64, 128, 32, 2, 2), (64, 64, 32, 2, 2), (64, 64, 64, 2, 2),
              (128, 32, 32, 4, 1), (64, 32, 32, 2, 1), (32, 64, 32, 1, 2), (32, 128, 32, 1, 4), (32, 32, 32, 1, 2),
              (16, 64, 32, 1, 2), (16, 64, 64, 1, 2), (16, 128, 64, 1, 4), (32, 64, 32, 2, 2),
              # round 3: deeper K stages and 1-warp / wide-N tiles (vLLM's small-M choice is 1-warp 16-row CTAs)
              (32, 128, 64, 1, 4), (128, 32, 64, 4, 1), (32, 64, 64, 1, 2), (64, 32, 64, 2, 1), (16, 256, 32, 1, 4),
              (32, 256, 32, 1, 4), (16, 64, 64, 1, 1), (16, 128, 64, 1, 1), (16, 32, 64, 1, 1), (32, 64, 64, 2, 1))
SPLITS = (1, 2, 3, 4, 6, 8)
MAX_LDS = 49152


def lds_bytes(tm, tn, tk): return 2 * (tm + tn) * (tk * 2 + 16)


def feasible(geom, split, m, n, k) -> bool:
  tm, tn, tk, wm, wn = geom
  threads, vpr = wm * wn * 32, tk * 2 // 16
  if m % tm or n % tn or k % split or (k // split) % tk or k // split < 2 * tk: return False
  if tm % (wm * 16) or tn % (wn * 8): return False
  if (tm * vpr) % threads or (tn * vpr) % threads or lds_bytes(tm, tn, tk) > MAX_LDS: return False
  return True


def run_config(geom, split, reps: int, rows=ROWS) -> None:
  from tinygrad import Tensor, Context, GlobalCounters, dtypes
  from tinygrad.codegen.opt import Opt, OptOps
  from tinygrad.codegen.opt.kernel_pipeline import KernelStage1PipelinePlan
  from tinygrad.llm.prefill_candidate_runtime import KernelLDSWindow, KernelTileGeometry, KernelCandidateContext
  import tinygrad.codegen.opt.postrange as pr
  tm, tn, tk, wm, wn = geom
  stride = tk * 2 + 16
  geometry = KernelTileGeometry((tm, tn, tk), (wm, wn), wm * wn * 32, 32,
    (KernelLDSWindow("A", 0, tm * stride, stride), KernelLDSWindow("B", tm * stride, (tm + tn) * stride, stride)))
  context = KernelCandidateContext("boltbeam.full_kernel_candidate.v1", "0" * 64, geometry,
                                   KernelStage1PipelinePlan(2, geometry.lds_bytes, 1))
  weights = {}
  for role, n, k in ROLES:
    for m in rows:
      if not feasible(geom, split, m, n, k) or split * m * n * 4 > 2 << 30: continue   # fp32 output/partials <= 2 GiB
      ks = k // split
      key = pr.warmstart_key({m, n} | ({split} if split > 1 else set()), ks)
      pr._WARMSTART_OPTS = {**(pr._WARMSTART_OPTS or {}), key: (Opt(OptOps.TC, 0, (-1, 2, 1)),)}
      pr._WARMSTART_CANDIDATE_CONTEXTS = {**(pr._WARMSTART_CANDIDATE_CONTEXTS or {}), key: context}
      if (n, k) not in weights:
        weights = {(n, k): Tensor.randn(n, k, dtype=dtypes.bfloat16).mul(0.02).cast(dtypes.bfloat16).realize()}
      w = weights[(n, k)]
      a = Tensor.randn(m, k, dtype=dtypes.bfloat16).cast(dtypes.bfloat16).realize()
      if split > 1:
        a3 = a.reshape(m, split, ks).permute(1, 0, 2).contiguous().realize()
        w3 = w.reshape(n, split, ks).permute(1, 0, 2).contiguous().realize()
        fn = lambda: a3.dot(w3.transpose(1, 2), dtype=dtypes.float).contiguous().sum(0)
      else:
        fn = lambda: a.dot(w.T, dtype=dtypes.float)
      row = {"role": role, "m": m, "n": n, "k": k, "geometry": list(geom), "split_k": split}
      try:
        out = fn().realize()
        ref = a.dot(w.T, dtype=dtypes.float).realize() if split > 1 else None
        times = []
        for _ in range(reps):
          GlobalCounters.reset()
          with Context(DEBUG=2), contextlib.redirect_stdout(io.StringIO()): fn().realize()
          times.append(GlobalCounters.time_sum_s)
        med = statistics.median(times)
        row.update(median_us=round(med * 1e6, 2), tflops=round(2 * m * n * k / med / 1e12, 2),
                   finite=bool(out.isfinite().all().item()))
        if ref is not None: row["max_rel_vs_unsplit"] = float(((out - ref).abs().max() / ref.abs().max()).item())
      except Exception as exc:  # a geometry the lowering refuses is a search result, not a crash
        row["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
      print(json.dumps(row), flush=True)


# Split-K partials live in the JIT's buffers for every layer of a captured graph; cap their extra footprint.
MAX_SPLIT_EXTRA_BYTES = 16 << 20


def select(search_rows: list[dict], controls: dict[tuple[str, int], float]) -> list[dict]:
  """Per exact (role, rows): the fastest finite candidate within the split-K memory cap that beats the ordinary path."""
  best: dict[tuple[str, int], dict] = {}
  for row in search_rows:
    if "median_us" not in row or not row.get("finite") or row.get("max_rel_vs_unsplit", 0.0) > 1e-4: continue
    if (row["split_k"] - 1) * row["m"] * row["n"] * 4 > MAX_SPLIT_EXTRA_BYTES: continue
    key = (row["role"], row["m"])
    if key in controls and row["median_us"] >= controls[key]: continue
    if key not in best or row["median_us"] < best[key]["median_us"]: best[key] = row
  return [{"role": r["role"], "m": r["m"], "n": r["n"], "k": r["k"], "geometry": r["geometry"], "split_k": r["split_k"],
           "search_median_us": r["median_us"], "search_tflops": r["tflops"], "ordinary_median_us": controls.get((r["role"], r["m"]))}
          for _, r in sorted(best.items())]


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", required=True)
  parser.add_argument("--select", nargs=2, metavar=("SEARCH_JSONL", "CONTROL_JSONL"),
                      help="write the selection (JSON) from search rows and ordinary-path control rows (role, m, median_us)")
  parser.add_argument("--reps", type=int, default=7)
  parser.add_argument("--config", help=argparse.SUPPRESS)
  parser.add_argument("--large", action="store_true", help="search the prefill LARGE_ROWS instead of ROWS")
  args = parser.parse_args()
  if args.select:
    rows = [json.loads(line) for line in Path(args.select[0]).read_text().splitlines() if line.strip()]
    controls = {}
    for line in Path(args.select[1]).read_text().splitlines():
      c = json.loads(line) if line.strip() else {}
      if "median_us" in c: controls[(c["role"], c["m"])] = min(c["median_us"], controls.get((c["role"], c["m"]), float("inf")))
    Path(args.out).write_text(json.dumps({"schema": "dense-bf16-geometry-selection.v1", "rows": select(rows, controls)}, indent=1) + "\n")
    return 0
  if args.config:
    geom, split = json.loads(args.config)
    run_config(tuple(geom), split, args.reps, LARGE_ROWS if args.large else ROWS)
    return 0
  with open(args.out, "a") as f:
    for geom in GEOMETRIES:
      for split in SPLITS:
        if not any(feasible(geom, split, m, n, k) for _, n, k in ROLES for m in ROWS): continue
        proc = subprocess.run([sys.executable, __file__, "--out", args.out, "--reps", str(args.reps),
                               "--config", json.dumps([geom, split])], capture_output=True, text=True)
        f.write(proc.stdout); f.flush()
        if proc.returncode: f.write(json.dumps({"geometry": geom, "split_k": split, "process_error": proc.stderr[-300:]}) + "\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
