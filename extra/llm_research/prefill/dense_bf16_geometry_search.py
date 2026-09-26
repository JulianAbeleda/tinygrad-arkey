#!/usr/bin/env python3
"""Offline geometry x split-K search for dense bf16 candidate GEMMs (off the production path).

The candidate space is the precontract tile family the promoted sm_120 schedule belongs to -- tile (m, n, k),
warps (m, n), LDS pipeline -- crossed with a split-K factor S.  The pipeline is either the original register-staged
two-buffer schedule or (``--family ring``) a cp.async ring of 2-8 stages whose fragments leave LDS as native matrix
loads over an XOR-swizzled (unpadded) or padded window, with launch-sized LDS up to the target's runtime maximum.  A split-K candidate is a batched GEMM over S
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
# Pipeline (stages, async_copy, matrix_fragments, xor_swizzle); the first is the original promoted schedule.
SYNC2 = (2, False, False, False)
# Ring family: one generator, not a list of picks.  Tile dims, warp grids and stage counts are enumerated; the
# constraints are the lowering's own (whole per-warp MMA subtiles, whole b128 vectors per thread, LDS within the runtime
# maximum) plus search bounds, each a stated physical argument rather than a tuned pick:
#   * XOR-swizzled windows only: a padded window is conflict-free too but strictly larger, so it can only lose stages;
#   * a per-thread accumulator budget (registers) and at least 2 warps;
#   * tile_m >= rows/4 (each extra M tile re-streams the whole weight);
#   * split-K exists to fill the machine: besides S=1, the smallest S reaching 1, 2 and 4 waves of CTAs.
RING_TILE_M, RING_TILE_N, RING_TILE_K = (16, 32, 64, 128), (32, 64, 128, 256), (32, 64)
RING_WARPS = ((1, 2), (2, 1), (1, 4), (2, 2), (4, 1), (2, 4), (4, 2))
RING_STAGES = (2, 3, 4, 6)
MAX_RING_SPLIT, MAX_ACC_PER_THREAD, SMS, SPLIT_WAVES = 16, 128, 170, (1, 2, 4)
MAX_RUNTIME_LDS = 101376   # CUDARenderer.max_runtime_local_bytes (sm_120 per-block opt-in maximum)


def lds_bytes(tm, tn, tk, pipe=SYNC2):
  stages, _, _, swizzle = pipe
  return stages * (tm + tn) * (tk * 2 + (0 if swizzle else 16))


def feasible(geom, split, m, n, k, pipe=SYNC2) -> bool:
  tm, tn, tk, wm, wn = geom
  threads, vpr = wm * wn * 32, tk * 2 // 16
  if m % tm or n % tn or k % split or (k // split) % tk or k // split < 2 * tk: return False
  if tm % (wm * 16) or tn % (wn * 8): return False
  limit = MAX_LDS if pipe == SYNC2 else MAX_RUNTIME_LDS
  if (tm * vpr) % threads or (tn * vpr) % threads or lds_bytes(tm, tn, tk, pipe) > limit: return False
  return True


def ring_splits(geom, m, n, k, pipe) -> list[int]:
  tm, tn = geom[0], geom[1]
  feasible_splits = [s for s in range(1, MAX_RING_SPLIT + 1) if feasible(geom, s, m, n, k, pipe)]
  grid = lambda s: (n // tn) * (m // tm) * s
  picks = {1} & set(feasible_splits)
  for waves in SPLIT_WAVES:
    picks |= set(next(([s] for s in feasible_splits if grid(s) >= waves * SMS), []))
  return sorted(picks)


def ring_configs(shapes) -> list[tuple]:
  """Every (geometry, split, pipeline) of the ring family admitted for at least one (m, n, k) in ``shapes``."""
  out = set()
  for tm in RING_TILE_M:
    for tn in RING_TILE_N:
      for tk in RING_TILE_K:
        for wm, wn in RING_WARPS:
          if tm % (wm * 16) or tn % (wn * 8) or (tm // wm // 16) * (tn // wn // 8) * 4 > MAX_ACC_PER_THREAD: continue
          for stages in RING_STAGES:
            pipe, geom = (stages, True, True, True), (tm, tn, tk, wm, wn)
            for m, n, k in shapes:
              if tm * 4 < m: continue
              out.update((geom, s, pipe) for s in ring_splits(geom, m, n, k, pipe))
  return sorted(out)


# Decode streams 42 layers of distinct weights through a 96 MB L2: a timing must rotate over enough weight copies that
# no copy is L2-resident when it is reused (the kernel audit's method: >= 384 MB per rotation).
ROTATION_BYTES = 384 << 20


def _context(geom, pipe):
  from tinygrad.codegen.opt.kernel_pipeline import KernelStage1PipelinePlan
  from tinygrad.llm.prefill_candidate_runtime import KernelLDSWindow, KernelTileGeometry, KernelCandidateContext
  tm, tn, tk, wm, wn = geom
  stages, async_copy, matrix, swizzle = pipe
  stride = tk * 2 + (0 if swizzle else 16)
  geometry = KernelTileGeometry((tm, tn, tk), (wm, wn), wm * wn * 32, 32,
    (KernelLDSWindow("A", 0, tm * stride, stride, xor_swizzle=swizzle),
     KernelLDSWindow("B", tm * stride, (tm + tn) * stride, stride, xor_swizzle=swizzle)))
  return KernelCandidateContext("boltbeam.full_kernel_candidate.v1", "0" * 64, geometry,
                                KernelStage1PipelinePlan(stages, geometry.lds_bytes, 1, async_copy=async_copy, matrix_fragments=matrix))


class _Operands:
  """One activation and a rotation of weight copies for an exact (m, n, k), with per-split strided layouts cached."""
  def __init__(self, m, n, k):
    from tinygrad import Tensor, dtypes
    self.m, self.n, self.k = m, n, k
    copies = max(1, -(-ROTATION_BYTES // (n * k * 2)))
    self.a = Tensor.randn(m, k, dtype=dtypes.bfloat16).cast(dtypes.bfloat16).realize()
    self.ws = [Tensor.randn(n, k, dtype=dtypes.bfloat16).mul(0.02).cast(dtypes.bfloat16).realize() for _ in range(copies)]
    self.split_views: dict[int, tuple] = {}
  def split(self, s):
    if s not in self.split_views:
      ks = self.k // s
      self.split_views[s] = (self.a.reshape(self.m, s, ks).permute(1, 0, 2).contiguous().realize(),
                             [w.reshape(self.n, s, ks).permute(1, 0, 2).contiguous().realize() for w in self.ws])
    return self.split_views[s]


def measure(role, ops: _Operands, geom, split, pipe, reps: int) -> dict:
  """One candidate on one exact shape: rotated-weight median GPU time of the GEMM (+ split-K sum), and its error."""
  from tinygrad import Context, GlobalCounters, dtypes
  from tinygrad.codegen.opt import Opt, OptOps
  import tinygrad.codegen.opt.postrange as pr
  m, n, k = ops.m, ops.n, ops.k
  ks = k // split
  key = pr.warmstart_key({m, n} | ({split} if split > 1 else set()), ks)
  pr._WARMSTART_OPTS = {**(pr._WARMSTART_OPTS or {}), key: (Opt(OptOps.TC, 0, (-1, 2, 1)),)}
  pr._WARMSTART_CANDIDATE_CONTEXTS = {**(pr._WARMSTART_CANDIDATE_CONTEXTS or {}), key: _context(geom, pipe)}
  if split > 1:
    a3, w3s = ops.split(split)
    fn = lambda i: a3.dot(w3s[i % len(w3s)].transpose(1, 2), dtype=dtypes.float).contiguous().sum(0)
  else:
    fn = lambda i: ops.a.dot(ops.ws[i % len(ops.ws)].T, dtype=dtypes.float)
  row = {"role": role, "m": m, "n": n, "k": k, "geometry": list(geom), "split_k": split}
  if pipe != SYNC2: row["pipeline"] = list(pipe)
  try:
    out = fn(0).realize()
    times = []
    for i in range(max(reps, len(ops.ws))):
      GlobalCounters.reset()
      with Context(DEBUG=2), contextlib.redirect_stdout(io.StringIO()): fn(i + 1).realize()
      times.append(GlobalCounters.time_sum_s)
    med = statistics.median(times)
    row.update(median_us=round(med * 1e6, 2), tflops=round(2 * m * n * k / med / 1e12, 2), finite=bool(out.isfinite().all().item()))
    if split > 1:
      ref = ops.a.dot(ops.ws[0].T, dtype=dtypes.float).realize()
      row["max_rel_vs_unsplit"] = float(((out - ref).abs().max() / ref.abs().max()).item())
  except Exception as exc:  # a geometry the lowering refuses is a search result, not a crash
    row["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
  return row


def run_config(geom, split, reps: int, rows=ROWS, pipe=SYNC2, roles=None) -> None:
  """Config-major: one candidate over every feasible (role, rows) in this process."""
  for role, n, k in ROLES:
    if roles is not None and role not in roles: continue
    for m in rows:
      if not feasible(geom, split, m, n, k, pipe) or split * m * n * 4 > 2 << 30: continue   # fp32 output/partials <= 2 GiB
      print(json.dumps(measure(role, _Operands(m, n, k), geom, split, pipe, reps)), flush=True)


def run_shape(role: str, m: int, configs, reps: int) -> None:
  """Shape-major: every config for one exact (role, rows) in this process.  Programs are cached by the
  pre-optimization AST (which does not carry the candidate), so both program caches are cleared per candidate."""
  import tinygrad.codegen as codegen, tinygrad.engine.realize as realize
  n, k = next((n, k) for r, n, k in ROLES if r == role)
  ops = _Operands(m, n, k)
  for geom, split, pipe in configs:
    if not feasible(geom, split, m, n, k, pipe) or split * m * n * 4 > 2 << 30: continue
    codegen.to_program_cache.clear(); realize.runtime_cache.clear()
    print(json.dumps(measure(role, ops, tuple(geom), split, tuple(pipe), reps)), flush=True)


def ring_admitted(geom, split, m, n, k, pipe) -> bool:
  """The ring family's lowering constraints and search bounds (see RING_* above) for one exact shape."""
  tm, tn, tk, wm, wn = geom
  return (tm in RING_TILE_M and tn in RING_TILE_N and tk in RING_TILE_K and (wm, wn) in RING_WARPS and pipe[0] in RING_STAGES
          and pipe[1:] == (True, True, True) and not tm % (wm * 16) and not tn % (wn * 8) and tm * 4 >= m
          and (tm // wm // 16) * (tn // wn // 8) * 4 <= MAX_ACC_PER_THREAD and split in ring_splits(geom, m, n, k, pipe))


def ring_neighbors(config, m, n, k) -> list[tuple]:
  """One-axis moves of a ring config to the adjacent enumerated value (tile m/n/k, warp grid, stages, split)."""
  (tm, tn, tk, wm, wn), split, pipe = config
  def adj(values, v): i = values.index(v) if v in values else -1; return [values[j] for j in (i - 1, i + 1) if i >= 0 and 0 <= j < len(values)]
  geoms = [(x, tn, tk, wm, wn) for x in adj(RING_TILE_M, tm)] + [(tm, x, tk, wm, wn) for x in adj(RING_TILE_N, tn)] + \
          [(tm, tn, x, wm, wn) for x in RING_TILE_K if x != tk] + [(tm, tn, tk, *w) for w in RING_WARPS if w != (wm, wn)]
  pipes = [(x, *pipe[1:]) for x in adj(RING_STAGES, pipe[0])]
  out = [(g, s, pipe) for g in geoms for s in ring_splits(g, m, n, k, pipe)] + [((tm, tn, tk, wm, wn), split, p) for p in pipes]
  out += [((tm, tn, tk, wm, wn), s, pipe) for s in ring_splits((tm, tn, tk, wm, wn), m, n, k, pipe) if s != split]
  return [c for c in dict.fromkeys(out) if ring_admitted(*c[:2], m, n, k, c[2])]


def climb_shape(role: str, m: int, seeds, reps: int, min_gain: float = 0.02) -> None:
  """Shape-major hill climb over the ring family: measure the seeds, then repeatedly every one-axis neighbour of the
  incumbent, moving while the best neighbour is at least ``min_gain`` faster.  Every measurement is printed."""
  import tinygrad.codegen as codegen, tinygrad.engine.realize as realize
  n, k = next((n, k) for r, n, k in ROLES if r == role)
  ops, seen = _Operands(m, n, k), {}
  def run(config):
    if config not in seen:
      codegen.to_program_cache.clear(); realize.runtime_cache.clear()
      row = measure(role, ops, *config[:2], config[2], reps)
      ok = "median_us" in row and row["finite"] and row.get("max_rel_vs_unsplit", 0.0) <= 1e-4
      seen[config] = row["median_us"] if ok else float("inf")
      print(json.dumps(row), flush=True)
    return seen[config]
  for control in [c for c in seeds if c[2] == SYNC2 and feasible(*c[:2], m, n, k, c[2])]: run(control)   # e.g. the promoted route
  seeds = [c for c in seeds if ring_admitted(*c[:2], m, n, k, c[2])]
  if not seeds: return
  best = min(seeds, key=run)
  while True:
    candidates = [c for c in ring_neighbors(best, m, n, k) if c not in seen]
    if not candidates: break
    top = min(candidates, key=run)
    if seen[top] >= seen[best] * (1 - min_gain): break
    best = top


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
           **({"pipeline": r["pipeline"]} if "pipeline" in r else {}),
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
  parser.add_argument("--family", choices=("sync2", "ring"), default="sync2", help="pipeline family to enumerate")
  parser.add_argument("--rows", help="comma list of GEMM row counts (default ROWS / LARGE_ROWS)")
  parser.add_argument("--roles", help="comma list of roles (default all)")
  parser.add_argument("--list", action="store_true", help="print the enumerated configs and exit")
  parser.add_argument("--shape", metavar="ROLE:ROWS", help="shape-major: measure the configs given as JSON lines on stdin")
  parser.add_argument("--climb", action="store_true", help="with --shape: hill-climb the ring family from the stdin seeds")
  args = parser.parse_args()
  rows = tuple(int(x) for x in args.rows.split(",")) if args.rows else LARGE_ROWS if args.large else ROWS
  roles = set(args.roles.split(",")) if args.roles else None
  if args.select:
    rows = [json.loads(line) for line in Path(args.select[0]).read_text().splitlines() if line.strip()]
    controls = {}
    for line in Path(args.select[1]).read_text().splitlines():
      c = json.loads(line) if line.strip() else {}
      if "median_us" in c: controls[(c["role"], c["m"])] = min(c["median_us"], controls.get((c["role"], c["m"]), float("inf")))
    Path(args.out).write_text(json.dumps({"schema": "dense-bf16-geometry-selection.v1", "rows": select(rows, controls)}, indent=1) + "\n")
    return 0
  if args.shape:
    role, m = args.shape.split(":")
    configs = [json.loads(line) for line in sys.stdin if line.strip()]
    configs = [(tuple(c[0]), c[1], tuple(c[2]) if len(c) > 2 else SYNC2) for c in configs]
    if args.climb: climb_shape(role, int(m), configs, args.reps)
    else: run_shape(role, int(m), configs, args.reps)
    return 0
  if args.config:
    geom, split, *pipe = json.loads(args.config)
    run_config(tuple(geom), split, args.reps, rows, tuple(pipe[0]) if pipe else SYNC2, roles)
    return 0
  shapes = [(m, n, k) for role, n, k in ROLES if roles is None or role in roles for m in rows]
  if args.family == "ring": configs = ring_configs(shapes)
  else: configs = [(geom, split, SYNC2) for geom in GEOMETRIES for split in SPLITS if any(feasible(geom, split, *s) for s in shapes)]
  if args.list:
    for config in configs: print(json.dumps(config))
    print(len(configs), "configs", file=sys.stderr)
    return 0
  with open(args.out, "a") as f:
    for geom, split, pipe in configs:
      config = [geom, split] + ([list(pipe)] if pipe != SYNC2 else [])
      extra = (["--rows", args.rows] if args.rows else []) + (["--roles", args.roles] if args.roles else []) + (["--large"] if args.large else [])
      proc = subprocess.run([sys.executable, __file__, "--out", args.out, "--reps", str(args.reps), "--config", json.dumps(config), *extra],
                            capture_output=True, text=True)
      f.write(proc.stdout); f.flush()
      if proc.returncode: f.write(json.dumps({"geometry": geom, "split_k": split, "pipeline": list(pipe), "process_error": proc.stderr[-300:]}) + "\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
