#!/usr/bin/env python3
"""Offline geometry x split-K search for dense bf16 candidate GEMMs (off the production path).

The candidate space is the precontract tile family the promoted sm_120 schedule belongs to -- tile (m, n, k),
warps (m, n), LDS pipeline -- crossed with a split-K factor S, and it is DERIVED (BoltBeam gemm_strategy) from the GPU's
facts and this compiler's lowering facts rather than listed: the pipeline is either the register-staged two-buffer
schedule within the static LDS limit or a cp.async ring of every admissible stage count whose fragments leave LDS as
native matrix loads over an XOR-swizzled (unpadded, when conflict free) or padded window, within the launch-sized LDS
limit.  A split-K candidate is a batched GEMM over S K-slices (the batch axis is an ordinary output range the
candidate lowering already accepts) followed by an fp32 sum over S.  Compiled programs are cached by the
pre-optimization AST, so config-major runs use one process per (geometry, S); shape-major runs clear both caches.

  DEV=NV python3 extra/llm_research/prefill/dense_bf16_geometry_search.py --out search.jsonl            # batch, model top-N
  DEV=NV python3 extra/llm_research/prefill/dense_bf16_geometry_search.py --scan --out scan.jsonl       # climb per shape
  python3 extra/llm_research/prefill/dense_bf16_geometry_search.py --list --rows 64 --roles ssm_in --out /dev/null

Winners are re-validated (oracle error, bit-exactness where S == 1) by dense_bf16_candidate_gate.py before minting.
"""
from __future__ import annotations
import argparse, contextlib, functools, io, json, statistics, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# Nemotron-H 4B BF16 projections: (role, N padded to 128, K)
ROLES = (("ssm_in", 17536, 3136), ("ssm_out", 3200, 7680), ("attn_q", 5120, 3136), ("attn_kv", 1024, 3136),
         ("attn_o", 3200, 5120), ("ffn_up", 12544, 3136), ("ffn_down", 3200, 12544), ("output", 131072, 3136))
ROWS = (16, 32, 64, 128, 256, 512)
# Prefill chunks: hi/lo routes stack two bf16 rows per token, so 8192-token chunks are 16384 GEMM rows.
LARGE_ROWS = (1024, 2048, 4096, 8192, 16384)
# Pipeline (stages, async_copy, matrix_fragments, xor_swizzle); the first is the original promoted schedule.
SYNC2 = (2, False, False, False)
# The candidate space is derived, not listed: BoltBeam's gemm_strategy derives every (geometry, split, pipeline) from
# the GPU's facts (its target registry scan: SM count, shared memory per SM/block, register file, thread/block limits,
# measured DRAM/L2 bandwidth and matrix rate) and this compiler's lowering facts (gemm_lowering_facts.py: the tensor-core
# atom and fragment registers, static and launch-sized LDS limits, cp.async / ldmatrix availability, which XOR-swizzled
# rows are bank-conflict free, whole-K-tile split-K), then prunes configs its roofline-with-quantization model puts
# beyond PRUNE_RATIO of its best.  Families the GPU admits but the lowering cannot emit are reported (``--deferred``).
TARGET_ID = "nvidia_sm120"
# Model-ranked distinct seeds per shape for the hill climb / batch listing (the prune keeps thousands per shape).
DEFAULT_TOP = 12


@functools.cache
def _strategy():
  from extra.llm_research.boltbeam_checkout import require_boltbeam
  require_boltbeam("boltbeam.search.gemm_strategy")
  from boltbeam.search import gemm_strategy
  from extra.llm_research.gemm_lowering_facts import lowering_facts
  return (gemm_strategy, gemm_strategy.GemmMachine.from_target(TARGET_ID),
          gemm_strategy.GemmLowering.from_json(lowering_facts("NV", "sm_120")))


@functools.cache
def derived_space(m, n, k) -> tuple[tuple, ...]:
  """The pruned derived space for one exact shape, model-best first, as (geometry, split, pipeline) tuples."""
  gs, machine, lowering = _strategy()
  shape = gs.Shape(m, n, k)
  return tuple((tuple(c.geometry), c.split_k, tuple(c.pipeline))
               for c in gs.prune(machine, lowering, shape, gs.derive_space(machine, lowering, shape)))


def model_top(m, n, k, top=DEFAULT_TOP, family=None) -> list[tuple]:
  """The model's best config for each of the ``top`` best distinct geometries (optionally one pipeline family)."""
  out, seen = [], set()
  for config in derived_space(m, n, k):
    if family == "sync2" and config[2] != SYNC2 or family == "ring" and config[2] == SYNC2: continue
    if config[0] in seen: continue
    seen.add(config[0]); out.append(config)
    if len(out) >= top: break
  return out


def lds_bytes(tm, tn, tk, pipe=SYNC2):
  gs, _, lowering = _strategy()
  return gs.lds_bytes(lowering, (tm, tn, tk), pipe)


def feasible(geom, split, m, n, k, pipe=SYNC2) -> bool:
  """Whether (geometry, split, pipeline) is in the derived space for this exact shape."""
  return (tuple(geom), split, tuple(pipe)) in set(derived_space(m, n, k))


def ring_configs(shapes, top=DEFAULT_TOP) -> list[tuple]:
  """The model's top ring configs for each (m, n, k) in ``shapes``."""
  return sorted({c for m, n, k in shapes for c in model_top(m, n, k, top, "ring")})


SELECTION = Path(__file__).with_name("dense_bf16_sm120_selection.json")


# Decode streams 42 layers of distinct weights through a 96 MB L2: a timing must rotate over enough weight copies that
# no copy is L2-resident when it is reused (the kernel audit's method: >= 384 MB per rotation).
ROTATION_BYTES = 384 << 20


def _context(geom, pipe):
  from tinygrad.codegen.opt.kernel_pipeline import KernelStage1PipelinePlan
  from tinygrad.llm.prefill_candidate_runtime import KernelLDSWindow, KernelTileGeometry, KernelCandidateContext
  tm, tn, tk, wm, wn = geom
  # pipe = (stages, async_copy, matrix_fragments, xor_swizzle[, barrier_group]); a 4-tuple is barrier group 1
  stages, async_copy, matrix, swizzle, *group = pipe
  stride = tk * 2 + (0 if swizzle else 16)
  geometry = KernelTileGeometry((tm, tn, tk), (wm, wn), wm * wn * 32, 32,
    (KernelLDSWindow("A", 0, tm * stride, stride, xor_swizzle=swizzle),
     KernelLDSWindow("B", tm * stride, (tm + tn) * stride, stride, xor_swizzle=swizzle)))
  return KernelCandidateContext("boltbeam.full_kernel_candidate.v1", "0" * 64, geometry,
                                KernelStage1PipelinePlan(stages, geometry.lds_bytes, 1, async_copy=async_copy, matrix_fragments=matrix,
                                                         barrier_group=group[0] if group else 1))


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


def run_shape(role: str, m: int, configs, reps: int, prototype: bool = False) -> None:
  """Shape-major: every config for one exact (role, rows) in this process.  Programs are cached by the
  pre-optimization AST (which does not carry the candidate), so both program caches are cleared per candidate."""
  import tinygrad.codegen as codegen, tinygrad.engine.realize as realize
  n, k = next((n, k) for r, n, k in ROLES if r == role)
  ops = _Operands(m, n, k)
  for geom, split, pipe in configs:
    # --prototype measures lowering axes the derived space does not enumerate yet (the lowering still fails closed)
    if (not prototype and not feasible(geom, split, m, n, k, pipe)) or split * m * n * 4 > 2 << 30: continue
    codegen.to_program_cache.clear(); realize.runtime_cache.clear()
    print(json.dumps(measure(role, ops, tuple(geom), split, tuple(pipe), reps)), flush=True)


def ring_admitted(geom, split, m, n, k, pipe) -> bool:
  return pipe != SYNC2 and feasible(geom, split, m, n, k, pipe)


def ring_neighbors(config, m, n, k) -> list[tuple]:
  """One-axis moves of a config inside the derived space: tile m/n/k halved or doubled, another warp grid, one stage
  more or fewer, another split."""
  (tm, tn, tk, wm, wn), split, pipe = config
  space = derived_space(m, n, k)
  def one_axis(c):
    (a, b, d, x, y), s, p = c
    diffs = [a != tm, b != tn, d != tk, (x, y) != (wm, wn), p[0] != pipe[0], s != split, p[1:] != pipe[1:]]
    if sum(diffs) != 1: return False
    if a != tm: return a in (tm // 2, tm * 2)
    if b != tn: return b in (tn // 2, tn * 2)
    if d != tk: return d in (tk // 2, tk * 2)
    if p[0] != pipe[0]: return abs(p[0] - pipe[0]) == 1
    return not diffs[6]
  return [c for c in space if one_axis(c)]


def climb_shape(role: str, m: int, seeds, reps: int, min_gain: float = 0.02) -> None:
  """Shape-major hill climb over the derived space: measure the seeds, then repeatedly every one-axis neighbour of the
  incumbent, moving while the best neighbour is at least ``min_gain`` faster.  Every measurement is printed; a seed
  outside the derived space (e.g. a route promoted before it) is measured as a control and not climbed from."""
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
  for control in [c for c in seeds if not feasible(*c[:2], m, n, k, c[2])]: run(control)
  seeds = [c for c in seeds if feasible(*c[:2], m, n, k, c[2])]
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
  parser.add_argument("--family", choices=("sync2", "ring"), help="restrict to one pipeline family (default: both)")
  parser.add_argument("--top", type=int, default=DEFAULT_TOP, help="model-ranked distinct geometries per shape")
  parser.add_argument("--scan", action="store_true",
                      help="per shape (--roles/--rows): hill-climb from the promoted route + the model's top seeds, one process per shape")
  parser.add_argument("--exhaustive", action="store_true",
                      help="with --scan: measure the whole cost-model-pruned derived space per shape instead of climbing")
  parser.add_argument("--deferred", action="store_true", help="print, per shape, the families the GPU admits but the lowering cannot emit")
  parser.add_argument("--rows", help="comma list of GEMM row counts (default ROWS / LARGE_ROWS)")
  parser.add_argument("--roles", help="comma list of roles (default all)")
  parser.add_argument("--list", action="store_true", help="print the enumerated configs and exit")
  parser.add_argument("--shape", metavar="ROLE:ROWS", help="shape-major: measure the configs given as JSON lines on stdin")
  parser.add_argument("--prototype", action="store_true", help="with --shape: measure configs outside the derived space")
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
    else: run_shape(role, int(m), configs, args.reps, prototype=args.prototype)
    return 0
  if args.config:
    geom, split, *pipe = json.loads(args.config)
    run_config(tuple(geom), split, args.reps, rows, tuple(pipe[0]) if pipe else SYNC2, roles)
    return 0
  shapes = [(m, n, k) for role, n, k in ROLES if roles is None or role in roles for m in rows]
  if args.deferred:
    gs, machine, lowering = _strategy()
    for role, n, k in ROLES:
      if roles is not None and role not in roles: continue
      for m in rows: print(json.dumps({"role": role, "m": m, "deferred": gs.deferred(machine, lowering, gs.Shape(m, n, k))}))
    return 0
  if args.scan:
    promoted = {(r["role"], r["m"]): r for r in json.loads(SELECTION.read_text())["rows"]}
    with open(args.out, "a") as f:
      for role, n, k in ROLES:
        if roles is not None and role not in roles: continue
        for m in rows:
          seeds = ([(tuple(p["geometry"]), p["split_k"], tuple(p.get("pipeline", SYNC2)))] if (p := promoted.get((role, m))) else [])
          space = [c for c in derived_space(m, n, k) if args.family is None or (c[2] == SYNC2) == (args.family == "sync2")]
          seeds += [c for c in (space if args.exhaustive else model_top(m, n, k, args.top, args.family)) if c not in seeds]
          print(json.dumps({"role": role, "m": m, "space_pruned": len(derived_space(m, n, k)), "measuring": len(seeds),
                            "mode": "exhaustive" if args.exhaustive else "climb"}), file=sys.stderr, flush=True)
          mode = [] if args.exhaustive else ["--climb"]
          proc = subprocess.run([sys.executable, __file__, "--out", args.out, "--reps", str(args.reps), "--shape", f"{role}:{m}", *mode],
                                input="".join(json.dumps([list(g), s_, list(pp)]) + "\n" for g, s_, pp in seeds), capture_output=True, text=True)
          f.write(proc.stdout); f.flush()
          if proc.returncode: f.write(json.dumps({"role": role, "m": m, "process_error": proc.stderr[-300:]}) + "\n")
    return 0
  configs = sorted({c for m, n, k in shapes for c in model_top(m, n, k, args.top, args.family)})
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
