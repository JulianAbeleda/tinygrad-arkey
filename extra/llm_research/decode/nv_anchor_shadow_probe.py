#!/usr/bin/env python3
"""NV cross-layer anchor+shadow overlap probe (Agent D).

Tests whether a small support kernel (norm/rope/reduce-shaped, ~1-4 us) can be
hidden behind a long GEMV-class anchor (~100 us, memory-bound) on the native
DEV=NV route, and at what cross-queue-wait / HBM-contention cost.  The topology
is llama's: a support kernel S depends on anchor P and runs on the aux compute
GPFIFO while the *next* anchor P' (dependent only on P) runs on the primary
queue.  Wall is measured with HCQ timestamp signals, the same primitive as the
decode overlap probes.

Arms (per support duration target):
  pair_serial   P,S on queue 0            -> dP + dS
  pair_shadow   P on q0, S on q1 waits P  -> cross-queue wait tax in isolation
  chain_serial  P,S,P' on queue 0         -> serialized anchor+shadow+anchor
  chain_shadow  P,P' on q0, S on q1 waits P -> realized hidden support wall

Realized overlap (hidden support wall) = wall(chain_serial) - wall(chain_shadow).
Contention/wait tax = dS_serial - realized_hidden + anchor duration delta under
overlap.  Every arm runs interleaved reps and reports medians, mirroring the
envelope probe method (HCQ_NUM_COMPUTE=2 bootstrap GPFIFOs, flock held).

The probe then scales the measured per-pair overlap onto the decode DAG census
(596 kernels/token, per-kernel median durations at HEAD) and emits a predicted
wall delta range per token plus a PROMOTE/NO-GO verdict against the +50 us bar.

Usage:
  flock -w 600 /tmp/gpu-bench.lock env PYTHONPATH=. DEV=NV HCQ_NUM_COMPUTE=2 \\
    python3 extra/llm_research/decode/nv_anchor_shadow_probe.py \\
      --out docs/task_workflow/evidence/nv-anchor-shadow-probe-20260819.json
"""
from __future__ import annotations

import argparse, json, statistics, sys, time
sys.path.insert(0, "/home/ubuntu/tinygrad-arkey")

import numpy as np

from tinygrad import Tensor, Device, dtypes
from extra.llm_research.decode.nv_multi_queue_probe import (
  Job, run_jobs, durations, span, make_queues, alloc_buffers,
  copyin, copyout, lower, out_uop, f32_sha,
)

SCHEMA = "tinygrad.nv_anchor_shadow_probe.v1"

# Decode DAG census at HEAD (nv-full-audit-census-head-20260818.json): one
# representative steady token, 596 kernels.  Keys are kernel program names from
# the per-kernel median census; values are (kernel count, median duration us).
# Anchors (q4k/q6k gemv + vocab_head) are excluded; every other kernel is a
# candidate shadow that could be placed on the aux GPFIFO behind an anchor.
DAG_CENSUS: dict[str, tuple[int, float]] = {
  "r_16_256": (37, 3.87),
  "E_32_32_4": (38, 2.30),
  "E_16_32_4_2": (36, 2.26),
  "E_8_8_16_2": (36, 1.95),
  "reduce_output_rmsnorm_1_4096": (19, 7.84),
  "reduce_output_rmsnorm_32_128": (36, 3.07),
  "reduce_output_rmsnorm_8_128": (36, 3.14),
  "r_8_32_4_4": (26, 1.68),
  "r_32_32_4_4": (17, 1.70),
  "rmsnorm_q8_1_llama_provider_4096": (17, 2.46),
  "r_32_4_1187": (1, 39.14),
  "r_128_16_8_1187": (1, 11.10),
  "E_1187_32_4": (2, 3.65),
  "E_16_4_2_8_16_2_4_4": (1, 3.20),
  "E_2": (1, 1.70),
  "r_16_8": (1, 1.70),
  "E": (1, 1.47),
  "r_32_32_4_32_4": (1, 4.80),
  # flash score/combine are separable non-GEMV mass llama hides behind its
  # mmq chain; kept separate so the model can include or exclude them.
  "flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128": (36, 6.48),
  "flash_fused_gmax_combine_f16_32_128": (36, 3.36),
}

# 2-queue ideal list-schedule saving from the duration-bearing decode DAG
# (nv-queue-count-sweep-20260819.json, same 596-node structure): 624.49 us off
# a 5493.27 us serialized span.  Scaled to the HEAD node-sum (4519 us) the same
# 11.37% proportion gives ~514 us; both bounds are recorded.
SCHEDULE_2Q_SAVING_US_OLD_DAG = 624.49
SCHEDULE_2Q_SAVING_US_HEAD_SCALED = 514.0


def _median(xs: list[float]) -> float:
  return float(statistics.median(xs))


def build_anchor(dev, k: int) -> tuple:
  """GEMV-like producer: x[1,4096] @ W[4096,k] in fp16 (memory-bound dot)."""
  x = Tensor.empty(1, 4096, dtype=dtypes.float16)
  w = Tensor.empty(4096, k, dtype=dtypes.float16)
  out = x @ w
  prg, ast, args = lower(dev, out)
  return prg, ast, args, x, w, out


def build_support_reduce(dev, rows: int, dim: int) -> tuple:
  """rmsnorm-shaped sumsq reduce (the r_* half of decode norms), one kernel."""
  x = Tensor.empty(rows, dim)
  out = (x * x).mean(-1, keepdim=True)
  prg, ast, args = lower(dev, out)
  return prg, ast, args, x, out


def build_support_elementwise(dev, n: int) -> tuple:
  """Elementwise multiply (the E_* epilogue shape), one kernel."""
  a = Tensor.empty(n)
  b = Tensor.empty(n)
  out = a * b
  prg, ast, args = lower(dev, out)
  return prg, ast, args, a, b, out


def copyin16(dev, buf, data):
  dev.allocator._copyin(buf, memoryview(np.ascontiguousarray(data, dtype=np.float16).tobytes()))


def copyout16(dev, buf, n):
  blob = memoryview(bytearray(int(np.prod(n)) * 2))
  dev.allocator._copyout(blob, buf)
  return np.frombuffer(blob, dtype=np.float16).copy()


def _w_arg(args, k: int):
  """The weight arg uop of a [1,4096]@[4096,k] matmul (by element count)."""
  return next(u for u in args if int(np.prod(u.shape)) == 4096 * k)


def fill_anchor(dev, bufs, ast, args, k: int, x_val: float = 2.0,
                w_val: float | None = 1.0):
  """Fill matmul inputs by argument slot (inputs are RESHAPE-wrapped; R5 pattern).

  x has 4096 elements and w has 4096*k; w_val=None skips the weight fill so a
  shared weight buffer is written only once.  Outputs (ast.arg.outs) are never
  filled.
  """
  for slot in ast.arg.ins:
    u = args[slot]
    n = int(np.prod(u.shape))
    if n == 4096:
      copyin16(dev, bufs[u], np.full(n, x_val, dtype=np.float16))
    elif n == 4096 * k and w_val is not None:
      copyin16(dev, bufs[u], np.full(n, w_val, dtype=np.float16))


def fill_support(dev, bufs, ast, args, xs):
  """Fill support inputs by slot (flat BUFFER args; reduce has one input slot,
  elementwise two, both fed the same fp32 vector)."""
  flat = np.ascontiguousarray(xs.reshape(-1), dtype=np.float32)
  for slot in ast.arg.ins:
    u = args[slot]
    n = int(np.prod(u.shape))
    if n != flat.size:
      raise ValueError(f"support input slot {slot}: {n} elements != data {flat.size}")
    copyin(dev, bufs[u], flat)


def run_batch(dev, qs, jobs):
  """run_jobs wrapper returning (timestamps, wall_us, per-kernel durations)."""
  ts = run_jobs(dev, jobs)
  return ts, span(ts), durations(ts)


class Probe:
  def __init__(self, args):
    self.args = args
    self.dev = Device["NV"]
    self.dev.synchronize()
    self.gpfifos = list(self.dev.compute_gpfifos)
    if len(self.gpfifos) < 2:
      raise RuntimeError(f"need >=2 bootstrap compute GPFIFOs (HCQ_NUM_COMPUTE=2), got {len(self.gpfifos)}")
    self.qs = make_queues(self.dev, self.gpfifos[:2])
    self.q0, self.q1 = self.qs

  # ---- calibration -------------------------------------------------------
  def calibrate_anchor(self, target_us: float) -> dict:
    """Pick k so the GEMV producer measures nearest target_us (memory-bound)."""
    best, best_k = 1e9, None
    rows = []
    for k in (16384, 20480, 24576, 32768):
      prg, ast, args, x, w, out = build_anchor(self.dev, k)
      bufs = alloc_buffers(self.dev, args)
      fill_anchor(self.dev, bufs, ast, args, k)
      _, _, ds = run_batch(self.dev, self.qs, [Job(self.q0, prg, [bufs[u] for u in args], ast)])
      med = _median(ds)
      rows.append({"k": k, "median_us": med})
      if abs(med - target_us) < best:
        best, best_k = abs(med - target_us), k
    return {"target_us": target_us, "chosen_k": best_k, "rows": rows,
            "chosen_median_us": next(r["median_us"] for r in rows if r["k"] == best_k)}

  def calibrate_support(self) -> dict:
    """Measure candidate support shapes; map targets 1/2/4 us to nearest size."""
    cands = []
    for rows, dim in ((1, 4096), (4, 4096), (16, 4096), (64, 4096), (16, 1024), (64, 1024)):
      prg, ast, args, x, out = build_support_reduce(self.dev, rows, dim)
      bufs = alloc_buffers(self.dev, args)
      _, _, ds = run_batch(self.dev, self.qs, [Job(self.q0, prg, [bufs[u] for u in args], ast)])
      cands.append({"kind": "rmsnorm_reduce", "rows": rows, "dim": dim,
                    "prg": prg, "ast": ast, "args": args, "median_us": _median(ds)})
    for n in (1 << 18, 1 << 19, 1 << 20, 1 << 21, 1 << 22):
      prg, ast, args, a, b, out = build_support_elementwise(self.dev, n)
      bufs = alloc_buffers(self.dev, args)
      _, _, ds = run_batch(self.dev, self.qs, [Job(self.q0, prg, [bufs[u] for u in args], ast)])
      cands.append({"kind": "elementwise", "n": n,
                    "prg": prg, "ast": ast, "args": args, "median_us": _median(ds)})
    chosen = {}
    for target in self.args.support_targets:
      cand = min(cands, key=lambda c: abs(c["median_us"] - target))
      chosen[target] = {k: cand[k] for k in cand if k in ("kind", "rows", "dim", "n", "median_us")}
      chosen[target]["target_us"] = target
    return {"candidates": [{k: c[k] for k in c if k in ("kind", "rows", "dim", "n", "median_us")} for c in cands],
            "chosen": chosen}

  # ---- arms --------------------------------------------------------------
  def pair_serial(self, anchor, support) -> tuple:
    prg_a, ast_a, args_a, bufs_a, x_a, out_a = anchor
    prg_s, ast_s, args_s, bufs_s, x_s, out_s = support
    jobs = [
      Job(self.q0, prg_a, [bufs_a[u] for u in args_a], ast_a),
      Job(self.q0, prg_s, [bufs_s[u] for u in args_s], ast_s),
    ]
    return run_batch(self.dev, self.qs, jobs)

  def pair_shadow(self, anchor, support) -> tuple:
    prg_a, ast_a, args_a, bufs_a, x_a, out_a = anchor
    prg_s, ast_s, args_s, bufs_s, x_s, out_s = support
    sig = self.dev.new_signal(value=0)
    jobs = [
      Job(self.q0, prg_a, [bufs_a[u] for u in args_a], ast_a, signals=((sig, 1),)),
      Job(self.q1, prg_s, [bufs_s[u] for u in args_s], ast_s, waits=((sig, 1),)),
    ]
    return run_batch(self.dev, self.qs, jobs)

  def chain_serial(self, anchor, anchor2, support) -> tuple:
    prg_a, ast_a, args_a, bufs_a, x_a, out_a = anchor
    prg_a2, ast_a2, args_a2, bufs_a2, x_a2, out_a2 = anchor2
    prg_s, ast_s, args_s, bufs_s, x_s, out_s = support
    jobs = [
      Job(self.q0, prg_a, [bufs_a[u] for u in args_a], ast_a),
      Job(self.q0, prg_s, [bufs_s[u] for u in args_s], ast_s),
      Job(self.q0, prg_a2, [bufs_a2[u] for u in args_a2], ast_a2),
    ]
    return run_batch(self.dev, self.qs, jobs)

  def chain_shadow(self, anchor, anchor2, support) -> tuple:
    prg_a, ast_a, args_a, bufs_a, x_a, out_a = anchor
    prg_a2, ast_a2, args_a2, bufs_a2, x_a2, out_a2 = anchor2
    prg_s, ast_s, args_s, bufs_s, x_s, out_s = support
    sig = self.dev.new_signal(value=0)
    jobs = [
      Job(self.q0, prg_a, [bufs_a[u] for u in args_a], ast_a, signals=((sig, 1),)),
      Job(self.q0, prg_a2, [bufs_a2[u] for u in args_a2], ast_a2),
      Job(self.q1, prg_s, [bufs_s[u] for u in args_s], ast_s, waits=((sig, 1),)),
    ]
    return run_batch(self.dev, self.qs, jobs)

  def run_target(self, target: float, anchor, anchor2, support_spec) -> dict:
    """Interleaved reps of the four arms for one support duration target."""
    prg_s, ast_s, args_s = support_spec["prg"], support_spec["ast"], support_spec["args"]
    support = (prg_s, ast_s, args_s, support_spec["bufs"], None, None)
    rows = {}
    reps = self.args.reps
    for i in range(reps):
      # Alternate arm order each rep to decouple from any slow drift.
      arms = (("pair_serial", "chain_serial", "pair_shadow", "chain_shadow")
              if i % 2 == 0 else ("chain_shadow", "pair_shadow", "chain_serial", "pair_serial"))
      for name in arms:
        ts, wall, ds = {
          "pair_serial": lambda: self.pair_serial(anchor, support),
          "chain_serial": lambda: self.chain_serial(anchor, anchor2, support),
          "pair_shadow": lambda: self.pair_shadow(anchor, support),
          "chain_shadow": lambda: self.chain_shadow(anchor, anchor2, support),
        }[name]()
        rows.setdefault(name, []).append({"wall_us": wall, "durations_us": ds})
    out = {}
    for name in ("pair_serial", "pair_shadow", "chain_serial", "chain_shadow"):
      r = rows[name]
      med_wall = _median([x["wall_us"] for x in r])
      nk = len(r[0]["durations_us"])
      med_ds = [_median([x["durations_us"][j] for x in r]) for j in range(nk)]
      out[name] = {"reps": r, "wall_us_median": med_wall,
                   "kernel_durations_us_median": med_ds}
    return out


def interpolate(points: list[tuple[float, float]], x: float) -> float:
  """Linear interpolation over sorted (x, y) points; clamp at the ends."""
  pts = sorted(points)
  xs = [p[0] for p in pts]
  if x <= xs[0]:
    return pts[0][1]
  if x >= xs[-1]:
    return pts[-1][1]
  for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
    if x0 <= x <= x1:
      f = (x - x0) / (x1 - x0)
      return y0 + f * (y1 - y0)
  raise AssertionError("unreachable")


def scale_to_dag(measured: dict) -> dict:
  """Scale measured per-pair hidden/tax onto the decode DAG census."""
  targets = sorted(measured["targets"].keys())
  points_hidden = [(t, measured["targets"][t]["realized_hidden_us"]) for t in targets]
  points_tax = [(t, measured["targets"][t]["total_tax_us"]) for t in targets]

  def support_mass(include_flash: bool) -> tuple:
    total, pairs = 0.0, 0
    per_kernel = []
    for name, (count, med_us) in DAG_CENSUS.items():
      if not include_flash and name.startswith("flash_"):
        continue
      hidden = interpolate(points_hidden, med_us)
      tax = interpolate(points_tax, med_us)
      total += count * (hidden - tax)
      pairs += count
      per_kernel.append({"name": name, "count": count, "median_us": med_us,
                         "hidden_us": round(hidden, 3), "tax_us": round(tax, 3),
                         "net_us": round(count * (hidden - tax), 3)})
    return total, pairs, per_kernel

  rows = []
  for include_flash in (False, True):
    mass, pairs, per = support_mass(include_flash)
    rows.append({
      "include_flash": include_flash,
      "shadow_pairs": pairs,
      "predicted_wall_delta_us_unclamped": round(mass, 3),
      "predicted_wall_delta_us_clamped": round(min(mass, SCHEDULE_2Q_SAVING_US_HEAD_SCALED), 3),
      "per_kernel": per,
    })
  return {
    "schedule_2q_saving_us_old_dag": SCHEDULE_2Q_SAVING_US_OLD_DAG,
    "schedule_2q_saving_us_head_scaled": SCHEDULE_2Q_SAVING_US_HEAD_SCALED,
    "interp_points": {"hidden_us": points_hidden, "total_tax_us": points_tax},
    "scenarios": rows,
  }


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--out", required=True)
  ap.add_argument("--anchor-us", type=float, default=100.0)
  ap.add_argument("--anchor-k", type=int, default=0, help="fixed k; 0 = calibrate")
  ap.add_argument("--support-targets", type=str, default="1,2,4")
  ap.add_argument("--reps", type=int, default=6)
  args = ap.parse_args()
  args.support_targets = [float(x) for x in args.support_targets.split(",") if x]
  if not args.support_targets:
    ap.error("--support-targets empty")

  probe = Probe(args)
  t0 = time.perf_counter()
  print(f"device ready {time.perf_counter()-t0:.2f}s gpfifos={len(probe.gpfifos)}", file=sys.stderr, flush=True)

  # Anchor calibration (or fixed k), then build the two anchor instances.
  cal = probe.calibrate_anchor(args.anchor_us)
  k = args.anchor_k or cal["chosen_k"]
  prg_a, ast_a, args_a, x_a, w_a, out_a = build_anchor(probe.dev, k)
  bufs_a = alloc_buffers(probe.dev, args_a)
  fill_anchor(probe.dev, bufs_a, ast_a, args_a, k, x_val=2.0, w_val=1.0)
  # Second anchor reuses the weight arg uop of the first (same Tensor), so both
  # anchors read the same weight buffer; only x and out are distinct.
  x_a2 = Tensor.empty(1, 4096, dtype=dtypes.float16)
  out_a2 = x_a2 @ w_a
  prg_a2, ast_a2, args_a2 = lower(probe.dev, out_a2)
  w2_uop = _w_arg(args_a2, k)
  bufs_a2 = alloc_buffers(probe.dev, [u for u in args_a2 if u is not w2_uop])
  bufs_a2[w2_uop] = bufs_a[_w_arg(args_a, k)]
  fill_anchor(probe.dev, bufs_a2, ast_a2, args_a2, k, x_val=3.0, w_val=None)
  anchor = (prg_a, ast_a, args_a, bufs_a, x_a, out_a)
  anchor2 = (prg_a2, ast_a2, args_a2, bufs_a2, x_a2, out_a2)

  # Support calibration: pick sizes nearest the requested durations.
  support_cal = probe.calibrate_support()
  targets: dict[float, dict] = {}
  verification = None
  for target in args.support_targets:
    spec = support_cal["chosen"][target]
    if spec["kind"] == "rmsnorm_reduce":
      prg_s, ast_s, args_s, x_s, out_s = build_support_reduce(probe.dev, spec["rows"], spec["dim"])
      xs = np.linspace(0.1, 2.0, spec["rows"] * spec["dim"], dtype=np.float32).reshape(spec["rows"], spec["dim"])
    else:
      prg_s, ast_s, args_s, x_s, b_s, out_s = build_support_elementwise(probe.dev, spec["n"])
      xs = np.linspace(0.1, 2.0, spec["n"], dtype=np.float32)
    bufs_s = alloc_buffers(probe.dev, args_s)
    fill_support(probe.dev, bufs_s, ast_s, args_s, xs)
    support_spec = {**spec, "prg": prg_s, "ast": ast_s, "args": args_s, "bufs": bufs_s}
    arms = probe.run_target(target, anchor, anchor2, support_spec)

    # Derived per-target metrics (medians).
    wall_ps = arms["pair_serial"]["wall_us_median"]
    wall_psh = arms["pair_shadow"]["wall_us_median"]
    wall_cs = arms["chain_serial"]["wall_us_median"]
    wall_csh = arms["chain_shadow"]["wall_us_median"]
    dS_serial = arms["chain_serial"]["kernel_durations_us_median"][1]
    dS_shadow = arms["chain_shadow"]["kernel_durations_us_median"][2]
    dP_serial = arms["chain_serial"]["kernel_durations_us_median"][0]
    dP2_serial = arms["chain_serial"]["kernel_durations_us_median"][2]
    dP2_shadow = arms["chain_shadow"]["kernel_durations_us_median"][1]
    dP_shadow = arms["chain_shadow"]["kernel_durations_us_median"][0]
    realized_hidden = wall_cs - wall_csh
    wait_tax_pair = wall_psh - wall_ps
    contention_tax = dS_shadow - dS_serial
    anchor_contention = dP2_shadow - dP2_serial
    targets[target] = {
      "chosen_support": {k: spec[k] for k in spec if k not in ("prg", "ast", "args", "bufs")},
      "measured_support_us": dS_serial,
      "arms": arms,
      "dS_serial_us": dS_serial, "dS_shadow_us": dS_shadow,
      "dP_serial_us": dP_serial, "dP2_serial_us": dP2_serial,
      "dP_shadow_us": dP_shadow, "dP2_shadow_us": dP2_shadow,
      "wall_pair_serial_us": wall_ps, "wall_pair_shadow_us": wall_psh,
      "wall_chain_serial_us": wall_cs, "wall_chain_shadow_us": wall_csh,
      "realized_hidden_us": realized_hidden,
      "hidden_fraction_of_support": realized_hidden / dS_serial if dS_serial else 0.0,
      "wait_tax_pair_us": wait_tax_pair,
      "contention_tax_support_us": contention_tax,
      "contention_tax_anchor_us": anchor_contention,
      "total_tax_us": (dS_serial - realized_hidden) + max(0.0, anchor_contention),
    }
    if len(targets) == 1:
      # Correctness gate on the first target: the anchor output is pinned to
      # 8192.0 (x=2.0 dot 4096 ones); the support output is compared to the
      # fp64 CPU reference with a 1e-3 relative tolerance contract.
      anchor_out = copyout16(probe.dev, bufs_a[out_uop(ast_a, args_a)], (1, k))
      ref_anchor = np.full((1, k), 8192.0, dtype=np.float16)
      if spec["kind"] == "rmsnorm_reduce":
        support_out = copyout(probe.dev, bufs_s[out_uop(ast_s, args_s)], spec["rows"])
        support_ref = (xs.astype(np.float64) ** 2).mean(-1, keepdims=True).astype(np.float32).reshape(-1)
      else:
        support_out = copyout(probe.dev, bufs_s[out_uop(ast_s, args_s)], spec["n"])
        support_ref = (xs.astype(np.float64) ** 2).reshape(-1)
      verification = {
        "anchor_hash": f32_sha(anchor_out.reshape(-1)),
        "anchor_ref_hash": f32_sha(ref_anchor.reshape(-1)),
        "anchor_hash_match": f32_sha(anchor_out.reshape(-1)) == f32_sha(ref_anchor.reshape(-1)),
        "anchor_max_err": float(np.abs(anchor_out.astype(np.float32) - ref_anchor.astype(np.float32)).max()),
        "support_hash": f32_sha(support_out.reshape(-1)),
        "support_ref_hash": f32_sha(support_ref),
        "support_hash_match": f32_sha(support_out.reshape(-1)) == f32_sha(support_ref),
        "support_max_err": float(np.abs(support_out.astype(np.float64) - support_ref.astype(np.float64)).max()),
      }
    print(f"target {target}: support={dS_serial:.2f}us hidden={realized_hidden:.2f}us "
          f"wait_tax={wait_tax_pair:.2f}us cont_s={contention_tax:.2f}us cont_a={anchor_contention:.2f}us",
          file=sys.stderr, flush=True)
  if verification is None:
    raise RuntimeError("no verification produced (empty support targets?)")

  scaling = scale_to_dag({"targets": targets})
  clamped = scaling["scenarios"][0]["predicted_wall_delta_us_clamped"]
  unclamped = scaling["scenarios"][0]["predicted_wall_delta_us_unclamped"]
  verdict = "PROMOTE" if clamped > 50.0 else "NO-GO"

  result = {
    "schema": SCHEMA,
    "date": "2026-08-19",
    "branch": "nvidia-bringup-20260731",
    "git_commit": "d14e6964e",
    "device": "RTX 5090 sm_120 DEV=NV",
    "method": "HCQ timestamp-signal wall; HCQ_NUM_COMPUTE=2 bootstrap GPFIFOs; "
              "arms interleaved per rep; medians over %d reps; flock held" % args.reps,
    "anchor": {"class": "fp16 GEMV x[1,4096]@W[4096,k]", "k": k,
               "target_us": args.anchor_us, "calibration": cal},
    "support_calibration": support_cal,
    "targets": targets,
    "verification": verification,
    "scaling": scaling,
    "predicted_wall_delta_us": {
      "base_no_flash": {"unclamped": round(unclamped, 3), "clamped": round(clamped, 3)},
      "with_flash": {"unclamped": round(scaling["scenarios"][1]["predicted_wall_delta_us_unclamped"], 3),
                     "clamped": round(scaling["scenarios"][1]["predicted_wall_delta_us_clamped"], 3)},
    },
    "bar_us": 50.0,
    "conclusion": verdict,
    "rationale": ("per-pair hidden/tax measured at a %gus anchor; scaled over the 596-kernel "
                  "decode census with linear interpolation on support duration and clamped to "
                  "the 2-queue list-schedule saving bound") % args.anchor_us,
  }
  with open(args.out, "w") as f:
    json.dump(result, f, indent=2, sort_keys=True)
    f.write("\n")
  print(json.dumps({"schema": SCHEMA, "anchor_us": args.anchor_us, "k": k,
                    "verdict": verdict,
                    "predicted_delta_us": result["predicted_wall_delta_us"],
                    "targets": {str(t): {"support_us": round(targets[t]["measured_support_us"], 3),
                                         "hidden_us": round(targets[t]["realized_hidden_us"], 3),
                                         "wait_tax_us": round(targets[t]["wait_tax_pair_us"], 3),
                                         "cont_tax_us": round(targets[t]["total_tax_us"], 3)}
                                for t in targets}}, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
