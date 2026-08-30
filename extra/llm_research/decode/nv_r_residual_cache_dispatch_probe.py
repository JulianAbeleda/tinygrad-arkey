#!/usr/bin/env python3
"""Adjudicate the production-conditioned residual R into cache-state versus
dispatch for the exact production cubins.

For each R row the retained ledger already separates the production command
interval ``P`` into exact body ``B``, clean chained-HCQ dispatch ``D = C - B``,
and the unnamed residual ``R = P - C``.  This probe measures the one term the
ledger did not: how much of ``R`` is L2 cache-state by replaying the exact
cubin on the native NV HCQ path in two arms,

  hot   N chained QMDs, no flush          (weights stay resident in L2)
  cold  N QMDs, a 128 MiB streaming write evicts L2 before each launch

Each kernel is bracketed by two distinct timestamp semaphores (the faithful
HCQ profile replica), so the flush kernel's own execution is outside the
measured interval.  The difference ``cold - hot`` is the cache-state share of
``R``; ``hot - B`` reproduces ``D`` (clean dispatch), and anything left of
``P - cold`` is a non-cache production residual (dispatch tail / predecessor
interference), reported without further mechanism assignment.

This is measurement tooling only.  It loads retained production cubins and
changes no production model, renderer, scheduler, runtime, or route policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import statistics
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

SCHEMA = "tinygrad.nv_r_residual_cache_dispatch_probe.v1"
EVIDENCE_ROOT = ROOT / "docs/task_workflow/evidence/nv-installed-islands-20260822"
CATCH_ROOT = ROOT / "docs/task_workflow/evidence/nv-catch-llama-ledger-20260822/phase1"

FLUSH_MIB = 128
FLUSH_FLOATS = FLUSH_MIB * 1024 * 1024 // 4
FLUSH_BLOCK = 256
FLUSH_GRID = (4096, 1, 1)
FLUSH_STRIDE = FLUSH_GRID[0] * FLUSH_BLOCK
FLUSH_XOR = 0xA5A5A5A5


ROWS = [
  {
    "key": "control_norm_8_128",
    "cubin": ROOT / "docs/task_workflow/evidence/nv-qk-head-norm-predecessor-20260822/reduce_output_rmsnorm_8_128.cubin",
    "symbol": "reduce_output_rmsnorm_8_128",
    "grid": (8, 1, 1), "block": (2, 16, 1),
    "buf_sizes": [4096, 4096, 4096], "vals": [],
    "production_p_us": None, "clean_hcq_c_us": 1.698, "body_b_us": 1.196,
    "count": 0, "control": True,
  },
  {
    "key": "control_norm_32_128",
    "cubin": ROOT / "docs/task_workflow/evidence/nv-qk-head-norm-predecessor-20260822/reduce_output_rmsnorm_32_128.cubin",
    "symbol": "reduce_output_rmsnorm_32_128",
    "grid": (32, 1, 1), "block": (4, 8, 1),
    "buf_sizes": [16384, 16384, 4096], "vals": [],
    "production_p_us": None, "clean_hcq_c_us": 1.698, "body_b_us": 1.190,
    "count": 0, "control": True,
  },
  {
    "key": "q_coop_4096",
    "cubin": EVIDENCE_ROOT / "phase7/q4k_warp_coop_q8_dp4a_partial_4096_4096.cubin",
    "symbol": "q4k_warp_coop_q8_dp4a_partial_4096_4096",
    "grid": (4096, 1, 1), "block": (128, 1, 1),
    "buf_sizes": [65536, 9437184, 8192], "vals": [512],
    "production_p_us": 8.416, "clean_hcq_c_us": 5.3093, "body_b_us": 4.8,
    "count": 17, "control": False,
  },
  {
    "key": "q_g3_4096",
    "cubin": EVIDENCE_ROOT / "phase7/q4k_g3_lanemap_gemv_4096_4096.cubin",
    "symbol": "q4k_g3_lanemap_gemv_4096_4096",
    "grid": (4096, 1, 1), "block": (32, 1, 1),
    "buf_sizes": [16384, 9437184, 8192], "vals": [],
    "production_p_us": 8.704, "clean_hcq_c_us": 7.7512, "body_b_us": 7.488,
    "count": 19, "control": False,
  },
  {
    "key": "o_epi_4096",
    "cubin": CATCH_ROOT / "q4k_g3_lanemap_gemv_epi_resadd_4096_4096.cubin",
    "symbol": "q4k_g3_lanemap_gemv_epi_resadd_4096_4096",
    "grid": (4096, 1, 1), "block": (32, 1, 1),
    "buf_sizes": [16384, 9437184, 8192, 16384], "vals": [],
    "production_p_us": 9.184, "clean_hcq_c_us": 7.6979, "body_b_us": 7.584,
    "count": 36, "control": False,
  },
  {
    "key": "flash_score",
    "cubin": CATCH_ROOT / "flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128.cubin",
    "symbol": "flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128",
    "grid": (48, 8, 1), "block": (32, 4, 1),
    "buf_sizes": [798720, 16384, 18874368], "vals": [512],
    "production_p_us": 6.272, "clean_hcq_c_us": 3.658, "body_b_us": 3.84,
    "count": 36, "control": False,
  },
  {
    "key": "kv_coop_1024",
    "cubin": EVIDENCE_ROOT / "phase7/q4k_warp_coop_q8_dp4a_partial_1024_4096.cubin",
    "symbol": "q4k_warp_coop_q8_dp4a_partial_1024_4096",
    "grid": (1024, 1, 1), "block": (128, 1, 1),
    "buf_sizes": [16384, 2359296, 8192], "vals": [512],
    "production_p_us": 3.712, "clean_hcq_c_us": 2.310, "body_b_us": 2.016,
    "count": 26, "control": False,
  },
  {
    "key": "kv_g3_1024",
    "cubin": CATCH_ROOT / "q4k_g3_lanemap_gemv_1024_4096.cubin",
    "symbol": "q4k_g3_lanemap_gemv_1024_4096",
    "grid": (1024, 1, 1), "block": (32, 1, 1),
    "buf_sizes": [4096, 2359296, 8192], "vals": [],
    "production_p_us": 4.768, "clean_hcq_c_us": 3.798, "body_b_us": 3.328,
    "count": 28, "control": False,
  },
]


def _sha256(path: pathlib.Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def _median(values: list[float]) -> float | None:
  return statistics.median(values) if values else None


def _alloc(dev, size: int):
  from tinygrad.device import BufferSpec
  return dev.allocator._alloc(size, BufferSpec())


def _make_queue(dev):
  from tinygrad.runtime.ops_nv import NVComputeQueue
  q = NVComputeQueue(queue_idx=0)
  q.setup(compute_class=dev.iface.compute_class, local_mem_window=dev.local_mem_window,
          shared_mem_window=dev.shared_mem_window)
  q.wait(dev.timeline_signal, dev.timeline_value - 1).memory_barrier()
  return q


def _compile_flush(dev):
  from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
  src = f"""
extern "C" __global__ void nv_r_flush_{FLUSH_MIB}mib(unsigned int* d) {{
  // The native NV loader does not populate NVRTC's dynamic blockDim/gridDim
  // ABI state.  tinygrad-rendered kernels bake these dimensions too.  Keep
  // the measurement kernel on that qualified path and use integer stores so
  // the readback is bit-exact.
  for (unsigned int i = blockIdx.x * {FLUSH_BLOCK}u + threadIdx.x; i < {FLUSH_FLOATS}u; i += {FLUSH_STRIDE}u) {{
    d[i] = i ^ {FLUSH_XOR}u;
  }}
}}
"""
  return NVRTCCompiler(dev.arch, ptx=False, cache_key=f"nv_r_flush_{FLUSH_MIB}mib_constgeom_v3").compile(src)


def _run_arm(dev, prg, bufs, flush_prg, flush_buf, arm: str, n: int, timeout_s: float, vals=()) -> dict:
  args_states = [prg.fill_kernargs(bufs, vals=tuple(vals)) for _ in range(n)]
  start_sigs = [dev.new_signal() for _ in range(n)]
  end_sigs = [dev.new_signal() for _ in range(n)]
  # A QMD lives inside its kernarg allocation and is mutable scheduler state.
  # Never alias one packet across multiple in-flight launches.
  flush_args = [flush_prg.fill_kernargs((flush_buf,)) for _ in range(n)]
  flush_grid = FLUSH_GRID

  for args_state, flush_state, st_sig, en_sig in zip(args_states, flush_args, start_sigs, end_sigs):
    # One queue/submission per sample avoids the old manual active_qmd reset.
    # The cold start timestamp is the flush QMD's completion release; the hot
    # start is an explicit WFI timestamp on an otherwise empty queue.
    q = _make_queue(dev)
    if arm == "cold":
      q.exec(flush_prg, flush_state, flush_grid, (FLUSH_BLOCK, 1, 1))
    q.timestamp(st_sig)
    q.exec(prg, args_state, prg_grid, prg_block)
    q.timestamp(en_sig)
    target = dev.next_timeline()
    q.signal(dev.timeline_signal, target).submit(dev)
    dev.synchronize(timeout=int(timeout_s * 1000))
  durations = [float(en.timestamp - st.timestamp) for st, en in zip(start_sigs, end_sigs)]
  return {"n": n, "durations_us": [round(x, 3) for x in durations],
          "median_us": round(statistics.median(durations), 3),
          "mean_us": round(sum(durations) / len(durations), 3)}


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--n", type=int, default=64, help="reps per arm")
  ap.add_argument("--timeout-s", type=float, default=120.0)
  ap.add_argument("--keys", default="", help="comma-separated row keys; empty runs all")
  ap.add_argument("--out", type=pathlib.Path, required=True)
  args = ap.parse_args()

  from tinygrad import Device
  from tinygrad.runtime.ops_nv import NVProgram

  dev = Device["NV"]
  flush_blob = _compile_flush(dev)
  flush_prg = NVProgram(dev, f"nv_r_flush_{FLUSH_MIB}mib", flush_blob)
  flush_buf = _alloc(dev, FLUSH_MIB * 1024 * 1024)
  dev.allocator._copyin(flush_buf, memoryview(bytearray(flush_buf.size)))
  dev.synchronize()

  keys = [k for k in args.keys.split(",") if k] if args.keys else None
  rows_out: list[dict] = []
  for row in ROWS:
    if keys and row["key"] not in keys:
      continue
    global prg_grid, prg_block
    prg_grid, prg_block = row["grid"], row["block"]
    cubin = row["cubin"].read_bytes()
    prg = NVProgram(dev, row["symbol"], cubin)
    bufs = tuple(_alloc(dev, sz) for sz in row["buf_sizes"])
    for buf in bufs:
      dev.allocator._copyin(buf, memoryview(bytearray(buf.size)))
    dev.synchronize()

    hot = _run_arm(dev, prg, bufs, flush_prg, flush_buf, "hot", args.n, args.timeout_s, row["vals"])
    cold = _run_arm(dev, prg, bufs, flush_prg, flush_buf, "cold", args.n, args.timeout_s, row["vals"])
    cache_state = round(cold["median_us"] - hot["median_us"], 3)
    dispatch = round(hot["median_us"] - row["body_b_us"], 3)
    non_cache_residual = (round(row["production_p_us"] - cold["median_us"], 3)
                          if row["production_p_us"] is not None else None)
    rows_out.append({
      "key": row["key"],
      "control": row.get("control", False),
      "symbol": row["symbol"],
      "cubin": str(row["cubin"]),
      "cubin_sha256": _sha256(row["cubin"]),
      "grid": list(row["grid"]), "block": list(row["block"]),
      "buf_sizes": row["buf_sizes"], "vals": row["vals"],
      "n": args.n,
      "hot_median_us": hot["median_us"],
      "cold_median_us": cold["median_us"],
      "cache_state_us": cache_state,
      "dispatch_us": dispatch,
      "production_p_us": row["production_p_us"],
      "clean_hcq_c_us": row["clean_hcq_c_us"],
      "body_b_us": row["body_b_us"],
      "production_r_us": (round(row["production_p_us"] - row["clean_hcq_c_us"], 3)
                          if row["production_p_us"] is not None else None),
      "non_cache_residual_us": (non_cache_residual if row["production_p_us"] is not None else None),
      "count": row["count"],
      "hot_durations_us": hot["durations_us"],
      "cold_durations_us": cold["durations_us"],
    })
    print(json.dumps({k: v for k, v in rows_out[-1].items()
                      if k not in ("hot_durations_us", "cold_durations_us")}, indent=2), flush=True)

  payload = {
    "schema": SCHEMA,
    "commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
    "method": "native NV HCQ chained QMD replay of exact production cubins; hot = no flush, "
              "cold = 128 MiB streaming write before each launch; two distinct timestamp semaphores per kernel",
    "flush_mib": FLUSH_MIB,
    "n_per_arm": args.n,
    "rows": rows_out,
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
