#!/usr/bin/env python3
"""Reverse-bracketed hot/cold cache-sensitivity adjudication for R.

The first probe ran ``hot`` then ``cold`` once and reported ``cold - hot``.
That leaves warmup and arm-order confounds open: the ``hot`` arm itself warms
up during its run, and no cache-sensitive control proved the 128 MiB flush
actually evicted L2.  This tool closes both gaps:

  * protocols ``H/C/H`` and ``C/H/C``, using the *last* instance of each
    temperature so every comparison is between settled arms;
  * a warmup discard of the first few reps per arm;
  * a cache-sensitive positive control (a 16 MiB read that fits in the 96 MiB
    L2) that must slow down measurably under the cold arm, and a retained
    flush-buffer readback.

Measurement tooling only; no production code path is touched.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import struct
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from nv_r_residual_cache_dispatch_probe import (  # noqa: E402
  ROWS, _alloc, _make_queue, _compile_flush, FLUSH_MIB, FLUSH_FLOATS, FLUSH_BLOCK, FLUSH_GRID,
  FLUSH_XOR, _sha256,
)

SCHEMA = "tinygrad.nv_r_residual_reverse_bracket.v1"
POSITIVE_MIB = 16
POSITIVE_FLOATS = POSITIVE_MIB * 1024 * 1024 // 4
POSITIVE_BLOCK = 256
POSITIVE_GRID = ((POSITIVE_FLOATS + POSITIVE_BLOCK - 1) // POSITIVE_BLOCK, 1, 1)


def _compile_positive(dev):
  from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
  src = f"""
extern "C" __global__ void nv_r_l2_read_{POSITIVE_MIB}mib(const float* __restrict__ d, float* __restrict__ out) {{
  unsigned int i = blockIdx.x * {POSITIVE_BLOCK}u + threadIdx.x;
  float v = d[i];
  if (v < 0.0f) out[0] = v;
}}
"""
  return NVRTCCompiler(dev.arch, ptx=False, cache_key=f"nv_r_l2_read_{POSITIVE_MIB}mib_v1").compile(src)


def _run_arm(dev, prg, bufs, grid, block, flush_prg, flush_buf, arm, n, timeout_s, vals=()):
  args_states = [prg.fill_kernargs(bufs, vals=tuple(vals)) for _ in range(n)]
  start_sigs = [dev.new_signal() for _ in range(n)]
  end_sigs = [dev.new_signal() for _ in range(n)]
  flush_args = [flush_prg.fill_kernargs((flush_buf,)) for _ in range(n)]
  flush_grid = FLUSH_GRID

  for args_state, flush_state, st_sig, en_sig in zip(args_states, flush_args, start_sigs, end_sigs):
    q = _make_queue(dev)
    if arm == "cold":
      q.exec(flush_prg, flush_state, flush_grid, (FLUSH_BLOCK, 1, 1))
    q.timestamp(st_sig)
    q.exec(prg, args_state, grid, block)
    q.timestamp(en_sig)
    target = dev.next_timeline()
    q.signal(dev.timeline_signal, target).submit(dev)
    dev.synchronize(timeout=int(timeout_s * 1000))
  durations = [float(en.timestamp - st.timestamp) for st, en in zip(start_sigs, end_sigs)]
  return [round(x, 3) for x in durations]


def _med(vals):
  return round(statistics.median(vals), 3)


def _flush_readback(dev, flush_buf):
  # Sample 8 spread words and confirm the bit-exact integer store.
  idx = sorted(set([0, 1, 7, 1023, FLUSH_FLOATS // 2, FLUSH_FLOATS - 1024, FLUSH_FLOATS - 2, FLUSH_FLOATS - 1]))
  out = []
  for j in idx:
    mv = memoryview(bytearray(4))
    dev.allocator._copyout(mv, flush_buf.offset(j * 4, 4))
    got = struct.unpack("<I", mv)[0]
    expect = j ^ FLUSH_XOR
    out.append({"idx": j, "expect": expect, "got": got, "match": got == expect})
  return out


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--n", type=int, default=40, help="reps per arm")
  ap.add_argument("--warmup", type=int, default=8, help="first reps discarded per arm")
  ap.add_argument("--timeout-s", type=float, default=120.0)
  ap.add_argument("--keys", default="", help="comma-separated row keys; empty runs all")
  ap.add_argument("--out", type=pathlib.Path, required=True)
  args = ap.parse_args()

  from tinygrad import Device
  from tinygrad.runtime.ops_nv import NVProgram

  dev = Device["NV"]
  flush_prg = NVProgram(dev, f"nv_r_flush_{FLUSH_MIB}mib", _compile_flush(dev))
  flush_buf = _alloc(dev, FLUSH_MIB * 1024 * 1024)
  dev.allocator._copyin(flush_buf, memoryview(bytearray(flush_buf.size)))
  pos_prg = NVProgram(dev, f"nv_r_l2_read_{POSITIVE_MIB}mib", _compile_positive(dev))
  pos_buf = _alloc(dev, POSITIVE_MIB * 1024 * 1024)
  pos_out = _alloc(dev, 4)
  dev.allocator._copyin(pos_buf, memoryview(bytearray(pos_buf.size)))
  dev.allocator._copyin(pos_out, memoryview(bytearray(4)))
  dev.synchronize()

  keys = [k for k in args.keys.split(",") if k] if args.keys else None
  rows_out = []

  def bracket(prg, bufs, grid, block, vals=()):
    protocols = {
      "H/C/H": ("hot", "cold", "hot"),
      "C/H/C": ("cold", "hot", "cold"),
    }
    result = {}
    for name, seq in protocols.items():
      arms = []
      for arm in seq:
        durs = _run_arm(dev, prg, bufs, grid, block, flush_prg, flush_buf, arm, args.n, args.timeout_s, vals)
        kept = durs[args.warmup:]
        arms.append({"arm": arm, "durations_us": durs, "kept_us": kept,
                     "median_us": _med(kept), "mean_us": round(sum(kept) / len(kept), 3)})
      if name == "H/C/H":
        hot, cold = arms[2], arms[1]
      else:
        hot, cold = arms[1], arms[2]
      result[name] = {
        "arms": arms,
        "hot_median_us": hot["median_us"],
        "cold_median_us": cold["median_us"],
        "cache_state_us": round(cold["median_us"] - hot["median_us"], 3),
      }
    return result

  # Positive cache-sensitive control first: if the flush cannot evict L2, this
  # must be reported as a failed control rather than an R conclusion.
  pos = bracket(pos_prg, (pos_buf, pos_out), POSITIVE_GRID, (POSITIVE_BLOCK, 1, 1))
  rows_out.append({
    "key": "positive_l2_read_16mib", "control": "positive_eviction",
    "symbol": f"nv_r_l2_read_{POSITIVE_MIB}mib",
    "grid": list(POSITIVE_GRID), "block": [POSITIVE_BLOCK, 1, 1],
    "buf_sizes": [pos_buf.size, pos_out.size],
    "protocols": pos,
  })
  print(json.dumps({"key": "positive_l2_read_16mib", **pos}, indent=2), flush=True)

  for row in ROWS:
    if keys and row["key"] not in keys:
      continue
    cubin = row["cubin"].read_bytes()
    prg = NVProgram(dev, row["symbol"], cubin)
    bufs = tuple(_alloc(dev, sz) for sz in row["buf_sizes"])
    for buf in bufs:
      dev.allocator._copyin(buf, memoryview(bytearray(buf.size)))
    dev.synchronize()
    prot = bracket(prg, bufs, row["grid"], row["block"], row["vals"])
    rows_out.append({
      "key": row["key"], "control": row.get("control", False),
      "symbol": row["symbol"],
      "cubin": str(row["cubin"]), "cubin_sha256": _sha256(row["cubin"]),
      "grid": list(row["grid"]), "block": list(row["block"]),
      "buf_sizes": row["buf_sizes"], "count": row["count"],
      "production_p_us": row["production_p_us"], "clean_hcq_c_us": row["clean_hcq_c_us"],
      "body_b_us": row["body_b_us"], "protocols": prot,
    })
    print(json.dumps({"key": row["key"], **{k: {kk: vv for kk, vv in v.items() if kk != "arms"} for k, v in prot.items()}}, indent=2), flush=True)

  readback = _flush_readback(dev, flush_buf)
  payload = {
    "schema": SCHEMA,
    "commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
    "method": "reverse-bracketed hot/cold (H/C/H and C/H/C), last instance per "
              "temperature, warmup-discarded median; 128 MiB flush plus a 16 MiB "
              "L2-resident positive eviction control",
    "flush_mib": FLUSH_MIB,
    "positive_mib": POSITIVE_MIB,
    "n_per_arm": args.n,
    "warmup_discard": args.warmup,
    "flush_readback": readback,
    "rows": rows_out,
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
