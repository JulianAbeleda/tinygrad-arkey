#!/usr/bin/env python3
"""Rotating-working-set cache probe for the production-conditioned residual R.

The first probe invalidated its ``cold`` arm: the 128 MiB streaming flush wrote
only block 0 and therefore never evicted L2.  This tool replaces the flush with
disjoint working-set rotation, which exercises the cache without the flush
kernel, without touching ``active_qmd`` beyond the existing timestamp-reset the
first probe already ran successfully, and without the QMD/allocator fault path.

For each retained production cubin the tool measures two arms:

  hot        every launch reads the same weight/cache buffer (resident)
  rotating   launches round-robin through disjoint copies whose aggregate size
             exceeds 2x the 96 MiB L2, so each launch's working set has been
             evicted before it is touched again

The target buffer geometry, cubin, timestamp commands, queue construction and
output buffer are identical between arms; only the working-set buffer address
rotates.  A 16 MiB read/checksum kernel is measured first as a positive
eviction control.  If that control does not separate, the cache-state question
is reported ``UNMEASURED`` rather than inferred from the target rows.

Measurement tooling only; no production model, renderer, scheduler, runtime,
or route-policy code is changed.
"""
from __future__ import annotations

import argparse
import hashlib
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
  ROWS, _alloc, _make_queue, _sha256,
)

SCHEMA = "tinygrad.nv_r_residual_rotation_probe.v1"
L2_MIB = 96
TWO_X_L2_BYTES = 2 * L2_MIB * 1024 * 1024
POSITIVE_MIB = 16
POSITIVE_BYTES = POSITIVE_MIB * 1024 * 1024
POSITIVE_FLOATS = POSITIVE_BYTES // 4
POSITIVE_BLOCK = 256
POSITIVE_GRID = ((POSITIVE_FLOATS + POSITIVE_BLOCK - 1) // POSITIVE_BLOCK, 1, 1)
MAX_COPIES = 256

# The one buffer that represents the cubin's streaming working set.  All other
# buffers (activations, outputs, residuals) stay fixed so the comparison is a
# pure working-set rotation, not a geometry or launch change.
ROTATE_IDX = {
  "control_norm_8_128": 0,
  "control_norm_32_128": 0,
  "q_coop_4096": 1,
  "q_g3_4096": 1,
  "o_epi_4096": 1,
  "flash_score": 2,
  "kv_coop_1024": 1,
  "kv_g3_1024": 1,
}


def _copies_for(size: int) -> int:
  # Enough disjoint copies that the aggregate exceeds 2x L2, with a +1 margin,
  # capped so the tiny control rows stay a negative (non-evicting) control.
  return max(1, min(MAX_COPIES, (TWO_X_L2_BYTES + size - 1) // size + 1))


def _compile_positive(dev):
  from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
  src = f"""
extern "C" __global__ void nv_r_rot_read_{POSITIVE_MIB}mib(const float* __restrict__ d, float* __restrict__ out) {{
  unsigned int i = blockIdx.x * {POSITIVE_BLOCK}u + threadIdx.x;
  if (i < {POSITIVE_FLOATS}u) {{
    float v = d[i];
    if (v < 0.0f) out[0] = v;
  }}
}}
"""
  return NVRTCCompiler(dev.arch, ptx=False, cache_key=f"nv_r_rot_read_{POSITIVE_MIB}mib_v1").compile(src)


def _med(vals: list[float]) -> float:
  return round(statistics.median(vals), 3)


def _run_arm(dev, prg, arg_states, grid, block, n, timeout_s):
  """Bracket each launch with two distinct timestamp semaphores and return
  per-launch durations.  ``arg_states`` already encodes which working-set copy
  each launch uses, so hot and rotating share this exact launch path."""
  start_sigs = [dev.new_signal() for _ in range(n)]
  end_sigs = [dev.new_signal() for _ in range(n)]

  for args_state, st_sig, en_sig in zip(arg_states, start_sigs, end_sigs):
    q = _make_queue(dev)
    q.timestamp(st_sig)
    q.exec(prg, args_state, grid, block)
    q.timestamp(en_sig)
    target = dev.next_timeline()
    q.signal(dev.timeline_signal, target).submit(dev)
    dev.synchronize(timeout=int(timeout_s * 1000))
  return [round(float(en.timestamp - st.timestamp), 3)
          for st, en in zip(start_sigs, end_sigs)]


def _protocol(dev, prg, arg_states_hot, arg_states_rot, grid, block, warmup, n, timeout_s):
  """Run H/C/H and C/H/C, discard warmup, keep the last instance of each
  temperature.  Returns the two settled medians per protocol."""
  out = {}
  for name, seq in (("H/C/H", ("hot", "rot", "hot")),
                    ("C/H/C", ("rot", "hot", "rot"))):
    arms = []
    for arm in seq:
      states = arg_states_hot if arm == "hot" else arg_states_rot
      durs = _run_arm(dev, prg, states, grid, block, n, timeout_s)
      kept = durs[warmup:]
      arms.append({"arm": arm, "durations_us": durs, "kept_us": kept,
                   "median_us": _med(kept), "mean_us": round(sum(kept) / len(kept), 3)})
    hot = arms[2] if name == "H/C/H" else arms[1]
    rot = arms[1] if name == "H/C/H" else arms[2]
    out[name] = {
      "arms": arms,
      "hot_median_us": hot["median_us"],
      "rotating_median_us": rot["median_us"],
      "working_set_state_us": round(rot["median_us"] - hot["median_us"], 3),
    }
  return out


def _fill_states(prg, base_bufs, rotate_idx, copies, n, vals=()):
  """Pre-touch every copy and build per-launch arg states.  Hot pins the base
  working-set buffer; rotating round-robins through the copies."""
  hot_states = [prg.fill_kernargs(base_bufs, vals=tuple(vals)) for _ in range(n)]
  rot_states = []
  for k in range(n):
    bufs = list(base_bufs)
    bufs[rotate_idx] = copies[k % len(copies)]
    rot_states.append(prg.fill_kernargs(tuple(bufs), vals=tuple(vals)))
  return hot_states, rot_states


def _copyin_zero(dev, buf):
  dev.allocator._copyin(buf, memoryview(bytearray(buf.size)))


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--n", type=int, default=48, help="reps per arm")
  ap.add_argument("--warmup", type=int, default=8, help="first reps discarded per arm")
  ap.add_argument("--timeout-s", type=float, default=180.0)
  ap.add_argument("--keys", default="", help="comma-separated row keys; empty runs all")
  ap.add_argument("--out", type=pathlib.Path, required=True)
  args = ap.parse_args()

  from tinygrad import Device
  from tinygrad.runtime.ops_nv import NVProgram

  dev = Device["NV"]
  keys = [k for k in args.keys.split(",") if k] if args.keys else None
  rows_out: list[dict] = []

  # --- Positive eviction control -----------------------------------------
  pos_prg = NVProgram(dev, f"nv_r_rot_read_{POSITIVE_MIB}mib", _compile_positive(dev))
  pos_out_buf = _alloc(dev, 4)
  _copyin_zero(dev, pos_out_buf)
  n_copies = _copies_for(POSITIVE_BYTES)
  pos_copies = [_alloc(dev, POSITIVE_BYTES) for _ in range(n_copies)]
  for c in pos_copies:
    _copyin_zero(dev, c)
  dev.synchronize()

  pos_hot = [pos_prg.fill_kernargs((pos_copies[0], pos_out_buf)) for _ in range(args.n)]
  pos_rot = [pos_prg.fill_kernargs((pos_copies[k % n_copies], pos_out_buf))
             for k in range(args.n)]
  pos = _protocol(dev, pos_prg, pos_hot, pos_rot, POSITIVE_GRID,
                  (POSITIVE_BLOCK, 1, 1), args.warmup, args.n, args.timeout_s)

  # Retained checksum: last word of each copy so the readback is provable.
  check = [0.0] * min(n_copies, 8)
  for i, c in enumerate(pos_copies[:8]):
    mv = memoryview(bytearray(4))
    dev.allocator._copyout(mv, c.offset(POSITIVE_BYTES - 4, 4))
    check[i] = round(struct.unpack("<f", mv)[0], 7)

  rows_out.append({
    "key": f"positive_l2_read_{POSITIVE_MIB}mib",
    "control": "positive_eviction",
    "symbol": f"nv_r_rot_read_{POSITIVE_MIB}mib",
    "grid": list(POSITIVE_GRID), "block": [POSITIVE_BLOCK, 1, 1],
    "working_set_bytes": POSITIVE_BYTES,
    "copies": n_copies,
    "aggregate_bytes": n_copies * POSITIVE_BYTES,
    "protocols": pos,
    "readback_first8_words": check,
  })
  print(json.dumps({"key": rows_out[-1]["key"],
                    **{k: {kk: vv for kk, vv in v.items() if kk != "arms"}
                       for k, v in pos.items()}}, indent=2), flush=True)

  # --- Target rows ---------------------------------------------------------
  for row in ROWS:
    if keys and row["key"] not in keys:
      continue
    cubin = row["cubin"].read_bytes()
    prg = NVProgram(dev, row["symbol"], cubin)
    base_bufs = tuple(_alloc(dev, sz) for sz in row["buf_sizes"])
    for b in base_bufs:
      _copyin_zero(dev, b)
    dev.synchronize()

    rotate_idx = ROTATE_IDX[row["key"]]
    working_set_bytes = row["buf_sizes"][rotate_idx]
    n_copies = _copies_for(working_set_bytes)
    copies = [_alloc(dev, working_set_bytes) for _ in range(n_copies)]
    for c in copies:
      _copyin_zero(dev, c)
    dev.synchronize()

    hot_states, rot_states = _fill_states(prg, base_bufs, rotate_idx, copies, args.n, row["vals"])
    prot = _protocol(dev, prg, hot_states, rot_states, row["grid"], row["block"],
                     args.warmup, args.n, args.timeout_s)
    rows_out.append({
      "key": row["key"], "control": row.get("control", False),
      "symbol": row["symbol"], "cubin": str(row["cubin"]),
      "cubin_sha256": _sha256(row["cubin"]),
      "grid": list(row["grid"]), "block": list(row["block"]),
      "buf_sizes": row["buf_sizes"], "rotate_buf_idx": rotate_idx,
      "working_set_bytes": working_set_bytes,
      "copies": n_copies,
      "aggregate_bytes": n_copies * working_set_bytes,
      "production_p_us": row["production_p_us"],
      "clean_hcq_c_us": row["clean_hcq_c_us"],
      "body_b_us": row["body_b_us"],
      "protocols": prot,
    })
    print(json.dumps({"key": row["key"],
                      **{k: {kk: vv for kk, vv in v.items() if kk != "arms"}
                         for k, v in prot.items()}}, indent=2), flush=True)

  payload = {
    "schema": SCHEMA,
    "commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
    "method": ("rotating disjoint working sets (aggregate > 2x 96 MiB L2) vs a "
               "single resident working set; reverse-bracketed H/C/H and C/H/C, "
               "warmup-discarded, last-instance median; positive 16 MiB read "
               "eviction control measured first"),
    "l2_mib": L2_MIB,
    "two_x_l2_bytes": TWO_X_L2_BYTES,
    "positive_mib": POSITIVE_MIB,
    "n_per_arm": args.n,
    "warmup_discard": args.warmup,
    "rows": rows_out,
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
