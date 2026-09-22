#!/usr/bin/env python3
"""NVRTC cache-byte gate for a producer-owned K/V cache sink.

Control:
  installed reduce_output_rmsnorm_rope_8_128 -> legacy K/V cache store

Candidate:
  the same reduction, affine and RoPE arithmetic, with the terminal epilogue
  writing K and already-final V directly to the selected cache slot.

This is measurement tooling only. It deliberately precedes production route
wiring and compares the entire cache allocation byte-for-byte for fp16/fp32
cache dtypes and boundary/interior start positions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tinygrad import Device, dtypes
from tinygrad.codegen import to_program
from tinygrad.codegen.late.reduce_output import emit_reduce_output, emit_reduce_output_rope_kv_cache
from tinygrad.device import BufferSpec
from tinygrad.helpers import Target
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.runtime.ops_nv import NVProgram
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, ReduceOutputSpec, UOp

ROWS, DIM, MAXC, EPS = 8, 128, 19, 1e-6


def _src(u: UOp, ren: CUDARenderer) -> str:
  return next(x.arg for x in to_program(u, ren).src if x.op is Ops.SOURCE)


def _launch_dims(u: UOp, ren: CUDARenderer) -> tuple[tuple[int, ...], tuple[int, ...]]:
  g, l = to_program(u, ren).arg.launch_dims({})
  return tuple(int(x) for x in g), tuple(int(x) for x in l)


def _vars(u: UOp, ren: CUDARenderer) -> tuple[str, ...]:
  return tuple(x.arg[0] for x in to_program(u, ren).arg.vars)


def _alloc(dev, size: int):
  return dev.allocator._alloc(size, BufferSpec())


def _copyin(dev, buf, data: bytes):
  dev.allocator._copyin(buf, memoryview(data))


def _copyout(dev, buf) -> bytes:
  host = memoryview(bytearray(buf.size))
  dev.allocator._copyout(host, buf)
  return bytes(host)


def _legacy_store(cache_dtype):
  def kernel(cache: UOp, k: UOp, v: UOp) -> UOp:
    sp = UOp.variable("start_pos", 0, MAXC - 1)
    row = UOp.range(ROWS, 0, AxisType.GLOBAL)
    elem = UOp.range(DIM, 1, AxisType.GLOBAL)
    flat = row * DIM + elem
    kst = cache[0, 0, row, sp, elem].store(k[flat].cast(cache_dtype))
    vst = cache.after(kst)[1, 0, row, sp, elem].store(v[flat].cast(cache_dtype))
    return vst.end(row, elem).sink(arg=KernelInfo(name=f"legacy_kv_cache_store_{cache_dtype.name}", opts_to_apply=()))
  return kernel


def _build(ren: CUDARenderer, cache_dtype) -> dict:
  spec = ReduceOutputSpec(ROWS, DIM, EPS, dtypes.float32, warps=ROWS, lanes=32, per_lane=4, epilogue="rope")
  cache_shape = (2, 1, ROWS, MAXC, DIM)
  p = UOp.placeholder

  control = emit_reduce_output(spec, dtypes.float32, dtypes.float16)(
    p((ROWS * DIM,), dtypes.float32, 0), p((ROWS * DIM,), dtypes.float32, 1),
    p((DIM,), dtypes.float16, 2), p((MAXC, DIM), dtypes.float32, 3))
  store = _legacy_store(cache_dtype)(
    p(cache_shape, cache_dtype, 0), p((ROWS * DIM,), dtypes.float32, 1), p((ROWS * DIM,), dtypes.float32, 2))
  candidate = emit_reduce_output_rope_kv_cache(spec, dtypes.float32, dtypes.float16, cache_dtype, MAXC)(
    p(cache_shape, cache_dtype, 0), p((ROWS * DIM,), dtypes.float32, 1), p((DIM,), dtypes.float16, 2),
    p((ROWS * DIM,), dtypes.float32, 3), p((MAXC, DIM), dtypes.float32, 4))

  def one(u: UOp) -> dict:
    return {"src": _src(u, ren), "name": u.arg.name, "grid": _launch_dims(u, ren)[0],
            "block": _launch_dims(u, ren)[1], "vars": _vars(u, ren)}
  return {"control": one(control), "store": one(store), "candidate": one(candidate)}


def _run(dev, ren: CUDARenderer, cache_dtype, manifest: dict, pos: int, seed: int) -> dict:
  rng = np.random.default_rng(seed)
  x_np = rng.normal(0.0, 0.35, ROWS * DIM).astype(np.float32)
  w_np = rng.normal(1.0, 0.08, DIM).astype(np.float16)
  v_np = rng.normal(0.0, 3.0, ROWS * DIM).astype(np.float32)
  v_np[:8] = np.asarray([0.0, -0.0, 65504.0, -65504.0, 2**-24, -(2**-24), 1.0, -1.0], dtype=np.float32)
  angles = np.arange(MAXC * (DIM // 2), dtype=np.float32).reshape(MAXC, DIM // 2) * np.float32(0.00091)
  freqs_np = np.concatenate((np.cos(angles), np.sin(angles)), axis=1).astype(np.float32)

  cache_np_dtype = np.float16 if cache_dtype == dtypes.float16 else np.float32
  cache_elems = 2 * ROWS * MAXC * DIM
  # Finite sentinels make accidental outside-slot writes visible without
  # relying on NaN payload behavior.
  sentinel = rng.uniform(-11.0, 11.0, cache_elems).astype(cache_np_dtype)
  initial = np.ascontiguousarray(sentinel).tobytes()

  x = _alloc(dev, x_np.nbytes); w = _alloc(dev, w_np.nbytes); v = _alloc(dev, v_np.nbytes); f = _alloc(dev, freqs_np.nbytes)
  kout = _alloc(dev, ROWS * DIM * 4)
  ctrl_cache = _alloc(dev, len(initial)); cand_cache = _alloc(dev, len(initial))
  for buf, data in ((x, x_np.tobytes()), (w, w_np.tobytes()), (v, v_np.tobytes()), (f, freqs_np.tobytes()),
                    (ctrl_cache, initial), (cand_cache, initial)):
    _copyin(dev, buf, data)

  programs = {key: NVProgram(dev, row["name"], ren.compiler.compile(row["src"])) for key, row in manifest.items()}
  for key, row in manifest.items():
    if row["vars"] != ("start_pos",):
      raise RuntimeError(f"{key} variable ABI changed: {row['vars']}")
  programs["control"](kout, x, w, f, vals=(pos,), global_size=manifest["control"]["grid"],
                      local_size=manifest["control"]["block"], wait=True)
  programs["store"](ctrl_cache, kout, v, vals=(pos,), global_size=manifest["store"]["grid"],
                    local_size=manifest["store"]["block"], wait=True)
  programs["candidate"](cand_cache, x, w, v, f, vals=(pos,), global_size=manifest["candidate"]["grid"],
                        local_size=manifest["candidate"]["block"], wait=True)

  ctrl_b, cand_b = _copyout(dev, ctrl_cache), _copyout(dev, cand_cache)
  itemsize = np.dtype(cache_np_dtype).itemsize
  slot_indexes = []
  for kv in range(2):
    for row in range(ROWS):
      base = ((((kv * 1) * ROWS + row) * MAXC + pos) * DIM)
      slot_indexes.extend(range(base * itemsize, (base + DIM) * itemsize))
  slot_mask = np.zeros(len(ctrl_b), dtype=np.bool_)
  slot_mask[np.asarray(slot_indexes, dtype=np.int64)] = True
  ctrl_u8, cand_u8, initial_u8 = np.frombuffer(ctrl_b, np.uint8), np.frombuffer(cand_b, np.uint8), np.frombuffer(initial, np.uint8)
  mismatch = ctrl_u8 != cand_u8
  outside_changed = (cand_u8 != initial_u8) & ~slot_mask
  return {
    "cache_dtype": cache_dtype.name, "start_pos": pos, "seed": seed,
    "full_cache_bit_exact": bool(not mismatch.any()), "mismatch_bytes": int(mismatch.sum()),
    "outside_slot_changed_bytes": int(outside_changed.sum()),
    "control_sha256": hashlib.sha256(ctrl_b).hexdigest(), "candidate_sha256": hashlib.sha256(cand_b).hexdigest(),
  }


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out-json", type=pathlib.Path, required=True)
  args = ap.parse_args()
  dev = Device["NV"]
  ren = CUDARenderer(Target("NV", arch="sm_120"), use_nvcc=False)

  results, source = [], {}
  for cache_dtype in (dtypes.float16, dtypes.float32):
    manifest = _build(ren, cache_dtype)
    source[cache_dtype.name] = {key: {k: v for k, v in row.items() if k != "src"} | {
      "source_sha256": hashlib.sha256(row["src"].encode()).hexdigest()} for key, row in manifest.items()}
    for i, pos in enumerate((0, 7, MAXC - 1)):
      results.append(_run(dev, ren, cache_dtype, manifest, pos, 20260824 + i))

  passed = all(r["full_cache_bit_exact"] and r["outside_slot_changed_bytes"] == 0 for r in results)
  out = {
    "schema": "tinygrad.nv_producer_kv_cache_sink_microgate.v1",
    "commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
    "method": "NVRTC production compiler; installed K norm+RoPE plus legacy store vs producer-owned direct cache sink",
    "shape": {"Hkv": ROWS, "Hd": DIM, "MAXC": MAXC}, "source": source, "results": results,
    "verdict": "PASS_CACHE_BYTES" if passed else "FAIL_CACHE_BYTES",
  }
  args.out_json.parent.mkdir(parents=True, exist_ok=True)
  args.out_json.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"verdict": out["verdict"], "cases": len(results)}, indent=2))
  for row in results:
    print(f"dtype={row['cache_dtype']} pos={row['start_pos']} exact={row['full_cache_bit_exact']} "
          f"mismatch_bytes={row['mismatch_bytes']} outside_changed={row['outside_slot_changed_bytes']}")
  return 0 if passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
