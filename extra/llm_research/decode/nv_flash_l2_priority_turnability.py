#!/usr/bin/env python3
"""Test whether PTX L2 eviction priorities preserve a dense decode KV footprint.

This is a primitive gate for the wide-flash production-conditioning residual.
It models all 36 dense layers at depth 512 as one 72 MiB read-only footprint,
then streams 256 MiB between the prime and timed reload.  The candidate arms
mark the footprint ``evict_last`` and/or the disturbance ``evict_first`` using
PTX cache policies.  Only the final normal-policy reload is timestamped.

Measurement tooling only; no production route is modified.
"""
from __future__ import annotations

import argparse, ctypes, json, os, pathlib, statistics, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from nv_r_residual_cache_dispatch_probe import _alloc, _make_queue

SCHEMA = "tinygrad.nv_flash_l2_priority_turnability.v1"
LAYERS = 36
DEPTH = 512
KV_HEADS = 8
HEAD_DIM = 128
DTYPE_BYTES = 2
KV_TENSORS = 2
FOOTPRINT_BYTES = LAYERS * DEPTH * KV_HEADS * HEAD_DIM * DTYPE_BYTES * KV_TENSORS
DISTURB_BYTES = 256 * 1024 * 1024
BLOCK = 256
GRID = (512, 1, 1)
LOCAL = (BLOCK, 1, 1)


def _compile(dev, name: str, words: int, policy: str):
  from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
  if policy == "normal":
    setup = ""
    load = 'asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(q));'
  elif policy in ("evict_last", "evict_first"):
    setup = f'unsigned long long pol; asm volatile("createpolicy.fractional.L2::{policy}.b64 %0, 1.0;" : "=l"(pol));'
    load = 'asm volatile("ld.global.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;" : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(q), "l"(pol));'
  else: raise ValueError(policy)
  src = f'''extern "C" __global__ void {name}(const unsigned int* d, unsigned int* out) {{
  {setup}
  unsigned int acc = 0;
  const unsigned int start = (blockIdx.x * {BLOCK}u + threadIdx.x) * 4u;
  const unsigned int stride = {GRID[0] * BLOCK * 4}u;
  for (unsigned int i = start; i + 3u < {words}u; i += stride) {{
    unsigned int a, b, c, e;
    const unsigned int* q = d + i;
    {load.replace('(d)', '(e)')}
    acc ^= a ^ b ^ c ^ e;
  }}
  if (acc == 0xdeadbeefu) out[blockIdx.x] = acc;
}}'''
  return NVRTCCompiler(dev.arch, ptx=False, cache_key=f"{name}_v1").compile(src)


def _run_sample(dev, programs, buffers, prime: str, disturb: str, timeout_s: float) -> float:
  q = _make_queue(dev)
  # QMDs are mutable scheduler state.  Every occurrence gets a fresh kernarg
  # allocation, including two launches of the same program in one submission.
  def launch(name: str):
    prg = programs[name]
    q.exec(prg, prg.fill_kernargs((buffers[name], buffers["out"])), GRID, LOCAL)
  # Standardize from an evicted footprint, then apply the requested priority.
  launch("disturb_normal")
  launch(f"footprint_{prime}")
  if disturb != "none": launch(f"disturb_{disturb}")
  start, end = dev.new_signal(), dev.new_signal()
  q.timestamp(start)
  launch("footprint_normal")
  q.timestamp(end)
  q.signal(dev.timeline_signal, dev.next_timeline()).submit(dev)
  dev.synchronize(timeout=int(timeout_s * 1000))
  return float(end.timestamp - start.timestamp)


def _summary(xs: list[float], warmup: int) -> dict:
  ys = xs[warmup:]
  return {"n":len(ys), "median_us":round(statistics.median(ys), 3),
          "mean_us":round(statistics.mean(ys), 3), "min_us":round(min(ys), 3),
          "max_us":round(max(ys), 3), "samples_us":[round(x, 3) for x in ys]}


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--n", type=int, default=32)
  ap.add_argument("--warmup", type=int, default=6)
  ap.add_argument("--timeout-s", type=float, default=180.0)
  ap.add_argument("--out", type=pathlib.Path, required=True)
  args = ap.parse_args()

  side_reserve=int(os.environ.get("NV_CUDA_SIDE_RESERVE", "0"))
  if side_reserve:
    cudart=ctypes.CDLL("libcudart.so");
    if (rc:=cudart.cudaSetDevice(0)) != 0: raise RuntimeError(f"cudaSetDevice failed {rc}")
    if (rc:=cudart.cudaDeviceSetLimit(6,ctypes.c_size_t(side_reserve))) != 0: raise RuntimeError(f"cudaDeviceSetLimit failed {rc}")
  from tinygrad import Device
  from tinygrad.runtime.ops_nv import NVProgram
  from extra.llm_research.decode.qk_norm_rope_wall_bracket import _gpu_state

  dev = Device["NV"]
  footprint = _alloc(dev, FOOTPRINT_BYTES)
  disturb = _alloc(dev, DISTURB_BYTES)
  out = _alloc(dev, GRID[0] * 4)
  for buf in (footprint, disturb, out): dev.allocator._copyin(buf, memoryview(bytearray(buf.size)))
  dev.synchronize()

  specs = {
    "footprint_normal": (FOOTPRINT_BYTES, "normal", footprint),
    "footprint_evict_last": (FOOTPRINT_BYTES, "evict_last", footprint),
    "disturb_normal": (DISTURB_BYTES, "normal", disturb),
    "disturb_evict_first": (DISTURB_BYTES, "evict_first", disturb),
  }
  programs, buffers = {}, {"out":out}
  for name, (size, policy, buf) in specs.items():
    programs[name] = NVProgram(dev, name, _compile(dev, name, size // 4, policy))
    buffers[name] = buf

  definitions = {
    "hot_control": ("normal", "none"),
    "normal_prime__normal_stream": ("normal", "normal"),
    "evict_last_prime__normal_stream": ("evict_last", "normal"),
    "normal_prime__evict_first_stream": ("normal", "evict_first"),
    "evict_last_prime__evict_first_stream": ("evict_last", "evict_first"),
  }
  rows = {}
  for arm, (prime, stream) in definitions.items():
    vals = [_run_sample(dev, programs, buffers, prime, stream, args.timeout_s) for _ in range(args.n)]
    rows[arm] = _summary(vals, args.warmup)
    print(json.dumps({arm:rows[arm]}, sort_keys=True), flush=True)

  hot = rows["hot_control"]["median_us"]
  cold = rows["normal_prime__normal_stream"]["median_us"]
  penalty = cold - hot
  recoveries = {arm:round(cold-row["median_us"], 3) for arm, row in rows.items() if arm not in ("hot_control", "normal_prime__normal_stream")}
  shares = {arm:round(val/penalty, 3) if penalty > 0 else None for arm, val in recoveries.items()}
  payload = {
    "schema":SCHEMA,
    "commit":subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
    "gpu_state":_gpu_state(),
    "model":{"layers":LAYERS, "depth":DEPTH, "kv_heads":KV_HEADS, "head_dim":HEAD_DIM,
             "footprint_bytes":FOOTPRINT_BYTES, "footprint_mib":FOOTPRINT_BYTES/1024/1024,
             "disturb_bytes":DISTURB_BYTES, "disturb_mib":DISTURB_BYTES/1024/1024},
    "method":"native NV HCQ; aggregate 36-layer footprint; cold-standardize, prime, disturb, timestamp normal-policy reload only",
    "n_per_arm":args.n-args.warmup,"cuda_side_context_reserve_bytes":side_reserve,
    "rows":rows,
    "reconciliation":{"hot_us":hot, "cold_us":cold, "cold_penalty_us":round(penalty, 3),
                      "candidate_recovery_us":recoveries, "candidate_share_of_cold_penalty":shares},
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
  print(json.dumps(payload["reconciliation"], indent=2, sort_keys=True))
  return 0


if __name__ == "__main__": raise SystemExit(main())
