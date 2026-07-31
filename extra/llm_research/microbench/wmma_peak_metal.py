#!/usr/bin/env python3
"""Metal equivalent of wmma_peak.cpp (this directory): the achievable isolated
simdgroup_multiply_accumulate rate `R`, for the crossover M* = (w/16)*(R/BW) in
docs/what-makes-a-token-fast-20260731.md.

Mirrors wmma_peak.cpp's structure element-for-element:
  - NACC independent accumulators (swept: latency-hiding count is unknown for this
    hardware, so it is measured rather than assumed from AMD's NACC=8).
  - operands (mat_a, mat_b) initialized ONCE, outside the loop -- zero loads in the
    hot loop.
  - hot loop contains nothing but simdgroup_multiply_accumulate, #pragma unroll'ed
    over the (compile-time-bound) NACC dimension; the outer trip count is a runtime
    kernel argument so the compiler cannot fold the whole loop to a compile-time
    constant.
  - result kept live by a never-taken branch (`if (s==1234.5f) out[0]=s;`).
  - FLOP counted as simdgroups * iters * NACC * 2*8*8*8 (Metal's tensor op is
    dims=(8,8,8) -- tinygrad/codegen/opt/tc.py:181 -- vs AMD's 16x16x16, so 1024
    FLOP/op not AMD's 8192).

Grid: global_size = (blocks,1,1) is the threadgroup COUNT (Metal calls this
dispatchThreadgroups), local_size = (tpb,1,1) is threads/threadgroup. One Apple
simdgroup is 32 threads (same width as AMD's wave32 gfx1100, so blocks*tpb/32 is
directly the wave/simdgroup analogue of wmma_peak.cpp's `waves`).

Uses tinygrad's real Device["METAL"] plumbing directly (MetalCompiler.compile ->
MetalProgram, both from tinygrad/runtime/ops_metal.py), the same machinery
scratchpad/t4_fused_generic_tc_execute.py used to dispatch a real compiled kernel,
NOT tinygrad's codegen/AST path -- this is raw hand-authored MSL, compiled and run
through the real Metal runtime. Search the emitted source: it contains
`simdgroup_multiply_accumulate` only, never the newer `simdgroup_matrix` template API.
"""
from __future__ import annotations
import sys, time, json, subprocess, tempfile, pathlib
sys.path.insert(0, "/Users/julianabeleda/env/tinygrad-arkey-exp")

from tinygrad import Device, dtypes
from tinygrad.device import Buffer
from tinygrad.runtime.ops_metal import MetalCompiler, MetalProgram

DEVICE = "METAL"
SIMD_WIDTH = 32          # Apple simdgroup width, every generation
FLOP_PER_OP = 2 * 8 * 8 * 8  # tinygrad/codegen/opt/tc.py:181 -- Metal TC dims=(8,8,8)


def kernel_src(nacc: int) -> str:
  return f"""#include <metal_stdlib>
using namespace metal;

kernel void wmma_peak(device float* out [[buffer(0)]],
                       constant int& iters [[buffer(1)]]) {{
  simdgroup_half8x8 mat_a, mat_b;
  // operands set ONCE, outside the loop -- zero loads in the hot loop below.
  mat_a.thread_elements()[0] = (half)1.0h;  mat_a.thread_elements()[1] = (half)1.001h;
  mat_b.thread_elements()[0] = (half)0.5h;  mat_b.thread_elements()[1] = (half)0.502h;

  simdgroup_float8x8 c[{nacc}];
  #pragma unroll
  for (int j = 0; j < {nacc}; j++) {{
    c[j].thread_elements()[0] = 0.0f;
    c[j].thread_elements()[1] = 0.0f;
  }}

  for (int t = 0; t < iters; t++) {{
    #pragma unroll
    for (int j = 0; j < {nacc}; j++) {{
      simdgroup_multiply_accumulate(c[j], mat_a, mat_b, c[j]);
    }}
  }}

  float s = 0.0f;
  #pragma unroll
  for (int j = 0; j < {nacc}; j++) {{
    s += c[j].thread_elements()[0] + c[j].thread_elements()[1];
  }}
  if (s == 1234.5f) out[0] = s;   // keep it live, never taken
}}
"""


def build(nacc: int) -> tuple[MetalProgram, bytes, str]:
  dev = Device[DEVICE]
  src = kernel_src(nacc)
  lib = MetalCompiler().compile(src)
  prog = MetalProgram(dev, "wmma_peak", lib)
  return prog, lib, src


def run_once(prog: MetalProgram, out_buf: Buffer, iters: int, blocks: int, tpb: int) -> float:
  """Returns GPU end-to-end wall time in seconds, host-timed with explicit synchronize()
  before and after (Metal is async -- timing enqueue alone once produced a bogus 63,583
  GFLOPS figure on this device, docs/what-makes-a-token-fast-20260731.md#9.6)."""
  Device[DEVICE].synchronize()
  t0 = time.perf_counter()
  prog(out_buf.get_buf(DEVICE), global_size=(blocks, 1, 1), local_size=(tpb, 1, 1), vals=(iters,), wait=False)
  Device[DEVICE].synchronize()
  t1 = time.perf_counter()
  return t1 - t0


def sweep_grid(nacc: int, tpb: int, iters: int, block_counts: list[int], warmup: int, reps: int) -> list[dict]:
  prog, lib, src = build(nacc)
  out_buf = Buffer(DEVICE, 1, dtypes.float32)
  out_buf.ensure_allocated()
  results = []
  for blocks in block_counts:
    simdgroups = blocks * (tpb // SIMD_WIDTH)
    for _ in range(warmup):
      run_once(prog, out_buf, iters, blocks, tpb)
    times = [run_once(prog, out_buf, iters, blocks, tpb) for _ in range(reps)]
    flop = simdgroups * iters * nacc * FLOP_PER_OP
    gflops = [flop / t / 1e9 for t in times]
    row = {"nacc": nacc, "blocks": blocks, "tpb": tpb, "simdgroups": simdgroups, "iters": iters,
           "times_s": times, "gflops": gflops, "mean_gflops": sum(gflops) / len(gflops),
           "max_gflops": max(gflops), "spread_gflops": max(gflops) - min(gflops)}
    results.append(row)
    print(f"nacc={nacc:2d} blocks={blocks:5d} tpb={tpb} simdgroups={simdgroups:6d} iters={iters} "
          f"-> gflops(per rep)={[round(g,1) for g in gflops]} mean={row['mean_gflops']:.1f} "
          f"max={row['max_gflops']:.1f} spread={row['spread_gflops']:.1f}")
  return results


def sweep_nacc(nacc_list: list[int], blocks: int, tpb: int, iters_for_nacc: dict[int, int],
               warmup: int, reps: int) -> list[dict]:
  results = []
  for nacc in nacc_list:
    iters = iters_for_nacc[nacc]
    prog, lib, src = build(nacc)
    out_buf = Buffer(DEVICE, 1, dtypes.float32)
    out_buf.ensure_allocated()
    simdgroups = blocks * (tpb // SIMD_WIDTH)
    for _ in range(warmup):
      run_once(prog, out_buf, iters, blocks, tpb)
    times = [run_once(prog, out_buf, iters, blocks, tpb) for _ in range(reps)]
    flop = simdgroups * iters * nacc * FLOP_PER_OP
    gflops = [flop / t / 1e9 for t in times]
    row = {"nacc": nacc, "blocks": blocks, "tpb": tpb, "simdgroups": simdgroups, "iters": iters,
           "times_s": times, "gflops": gflops, "mean_gflops": sum(gflops) / len(gflops),
           "max_gflops": max(gflops), "spread_gflops": max(gflops) - min(gflops), "lib_len": len(lib)}
    results.append(row)
    print(f"nacc={nacc:2d} blocks={blocks:5d} tpb={tpb} simdgroups={simdgroups:6d} iters={iters} "
          f"-> gflops(per rep)={[round(g,1) for g in gflops]} mean={row['mean_gflops']:.1f} "
          f"max={row['max_gflops']:.1f} spread={row['spread_gflops']:.1f}")
  return results


def disassemble_check(nacc: int) -> dict:
  """Compile with xcrun metal directly (not MTLCodeGenService) to get a standalone .air we
  can hand to metal-objdump, and confirm the hot loop contains only simdgroup_multiply_accumulate
  -- no loads, no address computation folded in. AIR is pre-register-allocation and upstream
  of Apple's private backend: it can show WHICH instructions the frontend emitted (loads vs
  pure matrix ops) but cannot show final register allocation, instruction scheduling, or
  occupancy -- those live in Apple's undocumented backend past this point."""
  src = kernel_src(nacc)
  with tempfile.TemporaryDirectory() as td:
    p = pathlib.Path(td)
    src_path, air_path = p / "x.metal", p / "x.air"
    src_path.write_text(src)
    r1 = subprocess.run(["xcrun", "metal", "-c", str(src_path), "-o", str(air_path)],
                        capture_output=True, text=True)
    if r1.returncode != 0:
      return {"ok": False, "stage": "compile", "stderr": r1.stderr}
    r2 = subprocess.run(["xcrun", "metal-objdump", "--disassemble", str(air_path)],
                        capture_output=True, text=True)
    if r2.returncode != 0:
      return {"ok": False, "stage": "objdump", "stderr": r2.stderr}
    dis = r2.stdout
    return {"ok": True, "disassembly": dis, "air_len": air_path.stat().st_size}


if __name__ == "__main__":
  print("This module is imported by wmma_peak_metal_run.py; run that to execute the sweep.")
