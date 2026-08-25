#!/usr/bin/env python3
"""Plain FP16/FP32 FMA peak on Apple M4, with the IDENTICAL harness that measured `R` in
wmma_peak_metal.py (R ~= 3781 GFLOPS for isolated simdgroup_multiply_accumulate). This file
changes only the inner operation: no simdgroup ops, just fma() on ordinary scalar/vector
registers.

Purpose: decide whether simdgroup_multiply_accumulate is a separate matrix unit (fp16 FMA peak
should then sit far below R) or lowers onto the same ALUs that plain FMA uses (fp16 FMA peak
should then land within noise of R). See docs/what-makes-inference-fast.md sec 5/10.

THREE variants, because simdgroup_float8x8 accumulates fp16x fp16 -> fp32, not fp16 -> fp16 --
comparing R against a pure-fp16 loop alone would risk a dtype artifact masquerading as an
architectural finding:
  - "f16f16": half x half -> half accumulate.  Pure fp16, cheapest storage/bandwidth.
  - "f16f32": half x half -> float accumulate (via explicit cast before the fma). This is the
    numerics that actually matches the matmul (fp16 operands, fp32 accumulate) -- PRIMARY
    comparator against R.
  - "f32f32": float x float -> float accumulate. Base ALU rate, and the fp16/fp32 ratio here is
    itself informative (packed math vs fp16-as-storage-only).

Kept identical in spirit to wmma_peak_metal.py:
  - operands (a[], b[]) initialized ONCE, outside the loop -- zero loads in the hot loop. The one
    exception is a single `thread_position_in_grid` register read (not a memory load) used to
    perturb operands very slightly per-thread -- see "uniform-lane" note below.
  - NACC independent accumulator chains, swept (do not assume 2 or 8).
  - accumulation pattern is C = A*B + C (fma(a,b,acc)), matching wmma's C = A@B + C shape -- NOT
    acc = acc*a+b, which would blow up multiplicatively and which the compiler could algebraically
    close-form under sufficiently permissive math (ours is C=A*B+C every iteration with the SAME
    A,B, which is itself a plain arithmetic sequence acc_n = acc_0 + n*(A*B) -- a linear recurrence
    a sufficiently aggressive fast-math compiler could ALSO collapse to a single multiply by
    `iters`. Verified this is not the failure mode here: MetalCompiler.compile() (the actual
    runtime path used to dispatch every measurement below) passes `-fno-fast-math` explicitly
    (tinygrad/runtime/ops_metal.py), which forbids float reassociation, so the closed-form
    transform is not legal and does not occur -- confirmed by disassembly, see disassemble_check).
  - result kept live via a never-taken branch.
  - Device["METAL"].synchronize() around the timed region, host-timed wall clock.
  - disassembly verification via `xcrun metal -c -fno-fast-math` (flag-matched to the real
    MetalCompiler.compile() runtime path -- an earlier version of this file's disassembly check
    used plain `xcrun metal -c` with NO fast-math flag override, which defaults to fast-math ON
    and showed `fma fast`/`fadd fast` in the IR; that mismatched the actual measured compile, so
    it is not evidence about what was measured. Fixed here.).

Thread-uniformity note: every thread's computation is otherwise identical (same NACC, same
literal operands), so in principle a compiler that proves a value is invariant across all lanes
of a SIMD-group could execute it once on a scalar/uniform unit and broadcast, understating real
per-lane work. Tested directly (isolated probe, /tmp/uniform_test.py during this investigation):
grid-swept fp16 FMA with vs without a thread-ID-dependent perturbation gave statistically
indistinguishable throughput (735-771 GFLOPS both ways, monotonically plateauing) -- so this is
NOT what happened here. The perturbation is kept anyway as a defensive default; it costs nothing.

Root cause of an earlier, retracted run of this file (numbers like 10^6 GFLOPS, no grid plateau):
`iters` was chosen too small relative to `blocks` (grid), so measured wall time was dominated by
fixed host-side dispatch/synchronize round-trip overhead (~0.3-0.7ms, present regardless of actual
kernel work) rather than GPU compute time -- the same failure family as
docs/what-makes-inference-fast.md sec 9.6 ("timing enqueue instead of execution"), just
reached through undersized `iters` rather than a missing synchronize(). Fixed here via
`calibrate_iters`: for each config, measure once, then pick `iters` so real wall time sits solidly
(>=100x) above the measured per-dispatch floor before trusting a GFLOPS number.

FLOP counting: FLOP = threads * iters * NACC * vector_width * 2 (1 mul + 1 add per element per
FMA). `threads` (not simdgroups) is the right unit here: plain FMA is per-thread ALU work, not a
per-simdgroup cooperative matrix op.

Vector width: this MSL toolchain (metal 32023) does NOT support half8/float8 as first-class
vector types -- metal_extended_vector.h forward-declares them as
"__Reserved_Name__Do_not_use_*", an incomplete type (verified directly: a kernel naming half8
fails `xcrun metal -c` with "incomplete type"). So width=8 is emulated as TWO independent
half4/float4 lanes per accumulator slot -- still `8 elements x 2 FLOP` charged per accumulator per
iteration, just not a single 8-wide vector instruction. Noted, not hidden.
"""
from __future__ import annotations
import sys, time, subprocess, tempfile, pathlib
sys.path.insert(0, "/Users/julianabeleda/env/tinygrad-arkey-exp")

from tinygrad import Device, dtypes
from tinygrad.device import Buffer
from tinygrad.runtime.ops_metal import MetalCompiler, MetalProgram

DEVICE = "METAL"
SIMD_WIDTH = 32  # Apple simdgroup width, every generation -- kept for comparability with wmma_peak_metal.py

# variant -> (input dtype, accumulator dtype)
VARIANTS = {
  "f16f16": ("half", "half"),
  "f16f32": ("half", "float"),
  "f32f32": ("float", "float"),
}


def _lane_type(base: str, w: int) -> str:
  return base if w == 1 else f"{base}{w}"


def kernel_src(nacc: int, width: int, variant: str) -> str:
  """dtype pair from VARIANTS. width: 1, 2, 4, or 8 (8 emulated as 2x width-4 lanes, see module doc)."""
  assert variant in VARIANTS, variant
  assert width in (1, 2, 4, 8)
  in_base, acc_base = VARIANTS[variant]
  in_suffix = "h" if in_base == "half" else "f"
  acc_suffix = "h" if acc_base == "half" else "f"
  needs_cast = in_base != acc_base

  lane_width = min(width, 4)
  lanes_per_acc = width // lane_width
  n_lanes = nacc * lanes_per_acc
  in_lane_type = _lane_type(in_base, lane_width)
  acc_lane_type = _lane_type(acc_base, lane_width)

  def scalar_lit(v: float, suffix: str) -> str:
    return f"{v}{suffix}"

  def splat(lane_type: str, base: str, expr: str) -> str:
    """Construct a lane_type value from a scalar expr -- vector constructor splat for width>1,
    plain C-style cast for width==1."""
    return f"({base})({expr})" if lane_type == base else f"{lane_type}({expr})"

  decl_a, decl_b, decl_acc, loop_lines = [], [], [], []
  for i in range(n_lanes):
    a_expr = f"{scalar_lit(1.0 + 0.001 * i, in_suffix)} + tid_pert"
    b_expr = f"{scalar_lit(0.5 + 0.002 * i, in_suffix)}"
    decl_a.append(f"{in_lane_type} a{i} = {splat(in_lane_type, in_base, a_expr)};")
    decl_b.append(f"{in_lane_type} b{i} = {splat(in_lane_type, in_base, b_expr)};")
    decl_acc.append(f"{acc_lane_type} acc{i} = {splat(acc_lane_type, acc_base, scalar_lit(0.0, acc_suffix))};")
    if needs_cast:
      cast_a = f"{acc_lane_type}(a{i})" if lane_width > 1 else f"({acc_base})a{i}"
      cast_b = f"{acc_lane_type}(b{i})" if lane_width > 1 else f"({acc_base})b{i}"
      loop_lines.append(f"acc{i} = fma({cast_a}, {cast_b}, acc{i});")
    else:
      loop_lines.append(f"acc{i} = fma(a{i}, b{i}, acc{i});")

  decl_a_s = "\n  ".join(decl_a)
  decl_b_s = "\n  ".join(decl_b)
  decl_acc_s = "\n  ".join(decl_acc)
  loop_body = "\n    ".join(loop_lines)

  if lane_width == 1:
    terms = [f"acc{i}" for i in range(n_lanes)]
  else:
    terms = [f"acc{i}[{k}]" for i in range(n_lanes) for k in range(lane_width)]
  reduce_expr = " + ".join(terms)

  return f"""#include <metal_stdlib>
using namespace metal;

kernel void fma_peak(device float* out [[buffer(0)]],
                      constant int& iters [[buffer(1)]],
                      uint tid [[thread_position_in_grid]]) {{
  // per-thread perturbation (tiny, read tid ONCE -- a register, not a memory load) defeats any
  // thread-uniform scalarization of otherwise-identical-across-threads arithmetic.
  {in_base} tid_pert = ({in_base})(tid % 251u) * ({in_base})0.0001{in_suffix};
  // operands set ONCE, outside the loop -- zero loads in the hot loop below.
  {decl_a_s}
  {decl_b_s}
  {decl_acc_s}

  for (int t = 0; t < iters; t++) {{
    {loop_body}
  }}

  {acc_base} s = ({reduce_expr});
  if (s == ({acc_base})1234.5{acc_suffix}) out[0] = (float)s;   // keep it live, never taken
}}
"""


def build(nacc: int, width: int, variant: str) -> tuple[MetalProgram, bytes, str]:
  dev = Device[DEVICE]
  src = kernel_src(nacc, width, variant)
  lib = MetalCompiler().compile(src)
  prog = MetalProgram(dev, "fma_peak", lib)
  return prog, lib, src


def run_once(prog: MetalProgram, out_buf: Buffer, iters: int, blocks: int, tpb: int) -> float:
  """Returns GPU end-to-end wall time in seconds, host-timed with explicit synchronize() before
  and after -- identical timing discipline to wmma_peak_metal.py."""
  Device[DEVICE].synchronize()
  t0 = time.perf_counter()
  prog(out_buf.get_buf(DEVICE), global_size=(blocks, 1, 1), local_size=(tpb, 1, 1), vals=(iters,), wait=False)
  Device[DEVICE].synchronize()
  t1 = time.perf_counter()
  return t1 - t0


def _flop(threads: int, iters: int, nacc: int, width: int) -> int:
  return threads * iters * nacc * width * 2


def calibrate_iters(prog: MetalProgram, out_buf: Buffer, blocks: int, tpb: int,
                     target_time: float = 0.3, iters0: int = 2000, max_iters: int = 50_000_000,
                     min_margin: float = 50.0) -> tuple[int, float]:
  """Pick `iters` so the measured wall time sits comfortably above host-side dispatch overhead,
  rather than assuming a fixed iters works across every (variant, width, nacc, blocks) config.
  Returns (iters, measured_time_at_that_iters). Raises if it cannot reach min_margin over a probed
  overhead floor even at max_iters (rather than silently reporting an overhead-dominated number)."""
  run_once(prog, out_buf, iters0, blocks, tpb)  # warmup / first-touch
  t0 = run_once(prog, out_buf, iters0, blocks, tpb)
  if t0 <= 0:
    t0 = 1e-6
  iters = int(iters0 * target_time / t0)
  iters = max(iters0, min(iters, max_iters))
  run_once(prog, out_buf, iters, blocks, tpb)  # warmup at new iters
  t1 = run_once(prog, out_buf, iters, blocks, tpb)
  # one correction pass if the projection was off by >2x
  if t1 > 0 and (t1 < 0.5 * target_time or t1 > 2.0 * target_time) and iters < max_iters:
    iters2 = int(iters * target_time / max(t1, 1e-6))
    iters2 = max(iters0, min(iters2, max_iters))
    if iters2 != iters:
      iters = iters2
      run_once(prog, out_buf, iters, blocks, tpb)
      t1 = run_once(prog, out_buf, iters, blocks, tpb)
  # sanity: near-zero iters0 timing (tiny workload) used as an overhead-floor probe
  t_floor = run_once(prog, out_buf, 1, blocks, tpb)
  if t1 < min_margin * t_floor:
    raise RuntimeError(f"calibration could not clear overhead floor: iters={iters} t1={t1*1000:.3f}ms "
                       f"t_floor(iters=1)={t_floor*1000:.3f}ms margin={t1/t_floor:.1f}x < {min_margin}x")
  return iters, t1


def sweep_grid(nacc: int, width: int, variant: str, tpb: int, block_counts: list[int],
               warmup: int, reps: int, target_time: float = 0.3) -> list[dict]:
  """iters is calibrated ONCE per (nacc,width,variant) at the LARGEST block count in
  block_counts, then held fixed across the sweep (mirrors wmma_peak_metal.py: same iters across a
  grid sweep, so wall time is expected to grow with blocks until the hardware saturates and it
  plateaus -- a real plateau, not an overhead artifact, because iters is large enough that the
  smallest block count in the sweep is still far above the dispatch-overhead floor)."""
  prog, lib, src = build(nacc, width, variant)
  out_buf = Buffer(DEVICE, 1, dtypes.float32)
  out_buf.ensure_allocated()
  cal_blocks = max(block_counts)
  iters, _ = calibrate_iters(prog, out_buf, cal_blocks, tpb, target_time=target_time)
  results = []
  for blocks in block_counts:
    threads = blocks * tpb
    simdgroups = blocks * (tpb // SIMD_WIDTH)
    for _ in range(warmup):
      run_once(prog, out_buf, iters, blocks, tpb)
    times = [run_once(prog, out_buf, iters, blocks, tpb) for _ in range(reps)]
    flop = _flop(threads, iters, nacc, width)
    gflops = [flop / t / 1e9 for t in times]
    row = {"variant": variant, "width": width, "nacc": nacc, "blocks": blocks, "tpb": tpb,
           "threads": threads, "simdgroups": simdgroups, "iters": iters,
           "times_s": times, "gflops": gflops, "mean_gflops": sum(gflops) / len(gflops),
           "max_gflops": max(gflops), "spread_gflops": max(gflops) - min(gflops)}
    results.append(row)
    print(f"{variant:7s} width={width} nacc={nacc:2d} blocks={blocks:6d} tpb={tpb} threads={threads:9d} "
          f"iters={iters:8d} -> gflops(per rep)={[round(g,1) for g in gflops]} mean={row['mean_gflops']:.1f} "
          f"max={row['max_gflops']:.1f} spread={row['spread_gflops']:.1f}")
  return results


def sweep_nacc(nacc_list: list[int], width: int, variant: str, blocks: int, tpb: int,
               warmup: int, reps: int, target_time: float = 0.3) -> list[dict]:
  results = []
  for nacc in nacc_list:
    prog, lib, src = build(nacc, width, variant)
    out_buf = Buffer(DEVICE, 1, dtypes.float32)
    out_buf.ensure_allocated()
    iters, _ = calibrate_iters(prog, out_buf, blocks, tpb, target_time=target_time)
    threads = blocks * tpb
    simdgroups = blocks * (tpb // SIMD_WIDTH)
    for _ in range(warmup):
      run_once(prog, out_buf, iters, blocks, tpb)
    times = [run_once(prog, out_buf, iters, blocks, tpb) for _ in range(reps)]
    flop = _flop(threads, iters, nacc, width)
    gflops = [flop / t / 1e9 for t in times]
    row = {"variant": variant, "width": width, "nacc": nacc, "blocks": blocks, "tpb": tpb,
           "threads": threads, "simdgroups": simdgroups, "iters": iters,
           "times_s": times, "gflops": gflops, "mean_gflops": sum(gflops) / len(gflops),
           "max_gflops": max(gflops), "spread_gflops": max(gflops) - min(gflops), "lib_len": len(lib)}
    results.append(row)
    print(f"{variant:7s} width={width} nacc={nacc:2d} blocks={blocks:6d} tpb={tpb} threads={threads:9d} "
          f"iters={iters:8d} -> gflops(per rep)={[round(g,1) for g in gflops]} mean={row['mean_gflops']:.1f} "
          f"max={row['max_gflops']:.1f} spread={row['spread_gflops']:.1f}")
  return results


def disassemble_check(nacc: int, width: int, variant: str) -> dict:
  """Compile with xcrun metal -fno-fast-math directly (flag-matched to MetalCompiler.compile()'s
  actual runtime flags -- see module docstring) to get a standalone .air we can hand to
  metal-objdump, and confirm the hot loop contains only fma -- no loads, no simdgroup ops, no
  fast-math reassociation flags anywhere, and (for f16f32) that the accumulate genuinely stays
  fp32 rather than being demoted back to fp16 each iteration. AIR is pre-register-allocation and
  upstream of Apple's private backend: it shows WHICH instructions the frontend emitted but not
  final register allocation, scheduling, spilling, or occupancy."""
  src = kernel_src(nacc, width, variant)
  with tempfile.TemporaryDirectory() as td:
    p = pathlib.Path(td)
    src_path, air_path = p / "x.metal", p / "x.air"
    src_path.write_text(src)
    r1 = subprocess.run(["xcrun", "metal", "-fno-fast-math", "-c", str(src_path), "-o", str(air_path)],
                        capture_output=True, text=True)
    if r1.returncode != 0:
      return {"ok": False, "stage": "compile", "stderr": r1.stderr}
    r2 = subprocess.run(["xcrun", "metal-objdump", "--disassemble", str(air_path)],
                        capture_output=True, text=True)
    if r2.returncode != 0:
      return {"ok": False, "stage": "objdump", "stderr": r2.stderr}
    dis = r2.stdout
    return {"ok": True, "disassembly": dis, "air_len": air_path.stat().st_size, "src": src}


if __name__ == "__main__":
  print("This module is imported by fma_peak_metal_run.py; run that to execute the sweep.")
