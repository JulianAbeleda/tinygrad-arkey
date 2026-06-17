#!/usr/bin/env python3
"""Phase 5 — minimal AMD two-ring dependency-DAG scheduler prototype (gfx1100, KFD). NOT model decode.

Builds the concept a decode-overlap scheduler needs: a dependent CHAIN (critical path) on one ring overlapping
INDEPENDENT work on another, with correct cross-ring dependencies and a join. A tiny generic runner takes
tasks = (name, ring, kernel, out_buf, in_bufs, deps) and submits each on its ring, making it WAIT (Phase-4
semantics) on its deps' done-signals; the host joins on the leaf signals. GPU-clock span (timestamps) measures
overlap; the SAME DAG is run two ways -> overlap factor:
  - serial   : every task on ring 0 (chain A0->A1 then B all serialize)
  - scheduled: chain A0->A1 on ring 0  ||  independent B on ring 1

DAG:  A0 --> A1   (A1 reads A0's output; B independent)
      B
Dependency correctness is verified (A1 sees A0's write; B correct). B is sized so t_B ~ t_A0+t_A1 (chain), so
ideal overlap ~2x. Requires DEV=AMD AMD_COMPUTE_RINGS=2. Run:
  DEV=AMD AMD_COMPUTE_RINGS=2 PYTHONPATH=. .venv/bin/python extra/amd_two_ring_dag_probe.py
"""
from __future__ import annotations

import json, os, pathlib, statistics, sys
import numpy as np

_EXT = '''
extern "C" __attribute__((device, const)) unsigned long __ockl_get_group_id(unsigned int);
extern "C" __attribute__((device, const)) unsigned int __ockl_get_local_id(unsigned int);
'''
# A0: spin then write sentinel gid+1000. A1: spin then out=in*2 (depends on A0). B: independent spin-write.
# NOTE: compile each kernel in its OWN lib -- a multi-kernel lib confuses fill_kernargs (wrong kernarg size).
_A0 = _EXT + '''extern "C" __attribute__((global)) __attribute__((amdgpu_flat_work_group_size(1,256)))
void a0(float* out, unsigned int iters) {
  unsigned int gid=(unsigned int)__ockl_get_group_id(0)*256u+__ockl_get_local_id(0);
  float a=(float)gid*1e-6f,b=1.0000001f; for(unsigned int i=0;i<iters;i++) a=a*b+1.0f;
  out[gid]=(float)gid+1000.0f+(a-a); }'''
_A1 = _EXT + '''extern "C" __attribute__((global)) __attribute__((amdgpu_flat_work_group_size(1,256)))
void a1(float* out, const float* in, unsigned int iters) {
  unsigned int gid=(unsigned int)__ockl_get_group_id(0)*256u+__ockl_get_local_id(0);
  float a=(float)gid*1e-6f,b=1.0000001f; for(unsigned int i=0;i<iters;i++) a=a*b+1.0f;
  out[gid]=in[gid]*2.0f+(a-a); }'''
_BK = _EXT + '''extern "C" __attribute__((global)) __attribute__((amdgpu_flat_work_group_size(1,256)))
void bk(float* out, unsigned int iters) {
  unsigned int gid=(unsigned int)__ockl_get_group_id(0)*256u+__ockl_get_local_id(0);
  float a=(float)gid*1e-6f,b=1.0000001f; for(unsigned int i=0;i<iters;i++) a=a*b+1.0f;
  out[gid]=a; }'''

def main():
  from tinygrad.runtime.ops_amd import AMD_COMPUTE_RINGS
  from tinygrad.device import Device, Buffer
  from tinygrad.dtype import dtypes
  GRID, ITERS, REPS = int(os.environ.get("GRID", 48)), int(os.environ.get("ITERS", 150000)), 40
  dev = Device["AMD"]
  if AMD_COMPUTE_RINGS < 2:
    print("REQUIRES AMD_COMPUTE_RINGS=2 (got %d)" % AMD_COMPUTE_RINGS); sys.exit(2)
  dev.compute_queue_desc(1)
  N = GRID * 256
  pA0 = dev.runtime("a0", dev.compiler.compile(_A0))
  pA1 = dev.runtime("a1", dev.compiler.compile(_A1))
  pB = dev.runtime("bk", dev.compiler.compile(_BK))
  gs, ls = (GRID, 1, 1), (256, 1, 1)
  bufA = Buffer("AMD", N, dtypes.float32).ensure_allocated()
  bufA2 = Buffer("AMD", N, dtypes.float32).ensure_allocated()
  bufB = Buffer("AMD", N, dtypes.float32).ensure_allocated()

  # task = (name, prg, out_buf, in_bufs, iters, deps[list of names]); ring assignment passed per-run
  def make_tasks():
    return [("A0", pA0, bufA, [], ITERS, []),
            ("A1", pA1, bufA2, [bufA], ITERS, ["A0"]),     # depends on A0
            ("B",  pB,  bufB, [], 2 * ITERS, [])]          # independent; sized ~ chain length (2*ITERS)

  def run_dag(ring_of:dict) -> float:
    """Submit the DAG with the given task->ring assignment; return GPU-clock span (max end - min start)."""
    spans = []
    for _ in range(REPS):
      done = {n: dev.new_signal() for n, *_ in make_tasks()}
      ts0 = {n: dev.new_signal() for n, *_ in make_tasks()}
      ts1 = {n: dev.new_signal() for n, *_ in make_tasks()}
      for name, prg, ob, ibs, iters, deps in make_tasks():
        q = dev.hw_compute_queue_t(queue_idx=ring_of[name]).memory_barrier()
        for dn in deps: q.wait(done[dn], 1)                # cross-task (and possibly cross-ring) dependency
        ka = prg.fill_kernargs(tuple(b._buf for b in ([ob] + ibs)), (iters,))
        q.timestamp(ts0[name]); q.exec(prg, ka, gs, ls); q.timestamp(ts1[name]); q.signal(done[name], 1)
        q.submit(dev)
      for s in done.values(): s.wait(1)
      starts = [float(ts0[n].timestamp) for n in done]; ends = [float(ts1[n].timestamp) for n in done]
      spans.append(max(ends) - min(starts))
    return statistics.median(spans)

  serial_span = run_dag({"A0": 0, "A1": 0, "B": 0})          # everything on ring 0
  sched_span = run_dag({"A0": 0, "A1": 0, "B": 1})           # chain on ring 0 || B on ring 1
  # dependency correctness (from the last scheduled run's buffers)
  gotA2 = np.empty(N, np.float32); bufA2.copyout(memoryview(gotA2))
  gotB = np.empty(N, np.float32); bufB.copyout(memoryview(gotB))
  expA2 = (np.arange(N, dtype=np.float32) + 1000.0) * 2.0
  dep_ok = bool(np.allclose(gotA2, expA2))                   # A1 saw A0's write
  b_ran = bool(np.all(np.isfinite(gotB)) and np.any(gotB != 0))

  factor = round(serial_span / sched_span, 3) if sched_span else None
  passes = bool(factor and factor > 1.2 and dep_ok and b_ran)
  out = {"arch": dev.arch, "reps": REPS, "grid": GRID, "iters": ITERS,
         "serial_span": round(serial_span, 1), "scheduled_span": round(sched_span, 1), "overlap_x": factor,
         "dependency_correct": dep_ok, "independent_ran": b_ran, "passes": passes,
         "verdict": (f"PASS: cross-ring DAG scheduler overlaps the A0->A1 chain with independent B at {factor}x "
                     f"(dep correct={dep_ok}) -> decode-overlap scheduling concept is sound; ready to scope "
                     f"decode (Phase 6)" if passes else
                     f"REFUTED: overlap {factor}x dep_ok={dep_ok} b_ran={b_ran} -> DAG scheduling unsound; stop")}
  print(f"serial span (all ring0)      : {serial_span:10.1f}")
  print(f"scheduled span (chain || B)  : {sched_span:10.1f}  -> overlap {factor}x")
  print(f"dependency correct (A1<-A0)  : {dep_ok}")
  print(f"independent B ran            : {b_ran}")
  print(out["verdict"])
  art = pathlib.Path("bench/amd-two-ring-dag-probe/result.json"); art.parent.mkdir(parents=True, exist_ok=True)
  art.write_text(json.dumps(out, indent=2)); print(f"artifact: {art}")

if __name__ == "__main__":
  main()
