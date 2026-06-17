#!/usr/bin/env python3
"""Phase 4 — AMD cross-ring dependency / wait semantics (gfx1100, KFD).

Phase 3 proved two rings run INDEPENDENT work concurrently (2.00x). A scheduler also needs ORDERING across
rings. This probe proves explicit signal/wait orders work across rings:
  - ring 0 runs a SLOW producer (spin, then write a sentinel to buf) and signals sigA;
  - ring 1 WAITS on sigA, then reads buf into out and signals sigB; host waits sigB and checks out == sentinel.
If cross-ring wait works, the consumer always sees the producer's write. A no-wait CONTROL (consumer doesn't
wait) should frequently read stale data (buf pre-initialized to a different value) -> proves the wait matters.
Both directions (0->1 and 1->0), plus a copy-queue-waits-on-compute case. GPU memory signals; many reps.

Requires: DEV=AMD AMD_COMPUTE_RINGS=2. Run:
  DEV=AMD AMD_COMPUTE_RINGS=2 PYTHONPATH=. .venv/bin/python extra/amd_two_ring_dependency_probe.py
"""
from __future__ import annotations

import json, os, pathlib, sys
import numpy as np

_EXT = '''
extern "C" __attribute__((device, const)) unsigned long __ockl_get_group_id(unsigned int);
extern "C" __attribute__((device, const)) unsigned int __ockl_get_local_id(unsigned int);
'''
def producer_src() -> str:  # spin `iters`, then write sentinel (gid+1000) -> a SLOW write the consumer must wait for
  return _EXT + '''
extern "C" __attribute__((global)) __attribute__((amdgpu_flat_work_group_size(1,256)))
void producer(float* buf, unsigned int iters) {
  unsigned int gid = (unsigned int)__ockl_get_group_id(0)*256u + __ockl_get_local_id(0);
  float a = (float)gid * 1e-6f, b = 1.0000001f;
  for (unsigned int i = 0; i < iters; i++) a = a*b + 1.0f;
  buf[gid] = (float)gid + 1000.0f + (a - a);  // (a-a)==0 keeps the spin from being optimized away
}'''
def consumer_src() -> str:  # read buf -> out
  return _EXT + '''
extern "C" __attribute__((global)) __attribute__((amdgpu_flat_work_group_size(1,256)))
void consumer(float* out, const float* buf) {
  unsigned int gid = (unsigned int)__ockl_get_group_id(0)*256u + __ockl_get_local_id(0);
  out[gid] = buf[gid];
}'''

def main():
  from tinygrad.runtime.ops_amd import AMD_COMPUTE_RINGS
  from tinygrad.device import Device, Buffer
  from tinygrad.dtype import dtypes
  GRID, ITERS, REPS = int(os.environ.get("GRID", 48)), int(os.environ.get("ITERS", 300000)), 40
  dev = Device["AMD"]
  if AMD_COMPUTE_RINGS < 2:
    print("REQUIRES AMD_COMPUTE_RINGS=2 (got %d)" % AMD_COMPUTE_RINGS); sys.exit(2)
  dev.compute_queue_desc(1)
  N = GRID * 256
  pP = dev.runtime("producer", dev.compiler.compile(producer_src()))
  pC = dev.runtime("consumer", dev.compiler.compile(consumer_src()))
  gs, ls = (GRID, 1, 1), (256, 1, 1)
  expect = (np.arange(N, dtype=np.float32) + 1000.0)
  stale = np.full(N, -1.0, np.float32)

  def cross_ring(prod_ring, cons_ring, do_wait:bool) -> int:
    """producer on prod_ring (slow write), consumer on cons_ring (optionally waits). Returns #reps consumer
    saw the fresh sentinel for ALL lanes."""
    ok = 0
    buf = Buffer("AMD", N, dtypes.float32).ensure_allocated()
    out = Buffer("AMD", N, dtypes.float32).ensure_allocated()
    for _ in range(REPS):
      buf.copyin(memoryview(stale.copy())); out.copyin(memoryview(stale.copy()))  # poison so a race is visible
      sigA, dC = dev.new_signal(), dev.new_signal()
      qp = dev.hw_compute_queue_t(queue_idx=prod_ring).memory_barrier()
      qp.exec(pP, pP.fill_kernargs((buf._buf,), (ITERS,)), gs, ls); qp.signal(sigA, 1)
      qc = dev.hw_compute_queue_t(queue_idx=cons_ring).memory_barrier()
      if do_wait: qc.wait(sigA, 1)
      qc.exec(pC, pC.fill_kernargs((out._buf, buf._buf), ()), gs, ls); qc.signal(dC, 1)
      qp.submit(dev); qc.submit(dev)
      dC.wait(1); sigA.wait(1)
      got = np.empty(N, np.float32); out.copyout(memoryview(got))
      if np.allclose(got, expect): ok += 1
    return ok

  def copy_after_compute() -> int:
    """compute on ring 0 writes buf (slow); SDMA copy queue WAITS on the compute signal, copies buf->dst."""
    ok = 0
    buf = Buffer("AMD", N, dtypes.float32).ensure_allocated()
    dst = Buffer("AMD", N, dtypes.float32).ensure_allocated()
    for _ in range(REPS):
      buf.copyin(memoryview(stale.copy())); dst.copyin(memoryview(stale.copy()))
      sigA, dCp = dev.new_signal(), dev.new_signal()
      qp = dev.hw_compute_queue_t(queue_idx=0).memory_barrier()
      qp.exec(pP, pP.fill_kernargs((buf._buf,), (ITERS,)), gs, ls); qp.signal(sigA, 1)
      qcopy = dev.hw_copy_queue_t().wait(sigA, 1)
      qcopy.copy(dst._buf, buf._buf, N * 4); qcopy.signal(dCp, 1)
      qp.submit(dev); qcopy.submit(dev)
      dCp.wait(1); sigA.wait(1)
      got = np.empty(N, np.float32); dst.copyout(memoryview(got))
      if np.allclose(got, expect): ok += 1
    return ok

  fwd = cross_ring(0, 1, do_wait=True)
  rev = cross_ring(1, 0, do_wait=True)
  ctrl = cross_ring(0, 1, do_wait=False)   # no-wait control: should usually read stale
  cp = copy_after_compute()
  passes = bool(fwd == REPS and rev == REPS and cp == REPS and ctrl < REPS)
  out = {"arch": dev.arch, "reps": REPS, "grid": GRID, "iters": ITERS,
         "fwd_0to1_correct": fwd, "rev_1to0_correct": rev, "copyq_after_compute_correct": cp,
         "nowait_control_correct": ctrl, "passes": passes,
         "verdict": (f"PASS: cross-ring waits order correctly both ways ({fwd}/{rev}/{REPS}), copy-queue waits on "
                     f"compute ({cp}/{REPS}); no-wait control raced ({ctrl}/{REPS}) -> wait/signal is real. "
                     f"DAG scheduling (Phase 5) is sound" if passes else
                     f"REFUTED: fwd={fwd} rev={rev} copyq={cp} nowait={ctrl} (want fwd=rev=copyq={REPS}, "
                     f"nowait<{REPS}) -> cross-ring ordering broken; stop")}
  print(f"fwd  ring0->ring1 (wait): {fwd}/{REPS} correct")
  print(f"rev  ring1->ring0 (wait): {rev}/{REPS} correct")
  print(f"copy-queue waits compute: {cp}/{REPS} correct")
  print(f"no-wait control (race)  : {ctrl}/{REPS} correct (want < {REPS})")
  print(out["verdict"])
  art = pathlib.Path("bench/amd-two-ring-dependency-probe/result.json"); art.parent.mkdir(parents=True, exist_ok=True)
  art.write_text(json.dumps(out, indent=2)); print(f"artifact: {art}")

if __name__ == "__main__":
  main()
