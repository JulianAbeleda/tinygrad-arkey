#!/usr/bin/env python3
"""CPU-side overhead measurement for ALLOC_TRACE (tinygrad/device.py). Pure Python timing of the record
path -- NO GPU workload, NO device opened. Run this to get real numbers for a given machine/Python; the
numbers quoted in the handoff doc were measured on the dev box and will vary by CPU/Python build.

Usage:
    python3 extra/gpu_fault_analysis/bench_alloc_trace.py [--n 200000]
"""
import argparse, sys, types, pathlib, time

def _import_device_standalone():
  if 'tinygrad.device' in sys.modules: return sys.modules['tinygrad.device']
  root = pathlib.Path(__file__).resolve().parents[2] / "tinygrad"
  if 'tinygrad' not in sys.modules:
    pkg = types.ModuleType('tinygrad')
    pkg.__path__ = [str(root)]
    pkg.__package__ = 'tinygrad'
    sys.modules['tinygrad'] = pkg
  import tinygrad.helpers, tinygrad.dtype, tinygrad.device  # noqa: F401
  return tinygrad.device

D = _import_device_standalone()

class _FakeBuf:
  def __init__(self, va, size): self.va_addr, self.size = va, size

def _reset():
  D._at_alloc_ring = None; D._at_dispatch_ring = None
  D._at_alloc_count[0] = 0; D._at_dispatch_count[0] = 0; D._at_seq[0] = 0
  D._at_device_ids.clear(); D._at_device_names.clear()
  D._at_kernel_ids.clear(); D._at_kernel_names.clear()

def _time_it(fn, n) -> float:
  t0 = time.perf_counter()
  for _ in range(n): fn()
  return (time.perf_counter() - t0) / n * 1e9  # ns/call

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--n", type=int, default=200_000)
  args = ap.parse_args()
  n = args.n
  bufs2 = [_FakeBuf(0x7f0000000000, 4096), _FakeBuf(0x7f0000001000, 8192)]

  print(f"n={n} per measurement, tinygrad/device.py ALLOC_TRACE ring, pure CPU, no GPU/device involved\n")

  # --- OFF (default, what every real run pays until someone sets the env var) ---
  D.ALLOC_TRACE.value = 0
  _reset()
  off_alloc = _time_it(lambda: D.alloc_trace_record_alloc("AMD", 0x1000, 0x1000, 0x1000), n)
  off_free  = _time_it(lambda: D.alloc_trace_record_free(-1), n)
  off_disp  = _time_it(lambda: D.alloc_trace_record_dispatch("AMD", "k", (1,1,1), (1,1,1), bufs2, 1), n)
  print("ALLOC_TRACE=0 (shipped default):")
  print(f"  alloc_trace_record_alloc:    {off_alloc:7.1f} ns/call")
  print(f"  alloc_trace_record_free:     {off_free:7.1f} ns/call")
  print(f"  alloc_trace_record_dispatch: {off_disp:7.1f} ns/call")

  # --- ON ---
  D.ALLOC_TRACE.value = 1
  _reset()
  on_alloc = _time_it(lambda: D.alloc_trace_record_alloc("AMD", 0x1000, 0x1000, 0x1000), n)
  _reset()
  ids = iter(range(10**9))
  def _alloc_then_free():
    aid = D.alloc_trace_record_alloc("AMD", 0x1000, 0x1000, 0x1000)
    D.alloc_trace_record_free(aid)
  on_alloc_free = _time_it(_alloc_then_free, n) / 2
  _reset()
  on_disp2 = _time_it(lambda: D.alloc_trace_record_dispatch("AMD", "k", (1,1,1), (1,1,1), bufs2, 1), n)
  _reset()
  bufs8 = [_FakeBuf(i, 4096) for i in range(D.ALLOC_TRACE_MAX_ARGS)]
  on_disp8 = _time_it(lambda: D.alloc_trace_record_dispatch("AMD", "k", (1,1,1), (1,1,1), bufs8, 1), n)
  print(f"\nALLOC_TRACE=1:")
  print(f"  alloc_trace_record_alloc (cold, 1 unique device):        {on_alloc:7.1f} ns/call")
  print(f"  alloc_trace_record_alloc + record_free pair (steady):    {on_alloc_free:7.1f} ns/call (avg of the two)")
  print(f"  alloc_trace_record_dispatch, 2 bufs:                     {on_disp2:7.1f} ns/call")
  print(f"  alloc_trace_record_dispatch, {D.ALLOC_TRACE_MAX_ARGS} bufs (max):{' '*13}{on_disp8:7.1f} ns/call")

  dump_t0 = time.perf_counter()
  path = D.alloc_trace_dump("/tmp/alloc_trace_bench_dump.json")
  dump_ms = (time.perf_counter() - dump_t0) * 1e3
  print(f"\n  alloc_trace_dump() with {n} allocs + {n} dispatches recorded: {dump_ms:.1f} ms (one-shot, at exit -- not hot path)")
  print(f"  dump written to {path}")

  D.ALLOC_TRACE.value = 0
  _reset()

  print("\nFor scale: a single AMD KFD dispatch submission (queue write + doorbell ring) is on the order of")
  print("several hundred ns to a few us; ALLOC_TRACE's added cost per dispatch is a small fraction of that,")
  print("and its added cost per allocation is irrelevant next to the ioctl(AMDKFD_IOC_ALLOC_MEMORY_OF_GPU)")
  print("+ mmap() a real allocation already pays (tens of us).")

if __name__ == "__main__":
  main()
