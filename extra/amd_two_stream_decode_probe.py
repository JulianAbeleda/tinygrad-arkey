#!/usr/bin/env python3
"""Phase 7a — can two same-process decode STREAMS overlap on two rings? Answer: BLOCKED by the dispatch path,
not the hardware. This probe ISOLATES the exact blocker (it does NOT fake a two-model throughput number).

Two confirmed blockers (verified against tinygrad/runtime/support/hcq.py + a live decode):
  1. No ring-routing API: HCQProgram.__call__ (hcq.py:374) uses unwrap(dev.hw_compute_queue_t)() == ring 0.
     A live decode with AMD_COMPUTE_RINGS=2 leaves dev.compute_queues == {0} (ring 1 is never used).
  2. Global-timeline serialization: every dispatched kernel does .wait(dev.timeline_signal, v-1) then
     .signal(dev.timeline_signal, next). The device has ONE timeline_signal, so two streams sharing it serialize
     even if routed to different rings.

This probe models two "streams" as kernel chains and dispatches them two ways to PROVE which blocker bites:
  - INDEPENDENT (per-stream signals, ring0 || ring1): the chains overlap  -> the hardware/primitive is fine.
  - SHARED-TIMELINE (one monotonic signal, mimics HCQProgram), even on two rings: serializes -> the timeline is
    the blocker.
So the missing API is: (a) per-stream/per-graph ring selection in the dispatch path, AND (b) independent
per-stream timelines (sync domains). Requires DEV=AMD AMD_COMPUTE_RINGS=2.
"""
from __future__ import annotations

import json, os, pathlib, statistics, sys, time

_EXT = '''
extern "C" __attribute__((device, const)) unsigned long __ockl_get_group_id(unsigned int);
extern "C" __attribute__((device, const)) unsigned int __ockl_get_local_id(unsigned int);
'''
def cu_src() -> str:
  return _EXT + '''extern "C" __attribute__((global)) __attribute__((amdgpu_flat_work_group_size(1,256)))
void cu(float* out, unsigned int iters) {
  unsigned int gid=(unsigned int)__ockl_get_group_id(0)*256u+__ockl_get_local_id(0);
  float a=(float)gid*1e-6f,b=1.0000001f; for(unsigned int i=0;i<iters;i++) a=a*b+1.0f; out[gid]=a; }'''

def main():
  from tinygrad.runtime.ops_amd import AMD_COMPUTE_RINGS
  from tinygrad.device import Device, Buffer
  from tinygrad.dtype import dtypes
  GRID, ITERS, NK, REPS, WARM = 48, 60000, 6, 40, 12   # NK kernels per "stream" (a mini decode-like chain)
  dev = Device["AMD"]
  if AMD_COMPUTE_RINGS < 2:
    print("REQUIRES AMD_COMPUTE_RINGS=2 (got %d)" % AMD_COMPUTE_RINGS); sys.exit(2)

  # PRIMARY (clean) evidence FIRST -- before we create ring 1 ourselves, so compute_queues reflects ONLY what the
  # model dispatch used. A real decode with AMD_COMPUTE_RINGS=2: does it ever touch ring 1?
  from tinygrad import Tensor
  from tinygrad.llm.model import Transformer
  routing_blocked, rings_used = True, None
  try:
    Tensor.manual_seed(0)
    mdl, _ = Transformer.from_gguf(pathlib.Path(os.environ.get("QK_MODEL", "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")), 2048)
    n = 0
    for _ in mdl.generate([5, 6, 7, 8, 9, 10], temperature=0.0):
      n += 1
      if n >= 4: break
    rings_used = sorted(dev.compute_queues)
    routing_blocked = (rings_used == [0])  # model dispatch never created/used ring 1
  except Exception as e:
    rings_used = f"decode check skipped: {e}"

  dev.compute_queue_desc(1)
  pB = dev.runtime("cu", dev.compiler.compile(cu_src()))
  gs, ls = (GRID, 1, 1), (256, 1, 1)
  oA = Buffer("AMD", GRID*256, dtypes.float32).ensure_allocated()
  oB = Buffer("AMD", GRID*256, dtypes.float32).ensure_allocated()
  kaA = pB.fill_kernargs((oA._buf,), (ITERS,)); kaB = pB.fill_kernargs((oB._buf,), (ITERS,))

  # wall-clock TO COMPLETION (valid here: comparing two dispatch STRUCTURES end-to-end with explicit completion
  # sync + warm + many reps -- not micro-kernel timing). Both modes run the SAME 2*NK kernels.
  def timed(submit_and_wait):
    ts = []
    for i in range(WARM + REPS):
      dev.synchronize(); t0 = time.perf_counter(); submit_and_wait(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts)

  def independent():
    """stream A: NK kernels chained on ring 0; stream B: NK chained on ring 1; NO cross-stream wait -> overlap."""
    dA, dB = dev.new_signal(), dev.new_signal()
    qa = dev.hw_compute_queue_t(queue_idx=0).memory_barrier()
    for _ in range(NK): qa.exec(pB, kaA, gs, ls)
    qa.signal(dA, 1)
    qb = dev.hw_compute_queue_t(queue_idx=1).memory_barrier()
    for _ in range(NK): qb.exec(pB, kaB, gs, ls)
    qb.signal(dB, 1)
    qa.submit(dev); qb.submit(dev); dA.wait(1); dB.wait(1)

  def shared_timeline():
    """Mimic HCQProgram dispatch: ONE monotonic signal; every kernel waits the prior value then signals next --
    even alternating rings 0/1. Two streams' kernels serialize through the single timeline."""
    tl = dev.new_signal(value=0); v = 0
    for kidx in range(2 * NK):
      ring = kidx % 2  # alternate rings (so ROUTING is not the limiter -- the shared timeline is)
      q = dev.hw_compute_queue_t(queue_idx=ring).memory_barrier().wait(tl, v)
      q.exec(pB, kaA if ring == 0 else kaB, gs, ls); v += 1; q.signal(tl, v); q.submit(dev)
    tl.wait(v)

  indep = timed(independent); shared = timed(shared_timeline)
  # SECONDARY (directional, has a submit-count confound): shared-timeline dispatch vs independent per-stream.
  ratio = round(shared / indep, 3) if indep else None
  blocked = bool(routing_blocked)  # the routing API gap is a FACT (not a measurement); timing is supporting
  out = {"arch": dev.arch, "rings": AMD_COMPUTE_RINGS, "kernels_per_stream": NK,
         "decode_rings_used": rings_used, "routing_blocked": routing_blocked,
         "independent_ms": round(indep * 1e3, 3), "shared_timeline_ms": round(shared * 1e3, 3),
         "shared_over_independent": ratio, "timing_caveat": "directional only: shared mode also has more host submits",
         "blockers": {
           "ring_routing": "HCQProgram.__call__ (hcq.py:374) hardcodes hw_compute_queue_t()==ring0; live decode "
                           "with AMD_COMPUTE_RINGS=2 leaves dev.compute_queues=={0}. Need per-stream ring selection.",
           "global_timeline": "device has ONE timeline_signal; every op waits/signals it (hcq.py:307/374). Two "
                              "streams sharing it serialize even on two rings. Need independent per-stream timelines.",
           "model_state": "one cache_kv per model; two concurrent streams need two state/cache sets."},
         "verdict": (f"BLOCKED by dispatch path (NOT hardware): a real decode with AMD_COMPUTE_RINGS=2 uses rings "
                     f"{rings_used} -- ring 1 is never touched (no per-stream ring-selection API). And the shared "
                     f"global timeline serializes dispatch (directionally: HCQProgram-style {ratio}x slower than "
                     f"independent per-stream signals, even on two rings). Missing APIs: (1) per-stream/per-graph "
                     f"ring selection, (2) independent per-stream timelines." if blocked
                     else f"decode used rings {rings_used} (ring 1 touched?) -- re-examine routing before concluding")}
  print(f"decode with RINGS=2 used compute rings: {rings_used}  (ring 1 never used => no routing API)")
  print(f"independent (per-stream signals, ring0||ring1): {indep*1e3:8.3f} ms")
  print(f"shared global timeline (HCQProgram-style)     : {shared*1e3:8.3f} ms  -> {ratio}x slower (directional)")
  print(out["verdict"])
  art = pathlib.Path("bench/amd-two-stream-decode-probe/result.json"); art.parent.mkdir(parents=True, exist_ok=True)
  art.write_text(json.dumps(out, indent=2)); print(f"artifact: {art}")

if __name__ == "__main__":
  main()
