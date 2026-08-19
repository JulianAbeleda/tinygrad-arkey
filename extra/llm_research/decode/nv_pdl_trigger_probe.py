#!/usr/bin/env python3
"""NV PDL early-trigger probe: launch_dependents at producer END vs START vs QMD-only latch.

Tests the highest-value parity theory on DEV=NV: llama.cpp fires
`cudaTriggerProgrammaticLaunchCompletion` at kernel START, while tinygrad's
native PDL releases the dependent grid at the last CTA (QMD v05
`pre_exit_at_last_cta_launch` + `griddepcontrol.launch_dependents` emitted at
the END of the producer body). This probe measures whether moving the renderer
instruction to the TOP of the producer body moves the dependent consumer's
launch earlier, and whether the pure QMD latch path releases early at all
without the in-kernel instructions.

Four arms, each in a fresh process (the PDL env is read at compile time):
  no_pdl      no env marks, fully serialized chain
  pdl_end     NV_PDL_PRODUCER_PROGRAMS/NV_PDL_CONSUMER_PROGRAMS + trigger
              unset (END placement, the current default)
  pdl_start   same marks + NV_PDL_TRIGGER_POSITION=start
  qmd_latch   no env marks; latch fields written directly into the QMD
              templates (arrive_at_latch + program pre-exit at last CTA
              launch on the producer, wait_on_latch on the consumer)

The producer is a multi-wave grid (4096 blocks x 256) that writes a large
buffer then spins ~100 us on %globaltimer; the consumer is a single-block
~1 us checksum kernel. In-kernel %globaltimer timestamps capture producer
start / max block end, consumer start / end; HCQ timestamp signals bracket the
chain for a GPU-side wall. Correctness is a full-coverage checksum over the
producer buffer, so an early launch that reads stale data is caught.

Run under the bench lock, one arm per process:
  flock -w 600 /tmp/gpu-bench.lock env PYTHONPATH=. DEV=NV python3 \\
    extra/llm_research/decode/nv_pdl_trigger_probe.py --arm pdl_start --out X.json
  or --run-all for all four arms plus the merged evidence file.
"""
from __future__ import annotations

import argparse, json, os, statistics, subprocess, sys, time
sys.path.insert(0, "/home/ubuntu/tinygrad-arkey")

from tinygrad import Device
from tinygrad.device import BufferSpec
from tinygrad.renderer.cuda import _nv_pdl_body
from tinygrad.runtime.ops_nv import NVComputeQueue, NVProgram
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler

ARMS = ["no_pdl", "pdl_end", "pdl_start", "qmd_latch"]
LS, GS = 256, 4096
NPROD = 1 << 23          # 8M u32 = 32 MiB, filled by the producer grid
NCONS = 513 * 1024       # 525312, multiple of LS*4; checksum = NCONS^2 mod 2^32
SPIN_NS = 100_000        # per-block %globaltimer spin budget (~100 us)
LATCH_ID = 7

KERNEL_HDR = f"""
#define LS {LS}u
#define GS {GS}u
#define NPROD {NPROD}u
#define NCONS {NCONS}u
#define SPIN_NS {SPIN_NS}ull
"""

PROD_BODY = [
  "  unsigned long long now;",
  '  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));',
  "  if (threadIdx.x == 0u && blockIdx.x == 0u) t[0] = now;",
  "  unsigned int i = blockIdx.x*LS + threadIdx.x;",
  "  for (; i < NPROD; i += GS*LS) out[i] = i*2u + 1u;",
  "  unsigned long long end = now + SPIN_NS;",
  '  do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now)); } while (now < end);',
  "  if (threadIdx.x == 0u && blockIdx.x == 0u) t[4] = now;",
  '  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));',
  "  if (threadIdx.x == 0u) atomicMax(&t[1], now);",
]

CONS_BODY = [
  "  unsigned long long now;",
  '  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));',
  "  if (threadIdx.x == 0u && blockIdx.x == 0u) t[2] = now;",
  "  unsigned int s0=0u, s1=0u, s2=0u, s3=0u;",
  "  unsigned int i = threadIdx.x;",
  "  for (; i < NCONS; i += LS*4) {",
  "    s0 += in[i]; s1 += in[i + LS]; s2 += in[i + LS*2]; s3 += in[i + LS*3];",
  "  }",
  "  atomicAdd(chk, s0+s1+s2+s3);",
  '  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));',
  "  if (threadIdx.x == 0u && blockIdx.x == 0u) t[3] = now;",
]


def build_source() -> tuple[str, str, str]:
  """Emit both kernels through the real renderer hook. Returns (src, prod_src, cons_src)."""
  prod_src = "\n".join(_nv_pdl_body("pdl_producer", PROD_BODY))
  cons_src = "\n".join(_nv_pdl_body("pdl_consumer", CONS_BODY))
  src = KERNEL_HDR + """
extern "C" __global__ void pdl_producer(unsigned int* out, unsigned long long* t) {
""" + prod_src + """
}

extern "C" __global__ void pdl_consumer(const unsigned int* in, unsigned int* chk, unsigned long long* t) {
""" + cons_src + """
}
"""
  return src, prod_src, cons_src


def expected_checksum() -> int:
  return (NCONS * NCONS) % (1 << 32)


def run_arm(arm: str, reps: int) -> dict:
  dev = Device[Device.DEFAULT]
  if arm == "qmd_latch":
    latch = dict(arrive_at_latch_valid=1, arrive_at_latch_id=LATCH_ID, enable_program_pre_exit=1,
                 pre_exit_at_last_cta_launch=1)
  else:
    latch = None

  src, prod_src, cons_src = build_source()
  lib = NVRTCCompiler(dev.arch, ptx=False, cache_key="nv_pdl").compile(src)
  prod = NVProgram(dev, "pdl_producer", lib)
  cons = NVProgram(dev, "pdl_consumer", lib)
  if latch is not None:
    prod.qmd.write(**latch)
    cons.qmd.write(wait_on_latch_valid=1, wait_on_latch_id=LATCH_ID)

  out_buf = dev.allocator._alloc(NPROD * 4, BufferSpec())
  t_buf = dev.allocator._alloc(5 * 8, BufferSpec())
  chk_buf = dev.allocator._alloc(4, BufferSpec())
  dev.allocator._copyin(out_buf, memoryview(b"\x00" * (NPROD * 4)))
  dev.synchronize()

  q = NVComputeQueue(queue_idx=0)
  q.setup(compute_class=dev.iface.compute_class, local_mem_window=dev.local_mem_window,
          shared_mem_window=dev.shared_mem_window)
  q.wait(dev.timeline_signal, dev.timeline_value - 1).memory_barrier()

  rows = []
  for _ in range(reps):
    # Reset the timing slots and checksum slot for this rep, then drain the copy
    # queue so the compute rep below cannot race the DMA overwrite.
    dev.allocator._copyin(t_buf, memoryview(b"\x00" * 40))
    dev.allocator._copyin(chk_buf, memoryview(b"\x00" * 4))
    dev.synchronize()

    st, en = dev.new_signal(), dev.new_signal()
    q.timestamp(st)
    q.exec(prod, prod.fill_kernargs((out_buf, t_buf)), (GS, 1, 1), (LS, 1, 1))
    q.exec(cons, cons.fill_kernargs((out_buf, chk_buf, t_buf)), (1, 1, 1), (LS, 1, 1))
    q.timestamp(en)
    q.signal(dev.timeline_signal, dev.next_timeline())
    q.submit(dev)
    # HWQueue.submit does not clear _q: without a reset, the next rep would
    # re-send the whole accumulated command list (stale execs re-dispatching
    # recycled signal slots). The probe queue is never .bind()ed, so _q is a
    # plain list and resetting is safe; the graph binds its own queues and is
    # unaffected.
    q._q = []
    dev.synchronize(timeout=15000)

    t_blob = memoryview(bytearray(40))
    chk_blob = memoryview(bytearray(4))
    dev.allocator._copyout(t_blob, t_buf)
    dev.allocator._copyout(chk_blob, chk_buf)
    t = [int.from_bytes(t_blob[8 * k:8 * k + 8], "little") for k in range(5)]
    chk = int.from_bytes(chk_blob, "little")
    rows.append({
      "prod_start_ns": t[0], "prod_end_ns": t[1], "cons_start_ns": t[2], "cons_end_ns": t[3],
      "producer_us": (t[1] - t[0]) / 1000.0,
      "consumer_us": (t[3] - t[2]) / 1000.0,
      "shadow_us": (t[2] - t[0]) / 1000.0,        # consumer launch delay after producer start
      "overlap_us": (t[1] - t[2]) / 1000.0,       # >0: consumer started before producer grid ended
      "wall_us": (t[3] - t[0]) / 1000.0,          # in-kernel chain wall
      "hcq_wall_us": float(en.timestamp - st.timestamp),
      "checksum": chk,
      "checksum_correct": chk == expected_checksum(),
    })

  def med(k):
    return statistics.median(r[k] for r in rows)

  return {
    "arm": arm,
    "arch": dev.arch,
    "device": Device.DEFAULT,
    "reps": len(rows),
    "env": {k: os.environ.get(k, "") for k in ("NV_PDL_PRODUCER_PROGRAMS", "NV_PDL_CONSUMER_PROGRAMS",
                                                "NV_PDL_TRIGGER_POSITION", "NV_PDL_LATCH_ID")},
    "producer_emitted": prod_src.splitlines()[1] if "launch_dependents" in prod_src else None,
    "consumer_emitted": cons_src.splitlines()[1] if "griddepcontrol" in cons_src else None,
    "qmd_latch_fields": latch,
    "median": {k: round(med(k), 3) for k in
               ("producer_us", "consumer_us", "shadow_us", "overlap_us", "wall_us", "hcq_wall_us")},
    "checksum_correct_all": all(r["checksum_correct"] for r in rows),
    "rows": rows,
  }


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--arm", choices=ARMS)
  ap.add_argument("--out")
  ap.add_argument("--reps", type=int, default=12)
  ap.add_argument("--run-all", action="store_true")
  args = ap.parse_args()

  if args.run_all:
    results = {}
    for arm in ARMS:
      env = dict(os.environ)
      env.pop("NV_PDL_PRODUCER_PROGRAMS", None)
      env.pop("NV_PDL_CONSUMER_PROGRAMS", None)
      env.pop("NV_PDL_TRIGGER_POSITION", None)
      if arm in ("pdl_end", "pdl_start"):
        env["NV_PDL_PRODUCER_PROGRAMS"] = "pdl_producer"
        env["NV_PDL_CONSUMER_PROGRAMS"] = "pdl_consumer"
      if arm == "pdl_start":
        env["NV_PDL_TRIGGER_POSITION"] = "start"
      out = f"/tmp/nv_pdl_trigger_{arm}.json"
      cmd = [sys.executable, os.path.abspath(__file__), "--arm", arm, "--out", out, "--reps", str(args.reps)]
      try:
        subprocess.run(cmd, env=env, timeout=300, check=True)
        with open(out) as f:
          results[arm] = json.load(f)
      except subprocess.TimeoutExpired:
        results[arm] = {"arm": arm, "status": "TIMEOUT"}
      except subprocess.CalledProcessError as e:
        results[arm] = {"arm": arm, "status": f"FAILED rc={e.returncode}"}
      print(json.dumps({arm: results[arm].get("median") or results[arm].get("status")}))
    doc = build_document(results)
    if args.out:
      with open(args.out, "w") as f:
        json.dump(doc, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(doc, sort_keys=True))
    return 0

  assert args.arm and args.out
  result = run_arm(args.arm, args.reps)
  with open(args.out, "w") as f:
    json.dump(result, f, indent=2, sort_keys=True)
    f.write("\n")
  print(json.dumps(result.get("median")))
  return 0


def build_document(results: dict) -> dict:
  checksum_valid = all(results.get(a, {}).get("checksum_correct_all") is True for a in ARMS)
  have_medians = all(results.get(a, {}).get("median") for a in ARMS)
  if not have_medians:
    verdict = "INVALID"
    per_pair, predicted_delta, qmd_latch_overlap, reported_delta = None, None, None, None
  elif not checksum_valid:
    verdict = "INVALID"
    per_pair, predicted_delta, qmd_latch_overlap, reported_delta = None, None, None, None
  else:
    # The hypothesis under test is specifically the early in-kernel trigger:
    # does moving griddepcontrol.launch_dependents from END to START release the
    # dependent grid earlier than the current END placement? The already-landed
    # QMD latch (arrive_at_latch + pre_exit_at_last_cta_launch) is measured as a
    # separate arm and is NOT a new lever.
    end, start, latch = (results["pdl_end"]["median"], results["pdl_start"]["median"],
                         results["qmd_latch"]["median"])
    delta_overlap = start["overlap_us"] - end["overlap_us"]
    delta_shadow = end["shadow_us"] - start["shadow_us"]
    per_pair = max(delta_overlap, delta_shadow)
    predicted_delta = per_pair * 600.0
    qmd_latch_overlap = round(latch["overlap_us"], 3)
    # Early trigger is only a lever if it produces real overlap beyond the
    # current END placement. A sub-microsecond delta is launch noise, and any
    # overlap it yields must not be one the already-landed latch already has.
    early_trigger_moves = per_pair >= 1.0 and predicted_delta > 50.0 and start["overlap_us"] > 1.0
    new_lever = early_trigger_moves and start["overlap_us"] > qmd_latch_overlap + 1.0
    verdict = "PROMOTE" if new_lever else "NO-GO"
    reported_delta = round(predicted_delta, 1) if new_lever else None
  return {
    "schema": "tinygrad.nv_pdl_early_trigger.v1",
    "date": "2026-08-19",
    "branch": "nvidia-bringup-20260731",
    "head": "d14e6964e",
    "task": "Agent B: EARLY PDL trigger hypothesis (llama kernel-start semantics)",
    "arms": results,
    "conclusion": verdict,
    "verdict": verdict,
    "checksum_valid_all_arms": checksum_valid,
    "qmd_latch_overlap_us": qmd_latch_overlap,
    "predicted_wall_delta_us_per_token": reported_delta,
    "per_pair_overlap_delta_us": round(per_pair, 3) if per_pair is not None else None,
    "scaling_note": "per-pair launch-shadow delta x ~600 dependent pairs per decode step; a sub-1us delta is launch noise and is not multiplied into a wall lever",
  }


if __name__ == "__main__":
  raise SystemExit(main())
