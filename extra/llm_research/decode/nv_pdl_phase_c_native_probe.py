#!/usr/bin/env python3
"""Phase C native-QMD half of the same-grid semantic discriminator.

Uses the same producer/consumer bodies, grid geometry, spin budget, and
checksum as ``nv_pdl_phase_c_cuda_probe.py``, but drives the native NV QMD
exec path and records the same eight ``%globaltimer`` slots:

  t0 producer grid start        t1 producer grid end
  t2 consumer grid start        t3 consumer end
  t4 pre-trigger point          t5 consumer data-wait exit
  t6 launch-complete trigger

The consumer grid-start timestamp is taken BEFORE the in-kernel
``griddepcontrol.wait`` so an early consumer launch is visible even when the
wait itself lasts until producer completion.
"""
from __future__ import annotations

import argparse, json, os, pathlib, statistics, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from tinygrad import Device
from tinygrad.device import BufferSpec
from tinygrad.runtime.ops_nv import NVComputeQueue, NVProgram
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler

SCHEMA = "tinygrad.nv_pdl_phase_c_native.v1"
ARMS = ("no_pdl", "pdl_end", "pdl_start", "pdl_prologue", "qmd_latch")
LS, GS = 256, 4096
NPROD = 1 << 23
NCONS = 513 * 1024
SPIN_NS = 100_000
LATCH_ID = 7

KERNEL_HDR = f"""
#define LS {LS}u
#define GS {GS}u
#define NPROD {NPROD}u
#define NCONS {NCONS}u
#define SPIN_NS {SPIN_NS}ull
"""

PROD_CORE = [
  '  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));',
  "  if (threadIdx.x == 0u && blockIdx.x == 0u) t[0] = now;",
  "  atomicMin(&t[7], now);",
  "  unsigned int i = blockIdx.x*LS + threadIdx.x;",
  "  for (; i < NPROD; i += GS*LS) out[i] = i*2u + 1u;",
  "  if (threadIdx.x == 0u && blockIdx.x == GS-1u) {",
  "    unsigned long long end = now + SPIN_NS;",
  '    do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now)); } while (now < end);',
  "  }",
  "  if (threadIdx.x == 0u && blockIdx.x == 0u) t[4] = now;",
]

TRIGGER_END = [
  '  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));',
  "  if (threadIdx.x == 0u && blockIdx.x == GS-1u) t[6] = now;",
  '  asm volatile("griddepcontrol.launch_dependents;");',
]

TRIGGER_START = [
  '  asm volatile("griddepcontrol.launch_dependents;");',
  '  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));',
  "  if (threadIdx.x == 0u && blockIdx.x == GS-1u) t[6] = now;",
]

PROD_TAIL = [
  '  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));',
  "  if (threadIdx.x == 0u) atomicMax(&t[1], now);",
]

CONS_HEAD = [
  "  unsigned long long now;",
  '  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));',
  "  if (threadIdx.x == 0u && blockIdx.x == 0u) t[2] = now;",
]

CONS_NO_WAIT_MARK = [
  "  if (threadIdx.x == 0u && blockIdx.x == 0u) t[5] = now;",
]

CONS_WAIT = [
  '  asm volatile("griddepcontrol.wait;");',
  '  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));',
  "  if (threadIdx.x == 0u && blockIdx.x == 0u) t[5] = now;",
]

CONS_PROLOGUE = [
  "  unsigned int s0=0u, s1=0u, s2=0u, s3=0u;",
  "  unsigned int i = threadIdx.x;",
]

CONS_CORE = [
  "  for (; i < NCONS; i += LS*4) {",
  "    s0 += in[i]; s1 += in[i + LS]; s2 += in[i + LS*2]; s3 += in[i + LS*3];",
  "  }",
  "  atomicAdd(chk, s0+s1+s2+s3);",
  '  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));',
  "  if (threadIdx.x == 0u && blockIdx.x == 0u) t[3] = now;",
]


def build_source(arm: str) -> str:
  decl = ["  unsigned long long now;"]
  if arm == "pdl_start":
    prod = decl + TRIGGER_START + PROD_CORE + PROD_TAIL
  elif arm in ("pdl_end", "pdl_prologue"):
    prod = decl + PROD_CORE + TRIGGER_END + PROD_TAIL
  else:
    prod = decl + PROD_CORE + PROD_TAIL
  if arm == "pdl_prologue":
    cons = CONS_HEAD + CONS_PROLOGUE + CONS_WAIT + CONS_CORE
  elif arm in ("pdl_start", "pdl_end"):
    cons = CONS_HEAD + CONS_WAIT + CONS_PROLOGUE + CONS_CORE
  else:
    cons = CONS_HEAD + CONS_NO_WAIT_MARK + CONS_PROLOGUE + CONS_CORE
  return KERNEL_HDR + """
extern "C" __global__ void pdl_producer(unsigned int* out, unsigned long long* t) {
""" + "\n".join(prod) + """
}

extern "C" __global__ void pdl_consumer(const unsigned int* in, unsigned int* chk, unsigned long long* t) {
""" + "\n".join(cons) + """
}
"""


def expected_checksum() -> int:
  return (NCONS * NCONS) % (1 << 32)


def run_arm(arm: str, reps: int, warmup: int = 1) -> dict:
  dev = Device[Device.DEFAULT]
  latch = None
  if arm == "qmd_latch":
    latch = dict(arrive_at_latch_valid=1, arrive_at_latch_id=LATCH_ID,
                 enable_program_pre_exit=1, pre_exit_at_last_cta_launch=1)
  src = build_source(arm)
  lib = NVRTCCompiler(dev.arch, ptx=False, cache_key=f"nv_pdl_phase_c_{arm}").compile(src)
  prod = NVProgram(dev, "pdl_producer", lib)
  cons = NVProgram(dev, "pdl_consumer", lib)
  if latch is not None:
    prod.qmd.write(**latch)
    cons.qmd.write(wait_on_latch_valid=1, wait_on_latch_id=LATCH_ID)

  out_buf = dev.allocator._alloc(NPROD * 4, BufferSpec())
  t_buf = dev.allocator._alloc(8 * 8, BufferSpec())
  chk_buf = dev.allocator._alloc(4, BufferSpec())
  dev.allocator._copyin(out_buf, memoryview(b"\x00" * (NPROD * 4)))
  dev.synchronize()

  q = NVComputeQueue(queue_idx=0)
  q.setup(compute_class=dev.iface.compute_class, local_mem_window=dev.local_mem_window,
          shared_mem_window=dev.shared_mem_window)
  q.wait(dev.timeline_signal, dev.timeline_value - 1).memory_barrier()

  rows = []
  out_zeros = memoryview(b"\x00" * (NPROD * 4))
  for _ in range(reps):
    # Refill the producer output every rep so the checksum certifies the
    # current grid body, not data retained from the first rep.
    dev.allocator._copyin(out_buf, out_zeros)
    dev.allocator._copyin(t_buf, memoryview(b"\x00" * 56 + b"\xff" * 8))
    dev.allocator._copyin(chk_buf, memoryview(b"\x00" * 4))
    dev.synchronize()

    q.exec(prod, prod.fill_kernargs((out_buf, t_buf)), (GS, 1, 1), (LS, 1, 1))
    q.exec(cons, cons.fill_kernargs((out_buf, chk_buf, t_buf)), (1, 1, 1), (LS, 1, 1))
    q.signal(dev.timeline_signal, dev.next_timeline())
    q.submit(dev)
    q._q = []
    # The queue is never .bind()ed, so the active QMD view belongs to the just
    # submitted command list.  Clear it so the next rep emits a fresh
    # SEND_PCAS instead of chaining into a QMD that is no longer in _q.
    q.active_qmd = None
    q.active_prg_name = None
    dev.synchronize(timeout=15000)

    t_blob = memoryview(bytearray(64))
    chk_blob = memoryview(bytearray(4))
    dev.allocator._copyout(t_blob, t_buf)
    dev.allocator._copyout(chk_blob, chk_buf)
    t = [int.from_bytes(t_blob[8 * k:8 * k + 8], "little") for k in range(8)]
    chk = int.from_bytes(chk_blob, "little")
    trigger_ns = t[6] if t[6] else None
    grid_start_ns = t[7] if t[7] != (1 << 64) - 1 else t[0]
    rows.append({
      "block0_start_ns": t[0], "grid_start_ns": grid_start_ns,
      "trigger_ns": trigger_ns, "prod_end_ns": t[1],
      "cons_grid_start_ns": t[2], "wait_exit_ns": t[5], "cons_end_ns": t[3],
      "producer_us": (t[1] - grid_start_ns) / 1000.0,
      "trigger_shadow_us": None if trigger_ns is None else (trigger_ns - grid_start_ns) / 1000.0,
      "launch_shadow_us": (t[2] - grid_start_ns) / 1000.0,
      "wait_us": (t[5] - t[2]) / 1000.0,
      "overlap_us": (t[1] - t[2]) / 1000.0,
      "wall_us": (t[3] - grid_start_ns) / 1000.0,
      "checksum": chk,
      "checksum_correct": chk == expected_checksum(),
    })

  raw_rows = rows
  rows = rows[warmup:]
  if not rows:
    raise RuntimeError("no measured rows remain after warmup")

  def med(key):
    values = [r[key] for r in rows if r[key] is not None]
    return round(statistics.median(values), 3) if values else None
  def fmt(value):
    return None if value is None else round(value, 3)
  return {
    "schema": SCHEMA,
    "arm": arm,
    "arch": dev.arch,
    "device": Device.DEFAULT,
    "reps": len(rows),
    "total_rows": len(raw_rows),
    "warmup": warmup,
    "env": {k: os.environ.get(k, "") for k in ("NV_PDL_PRODUCER_PROGRAMS", "NV_PDL_CONSUMER_PROGRAMS",
                                                "NV_PDL_TRIGGER_POSITION", "NV_PDL_LATCH_ID")},
    "qmd_latch_fields": latch,
    "consumer_latch_fields": {"wait_on_latch_valid": 1, "wait_on_latch_id": LATCH_ID} if latch is not None else None,
    "grid": {"GS": GS, "LS": LS, "spin_ns": SPIN_NS},
    "median": {k: fmt(med(k)) for k in
               ("producer_us", "trigger_shadow_us", "launch_shadow_us", "wait_us", "overlap_us", "wall_us")},
    "checksum_correct_all": all(r["checksum_correct"] for r in rows),
    "checksum_correct_all_rows": all(r["checksum_correct"] for r in raw_rows),
    "rows": rows,
  }


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--arm", required=True, choices=ARMS)
  ap.add_argument("--reps", type=int, default=12)
  ap.add_argument("--warmup", type=int, default=1, help="initial reps excluded from medians")
  ap.add_argument("--out", type=pathlib.Path, required=True)
  args = ap.parse_args()
  os.environ.pop("NV_PDL_PRODUCER_PROGRAMS", None)
  os.environ.pop("NV_PDL_CONSUMER_PROGRAMS", None)
  os.environ.pop("NV_PDL_TRIGGER_POSITION", None)
  if args.arm not in ("no_pdl", "qmd_latch"):
    os.environ["NV_PDL_PRODUCER_PROGRAMS"] = "pdl_producer"
    os.environ["NV_PDL_CONSUMER_PROGRAMS"] = "pdl_consumer"
    os.environ["NV_PDL_TRIGGER_POSITION"] = "start" if args.arm == "pdl_start" else "end"
  result = run_arm(args.arm, args.reps, args.warmup)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result["median"], indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
