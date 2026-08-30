#!/usr/bin/env python3
"""Stage 2 matched synthetic QMD latch-capability probes (edge-aware PDL hook).

Drives the native NV QMD latch path directly (``NVComputeQueue`` + QMD latch
fields), reusing the Phase C producer/consumer bodies, checksum, and
``%globaltimer`` slots.  Five probes:

  1. ``latch_id_sweep``   -- same consecutive producer/consumer pair, distinct
     latch IDs (default: ``NV_SPLIT_PHASE_LATCH_BASE`` .. ``+COUNT-1``).
  2. ``two_producer``     -- two producers arrive at one latch, single
     consumer waits on it.
  3. ``non_consecutive``  -- A-B-C same queue, A armed and C waiting while B
     stays unarmed.
  4. ``multi_consumer``   -- one producer latch waited on by two consumers.
  5. ``replay_flush``     -- same armed pair replayed across submissions with
     a synchronize (flush) between cycles, plus a no-flush chained variant.

Every GPU worker runs in a fresh process while the exclusive
``/tmp/gpu-bench.lock`` is held (blocking acquisition: it waits for other
holders instead of bypassing).  Driver modes acquire the lock themselves for
the whole run and spawn workers with ``NV_STAGE2_SKIP_LOCK=1``; standalone
worker modes acquire the lock themselves.  An outer ``flock`` wrapper is
detected via the inherited fd and never released by this process.

Usage (smoke)::

  DEV=NV .venv/bin/python extra/llm_research/decode/nv_edge_aware_pdl_stage2_capability.py \\
    --mode sweep --reps 2 --out docs/.../stage2_latch_sweep.json
"""
from __future__ import annotations

import argparse, errno, fcntl, json, os, pathlib, statistics, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

LOCK = "/tmp/gpu-bench.lock"
PYTHON = ROOT / ".venv/bin/python"
if not PYTHON.exists():
  PYTHON = pathlib.Path(__file__).with_name("python").resolve()
EVIDENCE_DIR = ROOT / "docs/task_workflow/evidence/nv-edge-aware-pdl-runtime-hook-20260821"
SCHEMA = "tinygrad.nv_edge_aware_pdl_stage2_capability.v1"

LS, GS, NPROD, NCONS, SPIN_NS = 256, 4096, 1 << 23, 513 * 1024, 100_000
EXP_CONS = (NCONS * NCONS) % (1 << 32)
EXP_CONS2 = (3 * NCONS * NCONS) % (1 << 32)

MISSING = (1 << 64) - 1
PDL_ENV_KEYS = ("NV_PDL_PRODUCER_PROGRAMS", "NV_PDL_CONSUMER_PROGRAMS",
                "NV_PDL_TRIGGER_POSITION", "NV_PDL_LATCH_ID", "NV_SPLIT_PHASE")
_LOCK_FDS: list[int] = []


def _git_head() -> str:
  try:
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()
  except Exception:
    return "unknown"


def _gpu_info() -> str:
  try:
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         check=True, capture_output=True, text=True).stdout.strip().splitlines()
    return out[0] if out else "nvidia-smi unavailable"
  except Exception:
    return "nvidia-smi unavailable"


# ---------------------------------------------------------------------------
# Lock discipline: blocking flock, inherited-lock aware.
# ---------------------------------------------------------------------------

def _lock_inode() -> int:
  return os.stat(LOCK).st_ino


def _already_locked_by_ancestor() -> bool:
  """True if this process already holds (or inherited) a lock on the lock file.

  Only fd identity on the lock inode is checked.  Never call ``flock()`` on an
  inherited fd here: ``LOCK_UN`` on a shared open-file-description releases the
  ancestor's lock too (verified on this host), which would silently break the
  serialization contract.
  """
  ino = _lock_inode()
  for name in os.listdir("/proc/self/fd"):
    try:
      st = os.fstat(int(name))
    except (OSError, ValueError):
      continue
    if st.st_ino == ino:
      return True
  return False


def _enter_lock() -> None:
  """Blocking exclusive flock held for this process' lifetime (waits).

  Skips when the caller already runs inside a lock: either an explicit
  ``NV_STAGE2_SKIP_LOCK=1`` (driver-spawned worker under the driver's lock) or
  an inherited fd from an outer ``flock`` wrapper.  A blocking ``flock`` means
  we wait for other holders instead of bypassing them.
  """
  if os.environ.get("NV_STAGE2_SKIP_LOCK") == "1":
    return
  if _already_locked_by_ancestor():
    return
  fd = os.open(LOCK, os.O_WRONLY | os.O_CREAT, 0o644)
  fcntl.flock(fd, fcntl.LOCK_EX)
  _LOCK_FDS.append(fd)


# ---------------------------------------------------------------------------
# Kernel sources
# ---------------------------------------------------------------------------

def _phase_c_source() -> str:
  """Byte-identical to the Phase C native probe's qmd_latch arm."""
  import nv_pdl_phase_c_native_probe as phase_c
  return phase_c.build_source("qmd_latch")


def _gen_producer_kernel(name: str, spin_ns: int) -> str:
  return f"""
extern "C" __global__ void {name}(unsigned int* out, unsigned long long* t, unsigned int base) {{
  unsigned long long now;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
  if (threadIdx.x == 0u && blockIdx.x == 0u) t[base + 0u] = now;
  atomicMin(&t[base + 3u], now);
  if (threadIdx.x == 0u && blockIdx.x == {GS - 1}u) t[base + 2u] = now;
  unsigned int i = blockIdx.x*LS + threadIdx.x;
  for (; i < NPROD; i += {GS}*LS) out[i] = i*2u + 1u;
  if (threadIdx.x == 0u && blockIdx.x == {GS - 1}u) {{
    unsigned long long end = now + {spin_ns}ull;
    do {{ asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now)); }} while (now < end);
  }}
  if (threadIdx.x == 0u) atomicMax(&t[base + 1u], now);
}}
"""


def _gen_consumer_kernel() -> str:
  return """
extern "C" __global__ void st2_consumer(const unsigned int* in, unsigned int* chk, unsigned long long* t, unsigned int base) {
  unsigned long long now;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
  if (threadIdx.x == 0u && blockIdx.x == 0u) t[base + 0u] = now;
  unsigned int s0=0u, s1=0u, s2=0u, s3=0u;
  unsigned int i = threadIdx.x;
  for (; i < NCONS; i += LS*4) {
    s0 += in[i]; s1 += in[i + LS]; s2 += in[i + LS*2]; s3 += in[i + LS*3];
  }
  atomicAdd(chk, s0+s1+s2+s3);
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
  if (threadIdx.x == 0u && blockIdx.x == 0u) t[base + 1u] = now;
}
"""


def _gen_consumer2_kernel() -> str:
  return """
extern "C" __global__ void st2_consumer2(const unsigned int* in1, const unsigned int* in2,
                                         unsigned int* chk1, unsigned int* chk2, unsigned long long* t, unsigned int base) {
  unsigned long long now;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
  if (threadIdx.x == 0u && blockIdx.x == 0u) t[base + 0u] = now;
  unsigned int s0=0u, s1=0u, s2=0u, s3=0u;
  unsigned int i = threadIdx.x;
  for (; i < NCONS; i += LS*4) {
    s0 += in1[i]; s1 += in1[i + LS]; s2 += in1[i + LS*2]; s3 += in1[i + LS*3];
  }
  atomicAdd(chk1, s0+s1+s2+s3);
  s0=0u; s1=0u; s2=0u; s3=0u;
  i = threadIdx.x;
  for (; i < NCONS; i += LS*4) {
    s0 += in2[i]; s1 += in2[i + LS]; s2 += in2[i + LS*2]; s3 += in2[i + LS*3];
  }
  atomicAdd(chk2, s0+s1+s2+s3);
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
  if (threadIdx.x == 0u && blockIdx.x == 0u) t[base + 1u] = now;
}
"""


def _gen_header() -> str:
  return f"""
#define LS {LS}u
#define GS {GS}u
#define NPROD {NPROD}u
#define NCONS {NCONS}u
"""


def _generic_source(producers: list[tuple[str, int]], consumers: list[str]) -> str:
  body = _gen_header()
  for name, spin_ns in producers:
    body += _gen_producer_kernel(name, spin_ns)
  if "st2_consumer" in consumers:
    body += _gen_consumer_kernel()
  if "st2_consumer2" in consumers:
    body += _gen_consumer2_kernel()
  return body


# ---------------------------------------------------------------------------
# Small GPU-side helpers (imported lazily inside workers)
# ---------------------------------------------------------------------------

def _alloc(dev, size: int):
  from tinygrad.device import BufferSpec
  return dev.allocator._alloc(size, BufferSpec())


def _t_init(n_slots: int, min_slots: list[int]) -> bytes:
  b = bytearray(n_slots * 8)
  for k in min_slots:
    b[k * 8:k * 8 + 8] = b"\xff" * 8
  return bytes(b)


def _read_u64s(dev, t_buf, n_slots: int) -> list[int]:
  blob = memoryview(bytearray(n_slots * 8))
  dev.allocator._copyout(blob, t_buf)
  return [int.from_bytes(blob[8 * k:8 * k + 8], "little") for k in range(n_slots)]


def _read_u32(dev, buf) -> int:
  blob = memoryview(bytearray(4))
  dev.allocator._copyout(blob, buf)
  return int.from_bytes(blob, "little")


def _make_queue(dev, idx=0):
  from tinygrad.runtime.ops_nv import NVComputeQueue
  q = NVComputeQueue(queue_idx=idx)
  q.setup(compute_class=dev.iface.compute_class, local_mem_window=dev.local_mem_window,
          shared_mem_window=dev.shared_mem_window)
  q.wait(dev.timeline_signal, dev.timeline_value - 1).memory_barrier()
  return q


def _new_program(dev, lib, symbol: str, latch_id: int | None, wait: bool):
  from tinygrad.runtime.ops_nv import NVProgram
  prg = NVProgram(dev, symbol, lib)
  if latch_id is not None:
    if wait:
      prg.qmd.write(wait_on_latch_valid=1, wait_on_latch_id=latch_id)
    else:
      prg.qmd.write(arrive_at_latch_valid=1, arrive_at_latch_id=latch_id,
                    enable_program_pre_exit=1, pre_exit_at_last_cta_launch=1)
  return prg


def _compile(dev, cache_key: str, src: str):
  from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
  return NVRTCCompiler(dev.arch, ptx=False, cache_key=cache_key).compile(src)


def _median(values: list[float]) -> float | None:
  values = [v for v in values if v is not None]
  return round(statistics.median(values), 3) if values else None


def _phase_summary(rows: list[dict], metrics: list[str]) -> dict:
  measured = [r for r in rows if not r.get("warmup")]
  return {
    "n_measured": len(measured),
    "n_total": len(rows),
    "checksum_correct_all": bool(measured) and all(r.get("checksum_correct", True) for r in measured),
    "median": {m: _median([r.get(m) for r in measured]) for m in metrics},
  }


def _base_env() -> dict:
  env = dict(os.environ)
  for key in PDL_ENV_KEYS:
    env.pop(key, None)
  env["DEV"] = "NV"
  return env


# ---------------------------------------------------------------------------
# Probe 1 worker: one latch ID, control/candidate/control (Phase C grid)
# ---------------------------------------------------------------------------

def _probe_latch_id_worker(args) -> dict:
  _enter_lock()
  from tinygrad import Device

  dev = Device[Device.DEFAULT]
  latch_id = args.latch_id
  src = _phase_c_source()
  lib = _compile(dev, f"nv_edge_aware_pdl_stage2_latch_{latch_id}", src)

  prod_plain = _new_program(dev, lib, "pdl_producer", None, False)
  cons_plain = _new_program(dev, lib, "pdl_consumer", None, True)
  prod_latch = _new_program(dev, lib, "pdl_producer", latch_id, False)
  cons_latch = _new_program(dev, lib, "pdl_consumer", latch_id, True)

  out_buf = _alloc(dev, NPROD * 4)
  t_buf = _alloc(dev, 8 * 8)
  chk_buf = _alloc(dev, 4)
  dev.allocator._copyin(out_buf, memoryview(b"\x00" * (NPROD * 4)))
  dev.synchronize()

  q = _make_queue(dev)
  out_zeros = memoryview(b"\x00" * (NPROD * 4))
  t_init = _t_init(8, [7])
  import nv_pdl_phase_c_native_probe as phase_c
  expected = phase_c.expected_checksum()

  rows: list[dict] = []
  for phase, prod, cons in (("control", prod_plain, cons_plain),
                            ("candidate", prod_latch, cons_latch),
                            ("control", prod_plain, cons_plain)):
    for rep in range(args.warmup + args.reps):
      dev.allocator._copyin(out_buf, out_zeros)
      dev.allocator._copyin(t_buf, memoryview(t_init))
      dev.allocator._copyin(chk_buf, memoryview(b"\x00" * 4))
      dev.synchronize()
      q.exec(prod, prod.fill_kernargs((out_buf, t_buf)), (GS, 1, 1), (LS, 1, 1))
      q.exec(cons, cons.fill_kernargs((out_buf, chk_buf, t_buf)), (1, 1, 1), (LS, 1, 1))
      q.signal(dev.timeline_signal, dev.next_timeline())
      q.submit(dev)
      q._q = []
      q.active_qmd = None
      q.active_prg_name = None
      dev.synchronize(timeout=15000)

      t = _read_u64s(dev, t_buf, 8)
      chk = _read_u32(dev, chk_buf)
      grid_start_ns = t[7] if t[7] != MISSING else t[0]
      rows.append({
        "phase": phase,
        "warmup": rep < args.warmup,
        "latch_id": latch_id,
        "block0_start_ns": t[0], "grid_start_ns": grid_start_ns,
        "prod_end_ns": t[1], "cons_grid_start_ns": t[2], "cons_end_ns": t[3],
        "pre_trigger_ns": t[4], "wait_exit_ns": t[5],
        "producer_us": round((t[1] - grid_start_ns) / 1000.0, 3),
        "launch_shadow_us": round((t[2] - grid_start_ns) / 1000.0, 3),
        "wait_us": round((t[5] - t[2]) / 1000.0, 3),
        "overlap_us": round((t[1] - t[2]) / 1000.0, 3),
        "wall_us": round((t[3] - grid_start_ns) / 1000.0, 3),
        "checksum": chk,
        "checksum_correct": chk == expected,
      })

  cand = [r for r in rows if r["phase"] == "candidate" and not r["warmup"]]
  med_overlap = _median([r["overlap_us"] for r in cand])
  chk_ok = bool(cand) and all(r["checksum_correct"] for r in cand)
  if chk_ok and med_overlap is not None and med_overlap > 1.0:
    verdict, reason = "supported", "candidate overlap positive and checksums correct on the matched grid"
  elif chk_ok:
    verdict, reason = "refuted", f"checksums correct but candidate median overlap {med_overlap} us: no launch-ahead"
  else:
    verdict, reason = "refuted", "candidate checksum failed: consumer read producer data before it was visible"

  phases = {}
  for phase in ("control", "candidate"):
    phases[phase] = _phase_summary([r for r in rows if r["phase"] == phase],
                                   ["producer_us", "launch_shadow_us", "wait_us", "overlap_us", "wall_us"])

  return {
    "schema": "tinygrad.nv_edge_aware_pdl_stage2_latch_id.v1",
    "probe": "latch_id_sweep",
    "latch_id": latch_id,
    "verdict": verdict,
    "reason": reason,
    "arch": dev.arch,
    "device": Device.DEFAULT,
    "latch_fields": {"producer": {"arrive_at_latch_id": latch_id, "arrive_at_latch_valid": 1,
                                  "enable_program_pre_exit": 1, "pre_exit_at_last_cta_launch": 1},
                     "consumer": {"wait_on_latch_id": latch_id, "wait_on_latch_valid": 1}},
    "grid": {"GS": GS, "LS": LS, "spin_ns": SPIN_NS, "NPROD": NPROD, "NCONS": NCONS},
    "phases": phases,
    "expected_checksum": expected,
    "rows": rows,
  }


# ---------------------------------------------------------------------------
# Probe 2 worker: two producers, one consumer (same latch)
# ---------------------------------------------------------------------------

def _probe_two_producer_worker(args) -> dict:
  _enter_lock()
  from tinygrad import Device

  dev = Device[Device.DEFAULT]
  latch_id = args.latch_id
  src = _generic_source([("st2_producer_p1", 100_000), ("st2_producer_p2", 400_000)], ["st2_consumer2"])
  lib = _compile(dev, "nv_edge_aware_pdl_stage2_two_producer", src)

  p1_plain = _new_program(dev, lib, "st2_producer_p1", None, False)
  p2_plain = _new_program(dev, lib, "st2_producer_p2", None, False)
  c_plain = _new_program(dev, lib, "st2_consumer2", None, True)
  p1_latch = _new_program(dev, lib, "st2_producer_p1", latch_id, False)
  p2_latch = _new_program(dev, lib, "st2_producer_p2", latch_id, False)
  c_latch = _new_program(dev, lib, "st2_consumer2", latch_id, True)

  out1 = _alloc(dev, NPROD * 4)
  out2 = _alloc(dev, NPROD * 4)
  t_buf = _alloc(dev, 10 * 8)
  chk1 = _alloc(dev, 4)
  chk2 = _alloc(dev, 4)
  dev.synchronize()
  q = _make_queue(dev)
  t_init = _t_init(10, [3, 7])
  zeros = memoryview(b"\x00" * (NPROD * 4))

  rows: list[dict] = []
  for phase, p1, p2, cons in (("control", p1_plain, p2_plain, c_plain),
                              ("candidate", p1_latch, p2_latch, c_latch),
                              ("control", p1_plain, p2_plain, c_plain)):
    for rep in range(args.warmup + args.reps):
      dev.allocator._copyin(out1, zeros)
      dev.allocator._copyin(out2, zeros)
      dev.allocator._copyin(t_buf, memoryview(t_init))
      dev.allocator._copyin(chk1, memoryview(b"\x00" * 4))
      dev.allocator._copyin(chk2, memoryview(b"\x00" * 4))
      dev.synchronize()
      q.exec(p1, p1.fill_kernargs((out1, t_buf), vals=(0,)), (GS, 1, 1), (LS, 1, 1))
      q.exec(p2, p2.fill_kernargs((out2, t_buf), vals=(4,)), (GS, 1, 1), (LS, 1, 1))
      q.exec(cons, cons.fill_kernargs((out1, out2, chk1, chk2, t_buf), vals=(8,)), (1, 1, 1), (LS, 1, 1))
      q.signal(dev.timeline_signal, dev.next_timeline())
      q.submit(dev)
      q._q = []
      q.active_qmd = None
      q.active_prg_name = None
      dev.synchronize(timeout=15000)

      t = _read_u64s(dev, t_buf, 10)
      c1 = _read_u32(dev, chk1)
      c2 = _read_u32(dev, chk2)
      p1_start, p1_end, p1_last, p1_min = t[0], t[1], t[2], t[3]
      p2_start, p2_end, p2_last, p2_min = t[4], t[5], t[6], t[7]
      c_start, c_end = t[8], t[9]
      rows.append({
        "phase": phase,
        "warmup": rep < args.warmup,
        "latch_id": latch_id,
        "p1_start_ns": p1_start, "p1_end_ns": p1_end, "p1_last_cta_ns": p1_last,
        "p2_start_ns": p2_start, "p2_end_ns": p2_end, "p2_last_cta_ns": p2_last,
        "c_start_ns": c_start, "c_end_ns": c_end,
        "p1_last_cta_us": round((p1_last - p1_start) / 1000.0, 3),
        "p2_last_cta_us": round((p2_last - p2_start) / 1000.0, 3),
        "c_vs_p1_arrival_us": round((c_start - p1_last) / 1000.0, 3),
        "c_vs_p2_arrival_us": round((c_start - p2_last) / 1000.0, 3),
        "c_vs_p2_end_us": round((c_start - p2_end) / 1000.0, 3),
        "p2_end_vs_c_start_us": round((p2_end - c_start) / 1000.0, 3),
        "checksum1": c1, "checksum1_correct": c1 == EXP_CONS,
        "checksum2": c2, "checksum2_correct": c2 == EXP_CONS,
      })

  cand = [r for r in rows if r["phase"] == "candidate" and not r["warmup"]]
  chk_ok = bool(cand) and all(r["checksum1_correct"] and r["checksum2_correct"] for r in cand)
  early = _median([r["c_vs_p2_arrival_us"] for r in cand])
  med_early_end = _median([r["c_vs_p2_end_us"] for r in cand])
  if not chk_ok:
    verdict, reason = "refuted", "consumer checksum failed for at least one producer buffer: the single WAIT_ON_LATCH_ID did not cover both producers"
  elif early is not None and early < -10.0:
    verdict, reason = "refuted", f"consumer grid started {abs(early)} us before producer 2's arrival: latch released on the first arrival"
  else:
    verdict, reason = "named-unavailable", (
      f"consumer launched {med_early_end} us before producer 2 end with both checksums correct on this matched grid; "
      "single-row aggregation-like behavior is not promoted to support per Stage 2 rules")

  return {
    "schema": "tinygrad.nv_edge_aware_pdl_stage2_two_producer.v1",
    "probe": "two_producer",
    "verdict": verdict,
    "reason": reason,
    "arch": dev.arch,
    "device": Device.DEFAULT,
    "latch_id": latch_id,
    "latch_fields": {"producers": {"arrive_at_latch_id": latch_id, "arrive_at_latch_valid": 1,
                                   "enable_program_pre_exit": 1, "pre_exit_at_last_cta_launch": 1},
                     "consumer": {"wait_on_latch_id": latch_id, "wait_on_latch_valid": 1}},
    "grid": {"GS": GS, "LS": LS, "spin_ns_p1": 100_000, "spin_ns_p2": 400_000, "NPROD": NPROD, "NCONS": NCONS},
    "phases": {phase: _phase_summary([r for r in rows if r["phase"] == phase],
                                     ["p1_last_cta_us", "p2_last_cta_us", "c_vs_p1_arrival_us",
                                      "c_vs_p2_arrival_us", "c_vs_p2_end_us"])
               for phase in ("control", "candidate")},
    "expected_checksum": EXP_CONS,
    "rows": rows,
  }


# ---------------------------------------------------------------------------
# Probe 3 worker: non-consecutive same-queue arming (A-B-C)
# ---------------------------------------------------------------------------

def _probe_non_consecutive_worker(args) -> dict:
  _enter_lock()
  from tinygrad import Device

  dev = Device[Device.DEFAULT]
  latch_id = args.latch_id
  src = _generic_source([("st2_producer_a", 100_000), ("st2_producer_b", 200_000)], ["st2_consumer"])
  lib = _compile(dev, "nv_edge_aware_pdl_stage2_non_consecutive", src)

  a_plain = _new_program(dev, lib, "st2_producer_a", None, False)
  b_plain = _new_program(dev, lib, "st2_producer_b", None, False)
  c_plain = _new_program(dev, lib, "st2_consumer", None, True)
  a_latch = _new_program(dev, lib, "st2_producer_a", latch_id, False)
  b_latch = _new_program(dev, lib, "st2_producer_b", None, False)
  c_latch = _new_program(dev, lib, "st2_consumer", latch_id, True)

  out_a = _alloc(dev, NPROD * 4)
  out_b = _alloc(dev, NPROD * 4)
  t_buf = _alloc(dev, 10 * 8)
  chk = _alloc(dev, 4)
  dev.synchronize()
  q = _make_queue(dev)
  t_init = _t_init(10, [3, 7])
  zeros = memoryview(b"\x00" * (NPROD * 4))

  rows: list[dict] = []
  for phase, pa, pb, pc in (("control", a_plain, b_plain, c_plain),
                            ("candidate", a_latch, b_latch, c_latch),
                            ("control", a_plain, b_plain, c_plain)):
    for rep in range(args.warmup + args.reps):
      dev.allocator._copyin(out_a, zeros)
      dev.allocator._copyin(out_b, zeros)
      dev.allocator._copyin(t_buf, memoryview(t_init))
      dev.allocator._copyin(chk, memoryview(b"\x00" * 4))
      dev.synchronize()
      q.exec(pa, pa.fill_kernargs((out_a, t_buf), vals=(0,)), (GS, 1, 1), (LS, 1, 1))
      q.exec(pb, pb.fill_kernargs((out_b, t_buf), vals=(4,)), (GS, 1, 1), (LS, 1, 1))
      q.exec(pc, pc.fill_kernargs((out_a, chk, t_buf), vals=(8,)), (1, 1, 1), (LS, 1, 1))
      q.signal(dev.timeline_signal, dev.next_timeline())
      q.submit(dev)
      q._q = []
      q.active_qmd = None
      q.active_prg_name = None
      dev.synchronize(timeout=15000)

      t = _read_u64s(dev, t_buf, 10)
      c = _read_u32(dev, chk)
      a_start, a_end, a_last, _ = t[0], t[1], t[2], t[3]
      b_start, b_end, _, _ = t[4], t[5], t[6], t[7]
      c_start, c_end = t[8], t[9]
      rows.append({
        "phase": phase,
        "warmup": rep < args.warmup,
        "latch_id": latch_id,
        "a_start_ns": a_start, "a_end_ns": a_end, "a_last_cta_ns": a_last,
        "b_start_ns": b_start, "b_end_ns": b_end,
        "c_start_ns": c_start, "c_end_ns": c_end,
        "a_last_cta_us": round((a_last - a_start) / 1000.0, 3),
        "b_run_us": round((b_end - b_start) / 1000.0, 3),
        "c_vs_b_start_us": round((c_start - b_start) / 1000.0, 3),
        "c_vs_b_end_us": round((c_start - b_end) / 1000.0, 3),
        "checksum": c, "checksum_correct": c == EXP_CONS,
      })

  cand = [r for r in rows if r["phase"] == "candidate" and not r["warmup"]]
  chk_ok = bool(cand) and all(r["checksum_correct"] for r in cand)
  med_before_b = _median([r["c_vs_b_end_us"] for r in cand])
  if not chk_ok:
    verdict, reason = "refuted", "consumer checksum failed: C did not see A's data"
  elif med_before_b is not None and med_before_b < -10.0:
    verdict, reason = "supported", (
      f"consumer C grid started {abs(med_before_b)} us before middle kernel B ended with correct checksums: "
      "the latch armed the non-consecutive same-queue A-C pair")
  else:
    verdict, reason = "refuted", (
      f"consumer C grid started {med_before_b} us relative to B end (>= -10): C waited for B completion, "
      "the latch did not cross the unarmed middle kernel")

  return {
    "schema": "tinygrad.nv_edge_aware_pdl_stage2_non_consecutive.v1",
    "probe": "non_consecutive",
    "verdict": verdict,
    "reason": reason,
    "arch": dev.arch,
    "device": Device.DEFAULT,
    "latch_id": latch_id,
    "latch_fields": {"A": {"arrive_at_latch_id": latch_id, "arrive_at_latch_valid": 1,
                           "enable_program_pre_exit": 1, "pre_exit_at_last_cta_launch": 1},
                     "B": None,
                     "C": {"wait_on_latch_id": latch_id, "wait_on_latch_valid": 1}},
    "grid": {"GS": GS, "LS": LS, "spin_ns_a": 100_000, "spin_ns_b": 200_000, "NPROD": NPROD, "NCONS": NCONS},
    "phases": {phase: _phase_summary([r for r in rows if r["phase"] == phase],
                                     ["a_last_cta_us", "b_run_us", "c_vs_b_start_us", "c_vs_b_end_us"])
               for phase in ("control", "candidate")},
    "expected_checksum": EXP_CONS,
    "rows": rows,
  }


# ---------------------------------------------------------------------------
# Probe 4 worker: one producer, two consumers on the same latch
# ---------------------------------------------------------------------------

def _probe_multi_consumer_worker(args) -> dict:
  _enter_lock()
  from tinygrad import Device

  dev = Device[Device.DEFAULT]
  latch_id = args.latch_id
  src = _generic_source([("st2_producer_p", 100_000)], ["st2_consumer"])
  lib = _compile(dev, "nv_edge_aware_pdl_stage2_multi_consumer", src)

  p_plain = _new_program(dev, lib, "st2_producer_p", None, False)
  c_plain = _new_program(dev, lib, "st2_consumer", None, True)
  p_latch = _new_program(dev, lib, "st2_producer_p", latch_id, False)
  c_latch = _new_program(dev, lib, "st2_consumer", latch_id, True)

  out = _alloc(dev, NPROD * 4)
  t_buf = _alloc(dev, 8 * 8)
  chk1 = _alloc(dev, 4)
  chk2 = _alloc(dev, 4)
  dev.synchronize()
  q = _make_queue(dev)
  t_init = _t_init(8, [3])
  zeros = memoryview(b"\x00" * (NPROD * 4))
  c1_view = out.offset(offset=0, size=NCONS * 4)
  c2_view = out.offset(offset=NCONS * 4, size=NCONS * 4)

  rows: list[dict] = []
  for phase, pp, pc in (("control", p_plain, c_plain),
                        ("candidate", p_latch, c_latch),
                        ("control", p_plain, c_plain)):
    for rep in range(args.warmup + args.reps):
      dev.allocator._copyin(out, zeros)
      dev.allocator._copyin(t_buf, memoryview(t_init))
      dev.allocator._copyin(chk1, memoryview(b"\x00" * 4))
      dev.allocator._copyin(chk2, memoryview(b"\x00" * 4))
      dev.synchronize()
      q.exec(pp, pp.fill_kernargs((out, t_buf), vals=(0,)), (GS, 1, 1), (LS, 1, 1))
      q.exec(pc, pc.fill_kernargs((c1_view, chk1, t_buf), vals=(4,)), (1, 1, 1), (LS, 1, 1))
      q.exec(pc, pc.fill_kernargs((c2_view, chk2, t_buf), vals=(6,)), (1, 1, 1), (LS, 1, 1))
      q.signal(dev.timeline_signal, dev.next_timeline())
      q.submit(dev)
      q._q = []
      q.active_qmd = None
      q.active_prg_name = None
      dev.synchronize(timeout=15000)

      t = _read_u64s(dev, t_buf, 8)
      c1 = _read_u32(dev, chk1)
      c2 = _read_u32(dev, chk2)
      p_start, p_end, p_last, _ = t[0], t[1], t[2], t[3]
      c1_start, c1_end = t[4], t[5]
      c2_start, c2_end = t[6], t[7]
      rows.append({
        "phase": phase,
        "warmup": rep < args.warmup,
        "latch_id": latch_id,
        "p_start_ns": p_start, "p_end_ns": p_end, "p_last_cta_ns": p_last,
        "c1_start_ns": c1_start, "c1_end_ns": c1_end,
        "c2_start_ns": c2_start, "c2_end_ns": c2_end,
        "p_last_cta_us": round((p_last - p_start) / 1000.0, 3),
        "c1_vs_p_end_us": round((c1_start - p_end) / 1000.0, 3),
        "c2_vs_p_end_us": round((c2_start - p_end) / 1000.0, 3),
        "checksum1": c1, "checksum1_correct": c1 == EXP_CONS,
        "checksum2": c2, "checksum2_correct": c2 == EXP_CONS2,
      })

  cand = [r for r in rows if r["phase"] == "candidate" and not r["warmup"]]
  chk_ok = bool(cand) and all(r["checksum1_correct"] and r["checksum2_correct"] for r in cand)
  both_launched = bool(cand) and all(r["c1_start_ns"] and r["c2_start_ns"] for r in cand)
  med_c1 = _median([r["c1_vs_p_end_us"] for r in cand])
  med_c2 = _median([r["c2_vs_p_end_us"] for r in cand])
  if not both_launched or not chk_ok:
    verdict, reason = "refuted", (
      f"consumers launched={both_launched} checksums_ok={chk_ok}: latch was not safely broadcast to both consumers")
  else:
    verdict, reason = "supported", (
      f"both consumers launched and checksums correct; C1 {med_c1} us / C2 {med_c2} us vs producer end on this matched grid")

  return {
    "schema": "tinygrad.nv_edge_aware_pdl_stage2_multi_consumer.v1",
    "probe": "multi_consumer",
    "verdict": verdict,
    "reason": reason,
    "arch": dev.arch,
    "device": Device.DEFAULT,
    "latch_id": latch_id,
    "latch_fields": {"producer": {"arrive_at_latch_id": latch_id, "arrive_at_latch_valid": 1,
                                  "enable_program_pre_exit": 1, "pre_exit_at_last_cta_launch": 1},
                     "consumers": {"wait_on_latch_id": latch_id, "wait_on_latch_valid": 1}},
    "grid": {"GS": GS, "LS": LS, "spin_ns": 100_000, "NPROD": NPROD, "NCONS": NCONS},
    "phases": {phase: _phase_summary([r for r in rows if r["phase"] == phase],
                                     ["p_last_cta_us", "c1_vs_p_end_us", "c2_vs_p_end_us"])
               for phase in ("control", "candidate")},
    "expected_checksums": {"c1": EXP_CONS, "c2": EXP_CONS2},
    "rows": rows,
  }


# ---------------------------------------------------------------------------
# Probe 5 worker: replay / flush reuse of the same armed pair
# ---------------------------------------------------------------------------

def _probe_replay_flush_worker(args) -> dict:
  _enter_lock()
  from tinygrad import Device

  dev = Device[Device.DEFAULT]
  latch_id = args.latch_id
  src = _generic_source([("st2_producer_p", 100_000)], ["st2_consumer"])
  lib = _compile(dev, "nv_edge_aware_pdl_stage2_replay_flush", src)

  p_plain = _new_program(dev, lib, "st2_producer_p", None, False)
  c_plain = _new_program(dev, lib, "st2_consumer", None, True)
  p_latch = _new_program(dev, lib, "st2_producer_p", latch_id, False)
  c_latch = _new_program(dev, lib, "st2_consumer", latch_id, True)

  out = _alloc(dev, NPROD * 4)
  t_buf = _alloc(dev, 8 * 8)
  chk = _alloc(dev, 4)
  dev.synchronize()
  q = _make_queue(dev)
  t_init = _t_init(8, [3])
  zeros = memoryview(b"\x00" * (NPROD * 4))

  def submit_pair(pp, pc, base_p, base_c):
    dev.allocator._copyin(out, zeros)
    dev.allocator._copyin(t_buf, memoryview(t_init))
    dev.allocator._copyin(chk, memoryview(b"\x00" * 4))
    dev.synchronize()
    q.exec(pp, pp.fill_kernargs((out, t_buf), vals=(base_p,)), (GS, 1, 1), (LS, 1, 1))
    q.exec(pc, pc.fill_kernargs((out, chk, t_buf), vals=(base_c,)), (1, 1, 1), (LS, 1, 1))
    q.signal(dev.timeline_signal, dev.next_timeline())
    q.submit(dev)
    q._q = []
    q.active_qmd = None
    q.active_prg_name = None
    dev.synchronize(timeout=15000)
    t = _read_u64s(dev, t_buf, 8)
    return t, _read_u32(dev, chk)

  rows: list[dict] = []
  for phase, pp, pc in (("control", p_plain, c_plain),
                        ("candidate", p_latch, c_latch),
                        ("control", p_plain, c_plain)):
    for cycle in range(args.cycles):
      for rep in range(args.warmup + args.reps):
        t, c = submit_pair(pp, pc, 0, 4)
        p_start, p_end, p_last, _ = t[0], t[1], t[2], t[3]
        c_start, c_end = t[4], t[5]
        rows.append({
          "phase": phase,
          "cycle": cycle,
          "warmup": rep < args.warmup,
          "latch_id": latch_id,
          "p_start_ns": p_start, "p_end_ns": p_end, "p_last_cta_ns": p_last,
          "c_start_ns": c_start, "c_end_ns": c_end,
          "p_last_cta_us": round((p_last - p_start) / 1000.0, 3),
          "overlap_us": round((p_end - c_start) / 1000.0, 3),
          "wall_us": round((c_end - p_start) / 1000.0, 3),
          "checksum": c, "checksum_correct": c == EXP_CONS,
        })

  cand = [r for r in rows if r["phase"] == "candidate" and not r["warmup"]]
  chk_ok = bool(cand) and all(r["checksum_correct"] for r in cand)
  overlap_ok = bool(cand) and all((r["overlap_us"] or 0) > 1.0 for r in cand)
  cycles_seen = sorted({r["cycle"] for r in cand})
  med_overlap = _median([r["overlap_us"] for r in cand])
  if not chk_ok:
    verdict, reason = "refuted", "at least one replay cycle checksum failed: latch state was not safely re-armed for the repeated pair"
  elif not overlap_ok:
    verdict, reason = "refuted", f"at least one replay cycle had no launch-ahead (median overlap {med_overlap} us)"
  else:
    verdict, reason = "supported", (
      f"all {len(cand)} candidate rows across cycles {cycles_seen} checksum-correct with median overlap {med_overlap} us: "
      "the armed pair replays safely with a flush between cycles on this matched grid")

  payload = {
    "schema": "tinygrad.nv_edge_aware_pdl_stage2_replay_flush.v1",
    "probe": "replay_flush",
    "verdict": verdict,
    "reason": reason,
    "arch": dev.arch,
    "device": Device.DEFAULT,
    "latch_id": latch_id,
    "latch_fields": {"producer": {"arrive_at_latch_id": latch_id, "arrive_at_latch_valid": 1,
                                  "enable_program_pre_exit": 1, "pre_exit_at_last_cta_launch": 1},
                     "consumer": {"wait_on_latch_id": latch_id, "wait_on_latch_valid": 1}},
    "grid": {"GS": GS, "LS": LS, "spin_ns": 100_000, "NPROD": NPROD, "NCONS": NCONS},
    "cycles": args.cycles,
    "phases": {phase: _phase_summary([r for r in rows if r["phase"] == phase], ["p_last_cta_us", "overlap_us", "wall_us"])
               for phase in ("control", "candidate")},
    "expected_checksum": EXP_CONS,
    "rows": rows,
  }

  if args.no_flush:
    payload["no_flush"] = _replay_no_flush_variant(dev, q, p_latch, c_latch, latch_id, args, lib, out)
  return payload


def _replay_no_flush_variant(dev, q, p_latch, c_latch, latch_id, args, lib, out) -> dict:
  """P1->C1->P2->C2 in one submission, all on the same latch (no flush between)."""
  out2 = _alloc(dev, NPROD * 4)
  t_buf = _alloc(dev, 16 * 8)
  chk1 = _alloc(dev, 4)
  chk2 = _alloc(dev, 4)
  t_init = _t_init(16, [3, 11])
  zeros = memoryview(b"\x00" * (NPROD * 4))
  c1_view = out.offset(offset=0, size=NCONS * 4)
  c2_view = out2.offset(offset=0, size=NCONS * 4)

  rows = []
  for rep in range(args.warmup + args.reps):
    dev.allocator._copyin(out, zeros)
    dev.allocator._copyin(out2, zeros)
    dev.allocator._copyin(t_buf, memoryview(t_init))
    dev.allocator._copyin(chk1, memoryview(b"\x00" * 4))
    dev.allocator._copyin(chk2, memoryview(b"\x00" * 4))
    dev.synchronize()
    q.exec(p_latch, p_latch.fill_kernargs((out, t_buf), vals=(0,)), (GS, 1, 1), (LS, 1, 1))
    q.exec(c_latch, c_latch.fill_kernargs((c1_view, chk1, t_buf), vals=(4,)), (1, 1, 1), (LS, 1, 1))
    q.exec(p_latch, p_latch.fill_kernargs((out2, t_buf), vals=(8,)), (GS, 1, 1), (LS, 1, 1))
    q.exec(c_latch, c_latch.fill_kernargs((c2_view, chk2, t_buf), vals=(12,)), (1, 1, 1), (LS, 1, 1))
    q.signal(dev.timeline_signal, dev.next_timeline())
    q.submit(dev)
    q._q = []
    q.active_qmd = None
    q.active_prg_name = None
    dev.synchronize(timeout=15000)
    t = _read_u64s(dev, t_buf, 16)
    c1 = _read_u32(dev, chk1)
    c2 = _read_u32(dev, chk2)
    p1_start, p1_end, p1_last, _ = t[0], t[1], t[2], t[3]
    c1_start, c1_end = t[4], t[5]
    p2_start, p2_end, p2_last, _ = t[8], t[9], t[10], t[11]
    c2_start, c2_end = t[12], t[13]
    rows.append({
      "warmup": rep < args.warmup,
      "latch_id": latch_id,
      "p1_start_ns": p1_start, "p1_end_ns": p1_end, "p1_last_cta_ns": p1_last,
      "c1_start_ns": c1_start, "c1_end_ns": c1_end,
      "p2_start_ns": p2_start, "p2_end_ns": p2_end, "p2_last_cta_ns": p2_last,
      "c2_start_ns": c2_start, "c2_end_ns": c2_end,
      "c2_vs_p2_arrival_us": round((c2_start - p2_last) / 1000.0, 3),
      "c2_vs_p2_end_us": round((c2_start - p2_end) / 1000.0, 3),
      "checksum1": c1, "checksum1_correct": c1 == EXP_CONS,
      "checksum2": c2, "checksum2_correct": c2 == EXP_CONS,
    })

  measured = [r for r in rows if not r["warmup"]]
  chk_ok = bool(measured) and all(r["checksum1_correct"] and r["checksum2_correct"] for r in measured)
  med = _median([r["c2_vs_p2_end_us"] for r in measured])
  if not chk_ok:
    verdict, reason = "refuted", "no-flush chained pair: a checksum failed, the latch did not re-arm for the second pair"
  elif med is not None and med < -10.0:
    verdict, reason = "supported", f"no-flush chained pair: C2 launched {abs(med)} us before P2 end with correct checksums, latch re-armed within the submission"
  else:
    verdict, reason = "refuted", f"no-flush chained pair: C2 launched {med} us relative to P2 end (>= -10): no re-arm evidence"
  return {
    "verdict": verdict,
    "reason": reason,
    "median_c2_vs_p2_end_us": med,
    "checksum_correct_all": chk_ok,
    "rows": rows,
  }


# ---------------------------------------------------------------------------
# Worker dispatcher + error capture
# ---------------------------------------------------------------------------

def _run_worker(args, worker_fn) -> int:
  payload = {"schema": None, "commit": _git_head(), "date": time.strftime("%Y-%m-%d"), "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
  wall0 = time.perf_counter()
  try:
    result = worker_fn(args)
    payload.update(result)
    payload["wall_s"] = round(time.perf_counter() - wall0, 3)
    payload["error"] = None
    rc = 0
  except Exception as e:
    payload["error"] = f"{type(e).__name__}: {e}"
    payload["traceback"] = __import__("traceback").format_exc()[-3000:]
    payload["wall_s"] = round(time.perf_counter() - wall0, 3)
    rc = 2
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  if payload["error"]:
    print(payload["error"], file=sys.stderr)
  return rc


def _spawn_worker(cmd: list[str], env: dict, timeout_s: int = 300) -> dict:
  worker_env = dict(env)
  worker_env["NV_STAGE2_SKIP_LOCK"] = "1"
  wrapped = ["timeout", str(timeout_s), *cmd]
  run = subprocess.run(wrapped, env=worker_env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  out = cmd[cmd.index("--out") + 1]
  if run.returncode == 0 and os.path.exists(out):
    return json.loads(pathlib.Path(out).read_text())
  payload = {"worker_rc": run.returncode, "worker_stderr": run.stderr[-4000:]}
  if os.path.exists(out):
    try:
      payload.update(json.loads(pathlib.Path(out).read_text()))
    except Exception:
      pass
  return payload


def _sweep_ids(args) -> list[int]:
  try:
    base = int(os.environ.get("NV_SPLIT_PHASE_LATCH_BASE", "0"))
  except ValueError:
    base = 0
  try:
    count = int(os.environ.get("NV_SPLIT_PHASE_LATCH_COUNT", "8"))
  except ValueError:
    count = 8
  ids = list(range(base, base + count))
  for spot in args.spot or []:
    if spot not in ids:
      ids.append(spot)
  return ids


def _run_sweep_driver(args) -> int:
  _enter_lock()
  ids = _sweep_ids(args)
  env = _base_env()
  per_id: dict[int, dict] = {}
  for latch_id in ids:
    worker_out = args.evidence_dir / f"stage2_latch_sweep_id{latch_id}.json"
    cmd = [str(PYTHON), str(pathlib.Path(__file__).resolve()), "--mode", "latch_id",
           "--latch-id", str(latch_id), "--reps", str(args.reps), "--warmup", str(args.warmup),
           "--out", str(worker_out)]
    per_id[latch_id] = _spawn_worker(cmd, env, timeout_s=args.worker_timeout)

  verdicts = [per_id[i].get("verdict", "named-unavailable") for i in ids]
  if any(v == "refuted" for v in verdicts):
    overall = "refuted"
  elif any(v == "named-unavailable" for v in verdicts):
    overall = "named-unavailable"
  elif all(v == "supported" for v in verdicts):
    overall = "supported"
  else:
    overall = "named-unavailable"
  per_id_summary = {str(i): {"verdict": per_id[i].get("verdict"),
                             "reason": per_id[i].get("reason"),
                             "error": per_id[i].get("error"),
                             "phases": per_id[i].get("phases")} for i in ids}
  reason = (f"tested latch IDs {ids} (env base/count and spot checks); per-ID verdicts "
            + ", ".join(f"{i}:{per_id_summary[str(i)]['verdict']}" for i in ids)
            + "; usable IDs are reported per row, the full pool size is not claimed")

  payload = {
    "schema": "tinygrad.nv_edge_aware_pdl_stage2_latch_sweep.v1",
    "probe": "latch_id_sweep",
    "verdict": overall,
    "reason": reason,
    "commit": _git_head(),
    "date": time.strftime("%Y-%m-%d"),
    "device": "NV",
    "gpu": _gpu_info(),
    "ids": ids,
    "latch_pool_env": {"NV_SPLIT_PHASE_LATCH_BASE": os.environ.get("NV_SPLIT_PHASE_LATCH_BASE", "0"),
                       "NV_SPLIT_PHASE_LATCH_COUNT": os.environ.get("NV_SPLIT_PHASE_LATCH_COUNT", "8")},
    "per_id": {str(i): per_id[i] for i in ids},
    "per_id_summary": per_id_summary,
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  print(json.dumps(per_id_summary, indent=2, sort_keys=True))
  return 0


def _run_full_driver(args) -> int:
  _enter_lock()
  env = _base_env()
  probes = {}
  for mode, out_name in (("sweep", "stage2_latch_sweep.json"),
                         ("two_producer", "stage2_two_producer_one_consumer.json"),
                         ("non_consecutive", "stage2_non_consecutive_arming.json"),
                         ("multi_consumer", "stage2_multi_consumer.json"),
                         ("replay_flush", "stage2_replay_flush.json")):
    out = args.evidence_dir / out_name
    cmd = [str(PYTHON), str(pathlib.Path(__file__).resolve()), "--mode", mode,
           "--reps", str(args.reps), "--warmup", str(args.warmup),
           "--out", str(out)]
    if mode == "sweep":
      for spot in args.spot or []:
        cmd += ["--spot", str(spot)]
    if mode == "replay_flush":
      cmd += ["--cycles", str(args.cycles)]
    probes[mode] = _spawn_worker(cmd, env, timeout_s=args.worker_timeout)

  merged = {
    "schema": SCHEMA,
    "commit": _git_head(),
    "date": time.strftime("%Y-%m-%d"),
    "device": "NV",
    "gpu": _gpu_info(),
    "method": {
      "path": "native QMD latch fields on NVComputeQueue (direct drive), Phase C producer/consumer bodies, "
              "%globaltimer slots, control/candidate/control where feasible, fresh process + flock per GPU worker",
      "latch_fields": {"arrive": ["arrive_at_latch_valid", "arrive_at_latch_id", "enable_program_pre_exit",
                                  "pre_exit_at_last_cta_launch"],
                       "wait": ["wait_on_latch_valid", "wait_on_latch_id"]},
      "grid": {"GS": GS, "LS": LS, "spin_ns": SPIN_NS, "NPROD": NPROD, "NCONS": NCONS},
      "checksum": "sum(out[i]=2i+1) mod 2^32 over NCONS elements, per consumed buffer",
    },
    "probes": probes,
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
  print(json.dumps({k: {"verdict": v.get("verdict"), "reason": v.get("reason"),
                        "error": v.get("error")} for k, v in probes.items()}, indent=2))
  return 0


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--mode", required=True,
                  choices=("driver", "sweep", "latch_id", "two_producer", "non_consecutive",
                           "multi_consumer", "replay_flush"))
  ap.add_argument("--latch-id", type=int, default=7)
  ap.add_argument("--reps", type=int, default=5)
  ap.add_argument("--warmup", type=int, default=1)
  ap.add_argument("--cycles", type=int, default=3)
  ap.add_argument("--no-flush", action="store_true", default=True)
  ap.add_argument("--spot", type=str, default="", help="comma-separated extra latch IDs for the sweep")
  ap.add_argument("--evidence-dir", type=pathlib.Path, default=EVIDENCE_DIR)
  ap.add_argument("--worker-timeout", type=int, default=300)
  ap.add_argument("--out", type=pathlib.Path, required=True)
  args = ap.parse_args()
  args.spot = [int(v) for v in args.spot.split(",") if v.strip()]

  if args.mode == "driver":
    return _run_full_driver(args)
  if args.mode == "sweep":
    return _run_sweep_driver(args)

  workers = {
    "latch_id": _probe_latch_id_worker,
    "two_producer": _probe_two_producer_worker,
    "non_consecutive": _probe_non_consecutive_worker,
    "multi_consumer": _probe_multi_consumer_worker,
    "replay_flush": _probe_replay_flush_worker,
  }
  return _run_worker(args, workers[args.mode])


if __name__ == "__main__":
  raise SystemExit(main())
