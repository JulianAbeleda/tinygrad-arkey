#!/usr/bin/env python3
"""Native NV QMD dependence-counter capability probes.

Two synthetic probes that close the gap between "the latch has no true
multi-producer merge" and "the QMD header exposes DEPENDENCE_COUNTER +
QMD_DECREMENT_DEPENDENCE":

  1. ``join`` -- two producers, one consumer, consumer ``dependence_counter=2``.
     P1 schedules P2 (slot 0) and decrements C (slot 1); P2 decrements C
     (slot 0). C must not start until both producers have finished, and we
     record whether its start is a completion-time join or an early
     launch-ahead relative to P2's tail.
  2. ``non_consecutive`` -- A, B, C on one queue. A schedules B (slot 0) and
     C (slot 1); B's schedule edge to C is disabled. C reads only A's output,
     so this tests whether a dependent schedule edge can skip the middle
     kernel and launch C while B is still running.

Each mode is intended to run in its own process under ``timeout`` so a QMD
wedge in one construction cannot corrupt the other. The checksum is the
correctness gate; ``%globaltimer`` slots are the timing gate.
"""
from __future__ import annotations

import argparse, json, os, pathlib, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import extra.llm_research.decode.nv_edge_aware_pdl_stage2_capability as st2

SCHEMA = "tinygrad.nv_qmd_dependence_counter_probe.v1"


def _git_head() -> str:
  try:
    import subprocess
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()
  except Exception:
    return "unknown"


def _probe_join(args) -> dict:
  st2._enter_lock()
  from tinygrad import Device

  dev = Device[Device.DEFAULT]
  src = st2._generic_source([("j_producer_p1", 100_000), ("j_producer_p2", 400_000)], ["st2_consumer2"])
  lib = st2._compile(dev, "nv_qmd_dependence_counter_join", src)

  p1 = st2._new_program(dev, lib, "j_producer_p1", None, False)
  p2 = st2._new_program(dev, lib, "j_producer_p2", None, False)
  cons = st2._new_program(dev, lib, "st2_consumer2", None, False)

  out1 = st2._alloc(dev, st2.NPROD * 4)
  out2 = st2._alloc(dev, st2.NPROD * 4)
  t_buf = st2._alloc(dev, 10 * 8)
  chk1 = st2._alloc(dev, 4)
  chk2 = st2._alloc(dev, 4)
  dev.synchronize()
  q = st2._make_queue(dev)
  t_init = st2._t_init(10, [3, 7])
  zeros = memoryview(b"\x00" * (st2.NPROD * 4))

  def build_and_submit(counter_join: bool):
    dev.allocator._copyin(out1, zeros)
    dev.allocator._copyin(out2, zeros)
    dev.allocator._copyin(t_buf, memoryview(t_init))
    dev.allocator._copyin(chk1, memoryview(b"\x00" * 4))
    dev.allocator._copyin(chk2, memoryview(b"\x00" * 4))
    dev.synchronize()

    q.exec(p1, p1.fill_kernargs((out1, t_buf), vals=(0,)), (st2.GS, 1, 1), (st2.LS, 1, 1))
    p1_qmd, p1_buf = q.active_qmd, q.active_qmd_buf
    q.exec(p2, p2.fill_kernargs((out2, t_buf), vals=(4,)), (st2.GS, 1, 1), (st2.LS, 1, 1))
    p2_qmd = q.active_qmd
    q.exec(cons, cons.fill_kernargs((out1, out2, chk1, chk2, t_buf), vals=(8,)), (1, 1, 1), (st2.LS, 1, 1))
    c_qmd, c_buf = q.active_qmd, q.active_qmd_buf

    if counter_join:
      # P1 schedules P2 on slot 0 (already written by exec) and decrements C on slot 1.
      p1_qmd.write(dependent_qmd1_pointer=c_buf.va_addr >> 8, dependent_qmd1_action=4,
                   dependent_qmd1_prefetch=1, dependent_qmd1_enable=1)
      # P2 already points at C on slot 0; switch that edge from schedule to decrement.
      p2_qmd.write(dependent_qmd0_action=4)
      c_qmd.write(dependence_counter=2)

    q.signal(dev.timeline_signal, dev.next_timeline())
    q.submit(dev)
    q._q = []
    q.active_qmd = None
    q.active_prg_name = None
    dev.synchronize(timeout=args.sync_timeout)
    return st2._read_u64s(dev, t_buf, 10), st2._read_u32(dev, chk1), st2._read_u32(dev, chk2)

  rows = []
  for phase, join in (("control", False), ("candidate", True), ("control", False)):
    for rep in range(args.warmup + args.reps):
      t, c1, c2 = build_and_submit(join)
      p1_start, p1_end, p1_last, _ = t[0], t[1], t[2], t[3]
      p2_start, p2_end, p2_last, _ = t[4], t[5], t[6], t[7]
      c_start, c_end = t[8], t[9]
      rows.append({
        "phase": phase,
        "warmup": rep < args.warmup,
        "p1_start_ns": p1_start, "p1_end_ns": p1_end, "p1_last_cta_ns": p1_last,
        "p2_start_ns": p2_start, "p2_end_ns": p2_end, "p2_last_cta_ns": p2_last,
        "c_start_ns": c_start, "c_end_ns": c_end,
        "c_vs_p1_end_us": round((c_start - p1_end) / 1000.0, 3),
        "c_vs_p2_end_us": round((c_start - p2_end) / 1000.0, 3),
        "c_vs_p2_last_cta_us": round((c_start - p2_last) / 1000.0, 3),
        "checksum1": c1, "checksum1_correct": c1 == st2.EXP_CONS,
        "checksum2": c2, "checksum2_correct": c2 == st2.EXP_CONS,
      })

  cand = [r for r in rows if r["phase"] == "candidate" and not r["warmup"]]
  chk_ok = bool(cand) and all(r["checksum1_correct"] and r["checksum2_correct"] for r in cand)
  med_vs_p2_end = st2._median([r["c_vs_p2_end_us"] for r in cand])
  med_vs_p2_last = st2._median([r["c_vs_p2_last_cta_us"] for r in cand])

  if not chk_ok:
    verdict, reason = "refuted", "consumer checksums failed: the counter join did not gate C on both producers"
  elif med_vs_p2_end is not None and med_vs_p2_end < -10.0:
    verdict, reason = "launch_ahead", (
      f"C started {abs(med_vs_p2_end)} us before P2 end with both checksums correct: "
      "the decrement fired at a pre-exit-style trigger, not at completion")
  elif med_vs_p2_last is not None and med_vs_p2_last >= -10.0:
    verdict, reason = "completion_join", (
      f"C started {med_vs_p2_end} us relative to P2 end ({med_vs_p2_last} vs P2 last CTA) with both "
      "checksums correct: counter=2 is a true completion-time join, not launch-ahead")
  else:
    verdict, reason = "named-unavailable", "counter join timing did not fit a clean completion or launch-ahead reading"

  return {
    "schema": SCHEMA,
    "probe": "dependence_counter_join",
    "verdict": verdict,
    "reason": reason,
    "arch": dev.arch,
    "device": Device.DEFAULT,
    "fields": {
      "p1": {"dependent_qmd1_pointer": "C", "dependent_qmd1_action": "QMD_DECREMENT_DEPENDENCE=4",
             "dependent_qmd1_prefetch": 1, "dependent_qmd1_enable": 1,
             "dependent_qmd0_pointer": "P2", "dependent_qmd0_action": "QMD_SCHEDULE=1"},
      "p2": {"dependent_qmd0_pointer": "C", "dependent_qmd0_action": "QMD_DECREMENT_DEPENDENCE=4"},
      "consumer": {"dependence_counter": 2},
    },
    "grid": {"GS": st2.GS, "LS": st2.LS, "spin_ns_p1": 100_000, "spin_ns_p2": 400_000,
             "NPROD": st2.NPROD, "NCONS": st2.NCONS},
    "expected_checksum": st2.EXP_CONS,
    "phases": {phase: st2._phase_summary([r for r in rows if r["phase"] == phase],
                                         ["c_vs_p1_end_us", "c_vs_p2_end_us", "c_vs_p2_last_cta_us"])
               for phase in ("control", "candidate")},
    "rows": rows,
  }


def _probe_non_consecutive(args) -> dict:
  st2._enter_lock()
  from tinygrad import Device

  dev = Device[Device.DEFAULT]
  src = st2._generic_source([("n_producer_a", 100_000), ("n_producer_b", 200_000)], ["st2_consumer"])
  lib = st2._compile(dev, "nv_qmd_dependence_counter_non_consecutive", src)

  a = st2._new_program(dev, lib, "n_producer_a", None, False)
  b = st2._new_program(dev, lib, "n_producer_b", None, False)
  c = st2._new_program(dev, lib, "st2_consumer", None, False)

  out_a = st2._alloc(dev, st2.NPROD * 4)
  out_b = st2._alloc(dev, st2.NPROD * 4)
  t_buf = st2._alloc(dev, 10 * 8)
  chk = st2._alloc(dev, 4)
  dev.synchronize()
  q = st2._make_queue(dev)
  t_init = st2._t_init(10, [3, 7])
  zeros = memoryview(b"\x00" * (st2.NPROD * 4))

  def build_and_submit(skip_edge: bool):
    dev.allocator._copyin(out_a, zeros)
    dev.allocator._copyin(out_b, zeros)
    dev.allocator._copyin(t_buf, memoryview(t_init))
    dev.allocator._copyin(chk, memoryview(b"\x00" * 4))
    dev.synchronize()

    q.exec(a, a.fill_kernargs((out_a, t_buf), vals=(0,)), (st2.GS, 1, 1), (st2.LS, 1, 1))
    a_qmd = q.active_qmd
    q.exec(b, b.fill_kernargs((out_b, t_buf), vals=(4,)), (st2.GS, 1, 1), (st2.LS, 1, 1))
    b_qmd = q.active_qmd
    q.exec(c, c.fill_kernargs((out_a, chk, t_buf), vals=(8,)), (1, 1, 1), (st2.LS, 1, 1))
    c_qmd, c_buf = q.active_qmd, q.active_qmd_buf

    if skip_edge:
      # A schedules B (slot 0, already written) and C (slot 1). Disable B's
      # schedule edge to C so C is only reachable through A's non-consecutive edge.
      a_qmd.write(dependent_qmd1_pointer=c_buf.va_addr >> 8, dependent_qmd1_action=1,
                  dependent_qmd1_prefetch=1, dependent_qmd1_enable=1)
      b_qmd.write(dependent_qmd0_enable=0)

    q.signal(dev.timeline_signal, dev.next_timeline())
    q.submit(dev)
    q._q = []
    q.active_qmd = None
    q.active_prg_name = None
    dev.synchronize(timeout=args.sync_timeout)
    return st2._read_u64s(dev, t_buf, 10), st2._read_u32(dev, chk)

  rows = []
  for phase, skip in (("control", False), ("candidate", True), ("control", False)):
    for rep in range(args.warmup + args.reps):
      t, cs = build_and_submit(skip)
      a_start, a_end, a_last, _ = t[0], t[1], t[2], t[3]
      b_start, b_end, _, _ = t[4], t[5], t[6], t[7]
      c_start, c_end = t[8], t[9]
      rows.append({
        "phase": phase,
        "warmup": rep < args.warmup,
        "a_start_ns": a_start, "a_end_ns": a_end, "a_last_cta_ns": a_last,
        "b_start_ns": b_start, "b_end_ns": b_end,
        "c_start_ns": c_start, "c_end_ns": c_end,
        "c_vs_a_end_us": round((c_start - a_end) / 1000.0, 3),
        "c_vs_b_start_us": round((c_start - b_start) / 1000.0, 3),
        "c_vs_b_end_us": round((c_start - b_end) / 1000.0, 3),
        "checksum": cs, "checksum_correct": cs == st2.EXP_CONS,
      })

  cand = [r for r in rows if r["phase"] == "candidate" and not r["warmup"]]
  chk_ok = bool(cand) and all(r["checksum_correct"] for r in cand)
  med_vs_b_end = st2._median([r["c_vs_b_end_us"] for r in cand])
  med_vs_a_end = st2._median([r["c_vs_a_end_us"] for r in cand])

  if not chk_ok:
    verdict, reason = "refuted", "C checksum failed: the non-consecutive schedule edge raced A's output"
  elif med_vs_b_end is not None and med_vs_b_end < -10.0:
    verdict, reason = "supported", (
      f"C started {abs(med_vs_b_end)} us before the middle kernel B ended ({med_vs_a_end} us after A end) with "
      "checksum correct: a dependent schedule edge skipped B")
  else:
    verdict, reason = "refuted", (
      f"C started {med_vs_b_end} us relative to B end ({med_vs_a_end} us after A end): "
      "the non-consecutive schedule edge did not skip B")

  return {
    "schema": SCHEMA,
    "probe": "non_consecutive_schedule_edge",
    "verdict": verdict,
    "reason": reason,
    "arch": dev.arch,
    "device": Device.DEFAULT,
    "fields": {
      "a": {"dependent_qmd0_pointer": "B", "dependent_qmd0_action": "QMD_SCHEDULE=1",
            "dependent_qmd1_pointer": "C", "dependent_qmd1_action": "QMD_SCHEDULE=1",
            "dependent_qmd1_prefetch": 1, "dependent_qmd1_enable": 1},
      "b": {"dependent_qmd0_enable": 0},
      "consumer": {"reads": "A only"},
    },
    "grid": {"GS": st2.GS, "LS": st2.LS, "spin_ns_a": 100_000, "spin_ns_b": 200_000,
             "NPROD": st2.NPROD, "NCONS": st2.NCONS},
    "expected_checksum": st2.EXP_CONS,
    "phases": {phase: st2._phase_summary([r for r in rows if r["phase"] == phase],
                                         ["c_vs_a_end_us", "c_vs_b_start_us", "c_vs_b_end_us"])
               for phase in ("control", "candidate")},
    "rows": rows,
  }


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--mode", required=True, choices=("join", "non_consecutive"))
  ap.add_argument("--reps", type=int, default=5)
  ap.add_argument("--warmup", type=int, default=1)
  ap.add_argument("--sync-timeout", type=int, default=15000)
  ap.add_argument("--out", type=pathlib.Path, required=True)
  args = ap.parse_args()

  payload = {"commit": _git_head(), "date": time.strftime("%Y-%m-%d"),
             "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
  wall0 = time.perf_counter()
  try:
    worker = _probe_join if args.mode == "join" else _probe_non_consecutive
    payload.update(worker(args))
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
  if payload.get("error"):
    print(payload["error"], file=sys.stderr)
  return rc


if __name__ == "__main__":
  raise SystemExit(main())
