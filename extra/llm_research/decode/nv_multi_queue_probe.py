#!/usr/bin/env python3
"""Native NV multi-compute-GPFIFO probe: can two compute channels run kernels concurrently?

Device-level P0 for the multi-compute-queue execution scope
(docs/task_workflow/input/nv-multi-compute-queue-execution-scope-20260803.md).
Creates extra compute GPFIFOs on the live NVDevice (closed-default slice:
persisted vaspace/ctxshare + `_new_gpu_fifo(debugger=)`), lowers kernels from
Tensor expressions, and executes them on hand-rolled ProbeComputeQueue instances
(NVComputeQueue with a per-instance GPFifo target). Timing uses HCQ timestamp
signals, the same primitive as the decode overlap measurement.

Answers, on this host (GB202, driver 595.84):
  E1: do cross-GPFIFO memory-semaphore dependencies work (numeric check)?
  E2: serial calibration (span must equal node-sum on one queue).
  E3/E4: do independent kernels on two/three compute channels overlap?
  E5: compute-heavy (matmul) flavor, to separate engine co-scheduling from DRAM
      contention on the elementwise flavor.

Usage:
  PYTHONPATH=/home/ubuntu/tinygrad-arkey python3 \
    extra/llm_research/decode/nv_multi_queue_probe.py [--out X.json] \
    [--engines 0,0] [--n 33554432]

Sequential GPU session required (house rule). No graph, no JIT, no model.
"""
from __future__ import annotations

import argparse, json, math, sys, time
sys.path.insert(0, "/home/ubuntu/tinygrad-arkey")

import numpy as np

from tinygrad import Tensor
from tinygrad.codegen import to_program
from tinygrad.device import BufferSpec, Device
from tinygrad.engine.realize import get_runtime
from tinygrad.runtime import ops_nv
from tinygrad.runtime.ops_nv import NVComputeQueue
from tinygrad.uop.ops import Ops, UOp


class ProbeComputeQueue(NVComputeQueue):
  """NVComputeQueue whose submits target a caller-provided GPFifo."""
  def __init__(self, gpfifo):
    self.gpfifo = gpfifo
    super().__init__()

  def _submit(self, dev): self._submit_to_gpfifo(dev, self.gpfifo)

  def submit(self, dev, var_vals=None):
    # HWQueue.submit does not clear _q, so a second submit re-sends the whole
    # accumulated command list. That is harmless in the graph (bound queues,
    # one submit per replay) but fatal for the probe: a stale entry re-executed
    # after its signal slots were freed and reused deadlocks the channel and
    # regresses the device timeline. Reset after every submit.
    super().submit(dev, var_vals)
    self._q = []


def lower(dev, t: Tensor):
  """Lower one Tensor expression to (NVProgram, program ast, arg buffer uops)."""
  linear, var_vals = Tensor.linear_with_vars(t)
  assert not var_vals, f"unexpected vars {var_vals}"
  assert len(linear.src) == 1, f"expected 1 program, got {len(linear.src)}"
  call = linear.src[0]
  ast = call.src[0]
  if ast.op is Ops.SINK: ast = to_program(ast, dev.renderer)
  assert ast.op is Ops.PROGRAM, f"expected PROGRAM, got {ast.op}"
  prg = get_runtime("NV", ast)
  arg_uops = [s for s in call.src[1:] if s.op is not Ops.BIND]
  assert len(arg_uops) == len(ast.arg.globals), (len(arg_uops), len(ast.arg.globals))
  return prg, ast, arg_uops


class Job:
  def __init__(self, q, prg, bufs, ast, waits=(), signals=(), grid=None):
    self.q, self.prg, self.bufs, self.ast, self.waits, self.signals, self.grid = q, prg, bufs, ast, tuple(waits), tuple(signals), grid


def run_jobs(dev, jobs):
  """Enqueue jobs (possibly across queues), submit, sync, return (start,end) us per job.

  Same-queue jobs are ordered by an in-order wait on a per-queue signal (one
  signal per queue, released only by that queue's channel, so it is monotonic
  by construction). Cross-queue jobs stay independent unless they carry
  explicit signal waits. Completion is a single device-timeline release on the
  first queue after it waits on every other queue's last per-queue target,
  mirroring HCQGraph (per-queue out_signals + one device-timeline bump per
  replay). Releasing the device timeline from every channel directly would be
  racy: releases land in arbitrary cross-channel order, the last writer wins,
  and the value can regress below the last issued target, hanging synchronize.
  """
  last_target: dict[int, int] = {}
  q_sig: dict[int, object] = {}
  ts: list[tuple[float, float]] = []
  for j in jobs:
    q = j.q
    qid = id(q)
    if qid not in q_sig: q_sig[qid] = dev.new_signal(value=0)
    if (lt := last_target.get(qid)) is not None: q.wait(q_sig[qid], lt)
    for sig, val in j.waits: q.wait(sig, val)
    st, en = dev.new_signal(), dev.new_signal()
    q.timestamp(st)
    gs, ls = j.ast.arg.launch_dims({})
    if j.grid is not None: gs = j.grid
    q.exec(j.prg, j.prg.fill_kernargs(tuple(j.bufs), vals=j.ast.arg.vals({})), gs or (1, 1, 1), ls or (1, 1, 1))
    q.timestamp(en)
    for sig, val in j.signals: q.signal(sig, val)
    target = last_target[qid] = (last_target.get(qid) or 0) + 1
    q.signal(q_sig[qid], target)
    ts.append((st, en))
  # Join: first queue waits on every other queue's last per-queue target, then
  # releases the device timeline once. This keeps the device timeline
  # monotonic; every later wait (copyin/copyout, dev.synchronize) is safe.
  join_q = jobs[0].q
  for qid, sig in q_sig.items():
    if qid != id(join_q): join_q.wait(sig, last_target[qid])
  join_q.signal(dev.timeline_signal, dev.next_timeline())
  # Submit each distinct queue exactly once, in first-use order. The join
  # commands were appended to the first job's queue, so they ride in that
  # queue's single ring entry, ordered after its job. Re-submitting the same
  # queue (or an empty queue) appends a duplicate/empty GPFIFO entry, which
  # stalls the frontend and deadlocks the next dispatch.
  seen: list[int] = []
  for j in jobs:
    if id(j.q) not in seen:
      j.q.submit(dev)
      seen.append(id(j.q))
  dev.synchronize(timeout=15000)
  return [(float(s.timestamp), float(e.timestamp)) for s, e in ts]


def durations(ts):
  return [max(0.0, en - st) for st, en in ts]


def span(ts):
  return max(en for _, en in ts) - min(st for st, _ in ts)


def make_queues(dev, gpfifos):
  queues = []
  for gf in gpfifos:
    q = ProbeComputeQueue(gf)
    q.setup(compute_class=dev.iface.compute_class, local_mem_window=dev.local_mem_window,
            shared_mem_window=dev.shared_mem_window)
    q.wait(dev.timeline_signal, 0).memory_barrier()
    queues.append(q)
  return queues


def alloc_buffers(dev, *uop_lists):
  """One HCQBuffer per unique arg uop, so shared buffers map to one allocation."""
  mapping: dict[UOp, object] = {}
  for uops in uop_lists:
    for u in uops:
      if u not in mapping:
        mapping[u] = dev.allocator._alloc(int(np.prod(u.shape)) * u.dtype.itemsize, BufferSpec())
  return mapping


def out_uop(ast, arg_uops):
  return arg_uops[ast.arg.outs[0]]


def copyin(dev, buf, data):
  dev.allocator._copyin(buf, memoryview(np.ascontiguousarray(data, dtype=np.float32).tobytes()))


def copyout(dev, buf, n):
  blob = memoryview(bytearray(int(np.prod(n)) * 4))
  dev.allocator._copyout(blob, buf)
  return np.frombuffer(blob, dtype=np.float32).copy()


def extra_gpfifos(dev, engine_types, separate_ctxshare=False):
  """Create one additional compute GPFifo per engineType; returns (fifos, errors).

  With separate_ctxshare, each extra channel gets its own FERMI_CONTEXT_SHARE_A
  (same vaspace). Channels inside one ctxshare may be serialized by the RM to
  preserve context state; separate context shares exercise the per-context
  parallel path CUDA uses for stream concurrency.
  """
  n = len(engine_types)
  if n == 0: return [], []
  area = dev.iface.alloc(0x200000 * n, contiguous=True, cpu_access=True, force_devmem=True,
                         map_flags=(ops_nv.nv_gpu.NVOS33_FLAGS_CACHING_TYPE_WRITECOMBINED << 23))
  fifos, errors = [], []
  for i, engine_type in enumerate(engine_types):
    try:
      ctxshare = dev.ctxshare
      if separate_ctxshare:
        ctxshare_params = ops_nv.nv_gpu.NV_CTXSHARE_ALLOCATION_PARAMETERS(
          hVASpace=dev.vaspace, flags=ops_nv.nv_gpu.NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC)
        ctxshare = dev.iface.rm_alloc(dev.channel_group, ops_nv.nv_gpu.FERMI_CONTEXT_SHARE_A, ctxshare_params)
      fifos.append(dev._new_gpu_fifo(area, ctxshare, dev.channel_group, offset=0x100000 * i,
                                     entries=0x4000, compute=True, debugger=False,
                                     engine_type=engine_type))
    except RuntimeError as e:
      errors.append({"engine_type": engine_type, "error": str(e)})
  if fifos:
    # The group-level GPFIFO schedule was issued at device init, before these
    # channels existed. Re-issue it so newly created channels (especially with
    # separate ctxshares) are actually scheduled onto the runlist.
    dev.iface.rm_control(dev.channel_group, ops_nv.nv_gpu.NVA06C_CTRL_CMD_GPFIFO_SCHEDULE,
                         ops_nv.nv_gpu.NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS(bEnable=1))
  return fifos, errors


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", type=str, default=None)
  ap.add_argument("--engines", type=str, default="0,0", help="comma list of engineType per extra GPFifo")
  ap.add_argument("--n", type=int, default=1 << 25, help="elementwise vector length")
  ap.add_argument("--matmul", type=int, default=2048, help="matmul NxNxN for the compute-heavy flavor")
  ap.add_argument("--stop-after", type=int, default=5, help="stop after experiment N (1-5)")
  ap.add_argument("--grid-div", type=int, default=1,
                  help="divide the E3/E4 elementwise grid by this factor; partial-SM kernels make "
                       "overlap physically possible (full-grid kernels saturate every SM either way)")
  ap.add_argument("--separate-ctxshare", action="store_true",
                  help="give every extra channel its own context share instead of sharing the device's")
  args = ap.parse_args()

  t0 = time.perf_counter()
  dev = Device["NV"]
  dev.synchronize()
  engines = [int(x) for x in args.engines.split(",") if x != ""]
  extra, fifo_errors = extra_gpfifos(dev, engines, separate_ctxshare=args.separate_ctxshare)
  gpfifos = [dev.compute_gpfifo, *extra]
  qs = make_queues(dev, gpfifos)
  print(f"device ready {time.perf_counter()-t0:.2f}s, compute gpfifos={len(gpfifos)} "
        f"engineTypes={[0, *engines]} errors={fifo_errors}", file=sys.stderr, flush=True)

  results = []
  N = args.n

  # --- E1: cross-GPFIFO semaphore dependency correctness ---------------------
  a = Tensor.empty(1 << 20, device="NV"); b = Tensor.empty(1 << 20, device="NV")
  c = a * b
  mul_prg, mul_ast, mul_args = lower(dev, c)
  b2 = Tensor.empty(1 << 20, device="NV")
  e = c + b2
  add_prg, add_ast, add_args = lower(dev, e)
  bufs = alloc_buffers(dev, mul_args, add_args)
  copyin(dev, bufs[a.uop], np.arange(1 << 20, dtype=np.float32))
  copyin(dev, bufs[b.uop], np.linspace(0.5, 2.0, 1 << 20, dtype=np.float32))
  copyin(dev, bufs[b2.uop], np.full(1 << 20, 1.0, dtype=np.float32))
  out_sig = dev.new_signal(value=0)
  e1_ts = run_jobs(dev, [
    Job(qs[0], mul_prg, [bufs[u] for u in mul_args], mul_ast, signals=((out_sig, 1),)),
    Job(qs[1], add_prg, [bufs[u] for u in add_args], add_ast, waits=((out_sig, 1),)),
  ])
  out_c = copyout(dev, bufs[out_uop(mul_ast, mul_args)], 1 << 20)
  out_e = copyout(dev, bufs[out_uop(add_ast, add_args)], 1 << 20)
  exp_c = np.arange(1 << 20, dtype=np.float32) * np.linspace(0.5, 2.0, 1 << 20, dtype=np.float32)
  exp_e = exp_c + 1.0
  e1_ok = np.allclose(out_c, exp_c, rtol=1e-3) and np.allclose(out_e, exp_e, rtol=1e-3)
  results.append({"name": "E1", "status": "pass" if e1_ok else "FAIL", "check": "cross-gpfifo semaphore dep",
                  "max_err_c": float(np.abs(out_c - exp_c).max()), "max_err_e": float(np.abs(out_e - exp_e).max())})
  print(f"E1 {'PASS' if e1_ok else 'FAIL'} cross-gpfifo dep (max err {float(np.abs(out_c-exp_c).max()):.2e}, "
        f"{float(np.abs(out_e-exp_e).max()):.2e})", file=sys.stderr, flush=True)
  if args.stop_after == 1: sys.exit(0)

  # --- E2: serial calibration (dependent chain, one queue) -------------------
  x1 = Tensor.empty(N, device="NV"); x2 = Tensor.empty(N, device="NV")
  y = x1 * x2
  s_prg, s_ast, s_args = lower(dev, y)
  y2 = Tensor.empty(N, device="NV")
  z = y + y2
  s2_prg, s2_ast, s2_args = lower(dev, z)
  s_bufs = alloc_buffers(dev, s_args, s2_args)
  copyin(dev, s_bufs[x1.uop], np.full(N, 2.0, dtype=np.float32))
  copyin(dev, s_bufs[x2.uop], np.full(N, 3.0, dtype=np.float32))
  copyin(dev, s_bufs[y2.uop], np.full(N, 1.0, dtype=np.float32))
  e2_ts = run_jobs(dev, [
    Job(qs[0], s_prg, [s_bufs[u] for u in s_args], s_ast),
    Job(qs[0], s2_prg, [s_bufs[u] for u in s2_args], s2_ast),
  ])
  d2 = durations(e2_ts); e2_span = span(e2_ts); e2_sum = sum(d2)
  e2_overlap = (e2_sum - e2_span) / e2_sum if e2_sum else 0.0
  results.append({"name": "E2", "status": "pass" if e2_overlap < 0.05 else "FAIL", "check": "serial calibration",
                  "dur_us": d2, "span_us": e2_span, "node_sum_us": e2_sum, "overlap": e2_overlap})
  print(f"E2 serial span={e2_span:.1f}us sum={e2_sum:.1f}us overlap={e2_overlap*100:.1f}% "
        f"durs={[f'{d:.1f}' for d in d2]}", file=sys.stderr, flush=True)
  if args.stop_after == 2: sys.exit(0)

  # --- E3: two independent elementwise kernels on two queues -----------------
  jobs3, q3_bufs = [], {}
  for qi in (0, 1):
    ta = Tensor.empty(N, device="NV"); tb = Tensor.empty(N, device="NV")
    tc = ta * tb
    p, ast, uops = lower(dev, tc)
    bb = alloc_buffers(dev, uops)
    copyin(dev, bb[ta.uop], np.full(N, float(qi + 1), dtype=np.float32))
    copyin(dev, bb[tb.uop], np.full(N, 2.0, dtype=np.float32))
    q3_bufs[qi] = bb
    g3, _ = ast.arg.launch_dims({})
    jobs3.append(Job(qs[qi], p, [bb[u] for u in uops], ast,
                     grid=(g3[0] // args.grid_div, g3[1], g3[2]) if args.grid_div > 1 else None))
  e3_ts = run_jobs(dev, jobs3)
  d3 = durations(e3_ts); e3_span = span(e3_ts); e3_sum = sum(d3)
  e3_overlap = (e3_sum - e3_span) / e3_sum if e3_sum else 0.0
  results.append({"name": "E3", "status": "pass" if e3_overlap >= 0.05 else "FAIL", "check": "2-queue elementwise overlap",
                  "dur_us": d3, "span_us": e3_span, "node_sum_us": e3_sum, "overlap": e3_overlap})
  print(f"E3 2-queue span={e3_span:.1f}us sum={e3_sum:.1f}us overlap={e3_overlap*100:.1f}% "
        f"durs={[f'{d:.1f}' for d in d3]}", file=sys.stderr, flush=True)
  if args.stop_after == 3: sys.exit(0)

  # --- E4: three independent kernels on three queues -------------------------
  if len(qs) >= 3:
    jobs4 = []
    for qi in (0, 1, 2):
      ta = Tensor.empty(N, device="NV"); tb = Tensor.empty(N, device="NV")
      tc = ta * tb
      p, ast, uops = lower(dev, tc)
      bb = alloc_buffers(dev, uops)
      copyin(dev, bb[ta.uop], np.full(N, float(qi + 1), dtype=np.float32))
      copyin(dev, bb[tb.uop], np.full(N, 2.0, dtype=np.float32))
      g4, _ = ast.arg.launch_dims({})
      jobs4.append(Job(qs[qi], p, [bb[u] for u in uops], ast,
                       grid=(g4[0] // args.grid_div, g4[1], g4[2]) if args.grid_div > 1 else None))
    e4_ts = run_jobs(dev, jobs4)
    d4 = durations(e4_ts); e4_span = span(e4_ts); e4_sum = sum(d4)
    e4_overlap = (e4_sum - e4_span) / e4_sum if e4_sum else 0.0
    results.append({"name": "E4", "status": "pass" if e4_overlap >= 0.05 else "FAIL", "check": "3-queue elementwise overlap",
                    "dur_us": d4, "span_us": e4_span, "node_sum_us": e4_sum, "overlap": e4_overlap})
    print(f"E4 3-queue span={e4_span:.1f}us sum={e4_sum:.1f}us overlap={e4_overlap*100:.1f}% "
          f"durs={[f'{d:.1f}' for d in d4]}", file=sys.stderr, flush=True)
    if args.stop_after == 4: sys.exit(0)
  else:
    results.append({"name": "E4", "status": "skipped", "check": "need 3 gpfifos"})

  # --- E5: compute-heavy (matmul) chains on two queues -----------------------
  M = args.matmul
  jobs5, m_bufs, m_asts, m_argss = [], {}, {}, {}
  for qi in (0, 1):
    ma = Tensor.empty(M, M, device="NV"); mb = Tensor.empty(M, M, device="NV")
    mc = ma @ mb
    linear, var_vals = Tensor.linear_with_vars(mc)
    calls = [si for si in linear.src]
    assert len(calls) == 1, f"matmul lowered to {len(calls)} programs"
    m_ast = calls[0].src[0]
    if m_ast.op is Ops.SINK: m_ast = to_program(m_ast, dev.renderer)
    m_prg = get_runtime("NV", m_ast)
    m_args = [s for s in calls[0].src[1:] if s.op is not Ops.BIND]
    bb = alloc_buffers(dev, m_args)
    # Matmul wraps its inputs in RESHAPE uops, so fill by argument slot. Both
    # inputs get sqrt(k)*I; A*B == k*I for any input order, which pins the check.
    for slot in m_ast.arg.ins:
      copyin(dev, bb[m_args[slot]], np.eye(M, dtype=np.float32) * math.sqrt(float(qi + 1)))
    m_bufs[qi], m_asts[qi], m_argss[qi] = bb, m_ast, m_args
    jobs5.append(Job(qs[qi], m_prg, [bb[u] for u in m_args], m_ast))
  e5_ts = run_jobs(dev, jobs5)
  d5 = durations(e5_ts); e5_span = span(e5_ts); e5_sum = sum(d5)
  e5_overlap = (e5_sum - e5_span) / e5_sum if e5_sum else 0.0
  out5 = [copyout(dev, m_bufs[qi][out_uop(m_asts[qi], m_argss[qi])], (M, M)) for qi in (0, 1)]
  e5_ok = all(np.allclose(out5[qi].reshape(M, M), np.eye(M, dtype=np.float32) * float(qi + 1), rtol=1e-2) for qi in (0, 1))
  results.append({"name": "E5", "status": "pass" if (e5_overlap >= 0.05 and e5_ok) else "FAIL",
                  "check": "2-queue matmul overlap + correctness", "M": M,
                  "dur_us": d5, "span_us": e5_span, "node_sum_us": e5_sum, "overlap": e5_overlap, "numeric_ok": e5_ok})
  print(f"E5 2-queue matmul span={e5_span:.1f}us sum={e5_sum:.1f}us overlap={e5_overlap*100:.1f}% "
        f"durs={[f'{d:.1f}' for d in d5]} numeric_ok={e5_ok}", file=sys.stderr, flush=True)

  payload = {
    "schema": "tinygrad.nv_multi_queue_probe.v1",
    "device": "NV sm_120 RTX 5090", "n": N, "matmul": M,
    "gpfifo_engine_types": [0, *engines], "gpfifo_creation_errors": fifo_errors,
    "verdict": "PASS" if all(r.get("status") in ("pass", "skipped") for r in results) else "FAIL",
    "experiments": results,
  }
  if args.out:
    with open(args.out, "w", encoding="utf-8") as f: json.dump(payload, f, indent=2)
  print(json.dumps(payload, indent=2))


if __name__ == "__main__":
  main()
