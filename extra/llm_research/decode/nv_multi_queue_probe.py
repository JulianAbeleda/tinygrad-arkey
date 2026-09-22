#!/usr/bin/env python3
"""Native NV multi-compute-GPFIFO probe: can two compute channels run kernels concurrently?

Device-level Phase 0 for the NV decode overlap implementation scope
(docs/task_workflow/input/nv-decode-overlap-implementation-scope-20260803.md).
Creates extra compute GPFIFOs on the live NVDevice (closed-default slice:
persisted vaspace/ctxshare + `_new_gpu_fifo(debugger=)`), lowers kernels from
Tensor expressions, and executes them on hand-rolled ProbeComputeQueue instances
(NVComputeQueue with a per-instance GPFifo target). Timing uses HCQ timestamp
signals, the same primitive as the decode overlap measurement.

Construction modes (see build_construction_plan):
  shared   control arm: all extra channels under dev.ctxshare in dev.channel_group,
           exactly the E1-E5 construction, with every RM operation recorded.
  ctxshare H1/H2: fresh FERMI_CONTEXT_SHARE_A per channel under dev.channel_group,
           then per-channel NVA06F bind/schedule, then a group-level NVA06C schedule
           on dev.channel_group.
  group    H3: fresh KEPLER_CHANNEL_GROUP_A per channel on dev.nvdevice with its own
           ctxshare, per-channel NVA06F bind/schedule, and the group-level NVA06C
           schedule on that fresh group (never dev.channel_group).

Every RM operation is recorded in an ordered rm_ops list ({op, kind, group, channel,
engine_type, status, error}) and the partial JSON payload is rewritten to --out after
each one, so a timed-out or failed arm still leaves an anchored record.

Answers, on this host (GB202, driver 595.84):
  R1: do cross-GPFIFO memory-semaphore dependencies work (anchored sha256 hash +
      max-error contract, not np.allclose)?
  R2: serial calibration (span must equal node-sum on one queue inside a declared
      timestamp tolerance).
  R3/R4: do independent kernels on two/three compute channels overlap (elementwise;
      full and partial-SM grids via --grid-div)?
  R5: compute-heavy (matmul 2048) flavor, to separate engine co-scheduling from DRAM
      contention on the elementwise flavor.

Usage:
  single arm (debugging):
    PYTHONPATH=/home/ubuntu/tinygrad-arkey python3 \
      extra/llm_research/decode/nv_multi_queue_probe.py --mode {shared,ctxshare,group} \
      [--out X.json] [--engines 0,0] [--n 33554432] [--grid-div 4] [--stop-after 5]
  driver (all three arms, each in a fresh subprocess with a hard timeout):
    PYTHONPATH=/home/ubuntu/tinygrad-arkey python3 \
      extra/llm_research/decode/nv_multi_queue_probe.py --run-all --out X.json \
      [--timeout 600] [--engines 0,0] [--grid-div 4]

Sequential GPU session required (house rule). No graph, no JIT, no model.
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, subprocess, sys, time
sys.path.insert(0, "/home/ubuntu/tinygrad-arkey")

import numpy as np

from tinygrad import Tensor
from tinygrad.codegen import to_program
from tinygrad.device import BufferSpec, Device
from tinygrad.engine.realize import get_runtime
from tinygrad.runtime import ops_nv
from tinygrad.runtime.ops_nv import GPFifo, NVComputeQueue
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
    if qid not in q_sig:
      # The probe queues persist across R1-R5 whereas every input copyin advances
      # the device timeline.  Their setup-time wait alone only guards timeline 0;
      # add the current frontier before this batch so an independent row cannot
      # read a still-in-flight DMA copy.  This is outside the timestamps.
      q.wait(dev.timeline_signal, dev.timeline_value - 1)
      q_sig[qid] = dev.new_signal(value=0)
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


def f32_sha(arr):
  return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.float32).tobytes()).hexdigest()


def r1_contract_pass(actual_hashes, ref_hashes, max_errs, ref_abs_maxes, tol=1e-3):
  """R1 'exact' verdict: anchored output hashes must equal the CPU reference hashes
  AND every output's max absolute error must be within tol * max(1, max|ref|).
  The verdict is never np.allclose."""
  if tuple(actual_hashes) != tuple(ref_hashes): return False
  return all(me <= tol * max(1.0, ra) for me, ra in zip(max_errs, ref_abs_maxes))


def serial_contract_ok(span_us, node_sum_us, max_pct=2.0, max_abs_us=10.0):
  """R2 serial-calibration verdict: span equals node-sum inside a declared timestamp
  tolerance (|pct_delta| <= max_pct OR abs_delta <= max_abs_us); never float equality."""
  pct = (abs(span_us - node_sum_us) / node_sum_us * 100.0) if node_sum_us else float("inf")
  return pct <= max_pct or abs(span_us - node_sum_us) <= max_abs_us


BOOT_GROUP = "boot_group"
BOOT_CTXSHARE = "boot_ctxshare"


def build_construction_plan(mode: str, engine_types: list[int], bind_policy: str = "required") -> list[dict]:
  """Ordered RM construction op plan for one arm (pure, no device needed).

  Group/ctxshare/channel refs are symbolic: BOOT_GROUP is dev.channel_group,
  BOOT_CTXSHARE is dev.ctxshare, and 'group:<i>' / 'ctxshare:<i>' / 'channel:<i>'
  are the objects created for engine i. Ops tagged requires_channel (resp.
  requires_any_channel) are skipped by the executor when that channel (resp. every
  channel) failed construction, so the failing step is the last record for it.
  bind_policy="required" issues NVA06F_CTRL_CMD_BIND per channel (amendment H1/H3
  sequence); "skip" omits it. RM evidence on driver 595.84 rejects BIND for
  group-allocated compute channels with NV_ERR_INVALID_ARGUMENT, so "skip" tests
  whether per-channel GPFIFO_SCHEDULE + group-level schedule alone co-schedules.
  """
  assert bind_policy in ("required", "skip"), bind_policy
  if mode in ("fresh_cuda_group", "fresh_cuda_group_ctxbuf", "fresh_cuda_group_cuda_params", "fresh_cuda_group_cuda_params_notifier4k"):
    # Exact topology of CUDA's first graphics group: one fresh group (graphics
    # engine), one fresh async ctxshare, all child GPFIFOs, then one schedule.
    plan = [
      {"op": "CHANNEL_GROUP_ALLOC", "kind": "KEPLER_CHANNEL_GROUP_A", "group": "cuda_group", "parent": "nvdevice",
       "engine_index": None, "engine_type": None},
      {"op": "CTXSHARE_ALLOC", "kind": "FERMI_CONTEXT_SHARE_A", "group": "cuda_group", "ctxshare": "cuda_ctxshare",
       "engine_index": None, "engine_type": None},
    ]
    for i, engine_type in enumerate(engine_types):
      plan.append({"op": "CHANNEL_ALLOC", "kind": "NVA06F", "group": "cuda_group", "ctxshare": "cuda_ctxshare",
                   "channel": f"channel:{i}", "engine_index": i, "engine_type": engine_type, "flags": 0x10 if i % 2 else 0})
    if engine_types:
      plan.append({"op": "NVA06C_GPFIFO_SCHEDULE", "kind": "NVA06C", "group": "cuda_group", "engine_index": None,
                   "engine_type": None, "requires_any_channel": True})
    return plan
  plan: list[dict] = []
  for i, engine_type in enumerate(engine_types):
    if mode in ("shared", "cuda_mirror"):
      plan.append({"op": "CHANNEL_ALLOC", "kind": "NVA06F", "group": BOOT_GROUP, "ctxshare": BOOT_CTXSHARE,
                   "channel": f"channel:{i}", "engine_index": i, "engine_type": engine_type})
    elif mode == "ctxshare":
      plan += [
        {"op": "CTXSHARE_ALLOC", "kind": "FERMI_CONTEXT_SHARE_A", "group": BOOT_GROUP, "ctxshare": f"ctxshare:{i}",
         "engine_index": i, "engine_type": engine_type},
        {"op": "CHANNEL_ALLOC", "kind": "NVA06F", "group": BOOT_GROUP, "ctxshare": f"ctxshare:{i}", "channel": f"channel:{i}",
         "engine_index": i, "engine_type": engine_type},
      ]
      if bind_policy == "required":
        plan += [{"op": "NVA06F_BIND", "kind": "NVA06F", "group": BOOT_GROUP, "channel": f"channel:{i}",
                  "engine_index": i, "engine_type": engine_type, "requires_channel": i}]
      plan += [
        {"op": "NVA06F_GPFIFO_SCHEDULE", "kind": "NVA06F", "group": BOOT_GROUP, "channel": f"channel:{i}",
         "engine_index": i, "engine_type": engine_type, "requires_channel": i},
        {"op": "NVA06C_GPFIFO_SCHEDULE", "kind": "NVA06C", "group": BOOT_GROUP,
         "engine_index": i, "engine_type": engine_type, "requires_channel": i},
      ]
    elif mode == "group":
      plan += [
        {"op": "CHANNEL_GROUP_ALLOC", "kind": "KEPLER_CHANNEL_GROUP_A", "group": f"group:{i}", "parent": "nvdevice",
         "engine_index": i, "engine_type": engine_type},
        {"op": "CTXSHARE_ALLOC", "kind": "FERMI_CONTEXT_SHARE_A", "group": f"group:{i}", "ctxshare": f"ctxshare:{i}",
         "engine_index": i, "engine_type": engine_type},
        {"op": "CHANNEL_ALLOC", "kind": "NVA06F", "group": f"group:{i}", "ctxshare": f"ctxshare:{i}", "channel": f"channel:{i}",
         "engine_index": i, "engine_type": engine_type},
      ]
      if bind_policy == "required":
        plan += [{"op": "NVA06F_BIND", "kind": "NVA06F", "group": f"group:{i}", "channel": f"channel:{i}",
                  "engine_index": i, "engine_type": engine_type, "requires_channel": i}]
      plan += [
        {"op": "NVA06F_GPFIFO_SCHEDULE", "kind": "NVA06F", "group": f"group:{i}", "channel": f"channel:{i}",
         "engine_index": i, "engine_type": engine_type, "requires_channel": i},
        {"op": "NVA06C_GPFIFO_SCHEDULE", "kind": "NVA06C", "group": f"group:{i}",
         "engine_index": i, "engine_type": engine_type, "requires_channel": i},
      ]
    else:
      raise ValueError(f"unknown mode {mode!r}")
  if mode == "cuda_mirror":
    # CUDA's first observed graphics group is constructed as one async ctxshare,
    # then all GPFIFOs are allocated before its one group schedule. Its stream
    # channels alternate the GROUP_CHANNEL_RUNQUEUE bit (0x10). The existing
    # native device has already created/scheduled bootstrap channels, so this
    # arm first disables the group and recreates that schedule boundary in a
    # fresh subprocess. It is deliberately probe-only and default-off.
    plan.insert(0, {"op": "NVA06C_GPFIFO_UNSCHEDULE", "kind": "NVA06C", "group": BOOT_GROUP,
                    "engine_index": None, "engine_type": None})
    for i, op in enumerate(p for p in plan if p["op"] == "CHANNEL_ALLOC"):
      op["flags"] = 0x10 if i % 2 else 0
  if mode in ("shared", "cuda_mirror") and engine_types:
    # Group-level re-schedule, exactly like the E1-E5 probe: only when at least one
    # extra channel was actually created.
    plan.append({"op": "NVA06C_GPFIFO_SCHEDULE", "kind": "NVA06C", "group": BOOT_GROUP,
                 "engine_index": None, "engine_type": None, "requires_any_channel": True})
  return plan


def _j(v):
  return int(v) if v is not None else None


def _resolve_group(ref, dev, handles):
  if ref is None: return None
  if ref == BOOT_GROUP: return dev.channel_group
  if ref == BOOT_CTXSHARE: return dev.ctxshare
  # .get: a CHANNEL_GROUP_ALLOC op resolves its own (not yet created) ref to None;
  # the record for that op is emitted with the new handle explicitly.
  return handles.get(ref)


def _resolve_channel(ref, handles):
  # .get: a CHANNEL_ALLOC op resolves its own (not yet created) ref to None; the
  # record for that op is emitted with the new handle explicitly.
  return None if ref is None else handles.get(ref)


def extra_gpfifos(dev, engine_types, mode="shared", on_rm_op=None, bind_policy="required", channel_flags=None):
  """One additional compute GPFifo per engineType under the selected construction mode.

  Executes build_construction_plan against `dev` (a live NVDevice or an RM/GPU-free
  fake with the same surface: iface.alloc/rm_alloc/rm_control + _new_gpu_fifo).
  Every RM operation is reported to on_rm_op immediately as
  {op, kind, group, channel, engine_type, status, error}; the caller flushes after
  each one. Returns (fifos, errors) where fifos holds only channels whose full
  per-channel plan completed without error; a failing step is recorded in errors.
  All NVA06F/NVA06C controls target raw RM channel/group handles, never a GPFifo
  wrapper (rm_control(GPFifo(...), ...) is forbidden).
  """
  if not engine_types: return [], []
  on_rm_op = on_rm_op or (lambda rec: None)
  plan = build_construction_plan(mode, engine_types, bind_policy=bind_policy)
  cuda_ctxbuf = mode in ("fresh_cuda_group_ctxbuf", "fresh_cuda_group_cuda_params", "fresh_cuda_group_cuda_params_notifier4k")
  cuda_subctx = mode in ("fresh_cuda_group_cuda_params", "fresh_cuda_group_cuda_params_notifier4k")
  cuda_notifier4k = mode == "fresh_cuda_group_cuda_params_notifier4k"
  if channel_flags is not None:
    if len(channel_flags) != len(engine_types): raise ValueError("--channel-flags count must equal --engines count")
    for op, flags in zip((p for p in plan if p["op"] == "CHANNEL_ALLOC"), channel_flags): op["flags"] = flags
  area = dev.iface.alloc(0x200000 * len(engine_types), contiguous=True, cpu_access=True, force_devmem=True,
                         map_flags=(ops_nv.nv_gpu.NVOS33_FLAGS_CACHING_TYPE_WRITECOMBINED << 23))
  handles: dict[str, object] = {}
  ok_channels: set[int] = set()
  fifos_by_channel: dict[int, GPFifo] = {}
  errors: list[dict] = []
  deferred_schedule = None

  def record(op, group, channel, status, error=None):
    on_rm_op({"op": op["op"], "kind": op.get("kind"), "group": _j(group), "channel": _j(channel),
              "engine_type": op.get("engine_type"), "status": status, "error": error})

  for op in plan:
    group = _resolve_group(op.get("group"), dev, handles)
    channel = _resolve_channel(op.get("channel"), handles)
    if op.get("requires_channel") is not None and op["requires_channel"] not in ok_channels: continue
    if op.get("requires_any_channel") and not ok_channels: continue
    if cuda_ctxbuf and op["op"] == "NVA06C_GPFIFO_SCHEDULE":
      # CUDA queries/registers every child context before its one group schedule.
      deferred_schedule = (op, group)
      continue
    try:
      record_channel = channel
      if op["op"] == "CHANNEL_GROUP_ALLOC":
        params = ops_nv.nv_gpu.NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS(engineType=ops_nv.nv_gpu.NV2080_ENGINE_TYPE_GRAPHICS)
        group = dev.iface.rm_alloc(dev.nvdevice, ops_nv.nv_gpu.KEPLER_CHANNEL_GROUP_A, params)
        handles[op["group"]] = group
        record_channel = None
      elif op["op"] == "CTXSHARE_ALLOC":
        params = ops_nv.nv_gpu.NV_CTXSHARE_ALLOCATION_PARAMETERS(hVASpace=dev.vaspace,
          flags=ops_nv.nv_gpu.NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC,
          **({"subctxId": 63} if cuda_subctx else {}))
        handles[op["ctxshare"]] = dev.iface.rm_alloc(group, ops_nv.nv_gpu.FERMI_CONTEXT_SHARE_A, params)
        record_channel = None
      elif op["op"] == "CHANNEL_ALLOC":
        i = op["engine_index"]
        fifo = dev._new_gpu_fifo(area, _resolve_group(op.get("ctxshare"), dev, handles), group,
                                 offset=0x100000 * i, entries=0x4000, compute=True, debugger=False,
                                 engine_type=op["engine_type"], flags=op.get("flags", 0), register_vm=not cuda_ctxbuf,
                                 error_notifier_size=4096 if cuda_notifier4k else 48 << 20)
        handles[op["channel"]] = fifo.handle
        fifos_by_channel[i] = fifo
        ok_channels.add(i)
        record_channel = fifo.handle
      elif op["op"] == "NVA06F_BIND":
        dev.iface.rm_control(channel, ops_nv.nv_gpu.NVA06F_CTRL_CMD_BIND,
                             ops_nv.nv_gpu.NVA06F_CTRL_BIND_PARAMS(engineType=op["engine_type"]))
      elif op["op"] == "NVA06F_GPFIFO_SCHEDULE":
        dev.iface.rm_control(channel, ops_nv.nv_gpu.NVA06F_CTRL_CMD_GPFIFO_SCHEDULE,
                             ops_nv.nv_gpu.NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS(bEnable=1))
      elif op["op"] == "NVA06C_GPFIFO_SCHEDULE":
        dev.iface.rm_control(group, ops_nv.nv_gpu.NVA06C_CTRL_CMD_GPFIFO_SCHEDULE,
                             ops_nv.nv_gpu.NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS(bEnable=1))
        record_channel = None
      elif op["op"] == "NVA06C_GPFIFO_UNSCHEDULE":
        dev.iface.rm_control(group, ops_nv.nv_gpu.NVA06C_CTRL_CMD_GPFIFO_SCHEDULE,
                             ops_nv.nv_gpu.NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS(bEnable=0))
        record_channel = None
      else:
        raise AssertionError(f"unhandled plan op {op['op']!r}")
      record(op, group, record_channel, "ok")
    except (RuntimeError, MemoryError) as e:
      record(op, group, channel, "error", str(e))
      errors.append({"op": op["op"], "engine_index": op.get("engine_index"), "engine_type": op.get("engine_type"), "error": str(e)})
      if op.get("engine_index") is not None:
        ok_channels.discard(op["engine_index"])
        fifos_by_channel.pop(op["engine_index"], None)
  if cuda_ctxbuf and fifos_by_channel:
    sizes = []
    for i, fifo in fifos_by_channel.items():
      try:
        p = dev.iface.rm_control(dev.subdevice, ops_nv.nv_gpu.NV2080_CTRL_CMD_GR_GET_CTX_BUFFER_SIZE,
          ops_nv.nv_gpu.NV2080_CTRL_GR_GET_CTX_BUFFER_SIZE_PARAMS(hChannel=fifo.handle))
        sizes.append(int(p.totalBufferSize))
        on_rm_op({"op": "GR_GET_CTX_BUFFER_SIZE", "kind": "NV2080", "group": None, "channel": fifo.handle,
                  "engine_type": engine_types[i], "status": "ok", "error": None, "ctx_size": int(p.totalBufferSize)})
      except RuntimeError as e:
        errors.append({"op": "GR_GET_CTX_BUFFER_SIZE", "engine_index": i, "engine_type": engine_types[i], "error": str(e)})
    if sizes and len(sizes) == len(fifos_by_channel) and len(set(sizes)) == 1:
      base, length = dev.iface._alloc_gpu_vaddr(sizes[0], force_low=True), sizes[0]
      for i, fifo in fifos_by_channel.items():
        try:
          dev.iface.setup_gpfifo_vm_at(fifo.handle, base, length)
          on_rm_op({"op": "UVM_REGISTER_CHANNEL_SHARED_CTX", "kind": "UVM", "group": None, "channel": fifo.handle,
                    "engine_type": engine_types[i], "status": "ok", "error": None, "ctx_base": base, "ctx_size": length})
        except RuntimeError as e:
          errors.append({"op": "UVM_REGISTER_CHANNEL_SHARED_CTX", "engine_index": i, "engine_type": engine_types[i], "error": str(e)})
  if deferred_schedule is not None and not errors:
    op, group = deferred_schedule
    try:
      dev.iface.rm_control(group, ops_nv.nv_gpu.NVA06C_CTRL_CMD_GPFIFO_SCHEDULE,
                           ops_nv.nv_gpu.NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS(bEnable=1))
      on_rm_op({"op": op["op"], "kind": op["kind"], "group": _j(group), "channel": None,
                "engine_type": None, "status": "ok", "error": None})
    except RuntimeError as e:
      errors.append({"op": op["op"], "engine_index": None, "engine_type": None, "error": str(e)})
  return [fifos_by_channel[i] for i in sorted(fifos_by_channel)], errors


def arm_payload_schema(mode, engines, n, matmul, grid_div, device="NV sm_120 RTX 5090"):
  """Fresh single-arm payload skeleton (pure; the driver merges these per arm)."""
  return {"schema": "tinygrad.nv_multi_queue_probe.v2", "mode": mode, "device": device,
          "n": n, "matmul": matmul, "engines": list(engines), "grid_div": grid_div,
          "gpfifo_engine_types": [0, *engines], "rm_ops": [], "construction_errors": [],
          "errors": [], "experiments": [], "arm": {"mode": mode, "exit_code": 0, "timed_out": False}}


def experiment_row(name, check, ts, arm):
  """Per-kernel HCQ timestamps, span, node-sum, overlap fraction, and arm state."""
  durs = durations(ts)
  node_sum = sum(durs)
  sp = span(ts)
  return {"name": name, "status": "pending", "check": check,
          "timestamps_us": [[st, en] for st, en in ts],
          "span_us": sp, "node_sum_us": node_sum,
          "overlap": (node_sum - sp) / node_sum if node_sum else 0.0,
          "arm": dict(arm)}


def run_r1(args, dev, qs, arm):
  """Cross-GPFIFO semaphore dependency: kernel on queue 1 waits on a signal released
  by queue 0. Exact = anchored sha256 hash match AND max-error contract; never np.allclose."""
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
  ts = run_jobs(dev, [
    Job(qs[0], mul_prg, [bufs[u] for u in mul_args], mul_ast, signals=((out_sig, 1),)),
    Job(qs[1], add_prg, [bufs[u] for u in add_args], add_ast, waits=((out_sig, 1),)),
  ])
  out_c = copyout(dev, bufs[out_uop(mul_ast, mul_args)], 1 << 20)
  out_e = copyout(dev, bufs[out_uop(add_ast, add_args)], 1 << 20)
  exp_c = np.arange(1 << 20, dtype=np.float32) * np.linspace(0.5, 2.0, 1 << 20, dtype=np.float32)
  exp_e = exp_c + 1.0
  h = {"out_c": f32_sha(out_c), "out_e": f32_sha(out_e), "ref_c": f32_sha(exp_c), "ref_e": f32_sha(exp_e)}
  me_c, me_e = float(np.abs(out_c - exp_c).max()), float(np.abs(out_e - exp_e).max())
  tol = 1e-3
  row = experiment_row("R1", "cross-gpfifo semaphore dep (hash + max-error contract)", ts, arm)
  row.update({"hashes": h, "max_err_c": me_c, "max_err_e": me_e, "error_bound": tol,
              "ref_abs_max_c": float(np.abs(exp_c).max()), "ref_abs_max_e": float(np.abs(exp_e).max())})
  row["status"] = "pass" if r1_contract_pass((h["out_c"], h["out_e"]), (h["ref_c"], h["ref_e"]),
                                             (me_c, me_e), (row["ref_abs_max_c"], row["ref_abs_max_e"]), tol) else "FAIL"
  print(f"R1 {'PASS' if row['status'] == 'pass' else 'FAIL'} cross-gpfifo dep "
        f"hashes={h} max_err=({me_c:.2e}, {me_e:.2e})", file=sys.stderr, flush=True)
  return row


def run_r2(args, dev, qs, arm):
  """Serial calibration on one fifo: span must equal node-sum inside a timestamp
  tolerance, never by float equality."""
  N = args.n
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
  ts = run_jobs(dev, [
    Job(qs[0], s_prg, [s_bufs[u] for u in s_args], s_ast),
    Job(qs[0], s2_prg, [s_bufs[u] for u in s2_args], s2_ast),
  ])
  row = experiment_row("R2", "serial calibration (span vs node-sum timestamp tolerance)", ts, arm)
  abs_delta = abs(row["span_us"] - row["node_sum_us"])
  pct_delta = (abs_delta / row["node_sum_us"] * 100.0) if row["node_sum_us"] else float("inf")
  row.update({"abs_delta_us": abs_delta, "pct_delta": pct_delta, "timestamp_tol_pct": 2.0, "timestamp_tol_abs_us": 10.0})
  row["status"] = "pass" if serial_contract_ok(row["span_us"], row["node_sum_us"]) else "FAIL"
  print(f"R2 serial span={row['span_us']:.1f}us sum={row['node_sum_us']:.1f}us "
        f"abs_delta={abs_delta:.3f}us pct_delta={pct_delta:.3f}%", file=sys.stderr, flush=True)
  return row


def run_elementwise_row(args, dev, qs, arm, name, n_queues):
  """n_queues independent elementwise kernels on n_queues fifos (R3: 2, R4: 3)."""
  # A reduced launch grid leaves most of the logical output undefined, which
  # made the old --grid-div=4 row fail its own full-array correctness contract.
  # Reduce the *fully computed* vector size instead: it still gives a
  # partial-SM workload while retaining an exact output oracle.
  N = args.n // args.grid_div
  assert N > 0
  jobs, asts, argss, bufs = [], [], [], []
  for replay in range(args.replays):
    for qi in range(n_queues):
      ta = Tensor.empty(N, device="NV"); tb = Tensor.empty(N, device="NV")
      tc = ta * tb
      p, ast, uops = lower(dev, tc)
      bb = alloc_buffers(dev, uops)
      copyin(dev, bb[ta.uop], np.full(N, float(qi + 1), dtype=np.float32))
      copyin(dev, bb[tb.uop], np.full(N, 2.0, dtype=np.float32))
      jobs.append(Job(qs[qi], p, [bb[u] for u in uops], ast))
      asts.append(ast); argss.append(uops); bufs.append(bb)
  ts = run_jobs(dev, jobs)
  refs = [np.full(N, float((i % n_queues) + 1) * 2.0, dtype=np.float32) for i in range(len(jobs))]
  outs = [copyout(dev, bb[out_uop(ast, uops)], N) for ast, uops, bb in zip(asts, argss, bufs)]
  row = experiment_row(name, f"{n_queues}-queue elementwise overlap", ts, arm)
  tol = 1e-3
  row.update({"grid_div": args.grid_div, "replays_per_queue": args.replays,
              "hashes": {"out": [f32_sha(o) for o in outs], "ref": [f32_sha(r) for r in refs]},
              "max_errs": [float(np.abs(o - r).max()) for o, r in zip(outs, refs)],
              "error_bound": tol, "ref_abs_maxes": [float(np.abs(r).max()) for r in refs],
              "hash_match": [f32_sha(o) == f32_sha(r) for o, r in zip(outs, refs)]})
  numeric_ok = all(me <= tol * max(1.0, ra) for me, ra in zip(row["max_errs"], row["ref_abs_maxes"]))
  row["numeric_ok"] = bool(numeric_ok)
  row["status"] = "pass" if (row["overlap"] >= 0.05 and numeric_ok) else "FAIL"
  print(f"{name} {n_queues}-queue span={row['span_us']:.1f}us sum={row['node_sum_us']:.1f}us "
        f"overlap={row['overlap'] * 100:.1f}% numeric_ok={numeric_ok}", file=sys.stderr, flush=True)
  return row


def run_r5(args, dev, qs, arm):
  """Compute-heavy (matmul 2048) flavor on two fifos, sqrt(k)*I input pin kept."""
  M = args.matmul
  jobs, asts, argss, bufs = [], [], [], []
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
    jobs.append(Job(qs[qi], m_prg, [bb[u] for u in m_args], m_ast))
    asts.append(m_ast); argss.append(m_args); bufs.append(bb)
  ts = run_jobs(dev, jobs)
  outs = [copyout(dev, bufs[qi][out_uop(asts[qi], argss[qi])], (M, M)).reshape(M, M) for qi in (0, 1)]
  refs = [np.eye(M, dtype=np.float32) * float(qi + 1) for qi in (0, 1)]
  row = experiment_row("R5", "2-queue matmul overlap + correctness", ts, arm)
  tol = 1e-2
  row.update({"M": M, "hashes": {"out": [f32_sha(o) for o in outs], "ref": [f32_sha(r) for r in refs]},
              "max_errs": [float(np.abs(o - r).max()) for o, r in zip(outs, refs)],
              "error_bound": tol, "ref_abs_maxes": [float(np.abs(r).max()) for r in refs],
              "hash_match": [f32_sha(o) == f32_sha(r) for o, r in zip(outs, refs)]})
  numeric_ok = all(me <= tol * max(1.0, ra) for me, ra in zip(row["max_errs"], row["ref_abs_maxes"]))
  row["numeric_ok"] = bool(numeric_ok)
  row["status"] = "pass" if (row["overlap"] >= 0.05 and numeric_ok) else "FAIL"
  print(f"R5 2-queue matmul span={row['span_us']:.1f}us sum={row['node_sum_us']:.1f}us "
        f"overlap={row['overlap'] * 100:.1f}% numeric_ok={numeric_ok}", file=sys.stderr, flush=True)
  return row


def run_experiments(args, dev, qs, payload, flush):
  """Run R1-R5, appending each row to payload['experiments'] and flushing after each."""
  arm = payload["arm"]
  results = payload["experiments"]

  def skipped(name, check, reason):
    results.append({"name": name, "status": "skipped", "check": check, "error": reason,
                    "timestamps_us": [], "span_us": None, "node_sum_us": None, "overlap": None, "arm": dict(arm)})
    payload["errors"].append(f"{name} skipped: {reason}")
    flush()

  def done(name, fn):
    try:
      results.append(fn())
    except (RuntimeError, MemoryError) as e:
      results.append({"name": name, "status": "FAIL", "check": "execution error", "error": str(e),
                      "timestamps_us": [], "span_us": None, "node_sum_us": None, "overlap": None, "arm": dict(arm)})
      payload["errors"].append(f"{name} execution error: {e}")
    flush()

  if len(qs) < 2:
    skipped("R1", "cross-gpfifo semaphore dep (hash + max-error contract)", f"need >= 2 compute gpfifos, got {len(qs)}")
  else:
    done("R1", lambda: run_r1(args, dev, qs, arm))
  if args.stop_after == 1: sys.exit(0)

  done("R2", lambda: run_r2(args, dev, qs, arm))
  if args.stop_after == 2: sys.exit(0)

  if len(qs) < 2:
    skipped("R3", "2-queue elementwise overlap", f"need >= 2 compute gpfifos, got {len(qs)}")
  else:
    done("R3", lambda: run_elementwise_row(args, dev, qs, arm, "R3", 2))
  if args.stop_after == 3: sys.exit(0)

  if len(qs) < 3:
    skipped("R4", "3-queue elementwise overlap", f"need >= 3 compute gpfifos, got {len(qs)}")
  else:
    done("R4", lambda: run_elementwise_row(args, dev, qs, arm, "R4", 3))
  if args.stop_after == 4: sys.exit(0)

  if len(qs) < 2:
    skipped("R5", "2-queue matmul overlap + correctness", f"need >= 2 compute gpfifos, got {len(qs)}")
  else:
    done("R5", lambda: run_r5(args, dev, qs, arm))


def run_arm(args) -> None:
  t0 = time.perf_counter()
  # The historical arms attached extra channels after tinygrad had already
  # scheduled its bootstrap group.  CUDA instead creates its stream channels
  # before that first group schedule.  This construction-only arm asks that
  # remaining question without changing the default one-channel runtime.
  if args.mode == "bootstrap_cuda":
    os.environ["HCQ_NUM_COMPUTE"] = "2"
    if args.bootstrap_dma: os.environ["NV_BOOT_COMPUTE_CHANNEL_DMA"] = "1"
  dev = Device["NV"]
  dev.synchronize()
  engines = [int(x) for x in args.engines.split(",") if x != ""]
  payload = arm_payload_schema(args.mode, engines, args.n, args.matmul, args.grid_div)

  def flush():
    if args.out:
      with open(args.out, "w", encoding="utf-8") as f: json.dump(payload, f, indent=2)

  def on_rm_op(rec):
    payload["rm_ops"].append(rec)
    flush()

  try:
    flags = None if args.channel_flags == "" else [int(x, 0) for x in args.channel_flags.split(",")]
    if args.mode == "bootstrap_cuda":
      extra, construction_errors = [], []
      on_rm_op({"op": "BOOTSTRAP_GROUP_BEFORE_SCHEDULE", "kind": "NVA06C", "group": int(dev.channel_group),
                "channel": None, "engine_type": 0, "status": "ok", "error": None,
                "compute_channels": len(dev.compute_gpfifos), "channel_dma": bool(args.bootstrap_dma)})
    else:
      extra, construction_errors = extra_gpfifos(dev, engines, mode=args.mode, on_rm_op=on_rm_op,
                                                 bind_policy=args.bind_policy, channel_flags=flags)
  except (RuntimeError, MemoryError) as e:
    construction_errors = [{"op": "CONSTRUCTION_ABORTED", "engine_index": None, "engine_type": None, "error": str(e)}]
    extra = []
  payload["construction_errors"] = construction_errors
  payload["errors"] += [f"construction {e['op']} engine {e['engine_index']}: {e['error']}" for e in construction_errors]
  gpfifos = list(dev.compute_gpfifos) if args.mode == "bootstrap_cuda" else [dev.compute_gpfifo, *extra]
  qs = make_queues(dev, gpfifos)
  print(f"device ready {time.perf_counter() - t0:.2f}s, compute gpfifos={len(gpfifos)} mode={args.mode} "
        f"engineTypes={[0, *engines]} errors={construction_errors}", file=sys.stderr, flush=True)
  flush()
  run_experiments(args, dev, qs, payload, flush)
  flush()
  print(json.dumps(payload, indent=2))


def g1_verdict(arms):
  """Gate G1 (belief-flip) classification over merged per-arm payloads.

  PASS: R1's hash/error contract passes in at least one successfully constructed
        arm AND at least one R3-R5 row in any arm shows overlap >= 5%.
  NO_OVERLAP: R1 passes somewhere, at least one corrected mode (ctxshare/group)
        successfully constructed (its R3-R5 rows executed rather than skipped),
        and every R3-R5 row in every arm is below 5%. The shared arm alone never
        earns NO_OVERLAP: it is the known-serialized control.
  CONSTRUCTION_BLOCKED: any other non-PASS shape - in particular, corrected modes
        rejected by an RM step or timing out before R1, or R1 failing its contract.
  Any non-PASS basis names the exact arm/operation boundary.
  """
  r1_pass_modes = [a["mode"] for a in arms if any(e["name"] == "R1" and e["status"] == "pass" for e in a["experiments"])]
  overlap_rows = [(a["mode"], e["name"], e["overlap"]) for a in arms for e in a["experiments"]
                  if e["name"] in ("R3", "R4", "R5") and isinstance(e.get("overlap"), (int, float)) and e["overlap"] >= 0.05]
  per_arm_max_overlap = {a["mode"]: max((e["overlap"] for e in a["experiments"]
                                         if e["name"] in ("R3", "R4", "R5") and isinstance(e.get("overlap"), (int, float))),
                                        default=0.0) for a in arms}
  corrected_executed = [a["mode"] for a in arms if a["mode"] in ("ctxshare", "group", "cuda_mirror", "bootstrap_cuda", "fresh_cuda_group", "fresh_cuda_group_ctxbuf", "fresh_cuda_group_cuda_params", "fresh_cuda_group_cuda_params_notifier4k")
                        and any(e["name"] in ("R3", "R4", "R5") and e.get("overlap") is not None for e in a["experiments"])]
  if r1_pass_modes and overlap_rows:
    return "PASS", (f"R1 hash/error contract passed in arm(s) {r1_pass_modes}; "
                    f"R3-R5 overlap >= 5% in {[(m, n, round(o, 4)) for m, n, o in overlap_rows]}")
  if r1_pass_modes and corrected_executed:
    return "NO_OVERLAP", (f"R1 passed in arm(s) {r1_pass_modes} but every R3-R5 row in every arm is below 5% "
                          f"(per-arm max overlap {per_arm_max_overlap}; corrected modes executed: {corrected_executed})")
  r1_fail_modes = [a["mode"] for a in arms if any(e["name"] == "R1" and e["status"] == "FAIL" for e in a["experiments"])]
  if r1_fail_modes:
    return "CONSTRUCTION_BLOCKED", (f"R1 executed but failed its hash/error contract in arm(s) {r1_fail_modes}; "
                                    f"no arm has a passing R1, so this is a blocked construction, not a no-overlap result")
  if r1_pass_modes and not corrected_executed:
    return "CONSTRUCTION_BLOCKED", (f"R1 passed in control arm(s) {r1_pass_modes} but no corrected mode (ctxshare/group) "
                                    f"successfully constructed and executed R3-R5 (per-arm max overlap {per_arm_max_overlap}); "
                                    f"the shared arm is the known-serialized control and cannot earn NO_OVERLAP alone")
  r1_states = {a["mode"]: [e.get("error", e.get("status")) for e in a["experiments"] if e["name"] == "R1"] for a in arms}
  return "CONSTRUCTION_BLOCKED", (f"R1 never ran in any arm: per-arm R1 states {r1_states}; "
                                  f"every arm failed construction or timed out before R1")


def run_all_driver(args) -> None:
  if not args.out: raise SystemExit("--run-all requires --out")
  script = os.path.abspath(__file__)
  engines = [int(x) for x in args.engines.split(",") if x != ""]
  arms = []
  for mode in ("shared", "ctxshare", "group", "cuda_mirror", "fresh_cuda_group", "fresh_cuda_group_ctxbuf", "fresh_cuda_group_cuda_params", "fresh_cuda_group_cuda_params_notifier4k"):
    arm_out = f"{args.out}.arm.{mode}.json"
    cmd = [sys.executable, script, "--mode", mode, "--out", arm_out, "--engines", args.engines,
           "--n", str(args.n), "--matmul", str(args.matmul), "--stop-after", str(args.stop_after),
           "--grid-div", str(args.grid_div), "--bind-policy", args.bind_policy]
    try:
      cp = subprocess.run(cmd, timeout=args.timeout, capture_output=True, text=True)
      exit_code, timed_out = cp.returncode, False
    except subprocess.TimeoutExpired:
      exit_code, timed_out = None, True
    arm = {"mode": mode, "exit_code": exit_code, "timed_out": timed_out, "rm_ops": [], "experiments": [], "errors": []}
    try:
      with open(arm_out, "r", encoding="utf-8") as f: sub_payload = json.load(f)
    except (OSError, json.JSONDecodeError):
      sub_payload = None
    if sub_payload is None:
      arm["errors"].append("arm produced no JSON payload (failed before the first flush)")
    else:
      arm["rm_ops"] = sub_payload.get("rm_ops", [])
      arm["experiments"] = sub_payload.get("experiments", [])
      arm["errors"] = list(sub_payload.get("errors", []))
    if timed_out: arm["errors"].append(f"timed out after {args.timeout}s")
    elif exit_code != 0: arm["errors"].append(f"arm exited with code {exit_code}")
    for row in arm["experiments"]:
      row["arm"] = {"mode": mode, "exit_code": exit_code, "timed_out": timed_out}
    arms.append(arm)
    print(f"arm {mode}: exit_code={exit_code} timed_out={timed_out} experiments={len(arm['experiments'])} "
          f"rm_ops={len(arm['rm_ops'])} errors={arm['errors']}", file=sys.stderr, flush=True)
  verdict, basis = g1_verdict(arms)
  payload = {"schema": "tinygrad.nv_multi_queue_probe.driver.v1", "device": "NV sm_120 RTX 5090",
             "n": args.n, "matmul": args.matmul, "engines": engines, "grid_div": args.grid_div,
             "per_arm_timeout_s": args.timeout, "arms": arms, "verdict": verdict, "verdict_basis": basis}
  with open(args.out, "w", encoding="utf-8") as f: json.dump(payload, f, indent=2)
  print(json.dumps(payload, indent=2))


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--mode", type=str, choices=["shared", "ctxshare", "group", "cuda_mirror", "bootstrap_cuda", "fresh_cuda_group", "fresh_cuda_group_ctxbuf", "fresh_cuda_group_cuda_params", "fresh_cuda_group_cuda_params_notifier4k"], default="shared",
                  help="construction mode for the extra GPFifos (default shared = E1-E5 control arm)")
  ap.add_argument("--run-all", action="store_true",
                  help="driver: run all three modes in fresh subprocesses with a hard timeout and merge the arms")
  ap.add_argument("--timeout", type=int, default=600, help="per-arm hard timeout in seconds (--run-all)")
  ap.add_argument("--out", type=str, default=None, help="incremental JSON output path")
  ap.add_argument("--engines", type=str, default="0,0", help="comma list of engineType per extra GPFifo")
  ap.add_argument("--n", type=int, default=1 << 25, help="elementwise vector length")
  ap.add_argument("--matmul", type=int, default=2048, help="matmul NxNxN for the compute-heavy flavor")
  ap.add_argument("--stop-after", type=int, default=5, help="stop after experiment N (1-5)")
  ap.add_argument("--grid-div", type=int, default=1,
                  help="divide the R3/R4 fully-computed vector size by this factor; this preserves the "
                       "exact output oracle while making a partial-SM workload possible")
  ap.add_argument("--replays", type=int, default=1,
                  help="independent R3/R4 kernels per queue, interleaved by queue (default 1)")
  ap.add_argument("--bind-policy", type=str, choices=["required", "skip"], default="required",
                  help="issue per-channel NVA06F_CTRL_CMD_BIND (required) or omit it (skip); "
                       "driver 595.84 rejects BIND for group-allocated compute channels")
  ap.add_argument("--channel-flags", type=str, default="",
                  help="comma-separated raw GPFIFO allocation flags (probe-only); empty uses the mode default")
  ap.add_argument("--bootstrap-dma", action="store_true",
                  help="bootstrap_cuda only: attach CUDA-like DMA objects to the two compute channels before scheduling")
  args = ap.parse_args()
  if args.run_all: run_all_driver(args)
  else: run_arm(args)


if __name__ == "__main__":
  main()
