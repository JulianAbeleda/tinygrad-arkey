#!/usr/bin/env python3
"""Opt-in native-NV decode pre-first-HCQGraph host breakdown.

All instrumentation is installed at runtime and restored before exit.  The
uninstrumented arm uses the same outer clocks but no internal perf-counter
calls, providing an explicit observer-tax control.
"""
from __future__ import annotations

import argparse, json, pathlib, statistics, subprocess, sys, time
from typing import cast

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def med(xs): return statistics.median(xs) if xs else None


def summarize(rows:list[dict]) -> dict:
  keys = sorted(set.intersection(*(set(r) for r in rows)) - {"token", "graph_calls"})
  return {k: med([r[k] for r in rows]) for k in keys if isinstance(rows[0][k], (int, float))}


def reconcile(row:dict) -> dict:
  parts = ["before_tinyjit_us", "prepare_inputs_us", "reuse_checks_us", "captured_before_concrete_us",
           "concrete_copy_rebind_us", "captured_to_run_linear_us", "run_linear_to_graph_lookup_us",
           "graph_runtime_lookup_us", "graph_lookup_to_hcq_entry_us"]
  explained = sum(row.get(x, 0.0) for x in parts)
  return {"explained_pre_first_us": explained, "closure_us": row["pre_first_graph_us"]-explained,
          "closure_pct": 100.0*(row["pre_first_graph_us"]-explained)/row["pre_first_graph_us"], "parts": parts}


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  ap.add_argument("--depth", type=int, default=512)
  ap.add_argument("--reps", type=int, default=9)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()

  import tinygrad.engine.jit as tj
  import tinygrad.engine.realize as tr
  import tinygrad.llm.model as tgm
  from tinygrad.device import Device, MultiBuffer
  from tinygrad.helpers import Context, DEBUG
  from tinygrad.runtime.graph.hcq import HCQGraph
  from tinygrad.runtime.support.hcq import HCQAllocator
  from tinygrad.tensor import Tensor
  from tinygrad.uop.ops import Ops, UOp, buffers, graph_rewrite
  from tinygrad.llm.model import Transformer

  tgm._CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS = frozenset()
  state = {"active": False, "row": None}
  original_prepare, original_tinyjit = tj._prepare_jit_inputs, tj.TinyJit.__call__
  original_captured, original_run_linear, original_copy_input = tj.CapturedJit.__call__, tj.run_linear, tj._copy_input
  original_get_graph, original_hcq = tr.get_graph_runtime, HCQGraph.__call__

  def stamp(name, st):
    if state["active"]: state["row"][name] = state["row"].get(name, 0) + time.perf_counter_ns()-st

  # Faithful copy of _prepare_jit_inputs, with timers only at existing semantic boundaries.
  def prepared(args_, kwargs_):
    if not state["active"]: return original_prepare(args_, kwargs_)
    st = time.perf_counter_ns()
    input_tensors = [(name,t) for name,t in list(enumerate(args_))+sorted(kwargs_.items()) if t.__class__ is Tensor]
    names, tensors = [name for name,_ in input_tensors], [t for _,t in input_tensors]
    for x in args_ + tuple(kwargs_.values()):
      it = x if isinstance(x, (tuple,list)) else x.values() if isinstance(x, dict) else []
      tensors += [t for t in it if t.__class__ is Tensor and not any(t is y for y in tensors)]
    def input_uops(): return tj.flatten([t.uop.src if t.uop.op is Ops.MULTI else [t.uop] for t in tensors])
    if any(u.device is None or u.base.op is Ops.CONST for u in input_uops()): raise tj.JitError("JIT inputs must be real buffers; use .clone()")
    if len(unrealized := [x for x in tensors if not x.uop.is_realized]): Tensor.realize(*unrealized)
    discovery_end = time.perf_counter_ns()
    ius = input_uops()
    ibus = [u.base for u in ius if u.base.realized is not None]
    if len(set(ibus)) != len(ibus): raise tj.JitError("duplicate inputs to JIT")
    structural_start = time.perf_counter_ns()
    inputs = [(*(graph_rewrite(u.substitute({u.base:UOp(Ops.NOOP)}, extra_pm=tj.mop_cleanup), tj.pm_jit_input_metadata).unbind_all()), u.dtype, u.device) for u in ius]
    vars_start = time.perf_counter_ns()
    vv = tj.merge_dicts([x[1] for x in inputs] + [dict(v.unbind() for v in (args_ + tuple(kwargs_.values())) if isinstance(v, UOp))])
    var_vals = {k.expr:v for k,v in vv.items()}
    info = [(x[0], tuple(sorted(x[1].keys(), key=lambda v: v.expr)), x[2], x[3]) for x in inputs]
    en = time.perf_counter_ns()
    row = state["row"]
    row["prepare_discovery_realize_ns"] = discovery_end-st
    row["prepare_structural_signature_ns"] = vars_start-structural_start
    row["prepare_var_expected_info_ns"] = en-vars_start
    row["prepare_inputs_ns"] = en-st
    row["prepare_end_ns"] = en
    return ibus, var_vals, names, info

  def tinyjit_call(self, *a, **kw):
    if not state["active"]: return original_tinyjit(self, *a, **kw)
    state["row"].setdefault("tinyjit_entry_ns", time.perf_counter_ns())
    return original_tinyjit(self, *a, **kw)

  def captured_call(self, input_uops, var_vals):
    if not state["active"]: return original_captured(self, input_uops, var_vals)
    entry = time.perf_counter_ns(); state["row"]["captured_entry_ns"] = entry
    state["row"]["captured_input_count"] = len(input_uops)
    state["row"]["captured_written_input_count"] = sum(u in self._written_uops for u in input_uops)
    concrete = tuple(tj._copy_input(u) if u in self._written_uops else u for u in input_uops)
    concrete_end = time.perf_counter_ns()
    if DEBUG >= 1 and len(self.linear.src) >= 10: print(f"jit execs {len(self.linear.src)} calls")
    observer_end = concrete_end
    if (observer:=tj._GRAPH_ADMISSION_OBSERVER.get()) is not None and hasattr(observer, "bind_execution"):
      observer.bind_execution(self.linear, concrete, var_vals); observer_end = time.perf_counter_ns()
    state["row"]["concrete_copy_rebind_ns"] = concrete_end-entry
    state["row"]["captured_observer_bind_ns"] = observer_end-concrete_end
    state["row"]["run_linear_call_ns"] = time.perf_counter_ns()
    try: tj.run_linear(self.linear, var_vals, input_uops=concrete, jit=True)
    except tj.GraphException as exc:
      if (observer:=tj._GRAPH_ADMISSION_OBSERVER.get()) is not None:
        try: observer(tj.GraphConstructorFailureObservation(type(exc).__name__, str(exc)))
        finally: raise
      raise
    return self.ret

  def copy_input(u):
    if not state["active"]: return original_copy_input(u)
    st = time.perf_counter_ns()
    call = u.copy_to_device(u.device).call(new:=UOp.new_buffer(u.device, u.arg, u.dtype), u, metadata=())
    built = time.perf_counter_ns()
    tj.run_linear(UOp(Ops.LINEAR, src=(call,)))
    en = time.perf_counter_ns()
    state["row"]["copy_input_build_ns"] = state["row"].get("copy_input_build_ns", 0) + built-st
    state["row"]["copy_input_run_ns"] = state["row"].get("copy_input_run_ns", 0) + en-built
    return new

  def run_linear(linear, var_vals=None, input_uops=(), update_stats=True, jit=False, wait=False):
    if not state["active"]: return original_run_linear(linear, var_vals, input_uops, update_stats, jit, wait)
    entry = time.perf_counter_ns(); state["row"]["run_linear_entry_ns"] = entry
    if not jit: linear = tr.compile_linear(linear, validate=tr.VALIDATE_WITH_CPU)
    ctx = tr.ExecContext(var_vals or {}, input_uops, update_stats, jit, wait or tr.DEBUG>=2)
    state["row"]["run_linear_context_ns"] = time.perf_counter_ns()-entry
    for call in linear.src: tr.pm_exec.rewrite(call, ctx)

  def get_graph(ast, input_uops=None):
    if not state["active"]: return original_get_graph(ast, input_uops)
    st = time.perf_counter_ns()
    if "first_graph_lookup_entry_ns" not in state["row"]: state["row"]["first_graph_lookup_entry_ns"] = st
    out = original_get_graph(ast, input_uops)
    en = time.perf_counter_ns()
    if "first_graph_lookup_ns" not in state["row"]: state["row"]["first_graph_lookup_ns"] = en-st
    return out

  def hcq_call(self, input_uops, var_vals, wait=False):
    if not state["active"]: return original_hcq(self, input_uops, var_vals, wait=wait)
    now = time.perf_counter_ns()
    state["row"].setdefault("first_hcq_entry_ns", now)
    state["row"]["graph_calls"] += 1
    st = now
    for dev in self.devices:
      for iidx, dev_idx in self.input_replace_map[dev]:
        buf = b.bufs[dev_idx] if isinstance(b:=input_uops[iidx].buffer, MultiBuffer) else b
        cast(HCQAllocator, dev.allocator).map(buf._buf)
    mapped = time.perf_counter_ns()
    self.kickoff_value += 1
    for dev in self.devices: self.last_timeline[dev][0].wait(self.last_timeline[dev][1])
    if tr.PROFILE and self.kickoff_value > 1:
      self.collect_timestamps(); self.collect_pmc()
    waited = time.perf_counter_ns()
    hcq_var_vals = {self.kickoff_var.expr: self.kickoff_value, **var_vals,
                    **{var.expr: dev.timeline_value - 1 for dev, var in self.virt_timeline_vals.items()},
                    **{sig.base_buf.va_addr.expr: dev.timeline_signal.base_buf.va_addr for dev, sig in self.virt_timeline_signals.items()}}
    vars_built = time.perf_counter_ns()
    for j, replace in enumerate(self.uop_replace):
      dev_idx = self.calls[j][0]
      for pos, iidx in replace:
        buf = b.bufs[dev_idx] if isinstance(b:=input_uops[iidx].buffer, MultiBuffer) else b
        hcq_var_vals[self.input_replace_to_var[(j,pos)].expr] = buf._buf.va_addr
    for (var, qp) in self.rdma_vars.values(): hcq_var_vals[var.expr] = qp.head
    nodes_updated = time.perf_counter_ns()
    for q in self.rdma_queues.values(): q.submit(q.dev, hcq_var_vals)
    for dev in self.devices:
      self.comp_queues[dev].submit(dev, local:=hcq_var_vals|self.device_vars.get(dev, {}))
      for copy_queue in self._dev_copy_queues(dev): copy_queue.submit(dev, local)
      self.last_timeline[dev] = (dev.timeline_signal, dev.next_timeline())
    submitted = time.perf_counter_ns()
    for sig in self.queue_signals_to_reset: sig.value = 0
    for sig in self.kick_signals.values(): sig.value = self.kickoff_value
    ended = time.perf_counter_ns()
    for name, value in {"hcq_map":mapped-st, "hcq_wait_restore":waited-mapped, "hcq_var_dict":vars_built-waited,
                        "hcq_node_var_updates":nodes_updated-vars_built, "hcq_queue_submit":submitted-nodes_updated,
                        "hcq_signal_finalize":ended-submitted, "hcq_cpu":ended-st}.items():
      state["row"][name+"_ns"] = state["row"].get(name+"_ns", 0) + value
    if wait:
      ws = time.perf_counter()
      for dev in self.devices: self.last_timeline[dev][0].wait(self.last_timeline[dev][1])
      return time.perf_counter()-ws
    return None

  tj._prepare_jit_inputs, tj.TinyJit.__call__, tj.CapturedJit.__call__, tj.run_linear, tj._copy_input = \
    prepared, tinyjit_call, captured_call, run_linear, copy_input
  tr.get_graph_runtime, HCQGraph.__call__ = get_graph, hcq_call

  model, _ = Transformer.from_gguf(args.model, 4608)
  prompt = [1] * args.depth

  def run(instrument:bool) -> dict:
    model.reset_generation_state()
    gen = model.generate(prompt.copy(), chunk_size=32, temperature=0.0)
    with Context(DEBUG=0): next(gen); next(gen); next(gen)
    Device["NV"].synchronize()
    row = {"graph_calls": 0}
    state.update(active=instrument, row=row)
    t0 = time.perf_counter_ns(); row["token_t0_ns"] = t0
    with Context(DEBUG=0): token = int(next(gen))
    Device["NV"].synchronize(); t1 = time.perf_counter_ns()
    state["active"] = False; gen.close()
    row.update(token=token, wall_us=(t1-t0)/1e3)
    if instrument:
      for k,v in list(row.items()):
        if k.endswith("_ns") and not k.endswith(("entry_ns", "end_ns", "t0_ns", "call_ns")): row[k[:-3]+"_us"] = v/1e3
      row["pre_first_graph_us"] = (row["first_hcq_entry_ns"]-t0)/1e3
      row["before_tinyjit_us"] = (row["tinyjit_entry_ns"]-t0)/1e3
      row["reuse_checks_us"] = (row["captured_entry_ns"]-row["prepare_end_ns"])/1e3
      row["captured_before_concrete_us"] = 0.0
      row["captured_to_run_linear_us"] = (row["run_linear_call_ns"]-(row["captured_entry_ns"]+row["concrete_copy_rebind_ns"]))/1e3
      row["run_linear_to_graph_lookup_us"] = (row["first_graph_lookup_entry_ns"]-row["run_linear_entry_ns"])/1e3
      row["graph_runtime_lookup_us"] = row["first_graph_lookup_ns"]/1e3
      row["graph_lookup_to_hcq_entry_us"] = (row["first_hcq_entry_ns"]-row["first_graph_lookup_entry_ns"]-row["first_graph_lookup_ns"])/1e3
      row["reconciliation"] = reconcile(row)
    return row

  controls, measured = [], []
  for _ in range(args.reps):
    controls.append(run(False)); measured.append(run(True))
  if len({x["token"] for x in controls+measured}) != 1: raise RuntimeError("instrumentation changed generated token")
  payload = {"schema":"tinygrad.nv_decode_predispatch_breakdown.v1", "evidence":"DIAGNOSTIC_HOST_PERF_COUNTERS",
             "commit":subprocess.check_output(["git","rev-parse","HEAD"], cwd=ROOT, text=True).strip(),
             "route":"DEV=NV", "depth":args.depth, "reps":args.reps, "control":controls, "instrumented":measured,
             "control_median":summarize(controls), "instrumented_median":summarize(measured)}
  payload["observer_tax_wall_us"] = payload["instrumented_median"]["wall_us"]-payload["control_median"]["wall_us"]
  # Reconcile each observation before taking a median.  Summing independent
  # component medians can combine different rows and manufacture closure error.
  payload["median_reconciliation"] = {
    "explained_pre_first_us": med([r["reconciliation"]["explained_pre_first_us"] for r in measured]),
    "closure_us": med([r["reconciliation"]["closure_us"] for r in measured]),
    "closure_pct": med([r["reconciliation"]["closure_pct"] for r in measured]),
    "parts": measured[0]["reconciliation"]["parts"],
  }
  pathlib.Path(args.out).write_text(json.dumps(payload, indent=2)+"\n")
  print(json.dumps({"control":payload["control_median"], "instrumented":payload["instrumented_median"],
                    "observer_tax_wall_us":payload["observer_tax_wall_us"], "reconciliation":payload["median_reconciliation"]}, indent=2))


if __name__ == "__main__": main()
