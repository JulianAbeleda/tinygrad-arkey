#!/usr/bin/env python3
"""Marker-controlled native-NV d512 decode device-window diagnostic.

This is deliberately outside production. It inserts native queue timestamp
markers immediately before the first and after the expected final HCQGraph submission,
then compares alternating marker-off/marker-on whole-token wall measurements.
The markers do not change graph programs, buffers, or dependencies.
"""
from __future__ import annotations

import argparse, hashlib, json, pathlib, statistics, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def med(values): return statistics.median(values) if values else None


def summarize(controls:list[dict], measured:list[dict]) -> dict:
  return {
    "marker_off_wall_us": med([x["wall_us"] for x in controls]),
    "marker_on_wall_us": med([x["wall_us"] for x in measured]),
    "device_window_us": med([x["device_window_us"] for x in measured]),
    "outside_window_us": med([x["outside_window_us"] for x in measured]),
    "marker_overhead_us": med([x["wall_us"] for x in measured])-med([x["wall_us"] for x in controls]),
  }


def host_partition(row:dict) -> dict:
  """Partition host wall around the graph window without double-counting overlap.

  Graph-call CPU and inter-call CPU intervals occur while previously submitted GPU
  work can still run, so they are reported but are *not* added to the outside-window
  equation.  Only the pre-first-call and post-device-drain boundaries are disjoint
  from the native queue-timestamp window.
  """
  calls, t0, t1 = row["host_calls_ns"], row["host_t0_ns"], row["host_t1_ns"]
  item = row["item_calls_ns"][-1]
  return {
    "pre_first_graph_us": (calls[0]["enter_ns"]-t0)/1e3,
    "graph_call_cpu_sum_us": sum(x["exit_ns"]-x["enter_ns"] for x in calls)/1e3,
    "inter_graph_host_sum_us": sum(calls[i+1]["enter_ns"]-calls[i]["exit_ns"] for i in range(len(calls)-1))/1e3,
    "last_graph_return_to_item_us": (item["enter_ns"]-calls[-1]["exit_ns"])/1e3,
    "item_total_us": (item["exit_ns"]-item["enter_ns"])/1e3,
    "item_pre_drain_us": item.get("drain_ns", 0)/1e3,
    "item_after_drain_us": (item["exit_ns"]-item.get("after_drain_ns", item["enter_ns"]))/1e3,
    "python_yield_tail_us": (row["next_return_ns"]-item["exit_ns"])/1e3,
    "post_next_synchronize_us": (t1-row["next_return_ns"])/1e3,
    "host_wall_us": (t1-t0)/1e3,
  }


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  ap.add_argument("--depth", type=int, default=512)
  ap.add_argument("--reps", type=int, default=5)
  ap.add_argument("--expected-groups", type=int, default=5)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()

  from tinygrad.device import Device
  from tinygrad.helpers import Context, unwrap
  from tinygrad.runtime.graph.hcq import HCQGraph
  from tinygrad.llm.model import Transformer
  import tinygrad.llm.model as tgm
  from tinygrad.tensor import Tensor

  # Existing decode-measurement seam: the promoted packed prefill attention
  # verifier is independently broken on this branch and is outside this test.
  tgm._CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS = frozenset()

  state = {"active": False, "markers": False, "calls": [], "start": None, "end": None, "mode": "normal", "items": []}
  original = HCQGraph.__call__
  original_item = Tensor.item

  def timestamp_marker(dev, sig) -> None:
    # Keep the marker on the device's ordinary compute submission path and
    # chain it through the same global timeline used by graph submissions.
    unwrap(dev.hw_compute_queue_t)().wait(dev.timeline_signal, dev.timeline_value-1) \
      .timestamp(sig).signal(dev.timeline_signal, dev.next_timeline()).submit(dev)

  def names_of(graph) -> list[str]:
    return [c[1].arg.name if hasattr(c[1].arg, "name") else c[1].op.name for c in graph.calls]

  def wrapped(self, input_uops, var_vals, wait=False):
    if not state["active"]: return original(self, input_uops, var_vals, wait=wait)
    if len(self.devices) != 1: raise RuntimeError(f"expected one native device, saw {len(self.devices)}")
    dev = self.devices[0]
    idx = len(state["calls"])
    if state["markers"] and idx == 0:
      state["start"] = dev.new_signal(value=0)
      timestamp_marker(dev, state["start"])
    enter_ns = time.perf_counter_ns()
    ret = original(self, input_uops, var_vals, wait=wait)
    exit_ns = time.perf_counter_ns()
    state["calls"].append({"programs": len(self.calls), "enter_ns": enter_ns, "exit_ns": exit_ns,
                           "name_sha256": hashlib.sha256("\n".join(names_of(self)).encode()).hexdigest()})
    if state["markers"] and idx == args.expected_groups-1:
      state["end"] = dev.new_signal(value=0)
      timestamp_marker(dev, state["end"])
    return ret

  def wrapped_item(self):
    if not state["active"]: return original_item(self)
    rec = {"enter_ns": time.perf_counter_ns()}
    if state["mode"] == "pre_item_drain":
      ds = time.perf_counter_ns()
      Device["NV"].synchronize()
      rec["drain_ns"] = time.perf_counter_ns()-ds
      rec["after_drain_ns"] = time.perf_counter_ns()
    ret = original_item(self)
    rec["exit_ns"] = time.perf_counter_ns()
    state["items"].append(rec)
    return ret

  HCQGraph.__call__ = wrapped
  Tensor.item = wrapped_item
  model, _ = Transformer.from_gguf(args.model, 4608)
  prompt = [1] * args.depth

  def run(markers:bool, mode:str="normal") -> dict:
    model.reset_generation_state()
    gen = model.generate(prompt.copy(), chunk_size=32, temperature=0.0)
    # Prefill, capture, and one settled replay; measure the following token.
    with Context(DEBUG=0): next(gen); next(gen); next(gen)
    Device["NV"].synchronize()
    state.update({"active": True, "markers": markers, "calls": [], "start": None, "end": None, "mode": mode, "items": []})
    t0_ns = time.perf_counter_ns()
    with Context(DEBUG=0): token = int(next(gen))
    next_return_ns = time.perf_counter_ns()
    Device["NV"].synchronize()
    t1_ns = time.perf_counter_ns()
    wall_us = (t1_ns-t0_ns)/1e3
    state["active"] = False
    if not markers:
      row = {"token": token, "wall_us": wall_us, "host_t0_ns": t0_ns, "next_return_ns": next_return_ns,
             "host_t1_ns": t1_ns, "host_calls_ns": state["calls"], "item_calls_ns": state["items"], "mode": mode}
      if len(state["calls"]) != args.expected_groups or len(state["items"]) != 1:
        raise RuntimeError(f"expected {args.expected_groups} graph groups and one item, saw {len(state['calls'])}/{len(state['items'])}")
      row["host_partition"] = host_partition(row)
      gen.close()
      return row
    if len(state["calls"]) != args.expected_groups or state["start"] is None or state["end"] is None:
      raise RuntimeError(f"expected {args.expected_groups} native graph groups and two markers, saw {len(state['calls'])}")
    span_us = float(state["end"].timestamp - state["start"].timestamp)
    row = {"token": token, "wall_us": wall_us, "device_window_us": span_us,
           "outside_window_us": wall_us-span_us, "groups": state["calls"],
           "host_t0_ns": t0_ns, "next_return_ns": next_return_ns, "host_t1_ns": t1_ns,
           "host_calls_ns": state["calls"], "item_calls_ns": state["items"], "mode": mode}
    if len(state["items"]) != 1: raise RuntimeError(f"expected one sampled item call, saw {len(state['items'])}")
    row["host_partition"] = host_partition(row)
    gen.close()
    return row

  controls, control_drained, measured, drained = [], [], [], []
  for _ in range(args.reps):
    controls.append(run(False))
    control_drained.append(run(False, "pre_item_drain"))
    measured.append(run(True, "normal"))
    drained.append(run(True, "pre_item_drain"))
  if len({x["token"] for x in controls+control_drained+measured+drained}) != 1: raise RuntimeError("instrumentation arms changed generated token")
  out = {
    "schema": "tinygrad.nv_decode_group_window_ledger.v1",
    "evidence": "OBSERVATIONAL_NATIVE_QUEUE_TIMESTAMPS",
    "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "route": "DEV=NV",
    "depth": args.depth,
    "reps": args.reps,
    "marker_off": controls, "marker_off_pre_item_drain": control_drained,
    "marker_on": measured, "pre_item_drain": drained,
    "median": summarize(controls, measured),
    "host_partition_median": {k: med([x["host_partition"][k] for x in measured]) for k in measured[0]["host_partition"]},
    "marker_off_host_median": {k: med([x["host_partition"][k] for x in controls]) for k in controls[0]["host_partition"]},
    "marker_off_pre_item_drain_median": {k: med([x["host_partition"][k] for x in control_drained]) for k in control_drained[0]["host_partition"]},
    "pre_item_drain_median": {k: med([x["host_partition"][k] for x in drained]) for k in drained[0]["host_partition"]},
  }
  pathlib.Path(args.out).write_text(json.dumps(out, indent=2)+"\n")
  print(json.dumps({"boundary": out["median"], "marker_off_host": out["marker_off_host_median"],
                    "marker_off_pre_item_drain": out["marker_off_pre_item_drain_median"], "host": out["host_partition_median"],
                    "pre_item_drain": out["pre_item_drain_median"]}, indent=2))


if __name__ == "__main__": main()
