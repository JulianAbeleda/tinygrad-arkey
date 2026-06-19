#!/usr/bin/env python3
"""Probe-local HCQ attribution for primitive analysis.

PMU-4 from docs/primitive-hcq-attribution-scope-20260619.md. This does not
collect PMU counters and does not change runtime defaults. It monkeypatches HCQ
launch/graph methods inside this process, runs small deterministic workloads,
and emits Level-3 runtime/graph attribution.
"""
from __future__ import annotations

import argparse, contextlib, functools, hashlib, json, os, pathlib, subprocess, time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-hcq-attribution"

def _now() -> int: return int(time.time())

def _git_commit() -> str:
  try:
    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=10).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10).returncode != 0
    return sha + ("-dirty" if dirty else "")
  except Exception:
    return "unknown"

def _read_json(path:pathlib.Path) -> dict[str, Any] | None:
  try:
    data = json.loads(path.read_text())
    return data if isinstance(data, dict) else None
  except Exception:
    return None

def _hash_bytes(data:bytes|None) -> str|None:
  return hashlib.sha256(data).hexdigest()[:16] if data else None

class Collector:
  def __init__(self):
    self.programs: list[dict[str, Any]] = []
    self.graphs: list[dict[str, Any]] = []
    self.graph_calls: list[dict[str, Any]] = []
    self._seq = 0
    self._graph_replays: dict[int, int] = {}

  def _next(self) -> int:
    self._seq += 1
    return self._seq

  def record_program(self, prg:Any, bufs:tuple[Any, ...], kwargs:dict[str, Any], host_ms:float, ret:Any):
    global_size = tuple(kwargs.get("global_size", (1, 1, 1)))
    local_size = tuple(kwargs.get("local_size", (1, 1, 1)))
    vals = tuple(kwargs.get("vals", ()))
    self.programs.append({
      "seq": self._next(),
      "kind": "eager_program",
      "program_name": getattr(prg, "name", type(prg).__name__),
      "runtime_class": type(prg).__name__,
      "device": getattr(getattr(prg, "dev", None), "device", None),
      "launch": {"global_size": global_size, "local_size": local_size},
      "metadata": {
        "kernargs_alloc_size": getattr(prg, "kernargs_alloc_size", None),
        "buffer_count": len(bufs),
        "vals_count": len(vals),
        "code_hash": _hash_bytes(getattr(prg, "lib", None)),
        "prof_prg_counter": getattr(prg, "prof_prg_counter", None),
      },
      "queue": {"type": "compute", "queue_idx": None},
      "sync": {"wait": bool(kwargs.get("wait", False)), "timeout": kwargs.get("timeout", None)},
      "timing": {"host_ms": round(host_ms, 6), "wait_return_raw": ret if isinstance(ret, (int, float)) else None,
                 "wait_return_units": "HCQProgram.__call__ return"},
      "graph": {"graph_id": None, "node_index": None},
    })

  def record_graph_init(self, graph:Any):
    runtimes = list(getattr(graph, "runtimes", []) or [])
    calls = list(getattr(graph, "calls", []) or [])
    jid = id(graph)
    runtime_names = [getattr(rt, "name", None) for rt in runtimes if rt is not None]
    devices = [getattr(d, "device", str(d)) for d in getattr(graph, "devices", [])]
    copy_count = sum(1 for (_, ast, _, _), rt in zip(calls, runtimes) if rt is None and getattr(getattr(ast, "op", None), "name", None) == "COPY")
    self.graphs.append({
      "graph_id": jid,
      "kind": "graph_construct",
      "call_count": len(calls),
      "runtime_count": sum(1 for rt in runtimes if rt is not None),
      "copy_count": copy_count,
      "devices": devices,
      "queue_count": {
        "compute": len(getattr(graph, "comp_queues", {}) or {}),
        "copy": len(getattr(graph, "copy_queues", {}) or {}),
        "rdma": len(getattr(graph, "rdma_queues", {}) or {}),
      },
      "rebind_count": sum(len(x) for x in getattr(graph, "uop_replace", []) or []),
      "program_names": runtime_names,
      "prof_signal_count": len(getattr(graph, "prof_signals", []) or []),
      "replay_count": 0,
      "wall_ms": None,
      "device_ms_sum": None,
    })
    for j, ((_, ast, _, _), rt) in enumerate(zip(calls, runtimes)):
      if rt is None: continue
      try:
        global_size, local_size = ast.arg.global_size, ast.arg.local_size
      except Exception:
        global_size, local_size = None, None
      self.programs.append({
        "seq": self._next(),
        "kind": "graph_program",
        "program_name": getattr(rt, "name", type(rt).__name__),
        "runtime_class": type(rt).__name__,
        "device": getattr(getattr(rt, "dev", None), "device", None),
        "launch": {"global_size": global_size, "local_size": local_size},
        "metadata": {
          "kernargs_alloc_size": getattr(rt, "kernargs_alloc_size", None),
          "buffer_count": len(calls[j][2]) if len(calls[j]) > 2 else None,
          "vals_count": None,
          "code_hash": _hash_bytes(getattr(rt, "lib", None)),
          "prof_prg_counter": getattr(rt, "prof_prg_counter", None),
        },
        "queue": {"type": "compute", "queue_idx": None},
        "sync": {"wait": None, "timeout": None},
        "timing": {"host_ms": None, "device_ms": None},
        "graph": {"graph_id": jid, "node_index": j},
      })

  def record_graph_call(self, graph:Any, wait:bool, host_ms:float, ret:Any):
    gid = id(graph)
    self._graph_replays[gid] = self._graph_replays.get(gid, 0) + 1
    replay = self._graph_replays[gid]
    for row in self.graphs:
      if row["graph_id"] == gid:
        row["replay_count"] = replay
        row["wall_ms"] = round(host_ms, 6)
        break
    self.graph_calls.append({
      "graph_id": gid,
      "replay_index": replay,
      "wait": wait,
      "wall_ms": round(host_ms, 6),
      "wait_return_s": ret if isinstance(ret, (int, float)) else None,
      "kickoff_value": getattr(graph, "kickoff_value", None),
    })

@contextlib.contextmanager
def attribution_context(collector:Collector):
  import tinygrad.runtime.support.hcq as hcq
  import tinygrad.runtime.graph.hcq as graph_hcq

  orig_call = hcq.HCQProgram.__call__
  orig_graph_init = graph_hcq.HCQGraph.__init__
  orig_graph_call = graph_hcq.HCQGraph.__call__

  @functools.wraps(orig_call)
  def wrapped_call(self, *bufs, **kwargs):
    st = time.perf_counter()
    ret = orig_call(self, *bufs, **kwargs)
    collector.record_program(self, bufs, kwargs, (time.perf_counter()-st)*1000.0, ret)
    return ret

  @functools.wraps(orig_graph_init)
  def wrapped_graph_init(self, *args, **kwargs):
    ret = orig_graph_init(self, *args, **kwargs)
    collector.record_graph_init(self)
    return ret

  @functools.wraps(orig_graph_call)
  def wrapped_graph_call(self, input_uops, var_vals, wait=False):
    st = time.perf_counter()
    ret = orig_graph_call(self, input_uops, var_vals, wait=wait)
    collector.record_graph_call(self, wait, (time.perf_counter()-st)*1000.0, ret)
    return ret

  hcq.HCQProgram.__call__ = wrapped_call
  graph_hcq.HCQGraph.__init__ = wrapped_graph_init
  graph_hcq.HCQGraph.__call__ = wrapped_graph_call
  try:
    yield
  finally:
    hcq.HCQProgram.__call__ = orig_call
    graph_hcq.HCQGraph.__init__ = orig_graph_init
    graph_hcq.HCQGraph.__call__ = orig_graph_call

def run_eager_smoke() -> dict[str, Any]:
  from tinygrad import Tensor, Device, dtypes
  Tensor.manual_seed(0)
  a = Tensor.randn(512, 512, dtype=dtypes.half).realize()
  b = Tensor.randn(512, 512, dtype=dtypes.half).realize()
  st = time.perf_counter()
  c = (a @ b).realize()
  Device[Device.DEFAULT].synchronize()
  return {"name": "tinygrad_hcq_eager_matmul", "shape": list(c.shape), "wall_ms": round((time.perf_counter()-st)*1000.0, 6)}

def run_graph_smoke() -> dict[str, Any]:
  from tinygrad import Tensor, Device, TinyJit, dtypes
  old_graph_one = os.environ.get("GRAPH_ONE_KERNEL")
  os.environ["GRAPH_ONE_KERNEL"] = "1"
  Tensor.manual_seed(1)
  a = Tensor.randn(256, 256, dtype=dtypes.half).realize()
  b = Tensor.randn(256, 256, dtype=dtypes.half).realize()
  def f(x, y): return (x @ y).realize()
  jf = TinyJit(f)
  walls = []
  out = None
  try:
    for _ in range(4):
      st = time.perf_counter()
      out = jf(a, b)
      Device[Device.DEFAULT].synchronize()
      walls.append(round((time.perf_counter()-st)*1000.0, 6))
  finally:
    if old_graph_one is None: os.environ.pop("GRAPH_ONE_KERNEL", None)
    else: os.environ["GRAPH_ONE_KERNEL"] = old_graph_one
  return {"name": "tinygrad_hcq_tinyjit_matmul", "shape": list(out.shape) if out is not None else None, "walls_ms": walls,
          "jit_count": getattr(jf, "cnt", None), "graph_one_kernel": True}

def run_tensile_eager_smoke() -> dict[str, Any]:
  try:
    from extra.qk_tensile_runtime import TensileRunner
    from extra.qk_tensile_hcq_launch import unbundle
    from tinygrad import Tensor, Device, dtypes
  except Exception as e:
    return {"name": "tensile_eager_smoke", "skipped": True, "reason": repr(e)}
  try:
    dev = Device[Device.DEFAULT]
    caps = {json.loads(l)["role"]: json.loads(l) for l in open(ROOT / "bench/qk-tensile-extraction/kernarg_all.jsonl")}
    runner = TensileRunner(dev, "attn_q_o", caps["attn_q_o"], unbundle())
    T, K, N = 512, 4096, 4096
    Tensor.manual_seed(2)
    A = Tensor.randn(K, T, dtype=dtypes.half).contiguous().realize()
    B = Tensor.randn(N, K, dtype=dtypes.half).contiguous().realize()
    C = Tensor.zeros(N, T, dtype=dtypes.half).contiguous().realize()
    st = time.perf_counter()
    runner(C.uop.buffer._buf, A.uop.buffer._buf, B.uop.buffer._buf, wait=True)
    wall_ms = (time.perf_counter()-st)*1000.0
    return {"name": "tensile_eager_attn_q_o", "skipped": False, "wall_ms": round(wall_ms, 6)}
  except Exception as e:
    return {"name": "tensile_eager_smoke", "skipped": True, "reason": repr(e)}

def classify(result:dict[str, Any]) -> list[str]:
  labels: list[str] = []
  pmu = result.get("pmu_probe") or {}
  if pmu.get("tinygrad_hcq", {}).get("classification") == "rocprof_hcq_visibility_gap": labels.append("rocprof_hcq_visibility_gap")
  if result["summary"]["graph_count"] == 0: labels.append("graph_capture_missing")
  elif result["summary"]["graph_replay_count"] > 0: labels.append("graph_rebind_ok")
  if result["summary"]["eager_program_count"] > 0 and result["summary"]["graph_count"] == 0: labels.append("host_sync")
  if not labels: labels.append("unknown")
  return labels

def build_result(args:argparse.Namespace) -> dict[str, Any]:
  os.environ.setdefault("DEV", "AMD")
  collector = Collector()
  workloads = []
  with attribution_context(collector):
    workloads.append(run_eager_smoke())
    workloads.append(run_graph_smoke())
    if args.include_tensile: workloads.append(run_tensile_eager_smoke())

  pmu_probe = _read_json(ROOT / "bench/qk-pmu-observability/result.json")
  summary = {
    "program_count": len(collector.programs),
    "eager_program_count": sum(1 for p in collector.programs if p["kind"] == "eager_program"),
    "graph_program_count": sum(1 for p in collector.programs if p["kind"] == "graph_program"),
    "graph_count": len(collector.graphs),
    "graph_replay_count": len(collector.graph_calls),
    "copy_count": sum(g.get("copy_count", 0) for g in collector.graphs),
    "wait_return_raw_sum": round(sum(p["timing"].get("wait_return_raw") or 0.0 for p in collector.programs), 6),
    "host_ms_sum": round(sum(p["timing"].get("host_ms") or 0.0 for p in collector.programs) +
                         sum(c.get("wall_ms") or 0.0 for c in collector.graph_calls), 6),
  }
  result = {
    "schema": "qk_hcq_attribution_v1",
    "generated_at": _now(),
    "commit": _git_commit(),
    "device": os.environ.get("DEV", "AMD"),
    "backend": "AMD",
    "mode": "probe_local",
    "workload": {"name": "pmu4_hcq_attribution_smokes", "items": workloads},
    "summary": summary,
    "programs": collector.programs,
    "graphs": collector.graphs,
    "graph_calls": collector.graph_calls,
    "pmu_probe": {
      "path": "bench/qk-pmu-observability/result.json",
      "verdict": pmu_probe.get("verdict") if pmu_probe else None,
      "tinygrad_hcq": pmu_probe.get("tinygrad_hcq") if pmu_probe else None,
      "hip_control": pmu_probe.get("hip_control", {}).get("verdict") if pmu_probe else None,
    },
    "classification": [],
    "provenance": ["extra/qk_hcq_attribution.py", "docs/primitive-hcq-attribution-scope-20260619.md"],
  }
  result["classification"] = classify(result)
  return result

def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", type=pathlib.Path, default=OUT / "result.json")
  ap.add_argument("--include-tensile", action="store_true")
  args = ap.parse_args()
  result = build_result(args)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"out": str(args.out.relative_to(ROOT)), "classification": result["classification"],
                    "programs": result["summary"]["program_count"], "graphs": result["summary"]["graph_count"],
                    "graph_replays": result["summary"]["graph_replay_count"]}, indent=2))
  return 0 if result["summary"]["program_count"] > 0 else 2

if __name__ == "__main__":
  raise SystemExit(main())
