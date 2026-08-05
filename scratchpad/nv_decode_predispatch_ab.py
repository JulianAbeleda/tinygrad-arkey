#!/usr/bin/env python3
"""Default-off A/B for settled TinyJit input preparation and written-input shadows.

This is intentionally a runtime diagnostic: importing it changes nothing.  The
two candidate paths are installed only inside ``installed_candidates`` and the
original tinygrad methods are restored on exit.
"""
from __future__ import annotations

import argparse, contextlib, hashlib, json, pathlib, statistics, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def median(xs): return statistics.median(xs) if xs else None


class StructuralInputCache:
  """Narrow identity cache for immutable input UOps.

  Identity is deliberately stricter than structural equality.  A changed view
  or tensor misses and takes the full oracle path; concrete buffers and bound
  variables are never cached.
  """
  def __init__(self): self.entries = {}

  def lookup(self, uops):
    key = tuple(id(u) for u in uops)
    ent = self.entries.get(key)
    if ent is None or any(a is not b for a,b in zip(ent[0], uops)) or len(ent[0]) != len(uops): return None
    return ent[1]

  def store(self, uops, structural): self.entries[tuple(id(u) for u in uops)] = (tuple(uops), structural)


def discover_inputs(args, kwargs, Tensor, Ops, flatten, JitError):
  input_tensors = [(name,t) for name,t in list(enumerate(args))+sorted(kwargs.items()) if t.__class__ is Tensor]
  names, tensors = [name for name,_ in input_tensors], [t for _,t in input_tensors]
  for x in args + tuple(kwargs.values()):
    it = x if isinstance(x, (tuple,list)) else x.values() if isinstance(x, dict) else []
    tensors += [t for t in it if t.__class__ is Tensor and not any(t is y for y in tensors)]
  def get_uops(): return flatten([t.uop.src if t.uop.op is Ops.MULTI else [t.uop] for t in tensors])
  if any(u.device is None or u.base.op is Ops.CONST for u in get_uops()): raise JitError("JIT inputs must be real buffers; use .clone()")
  unrealized = [x for x in tensors if not x.uop.is_realized]
  if unrealized: Tensor.realize(*unrealized)
  input_uops = get_uops()
  input_buf_uops = [u.base for u in input_uops if u.base.realized is not None]
  if len(set(input_buf_uops)) != len(input_buf_uops): raise JitError("duplicate inputs to JIT")
  return names, input_uops, input_buf_uops


@contextlib.contextmanager
def installed_candidates(enable_a:bool, enable_b:bool, metrics:dict):
  import tinygrad.engine.jit as tj
  from tinygrad.device import MultiBuffer
  from tinygrad.helpers import DEBUG
  from tinygrad.tensor import Tensor
  from tinygrad.uop.ops import Ops, UOp, graph_rewrite

  original_prepare, original_captured = tj._prepare_jit_inputs, tj.CapturedJit.__call__
  structural_cache, shadows = StructuralInputCache(), {}

  def prepare(args, kwargs):
    st = time.perf_counter_ns()
    if not enable_a:
      out = original_prepare(args, kwargs); metrics.setdefault("prepare_ns", []).append(time.perf_counter_ns()-st); return out
    names, input_uops, input_buf_uops = discover_inputs(args, kwargs, Tensor, Ops, tj.flatten, tj.JitError)
    structural = structural_cache.lookup(input_uops)
    hit = structural is not None
    if structural is None:
      structural = [(*(graph_rewrite(u.substitute({u.base:UOp(Ops.NOOP)}, extra_pm=tj.mop_cleanup),
                                      tj.pm_jit_input_metadata).unbind_all()), u.dtype, u.device) for u in input_uops]
      structural_cache.store(input_uops, structural)
    # Values are invocation-owned.  Rebuild them even on a structural hit.
    bound = tj.merge_dicts([x[1] for x in structural] +
                           [dict(v.unbind() for v in (args + tuple(kwargs.values())) if isinstance(v, UOp))])
    var_vals = {k.expr:v for k,v in bound.items()}
    expected = [(x[0], tuple(sorted(x[1].keys(), key=lambda v: v.expr)), x[2], x[3]) for x in structural]
    metrics.setdefault("a_hits", []).append(hit)
    metrics.setdefault("prepare_ns", []).append(time.perf_counter_ns()-st)
    return input_buf_uops, var_vals, names, expected

  def captured(self, input_uops, var_vals):
    if not enable_b: return original_captured(self, input_uops, var_vals)
    concrete = []
    for idx,u in enumerate(input_uops):
      if u not in self._written_uops: concrete.append(u); continue
      key = (id(self), idx)
      shadow = shadows.get(key)
      if shadow is None:
        shadow = UOp.new_buffer(u.device, u.arg, u.dtype)
        shadows[key] = shadow
      # Preserve the defensive device copy and its stream-ordered dependency.
      st = time.perf_counter_ns()
      call = u.copy_to_device(u.device).call(shadow, u, metadata=())
      tj.run_linear(UOp(Ops.LINEAR, src=(call,)))
      metrics.setdefault("shadow_copy_ns", []).append(time.perf_counter_ns()-st)
      concrete.append(shadow)
    concrete = tuple(concrete)
    if DEBUG >= 1 and len(self.linear.src) >= 10: print(f"jit execs {len(self.linear.src)} calls")
    if (observer:=tj._GRAPH_ADMISSION_OBSERVER.get()) is not None and hasattr(observer, "bind_execution"):
      observer.bind_execution(self.linear, concrete, var_vals)
    try: tj.run_linear(self.linear, var_vals, input_uops=concrete, jit=True)
    except tj.GraphException as exc:
      if (observer:=tj._GRAPH_ADMISSION_OBSERVER.get()) is not None:
        try: observer(tj.GraphConstructorFailureObservation(type(exc).__name__, str(exc)))
        finally: raise
      raise
    return self.ret

  tj._prepare_jit_inputs, tj.CapturedJit.__call__ = prepare, captured
  try: yield
  finally: tj._prepare_jit_inputs, tj.CapturedJit.__call__ = original_prepare, original_captured


def summarize(rows):
  steady = rows[3:]
  return {"samples":len(steady), "wall_us_median":median([x["wall_us"] for x in steady]),
          "wall_us_mad":median([abs(x["wall_us"]-median([y["wall_us"] for y in steady])) for x in steady]),
          "prepare_us_median":median([x["prepare_us"] for x in steady]),
          "shadow_copy_us_median":median([x["shadow_copy_us"] for x in steady if x["shadow_copy_us"] is not None]),
          "tokens_sha256":hashlib.sha256(json.dumps([x["token"] for x in steady]).encode()).hexdigest()}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  ap.add_argument("--depth", type=int, default=512)
  ap.add_argument("--samples", type=int, default=30)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()
  import tinygrad.llm.model as tgm
  from tinygrad.device import Device
  from tinygrad.helpers import Context
  from tinygrad.llm.model import Transformer

  tgm._CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS = frozenset()
  model, _ = Transformer.from_gguf(args.model, 4608)

  def arm(name, a, b):
    model.reset_generation_state(); gen = model.generate([1]*args.depth, chunk_size=32, temperature=0.0)
    metrics, rows = {}, []
    with installed_candidates(a, b, metrics):
      with Context(DEBUG=0): next(gen); next(gen); next(gen)
      Device["NV"].synchronize()
      for _ in range(args.samples+3):
        p0, c0 = len(metrics.get("prepare_ns", [])), len(metrics.get("shadow_copy_ns", []))
        st = time.perf_counter_ns()
        with Context(DEBUG=0): token = int(next(gen))
        Device["NV"].synchronize(); en = time.perf_counter_ns()
        prep = metrics.get("prepare_ns", [])[p0:]
        copy = metrics.get("shadow_copy_ns", [])[c0:]
        rows.append({"token":token, "wall_us":(en-st)/1e3,
                     "prepare_us":sum(prep)/1e3, "shadow_copy_us":sum(copy)/1e3 if copy else None})
    gen.close()
    return {"name":name, "enable_a":a, "enable_b":b, "rows":rows, "summary":summarize(rows),
            "a_cache_hits":sum(metrics.get("a_hits", [])), "a_cache_lookups":len(metrics.get("a_hits", []))}

  # Each candidate and the combination receive an independent reverse bracket.
  spec = [("control_a0",False,False),("candidate_a",True,False),("control_a1",False,False),
          ("candidate_b",False,True),("control_b1",False,False),
          ("candidate_ab",True,True),("control_ab1",False,False)]
  arms = [arm(*x) for x in spec]
  # Determinism gate: every arm must produce the exact same steady token stream.
  hashes = {x["summary"]["tokens_sha256"] for x in arms}
  if len(hashes) != 1: raise RuntimeError(f"candidate changed token stream: {hashes}")
  payload = {"schema":"tinygrad.nv_decode_predispatch_ab.v1", "evidence":"NATIVE_NV_REVERSE_BRACKET",
             "commit":subprocess.check_output(["git","rev-parse","HEAD"], cwd=ROOT, text=True).strip(),
             "route":"DEV=NV", "depth":args.depth, "steady_samples":args.samples, "arms":arms,
             "exact_token_stream":True}
  pathlib.Path(args.out).write_text(json.dumps(payload, indent=2)+"\n")
  print(json.dumps({x["name"]:x["summary"] for x in arms}, indent=2))


if __name__ == "__main__": main()
