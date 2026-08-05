#!/usr/bin/env python3
"""Native-NV P2b attention-O qualification: logits, census, then wall.

Each invocation owns one fresh-process arm. The caller compares control and
attention_o artifacts before admitting timing; the production policy remains
closed throughout.
"""
from __future__ import annotations

import argparse, contextlib, dataclasses, hashlib, io, json, pathlib, re, statistics, time
import numpy as np

from extra.llm_research.decode.nv_predispatch_full_logits_qualification import DEFAULT_MODEL, _load, _prompt


TM_RE = re.compile(r"^\*\*\* NV\s+\d+\s+(\S+)\s+arg\s+\d+.*?tm\s+([\d.]+)(us|ms)/")


def _call_name(call) -> str:
  """The rendered-program name of one scheduled opaque CALL."""
  return str(getattr(getattr(call.src[0], "arg", None), "name", ""))


def _param_slots_written(call) -> tuple[int, ...]:
  """Recover output argument slots from the post-rangeify body, read-only.

  A scheduled CALL does not carry a separate output list: its body STOREs to
  PARAM slots.  Reading those STORE targets is therefore the only reliable
  way to distinguish an input buffer from a copy's written buffer.  This is
  deliberately analysis-only and works after SINK -> LINEAR lowering.
  """
  from tinygrad.uop.ops import Ops, ParamArg
  # A captured JIT has already compiled SINK bodies to PROGRAM. ProgramInfo's
  # immutable outs contract is the exact post-rangeify answer in that form.
  if call.src[0].op in (Ops.PROGRAM, Ops.COPY, Ops.SLICE):
    from tinygrad.engine.realize import get_call_outs_ins
    return tuple(get_call_outs_ins(call)[0])
  slots = set()
  for u in call.src[0].toposort():
    if u.op is not Ops.STORE: continue
    for v in u.src[0].toposort():
      if v.op is Ops.PARAM and isinstance(v.arg, ParamArg):
        slots.add(v.arg.slot)
  return tuple(sorted(slot for slot in slots if slot < len(call.src)-1))


def post_callify_copy_trace(linear, copy_name_fragment:str="86a2") -> list[dict]:
  """Map every E_86a2 writer through post-callify buffer identities.

  The result names the exact written argument slot(s), direct consumers and
  any reachable epilogue input slots.  It never rewrites, realizes, or changes
  allocation.  `buf_uop` identity is intentionally used only while examining
  this one in-memory linear graph; the emitted record contains logical program
  names/slots, shapes and dtypes rather than unstable object ids.
  """
  from tinygrad.uop.ops import Ops
  calls = [u for u in linear.toposort() if u.op is Ops.CALL]
  writes = {call: _param_slots_written(call) for call in calls}
  writer, consumers = {}, {}
  def key(arg):
    try: return arg.buf_uop
    except RuntimeError: return None
  for call in calls:
    for slot in writes[call]:
      if (buf := key(call.src[1+slot])) is not None: writer[buf] = (call, slot)
    for slot, arg in enumerate(call.src[1:]):
      if (buf := key(arg)) is not None: consumers.setdefault(buf, []).append((call, slot, arg))
  records = []
  for copy in calls:
    if copy_name_fragment not in _call_name(copy): continue
    out = {"copy": _call_name(copy), "written_slots": list(writes[copy]), "edges": []}
    for written_slot in writes[copy]:
      buf = key(copy.src[1+written_slot])
      for consumer, slot, arg in consumers.get(buf, []):
        # A copy can feed a transport node first. Walk the unique writer /
        # consumer buffer graph until an O epilogue is reached; branch records
        # are retained rather than guessing which use is semantically relevant.
        chain, seen, frontier = [], set(), [(consumer, slot, arg)]
        while frontier:
          nxt, nslot, narg = frontier.pop(0)
          marker = (id(nxt), nslot)
          if marker in seen: continue
          seen.add(marker)
          edge = {"program": _call_name(nxt), "slot": nslot,
                  "shape": [str(v) for v in narg.shape], "dtype": str(narg.dtype)}
          chain.append(edge)
          if "epi_resadd_4096_4096" in edge["program"]: continue
          # Follow every buffer written by this intermediary to its consumers.
          for next_out in writes[nxt]:
            nbuf = key(nxt.src[1+next_out])
            frontier.extend(consumers.get(nbuf, []))
        out["edges"].append({"written_slot": written_slot, "chain": chain})
    records.append(out)
  return records


def captured_decode_copy_trace(model) -> list[dict]:
  """Read exact compiled decode LINEARs retained by this model's TinyJits."""
  from tinygrad.engine.jit import TinyJit
  from tinygrad.uop.ops import Ops
  seen, traces = set(), []
  def visit(obj):
    if isinstance(obj, TinyJit):
      captured = getattr(obj, "captured", None)
      if captured is None or id(captured) in seen: return
      seen.add(id(captured))
      names = {_call_name(u) for u in captured.linear.toposort() if u.op is Ops.CALL}
      # Exclude prefill and unrelated diagnostic captures. The composed decode
      # authority contains the exact admitted O epilogue family.
      if any("epi_resadd_4096_4096" in name for name in names):
        traces.extend(post_callify_copy_trace(captured.linear))
      return
    if isinstance(obj, dict):
      for value in obj.values(): visit(value)
    elif isinstance(obj, (list, tuple)):
      for value in obj: visit(value)
  visit(vars(model))
  return traces


def _configure(arm:str):
  from tinygrad.callify import CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT
  import tinygrad.llm.model_route_plan as mrp
  expected_redirect = arm != "redirect_off"
  if bool(CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT) != expected_redirect:
    raise RuntimeError(f"{arm=} requires CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT={int(expected_redirect)}")
  mrp._DECODE_Q4K_EPILOGUE_FUSION_PROMOTED_TARGETS = \
    frozenset({("NV", "sm_120")}) if arm == "attention_o" else frozenset()
  # The treatment is the exact composed reopen: fp16 flash combine removes the
  # activation adapter while the q4 epilogue absorbs the residual.
  mrp._DECODE_FLASH_COMBINE_FUSION_PROMOTED_TARGETS = \
    frozenset({("NV", "sm_120")}) if arm == "attention_o" else frozenset()


def _gate_linears(model, arm:str) -> dict[str, int]:
  from tinygrad import Tensor, UOp
  from tinygrad.llm.qk_primitives import Q4KPrimitiveLinear
  seen, counts = set(), {"admitted": 0, "closed": 0}
  def walk(obj):
    if isinstance(obj, Q4KPrimitiveLinear):
      adm = getattr(obj, "route_admission", None)
      if adm is None: return
      want = arm == "attention_o" and getattr(obj, "route_role", "") == "attn_qo"
      obj.route_admission = dataclasses.replace(adm, q4k_epilogue_fusion_promoted=want)
      counts["admitted" if want else "closed"] += 1
      return
    if id(obj) in seen: return
    seen.add(id(obj))
    if isinstance(obj, dict):
      for value in obj.values(): walk(value)
    elif isinstance(obj, (list, tuple, set, frozenset)):
      for value in obj: walk(value)
    elif not isinstance(obj, (Tensor, UOp)) and hasattr(obj, "__dict__"):
      for value in vars(obj).values(): walk(value)
  walk(model)
  # Block zero's residual is the embedding/gather expression, not a precompiled
  # block output. Keep its ordinary residual add: forcing it through the opaque
  # epilogue merely exchanges that add for one adapter. Blocks 1..35 have the
  # exact call-output alias contract proved by the CPU gate.
  if arm == "attention_o" and getattr(model, "blk", None):
    first = getattr(model.blk[0], "attn_output", None)
    adm = getattr(first, "route_admission", None)
    if adm is not None and getattr(adm, "q4k_epilogue_fusion_promoted", False):
      first.route_admission = dataclasses.replace(adm, q4k_epilogue_fusion_promoted=False)
      counts["admitted"] -= 1; counts["closed"] += 1
  return counts


def _model(arm:str, model_path:str, max_context:int):
  _configure(arm)
  from tinygrad.callify import CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT
  model = _load(model_path, max_context)
  gates = _gate_linears(model, arm)
  gates["callify_owned_redirect"] = int(bool(CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT))
  return model, gates


def logits(arm:str, model_path:str, depth:int, count:int, max_context:int) -> tuple[dict, np.ndarray]:
  from tinygrad import Tensor, UOp
  from tinygrad.helpers import Context
  model, gates = _model(arm, model_path, max_context)
  gen = model.generate(_prompt(model_path, depth), chunk_size=32, temperature=0.0)
  try: prelude = int(next(gen))
  finally: gen.close()
  token, temp = Tensor([[1]], dtype="int32").contiguous(), Tensor([0.0])
  start_pos = UOp.variable("start_pos", 0, max_context - 1)
  with Context(JIT=0): _, eager_logits = model.forward_with_logits(token, start_pos.bind(depth), temp)
  eager = eager_logits.numpy()
  if not np.isfinite(eager).all(): raise RuntimeError("non-finite eager logits")
  samples, rows = [], []
  for idx in range(count):
    sample, full = model.decode_with_logits(token, start_pos.bind(depth + 1 + idx), temp)
    row, sid = full.numpy(), int(sample.item())
    if not np.isfinite(row).all() or sid != int(row.argmax(axis=-1).item()):
      raise RuntimeError(f"invalid diagnostic output at row {idx}")
    samples.append(sid); rows.append(row)
  stacked = np.stack(rows)
  digest = hashlib.sha256(np.ascontiguousarray(stacked).view(np.uint8)).hexdigest()
  return {"arm": arm, "mode": "logits", "gates": gates, "prelude_token": prelude, "tokens": samples,
          "shape": list(stacked.shape), "dtype": str(stacked.dtype), "logits_sha256": digest}, stacked


def census(arm:str, model_path:str, depth:int, max_context:int) -> dict:
  from tinygrad import Tensor
  from tinygrad.helpers import Context
  from tinygrad.uop.ops import Ops
  model, gates = _model(arm, model_path, max_context)
  # Capture the exact pre-boundary value forms. This is structural evidence for
  # why an adapter remains; it does not alter program selection or execution.
  import tinygrad.llm.decode_routes as routes
  original_execute, boundary_inputs = routes.execute_promoted_program, []
  def traced_execute(output, *inputs, program):
    if "epi_resadd" in getattr(program.emitter, "__name__", "") or program.program_id.endswith(".gemv") and \
       len(inputs) == 3 and getattr(program, "route_id", "").startswith("decode_q4k"):
      def chain(u):
        ret = []
        while True:
          ret.append(u.op.name)
          if not len(u.src) or u.op in (Ops.BUFFER, Ops.PARAM): break
          u = u.src[0]
        return ret
      boundary_inputs.append([{"op": x.uop.op.name, "shape": [str(v) for v in x.shape],
                               "precompiled_identity": x.uop.has_precompiled_output_identity(),
                               "chain": chain(x.uop)} for x in inputs])
    return original_execute(output, *inputs, program=program)
  routes.execute_promoted_program = traced_execute
  # Post-callify identity audit: link every 86a2 copy output buffer to the exact
  # input slot of an epilogue invocation in the same executable linear graph.
  original_linear, copy_traces = Tensor.linear_with_vars, []
  def traced_linear(self, *others):
    # During TinyJit capture create_linear_with_vars deliberately returns an
    # empty LINEAR after handing the real schedule to capturing[0].add_linear.
    # Remember that list boundary so this read-only observer sees the exact
    # pre-memory-plan schedule instead of the empty capture sentinel.
    from tinygrad.engine.realize import capturing
    capture_owner = capturing[0] if capturing and hasattr(capturing[0], "_linears") else None
    capture_count = len(capture_owner._linears) if capture_owner is not None else 0
    linear, var_vals = original_linear(self, *others)
    # This is after recursive callify/rangeify scheduling, not the raw custom
    # kernel argument graph.  It is the authoritative location for the 86a2
    # copies because the physical writer/reader slots are now concrete.
    observed = capture_owner._linears[capture_count:] if capture_owner is not None else (linear,)
    for scheduled in observed: copy_traces.extend(post_callify_copy_trace(scheduled))
    return linear, var_vals
  Tensor.linear_with_vars = traced_linear
  # The first decode after prefill is TinyJit's cnt==0 eager arm. It has no
  # retained CapturedJit yet, and DEBUG observes exactly that eager execution.
  # Observe the compiled eager LINEAR at the compilation boundary as well.
  import tinygrad.engine.realize as realize_module
  import tinygrad.engine.jit as jit_module
  original_realize_compile, original_jit_compile = realize_module.compile_linear, jit_module.compile_linear
  def traced_compile(linear, *args, **kwargs):
    compiled = original_realize_compile(linear, *args, **kwargs)
    copy_traces.extend(post_callify_copy_trace(compiled))
    return compiled
  realize_module.compile_linear = traced_compile
  jit_module.compile_linear = traced_compile
  gen = model.generate(_prompt(model_path, depth), chunk_size=32, temperature=0.0)
  with Context(DEBUG=0): next(gen)
  capture = io.StringIO()
  with contextlib.redirect_stdout(capture):
    with Context(DEBUG=2): token = int(next(gen))
  # TinyJit retains the compiled, memory-planned graph that DEBUG just ran.
  # This is later than the per-Tensor schedule hook and therefore carries both
  # rendered E hashes and ProgramInfo output slots.
  copy_traces.extend(captured_decode_copy_trace(model))
  gen.close(); routes.execute_promoted_program = original_execute; Tensor.linear_with_vars = original_linear
  realize_module.compile_linear, jit_module.compile_linear = original_realize_compile, original_jit_compile
  rows = []
  for line in capture.getvalue().splitlines():
    if (match := TM_RE.match(line)):
      us = float(match.group(2)) * (1000.0 if match.group(3) == "ms" else 1.0)
      rows.append((match.group(1), us))
  hist: dict[str, list[float]] = {}
  for name, us in rows: hist.setdefault(name, []).append(us)
  return {"arm": arm, "mode": "census", "gates": gates, "token": token, "kernels": len(rows),
          "E_kernels": sum(name.startswith("E_") for name, _ in rows),
          "adapter_86a2": sum("86a2" in name for name, _ in rows),
          "residual_adds": sum(name.startswith("E_32_32_4_02a") for name, _ in rows),
          "fused_attention_o": sum("epi_resadd_4096_4096" in name for name, _ in rows),
          "kernel_us": sum(us for _, us in rows),
          "boundary_inputs": boundary_inputs[:4],
          "post_callify_86a2_trace": copy_traces,
          "histogram": sorted(((name, len(vals), statistics.median(vals)) for name, vals in hist.items()),
                              key=lambda row: (-row[1], -row[2]))}


def timing(arm:str, model_path:str, depth:int, count:int, reps:int, max_context:int) -> dict:
  from tinygrad import Device
  model, gates = _model(arm, model_path, max_context)
  prompt, dev, samples, hashes = _prompt(model_path, depth), Device[Device.DEFAULT], [], []
  for _ in range(reps):
    model.reset_generation_state(); gen = model.generate(prompt.copy(), chunk_size=32, temperature=0.0)
    out = []
    try:
      next(gen); dev.synchronize(); started = time.perf_counter_ns()
      for _ in range(count): out.append(int(next(gen)))
      dev.synchronize(); samples.append((time.perf_counter_ns() - started) / count / 1e6)
    finally: gen.close()
    hashes.append(hashlib.sha256(",".join(map(str, out)).encode()).hexdigest())
  return {"arm": arm, "mode": "timing", "gates": gates, "samples_ms": samples,
          "median_ms": statistics.median(samples), "token_hashes": hashes, "tokens_identical": len(set(hashes)) == 1}


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--arm", choices=("redirect_off", "redirect_on", "attention_o"), required=True)
  ap.add_argument("--mode", choices=("logits", "census", "timing"), required=True)
  ap.add_argument("--model", default=DEFAULT_MODEL); ap.add_argument("--depth", type=int, default=512)
  ap.add_argument("--count", type=int, default=8); ap.add_argument("--reps", type=int, default=3)
  ap.add_argument("--max-context", type=int, default=1024); ap.add_argument("--out", required=True)
  args = ap.parse_args(); path = pathlib.Path(args.out)
  if args.mode == "logits": result, array = logits(args.arm, args.model, args.depth, args.count, args.max_context); np.savez_compressed(path.with_suffix(".npz"), logits=array)
  elif args.mode == "census": result = census(args.arm, args.model, args.depth, args.max_context)
  else: result = timing(args.arm, args.model, args.depth, args.count, args.reps, args.max_context)
  path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n"); print(json.dumps(result, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
