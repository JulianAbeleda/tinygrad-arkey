"""Fail-closed probe for Q6 model-boundary E-family attribution.

Captured TinyJit PROGRAM/CALL objects are process-local; the role arm does not
serialize executable runtimes or buffer handles.  This probe therefore reports
stable doubled identities only and never invents device timing.
"""
from __future__ import annotations
import argparse, json, pathlib

def _uop_dump(u, depth=0, seen=None):
  if seen is None: seen=set()
  if u is None or depth > 5: return "..."
  key=id(u)
  if key in seen: return {"ref": key}
  seen.add(key)
  return {"op": str(getattr(u, "op", None)), "dtype": str(getattr(u, "dtype", None)),
    "arg": str(getattr(u, "arg", None)), "src": [_uop_dump(x, depth+1, seen) for x in getattr(u, "src", ())]}

def probe_live(captured=None, *, out: str | pathlib.Path, downstream_finite: bool, program_calls=None) -> dict:
  """Inspect a live GraphRunner without guessing an invocation ABI."""
  calls = tuple(program_calls or getattr(captured, "calls", ()))
  if not calls and hasattr(captured, "toposort"):
    calls = tuple(x for x in captured.toposort() if getattr(getattr(x, "op", None), "name", "") == "CALL")
  from tinygrad.engine.realize import get_call_arg_uops, get_call_outs_ins
  runtimes = getattr(captured, "runtimes", ())
  rows = []
  call_meta = []
  for i, call in enumerate(calls):
    if not isinstance(call, tuple):
      program = call.src[0]; bufs = list(get_call_arg_uops(call)); device_vars = {}
    else: _, program, bufs, device_vars = call
    name = getattr(getattr(program, "arg", None), "name", "")
    if not name.startswith("E_"): continue
    info = getattr(program, "arg", None)
    outs, ins = get_call_outs_ins(call) if not isinstance(call, tuple) else (tuple(getattr(info, "outs", ())), tuple(getattr(info, "ins", ())))
    if not outs and name: outs = ()
    call_meta.append((i, name, bufs, outs, info))
    rows.append({"index": i, "name": name, "runtime_present": i < len(runtimes) and runtimes[i] is not None,
      "args": [{"dtype": str(getattr(b, "dtype", None)), "shape": [getattr(b, "size", None)]} for b in bufs],
      "device_vars": dict(device_vars), "global_size": getattr(info, "global_size", None),
      "local_size": getattr(info, "local_size", None), "output_dtype": str(getattr(bufs[0], "dtype", None)) if bufs else None})
  edges = []
  for i, name, bufs, outs, info in call_meta:
    if not (name.startswith("q6k") or "q6" in name or name.startswith("r_8_32_32") or name.startswith("r_8_128_32")): continue
    if len(outs) != 1 or not isinstance(outs[0], int) or outs[0] >= len(bufs):
      continue
    produced = id(bufs[outs[0]])
    for j, cname, cbufs, couts, cinfo in call_meta[i+1:]:
      for slot, buf in enumerate(cbufs):
        if id(buf) == produced and cname.startswith("E_"):
          edges.append({"producer_index": i, "producer": name, "consumer_index": j, "consumer": cname,
            "producer_slot": outs[0], "consumer_slot": slot,
            "buffer_dtype": str(getattr(buf, "dtype", None)), "buffer_size": getattr(buf, "size", None),
            "producer_outs": list(outs)})
  try:
    from extra.llm_research.prefill.nv_compiler_q6k_pp512_binding import binding_for
    expected_v_main = binding_for("NV").roles["attn_v"].main_program.arg.name
  except Exception:
    expected_v_main = None
  anchors = [x for x in call_meta if x[1] == expected_v_main] if expected_v_main else []
  raw = []
  if anchors:
    start = anchors[0][0]
    for i, name, bufs, outs, info in call_meta[start:start+5]:
      raw.append({"index": i, "name": name, "program_uop": _uop_dump(info),
        "buffers": [{"slot": s, "object_id": id(b), "dtype": str(getattr(b, "dtype", None)),
                     "size": getattr(b, "size", None)} for s,b in enumerate(bufs)],
      "outs": list(outs), "ins": list(ins)})
  intersections = []
  for left in raw:
    for right in raw:
      if left["index"] < right["index"]:
        common = sorted(set(x["object_id"] for x in left["buffers"]) & set(x["object_id"] for x in right["buffers"]))
        intersections.append({"left": left["index"], "right": right["index"], "shared_object_ids": common})
  payload = {"schema":"tinygrad.nv_compiler_q6k_boundary_runtime_probe.live.v1", "status":"BLOCKED",
    "families": rows, "runtime_timings_us": None, "producer_consumer_edges": edges,
    "raw_call_window": raw, "raw_buffer_intersections": intersections,
    "call_inventory": [{"index": i, "name": name, "buffer_count": len(bufs)} for i,name,bufs,_,_ in call_meta],
    "op_histogram": {},
    "matching": {"expected_v_main": expected_v_main, "matched_indices": [x[0] for x in anchors],
                  "matched_count": len(anchors)},
    "downstream_finite": downstream_finite,
    "limitation":"live runtime invocation requires scheduler-owned launch replacement and cannot be safely reconstructed by this probe"}
  p=pathlib.Path(out); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(payload,indent=2,sort_keys=True,default=str)+"\n")
  return payload

def main() -> int:
  ap = argparse.ArgumentParser(); ap.add_argument("--candidate", required=True); ap.add_argument("--control", required=True); ap.add_argument("--out", required=True)
  a = ap.parse_args(); cand=json.loads(pathlib.Path(a.candidate).read_text()); ctrl=json.loads(pathlib.Path(a.control).read_text())
  ca, co = cand.get("program_names", {}), ctrl.get("program_names", {})
  doubled = {name: {"candidate": ca[name], "control": co[name], "delta": ca[name]-co[name]}
             for name in ca.keys() & co.keys() if ca[name] > co[name]}
  payload = {"schema":"tinygrad.nv_compiler_q6k_boundary_runtime_probe.v1", "status":"BLOCKED",
    "doubled_families": doubled, "runtime_invocation": "UNAVAILABLE",
    "device_durations_us": None, "dtype_shape_edges": None, "pure_conversion_proven": False,
    "limitation":"TinyJit runtimes and buffer handles are process-local and absent from saved model-arm JSON; rerun must instrument the live process"}
  p=pathlib.Path(a.out); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
  print(json.dumps(payload,sort_keys=True)); return 2
if __name__ == "__main__": raise SystemExit(main())
