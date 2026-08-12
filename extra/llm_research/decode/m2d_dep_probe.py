#!/usr/bin/env python3
"""Scratch probe: dump the captured candidate decode graph's call/buffer/dep
structure to find why the fp16 combine->o_proj dependency is missing on NV."""
import os, sys, json
sys.path.insert(0, "/home/ubuntu/tinygrad-arkey")
os.environ.setdefault("PYTHONPATH", "/home/ubuntu/tinygrad-arkey")

from tinygrad import Context, Tensor, UOp
from tinygrad.engine.jit import DepsTracker, GraphRunner, _prepare_jit_inputs
from tinygrad.engine.realize import get_call_outs_ins, resolve_params, unwrap_multi
from tinygrad.uop.ops import Ops
from extra.llm_research.decode.nv_epilogue_absorption_ab import (
  COMBINE_F16_PREFIX, COMBINE_F32_PREFIX, _m2_arm_context, _model, _prompt,
)
from extra.llm_research.decode.decode_harness import DEFAULT_MODEL


def main() -> None:
  model_path = os.environ.get("QK_MODEL", DEFAULT_MODEL)
  depth, max_context = 512, 1024
  out_path = os.environ.get("M2D_PROBE_OUT", "/tmp/m2d-dep-probe.json")
  arm = os.environ.get("M2D_PROBE_ARM", "candidate")
  with _m2_arm_context(arm):
    model, _gates = _model(arm, model_path, max_context)
    gen = model.generate(_prompt(model_path, depth), chunk_size=32, temperature=0.0)
    try: prelude = int(next(gen))
    finally: gen.close()
    token, temp = Tensor([[1]], dtype="int32").contiguous(), Tensor([0.0])
    start_pos = UOp.variable("start_pos", 0, max_context - 1)
    with Context(JIT=0): model.forward_with_logits(token, start_pos.bind(depth), temp)
    model.decode_with_logits(token, start_pos.bind(depth + 1), temp)
    model.decode_with_logits(token, start_pos.bind(depth + 2), temp)
    print("direct_greedy", getattr(model, "_decode_direct_greedy_promoted", False),
          "temp", float(temp.item()))
    for name in ("rollout_greedy_logits_jit", "rollout_logits_jit",
                 "rollout_greedy_logits_jit_flash", "rollout_logits_jit_flash",
                 "rollout_greedy_jit", "rollout_greedy_jit_flash"):
      j = getattr(model, name, None)
      if j is not None:
        print(name, "cnt", j.cnt, "captured", j.captured is not None)
    jit = model.rollout_greedy_logits_jit
    if jit.captured is None: jit = model.rollout_logits_jit
    if jit.captured is None: jit = model.rollout_greedy_logits_jit_flash
    captured = jit.captured
    assert captured is not None, "decode jit did not capture"
    input_buf_uops, var_vals, names, _ = _prepare_jit_inputs(
      (token, start_pos.bind(depth + 1), temp), {})
    linear = captured.linear
    result: dict = {"prelude": prelude, "n_outer_calls": len(linear.src), "graphs": []}
    global_deps = DepsTracker()

    def process_batch(member_calls, graph_index: int, outer_index: int) -> None:
      rows = []
      deps = DepsTracker()
      for j, call in enumerate(member_calls):
        ast = call.src[0]
        if ast.op is not Ops.PROGRAM:
          continue
        resolved = resolve_params(call, input_buf_uops)
        bufs = [b.ensure_allocated() for b in (b.buffer for b in resolved)]
        outs, _ins = get_call_outs_ins(call)
        wait = deps.access_resources(bufs, list(outs), j)
        gwait = global_deps.access_resources(bufs, list(outs), (graph_index, j))
        rows.append({
          "j": j, "name": ast.arg.name, "outs": list(outs),
          "bufs": [{"base": id(b.base), "off": b.offset, "n": b.nbytes, "dev": str(b.device)}
                   for b in bufs],
          "waits": [w for w in wait],
          "gwaits": [f"{g[0]}:{g[1]}" for g in gwait],
        })
      result["graphs"].append({"outer_index": outer_index, "graph_index": graph_index,
                               "calls": rows})

    graph_index = 0
    for oi, call in enumerate(linear.src):
      ast = call.src[0]
      if ast.op is Ops.CUSTOM_FUNCTION and ast.arg == "graph":
        batch = ast.src[0].src
        process_batch(batch, graph_index, oi)
        graph_index += 1

    # pair every combine call (any graph) with later readers of its output buffer
    combines, pairs = [], []
    all_rows = [(g["outer_index"], r) for g in result["graphs"] for r in g["calls"]]
    for gidx, r in all_rows:
      if r["name"].startswith(COMBINE_F16_PREFIX) or r["name"].startswith(COMBINE_F32_PREFIX):
        combines.append((gidx, r))
    for gidx, c in combines:
      out_buf = c["bufs"][0]
      readers = []
      for g2, r in all_rows:
        if (g2, r["j"]) <= (gidx, c["j"]): continue
        for bi, b in enumerate(r["bufs"]):
          if b["base"] == out_buf["base"] and bi not in r["outs"]:
            readers.append({"graph": g2, "j": r["j"], "name": r["name"], "slot": bi,
                            "batch_waits": r["waits"], "global_waits": r["gwaits"]})
      pairs.append({"combine_graph": gidx, "combine_j": c["j"], "combine": c["name"],
                    "out": out_buf, "readers": readers})
    result["combine_calls"] = len(combines)
    result["pairs"] = pairs
    with open(out_path, "w") as f: json.dump(result, f, indent=1, sort_keys=True)
    print(json.dumps({"n_outer_calls": len(linear.src), "n_graphs": graph_index,
                      "combines": len(combines),
                      "pairs": [(p["combine_graph"], p["combine_j"], p["combine"][:48],
                                 [(r["graph"], r["j"], r["name"][:48], r["batch_waits"][:6])
                                  for r in p["readers"]])
                                for p in pairs]}, sort_keys=True))


if __name__ == "__main__":
  main()
