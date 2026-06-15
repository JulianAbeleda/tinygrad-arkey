#!/usr/bin/env python3
"""Step 3: warm-start injection -- force the loop's schedule onto the decode forward's FFN matmuls (no BEAM).

--find  : run the GPU-safe loop on the FFN verify shapes, save the guided-best opts per shape.
--measure [Q4K_WARMSTART=1]: load opts, set postrange._WARMSTART_OPTS, run the T=16 forward, report
          warmstart apply/error stats + ms/tok. Without Q4K_WARMSTART -> baseline (heuristic).

Run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_decode_warmstart.py --find
     DEV=AMD Q4K_PRIMITIVE=1 PYTHONPATH=. python extra/qk_decode_warmstart.py --measure          # baseline
     DEV=AMD Q4K_PRIMITIVE=1 Q4K_WARMSTART=1 PYTHONPATH=. python extra/qk_decode_warmstart.py --measure
"""
from __future__ import annotations
import json, pathlib, statistics, sys, time
import numpy as np
from tinygrad import Device, Tensor
from tinygrad.helpers import getenv
from tinygrad.codegen.opt import Opt, OptOps
import tinygrad.codegen.opt.postrange as pr

OPTS_JSON = pathlib.Path("bench/amd-decode-flywheel-proof-20260614/batch-ceiling-probe/warmstart_opts.json")
# FFN matmul shapes (M=out, K=in) at speculative batch N -- gate/up and down.
FFN_SHAPES = [(12288, 4096, 16), (4096, 12288, 16)]


# FORWARD-LAYOUT matmuls (x @ W.T), 2D with the model's batch-1-squeezed layout: x=(T=16, in), W=(out, in).
# axis 0 = 16 (seq), axis 1 = out -- matching the model's kernel (out-dims {16,out}, reduce=in).
FFN_FWD = [((16, 4096), (12288, 4096)), ((16, 12288), (4096, 12288))]


def _find():
  import math
  from tinygrad import Tensor, dtypes
  from tinygrad.codegen import to_program
  from tinygrad.codegen.opt.search import _time_program
  from test.backend.test_linearizer import helper_realized_ast
  from test.helpers import replace_opts
  from extra.qk_beam_log import gen_candidates
  ren = Device["AMD"].renderer
  out = []
  for xs, ws in FFN_FWD:
    x = Tensor.randn(*xs, dtype=dtypes.float16, device="AMD").realize()
    W = Tensor.randn(*ws, dtype=dtypes.float16, device="AMD").realize()
    ast, bufs = helper_realized_ast(x @ W.T)
    out_dims = sorted([xs[0], ws[0]]); reduce = ws[1]; flops = 2*xs[0]*ws[1]*ws[0]  # out {16,out}, reduce=in
    cands = gen_candidates()
    best_tf, best_opts = 0.0, None
    for opts in cands:
      try:
        t = min(_time_program(to_program(replace_opts(ast, opts), ren), {}, bufs, cnt=2))
        tf = flops/t/1e12 if math.isfinite(t) and t > 0 else 0
        if tf > best_tf: best_tf, best_opts = tf, opts
      except Exception: pass
    opts_repr = [{"op": o.op.name, "axis": o.axis, "arg": o.arg} for o in best_opts]
    out.append({"out_dims": out_dims, "reduce": reduce, "opts": opts_repr, "tflops": round(best_tf, 2)})
    print(f"fwd-layout x{xs}@W{ws}: best tflops={best_tf:.2f} opts={[o['op']+':'+str(o['axis'])+':'+str(o['arg']) for o in opts_repr]}",
          file=sys.__stdout__)
  OPTS_JSON.parent.mkdir(parents=True, exist_ok=True)
  OPTS_JSON.write_text(json.dumps(out, indent=2) + "\n")
  return 0


def _load_warmstart_map():
  wmap = {}
  for e in json.loads(OPTS_JSON.read_text()):
    opts = tuple(Opt(OptOps[o["op"]], o["axis"], tuple(o["arg"]) if isinstance(o["arg"], list) else o["arg"]) for o in e["opts"])
    wmap[(frozenset(e["out_dims"]), e["reduce"])] = opts  # keyed by the forward matmul's (out-dims, reduce)
  return wmap


def _measure():
  from tinygrad.uop.ops import UOp
  from tinygrad.helpers import GlobalCounters
  from tinygrad.llm.model import Transformer
  warm = bool(getenv("Q4K_WARMSTART"))
  if warm:
    pr._WARMSTART_OPTS = _load_warmstart_map()
    print(f"warmstart ENABLED, {len(pr._WARMSTART_OPTS)} shape keys", file=sys.__stdout__)
  model, kv = Transformer.from_gguf("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf", 4096)
  mc = model.max_context
  v_sp = UOp.variable("start_pos", 0, mc-1); v_tk = UOp.variable("toks", 1, 32)
  temp = Tensor([0.0]); t = Tensor([[1]*mc], dtype="int32")
  T = 16
  if getenv("CONCRETE"):  # bypass the symbolic JIT batch var -> concrete N=16 so TC can apply
    from tinygrad import TinyJit
    toks = Tensor([[1]*T], dtype="int32")
    jfwd = TinyJit(model.forward)
    call = lambda: jfwd(toks, 0, temp)
  else:
    sp = v_sp.bind(0); nt = v_tk.bind(T); inp = t[:, sp:sp+nt]
    call = lambda: model(inp, sp, temp)
  for _ in range(3): call().realize()  # warmup/JIT (warmstart fires here)
  GlobalCounters.reset(); R = 5; st = time.perf_counter()
  for _ in range(R): call().realize()
  dt = (time.perf_counter()-st)/R
  print(f"{'WARMSTART' if warm else 'BASELINE '} T=16: {dt*1000:.2f} ms ({dt/T*1000:.2f} ms/tok)", file=sys.__stdout__)
  print(f"warmstart_stats: {pr._warmstart_stats}", file=sys.__stdout__)
  return 0


def main():
  if "--find" in sys.argv: return _find()
  if "--measure" in sys.argv: return _measure()
  print("use --find or --measure", file=sys.__stdout__); return 1


if __name__ == "__main__":
  raise SystemExit(main())
