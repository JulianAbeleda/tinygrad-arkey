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


def _find():
  from extra.qk_beam_log import gen_candidates
  from extra.qk_loop_live import live_time_shape, _cand_feature_rows
  from extra.qk_loop_learnability import load_merged, _train_predict
  corpus = load_merged(); ren = Device["AMD"].renderer
  out = []
  for (M, K, N) in FFN_SHAPES:
    cands = gen_candidates()
    live = live_time_shape(M, K, N, ren, cands)
    pred = _train_predict(corpus, _cand_feature_rows(M, K, N, cands))
    order = [int(i) for i in np.argsort(-pred) if live[int(i)]["valid"]]
    bi = order[0]
    opts = [{"op": o.op.name, "axis": o.axis, "arg": o.arg} for o in cands[bi]]
    out.append({"M": M, "K": K, "N": N, "opts": opts, "tflops": round(live[bi]["tflops"], 2)})
    print(f"({M},{K},{N}) guided-best tflops={live[bi]['tflops']:.2f} opts={[o['op']+':'+str(o['axis'])+':'+str(o['arg']) for o in opts]}",
          file=sys.__stdout__)
  OPTS_JSON.parent.mkdir(parents=True, exist_ok=True)
  OPTS_JSON.write_text(json.dumps(out, indent=2) + "\n")
  return 0


def _load_warmstart_map():
  wmap = {}
  for e in json.loads(OPTS_JSON.read_text()):
    opts = tuple(Opt(OptOps[o["op"]], o["axis"], tuple(o["arg"]) if isinstance(o["arg"], list) else o["arg"]) for o in e["opts"])
    wmap[(frozenset({e["M"], e["N"]}), e["K"])] = opts  # forward matmul: out dims {M,N}, reduce K
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
  T = 16; sp = v_sp.bind(0); nt = v_tk.bind(T); inp = t[:, sp:sp+nt]
  for _ in range(3): model(inp, sp, temp).realize()  # warmup/JIT (warmstart fires here)
  GlobalCounters.reset(); R = 5; st = time.perf_counter()
  for _ in range(R): model(inp, sp, temp).realize()
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
