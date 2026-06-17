#!/usr/bin/env python3
"""Prefill v2 — Stage 0 make-or-break gate.

Prior work (`9a17aae4e`) found: injecting the loop's TC opts via `_WARMSTART_OPTS` onto the in-model
prefill matmul ERRORS ("symbolic-batch JIT blocks tensor cores") — the forward's batch dim is the symbolic
`v_toks`, and TC needs concrete dims. The UNTESTED combination is **concrete ubatch + fp16 + warmstart-TC**:
M1 tried concrete batch alone (no warmstart -> heuristic), Step-3 tried warmstart alone (symbolic -> error).

This gate answers: with a CONCRETE batch, does the loop's best TC schedule (a) APPLY (no KernelOptError) and
(b) lift the matmul toward standalone peak? If yes -> the per-matmul wall is breakable by a concrete-ubatch
prefill-mode forward (GREENLIGHT Stage 1). If TC still errors / no recovery -> structural wall (rocBLAS/park).

Measures, fp16, on real 8B FFN shapes:
  1. standalone best (enumerate TC+UPCAST/LOCAL, replace_opts) -> anchor %peak.
  2. warmstart-concrete: populate _WARMSTART_OPTS with that best, realize A@B at concrete N -> apply/error
     stats + achieved %peak (the untested fix).
  3. warmstart-symbolic: same but bind a symbolic N -> reproduce the prior TC-error (control).
"""
from __future__ import annotations

import itertools, json, math, sys

PEAK_TF = 83.6  # fp16 compute peak, gfx1100 (measured, docs)
# 8B FFN matmul shapes (M=out, K=in, N=ubatch). gate/up: 12288x4096; down: 4096x12288.
SHAPES = [(12288, 4096, 512), (4096, 12288, 512), (4096, 4096, 512)]

def _candidates():
  from tinygrad.codegen.opt import Opt, OptOps
  tcs = [Opt(OptOps.TC, 0, (-1, 2, 1)), Opt(OptOps.TC, 0, (-1, 1, 1))]
  extras = ([Opt(OptOps.UPCAST, 0, a) for a in (2, 4)] + [Opt(OptOps.UPCAST, 1, a) for a in (2, 4)] +
            [Opt(OptOps.LOCAL, 1, a) for a in (2, 4)])
  cands = []
  for tc in tcs:
    cands.append([tc])
    for e in extras: cands.append([tc, e])
    for e1, e2 in itertools.combinations(extras, 2): cands.append([tc, e1, e2])
  return cands

def best_standalone(M, K, N) -> dict:
  from tinygrad import Tensor, dtypes, Device
  from tinygrad.codegen import to_program
  from tinygrad.codegen.opt.search import _time_program
  from test.backend.test_linearizer import helper_realized_ast
  from test.helpers import replace_opts
  ren = Device[Device.DEFAULT].renderer
  A = Tensor.randn(M, K, dtype=dtypes.float16, device="AMD").realize()
  B = Tensor.randn(K, N, dtype=dtypes.float16, device="AMD").realize()
  ast, bufs = helper_realized_ast(A @ B)
  flops = 2 * M * K * N
  best = None
  for opts in _candidates():
    try:
      prg = to_program(replace_opts(ast, opts), ren)
      t = min(_time_program(prg, {}, bufs, cnt=3))
      if math.isfinite(t) and t > 0:
        tf = flops / t / 1e12
        if best is None or tf > best["tflops"]: best = {"tflops": round(tf, 2), "opts": opts}
    except Exception:
      pass
  if best is None: return {"tflops": None, "pct_peak": None, "opts": None}
  best["pct_peak"] = round(100 * best["tflops"] / PEAK_TF, 1)
  return best

def warmstart_concrete(M, K, N, opts) -> dict:
  """Populate _WARMSTART_OPTS with `opts` for this shape, realize a CONCRETE-N matmul, report apply/error +
  achieved tflops. This is the untested fix: concrete batch should let TC apply (vs the symbolic-batch error)."""
  import tinygrad.codegen.opt.postrange as pr
  from tinygrad import Tensor, dtypes, GlobalCounters
  import time
  key = (frozenset({M, N}), K)
  pr._WARMSTART_OPTS = {key: tuple(opts)}
  pr._warmstart_stats = {"match": 0, "apply": 0, "error": 0}
  A = Tensor.randn(M, K, dtype=dtypes.float16, device="AMD").realize()
  B = Tensor.randn(K, N, dtype=dtypes.float16, device="AMD").realize()
  (A @ B).realize()                       # first: compile (warmstart fires here)
  ts = []
  for _ in range(5):
    GlobalCounters.reset(); t0 = time.perf_counter(); (A @ B).realize(); ts.append(time.perf_counter() - t0)
  pr._WARMSTART_OPTS = None
  t = min(ts); flops = 2 * M * K * N
  tf = flops / t / 1e12 if t > 0 else 0.0
  st = pr._warmstart_stats
  return {"match": st["match"], "apply": st["apply"], "error": st["error"],
          "tflops": round(tf, 2), "pct_peak": round(100 * tf / PEAK_TF, 1)}

def chained_ffn(N, per_shape_opts) -> dict:
  """The dominant ~27x factor was the @function block fusing the matmul CHAIN. Test the FFN chain
  (gate->silu*up->down) with concrete N, fp16, .contiguous() ISOLATION between matmuls, and warmstart on
  each shape. If the aggregate stays near the per-matmul 43% (not collapsing to ~5%), isolation+warmstart
  handles the chaining -> the prefill-mode forward is viable."""
  import tinygrad.codegen.opt.postrange as pr
  from tinygrad import Tensor, dtypes, GlobalCounters
  import time
  H, FF = 4096, 12288
  pr._WARMSTART_OPTS = {(frozenset({M, N}), K): tuple(o) for (M, K), o in per_shape_opts.items()}
  pr._warmstart_stats = {"match": 0, "apply": 0, "error": 0}
  x  = Tensor.randn(N, H, dtype=dtypes.float16, device="AMD").realize()
  Wg = Tensor.randn(FF, H, dtype=dtypes.float16, device="AMD").realize()
  Wu = Tensor.randn(FF, H, dtype=dtypes.float16, device="AMD").realize()
  Wd = Tensor.randn(H, FF, dtype=dtypes.float16, device="AMD").realize()
  def ffn():
    g = (x @ Wg.T).contiguous()          # isolate each matmul so warmstart matches a clean kernel
    u = (x @ Wu.T).contiguous()
    h = (g.silu() * u).contiguous()
    return (h @ Wd.T).contiguous()
  ffn().realize()
  ts = []
  for _ in range(5):
    GlobalCounters.reset(); t0 = time.perf_counter(); ffn().realize(); ts.append(time.perf_counter() - t0)
  pr._WARMSTART_OPTS = None
  t = min(ts); flops = 2 * N * (FF*H + FF*H + H*FF)  # gate + up + down
  tf = flops / t / 1e12 if t > 0 else 0.0
  st = pr._warmstart_stats
  return {"apply": st["apply"], "error": st["error"], "tflops": round(tf, 2),
          "pct_peak": round(100 * tf / PEAK_TF, 1)}

def main():
  out = {"peak_tf": PEAK_TF, "shapes": []}
  per_shape_opts = {}
  for (M, K, N) in SHAPES:
    base = best_standalone(M, K, N)
    ws = warmstart_concrete(M, K, N, base["opts"]) if base["opts"] else {"error": "no standalone opt"}
    if base["opts"]: per_shape_opts[(M, K)] = base["opts"]
    row = {"M": M, "K": K, "N": N, "standalone_best": base, "warmstart_concrete": ws}
    out["shapes"].append(row)
    print(f"{M}x{K}x{N}: standalone {base['pct_peak']}% peak; warmstart concrete -> "
          f"apply={ws.get('apply')} error={ws.get('error')} {ws.get('pct_peak')}% peak", file=sys.__stdout__)
  chain = chained_ffn(512, per_shape_opts)
  out["chained_ffn_512"] = chain
  print(f"CHAINED FFN (concrete 512, isolated, warmstart): apply={chain['apply']} error={chain['error']} "
        f"-> {chain['pct_peak']}% peak (vs ~5% fused-collapse, ~1.3% in-model today)", file=sys.__stdout__)
  # gate: per-matmul recovers (>=17% on >=1 FFN shape) AND the isolated chain does NOT collapse (>=15%)
  per_ok = any(r["warmstart_concrete"].get("apply", 0) > 0 and r["warmstart_concrete"].get("error", 1) == 0
               and (r["warmstart_concrete"].get("pct_peak") or 0) >= 17 for r in out["shapes"])
  chain_ok = chain.get("error", 1) == 0 and (chain.get("pct_peak") or 0) >= 15
  out["gate_pass"] = per_ok and chain_ok
  print("GATE:", "PASS (concrete warmstart-TC applies per-matmul AND survives the chain -> Stage 1)"
        if out["gate_pass"] else
        f"PARTIAL/FAIL (per_matmul={per_ok}, chain={chain_ok}) -> chaining wall remains; rocBLAS or deeper restructure",
        file=sys.__stdout__)
  print(json.dumps(out, default=str))

if __name__ == "__main__":
  main()
