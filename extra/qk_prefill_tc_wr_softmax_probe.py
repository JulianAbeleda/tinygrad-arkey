#!/usr/bin/env python3
"""Option B probe (before SHAPED_WMMA codegen surgery): can proven optimizer-TC matmuls + normal softmax beat
tinygrad SDPA on the Qwen prefill attention shape, even with materialized scores?

WR4 found the manual SHAPED_WMMA custom-kernel idiom is stale (codegen-spec wall). But OptOps.TC still works
through the normal optimizer (it powers the prefill-v2 FFN). So before opening that codegen arc, test the
lowest-risk stack: pure Tensor ops, explicit Q@Kᵀ -> materialized scores -> softmax -> P@V, with GQA via
BROADCAST (K/V stored per kv-head, expanded over the G group dim -- NO repeat_interleave), fp16. NOTE: with
materialized scores the softmax is a normal axis reduction; the WR1-3 warp reductions only help a FUSED kernel
(= Option A), so they don't apply here -- this probe isolates whether concrete-shape TC matmuls alone win.

GPU time = GlobalCounters.time_sum_s under DEBUG>=2 (authoritative; never wall-clock). Gate: explicit beats
SDPA at KV=3584 by >=1.5x AND TC fires for QK/PV -> Option A may be avoidable. Else refuted -> Option A earned.

Run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_prefill_tc_wr_softmax_probe.py
"""
from __future__ import annotations

import json, math, pathlib, re, subprocess, sys

Hq, Hkv, Hd, T = 32, 8, 128, 512
G = Hq // Hkv
KVS = [512, 1024, 3584]

def _explicit(q, k, v, mask, KV):
  # GQA via broadcast: q[Hkv,G,T,Hd] @ k[Hkv,1,Hd,KV] -> [Hkv,G,T,KV] (K/V read once per kv-head, expanded over G)
  from tinygrad import dtypes
  scale = 1.0 / math.sqrt(Hd)
  qg = q.reshape(Hkv, G, T, Hd)
  kg = k.reshape(Hkv, 1, KV, Hd); vg = v.reshape(Hkv, 1, KV, Hd)
  s = (qg @ kg.transpose(-1, -2)).cast(dtypes.float32) * scale     # TC matmul (fp16 inputs)
  s = (s + mask.reshape(1, 1, T, KV)).softmax(-1)                  # materialized scores + normal softmax
  o = s.cast(dtypes.float16) @ vg                                  # TC matmul
  return o.reshape(Hq, T, Hd)

def _sdpa(q, k, v, mask, KV):
  return q.reshape(1, Hq, T, Hd).scaled_dot_product_attention(
    k.reshape(1, Hkv, KV, Hd), v.reshape(1, Hkv, KV, Hd), attn_mask=mask.reshape(1, 1, T, KV), enable_gqa=True).reshape(Hq, T, Hd)

def _child(KV:int):
  import os
  os.environ["DEBUG"] = "2"
  from tinygrad import Tensor, dtypes, GlobalCounters
  sp = KV - T
  def fresh():
    return (Tensor.randn(Hq, T, Hd, dtype=dtypes.float16).realize(),
            Tensor.randn(Hkv, KV, Hd, dtype=dtypes.float16).realize(),
            Tensor.randn(Hkv, KV, Hd, dtype=dtypes.float16).realize())
  qi = Tensor.arange(T).reshape(T, 1); kj = Tensor.arange(KV).reshape(1, KV)
  mask = (kj > sp + qi).where(Tensor(-float("inf")), Tensor(0.0)).cast(dtypes.float16).realize()
  q, k, v = fresh()
  ref = _sdpa(q, k, v, mask, KV).realize().numpy()
  ex = _explicit(q, k, v, mask, KV).realize().numpy()
  import numpy as np
  err = float(np.abs(ex - ref).max()); rmean = float(np.abs(ref).mean())
  def gpu_ms(fn, iters=6):
    best = 1e9
    for _ in range(iters):
      a, b, c = fresh(); GlobalCounters.reset(); fn(a, b, c, mask, KV).realize(); best = min(best, GlobalCounters.time_sum_s)
    return best * 1e3
  t_sdpa = gpu_ms(_sdpa); t_ex = gpu_ms(_explicit)
  print(f"@@R@@{json.dumps({'KV':KV,'err':round(err,5),'rel':round(err/max(rmean,1e-9),4),'sdpa_ms':round(t_sdpa,3),'explicit_ms':round(t_ex,3),'speedup':round(t_sdpa/t_ex,3) if t_ex else None,'score_numel':Hq*T*KV})}")

_TC = re.compile(r"WMMA|wmma|__hip_wmma|tensor")
def _tc_fired(KV:int) -> bool:
  import os
  env = {**os.environ, "PYTHONPATH": ".", "DEBUG": "4"}
  p = subprocess.run([sys.executable, "-c",
    f"import sys; sys.argv=['x']; "
    f"from extra.qk_prefill_tc_wr_softmax_probe import _explicit, Hq,Hkv,Hd,T; "
    f"from tinygrad import Tensor,dtypes; KV={KV}; sp=KV-T; "
    f"q=Tensor.randn(Hq,T,Hd,dtype=dtypes.float16).realize(); k=Tensor.randn(Hkv,KV,Hd,dtype=dtypes.float16).realize(); v=Tensor.randn(Hkv,KV,Hd,dtype=dtypes.float16).realize(); "
    f"qi=Tensor.arange(T).reshape(T,1); kj=Tensor.arange(KV).reshape(1,KV); "
    f"mask=(kj>sp+qi).where(Tensor(-float('inf')),Tensor(0.0)).cast(dtypes.float16).realize(); "
    f"_explicit(q,k,v,mask,KV).realize()"],
    capture_output=True, text=True, env=env, timeout=120)
  return bool(_TC.search(p.stdout + p.stderr))

def main():
  if len(sys.argv) >= 3 and sys.argv[1] == "--kv":
    _child(int(sys.argv[2])); return
  import os
  rows = []
  for KV in KVS:
    p = subprocess.run([sys.executable, __file__, "--kv", str(KV)], capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": "."}, timeout=240)
    line = next((l for l in p.stdout.splitlines() if l.startswith("@@R@@")), None)
    if line is None:
      print(f"KV={KV}: child failed:\n{p.stderr[-300:]}"); rows.append({"KV": KV, "faulted": True}); continue
    r = json.loads(line[5:]); rows.append(r)
    print(f"KV={r['KV']:5d}: err={r['err']:.4f}(rel {r['rel']}) | sdpa {r['sdpa_ms']:.2f}ms | explicit {r['explicit_ms']:.2f}ms -> {r['speedup']}x")
  tc = _tc_fired(max(KVS))
  ok = [r for r in rows if not r.get("faulted")]
  long = next((r for r in ok if r["KV"] == max(KVS)), None)
  out = {"shape": {"Hq": Hq, "Hkv": Hkv, "Hd": Hd, "T": T}, "rows": rows, "tc_fired_qk_or_pv": tc,
         "correct": all(r["err"] <= 0.02 for r in ok) if ok else False}
  if long:
    out["beats_sdpa_1p5x"] = long["speedup"] is not None and long["speedup"] >= 1.5
    out["verdict"] = (f"OPTION B VIABLE: explicit TC+softmax {long['speedup']}x over SDPA at KV={long['KV']}, "
                      f"tc_fired={tc} -> SHAPED_WMMA surgery may be avoidable"
                      if (out["beats_sdpa_1p5x"] and out["correct"]) else
                      f"OPTION B REFUTED: {long['speedup']}x at KV={long['KV']} (tc_fired={tc}) -> "
                      f"materialized-scores path doesn't beat SDPA enough; Option A (fused SHAPED_WMMA) earned")
    print(f"tc_fired={tc} | {out['verdict']}")
  art = pathlib.Path("bench/qk-prefill-tc-wr-softmax-probe/result.json"); art.parent.mkdir(parents=True, exist_ok=True)
  art.write_text(json.dumps(out, indent=2)); print(f"artifact: {art}")

if __name__ == "__main__":
  main()
