#!/usr/bin/env python3
"""Target A — llama-shaped GQA-batched cooperative decode-attention tile (isolated build + gates).

llama's flash_attn_tile reuses one K/V tile across the GQA group; tinygrad's hoisted flash_partial_v2 makes the
query head `h` a GLOBAL axis, so V[h//G] is re-read G=4x across the group. This builds the smallest candidate
that isolates that lever: a cooperative partial where V[kv,t,d] is read ONCE per (kv-head,split,d) thread and
reused across the G=4 query heads (G register accumulators), fed the SAME prob as the hoisted path.

Phase 0: hoisted per-kernel baseline (DEBUG=2 tm) at KV 512/1024/2048/4096.
Phase 1: flash_partial_coop vs flash_partial_v2 on identical prob/V -- correctness + tm. Gate >=1.2x @KV1024/4096.

Run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_gqa_coop_decode_attention.py
"""
from __future__ import annotations
import io, json, os, pathlib, re, sys, contextlib
import numpy as np

from tinygrad import Tensor, UOp, Context, GlobalCounters, Device, dtypes
from tinygrad.uop.ops import AddrSpace, AxisType, KernelInfo
from extra.qk_flash_decode import flash_decode_attention, flash_partial_v2_kernel

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_LINE = re.compile(r"\*\*\*\s+(\S+)\s+\d+\s+(.+?)\s+arg\s+\d+\s+mem\s+[\d.]+\s+GB\s+tm\s+([\d.]+)us")
_F32 = dtypes.float32
def _fc(v): return UOp.const(_F32, v)
def _fki(name): return KernelInfo(name=name, opts_to_apply=())
Hd, Hq, Hkv, MAXC = 128, 32, 8, 4608
G = Hq // Hkv
KVS = [512, 1024, 2048, 4096]

def flash_partial_coop_kernel(Hd, Hq, Hkv, MAXC, L, S, Tc):
  """Cooperative GQA partial: V[kv,t,d] read ONCE per (kv,s,d) thread, reused across G query heads via G
  register accumulators (c_regs idiom). pout layout identical to flash_partial_v2 (pout[(h*S+s)*W+d])."""
  Gg = Hq // Hkv; W = Hd + 1
  def kernel(pout, prob, vc):
    kvh = UOp.range(Hkv, 0, AxisType.GLOBAL)
    s = UOp.range(S, 1, AxisType.GLOBAL)
    d = UOp.range(W, 2, AxisType.GLOBAL)
    is_v = d < Hd
    j = UOp.range(L, 3, axis_type=AxisType.REDUCE)
    t = s * L + j; in_r = t < Tc
    t_safe = in_r.where(t, t.const_like(0))
    vd = is_v.where(vc[(kvh * MAXC + t_safe) * Hd + is_v.where(d, d.const_like(0))].cast(_F32), _fc(1.0))  # ONCE
    # G register accumulators (proven multi-reg-reduce pattern from extra/lds_attention_tile.py):
    c = UOp.placeholder((Gg,), _F32, 110, addrspace=AddrSpace.REG)
    zi = UOp.range(Gg, 4); c = c.after(c[zi].store(0.0).end(zi))
    g = UOp.range(Gg, 5)
    p = in_r.where(prob[(kvh * Gg + g) * MAXC + t], _fc(0.0))
    acc = c[g].store(c.after(j)[g] + p * vd).end(g).end(j)
    g2 = UOp.range(Gg, 6); fin = c.after(acc)
    return pout[((kvh * Gg + g2) * S + s) * W + d].store(fin[g2]).end(g2).end(kvh, s, d).sink(
      arg=_fki(f"flash_partial_coop_{Hq}_{Hd}"))
  return kernel

def gpu_tm(fn, prefix=None, warm=3):
  for _ in range(warm): fn()
  best = None
  for _ in range(5):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), Context(DEBUG=2):
      GlobalCounters.reset(); fn()
    per = {}
    for l in buf.getvalue().splitlines():
      if (m := _LINE.search(_ANSI.sub("", l))):
        nm = m.group(2).strip(); key = "_".join(x for x in nm.lower().split("_") if not x.isdigit())
        per[key] = per.get(key, 0.0) + float(m.group(3))
    tot = (per.get(prefix, 0.0) if prefix else sum(per.values()))
    best = tot if best is None else min(best, tot)
  return best, per

def phase0_baseline():
  rng = np.random.default_rng(0)
  q = Tensor(rng.standard_normal((Hq, Hd)).astype(np.float16)).realize()
  k = Tensor(rng.standard_normal((Hkv, MAXC, Hd)).astype(np.float16)).realize()
  v = Tensor(rng.standard_normal((Hkv, MAXC, Hd)).astype(np.float16)).realize()
  rows = []
  for KV in KVS:
    spb = UOp.variable("start_pos", 0, MAXC - 1).bind(KV - 1); spu = UOp.variable("start_pos", 0, MAXC - 1)
    def build(): return flash_decode_attention(q, k, v, spb + 1, spu + 1, Hd, Hq, Hkv, MAXC, 128, variant="hoisted")
    tot, per = gpu_tm(lambda: build().realize())
    rows.append({"KV": KV, "total_us": round(tot, 1),
                 "flash_partial_v2_us": round(per.get("flash_partial_v2", 0.0), 1),
                 "flash_prob_us": round(per.get("flash_prob", 0.0), 1),
                 "breakdown": {k2: round(u, 1) for k2, u in sorted(per.items(), key=lambda x: -x[1])}})
    print(f"  KV={KV:5}: total {tot:7.1f}us | partial_v2 {per.get('flash_partial_v2',0):7.1f}us", file=sys.stderr)
  return rows

def phase1_coop_partial():
  rng = np.random.default_rng(1)
  vc_np = rng.standard_normal((Hkv, MAXC, Hd)).astype(np.float16)
  vc = Tensor(vc_np).reshape(Hkv * MAXC * Hd).realize()
  rows = []
  for KV in KVS:
    L = 128; S = (KV + L - 1) // L; Smax = (MAXC + L - 1) // L; W = Hd + 1
    prob_np = np.zeros((Hq, MAXC), np.float32); prob_np[:, :KV] = rng.standard_normal((Hq, KV)).astype(np.float32)
    prob = Tensor(prob_np).reshape(Hq * MAXC).realize()
    def v2(): return Tensor.empty(Hq * Smax * W, dtype=_F32).custom_kernel(prob, vc, fxn=flash_partial_v2_kernel(Hd, Hq, Hkv, MAXC, L, S, KV))[0]
    def coop(): return Tensor.empty(Hq * Smax * W, dtype=_F32).custom_kernel(prob, vc, fxn=flash_partial_coop_kernel(Hd, Hq, Hkv, MAXC, L, S, KV))[0]
    # both kernels pack output as [h, s, d] with stride S (hs=h*S+s), writing only the first Hq*S*W elements.
    ref = v2().numpy()[:Hq * S * W].reshape(Hq, S, W)
    try:
      got = coop().numpy()[:Hq * S * W].reshape(Hq, S, W)
      err = float(np.abs(got - ref).max())
      ok = err < 1e-3
    except Exception as e:
      rows.append({"KV": KV, "error": f"{type(e).__name__}: {str(e)[:160]}"}); print(f"  KV={KV}: COOP BUILD FAILED: {str(e)[:120]}", file=sys.stderr); continue
    tm_v2, dv2 = gpu_tm(lambda: v2().realize(), "flash_partial_v2")
    tm_co, dco = gpu_tm(lambda: coop().realize(), "flash_partial_coop")
    devs = sorted(set(list(dv2.keys()) + list(dco.keys())))  # sanity
    sp = tm_v2 / tm_co if tm_co else 0
    rows.append({"KV": KV, "v2_us": round(tm_v2, 1), "coop_us": round(tm_co, 1), "speedup": round(sp, 3),
                 "max_err_vs_v2": round(err, 6), "correct": ok})
    print(f"  KV={KV:5}: v2 {tm_v2:7.1f}us | coop {tm_co:7.1f}us -> {sp:.3f}x | err {err:.2g} {'OK' if ok else 'FAIL'}", file=sys.stderr)
  return rows

def main():
  assert Device.DEFAULT == "AMD"
  print("=== Phase 0: hoisted baseline ===", file=sys.stderr)
  base = phase0_baseline()
  print("=== Phase 1: cooperative GQA V-reuse partial vs hoisted flash_partial_v2 ===", file=sys.stderr)
  p1 = phase1_coop_partial()
  wins = [r for r in p1 if r.get("correct") and r.get("speedup", 0) >= 1.2 and r["KV"] in (1024, 4096)]
  verdict = ("PROCEED" if wins else "STOP: GQA V-reuse <1.2x (cache hides the 4x V reread) -- "
             "consistent with decode-block map (V IC-served) + lds_attention_tile refutation")
  out = {"shape": {"Hq": Hq, "Hkv": Hkv, "G": G, "Hd": Hd}, "baseline": base, "phase1_coop_partial": p1,
         "gate": ">=1.2x vs flash_partial_v2 @KV1024 or 4096", "verdict": verdict}
  art = pathlib.Path("bench/qk-gqa-coop-decode-attention/phase1_partial.json"); art.parent.mkdir(parents=True, exist_ok=True)
  art.write_text(json.dumps(out, indent=2))
  print(f"\nVERDICT: {verdict}\nartifact: {art}", file=sys.__stderr__); print("@@DONE@@", file=sys.__stderr__)

if __name__ == "__main__":
  main()
