#!/usr/bin/env python3
"""L2 in-model validation (via model.__call__ so warmstart APPLIES): gate/up schedule production vs new.

Builds two prefill_v2_jit (old/new warmstart) through model.__call__ (the only path where warmstart applies),
then interleave-measures by swapping model.prefill_v2_jit. Clock-controlled. + correctness.
Run: DEV=AMD PREFILL_V2=1 PYTHONPATH=. .venv/bin/python extra/qk_gateup_sched_ab.py
"""
from __future__ import annotations
import os, time, statistics
import numpy as np
from tinygrad import Tensor, TinyJit, Device, UOp
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.llm.model import Transformer, PREFILL_UBATCH

def main():
  assert os.environ.get("PREFILL_V2")
  Tensor.manual_seed(0)
  model,_=Transformer.from_gguf("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf",2048)
  N=PREFILL_UBATCH
  t=Tensor([5,6,7,8,9,10]*200+[0]*(2048-1200),dtype="int32").reshape(1,2048); chunk=t[:,0:N].contiguous(); temp=Tensor([0.0])
  vsp=UOp.variable("start_pos",0,2047)
  old=dict(model._pf16_warmstart); gu=(frozenset({12288,N}),4096)
  new=dict(old); new[gu]=(Opt(OptOps.TC,0,(-1,2,1)),Opt(OptOps.UPCAST,0,2),Opt(OptOps.UPCAST,1,4),Opt(OptOps.UNROLL,0,16),Opt(OptOps.LOCAL,1,4))
  def build(ws):
    model._pf16_warmstart=ws; model.prefill_v2_jit=TinyJit(model.forward)
    model(chunk, vsp.bind(0), temp).realize(); Device["AMD"].synchronize()
    return model.prefill_v2_jit
  jold=build(old); jnew=build(new)
  def run(j, ws):
    model.prefill_v2_jit=j; model._pf16_warmstart=ws
    r=model(chunk, vsp.bind(0), temp).realize(); Device["AMD"].synchronize(); return r
  ro=run(jold,old).float().numpy(); rn=run(jnew,new).float().numpy()
  rel=float(np.sqrt(((rn-ro)**2).mean())/(np.sqrt((ro**2).mean())+1e-9))
  print(f"correctness rel_err(new vs old) = {rel:.6f}  {'OK' if rel<2e-2 else 'WRONG'}")
  for _ in range(25): run(jold,old); run(jnew,new)
  told,tnew=[],[]
  for _ in range(25):
    for j,ws,acc in ((jold,old,told),(jnew,new,tnew)):
      model.prefill_v2_jit=j; model._pf16_warmstart=ws
      t0=time.perf_counter(); model(chunk, vsp.bind(0), temp).realize(); Device["AMD"].synchronize(); acc.append(time.perf_counter()-t0)
  om,nm=statistics.median(told),statistics.median(tnew)
  print(f"OLD gate/up (UNROLL8):       {N/om:6.0f} tok/s ({1000*om:.1f}ms)")
  print(f"NEW gate/up (UNROLL16+LOCAL): {N/nm:6.0f} tok/s ({1000*nm:.1f}ms)")
  print(f"SPEEDUP = {om/nm:.3f}x")

if __name__=="__main__": main()
