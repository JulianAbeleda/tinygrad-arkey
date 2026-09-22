#!/usr/bin/env python3
"""Adversarial two-projection finalized-PROGRAM @function composition gate."""
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.function import function
from extra.llm_research.prefill.nv_q4_imma_pp512_binding import binding_for
from extra.llm_research.prefill.nv_q4_imma_provider import M,N,K

def valid_words(seed):
  rng=np.random.default_rng(seed); w=rng.integers(0,2**32,(N,K//256,36),dtype=np.uint32)
  # finite, modest Q4 block scales: D=.00390625, Dmin=.001953125
  lo=np.float16(1/256).view(np.uint16).item(); hi=np.float16(1/512).view(np.uint16).item()
  w[:,:,0]=np.uint32(lo|(hi<<16)); return w

def main():
  b=binding_for();b.prepare_outputs(6);b.begin_trace()
  rng=np.random.default_rng(20260828)
  x1=Tensor(rng.standard_normal((M,K),dtype=np.float32),device="NV").contiguous()
  x2=Tensor(rng.standard_normal((M,K),dtype=np.float32),device="NV").contiguous()
  w1=Tensor(valid_words(1),device="NV").contiguous();w2=Tensor(valid_words(2),device="NV").contiguous()
  def project(x,w,role): return b.project(x,w,model_family="qwen3_8b",role=role)
  eg,eu=project(x1,w1,"ffn_gate"),project(x1,w2,"ffn_up")
  Tensor.realize(eg,eu); er=(eg.numpy(),eu.numpy())
  @function(precompile=True,allow_implicit=True)
  def pair(x,wg,wu): return project(x,wg,"ffn_gate"),project(x,wu,"ffn_up")
  fg,fu=pair(x1,w1,w2); sg,su=pair(x2,w2,w1)
  Tensor.realize(fg,fu,sg,su)
  rows=[]
  for name,a,r in (("gate",fg.numpy(),er[0]),("up",fu.numpy(),er[1])):
    d=np.abs(a-r);rows.append((name,float(d.max()),float(d.mean()),bool(np.allclose(a,r,rtol=2e-5,atol=2e-3))))
  # Distinct second invocation must not alias either first result.
  ga,ua=sg.numpy(),su.numpy(); distinct=not np.array_equal(ga,fg.numpy()) and not np.array_equal(ua,fu.numpy())
  print({"rows":rows,"second_distinct":distinct,"finite":all(np.isfinite(x).all() for x in (ga,ua))})
  if not all(x[3] for x in rows) or not distinct: raise SystemExit(1)
if __name__ == "__main__": main()
