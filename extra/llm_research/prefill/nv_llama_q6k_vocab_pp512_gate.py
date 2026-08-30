#!/usr/bin/env python3
"""Isolated native-NV Q8 producer -> llama Q6_K vocabulary consumer fixture."""
import argparse, json, pathlib, time
import numpy as np
from tinygrad import Tensor, dtypes
from extra.llm_research.prefill.nv_llama_q6k_vocab_pp512_binding import programs

def main():
  ap=argparse.ArgumentParser(); ap.add_argument('--fixture',type=pathlib.Path,required=True); ap.add_argument('--reps',type=int,default=20); ap.add_argument('--out',type=pathlib.Path,required=True); a=ap.parse_args()
  def mark(s): print(s,flush=True)
  p=a.fixture; mark('stage:mmap')
  hidden=np.memmap(p/'final-hidden-row.f32',dtype=np.float32,mode='r',shape=(4096,))
  weights=np.memmap(p/'output.weight.q6_k.bin',dtype=np.uint8,mode='r')
  ref=np.memmap(p/'reference-logits.f32',dtype=np.float32,mode='r',shape=(151936,))
  mark('stage:h2d-hidden'); x=Tensor(np.asarray(hidden),dtype=dtypes.float32,device='NV').reshape(4096).cast(dtypes.float16).contiguous()
  mark('stage:h2d-weights')
  w=Tensor(np.asarray(weights),dtype=dtypes.uint8,device='NV').bitcast(dtypes.uint16).contiguous()
  mark('stage:compile-load'); q8,q6=programs(); packet=Tensor.empty((1152,),dtype=dtypes.uint32,device='NV'); out=Tensor.empty((151936,),dtype=dtypes.float32,device='NV')
  mark('stage:q8-launch'); _,packet=x.uop_program(packet,fxn=lambda *_:q8); packet.realize(); mark('stage:q8-sync')
  mark('stage:q6-launch'); w,packet,out=out.uop_program(w,packet,fxn=lambda *_:q6); out.realize(); mark('stage:q6-sync')
  vals=[]
  for _ in range(a.reps):
    t=time.perf_counter(); _,packet=x.uop_program(packet,fxn=lambda *_:q8); packet.realize(); w,packet,out=out.uop_program(w,packet,fxn=lambda *_:q6); out.realize(); vals.append((time.perf_counter()-t)*1e6)
  mark('stage:d2h'); got=out.numpy(); delta=np.abs(got-ref)
  result={'schema':'tinygrad.nv_llama_q6k_vocab_native_fixture.v1','reps':a.reps,'median_us':float(np.median(vals)),'samples_us':vals,'max_abs':float(delta.max()),'mean_abs':float(delta.mean()),'argmax':int(got.argmax()),'reference_argmax':int(ref.argmax()),'finite':bool(np.isfinite(got).all()),'guards':'not-applicable','verdict':'PASS' if float(delta.max())<=0.5 and int(got.argmax())==int(ref.argmax()) else 'STOP'}
  a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,sort_keys=True))
if __name__=='__main__': main()
