#!/usr/bin/env python3
"""Fast real-weight 72-main capture proxy for pp512 Q4 IMMA kernel sweeps."""
from __future__ import annotations
import argparse, json, time
import numpy as np
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.llm.generate import load_model_and_tokenizer
from extra.llm_research.prefill.nv_q4_imma_provider import M,N,K,PARTIAL_SLOTS,MAIN_GRID,BLOCK,DYNAMIC_SHARED_BYTES,NV_RUNTIME_SHARED_BYTES,compile_provider,provider_programs
from extra.llm_research.prefill.nv_native_program_uop import call_native,native_nv_program

MODEL="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"

def main():
  ap=argparse.ArgumentParser(); ap.add_argument("--model",default=MODEL); ap.add_argument("--warmups",type=int,default=3)
  ap.add_argument("--main-cubin",default="",help="qualified one-symbol main cubin override")
  ap.add_argument("--rounds",type=int,default=9); args=ap.parse_args()
  dev=Device["NV"]
  model,_=load_model_and_tokenizer(args.model,4608,seed=20260617)
  words=[p.prefill_packed_weight().contiguous().realize() for b in model.blk for p in (b.ffn_gate,b.ffn_up)]
  if len(words)!=72 or any(w.dtype!=dtypes.uint32 or w.numel()!=N*(K//256)*36 for w in words):
    raise RuntimeError("proxy requires 72 canonical real Qwen3-8B gate/up Q4_K buffers")
  provider=compile_provider(dev); main_program,_=provider_programs(provider)
  if args.main_cubin:
    main_program=native_nv_program("q4k_imma_stream",open(args.main_cubin,"rb").read(),global_size=MAIN_GRID,local_size=BLOCK,
      globals=tuple(range(7)),outs=(0,1,2),ins=(3,4,5,6),vals=(M,N,K),shared_mem=DYNAMIC_SHARED_BYTES+NV_RUNTIME_SHARED_BYTES)
  # Distinct buffers reproduce the captured model's address/TLB/cache pressure.
  q8=[]; scales=[]; sums=[]; outs=[]; partials=[]; ids=[]
  base_q=((Tensor.arange(M*K)*17+3)%255-127).cast(dtypes.int8).realize()
  for _ in words:
    q8.append((base_q+Tensor.zeros(1,device="NV",dtype=dtypes.int8)).contiguous().realize())
    scales.append(Tensor.ones(M*(K//32),device="NV").realize())
    sums.append(Tensor.zeros(M*(K//32),device="NV").realize())
    outs.append(Tensor.empty(M*N,device="NV").realize())
    partials.append(Tensor.empty(PARTIAL_SLOTS*128*128,device="NV").realize())
    ids.append(Tensor.empty(PARTIAL_SLOTS,dtype=dtypes.int32,device="NV").realize())
  dummy=Tensor([0],device="NV").realize()
  @TinyJit
  def batch(token):
    for a in range(72): call_native(main_program,outs[a],partials[a],ids[a],words[a],q8[a],scales[a],sums[a])
    return token
  times=[]
  for _ in range(args.warmups+args.rounds):
    st=time.perf_counter(); batch(dummy).realize(); dev.synchronize(); times.append((time.perf_counter()-st)*1e3)
  # Replay must deterministically reproduce representative direct/partial/id locations.
  out_idx=np.asarray([0,127,128,16383,M*N-1]); partial_idx=np.asarray([0,127,128,16383,partials[0].numel()-1])
  before=[(o.numpy().reshape(-1)[out_idx].copy(),p.numpy().reshape(-1)[partial_idx].copy(),i.numpy()[:8].copy())
          for o,p,i in zip(outs[::35],partials[::35],ids[::35])]
  batch(dummy).realize(); dev.synchronize()
  after=[(o.numpy().reshape(-1)[out_idx],p.numpy().reshape(-1)[partial_idx],i.numpy()[:8])
         for o,p,i in zip(outs[::35],partials[::35],ids[::35])]
  exact=all(all(np.array_equal(x,y) for x,y in zip(a,b)) for a,b in zip(before,after))
  finite=all(np.isfinite(x).all() and np.isfinite(y).all() for x,y,_ in after)
  hot=times[args.warmups:]
  result={"status":"PASS" if exact and finite else "FAIL","main_cubin":args.main_cubin or "provider-default",
          "real_weight_buffers":len(words),"distinct_scratch":True,
          "captured_main_calls":72,"sample_replay_exact":exact,"sample_finite":finite,"r_ms":hot,
          "wall_min_ms":min(hot),"wall_median_ms":float(np.median(hot)),"per_main_wall_us":min(hot)*1000/72}
  print(json.dumps(result,sort_keys=True))
  if result["status"]!="PASS": raise SystemExit(1)

if __name__=="__main__": main()
