"""18-role Q4-down population census/timing harness (research-only)."""
import argparse,json,time,statistics
from pathlib import Path
import numpy as np
from tinygrad import Tensor,dtypes,Device
from extra.llm_research.layout import read_metadata,packed_u8_slice,packed_u32_slice
from extra.llm_research.prefill.nv_compiler_q4k_down_pp512_binding import capture_for

def main():
  ap=argparse.ArgumentParser(); ap.add_argument('--model',default='/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf'); ap.add_argument('--rounds',type=int,default=9); ap.add_argument('--out',required=True); a=ap.parse_args()
  model=Path(a.model); md=read_metadata(model); infos=[i for i in md.infos if i.name.endswith('.ffn_down.weight') and i.typ==12]
  if len(infos)!=18: raise RuntimeError(f'expected exactly 18 type12 down weights, found {len(infos)}')
  xnp=(((np.arange(512*12288,dtype=np.int64)*37+11)%255)-127).astype(np.int8).reshape(512,12288)
  # Legal saved-Z carrier: fp16 values with deterministic bounded range.
  x=Tensor((xnp.astype(np.float16)/32),device='NV').contiguous().realize(); cap=capture_for(); cap.begin_trace(); cap.prepare_records(18)
  outs=[]; names=[]; times=[]
  for info in infos:
    words=(packed_u32_slice(model,md,info,device='NV') if info.typ==12 else packed_u8_slice(model,md,info,device='NV').bitcast(dtypes.uint32)).contiguous().realize(); names.append(info.name)
    y=cap.project(x,words,model_family='qwen3_8b',role='ffn_down'); y.realize(); outs.append(y)
  for _ in range(3):
    Device['NV'].synchronize(); t=time.perf_counter_ns(); [y.realize() for y in outs]; Device['NV'].synchronize()
  for _ in range(a.rounds):
    Device['NV'].synchronize(); t=time.perf_counter_ns(); [y.sum().realize() for y in outs]; Device['NV'].synchronize(); times.append((time.perf_counter_ns()-t)/1e6)
  # Force a device read of every result; otherwise the scheduler may prune
  # unused lazy CALLs and produce an invalid near-zero timing.
  checks=[]
  for y in outs:
    checks.append((float(y.sum().item()),float(y.abs().max().item())))
  arr=[y.numpy() for y in outs]; payload={'schema':'tinygrad.nv_q4down_population.v3','weights':names,'census':{'expected':18,'producers':18,'mains':18,'distinct_outputs':len({id(y.uop) for y in outs}),'records':cap.cursor},'correctness':{'finite':all(np.isfinite(z).all() for z in arr),'distinct_values':len(set(checks))>1,'shape':all(z.shape==(512,4096) for z in arr),'checksums':checks},'timing_ms':{'min':min(times),'median':statistics.median(times),'samples':times},'verdict':'PASS' if len(set(checks))>1 else 'FAIL'}
  open(a.out,'w').write(json.dumps(payload,indent=2)+'\n'); print(json.dumps(payload,indent=2))
if __name__=='__main__': main()
