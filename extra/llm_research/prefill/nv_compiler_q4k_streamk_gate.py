from __future__ import annotations

import argparse, json, statistics, time
import numpy as np

from tinygrad import Device, Tensor, dtypes
from tinygrad.runtime.ops_nv import NVProgram
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
from extra.llm_research.prefill.nv_compiler_q4k_pp512_binding import M, N, K, RECORD_U32, binding_for as compiler_binding_for
from extra.llm_research.prefill.nv_q4_imma_pp512_binding import binding_for as fixup_binding_for, call_native, native_nv_program
from extra.llm_research.prefill.nv_q4_imma_provider import DYNAMIC_SHARED_BYTES, NV_RUNTIME_SHARED_BYTES, PARTIAL_SLOTS
from extra.llm_research.prefill.nv_q4_imma_72main_proxy import load_model_and_tokenizer
from tinygrad.codegen.opt.persistent_accumulator import owner_segments
from tinygrad.codegen.opt.stream_k import StreamKSchedule
from extra.llm_research.prefill.nv_compiler_q4k_streamk_transform import active_fixup_source

def main() -> None:
  ap=argparse.ArgumentParser(); ap.add_argument("--model",required=True); ap.add_argument("--main-cubin",required=True)
  ap.add_argument("--rounds",type=int,default=5); args=ap.parse_args()
  dev=Device["NV"]
  model,_=load_model_and_tokenizer(args.model,4608,seed=20260617)
  words=model.blk[0].ffn_gate.prefill_packed_weight().contiguous().realize()
  compiler=compiler_binding_for("NV"); fixup=fixup_binding_for("NV")
  schedule=StreamKSchedule(M,N,K,128,128,64,170,8)
  fixup_map=np.full((schedule.output_tiles,2),-1,dtype=np.int32)
  for owner in range(schedule.owners):
    for segment in owner_segments(schedule,owner):
      if not segment.direct:
        row=fixup_map[segment.output_tile]
        row[int(row[0] >= 0)]=segment.partial_slot
  fixup_map_tensor=Tensor(fixup_map,device="NV").realize()
  active_tiles=Tensor(np.asarray(sorted({segment.output_tile for owner in range(schedule.owners)
    for segment in owner_segments(schedule,owner) if not segment.direct}),dtype=np.int32),device="NV").realize()
  fixup_cubin=NVRTCCompiler(dev.arch,ptx=False,cache_key="q4_compiler_streamk_active_fixup_v1").compile(active_fixup_source())
  active_fixup=native_nv_program("q4k_imma_fixup_active",fixup_cubin,global_size=(active_tiles.numel(),1,1),local_size=(256,1,1),
    globals=tuple(range(4)),outs=(0,),ins=(1,2,3),vals=(M,N))
  x=Tensor(np.random.default_rng(20260830).standard_normal((M,K),dtype=np.float32).astype(np.float16),device="NV").realize()
  record=Tensor.empty(RECORD_U32,dtype=dtypes.uint32,device="NV")
  _,record=x.uop_program(record,fxn=lambda *_:compiler.producer); record.realize()
  reference=Tensor.empty(M*N,dtype=dtypes.float32,device="NV").realize()
  call_native(compiler.main_program,reference,record,words); dev.synchronize()
  main_program=native_nv_program("q4k_imma_stream",open(args.main_cubin,"rb").read(),global_size=(170,1,1),local_size=(32,2,4),
    globals=tuple(range(5)),outs=(0,1,2),ins=(3,4),vals=(M,N,K),shared_mem=0)
  out=Tensor.empty(M*N,dtype=dtypes.float32,device="NV").realize()
  partials=Tensor.empty(PARTIAL_SLOTS*128*128,dtype=dtypes.float32,device="NV").realize()
  ids=Tensor.empty(PARTIAL_SLOTS,dtype=dtypes.int32,device="NV").realize()
  def run():
    call_native(main_program,out,partials,ids,words,record)
    call_native(active_fixup,out,partials,fixup_map_tensor,active_tiles)
    return out
  candidate=run(); dev.synchronize()
  ref=reference.numpy(); got=candidate.numpy()
  finite=bool(np.isfinite(got).all()); delta=np.abs(got-ref)
  exact=finite and bool(np.allclose(got,ref,rtol=2e-5,atol=2e-3))
  split={segment.output_tile for owner in range(schedule.owners) for segment in owner_segments(schedule,owner) if not segment.direct}
  tile_max={}
  delta2=delta.reshape(M,N)
  for tile in range(schedule.output_tiles):
    tm,tn=divmod(tile,schedule.tiles_n)
    tile_max[tile]=float(delta2[tm*128:(tm+1)*128,tn*128:(tn+1)*128].max())
  samples=[]
  for _ in range(args.rounds):
    st=time.perf_counter(); run(); dev.synchronize(); samples.append((time.perf_counter()-st)*1e6)
  main_samples=[]
  for _ in range(args.rounds):
    st=time.perf_counter(); call_native(main_program,out,partials,ids,words,record); dev.synchronize(); main_samples.append((time.perf_counter()-st)*1e6)
  fixup_samples=[]
  for _ in range(args.rounds):
    st=time.perf_counter(); call_native(active_fixup,out,partials,fixup_map_tensor,active_tiles); dev.synchronize(); fixup_samples.append((time.perf_counter()-st)*1e6)
  raw_main=NVProgram(dev,"q4k_imma_stream",open(args.main_cubin,"rb").read())
  raw_args=tuple(t.uop.buffer.get_buf("NV") for t in (out,partials,ids,words,record))
  raw_main_samples=[raw_main(*raw_args,global_size=(170,1,1),local_size=(32,2,4),wait=True)*1e6 for _ in range(args.rounds)]
  raw_fixup=NVProgram(dev,"q4k_imma_fixup_active",fixup_cubin)
  raw_fixup_args=tuple(t.uop.buffer.get_buf("NV") for t in (out,partials,fixup_map_tensor,active_tiles))
  raw_fixup_samples=[raw_fixup(*raw_fixup_args,global_size=(active_tiles.numel(),1,1),local_size=(256,1,1),vals=(M,N),wait=True)*1e6
                     for _ in range(args.rounds)]
  result={"passed":exact,"finite":finite,"max_abs":float(delta.max()),"mean_abs":float(delta.mean()),
          "samples_us":samples,"min_us":min(samples),"median_us":statistics.median(samples),
          "main_min_us":min(main_samples),"fixup_min_us":min(fixup_samples),
          "raw_main_min_us":min(raw_main_samples),"raw_main_median_us":statistics.median(raw_main_samples),
          "raw_fixup_min_us":min(raw_fixup_samples),"raw_chain_min_us":min(raw_main_samples)+min(raw_fixup_samples),
          "direct_tile_max_abs":max(v for t,v in tile_max.items() if t not in split),
          "split_tile_max_abs":max(tile_max[t] for t in split),"split_tiles":len(split),
          "worst_tiles":sorted(tile_max.items(),key=lambda item:item[1],reverse=True)[:12],
          "local_size":[32,2,4],"owners":170,"compact_record":True}
  print(json.dumps(result,sort_keys=True))
  if not exact: raise SystemExit(1)

if __name__ == "__main__": main()
