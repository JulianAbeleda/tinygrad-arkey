"""Research benchmark for the 340-owner Q6_K Stream-K slot producer."""
import statistics, time
import numpy as np
from tinygrad import Device, dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.device import BufferSpec
from tinygrad.runtime.ops_nv import NVProgram
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
from tinygrad.uop.ops import Ops, UOp
from extra.llm_research.prefill.nv_generated_q6k_streamk_slots import q6_streamk_slot_kernel
from extra.llm_research.prefill.nv_generated_q6k_streamk import owner_metadata

def run(samples=10):
  total, slots, maxseg = 48, 340, 37
  rows=[]
  for owner in range(170):
    lo=(owner*6144)//170; hi=((owner+1)*6144)//170
    seg=[]
    for linear in range(lo,hi):
      tile,k=divmod(linear,total)
      if not seg or seg[-1][0]!=tile: seg.append([tile,k,k+1])
      else: seg[-1][2]=k+1
    rows.extend(seg+([[-1,0,0]]*(2-len(seg))))
  desc=np.array(rows,np.int32).reshape(-1)
  rng=np.random.default_rng(20260908)
  blocks=rng.integers(0,256,(4096,total,210),dtype=np.uint8).view(np.uint16).reshape(-1)
  b=rng.integers(-4,5,(total*256,512),dtype=np.int8)
  db=np.full((total,8,512),.0625,np.float32)
  ph=lambda n,dt,i: UOp.placeholder((n,),dt,i)
  ast=q6_streamk_slot_kernel(ph(slots*16384,dtypes.float32,0),ph(slots,dtypes.int32,1),ph(slots*3,dtypes.int32,2),
    ph(blocks.size,dtypes.uint16,3),ph(b.size,dtypes.int8,4),ph(db.size,dtypes.float32,5),total,slots,maxseg)
  src=next(x.arg for x in to_program(ast,CUDARenderer(Target.parse('NV:CUDA:sm_120'))).src if x.op is Ops.SOURCE)
  dev=Device['NV']; host=(np.empty(slots*16384,np.float32),np.empty(slots,np.int32),desc,blocks,b,db)
  bufs=[dev.allocator._alloc(x.nbytes,BufferSpec()) for x in host]
  for buf,x in zip(bufs[2:],host[2:]): dev.allocator._copyin(buf,memoryview(x.tobytes()))
  prog=NVProgram(dev,'nv_generated_q6k_streamk_slots',NVRTCCompiler(dev.arch,ptx=False,cache_key='q6_streamk_slots_bench_q8_shared_v1').compile(src),shared_mem=58880)
  print({"registers":prog.regs_usage,"shared_bytes":prog.shmem_usage,"local_bytes":prog.lcmem_usage})
  for _ in range(1): prog(*bufs,global_size=(slots,1,1),local_size=(256,1,1),wait=True,timeout=120000)
  ts=[]
  for _ in range(samples):
    t=time.perf_counter(); prog(*bufs,global_size=(slots,1,1),local_size=(256,1,1),wait=True,timeout=120000); ts.append((time.perf_counter()-t)*1e6)
  print({"owners":slots,"active":sum(x[1]!=x[2] for x in rows),"warmup":1,"samples":samples,
         "min_us":min(ts),"median_us":statistics.median(ts),"shape":"M512 N4096 K12288"})

if __name__ == '__main__': run()
