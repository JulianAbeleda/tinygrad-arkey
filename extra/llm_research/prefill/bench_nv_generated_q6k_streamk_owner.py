"""Correctness qualification and benchmark for the 170-owner Q6_K Stream-K main."""
import argparse, json, statistics, time
import numpy as np
from tinygrad import Device, dtypes
from tinygrad.codegen import to_program
from tinygrad.device import BufferSpec
from tinygrad.helpers import Target
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.runtime.ops_nv import NVProgram
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
from tinygrad.uop.ops import Ops, UOp
from extra.llm_research.prefill.nv_generated_q6k_streamk import generated_q6k_streamk_owner_partials, owner_metadata
from extra.llm_research.prefill.nv_generated_q6k_streamk_slots import q6_streamk_slot_kernel

M, N, K = 512, 4096, 12288
K_BLOCKS, OWNERS, SLOTS, TILE_VALUES = 48, 170, 340, 128*128
LLAMA_MAIN_US, LLAMA_TOTAL_US = 201.216, 209.856

def _source(ast):
  return next(x.arg for x in to_program(ast,CUDARenderer(Target.parse('NV:CUDA:sm_120'))).src if x.op is Ops.SOURCE)

def run(samples=9):
  descriptors=np.array([(s.tile_id,s.begin,s.end) for s in owner_metadata()],dtype=np.int32).reshape(-1)
  rng=np.random.default_rng(20260908)
  blocks=rng.integers(0,256,(N,K_BLOCKS,210),dtype=np.uint8).view(np.uint16).reshape(-1)
  blocks.reshape(N,K_BLOCKS,105).view(np.uint8).reshape(N,K_BLOCKS,210)[:,:,208:210]=np.frombuffer(np.float16(.03125).tobytes(),np.uint8)
  q8=rng.integers(-4,5,(K, M),dtype=np.int8)
  q8_scales=np.full((K_BLOCKS,8,M),.0625,np.float32)
  ph=lambda n,dt,i: UOp.placeholder((n,),dt,i)
  owner_ast=generated_q6k_streamk_owner_partials(ph(SLOTS*TILE_VALUES,dtypes.float32,0),ph(SLOTS,dtypes.int32,1),
    ph(blocks.size,dtypes.uint16,2),ph(q8.size,dtypes.int8,3),ph(q8_scales.size,dtypes.float32,4))
  slot_ast=q6_streamk_slot_kernel(ph(SLOTS*TILE_VALUES,dtypes.float32,0),ph(SLOTS,dtypes.int32,1),
    ph(SLOTS*3,dtypes.int32,2),ph(blocks.size,dtypes.uint16,3),ph(q8.size,dtypes.int8,4),
    ph(q8_scales.size,dtypes.float32,5),K_BLOCKS,SLOTS,37)
  owner_src,slot_src=_source(owner_ast),_source(slot_ast)
  dev=Device['NV']; alloc=lambda x: dev.allocator._alloc(x.nbytes,BufferSpec())
  owner_host=(np.zeros(SLOTS*TILE_VALUES,np.float32),np.full(SLOTS,-2,np.int32))
  slot_host=(np.zeros(SLOTS*TILE_VALUES,np.float32),np.full(SLOTS,-2,np.int32))
  owner_bufs=[alloc(x) for x in owner_host]; slot_bufs=[alloc(x) for x in slot_host]
  desc_buf,blocks_buf,q8_buf,scales_buf=map(alloc,(descriptors,blocks,q8,q8_scales))
  for buf,x in zip((*owner_bufs,*slot_bufs,desc_buf,blocks_buf,q8_buf,scales_buf),
                   (*owner_host,*slot_host,descriptors,blocks,q8,q8_scales)):
    dev.allocator._copyin(buf,memoryview(x.tobytes()))
  owner=NVProgram(dev,'nv_generated_q6k_streamk_owner_partials',
    NVRTCCompiler(dev.arch,ptx=False,cache_key='q6_streamk_owner_qualification_v1').compile(owner_src),shared_mem=58880)
  slot=NVProgram(dev,'nv_generated_q6k_streamk_slots',
    NVRTCCompiler(dev.arch,ptx=False,cache_key='q6_streamk_slots_qualification_v1').compile(slot_src),shared_mem=58880)
  slot(*slot_bufs,desc_buf,blocks_buf,q8_buf,scales_buf,global_size=(SLOTS,1,1),local_size=(256,1,1),wait=True,timeout=120000)
  owner(*owner_bufs,blocks_buf,q8_buf,scales_buf,global_size=(OWNERS,1,1),local_size=(256,1,1),wait=True,timeout=120000)
  copied=[]
  for buf,dt,count in ((*owner_bufs[0:1],np.float32,SLOTS*TILE_VALUES),(*owner_bufs[1:2],np.int32,SLOTS),
                       (*slot_bufs[0:1],np.float32,SLOTS*TILE_VALUES),(*slot_bufs[1:2],np.int32,SLOTS)):
    raw=memoryview(bytearray(buf.size)); dev.allocator._copyout(raw,buf); copied.append(np.frombuffer(raw,dt,count=count).copy())
  owner_values,owner_ids,slot_values,slot_ids=copied
  slot_row_major=slot_values.reshape(SLOTS,128,128).transpose(0,2,1).reshape(-1)
  values_exact=bool(np.array_equal(owner_values,slot_row_major)); ids_exact=bool(np.array_equal(owner_ids,slot_ids))
  max_abs=float(np.max(np.abs(owner_values-slot_row_major)))
  if not (values_exact and ids_exact):
    finite=np.isfinite(owner_values)&np.isfinite(slot_row_major)
    delta=np.abs(owner_values-slot_row_major)
    bad=np.flatnonzero(~finite | (delta != 0))
    raise AssertionError(f"owner mismatch: values_exact={values_exact} ids_exact={ids_exact} max_abs={max_abs} "
      f"owner_nan={int((~np.isfinite(owner_values)).sum())} slot_nan={int((~np.isfinite(slot_row_major)).sum())} "
      f"finite_max={float(np.max(delta[finite])) if finite.any() else None} first_bad={int(bad[0]) if bad.size else None} "
      f"owner_first={owner_values[bad[0]] if bad.size else None} slot_first={slot_row_major[bad[0]] if bad.size else None}")
  timings=[]
  for _ in range(samples):
    start=time.perf_counter()
    owner(*owner_bufs,blocks_buf,q8_buf,scales_buf,global_size=(OWNERS,1,1),local_size=(256,1,1),wait=True,timeout=120000)
    timings.append((time.perf_counter()-start)*1e6)
  minimum=min(timings); median=statistics.median(timings)
  return {"schema":"tinygrad.nv_generated_q6k_streamk_owner_qualification.v1","status":"PASS" if minimum<=LLAMA_MAIN_US*1.05 else "FAIL_PERFORMANCE",
    "shape":{"M":M,"N":N,"K":K,"weight":"Q6_K","owners":OWNERS},
    "correctness":{"values_compared":int(owner_values.size),"values_exact":values_exact,"tile_ids_exact":ids_exact,"max_abs":max_abs},
    "timing_us":{"samples":timings,"min":minimum,"median":median,"llama_main":LLAMA_MAIN_US,"llama_main_5pct_gate":LLAMA_MAIN_US*1.05,
      "llama_pair_total":LLAMA_TOTAL_US},
    "resources":{"registers":owner.regs_usage,"shared_bytes":owner.shmem_usage,"local_bytes":owner.lcmem_usage}}

if __name__ == '__main__':
  parser=argparse.ArgumentParser(); parser.add_argument('--samples',type=int,default=9)
  print(json.dumps(run(parser.parse_args().samples),indent=2))
