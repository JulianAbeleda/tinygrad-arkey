#!/usr/bin/env python3
"""Replay the captured heterogeneous projection population through NV HCQ or CUDA graph.

Run in separate processes with ``--backend nv`` and ``--backend cuda-graph``.
The exact cubins, ABIs, buffer sizes, serial order, and prefix populations are
identical. Prefix-length regression removes each runtime's fixed drain/event
intercept; this is a front-end cadence discriminator, not a production wall.
"""
from __future__ import annotations
import argparse, ctypes, hashlib, json, pathlib, statistics, subprocess, sys, time

ROOT=pathlib.Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT))
CAP=ROOT/'docs/task_workflow/evidence/nv-active-body-ledger-20260827/tiny-projection-capture.json'
COUNTS={
 'q4k_g3_lanemap_gemv_vec_4096_4096':18, 'q4k_g3_lanemap_gemv_vec_1024_4096':10,
 'q6k_v_four_warp_fp16_direct_1024_4096':10, 'q4k_g3_lanemap_gemv_vec_epi_resadd_4096_4096':36,
 'q4k_gate_up_four_warp_vec_fp16_12288_4096':36, 'q6k_fp16_packed_lanemap_u4_4096_12288_epi_ffnresadd':18,
 'rmsnorm_q8_1_llama_provider_4096':18, 'q4k_warp_coop_q8_dp4a_direct_4096_4096':18,
 'q4k_q6k_warp_coop_q8_dp4a_pair_direct_1024_4096':8, 'q4k_warp_coop_q8_dp4a_pair_direct_1024_4096':10,
 'q4k_fp16_mmvq_direct_vec_4096_12288_epi_ffnresadd':18, 'q4k_g3_lanemap_gemv_pair_vec_1024_4096':8}
NS=(16,32,64,128,208)

def ols(xs,ys):
 mx,my=statistics.mean(xs),statistics.mean(ys); m=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/sum((x-mx)**2 for x in xs)
 return m,my-m*mx

def sequence(rows):
 remain=COUNTS.copy(); out=[]
 while any(remain.values()):
  for r in rows:
   if remain[r['name']]: out.append(r); remain[r['name']]-=1
 return out

def run_nv(rows, seq, warmup, reps):
 from tinygrad import Device
 from tinygrad.runtime.ops_nv import NVProgram
 from tinygrad.runtime.ops_nv import NVComputeQueue
 from tinygrad.device import BufferSpec
 dev=Device['NV']; state={}
 for r in rows:
  p=NVProgram(dev,r['name'],pathlib.Path(r['cubin_path']).read_bytes()); c=r['calls'][0]; bs=[]
  for b in c['buf_meta']:
   q=dev.allocator._alloc(b['size'],BufferSpec()); dev.allocator._copyin(q,memoryview(bytearray(b['size']))); bs.append(q)
  state[r['name']]=(p,tuple(bs),tuple(c['global_size']),tuple(c['local_size']))
 dev.synchronize(); result=[]
 for n in NS:
  samples=[]
  for rep in range(warmup+reps):
   q=NVComputeQueue(queue_idx=0); q.setup(compute_class=dev.iface.compute_class,local_mem_window=dev.local_mem_window,shared_mem_window=dev.shared_mem_window)
   q.wait(dev.timeline_signal,dev.timeline_value-1).memory_barrier()
   for r in seq[:n]:
    p,bs,g,b=state[r['name']]; q.exec(p,p.fill_kernargs(bs),g,b)
   target=dev.next_timeline(); q.signal(dev.timeline_signal,target); q.submit(dev); t=time.perf_counter_ns(); dev.synchronize(); samples.append((time.perf_counter_ns()-t)/1e3)
  result.append({'n':n,'samples_us':samples[warmup:],'median_us':statistics.median(samples[warmup:])})
 return result

def run_cuda(rows,seq,warmup,reps):
 from tinygrad.runtime.autogen import cuda
 from tinygrad.runtime.ops_cuda import check,encode_args
 check(cuda.cuInit(0)); dev=ctypes.c_int(); check(cuda.cuDeviceGet(ctypes.byref(dev),0)); ctx=cuda.CUcontext(); check(cuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx),dev)); check(cuda.cuCtxSetCurrent(ctx))
 stream=cuda.CUstream(); check(cuda.cuStreamCreate(ctypes.byref(stream),cuda.CU_STREAM_NON_BLOCKING)); state={}; keep=[]
 try:
  for r in rows:
   mod,fn=cuda.CUmodule(),cuda.CUfunction(); check(cuda.cuModuleLoadData(ctypes.byref(mod),pathlib.Path(r['cubin_path']).read_bytes())); check(cuda.cuModuleGetFunction(ctypes.byref(fn),mod,r['name'].encode())); keep.append(mod)
   ps=[]
   for b in r['calls'][0]['buf_meta']:
    p=cuda.CUdeviceptr(); check(cuda.cuMemAlloc_v2(ctypes.byref(p),b['size'])); check(cuda.cuMemsetD8_v2(p,0,b['size'])); ps.append(p); keep.append(p)
   ca,va=encode_args(ps,[]); keep.extend((ca,va)); state[r['name']]=(fn,tuple(r['calls'][0]['global_size']),tuple(r['calls'][0]['local_size']),va)
  result=[]
  for n in NS:
   graph,inst=cuda.CUgraph(),cuda.CUgraphExec(); check(cuda.cuGraphCreate(ctypes.byref(graph),0)); prior=None; kps=[]
   for r in seq[:n]:
    fn,g,b,va=state[r['name']]; node=cuda.CUgraphNode(); deps=None if prior is None else (cuda.CUgraphNode*1)(prior)
    kp=cuda.CUDA_KERNEL_NODE_PARAMS_v1(fn,*g,*b,0,ctypes.cast(0,ctypes.POINTER(ctypes.c_void_p)),va); check(cuda.cuGraphAddKernelNode(ctypes.byref(node),graph,deps,0 if prior is None else 1,ctypes.byref(kp))); kps.append(kp); prior=node
   check(cuda.cuGraphInstantiate_v2(ctypes.byref(inst),graph,None,None,0))
   for _ in range(warmup): check(cuda.cuGraphLaunch(inst,stream))
   check(cuda.cuStreamSynchronize(stream)); samples=[]
   for _ in range(reps):
    a,beg=cuda.CUevent(),cuda.CUevent(); check(cuda.cuEventCreate(ctypes.byref(a),0)); check(cuda.cuEventCreate(ctypes.byref(beg),0)); check(cuda.cuEventRecord(a,stream)); check(cuda.cuGraphLaunch(inst,stream)); check(cuda.cuEventRecord(beg,stream)); check(cuda.cuEventSynchronize(beg)); ms=ctypes.c_float(); check(cuda.cuEventElapsedTime(ctypes.byref(ms),a,beg)); samples.append(ms.value*1000); check(cuda.cuEventDestroy_v2(a)); check(cuda.cuEventDestroy_v2(beg))
   result.append({'n':n,'samples_us':samples,'median_us':statistics.median(samples)}); check(cuda.cuGraphExecDestroy(inst)); check(cuda.cuGraphDestroy(graph))
  return result
 finally:
  # Process exit releases allocations/modules; explicit mixed-type cleanup is intentionally avoided in measurement tooling.
  check(cuda.cuStreamDestroy_v2(stream)); check(cuda.cuDevicePrimaryCtxRelease(dev))

def main():
 ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--backend',choices=('nv','cuda-graph'),required=True); ap.add_argument('--warmup',type=int,default=5); ap.add_argument('--reps',type=int,default=11); ap.add_argument('--out',type=pathlib.Path,required=True); a=ap.parse_args()
 rows=json.loads(CAP.read_text())['captured']; seq=sequence(rows); vals=run_nv(rows,seq,a.warmup,a.reps) if a.backend=='nv' else run_cuda(rows,seq,a.warmup,a.reps); m,i=ols([r['n'] for r in vals],[r['median_us'] for r in vals])
 out={'schema':'tinygrad.nv_projection_frontend_bridge.v1','backend':a.backend,'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'population':len(seq),'ns':NS,'warmup':a.warmup,'reps':a.reps,'rows':vals,'slope_us_per_node':m,'intercept_us':i,'sequence':[r['name'] for r in seq]}; a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps({'backend':a.backend,'slope_us_per_node':m,'intercept_us':i,'medians':[(r['n'],r['median_us']) for r in vals]},indent=2))
if __name__=='__main__': main()
