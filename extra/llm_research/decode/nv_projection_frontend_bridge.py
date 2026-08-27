#!/usr/bin/env python3
"""Replay the captured heterogeneous projection population through NV HCQ or CUDA graph.

Run in separate processes with ``--backend nv`` and ``--backend cuda-graph``.
The exact cubins, ABIs, buffer sizes, serial order, and prefix populations are
identical. Prefix-length regression removes each runtime's fixed drain/event
intercept; this is a front-end cadence discriminator, not a production wall.
"""
from __future__ import annotations
import argparse, ctypes, hashlib, json, os, pathlib, statistics, subprocess, sys, time

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

def trace_snapshot(label):
 if not os.getenv('NV_IOCTL_TRACE_SNAPSHOT'): return
 fn=ctypes.CDLL(None).nv_ioctl_trace_snapshot; fn.argtypes=(ctypes.c_char_p,); fn(label.encode())

def ols(xs,ys):
 mx,my=statistics.mean(xs),statistics.mean(ys); m=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/sum((x-mx)**2 for x in xs)
 return m,my-m*mx

def sequence(rows):
 remain=COUNTS.copy(); out=[]
 while any(remain.values()):
  for r in rows:
   if remain[r['name']]: out.append(r); remain[r['name']]-=1
 return out

def run_nv(rows, seq, warmup, reps, prologue, completion, chain, replay, prefetch):
 from tinygrad import Device
 from tinygrad.runtime.ops_nv import NVProgram
 from tinygrad.runtime.ops_nv import NVComputeQueue
 from tinygrad.device import BufferSpec
 dev=Device['NV']; state={}
 for r in rows:
  p=NVProgram(dev,r['name'],pathlib.Path(r['cubin_path']).read_bytes()); c=r['calls'][0]; bs=[]
  for bi,b in enumerate(c['buf_meta']):
   q=dev.allocator._alloc(b['size'],BufferSpec()); seed=(rows.index(r)*17+bi*29+1)&255; dev.allocator._copyin(q,memoryview(bytes([seed])*b['size'])); bs.append(q)
  state[r['name']]=(p,tuple(bs),tuple(b['size'] for b in c['buf_meta']),tuple(c['global_size']),tuple(c['local_size']),tuple(c['vals']))
 dev.synchronize(); result=[]
 def build_queue(n, done=None):
  q=NVComputeQueue(queue_idx=0); q.setup(compute_class=dev.iface.compute_class,local_mem_window=dev.local_mem_window,shared_mem_window=dev.shared_mem_window)
  if prologue in ('full','wait'): q.wait(dev.timeline_signal,dev.timeline_value-1)
  if prologue in ('full','barrier'): q.memory_barrier()
  for r in seq[:n]:
   p,bs,_sizes,g,b,vals=state[r['name']]; prev=q.active_qmd; q.exec(p,p.fill_kernargs(bs,vals),g,b)
   if prev is not None: prev.write(dependent_qmd0_prefetch=prefetch)
   if chain=='pcas': q.active_qmd=None
  if completion=='pushbuffer': q.active_qmd=None
  q.signal(done or dev.timeline_signal,1 if done is not None else dev.next_timeline())
  return q
 for n in NS:
  samples=[]
  if replay=='bound':
   done=dev.new_signal(); q=build_queue(n,done); q.bind(dev)
   for rep in range(warmup+reps): done.value=0; t=time.perf_counter_ns(); q.submit(dev); done.wait(1); samples.append((time.perf_counter_ns()-t)/1e3)
  else:
   for rep in range(warmup+reps):
    q=build_queue(n); t=time.perf_counter_ns(); q.submit(dev); dev.synchronize(); samples.append((time.perf_counter_ns()-t)/1e3)
  result.append({'n':n,'samples_us':samples[warmup:],'median_us':statistics.median(samples[warmup:])})
 hashes={}
 for name,(_p,bs,sizes,_g,_b,_vals) in state.items():
  hashes[name]=[]
  for q,size in zip(bs,sizes):
   host=bytearray(size); dev.allocator._copyout(memoryview(host),q); hashes[name].append(hashlib.sha256(host).hexdigest())
 return result,hashes

def run_cuda(rows,seq,warmup,reps,upload):
 from tinygrad.runtime.autogen import cuda
 from tinygrad.runtime.ops_cuda import check,encode_args
 check(cuda.cuInit(0)); dev=ctypes.c_int(); check(cuda.cuDeviceGet(ctypes.byref(dev),0)); ctx=cuda.CUcontext(); check(cuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx),dev)); check(cuda.cuCtxSetCurrent(ctx))
 stream=cuda.CUstream(); check(cuda.cuStreamCreate(ctypes.byref(stream),cuda.CU_STREAM_NON_BLOCKING)); state={}; keep=[]
 try:
  for r in rows:
   mod,fn=cuda.CUmodule(),cuda.CUfunction(); check(cuda.cuModuleLoadData(ctypes.byref(mod),pathlib.Path(r['cubin_path']).read_bytes())); check(cuda.cuModuleGetFunction(ctypes.byref(fn),mod,r['name'].encode())); keep.append(mod)
   ps=[]
   sizes=[]
   for bi,b in enumerate(r['calls'][0]['buf_meta']):
    p=cuda.CUdeviceptr(); check(cuda.cuMemAlloc_v2(ctypes.byref(p),b['size'])); seed=(rows.index(r)*17+bi*29+1)&255; check(cuda.cuMemsetD8_v2(p,seed,b['size'])); ps.append(p); sizes.append(b['size']); keep.append(p)
   vals=tuple(r['calls'][0]['vals']); ca,va=encode_args(ps,vals); keep.extend((ca,va)); state[r['name']]=(fn,tuple(r['calls'][0]['global_size']),tuple(r['calls'][0]['local_size']),va,vals,ps,sizes)
  result=[]
  for n in NS:
   graph,inst=cuda.CUgraph(),cuda.CUgraphExec(); check(cuda.cuGraphCreate(ctypes.byref(graph),0)); prior=None; kps=[]
   for r in seq[:n]:
    fn,g,b,va,_vals,_ps,_sizes=state[r['name']]; node=cuda.CUgraphNode(); deps=None if prior is None else (cuda.CUgraphNode*1)(prior)
    kp=cuda.CUDA_KERNEL_NODE_PARAMS_v1(fn,*g,*b,0,ctypes.cast(0,ctypes.POINTER(ctypes.c_void_p)),va); check(cuda.cuGraphAddKernelNode(ctypes.byref(node),graph,deps,0 if prior is None else 1,ctypes.byref(kp))); kps.append(kp); prior=node
   check(cuda.cuGraphInstantiate_v2(ctypes.byref(inst),graph,None,None,0))
   if upload: check(cuda.cuGraphUpload(inst,stream)); check(cuda.cuStreamSynchronize(stream))
   for _ in range(warmup): check(cuda.cuGraphLaunch(inst,stream))
   check(cuda.cuStreamSynchronize(stream)); trace_snapshot(f'cuda-n{n}-warm'); samples=[]
   for _ in range(reps):
    a,beg=cuda.CUevent(),cuda.CUevent(); check(cuda.cuEventCreate(ctypes.byref(a),0)); check(cuda.cuEventCreate(ctypes.byref(beg),0)); check(cuda.cuEventRecord(a,stream)); check(cuda.cuGraphLaunch(inst,stream)); check(cuda.cuEventRecord(beg,stream)); check(cuda.cuEventSynchronize(beg)); trace_snapshot(f'cuda-n{n}-rep'); ms=ctypes.c_float(); check(cuda.cuEventElapsedTime(ctypes.byref(ms),a,beg)); samples.append(ms.value*1000); check(cuda.cuEventDestroy_v2(a)); check(cuda.cuEventDestroy_v2(beg))
   result.append({'n':n,'samples_us':samples,'median_us':statistics.median(samples)}); check(cuda.cuGraphExecDestroy(inst)); check(cuda.cuGraphDestroy(graph))
  hashes={}
  for name,(_fn,_g,_b,_va,_vals,ps,sizes) in state.items():
   hashes[name]=[]
   for p,size in zip(ps,sizes):
    host=bytearray(size); check(cuda.cuMemcpyDtoH_v2(ctypes.addressof(ctypes.c_ubyte.from_buffer(host)),p,size)); hashes[name].append(hashlib.sha256(host).hexdigest())
  return result,hashes
 finally:
  # Process exit releases allocations/modules; explicit mixed-type cleanup is intentionally avoided in measurement tooling.
  check(cuda.cuStreamDestroy_v2(stream)); check(cuda.cuDevicePrimaryCtxRelease(dev))

def main():
 ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--backend',choices=('nv','cuda-graph'),required=True); ap.add_argument('--warmup',type=int,default=5); ap.add_argument('--reps',type=int,default=11); ap.add_argument('--out',type=pathlib.Path,required=True)
 ap.add_argument('--nv-prologue',choices=('full','wait','barrier','none'),default='full'); ap.add_argument('--nv-completion',choices=('qmd','pushbuffer'),default='qmd'); ap.add_argument('--nv-chain',choices=('dependent','pcas'),default='dependent'); ap.add_argument('--nv-replay',choices=('fresh','bound'),default='fresh'); ap.add_argument('--nv-prefetch',type=int,choices=(0,1),default=1); ap.add_argument('--cuda-upload',type=int,choices=(0,1),default=0); a=ap.parse_args()
 rows=json.loads(CAP.read_text())['captured']; assert {r['name'] for r in rows} == set(COUNTS)
 for r in rows:
  signatures={(c['n_bufs'],tuple(b['size'] for b in c['buf_meta']),tuple(c['global_size']),tuple(c['local_size']),tuple(c['vals'])) for c in r['calls']}
  assert len(signatures)==1, f"non-constant captured ABI for {r['name']}: {signatures}"
 seq=sequence(rows); vals,hashes=run_nv(rows,seq,a.warmup,a.reps,a.nv_prologue,a.nv_completion,a.nv_chain,a.nv_replay,a.nv_prefetch) if a.backend=='nv' else run_cuda(rows,seq,a.warmup,a.reps,a.cuda_upload); m,i=ols([r['n'] for r in vals],[r['median_us'] for r in vals])
 out={'schema':'tinygrad.nv_projection_frontend_bridge.v1','backend':a.backend,'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'population':len(seq),'ns':NS,'warmup':a.warmup,'reps':a.reps,'rows':vals,'slope_us_per_node':m,'intercept_us':i,'sequence':[r['name'] for r in seq],'final_buffer_sha256':hashes,'nv_options':{'prologue':a.nv_prologue,'completion':a.nv_completion,'chain':a.nv_chain,'replay':a.nv_replay,'prefetch':a.nv_prefetch},'cuda_options':{'upload':a.cuda_upload}}; a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps({'backend':a.backend,'slope_us_per_node':m,'intercept_us':i,'medians':[(r['n'],r['median_us']) for r in vals],'buffer_hashes':sum(map(len,hashes.values())),'nv_options':out['nv_options'],'cuda_options':out['cuda_options']},indent=2))
if __name__=='__main__': main()
