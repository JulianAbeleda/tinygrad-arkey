#!/usr/bin/env python3
"""Adversarial producer/consumer visibility gate for dependent-QMD membars."""
from __future__ import annotations
import argparse, json, pathlib, statistics, subprocess, sys, time

ROOT=pathlib.Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT))

SRC=r'''
extern "C" __global__ void membar_producer(unsigned int *data, unsigned int value) {
  unsigned int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<8192) data[i]=value^i;
}
extern "C" __global__ void membar_consumer(const unsigned int *data, unsigned int *errors, unsigned int value) {
  unsigned int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<8192 && data[i]!=(value^i)) atomicAdd(errors,1u);
}
'''

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--membar',choices=('system','internal-none'),required=True); ap.add_argument('--pairs',type=int,default=256); ap.add_argument('--warmup',type=int,default=5); ap.add_argument('--reps',type=int,default=15); ap.add_argument('--out',type=pathlib.Path,required=True); a=ap.parse_args()
 from tinygrad import Device
 from tinygrad.device import BufferSpec
 from tinygrad.runtime.ops_nv import NVComputeQueue,NVProgram
 from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
 dev=Device['NV']; cubin=NVRTCCompiler(dev.arch,ptx=False,cache_key='nv_qmd_membar_semantic_v1').compile(SRC)
 prod=NVProgram(dev,'membar_producer',cubin); cons=NVProgram(dev,'membar_consumer',cubin)
 data=dev.allocator._alloc(8192*4,BufferSpec()); errors=dev.allocator._alloc(4,BufferSpec()); dev.allocator._copyin(data,memoryview(bytes(8192*4))); dev.allocator._copyin(errors,memoryview(bytes(4))); dev.synchronize()
 q=NVComputeQueue(queue_idx=0); q.setup(compute_class=dev.iface.compute_class,local_mem_window=dev.local_mem_window,shared_mem_window=dev.shared_mem_window); q.wait(dev.timeline_signal,dev.timeline_value-1).memory_barrier()
 for pair in range(a.pairs):
  value=(pair*0x9e3779b9+0x13579bdf)&0xffffffff
  prev=q.active_qmd; q.exec(prod,prod.fill_kernargs((data,),(value,)),(32,1,1),(256,1,1));
  if a.membar=='internal-none' and prev is not None: prev.write(cwd_membar_type=0)
  prev=q.active_qmd; q.exec(cons,cons.fill_kernargs((data,errors),(value,)),(32,1,1),(256,1,1));
  if a.membar=='internal-none': prev.write(cwd_membar_type=0)
 done=dev.new_signal(); q.signal(done,1); q.bind(dev); samples=[]
 for rep in range(a.warmup+a.reps): done.value=0; t=time.perf_counter_ns(); q.submit(dev); done.wait(1); samples.append((time.perf_counter_ns()-t)/1e3)
 host=bytearray(4); dev.allocator._copyout(memoryview(host),errors); error_count=int.from_bytes(host,'little')
 out={'schema':'tinygrad.nv_qmd_membar_semantic.v1','commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'membar':a.membar,'pairs':a.pairs,'transitions':a.pairs*2-1,'warmup':a.warmup,'reps':a.reps,'samples_us':samples[a.warmup:],'median_us':statistics.median(samples[a.warmup:]),'error_count':error_count,'pass':error_count==0}
 a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
