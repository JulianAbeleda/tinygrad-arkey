#!/usr/bin/env python3
"""Region C: deterministic int32 MMA tiles plus FP32 Q6 scale/accumulate."""
import argparse, json, pathlib, re, subprocess
import numpy as np
from tinygrad import Device, dtypes
from tinygrad.codegen import to_program
from tinygrad.device import BufferSpec
from tinygrad.helpers import Target
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.runtime.ops_nv import NVProgram
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
from tinygrad.uop.ops import KernelInfo, Ops, UOp

ROOT = pathlib.Path(__file__).resolve().parents[3]

def region_c_kernel(out, dots, scales, d, dB):
  lane = UOp.special(32, "lidx0"); bid = UOp.special(4096, "gidx0")
  lr, lc = lane >> 2, lane & 3
  writes = []
  for r in range(4):
    idx = (lr + 8*(r >> 1))*8 + 2*lc + (r & 1)
    col = 2*lc + (r & 1)
    acc = None
    for p in range(8):
      z0 = dots[bid*2048 + (2*p)*128 + idx].cast(dtypes.float32)
      z1 = dots[bid*2048 + (2*p+1)*128 + idx].cast(dtypes.float32)
      term = (scales[p*32 + col].cast(dtypes.float32)*z0 +
              scales[p*32 + 8 + col].cast(dtypes.float32)*z1) * \
             (d[p].cast(dtypes.float32) * dB[p*8 + col].cast(dtypes.float32))
      acc = term if acc is None else acc + term
    writes.append(out[bid*128 + idx].store(acc))
  return UOp.sink(*writes, arg=KernelInfo(name="nv_q6_region_c_scale_accum", opts_to_apply=()))

def _alloc(dev, n): return dev.allocator._alloc(n, BufferSpec())
def _copyin(dev, buf, value): dev.allocator._copyin(buf, memoryview(np.ascontiguousarray(value).tobytes()))
def _copyout(dev, buf, dtype, shape):
  host = memoryview(bytearray(buf.size)); dev.allocator._copyout(host, buf)
  return np.frombuffer(host, dtype=dtype, count=int(np.prod(shape))).reshape(shape).copy()
def _sass(path):
  tool = ROOT/'.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/nvdisasm'
  text = subprocess.check_output([str(tool), '-c', str(path)], text=True)
  def count(op): return len(re.findall(rf'\b{op}(?:\.|\s)', text))
  return {k: count(k.upper()) for k in ('ffma','fadd','fmul','i2fp','stg','ldg','ldl','stl','bar','imma')}

def main():
  ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--artifacts', required=True)
  ap.add_argument('--replicas', type=int, default=4096); ap.add_argument('--rounds', type=int, default=15); args = ap.parse_args()
  if args.replicas < 1 or args.rounds < 5: raise ValueError('invalid replicas/rounds')
  art = pathlib.Path(args.artifacts); art.mkdir(parents=True, exist_ok=True)
  ph = lambda n, dt, i: UOp.placeholder((n,), dt, i)
  ast = region_c_kernel(ph(args.replicas*128, dtypes.float32, 0), ph(args.replicas*2048, dtypes.int32, 1),
                        ph(8*32, dtypes.float32, 2), ph(8, dtypes.float32, 3), ph(8*8, dtypes.float32, 4))
  pa = to_program(ast, CUDARenderer(Target.parse('NV:CUDA:sm_120'))); source = next(x.arg for x in pa.src if x.op is Ops.SOURCE)
  dev = Device['NV']; cubin = NVRTCCompiler(dev.arch, ptx=False, cache_key='q6_region_c_v1').compile(source)
  (art/'region_c.cu').write_text(source); (art/'region_c.cubin').write_bytes(cubin)
  counts = _sass(art/'region_c.cubin')
  rng = np.random.default_rng(20260831)
  dots = rng.integers(-8192, 8193, args.replicas*2048, dtype=np.int32)
  dots = dots.reshape(args.replicas, 2048)
  scales = rng.uniform(-2, 2, 8*32).astype(np.float32); d = rng.uniform(.1, 2, 8).astype(np.float32); dB = rng.uniform(.1, 2, 64).astype(np.float32)
  bufs = [_alloc(dev, x.nbytes) for x in (dots, scales, d, dB)]
  for b, x in zip(bufs, (dots, scales, d, dB)): _copyin(dev, b, x)
  out = _alloc(dev, args.replicas*128*4); program = NVProgram(dev, pa.arg.name, cubin)
  program(out, *bufs, global_size=(args.replicas,1,1), local_size=(32,1,1), wait=True, timeout=10)
  got = _copyout(dev, out, np.float32, (args.replicas,128))
  ref = np.zeros((args.replicas,128), np.float32)
  # Reconstruct the exact lane-owned 16x8 output indexing used by the kernel.
  for row in range(16):
    for col in range(8):
      idx = row*8+col
      for p in range(8): ref[:,idx] += (scales[p*32+col]*dots[:,2*p*128+idx] + scales[p*32+8+col]*dots[:,(2*p+1)*128+idx]) * d[p]*dB[p*8+col]
  max_abs = float(np.max(np.abs(got-ref))); max_rel = float(np.max(np.abs(got-ref)/(np.abs(ref)+1e-6)))
  exact = bool(np.allclose(got, ref, rtol=2e-5, atol=2e-2))
  samples = [program(out, *bufs, global_size=(args.replicas,1,1), local_size=(32,1,1), wait=True, timeout=10)*1e6 for _ in range(args.rounds)]
  kept=samples[3:]; result={'schema':'nv.q6.region_c_scale_accum.v1','passed':exact,'exact_fp32':exact,'max_abs':max_abs,'max_rel':max_rel,'replicas':args.replicas,'timing_us':{'min':float(min(kept)),'median':float(np.median(kept)),'samples':samples},'sass':counts,'resources':{'registers':program.regs_usage,'shared_bytes':program.shmem_usage,'local_bytes':program.lcmem_usage},'artifacts':str(art)}
  pathlib.Path(args.out).write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,sort_keys=True))
  if not exact: raise SystemExit(1)
if __name__ == '__main__': main()
