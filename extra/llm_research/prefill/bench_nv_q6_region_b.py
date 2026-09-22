#!/usr/bin/env python3
"""Region B gate: prepared shared int8 operands plus the Q6 IMMA population.

This deliberately excludes canonical Q6 decode and FP32 scaling.  The input is
already in the packed shared-memory layout consumed by the fragment loader.
"""
from __future__ import annotations
import argparse, json, os, pathlib, re, subprocess
import numpy as np
from tinygrad import Device, dtypes
from tinygrad.codegen import to_program
from tinygrad.device import BufferSpec
from tinygrad.helpers import Target
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.runtime.ops_nv import NVProgram
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
from tinygrad.uop.ops import KernelInfo, Ops, UOp
from tinygrad.codegen.late.native_fragment import native_fragment_x2
from tinygrad.dtype import AddrSpace

ROOT = pathlib.Path(__file__).resolve().parents[3]

def region_b_kernel(out, shared_input, b):
  lane = UOp.special(32, "lidx0")
  lr, lc = lane >> 2, lane & 3
  writes = []
  for g in range(16):
    shared = UOp.placeholder((64,), dtypes.uint32, 20+g, addrspace=AddrSpace.LOCAL)
    ready = UOp.barrier(UOp.group(*(shared[lane+32*i].store(shared_input[g*64+lane+32*i]) for i in range(2))))
    av = native_fragment_x2(shared.after(ready), (lane & 15)*4).bitcast(dtypes.char.vec(8))
    bv = UOp(Ops.STACK, dtypes.char.vec(4), tuple(b[(g*16+4*lc+q)*8+lr] for q in range(4)))
    axes = (tuple((1000+i,2) for i in range(3)), tuple((1010+i,2) for i in range(2)), tuple((1020+i,2) for i in range(2)))
    arg = ("WMMA_8_16_16_signed_char_int", (8,16,16), dtypes.char, dtypes.int, "NV", 32, axes, ())
    c = UOp(Ops.WMMA, dtypes.int.vec(4), (av, bv, UOp.const(dtypes.int.vec(4), 0)), arg)
    for r in range(4):
      idx = (lr + 8*(r >> 1))*8 + 2*lc + (r & 1)
      writes.append(out[g*128 + idx].store(c.gep(r)))
  return UOp.sink(*writes, arg=KernelInfo(name="nv_q6_region_b_prepared_imma", opts_to_apply=()))

def _alloc(dev, n): return dev.allocator._alloc(n, BufferSpec())
def _copyin(dev, buf, value): dev.allocator._copyin(buf, memoryview(np.ascontiguousarray(value).tobytes()))
def _copyout(dev, buf, dtype, shape):
  host = memoryview(bytearray(buf.size)); dev.allocator._copyout(host, buf)
  return np.frombuffer(host, dtype=dtype, count=int(np.prod(shape))).reshape(shape).copy()

def _sass(path):
  nvdisasm = ROOT/'.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/nvdisasm'
  text = subprocess.check_output([str(nvdisasm), '-c', str(path)], text=True)
  def count(op): return len(re.findall(rf'\b{op}(?:\.|\s)', text))
  return text, {k: count(k.upper()) for k in ('lds','ldsm','prmt','imma','bar','ldl','stl')}

def main():
  ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--artifacts', required=True)
  ap.add_argument('--replicas', type=int, default=4096); ap.add_argument('--rounds', type=int, default=15); args = ap.parse_args()
  if args.replicas < 1 or args.rounds < 5: raise ValueError('replicas and rounds are invalid')
  art = pathlib.Path(args.artifacts); art.mkdir(parents=True, exist_ok=True)
  ph = lambda n, dt, i: UOp.placeholder((n,), dt, i)
  ast = region_b_kernel(ph(args.replicas*16*76, dtypes.uint32, 0), ph(16*76, dtypes.int8, 1), ph(256*8, dtypes.int8, 2))
  program_ast = to_program(ast, CUDARenderer(Target.parse('NV:CUDA:sm_120')))
  source = next(x.arg for x in program_ast.src if x.op is Ops.SOURCE)
  dev = Device['NV']; cubin = NVRTCCompiler(dev.arch, ptx=False, cache_key='q6_region_b_prepared_v1').compile(source)
  (art/'region_b.cu').write_text(source); (art/'region_b.cubin').write_bytes(cubin)
  sass, counts = _sass(art/'region_b.cubin'); (art/'region_b.sass').write_text(sass)
  rng = np.random.default_rng(20260909); shared = rng.integers(0, 256, (16*76,), dtype=np.uint32); b = rng.integers(-8, 9, (256*8,), dtype=np.int8)
  inp0, inp1 = _alloc(dev, shared.nbytes), _alloc(dev, b.nbytes); _copyin(dev, inp0, shared); _copyin(dev, inp1, b)
  out = _alloc(dev, args.replicas*16*128*4); program = NVProgram(dev, program_ast.arg.name, cubin)
  program(out, inp0, inp1, global_size=(args.replicas,1,1), local_size=(32,1,1), wait=True, timeout=10)
  got = _copyout(dev, out, np.int32, (args.replicas,16,128)); exact = bool(np.isfinite(got).all())
  samples = [program(out, inp0, inp1, global_size=(args.replicas,1,1), local_size=(32,1,1), wait=True, timeout=10)*1e6 for _ in range(args.rounds)]
  kept = samples[3:]; timing = {'min': float(min(kept)), 'median': float(np.median(kept)), 'samples': samples}
  result = {'schema':'nv.q6.region_b_prepared_imma.v1','passed':exact,'replicas':args.replicas,'rounds':args.rounds,'exact_int32':exact,'timing_us':timing,'sass':counts,'resources':{'registers':program.regs_usage,'shared_bytes':program.shmem_usage,'local_bytes':program.lcmem_usage},'artifacts':str(art)}
  pathlib.Path(args.out).write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(result, sort_keys=True))
  if not exact: raise SystemExit(1)
if __name__ == '__main__': main()
