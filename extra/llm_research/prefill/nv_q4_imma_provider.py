"""Fail-closed research provider for the qualified pp512 Q4_K x Q8_1 IMMA projection."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from tinygrad.runtime.ops_nv import NVProgram
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
from extra.llm_research.prefill.nv_q4k_imma_fragment_microgate import SRC, lexical_src, production_slotmap

M, N, K = 512, 12288, 4096
MAIN_GRID, FIXUP_GRID, BLOCK = (170, 1, 1), (384, 1, 1), (256, 1, 1)
DYNAMIC_SHARED_BYTES, NV_RUNTIME_SHARED_BYTES = 57856, 1024
PARTIAL_SLOTS = 340


def main_source() -> str:
  src = lexical_src(SRC, True)
  src = src.replace("int ntx=(N+127)/128,mb=(tile/ntx)*128,nb=(tile%ntx)*128,blocks=K/256;",
                    "int ntx=96,mb=(tile/96)*128,nb=(tile%96)*128,blocks=16;")
  src = src.replace("row*K+blk*256", "row*4096+blk*256")
  src = src.replace("total=(M/128)*(N/128)*(K/256),owners=min(170,total)", "total=6144,owners=170").replace("K/256", "16")
  src = src.replace("if(col<N) { unsigned raw=", "{ unsigned raw=").replace("w0=col<N?words[base]:0;", "w0=words[base];")
  src = src.replace("if(col<N) {", "{").replace("if(row<M) v=", "v=").replace("if(row<M) { int xm=", "{ int xm=")
  src = src.replace("if(row<M&&col<N) {", "{").replace("row*N+col", "row*12288+col")
  assert src.count('extern "C" __global__') == 1
  return src


def fixup_source() -> str:
  return r'''extern "C" __global__ void q4k_imma_fixup(float *o,const float*p,const int*map,int M,int N){
    int t=blockIdx.x,s0=map[2*t];if(s0<0)return;int s1=map[2*t+1],nb=(t%(N/128))*128,mb=(t/(N/128))*128;
    for(int z=threadIdx.x;z<16384;z+=256){int r=z/128,c=z%128;
      o[(mb+r)*N+nb+c]=p[s0*16384+z]+(s1>=0?p[s1*16384+z]:0);}}
  '''


@dataclass(frozen=True)
class Provider:
  main: NVProgram
  fixup: NVProgram
  slotmap: np.ndarray

  def validate_buffers(self, out, partials, ids, words, q8, scales, sums, map_buffer) -> None:
    expected = (M*N*4, PARTIAL_SLOTS*128*128*4, PARTIAL_SLOTS*4, N*(K//256)*36*4,
                M*K, M*(K//32)*4, M*(K//32)*4, 384*2*4)
    actual = tuple(x.size for x in (out, partials, ids, words, q8, scales, sums, map_buffer))
    if any(got < need for got,need in zip(actual, expected)):
      raise ValueError(f"Q4 IMMA provider ABI mismatch: {actual=} expected_minimum={expected}")

  def launch(self, out, partials, ids, words, q8, scales, sums, map_buffer, *, wait=False):
    self.validate_buffers(out, partials, ids, words, q8, scales, sums, map_buffer)
    main_t = self.main(out, partials, ids, words, q8, scales, sums, vals=(M,N,K),
                       global_size=MAIN_GRID, local_size=BLOCK, wait=wait)
    fix_t = self.fixup(out, partials, map_buffer, vals=(M,N), global_size=FIXUP_GRID, local_size=BLOCK, wait=wait)
    return main_t, fix_t


def compile_provider(dev) -> Provider:
  # Separate cubins are mandatory: combined entry-point ELF modules can corrupt
  # native dynamic-shared QMD state on sm_120.
  # v4 fully unrolls the two K128 panels and four K32 groups: 256 static
  # IMMA / 32 LDSM sites, full-real-qualified with no local spill traffic.
  main_lib = NVRTCCompiler(dev.arch, ptx=False, cache_key="q4_imma_provider_main_v4").compile(main_source())
  fix_lib = NVRTCCompiler(dev.arch, ptx=False, cache_key="q4_imma_provider_fix_v1").compile(fixup_source())
  main = NVProgram(dev, "q4k_imma_stream", main_lib, shared_mem=DYNAMIC_SHARED_BYTES+NV_RUNTIME_SHARED_BYTES)
  fixup = NVProgram(dev, "q4k_imma_fixup", fix_lib)
  return Provider(main, fixup, production_slotmap())


def provider_programs(provider:Provider):
  """Return finalized, TinyJit-capturable PROGRAM UOps for the fixed ABI."""
  from extra.llm_research.prefill.nv_native_program_uop import native_nv_program
  main = native_nv_program("q4k_imma_stream", provider.main.lib, global_size=MAIN_GRID, local_size=BLOCK,
    globals=tuple(range(7)), outs=(0,1,2), ins=(3,4,5,6), vals=(M,N,K),
    shared_mem=DYNAMIC_SHARED_BYTES+NV_RUNTIME_SHARED_BYTES)
  # out is both an input (direct tiles already written by main) and output.
  fixup = native_nv_program("q4k_imma_fixup", provider.fixup.lib, global_size=FIXUP_GRID, local_size=BLOCK,
    globals=(0,1,2), outs=(0,), ins=(0,1,2), vals=(M,N))
  return main, fixup
