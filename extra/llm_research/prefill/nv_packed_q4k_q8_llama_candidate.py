"""Pinned llama Q4_K MMQ extraction for the pp512 gate/up discriminator."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

from tinygrad.runtime.ops_nv import NVProgram

M, N, K = 512, 12288, 4096
MMQ_X = MMQ_Y = 128
MAIN_GRID, MAIN_BLOCK = (170, 1, 1), (32, 8, 1)
FIXUP_GRID, FIXUP_BLOCK = (170, 4, 1), (32, 4, 1)
PRODUCER_GRID, PRODUCER_BLOCK = (M, K//(4*128), 1), (128, 1, 1)
SHARED_BYTES = 58880
Q8_RECORD_BYTES = M*(K//128)*144 + MMQ_X*144
SCRATCH_FLOATS = MAIN_GRID[0]*MMQ_X*MMQ_Y

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT/"docs/task_workflow/evidence/nv-packed-q4k-q8-llama-extracted-20260830"
MAIN_SYMBOL = "_Z9mul_mat_qIL9ggml_type12ELi128ELb0EEvPKcPKiS4_S4_PfS5_5uint3iiiiiS6_S6_iiiS6_S6_iiiS6_"
FIXUP_SYMBOL = "_Z24mul_mat_q_stream_k_fixupIL9ggml_type12ELi128ELb0EEvPKiS2_PfS3_5uint3iiiS4_iS4_iS4_"
PRODUCER_SYMBOL = "_Z17quantize_mmq_q8_1IL18mmq_q8_1_ds_layout1EEvPKfPKiPvlllllii"

class _NullBuffer:
  va_addr = 0

NULL = _NullBuffer()

def fastdiv(divisor:int) -> tuple[int,int,int]:
  level = 0
  while (1 << level) < divisor: level += 1
  multiplier = ((1 << 32)*((1 << level)-divisor)//divisor + 1) & 0xffffffff
  return multiplier, level, divisor

def i64(value:int) -> tuple[int,int]: return value & 0xffffffff, (value >> 32) & 0xffffffff

@dataclass(frozen=True)
class Metadata:
  producer_grid: tuple[int,int,int] = PRODUCER_GRID
  producer_block: tuple[int,int,int] = PRODUCER_BLOCK
  main_grid: tuple[int,int,int] = MAIN_GRID
  main_block: tuple[int,int,int] = MAIN_BLOCK
  fixup_grid: tuple[int,int,int] = FIXUP_GRID
  fixup_block: tuple[int,int,int] = FIXUP_BLOCK
  shared_bytes: int = SHARED_BYTES
  q8_record_bytes: int = Q8_RECORD_BYTES
  scratch_bytes: int = SCRATCH_FLOATS*4
  producer_input_dtype: str = "float32"
  record_layout: str = "llama block_q8_1_mmq DS4"

@dataclass
class Candidate:
  producer: NVProgram
  main: NVProgram
  fixup: NVProgram
  metadata: Metadata = Metadata()

  @classmethod
  def compile(cls, dev):
    main_lib=(ARTIFACTS/"q4k-mmq.sm_120a.cubin").read_bytes()
    fixup_lib=(ARTIFACTS/"q4k-fixup.sm_120a.cubin").read_bytes()
    producer_lib=(ARTIFACTS/"q8-ds4.sm_120a.cubin").read_bytes()
    return cls(NVProgram(dev,PRODUCER_SYMBOL,producer_lib), NVProgram(dev,MAIN_SYMBOL,main_lib,shared_mem=SHARED_BYTES),
      NVProgram(dev,FIXUP_SYMBOL,fixup_lib))

  def launch_producer(self, x, record, *, wait=False):
    # x, ids=null, record; ne00/s01/s02/s03/ne0 are int64, ne1/ne2 are int32.
    vals=(*i64(K),*i64(K),*i64(M*K),*i64(M*K),*i64(K),M,1)
    return self.producer(x,NULL,record,vals=vals,global_size=PRODUCER_GRID,local_size=PRODUCER_BLOCK,wait=wait)

  def launch_main(self, words, record, out, scratch, *, wait=False):
    fd1,fd16,fd4=fastdiv(1),fastdiv(K//256),fastdiv(M//MMQ_X)
    stride_x=N*(K//256); stride_y=M*(K//32)*9; stride_dst=M*N
    vals=(*fd16,N,M,K//256,M,N,*fd1,*fd1,stride_x,stride_y,stride_dst,
          *fd1,*fd1,stride_x,stride_y,stride_dst,*fd4)
    return self.main(words,record,NULL,NULL,out,scratch,vals=vals,global_size=MAIN_GRID,local_size=MAIN_BLOCK,wait=wait)

  def launch_fixup(self, out, scratch, *, wait=False):
    fd1,fd16,fd4=fastdiv(1),fastdiv(K//256),fastdiv(M//MMQ_X)
    stride_dst=M*N
    vals=(*fd16,N,M,N,*fd1,stride_dst,*fd1,stride_dst,*fd4)
    return self.fixup(NULL,NULL,out,scratch,vals=vals,global_size=FIXUP_GRID,local_size=FIXUP_BLOCK,wait=wait)

  def launch(self, x, words, record, out, scratch, *, wait=False):
    self.launch_producer(x,record,wait=wait)
    self.launch_main(words,record,out,scratch,wait=wait)
    self.launch_fixup(out,scratch,wait=wait)

def compile_candidate(dev): return Candidate.compile(dev)
