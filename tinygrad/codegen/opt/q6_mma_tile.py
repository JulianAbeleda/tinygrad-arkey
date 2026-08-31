"""Backend-neutral Q6_K packed staging contract for a 64xK32 MMQ tile."""
from __future__ import annotations
from dataclasses import dataclass
import struct

@dataclass(frozen=True)
class Q6KMMATile:
  rows: int = 64
  warp_size: int = 32
  nwarps: int = 2
  threads_per_row: int = 4
  shared_stride_words: int = 77
  ql_bytes: int = 128
  qh_bytes: int = 64
  scale_bytes: int = 16
  block_bytes: int = 210

  def validate(self) -> None:
    if (self.rows,self.warp_size,self.nwarps,self.threads_per_row,self.shared_stride_words) != (64,32,2,4,77):
      raise ValueError("unsupported Q6_K MMQ tile")
    if self.ql_bytes+self.qh_bytes+self.scale_bytes+2 != self.block_bytes: raise ValueError("invalid Q6_K block layout")

  @staticmethod
  def _u32(data:bytes, offset:int) -> int: return int.from_bytes(data[offset:offset+4], "little")

  def stage_cpu(self, blocks:bytes) -> tuple[int, ...]:
    self.validate()
    if len(blocks) != self.rows*self.block_bytes: raise ValueError("expected 64 canonical Q6_K blocks")
    out = [0] * (self.rows*self.shared_stride_words)
    for y in range(self.nwarps):
      for x in range(self.warp_size):
        txi=x%self.threads_per_row
        for i0 in range(0,self.rows,8*self.nwarps):
          row=i0+y*8+x//self.threads_per_row; base=row*self.block_bytes
          ql=self._u32(blocks,base+4*txi); qh=self._u32(blocks,base+self.ql_bytes+4*txi)
          q0=((ql&0x0f0f0f0f)|((qh<<4)&0x30303030))
          q1=(((ql>>4)&0x0f0f0f0f)|(qh&0x30303030))
          # Per-byte subtraction is borrow-free only when performed lane-wise.
          pack=lambda q: sum((((q>>(8*j))&0xff)-32 & 0xff)<<(8*j) for j in range(4))
          out[row*self.shared_stride_words+2*txi]=pack(q0)
          out[row*self.shared_stride_words+2*txi+8]=pack(q1)
        row=(y*self.warp_size+x)%self.rows; base=row*self.block_bytes
        d=struct.unpack_from("<e",blocks,base+self.ql_bytes+self.qh_bytes+self.scale_bytes)[0]
        out[row*self.shared_stride_words+64]=struct.unpack("<I",struct.pack("<f",d))[0]
        for i0 in range(0,self.rows,8*self.nwarps):
          row=i0+y*8+x//4; base=row*self.block_bytes
          out[row*self.shared_stride_words+66+x%4]=self._u32(blocks,base+self.ql_bytes+self.qh_bytes+4*(x%4))
    return tuple(out)

  def emit_cuda(self, name:str="q6k_stage_64") -> str:
    self.validate()
    return f'''#include <cuda_fp16.h>
__device__ __forceinline__ unsigned int q6_get_b2(const unsigned char *p,int i) {{
  const unsigned short *s=(const unsigned short*)p; return (unsigned int)s[2*i]|((unsigned int)s[2*i+1]<<16);
}}
extern "C" __global__ void {name}(const unsigned char *blocks, unsigned int *out) {{
  __shared__ int tile[{self.rows*self.shared_stride_words}];
  int x=threadIdx.x,y=threadIdx.y,txi=x&3;
  for (int z=x+32*y;z<{self.rows*self.shared_stride_words};z+=64) tile[z]=0;
  __syncthreads();
  for (int i0=0;i0<{self.rows};i0+=16) {{
    int row=i0+y*8+x/4; const unsigned char *b=blocks+row*{self.block_bytes};
    unsigned int ql=q6_get_b2(b,txi),qh=q6_get_b2(b+{self.ql_bytes},txi);
    unsigned int q0=(ql&0x0f0f0f0fu)|((qh<<4)&0x30303030u);
    unsigned int q1=((ql>>4)&0x0f0f0f0fu)|(qh&0x30303030u);
    tile[row*{self.shared_stride_words}+2*txi]=__vsubss4(q0,0x20202020u);
    tile[row*{self.shared_stride_words}+2*txi+8]=__vsubss4(q1,0x20202020u);
  }}
  int row=y*32+x; const unsigned char *db=blocks+row*{self.block_bytes};
  ((float*)(tile+64))[row*{self.shared_stride_words}]=__half2float(*(const half*)(db+208));
  for (int i0=0;i0<{self.rows};i0+=16) {{
    row=i0+y*8+x/4; const unsigned char *sb=blocks+row*{self.block_bytes}+192;
    tile[row*{self.shared_stride_words}+66+x%4]=q6_get_b2(sb,x%4);
  }}
  __syncthreads();
  for (int z=x+32*y;z<{self.rows*self.shared_stride_words};z+=64) out[z]=(unsigned int)tile[z];
}}'''

__all__ = ["Q6KMMATile"]
