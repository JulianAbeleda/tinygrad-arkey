"""Backend-neutral Q6_K packed staging contract for a 64xK32 MMQ tile."""
from __future__ import annotations
from dataclasses import dataclass
import struct

@dataclass(frozen=True)
class Q6KMMATile:
  rows: int = 64
  warp_size: int = 32
  nwarps: int = 2
  threads_per_row: int = 32
  shared_stride_words: int = 76
  ql_bytes: int = 128
  qh_bytes: int = 64
  scale_bytes: int = 16
  block_bytes: int = 210

  def validate(self) -> None:
    if (self.rows,self.warp_size,self.nwarps,self.threads_per_row,self.shared_stride_words) != (64,32,2,32,76):
      raise ValueError("unsupported Q6_K MMQ tile")
    if self.ql_bytes+self.qh_bytes+self.scale_bytes+2 != self.block_bytes: raise ValueError("invalid Q6_K block layout")

  @staticmethod
  def _u32(data:bytes, offset:int) -> int: return int.from_bytes(data[offset:offset+4], "little")

  def stage_cpu(self, blocks:bytes) -> tuple[int, ...]:
    self.validate()
    if len(blocks) != self.rows*self.block_bytes: raise ValueError("expected 64 canonical Q6_K blocks")
    out = [0] * (self.rows*self.shared_stride_words)
    for row in range(self.rows):
      base=row*self.block_bytes
      for txi in range(self.warp_size):
          ql=self._u32(blocks,base+4*txi)
          qhi=(txi//16)*8+txi%8; qh=self._u32(blocks,base+self.ql_bytes+4*qhi); shift=(txi&8)>>2
          q0=((ql&0x0f0f0f0f)|(((qh>>shift)<<4)&0x30303030))
          q1=(((ql>>4)&0x0f0f0f0f)|((qh>>shift)&0x30303030))
          # Per-byte subtraction is borrow-free only when performed lane-wise.
          pack=lambda q: sum((((q>>(8*j))&0xff)-32 & 0xff)<<(8*j) for j in range(4))
          kq0=2*txi-txi%16
          out[row*self.shared_stride_words+kq0]=pack(q0)
          out[row*self.shared_stride_words+kq0+16]=pack(q1)
      d=struct.unpack_from("<e",blocks,base+self.ql_bytes+self.qh_bytes+self.scale_bytes)[0]
      out[row*self.shared_stride_words+64]=struct.unpack("<I",struct.pack("<f",d))[0]
      for word in range(4): out[row*self.shared_stride_words+65+word]=self._u32(blocks,base+self.ql_bytes+self.qh_bytes+4*word)
    return tuple(out)

  def emit_cuda(self, name:str="q6k_stage_64") -> str:
    self.validate()
    return f'''#include <cuda_fp16.h>
__device__ __forceinline__ unsigned int q6_get_b2(const unsigned char *p,int i) {{
  const unsigned short *s=(const unsigned short*)p; return (unsigned int)s[2*i]|((unsigned int)s[2*i+1]<<16);
}}
extern "C" __global__ void {name}(const unsigned char *blocks, unsigned int *out) {{
  __shared__ int tile[{self.rows*self.shared_stride_words}];
  int x=threadIdx.x,y=threadIdx.y,txi=x;
  for (int z=x+32*y;z<{self.rows*self.shared_stride_words};z+=64) tile[z]=0;
  __syncthreads();
  for (int row=y;row<{self.rows};row+=2) {{
    const unsigned char *b=blocks+row*{self.block_bytes}; int qhi=(txi/16)*8+txi%8,shift=(txi&8)>>2;
    unsigned int ql=q6_get_b2(b,txi),qh=q6_get_b2(b+{self.ql_bytes},qhi);
    unsigned int q0=(ql&0x0f0f0f0fu)|(((qh>>shift)<<4)&0x30303030u);
    unsigned int q1=((ql>>4)&0x0f0f0f0fu)|((qh>>shift)&0x30303030u); int kq0=2*txi-txi%16;
    tile[row*{self.shared_stride_words}+kq0]=__vsubss4(q0,0x20202020u);
    tile[row*{self.shared_stride_words}+kq0+16]=__vsubss4(q1,0x20202020u);
  }}
  int row=y*32+x; const unsigned char *db=blocks+row*{self.block_bytes};
  ((float*)(tile+64))[row*{self.shared_stride_words}]=__half2float(*(const half*)(db+208));
  if (x<4) for (int i=y;i<{self.rows};i+=2) tile[i*{self.shared_stride_words}+65+x]=q6_get_b2(blocks+i*{self.block_bytes}+192,x);
  __syncthreads();
  for (int z=x+32*y;z<{self.rows*self.shared_stride_words};z+=64) out[z]=(unsigned int)tile[z];
}}'''

__all__ = ["Q6KMMATile"]
