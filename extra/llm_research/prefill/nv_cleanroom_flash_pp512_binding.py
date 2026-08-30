"""Default-off descriptor for the clean-room pp512 Flash candidate."""
from __future__ import annotations
from dataclasses import dataclass
from tinygrad import dtypes
from tinygrad import Tensor
from tinygrad.helpers import getenv
from tinygrad.runtime.support.compiler_cuda import NVCCCompiler
from extra.llm_research.prefill.nv_native_program_uop import native_nv_program
from pathlib import Path

@dataclass(frozen=True)
class CleanroomFlashPP512Binding:
  name: str = 'nv_cleanroom_flash_phase2_kstage'
  shape: tuple[int,...] = (1,32,512,128)
  kv_shape: tuple[int,...] = (1,8,512,128)
  gqa: int = 4
  output_shape: tuple[int,...] = (1,32,512,128)

  def supports(self, q, k, v, *, start_pos: int, ring: bool=False) -> bool:
    return (tuple(q.shape)==self.shape and tuple(k.shape)==self.kv_shape and tuple(v.shape)==self.kv_shape
      and q.dtype==dtypes.float16 and k.dtype==dtypes.float16 and v.dtype==dtypes.float16
      and start_pos==0 and not ring and q.device=='NV')

  def reject_reason(self, q, k, v, *, start_pos: int, ring: bool=False) -> str:
    return 'admitted_exact_b1_t512_start0_nonring' if self.supports(q,k,v,start_pos=start_pos,ring=ring) else 'contract_mismatch'

  @classmethod
  def compile(cls):
    src=Path(__file__).with_name('nv_pp512_flash_phase2_fp16_reference.cu').read_text()
    cubin=NVCCCompiler(getenv('NV_CLEANROOM_FLASH_ARCH','sm_120a'),ptx=False,cache_key='nv_cleanroom_flash_pp512_fp16_v2',extra_options=['-std=c++17']).compile(src)
    return native_nv_program('nv_pp512_flash_phase2_fp16_reference',cubin,
      global_size=(512,32,1),local_size=(32,4,1),globals=(0,1,2,3),outs=(3,),ins=(0,1,2),vals=(0,512,512))

  def project(self, q: Tensor, k: Tensor, v: Tensor, *, start_pos:int=0, ring:bool=False):
    if not self.supports(q,k,v,start_pos=start_pos,ring=ring): raise ValueError('clean-room Flash contract mismatch')
    out=Tensor.empty((1,32,512,128),dtype=dtypes.float32,device=q.device)
    q,k,v,out=q.uop_program(k,v,out,fxn=lambda *_: program_for())
    return out

BINDING=CleanroomFlashPP512Binding()
_PROGRAM=None
def program_for():
  global _PROGRAM
  if _PROGRAM is None: _PROGRAM=BINDING.compile()
  return _PROGRAM
