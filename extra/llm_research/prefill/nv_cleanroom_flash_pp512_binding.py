"""Default-off descriptor for the clean-room pp512 Flash candidate."""
from __future__ import annotations
from dataclasses import dataclass
from tinygrad import dtypes

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

BINDING=CleanroomFlashPP512Binding()
