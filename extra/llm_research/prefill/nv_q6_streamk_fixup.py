"""Deterministic Q6 Stream-K partial fixup contract (research-only)."""
import numpy as np
from tinygrad import dtypes
from tinygrad.uop.ops import KernelInfo, UOp

TILE, M, N = 128, 512, 4096

def fixup_numpy(partials, tile_ids, out=None):
  """Compose owner-ordered partial slots into the row-major MxN output."""
  p=np.asarray(partials, dtype=np.float32); ids=np.asarray(tile_ids, dtype=np.int32)
  if ids.ndim != 2 or ids.shape[1] not in (2,3): raise ValueError("tile_ids must be [tiles,2|3]")
  if p.ndim != 2 or p.shape[1] != TILE*TILE: raise ValueError("partials must contain 128x128 tiles")
  result=np.zeros((M,N),np.float32) if out is None else np.asarray(out,dtype=np.float32).reshape(M,N)
  tiles=N//TILE
  for tile, slots in enumerate(ids):
    r0,c0=divmod(int(tile),tiles); acc=np.zeros((TILE,TILE),np.float32)
    for slot in slots:
      if slot >= 0: acc += p[int(slot)].reshape(TILE,TILE)
    result[r0*TILE:(r0+1)*TILE,c0*TILE:(c0+1)*TILE]=acc
  return result.reshape(-1)

def emit_fixup_kernel(out, partials, tile_ids):
  """Emit one deterministic 128-CTA/256-thread FP32 fixup kernel."""
  tile=UOp.special(128,"gidx0"); lane=UOp.special(256,"lidx0"); base=tile*3
  writes=[]
  for i in range(64):
    z=lane+256*i; acc=UOp.const(dtypes.float32,0.0)
    for j in range(3):
      slot=tile_ids[base+j].cast(dtypes.weakint); value=partials[slot*16384+z]
      acc=(slot < 0).where(acc,acc+value)
    tr=tile//32; tc=tile&31; rr=z//128; cc=z&127
    writes.append(out[(tr*128+rr)*4096+tc*128+cc].store(acc))
  return UOp.sink(*writes,arg=KernelInfo(name="nv_q6_streamk_fixup",opts_to_apply=()))

__all__=["M","N","TILE","emit_fixup_kernel","fixup_numpy"]
