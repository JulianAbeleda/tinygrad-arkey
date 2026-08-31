"""Research-only generated Q6_K Stream-K ownership contract.

This module owns partitioning and metadata; Q6 arithmetic remains in the
canonical packed CTA segment kernel.
"""
from dataclasses import dataclass
from tinygrad import dtypes
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp
from extra.llm_research.prefill.nv_native_fragment_k16_gate import q6_packed_cta_kernel

OWNERS, SLOTS, ACTIVE_SLOTS, TILES, K_BLOCKS = 170, 340, 294, 128, 48
TILE_M, TILE_N = 4, 32

@dataclass(frozen=True)
class OwnerSegment:
  owner: int
  tile_id: int
  begin: int
  end: int
  slot: int

def owner_bounds(owner: int):
  if not 0 <= owner < OWNERS: raise ValueError("owner outside Stream-K grid")
  return (owner * (TILES*K_BLOCKS)) // OWNERS, ((owner+1) * (TILES*K_BLOCKS)) // OWNERS

def owner_work_units(owner: int):
  """Exact count of Q6_K `MMQ_ITER_K` units assigned to one Stream-K CTA."""
  lo,hi=owner_bounds(owner)
  return hi-lo

def streamk_segments(owner: int):
  """Return the contiguous tile/K segments assigned to an owner."""
  begin, end = owner_bounds(owner); out=[]
  for linear in range(begin, end):
    tile, kb = divmod(linear, K_BLOCKS)
    if not out or out[-1].tile_id != tile:
      out.append(OwnerSegment(owner, tile, kb, kb+1, owner*2+len(out)))
    else:
      p=out[-1]; out[-1]=OwnerSegment(owner,tile,p.begin,kb+1,p.slot)
  if len(out) > 2: raise AssertionError("Stream-K owner exceeds two segments")
  return tuple(out)

def tile_coordinates(tile_id: int):
  if not 0 <= tile_id < TILES: raise ValueError("tile outside 4x32 grid")
  return divmod(tile_id, TILE_N)

def owner_metadata():
  """340 post-scale partial slots: two slots per owner, with tile metadata."""
  rows=[]
  for owner in range(OWNERS):
    segs=streamk_segments(owner)
    rows.extend(segs)
    rows.extend(OwnerSegment(owner,-1,0,0,owner*2+i) for i in range(len(segs),2))
  return tuple(rows)

def fixup_slot_map():
  """Return the deterministic [128,3] owner-ordered partial-slot map."""
  by_tile=[[] for _ in range(TILES)]
  for seg in owner_metadata():
    if seg.tile_id >= 0: by_tile[seg.tile_id].append(seg.slot)
  if max(map(len,by_tile)) > 3 or min(map(len,by_tile)) < 2:
    raise AssertionError("representative Q6 tile contributor bound changed")
  return tuple(tuple(slots+[-1]*(3-len(slots))) for slots in by_tile)

def generated_q6k_streamk_owner_partials(partials, tile_ids, blocks, b, dB):
  """170-CTA exact Stream-K main which emits at most two ordered tile partials.

  This is the schedule substrate: each owner has one mandatory segment and an
  optional second segment. A later writeback/fixup pass decides direct output
  versus reduction ownership; no inactive fixed-length MMA loop is emitted.
  """
  owner=UOp.special(OWNERS,"gidx0"); total=UOp.const(dtypes.int32,TILES*K_BLOCKS)
  lo=(owner*total)//OWNERS; hi=((owner+1)*total)//OWNERS
  tile0=lo//K_BLOCKS; tile0_stop=(tile0+1)*K_BLOCKS
  first_stop=(hi<tile0_stop).where(hi,tile0_stop); first_len=first_stop-lo
  tile1=first_stop//K_BLOCKS; second_len=hi-first_stop
  mt0,nt0=tile0//TILE_N,tile0%TILE_N; mt1,nt1=tile1//TILE_N,tile1%TILE_N
  out0=partials.index(owner*2*16384,ptr=True); out1=partials.index((owner*2+1)*16384,ptr=True)
  rows0=blocks.index(nt0*128*K_BLOCKS*105,ptr=True); rows1=blocks.index(nt1*128*K_BLOCKS*105,ptr=True)
  body0=q6_packed_cta_kernel(out0,rows0,b,dB,K_BLOCKS,col_groups=8,block_start=lo%K_BLOCKS,
    segment_blocks=first_len,total_k_blocks=K_BLOCKS,activation_stride=512,activation_offset=mt0*128,allocation_base=0,axis_base=0)
  body1=q6_packed_cta_kernel(out1,rows1,b,dB,K_BLOCKS,col_groups=8,block_start=UOp.const(dtypes.int32,0),
    segment_blocks=second_len,total_k_blocks=K_BLOCKS,activation_stride=512,activation_offset=mt1*128,
    allocation_base=0,register_base=10,axis_base=2)
  meta=UOp.group(tile_ids[owner*2].store(tile0),tile_ids[owner*2+1].store((second_len>0).where(tile1,-1)))
  return UOp.sink(*(body0.src+body1.src+meta.src),arg=KernelInfo(name="nv_generated_q6k_streamk_owner_partials",opts_to_apply=()))

__all__=["OWNERS","SLOTS","ACTIVE_SLOTS","TILES","K_BLOCKS","OwnerSegment","owner_bounds","owner_work_units","streamk_segments",
         "tile_coordinates","owner_metadata","fixup_slot_map","generated_q6k_streamk_owner_partials"]
