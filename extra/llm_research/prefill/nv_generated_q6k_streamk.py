"""Research-only generated Q6_K Stream-K ownership contract.

This module owns partitioning and metadata; Q6 arithmetic remains in the
canonical packed CTA segment kernel.
"""
from dataclasses import dataclass
from tinygrad import dtypes
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp
from extra.llm_research.prefill.nv_native_fragment_k16_gate import q6_packed_cta_kernel
from extra.llm_research.prefill.nv_generated_q6k_streamk_slots import q6_streamk_owner_kernel

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

def generated_owner_boundary_gate(out, values):
  """Single-register executable gate for owner boundary flush/reset semantics."""
  owner=UOp.special(OWNERS,"gidx0"); total=UOp.const(dtypes.int32,TILES*K_BLOCKS)
  lo=(owner*total)//OWNERS; hi=((owner+1)*total)//OWNERS
  blk=UOp.range(hi-lo,40,axis_type=AxisType.REDUCE); linear=lo+blk
  tile=linear//K_BLOCKS; previous=(linear-1)//K_BLOCKS
  transition=(blk>0)&(tile!=previous)
  acc=UOp.placeholder((1,),dtypes.float32,940,addrspace=__import__('tinygrad.dtype',fromlist=['AddrSpace']).AddrSpace.REG)
  init=acc[0].store(0.0); state=acc.after(init).after(blk)
  flush=out[owner*2].store(state[0],gate=transition)
  update=state.after(flush)[0].store(transition.where(values[linear],state[0]+values[linear]))
  done=update.end(blk); crossed=(lo//K_BLOCKS)!=((hi-1)//K_BLOCKS)
  final=out[owner*2+crossed.cast(dtypes.int32)].store(acc.after(done)[0])
  return UOp.sink(final,arg=KernelInfo(name="nv_generated_q6_owner_boundary_gate",opts_to_apply=()))

def generated_q6k_streamk_owner_partials(partials, tile_ids, blocks, b, dB):
  """170-CTA exact Stream-K main which emits at most two ordered tile partials.

  This is the schedule substrate: each owner has one mandatory segment and an
  optional second segment. A later writeback/fixup pass decides direct output
  versus reduction ownership; no inactive fixed-length MMA loop is emitted.
  """
  return q6_streamk_owner_kernel(partials,tile_ids,blocks,b,dB,total_k_blocks=K_BLOCKS,owners=OWNERS,
    kernel_name="nv_generated_q6k_streamk_owner_partials")

__all__=["OWNERS","SLOTS","ACTIVE_SLOTS","TILES","K_BLOCKS","OwnerSegment","owner_bounds","owner_work_units","streamk_segments",
         "tile_coordinates","owner_metadata","fixup_slot_map","generated_owner_boundary_gate","generated_q6k_streamk_owner_partials"]
