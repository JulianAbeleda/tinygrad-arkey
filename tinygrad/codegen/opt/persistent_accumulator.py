from dataclasses import dataclass

from tinygrad.codegen.opt.stream_k import StreamKSchedule

@dataclass(frozen=True)
class PersistentAccumulatorABI:
  accumulator_dtype: str = "float32"
  partial_slots_per_owner: int = 2

  def __post_init__(self):
    if self.accumulator_dtype != "float32": raise ValueError("matrix-MMA partials require float32 accumulation")
    if self.partial_slots_per_owner != 2: raise ValueError("Stream-K owner intervals require head and tail slots")

@dataclass(frozen=True)
class PersistentWorkItem:
  owner: int
  serial: int
  linear: int
  output_tile: int
  k_block: int
  reset: bool
  direct: bool
  partial: bool
  partial_slot: int|None

@dataclass(frozen=True)
class PersistentSegment:
  owner: int
  output_tile: int
  linear_start: int
  linear_stop: int
  direct: bool
  partial_slot: int|None

def owner_segments(schedule:StreamKSchedule, owner:int,
                   abi:PersistentAccumulatorABI=PersistentAccumulatorABI()) -> tuple[PersistentSegment, ...]:
  start,stop=schedule.interval(owner)
  first,last=start//schedule.k_blocks,(stop-1)//schedule.k_blocks
  segments=[]
  for tile in range(first,last+1):
    lo=max(start,tile*schedule.k_blocks)
    hi=min(stop,(tile+1)*schedule.k_blocks)
    direct=lo == tile*schedule.k_blocks and hi == (tile+1)*schedule.k_blocks
    tail=hi == stop and hi != (tile+1)*schedule.k_blocks
    has_head=start%schedule.k_blocks != 0
    slot=None if direct else owner*abi.partial_slots_per_owner+int(tail and has_head)
    segments.append(PersistentSegment(owner,tile,lo,hi,direct,slot))
  return tuple(segments)

def persistent_work_item(schedule:StreamKSchedule, owner:int, serial:int,
                         abi:PersistentAccumulatorABI=PersistentAccumulatorABI()) -> PersistentWorkItem:
  start,stop=schedule.interval(owner)
  if not 0 <= serial < stop-start: raise IndexError("serial work index outside owner interval")
  linear=start+serial
  tile,k_block=divmod(linear,schedule.k_blocks)
  tile_end=k_block == schedule.k_blocks-1
  owner_end=linear == stop-1
  saw_tile_start=start <= tile*schedule.k_blocks
  reset=serial == 0 or k_block == 0
  direct=tile_end and saw_tile_start
  partial=(tile_end and not saw_tile_start) or (owner_end and not tile_end)
  has_head=start%schedule.k_blocks != 0
  slot=owner*abi.partial_slots_per_owner+int(owner_end and not tile_end and has_head) if partial else None
  return PersistentWorkItem(owner,serial,linear,tile,k_block,reset,direct,partial,slot)

def fixup_contributors(schedule:StreamKSchedule, output_tile:int) -> tuple[PersistentWorkItem, ...]:
  if not 0 <= output_tile < schedule.output_tiles: raise IndexError("output tile outside schedule")
  contributors=[]
  for owner in range(schedule.owners):
    start,stop=schedule.interval(owner)
    for serial in range(stop-start):
      item=persistent_work_item(schedule,owner,serial)
      if item.output_tile == output_tile and item.partial: contributors.append(item)
  return tuple(sorted(contributors,key=lambda item:item.owner,reverse=True))
