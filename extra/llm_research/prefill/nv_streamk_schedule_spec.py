"""Generic GPU-free Stream-K schedule contract shared by Q4/Q6 research paths."""
from dataclasses import dataclass
@dataclass(frozen=True)
class StreamKScheduleSpec:
  M:int; N:int; K:int; tile_m:int; tile_n:int; tile_k:int; owners:int=170
  @property
  def tiles_m(self): return self.M//self.tile_m
  @property
  def tiles_n(self): return self.N//self.tile_n
  @property
  def output_tiles(self): return self.tiles_m*self.tiles_n
  @property
  def k_blocks(self): return self.K//self.tile_k
  @property
  def total_work(self): return self.output_tiles*self.k_blocks
  @property
  def partial_slots(self): return 2*self.owners
  @property
  def fixup_grid(self): return self.output_tiles
  def interval(self, owner): return owner*self.total_work//self.owners,(owner+1)*self.total_work//self.owners
  def tile_segment(self, owner):
    a,b=self.interval(owner); return a//self.k_blocks,b//self.k_blocks,a%self.k_blocks,b%self.k_blocks
  def validate(self):
    if min(self.M,self.N,self.K,self.tile_m,self.tile_n,self.tile_k,self.owners)<=0: raise ValueError("invalid Stream-K dimensions")
    if self.M%self.tile_m or self.N%self.tile_n or self.K%self.tile_k or self.owners>self.total_work: raise ValueError("Stream-K divisibility/owner invariant")
    if self.interval(0)[0]!=0 or self.interval(self.owners-1)[1]!=self.total_work: raise ValueError("Stream-K coverage")
    return self
