"""Backend-neutral deterministic Stream-K work ownership."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StreamKSchedule:
  m:int; n:int; k:int; tile_m:int; tile_n:int; tile_k:int; owners:int; boundary_quantum_k_blocks:int=1
  def __post_init__(self):
    values=(self.m,self.n,self.k,self.tile_m,self.tile_n,self.tile_k,self.owners,self.boundary_quantum_k_blocks)
    if any(not isinstance(x,int) or isinstance(x,bool) or x<=0 for x in values): raise ValueError("Stream-K values must be positive integers")
    if any(dim%tile for dim,tile in ((self.m,self.tile_m),(self.n,self.tile_n),(self.k,self.tile_k))):
      raise ValueError("Stream-K dimensions must be tile divisible")
    if self.owners>self.work_units: raise ValueError("Stream-K owners exceed work units")
    if self.k_blocks%self.boundary_quantum_k_blocks: raise ValueError("Stream-K K blocks must contain whole boundary quanta")
  @property
  def tiles_m(self): return self.m//self.tile_m
  @property
  def tiles_n(self): return self.n//self.tile_n
  @property
  def k_blocks(self): return self.k//self.tile_k
  @property
  def output_tiles(self): return self.tiles_m*self.tiles_n
  @property
  def work_units(self): return self.output_tiles*self.k_blocks
  def interval(self,owner:int) -> tuple[int,int]:
    if not 0<=owner<self.owners: raise IndexError(owner)
    q=self.boundary_quantum_k_blocks
    begin=(owner*self.work_units//self.owners//q)*q
    end=((owner+1)*self.work_units//self.owners//q)*q
    return begin,end
  def work_coordinate(self,work_unit:int) -> tuple[int,int]: return divmod(work_unit,self.k_blocks)
  def tail_is_partial(self,owner:int) -> bool:
    begin,end=self.interval(owner)
    return begin<end and end%self.k_blocks!=0
  def fixup_predecessors(self,owner:int) -> tuple[int,...]:
    begin,_=self.interval(owner)
    if begin%self.k_blocks==0:return ()
    tile=begin//self.k_blocks; out=[]
    for prior in range(owner-1,-1,-1):
      pbegin,pend=self.interval(prior)
      if pbegin==pend:continue
      if (pend-1)//self.k_blocks!=tile:break
      out.append(prior)
      if pbegin%self.k_blocks==0:break
    return tuple(out)
  def validate_exact_cover(self):
    intervals=tuple(self.interval(owner) for owner in range(self.owners))
    if intervals[0][0]!=0 or intervals[-1][1]!=self.work_units or any(a[1]!=b[0] for a,b in zip(intervals,intervals[1:])):
      raise ValueError("Stream-K intervals do not exactly cover work units")
    return self
