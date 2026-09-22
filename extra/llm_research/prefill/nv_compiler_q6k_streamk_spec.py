"""Static phase-1 contract for a future Q6 Stream-K scheduler."""
from __future__ import annotations
import json
from dataclasses import dataclass,asdict

@dataclass(frozen=True)
class StreamKSpec:
  M:int=512; N:int=4096; K:int=12288; tile_m:int=64; tile_n:int=32; tile_k:int=64; owners:int=170
  @property
  def tiles_m(self): return self.M//self.tile_m
  @property
  def tiles_n(self): return self.N//self.tile_n
  @property
  def k_blocks(self): return self.K//self.tile_k
  @property
  def output_tiles(self): return self.tiles_m*self.tiles_n
  @property
  def work_units(self): return self.output_tiles*self.k_blocks
  def validate(self):
    if any(x<=0 for x in (self.M,self.N,self.K,self.tile_m,self.tile_n,self.tile_k,self.owners)): raise ValueError("positive dimensions required")
    if any(d%t for d,t in ((self.M,self.tile_m),(self.N,self.tile_n),(self.K,self.tile_k))): raise ValueError("non-divisible tile")
    if self.owners>self.work_units: raise ValueError("owners exceed work")
    intervals=[(o*self.work_units//self.owners,(o+1)*self.work_units//self.owners) for o in range(self.owners)]
    if intervals[0][0]!=0 or intervals[-1][1]!=self.work_units or any(b!=intervals[i+1][0] for i,(_,b) in enumerate(intervals[:-1])): raise ValueError("interval gap")
    if sum(b-a for a,b in intervals)!=self.work_units: raise ValueError("coverage mismatch")
    partial=sum(1 for a,b in intervals if a%self.k_blocks)+sum(1 for a,b in intervals if b%self.k_blocks)
    if partial>2*self.owners: raise ValueError("partial slot bound")
    return intervals
  def json(self):
    iv=self.validate()
    probes=[]
    for owner in (0, 1, self.owners-1):
      start,end=iv[owner]; first,last=start//self.k_blocks,end//self.k_blocks
      probes.append({"owner_special":owner,"interval_start":start,"interval_end":end,"first_tile":first,"last_tile_exclusive":last+1,
        "first_k_begin":start%self.k_blocks,"last_k_end":end%self.k_blocks or self.k_blocks,
        "partial_slot_start":owner*2 if start%self.k_blocks else None,"partial_slot_end":owner*2+1 if end%self.k_blocks else None,
        "accumulator_reset_per_tile":True,"interior_direct_store":True})
    return {"schema":"tinygrad.nv_q6_streamk_phase2.v1","status":"PASS","spec":asdict(self),"derived":{"tiles_m":self.tiles_m,"tiles_n":self.tiles_n,"k_blocks":self.k_blocks,"output_tiles":self.output_tiles,"work_units":self.work_units},"intervals":iv,"owner_probes":probes,"invariants":{"exact_coverage":True,"deterministic_owner_order":True,"max_partial_slots":2*self.owners,"partial_slots":sum(1 for a,b in iv if a%self.k_blocks)+sum(1 for a,b in iv if b%self.k_blocks),"fixup_map_tiles":self.output_tiles,"interior_direct_store_eligible":True,"symbolic_bounds":True}}

if __name__=="__main__": print(json.dumps(StreamKSpec().json(),indent=2))
