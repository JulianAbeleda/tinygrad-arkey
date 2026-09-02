"""Deterministic reduction policies for the admitted packed Q6 Stream-K body."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import inspect
import numpy as np

from tinygrad import dtypes
from tinygrad.uop.ops import KernelInfo, Ops, UOp
from extra.llm_research.prefill.nv_q6_oracle_broad_cta import (
  ROWS, COLS, q6_oracle_broad_cta_kernel)

M, N, K = 512, 4096, 12288
OWNERS, K256 = 170, 48
TILES_M, TILES_N, TILES = 4, 32, 128
TILE_ELEMS = ROWS * COLS

ALL_PARTIALS_ASCENDING = "all_partials_ascending"
LLAMA_STANDALONE_ASCENDING = "llama_standalone_ascending"
LLAMA_STANDALONE_NONFINAL_FIRST = "llama_standalone_nonfinal_first"
ARM_NAMES = (ALL_PARTIALS_ASCENDING, LLAMA_STANDALONE_ASCENDING, LLAMA_STANDALONE_NONFINAL_FIRST)

SEGMENT_ORDER_BUILDER_PATCH = {
  "status": "blocked_until_builder_toggle",
  "file": "extra/llm_research/prefill/nv_q6_oracle_broad_cta.py",
  "function": "q6_oracle_broad_cta_kernel",
  "parameter": "streamk_segment_order",
  "allowed_values": ["ascending", "nonfinal_first"],
  "required_change": (
    "Keep RANGE 1498 ascending, but map its logical iteration to segment_count-1-segment_loop "
    "when nonfinal_first is selected; use that mapped ordinal for tile, epoch_start, depth, and slot."
  ),
  "reason": "B-prime must change only owner-local segment order without rewriting rendered CUDA or duplicating the admitted body",
}


@dataclass(frozen=True)
class SegmentDescriptor:
  owner: int
  plane: int
  slot: int
  tile: int
  k_begin: int
  k_end: int
  is_final: bool

  def record(self, fold_position:int) -> dict[str, object]:
    return asdict(self) | {"fold_position": fold_position}


@dataclass(frozen=True)
class ReductionSchedule:
  records: tuple[SegmentDescriptor, ...]
  ordered_by_tile: tuple[tuple[SegmentDescriptor, ...], ...]

  @property
  def predecessor_slots(self) -> tuple[int, ...]:
    return tuple(x.slot for x in self.records if not x.is_final)

  @property
  def final_slots(self) -> tuple[int, ...]:
    return tuple(x.slot for x in self.records if x.is_final)

  @property
  def active_slots(self) -> tuple[int, ...]:
    return tuple(x.slot for x in self.records)

  def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    slots=np.zeros((TILES,3),dtype=np.int32)
    counts=np.empty((TILES,),dtype=np.int32)
    final_indices=np.empty((TILES,),dtype=np.int32)
    for tile,descriptors in enumerate(self.ordered_by_tile):
      counts[tile]=len(descriptors)
      slots[tile,:len(descriptors)]=[x.slot for x in descriptors]
      final_indices[tile]=next(i for i,x in enumerate(descriptors) if x.is_final)
    return slots,counts,final_indices

  def json_record(self) -> dict[str, object]:
    slots,counts,final_indices=self.arrays()
    current_plane_order=[]
    for tile in range(TILES):
      current_plane_order.append(sorted(x.slot for x in self.records if x.tile == tile))
    return {
      "owners": OWNERS, "tiles": TILES, "k256_per_tile": K256,
      "slot_layout": "plane_major: slot=plane*170+owner", "allocated_slots": 2*OWNERS,
      "active_slots": len(self.records), "preceding_segments": len(self.predecessor_slots),
      "final_segments": len(self.final_slots),
      "fold_order": "nonfinal contributors in descending owner order, then the final contributor",
      "descriptors": [[x.record(i) for i,x in enumerate(row)] for row in self.ordered_by_tile],
      "slot_map": slots.tolist(), "counts": counts.tolist(), "final_indices": final_indices.tolist(),
      "admitted_plane_major_slot_order": current_plane_order,
    }


def build_reduction_schedule() -> ReductionSchedule:
  records=[]
  for owner in range(OWNERS):
    start=owner*TILES*K256//OWNERS
    stop=(owner+1)*TILES*K256//OWNERS
    tile0=start//K256
    boundary=(tile0+1)*K256
    first_stop=min(stop,boundary)
    segments=[(0,tile0,start-tile0*K256,first_stop-tile0*K256)]
    if stop>boundary: segments.append((1,tile0+1,0,stop-boundary))
    for plane,tile,k_begin,k_end in segments:
      records.append(SegmentDescriptor(owner,plane,plane*OWNERS+owner,tile,k_begin,k_end,k_end==K256))
  ordered=[]
  for tile in range(TILES):
    contributors=[x for x in records if x.tile == tile]
    final=[x for x in contributors if x.is_final]
    if len(final) != 1: raise ValueError(f"tile {tile} has {len(final)} final contributors")
    preceding=sorted((x for x in contributors if not x.is_final),key=lambda x:x.owner,reverse=True)
    ordered.append(tuple(preceding+final))
  schedule=ReductionSchedule(tuple(records),tuple(ordered))
  counts=[len(x) for x in schedule.ordered_by_tile]
  if len(records)!=294 or counts.count(2)!=90 or counts.count(3)!=38:
    raise ValueError("fixed Q6 Stream-K ownership drift")
  if len(schedule.predecessor_slots)!=166 or len(schedule.final_slots)!=128:
    raise ValueError("fixed Q6 final/predecessor census drift")
  return schedule


def fold_values(values:tuple[np.ndarray, ...]) -> np.ndarray:
  if not values: raise ValueError("a reduction requires at least one contributor")
  acc=np.zeros_like(np.asarray(values[0],dtype=np.float32),dtype=np.float32)
  for value in values: np.add(acc,np.asarray(value,dtype=np.float32),out=acc)
  return acc


def _tile_view(output:np.ndarray, tile:int) -> np.ndarray:
  mt,nt=tile%TILES_M,tile//TILES_M
  return output[mt*COLS:(mt+1)*COLS,nt*ROWS:(nt+1)*ROWS].T


def cpu_all_partials(partials:np.ndarray, schedule:ReductionSchedule) -> np.ndarray:
  raw=np.asarray(partials,dtype=np.float32).reshape(2*OWNERS,ROWS,COLS)
  output=np.empty((M,N),dtype=np.float32)
  for tile,descriptors in enumerate(schedule.ordered_by_tile):
    _tile_view(output,tile)[:]=fold_values(tuple(raw[x.slot] for x in descriptors))
  return output


def cpu_direct_final(partials:np.ndarray, direct_output:np.ndarray, schedule:ReductionSchedule) -> np.ndarray:
  raw=np.asarray(partials,dtype=np.float32).reshape(2*OWNERS,ROWS,COLS)
  direct=np.asarray(direct_output,dtype=np.float32).reshape(M,N)
  output=np.empty((M,N),dtype=np.float32)
  for tile,descriptors in enumerate(schedule.ordered_by_tile):
    values=tuple(_tile_view(direct,tile) if x.is_final else raw[x.slot] for x in descriptors)
    _tile_view(output,tile)[:]=fold_values(values)
  return output


def direct_output_from_partials(partials:np.ndarray, schedule:ReductionSchedule) -> np.ndarray:
  raw=np.asarray(partials,dtype=np.float32).reshape(2*OWNERS,ROWS,COLS)
  output=np.empty((M,N),dtype=np.float32)
  for tile,descriptors in enumerate(schedule.ordered_by_tile):
    final=next(x for x in descriptors if x.is_final)
    _tile_view(output,tile)[:]=raw[final.slot]
  return output


def supports_nonfinal_first() -> bool:
  return "streamk_segment_order" in inspect.signature(q6_oracle_broad_cta_kernel).parameters


def build_packed_one_body_ast(segment_order:str="ascending", partial_output_layout:str="tile_row_major", *,
                              region_load_bridge_q8_panel1:bool=False, fp32_contraction:str="implicit",
                              weight_scale_contract:str="trusted_fp16_packed", factor_dA:bool=False,
                              fp32_p_tree:str="legacy", fp32_scale_grouping:str="legacy",
                              q6_fragment_schedule:str="preload", q6_metadata_schedule:str="preload",
                              strict_after_q8_panel1:bool=False, q8_panel1_anchor_cg:int=5,
                              q6_d_storage:str="fp16_bits",
                              streamk_segments_in_cta:bool=True, streamk_segment:int=0) -> UOp:
  if segment_order not in ("ascending","nonfinal_first"): raise ValueError(segment_order)
  if segment_order == "nonfinal_first" and not supports_nonfinal_first():
    raise RuntimeError(SEGMENT_ORDER_BUILDER_PATCH["reason"])
  ph=lambda n,dt,i:UOp.placeholder((n,),dt,i)
  kwargs={"prefetch_second_panel":not region_load_bridge_q8_panel1 and not strict_after_q8_panel1,"combined_initial_publish":True,"factor_dA":factor_dA,"oracle_publisher":True,
    "weight_scale_contract":weight_scale_contract,"fp32_p_tree":fp32_p_tree,"fp32_scale_grouping":fp32_scale_grouping,
    "q6_fragment_schedule":q6_fragment_schedule,
    "q6_metadata_schedule":q6_metadata_schedule,
    "strict_after_q8_panel1":strict_after_q8_panel1,"q8_panel1_anchor_cg":q8_panel1_anchor_cg,
    "q6_d_storage":q6_d_storage,
    "streamk_owners":OWNERS,"streamk_segment":streamk_segment,
    "streamk_segments_in_cta":streamk_segments_in_cta,"partial_output_layout":partial_output_layout,
    "region_load_bridge_q8_panel1":region_load_bridge_q8_panel1,"fp32_contraction":fp32_contraction}
  if supports_nonfinal_first(): kwargs["streamk_segment_order"]=segment_order
  return q6_oracle_broad_cta_kernel(ph(2*OWNERS*TILE_ELEMS,dtypes.float32,0),ph(N*K256*105,dtypes.uint16,1),
    ph(TILES_M*K256*2*COLS*36,dtypes.uint32,2),**kwargs)


def _root_param(u:UOp) -> UOp:
  while u.op is not Ops.PARAM:
    if not u.src: raise ValueError("store target has no PARAM root")
    u=u.src[0]
  return u


def direct_final_writeback(base_ast:UOp, destination:UOp, *, name:str) -> UOp:
  """Redirect only final segment stores; retain the admitted compute graph verbatim."""
  if base_ast.op is not Ops.SINK or len(base_ast.src)!=1 or base_ast.src[0].op is not Ops.END:
    raise ValueError("admitted one-body terminal shape changed")
  body_end=base_ast.src[0]
  if len(body_end.src)!=2 or body_end.src[0].op is not Ops.GROUP or body_end.src[1].op is not Ops.RANGE:
    raise ValueError("admitted segment END shape changed")
  stores,segment=body_end.src
  if segment.arg[0]!=1498 or len(stores.src)!=64 or any(x.op is not Ops.STORE for x in stores.src):
    raise ValueError("admitted output-store group changed")
  if any(_root_param(x.src[0]).arg.slot != 0 for x in stores.src):
    raise ValueError("terminal group is not the partial workspace writeback")
  bid=next((x for x in base_ast.toposort() if x.op is Ops.SPECIAL and x.arg=="gidx0"),None)
  if bid is None: raise ValueError("owner special missing")
  work=TILES*K256
  owner_start=bid*work//OWNERS
  owner_stop=(bid+1)*work//OWNERS
  tile0=owner_start//K256
  boundary=(tile0+1)*K256
  first_stop=owner_stop.minimum(boundary)
  second=segment>0
  tile=second.where(tile0+1,tile0)
  epoch_start=second.where(owner_start*0,owner_start-tile0*K256)
  segment_depth=second.where(owner_stop-first_stop,first_stop-owner_start)
  is_final=(epoch_start+segment_depth).eq(K256)
  rewritten=[]
  for store in stores.src:
    scratch_index=store.src[0].src[1]
    z=scratch_index%TILE_ELEMS
    wr,mc=z//COLS,z%COLS
    mt,nt=tile%TILES_M,tile//TILES_M
    destination_index=(mt*COLS+mc)*N+nt*ROWS+wr
    gate=store.src[2]
    rewritten.append(store.replace(src=(store.src[0],store.src[1],gate&is_final.eq(False))))
    rewritten.append(destination[destination_index].store(store.src[1],gate=gate&is_final))
  terminal=body_end.replace(src=(UOp.group(*rewritten),segment))
  return base_ast.replace(src=(terminal,),arg=KernelInfo(name=name,opts_to_apply=()))


def emit_ordered_fixup(destination:UOp, partials:UOp, slots:UOp, counts:UOp,
                       final_indices:UOp|None=None, *, direct_final:bool=False) -> UOp:
  if direct_final != (final_indices is not None): raise ValueError("direct-final fixup requires final indices")
  tile=UOp.special(TILES,"gidx0")
  lane=UOp.special(256,"lidx0")
  count=counts[tile].load()
  final_index=final_indices[tile].load() if final_indices is not None else None
  zero=UOp.const(dtypes.float32,0.0)
  fp_add=lambda a,b:UOp(Ops.CUSTOMI,dtypes.float32,(a,b),arg="__fadd_rn({0},{1})")
  writes=[]
  for i in range(64):
    z=lane+256*i
    wr,mc=z//COLS,z%COLS
    mt,nt=tile%TILES_M,tile//TILES_M
    destination_index=(mt*COLS+mc)*N+nt*ROWS+wr
    acc=zero
    for j in range(3):
      valid=count>j
      slot=slots[tile*3+j].load()
      safe_slot=valid.where(slot,UOp.const(dtypes.int32,0))
      if direct_final:
        assert final_index is not None
        final=valid&final_index.eq(j)
        partial_value=partials[safe_slot*TILE_ELEMS+z].load(zero,valid&final.eq(False))
        direct_value=destination[destination_index].load(zero,final)
        value=final.where(direct_value,partial_value)
      else:
        value=partials[safe_slot*TILE_ELEMS+z].load(zero,valid)
      added=fp_add(acc,value)
      acc=valid.where(added,acc)
    writes.append(destination[destination_index].store(acc))
  suffix="direct_final" if direct_final else "all_partials"
  return UOp.sink(*writes,arg=KernelInfo(name=f"nv_q6_ordered_fixup_{suffix}",opts_to_apply=()))


def ast_signature(ast:UOp) -> dict[str, object]:
  topo=ast.toposort()
  roots={}
  for store in (x for x in topo if x.op is Ops.STORE):
    try: root=_root_param(store.src[0])
    except ValueError: continue
    roots[root.arg.slot]=roots.get(root.arg.slot,0)+1
  return {"wmma":sum(x.op is Ops.WMMA for x in topo),"barriers":sum(x.op is Ops.BARRIER for x in topo),
    "ranges":sorted(x.arg[0] for x in topo if x.op is Ops.RANGE),"parameter_store_counts":roots,
    "terminal_store_count":len(ast.src[0].src[0].src) if ast.src and ast.src[0].op is Ops.END else None}


__all__=["ALL_PARTIALS_ASCENDING","LLAMA_STANDALONE_ASCENDING","LLAMA_STANDALONE_NONFINAL_FIRST","ARM_NAMES",
  "SEGMENT_ORDER_BUILDER_PATCH","SegmentDescriptor","ReductionSchedule","build_reduction_schedule","fold_values",
  "cpu_all_partials","cpu_direct_final","direct_output_from_partials","supports_nonfinal_first","build_packed_one_body_ast",
  "direct_final_writeback","emit_ordered_fixup","ast_signature"]
