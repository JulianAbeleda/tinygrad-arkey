"""Content-addressed compiler and numeric evidence for shared prefill attention."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import hashlib, json, math, re
from typing import Any, Mapping

from tinygrad.uop.ops import (AttentionWMMARole, KernelInfo, Ops, ProgramInfo,
  SharedAttentionCandidateContext, StateHandle, UOp)
from extra.qk.attention_harness_common import content_sha as _sha

CAPTURE_SCHEMA = "tinygrad.shared_attention_compiler_capture.v2"
ACC_SLICE_CAPTURE_SCHEMA = "tinygrad.shared_attention_compiler_capture.acc_slice_v3"
ACC_SLICE_PASS_SCHEMA = "tinygrad.shared_attention_acc_slice_pass.v1"
PHASE_CAPTURE_SCHEMA = "tinygrad.shared_attention_compiler_capture.phase_v4"
PHASE_PLAN_SCHEMA = "tinygrad.shared_attention_phase_plan.v1"
_RESOURCE_FIELDS = ("vgpr", "sgpr", "lds_bytes", "scratch_bytes", "vgpr_spills", "sgpr_spills", "wavefront_size")

def _text_sha(value:str) -> str: return _sha(value.encode())
def _is_sha(value:Any) -> bool:
  return isinstance(value,str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
def _canonical(value:Mapping[str,Any]) -> bytes:
  return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()

def _context_json(ctx:SharedAttentionCandidateContext, *, include_slice:bool=True) -> dict[str,Any]:
  names=ctx._fields if include_slice else tuple(x for x in ctx._fields if x != "acc_blocks")
  return {name:getattr(ctx,name) for name in names}

def _context_from_json(row:Mapping[str,Any]) -> SharedAttentionCandidateContext:
  full=set(SharedAttentionCandidateContext._fields); legacy=full-{"acc_blocks"}; old=full|{"output_block_base"}
  if frozenset(row) not in {frozenset(full),frozenset(legacy),frozenset(old)}:
    raise ValueError("candidate_context fields are malformed")
  values={**row}
  values.setdefault("acc_blocks",8)
  ret=SharedAttentionCandidateContext(*(values[name] for name in SharedAttentionCandidateContext._fields))
  _validate_candidate_context(ret)
  return ret

def _validate_candidate_context(ctx:SharedAttentionCandidateContext) -> None:
  # Keep evidence compatible while legacy compiler contexts move slice base to
  # pass metadata. Do not depend on the removed output_block_base attribute.
  if not isinstance(ctx,SharedAttentionCandidateContext): raise TypeError("shared attention candidate context is malformed")
  if not isinstance(ctx.profile,str) or not ctx.profile or ctx.strategy not in {"FULL_RESIDENT_OVERLAY","BOUNDED_PACKED_TILES"}:
    raise ValueError("invalid shared attention profile/strategy")
  if not all(isinstance(x,int) and not isinstance(x,bool) and x > 0 for x in
             (ctx.q_tokens,ctx.kv_tokens,ctx.hq,ctx.hkv,ctx.hd,ctx.acc_blocks)) or ctx.start_pos < 0:
    raise ValueError("invalid shared attention geometry")
  if ctx.hd != 128 or ctx.hq % ctx.hkv or ctx.acc_blocks not in {4,8}:
    raise ValueError("invalid shared attention GQA or accumulator ownership")

@dataclass(frozen=True)
class SharedAttentionStateHandleOwnership:
  region: str
  producer_phase_id: str
  consumer_phase_id: str
  boundary_ordinal: int
  generation: int
  block: int
  block_count: int
  lane: int
  lane_count: int
  lds_offset: int
  storage_slot: int
  storage_elements: int
  lane_stride: int
  element_offset: int
  dtype: str
  itemsize: int
  shape: tuple[int,...]

  @classmethod
  def from_state_handle(cls, handle:StateHandle, *, block:int, block_count:int, lane:int, lane_count:int,
                        lds_offset:int) -> SharedAttentionStateHandleOwnership:
    handle.validate()
    if handle.storage is None: raise ValueError("captured StateHandle must own local storage")
    return cls(handle.region.name,handle.boundary.publish_phase,handle.boundary.reload_phase,handle.boundary.ordinal,
               handle.generation,block,block_count,lane,lane_count,lds_offset,int(handle.storage.arg),handle.storage.ptrdtype.size,
               handle.lane_stride,handle.element_offset,str(handle.region.dtype),
               handle.region.dtype.itemsize,(handle.region.lanes,)).validate()

  def validate(self) -> SharedAttentionStateHandleOwnership:
    for name in (self.region,self.producer_phase_id,self.consumer_phase_id):
      if not isinstance(name,str) or not re.fullmatch(r"[A-Za-z0-9_.:-]+",name):
        raise ValueError("StateHandle region/phase ID is malformed")
    if not isinstance(self.dtype,str) or not self.dtype: raise ValueError("StateHandle dtype is malformed")
    if any(not isinstance(x,int) or isinstance(x,bool) for x in
            (self.boundary_ordinal,self.generation,self.block,self.block_count,self.lane,self.lane_count,
            self.lds_offset,self.storage_slot,self.storage_elements,self.lane_stride,self.element_offset,self.itemsize)):
      raise ValueError("StateHandle ownership fields are malformed")
    if self.boundary_ordinal < 0 or self.generation < 0: raise ValueError("StateHandle boundary/generation is malformed")
    if self.block_count <= 0 or not 0 <= self.block < self.block_count or self.lane_count <= 0 or not 0 <= self.lane < self.lane_count:
      raise ValueError("StateHandle logical ownership is out of range")
    if self.itemsize <= 0 or self.lds_offset < 0 or self.lds_offset % self.itemsize:
      raise ValueError("StateHandle LDS offset is invalid for dtype")
    if any(not isinstance(x,int) or isinstance(x,bool) or x <= 0 for x in self.shape):
      raise ValueError("StateHandle shape is malformed")
    if self.storage_slot < 0 or self.storage_elements <= 0 or self.lane_stride < self.shape[0] or self.element_offset < 0 or \
       self.element_offset+self.shape[0] > self.lane_stride:
      raise ValueError("StateHandle storage/lane contract is malformed")
    elements=math.prod(self.shape)
    if self.lds_offset+elements*self.itemsize > self.storage_elements*self.itemsize:
      raise ValueError("StateHandle ownership exceeds local storage")
    return self

  def to_json(self) -> dict[str,Any]:
    return {"region":self.region,"producer_phase_id":self.producer_phase_id,"consumer_phase_id":self.consumer_phase_id,
            "boundary_ordinal":self.boundary_ordinal,"generation":self.generation,"block":self.block,
            "block_count":self.block_count,"lane":self.lane,"lane_count":self.lane_count,
            "lds_offset":self.lds_offset,"storage_slot":self.storage_slot,"storage_elements":self.storage_elements,
            "lane_stride":self.lane_stride,"element_offset":self.element_offset,"dtype":self.dtype,
            "itemsize":self.itemsize,"shape":list(self.shape)}

  @classmethod
  def from_json(cls,row:Mapping[str,Any]) -> SharedAttentionStateHandleOwnership:
    required={"region","producer_phase_id","consumer_phase_id","boundary_ordinal","generation","block","block_count",
              "lane","lane_count","lds_offset","storage_slot","storage_elements","lane_stride","element_offset",
              "dtype","itemsize","shape"}
    if not isinstance(row,Mapping) or set(row) != required: raise ValueError("StateHandle fields are malformed")
    if not isinstance(row["shape"],list): raise ValueError("StateHandle shape is malformed")
    return cls(str(row["region"]),str(row["producer_phase_id"]),str(row["consumer_phase_id"]),
               int(row["boundary_ordinal"]),int(row["generation"]),int(row["block"]),int(row["block_count"]),
               int(row["lane"]),int(row["lane_count"]),int(row["lds_offset"]),int(row["storage_slot"]),
               int(row["storage_elements"]),int(row["lane_stride"]),int(row["element_offset"]),str(row["dtype"]),int(row["itemsize"]),
               tuple(int(x) for x in row["shape"])).validate()

@dataclass(frozen=True)
class SharedAttentionPhasePlan:
  schema: str
  phase_ids: tuple[str,...]
  logical_graph_sha256: str
  state_handles: tuple[SharedAttentionStateHandleOwnership,...]

  def validate(self) -> SharedAttentionPhasePlan:
    if self.schema != PHASE_PLAN_SCHEMA: raise ValueError("shared attention phase-plan schema mismatch")
    if len(self.phase_ids) < 2 or len(set(self.phase_ids)) != len(self.phase_ids) or any(
        not isinstance(x,str) or not re.fullmatch(r"[A-Za-z0-9_.:-]+",x) for x in self.phase_ids):
      raise ValueError("compiler phase IDs are malformed or duplicated")
    if not _is_sha(self.logical_graph_sha256): raise ValueError("phase-plan logical graph hash is malformed")
    if not self.state_handles: raise ValueError("phase plan has no StateHandle ownership")
    order={phase_id:index for index,phase_id in enumerate(self.phase_ids)}
    groups:dict[tuple[str,str,str,int,int],list[SharedAttentionStateHandleOwnership]] = {}
    for handle in self.state_handles:
      handle.validate()
      if handle.producer_phase_id not in order or handle.consumer_phase_id not in order or \
         order[handle.producer_phase_id] >= order[handle.consumer_phase_id]:
        raise ValueError("StateHandle producer/consumer phase mismatch")
      groups.setdefault((handle.region,handle.producer_phase_id,handle.consumer_phase_id,
                         handle.boundary_ordinal,handle.generation),[]).append(handle)
    for handles in groups.values():
      contracts={(x.block_count,x.lane_count,x.storage_slot,x.storage_elements,x.lane_stride,x.element_offset,
                  x.dtype,x.itemsize,x.shape) for x in handles}
      if len(contracts) != 1: raise ValueError("StateHandle shape/ownership contract mismatch")
      block_count,lane_count,*_=next(iter(contracts)); pairs=[(x.block,x.lane) for x in handles]
      if len(pairs) != len(set(pairs)): raise ValueError("StateHandle ownership overlap")
      expected={(block,lane) for block in range(block_count) for lane in range(lane_count)}
      if set(pairs) != expected: raise ValueError("StateHandle ownership gap")
      intervals=sorted((x.lds_offset,x.lds_offset+math.prod(x.shape)*x.itemsize) for x in handles)
      if any(right[0] < left[1] for left,right in zip(intervals,intervals[1:])):
        raise ValueError("StateHandle LDS storage overlap")
    return self

  def to_json(self) -> dict[str,Any]:
    return {"schema":self.schema,"phase_ids":list(self.phase_ids),"logical_graph_sha256":self.logical_graph_sha256,
            "state_handles":[x.to_json() for x in self.state_handles]}

  @classmethod
  def from_json(cls,row:Mapping[str,Any]) -> SharedAttentionPhasePlan:
    if not isinstance(row,Mapping) or set(row) != {"schema","phase_ids","logical_graph_sha256","state_handles"}:
      raise ValueError("shared attention phase-plan fields are malformed")
    if not isinstance(row["phase_ids"],list) or not isinstance(row["state_handles"],list):
      raise ValueError("shared attention phase-plan lists are malformed")
    return cls(str(row["schema"]),tuple(str(x) for x in row["phase_ids"]),str(row["logical_graph_sha256"]),
               tuple(SharedAttentionStateHandleOwnership.from_json(x) for x in row["state_handles"])).validate()

@dataclass(frozen=True)
class SharedAttentionAccSlicePass:
  schema: str
  output_block_base: int
  acc_blocks: int
  qk_recomputed: bool
  logical_graph_sha256: str

  def validate(self) -> SharedAttentionAccSlicePass:
    if self.schema != ACC_SLICE_PASS_SCHEMA: raise ValueError("shared attention accumulator-slice pass schema mismatch")
    if (self.output_block_base,self.acc_blocks) not in {(0,4),(4,4)}:
      raise ValueError("shared attention accumulator-slice ownership is not an exact half")
    if self.qk_recomputed is not True: raise ValueError("shared attention accumulator-slice pass must declare QK recomputation")
    if not _is_sha(self.logical_graph_sha256): raise ValueError("shared attention accumulator-slice logical graph hash is malformed")
    return self

  def to_json(self) -> dict[str,Any]:
    return {"schema":self.schema,"output_block_base":self.output_block_base,"acc_blocks":self.acc_blocks,
            "qk_recomputed":self.qk_recomputed,"logical_graph_sha256":self.logical_graph_sha256}

  @classmethod
  def from_json(cls,row:Mapping[str,Any]) -> SharedAttentionAccSlicePass:
    if set(row) != {"schema","output_block_base","acc_blocks","qk_recomputed","logical_graph_sha256"}:
      raise ValueError("shared attention accumulator-slice pass fields are malformed")
    if not isinstance(row["qk_recomputed"],bool): raise ValueError("shared attention accumulator-slice recomputation flag is malformed")
    return cls(str(row["schema"]),int(row["output_block_base"]),int(row["acc_blocks"]),
               row["qk_recomputed"],str(row["logical_graph_sha256"])).validate()

@dataclass(frozen=True)
class SharedAttentionSynchronization:
  scope: str
  workgroup_waves: int
  lds_wait_sites: int
  workgroup_barriers: int

  def validate(self) -> SharedAttentionSynchronization:
    if self.scope not in {"wave", "workgroup"}: raise ValueError("shared attention synchronization scope is invalid")
    if not all(isinstance(x,int) and not isinstance(x,bool) and x >= 0 for x in
               (self.workgroup_waves,self.lds_wait_sites,self.workgroup_barriers)):
      raise ValueError("shared attention synchronization counts are malformed")
    expected = ("wave",1,1,0) if self.scope == "wave" else ("workgroup",self.workgroup_waves,0,1)
    if (self.scope,self.workgroup_waves,self.lds_wait_sites,self.workgroup_barriers) != expected or self.workgroup_waves < 1:
      raise ValueError("shared attention synchronization contract is not exact")
    return self

  def to_json(self) -> dict[str,Any]:
    return {"scope":self.scope,"workgroup_waves":self.workgroup_waves,
            "lds_wait_sites":self.lds_wait_sites,"workgroup_barriers":self.workgroup_barriers}

  @classmethod
  def from_json(cls,row:Mapping[str,Any]) -> SharedAttentionSynchronization:
    if set(row) != {"scope","workgroup_waves","lds_wait_sites","workgroup_barriers"}:
      raise ValueError("shared attention synchronization fields are malformed")
    return cls(str(row["scope"]),int(row["workgroup_waves"]),int(row["lds_wait_sites"]),
               int(row["workgroup_barriers"])).validate()

def _is_wave_lds_wait(line:str) -> bool:
  line=line.lower()
  if "s_waitcnt" not in line: return False
  if all(x in line for x in ("vmcnt(63)","lgkmcnt(0)","expcnt(7)")): return True
  match=re.search(r"\bs_waitcnt\((\d*)\)",line)
  if match is None: return False
  simm16=int(match.group(1) or 0)
  return simm16 == (63 << 10) | 7

def _ordered_lds_wait_sites(text:str, *, source:bool) -> int:
  if source:
    stores=[m.start() for m in re.finditer(r"\*\(buf0\+[^;\n]+\)\s*=",text)]
    loads=[m.start() for m in re.finditer(r"half\s+val\d+\s*=\s*\(\*\(buf0",text)]
    waits=[m.start() for m in re.finditer(r"__builtin_amdgcn_s_waitcnt\([^)]*\);",text)]
  else:
    lines=text.splitlines(); stores=[i for i,x in enumerate(lines) if re.search(r"\bds_store",x,re.I)]
    loads=[i for i,x in enumerate(lines) if re.search(r"\bds_load",x,re.I)]
    waits=[i for i,x in enumerate(lines) if _is_wave_lds_wait(x)]
  if not stores or not loads: raise ValueError("shared attention LDS publication/reload markers are missing")
  last_store=max(stores); first_reload=min((x for x in loads if x > last_store),default=-1)
  if first_reload < 0: raise ValueError("shared attention LDS reload does not follow publication")
  return sum(last_store < x < first_reload for x in waits)

def _derive_synchronization(local_size:tuple[int,...]|None, wavefront_size:int, hip_source:str,
                            amd_isa_text:str) -> SharedAttentionSynchronization:
  if local_size is None or not local_size or any(not isinstance(x,int) or x <= 0 for x in local_size):
    raise ValueError("shared attention synchronization requires static local size")
  threads=math.prod(local_size)
  if wavefront_size <= 0 or threads % wavefront_size: raise ValueError("workgroup is not a whole number of waves")
  waves=threads//wavefront_size
  source_barriers=hip_source.count("__builtin_amdgcn_s_barrier")
  isa_barriers=len(re.findall(r"\bs_barrier\b",amd_isa_text,re.I))
  if source_barriers != isa_barriers: raise ValueError("HIP/ISA workgroup barrier counts differ")
  if waves == 1:
    source_waits=_ordered_lds_wait_sites(hip_source,source=True)
    isa_waits=_ordered_lds_wait_sites(amd_isa_text,source=False)
    if source_barriers or source_waits != 1 or isa_waits != 1:
      raise ValueError(f"single-wave LDS wait is missing, duplicated, or misordered: source_waits={source_waits}, "
                       f"isa_waits={isa_waits}, source_barriers={source_barriers}")
    return SharedAttentionSynchronization("wave",1,1,0).validate()
  if source_barriers != 1: raise ValueError("multi-wave workgroup requires one workgroup barrier")
  return SharedAttentionSynchronization("workgroup",waves,0,1).validate()

@dataclass(frozen=True)
class SharedAttentionCompilerCapture:
  schema: str
  candidate_context: SharedAttentionCandidateContext
  canonical_graph_sha256: str
  compute_call_count: int
  copy_call_count: int
  param_ownership: tuple[tuple[int,int], ...]
  allocation_elements: tuple[int, ...]
  allocation_complete: bool
  expanded_kv_buffers: int
  score_probability_buffers: int
  wmma_roles: tuple[tuple[int,AttentionWMMARole], ...]
  hip_source: str
  hip_source_sha256: str
  hip_resources: tuple[tuple[str,int], ...]
  amd_isa_text: str
  amd_isa_sha256: str
  loop_count: int
  barrier_count: int
  synchronization: SharedAttentionSynchronization
  static_wmma_count: int
  highest_vgpr: int
  highest_sgpr: int
  spill_count: int
  scratch_bytes: int
  lds_bytes: int
  numeric_max_abs: float
  numeric_max_rel: float
  numeric_rel_l2: float
  reference_sha256: str
  capture_sha256: str = ""
  acc_slice_pass: SharedAttentionAccSlicePass|None = None
  phase_plan: SharedAttentionPhasePlan|None = None

  def _payload(self) -> dict[str,Any]:
    payload = {"schema":self.schema,"candidate_context":_context_json(self.candidate_context,include_slice=self.schema!=CAPTURE_SCHEMA),
      "canonical_graph_sha256":self.canonical_graph_sha256,"compute_call_count":self.compute_call_count,
      "copy_call_count":self.copy_call_count,"param_ownership":[list(x) for x in self.param_ownership],
      "allocation_elements":list(self.allocation_elements),"allocation_complete":self.allocation_complete,
      "expanded_kv_buffers":self.expanded_kv_buffers,"score_probability_buffers":self.score_probability_buffers,
      "wmma_roles":[[site,role.contraction,role.tile] for site,role in self.wmma_roles],
      "hip_source":self.hip_source,"hip_source_sha256":self.hip_source_sha256,
      "hip_resources":{key:value for key,value in self.hip_resources},"amd_isa_text":self.amd_isa_text,
      "amd_isa_sha256":self.amd_isa_sha256,"loop_count":self.loop_count,"barrier_count":self.barrier_count,
      "synchronization":self.synchronization.to_json(),
      "static_wmma_count":self.static_wmma_count,"highest_vgpr":self.highest_vgpr,"highest_sgpr":self.highest_sgpr,
      "spill_count":self.spill_count,"scratch_bytes":self.scratch_bytes,"lds_bytes":self.lds_bytes,
      "numeric":{"max_abs":self.numeric_max_abs,"max_rel":self.numeric_max_rel,"rel_l2":self.numeric_rel_l2,
                 "reference_sha256":self.reference_sha256}}
    if self.schema == ACC_SLICE_CAPTURE_SCHEMA:
      payload["acc_slice_pass"] = self.acc_slice_pass.to_json() if self.acc_slice_pass is not None else None
    if self.schema == PHASE_CAPTURE_SCHEMA:
      payload["phase_plan"] = self.phase_plan.to_json() if self.phase_plan is not None else None
    return payload

  def with_hash(self) -> SharedAttentionCompilerCapture:
    return replace(self,capture_sha256=_sha(_canonical(self._payload())))

  def validate(self) -> SharedAttentionCompilerCapture:
    _validate_candidate_context(self.candidate_context)
    if self.schema not in {CAPTURE_SCHEMA,ACC_SLICE_CAPTURE_SCHEMA,PHASE_CAPTURE_SCHEMA}:
      raise ValueError("shared attention capture schema mismatch")
    if not all(_is_sha(x) for x in (self.canonical_graph_sha256,self.hip_source_sha256,self.amd_isa_sha256,
                                    self.reference_sha256,self.capture_sha256)):
      raise ValueError("shared attention capture hash is malformed")
    if self.hip_source_sha256 != _text_sha(self.hip_source) or self.amd_isa_sha256 != _text_sha(self.amd_isa_text):
      raise ValueError("shared attention source/ISA hash mismatch")
    if self.capture_sha256 != _sha(_canonical(self._payload())): raise ValueError("shared attention capture hash mismatch")
    ctx = self.candidate_context
    expected = ((0,ctx.hq*ctx.q_tokens*ctx.hd),(1,ctx.hq*ctx.q_tokens*ctx.hd),
                (2,ctx.hkv*ctx.kv_tokens*ctx.hd),(3,ctx.hkv*ctx.kv_tokens*ctx.hd))
    if self.compute_call_count != 1 or self.copy_call_count < 0 or self.param_ownership != expected:
      raise ValueError("shared attention call or PARAM ownership is not exact")
    if not self.allocation_complete or self.expanded_kv_buffers or self.score_probability_buffers:
      raise ValueError("shared attention allocation census is incomplete or materialized")
    if tuple(sorted(self.hip_resources)) != tuple(sorted((key,dict(self.hip_resources)[key]) for key in _RESOURCE_FIELDS)):
      raise ValueError("HIP resource metadata fields are malformed")
    if any(not isinstance(v,int) or v < 0 for _,v in self.hip_resources): raise ValueError("HIP resources must be non-negative integers")
    if any(not isinstance(x,int) or x < 0 for x in (self.copy_call_count,self.loop_count,self.barrier_count,self.static_wmma_count,
      self.highest_vgpr,self.highest_sgpr,self.spill_count,self.scratch_bytes,self.lds_bytes)):
      raise ValueError("AMD ISA counts/resources are malformed")
    self.synchronization.validate()
    if self.barrier_count != self.synchronization.workgroup_barriers:
      raise ValueError("captured barrier count differs from synchronization contract")
    expected_wmma = 12 if self.schema == ACC_SLICE_CAPTURE_SCHEMA else 16
    if self.loop_count < 1 or self.static_wmma_count != expected_wmma or self.spill_count or self.scratch_bytes:
      raise ValueError("AMD ISA loop/WMMA/resource contract failed")
    wavefront=dict(self.hip_resources)["wavefront_size"]
    derived=_derive_synchronization((self.synchronization.workgroup_waves*wavefront,),wavefront,self.hip_source,self.amd_isa_text)
    if derived != self.synchronization: raise ValueError("shared attention synchronization evidence is not reproducible")
    if self.schema == CAPTURE_SCHEMA:
      if self.acc_slice_pass is not None or self.phase_plan is not None or ctx.acc_blocks != 8:
        raise ValueError("v2 shared attention capture cannot claim accumulator-slice ownership")
      pv_tiles=list(range(8))
    elif self.schema == ACC_SLICE_CAPTURE_SCHEMA:
      if self.acc_slice_pass is None or self.phase_plan is not None: raise ValueError("accumulator-slice capture metadata is missing")
      self.acc_slice_pass.validate()
      if ctx.acc_blocks != self.acc_slice_pass.acc_blocks:
        raise ValueError("capture context and accumulator-slice ownership differ")
      pv_tiles=list(range(4))
    else:
      if self.acc_slice_pass is not None or self.phase_plan is None:
        raise ValueError("phase capture metadata is missing or mixed")
      self.phase_plan.validate()
      if self.phase_plan.logical_graph_sha256 != self.canonical_graph_sha256:
        raise ValueError("phase plan and capture logical graphs differ")
      if ctx.acc_blocks != 8:
        raise ValueError("phase capture must own the complete output")
      pv_tiles=list(range(8))
    sites = tuple(site for site,_ in self.wmma_roles)
    if len(self.wmma_roles) != expected_wmma or sites != tuple(sorted(set(sites))): raise ValueError("WMMA role sites are malformed")
    for role in (role for _,role in self.wmma_roles): role.validate()
    if sorted(role.tile for _,role in self.wmma_roles if role.contraction == "QK") != list(range(8)):
      raise ValueError("shared attention capture requires exactly 8 QK roles")
    if sorted(role.tile for _,role in self.wmma_roles if role.contraction == "PV") != pv_tiles:
      raise ValueError(f"shared attention capture requires exactly {len(pv_tiles)} PV roles")
    if not all(isinstance(x,(int,float)) and math.isfinite(x) and x >= 0 for x in
               (self.numeric_max_abs,self.numeric_max_rel,self.numeric_rel_l2)):
      raise ValueError("numeric metrics are malformed")
    return self

  def to_json(self) -> dict[str,Any]: return {**self._payload(),"capture_sha256":self.capture_sha256}

  @classmethod
  def from_json(cls,row:Mapping[str,Any]) -> SharedAttentionCompilerCapture:
    required = {"schema","candidate_context","canonical_graph_sha256","compute_call_count","copy_call_count","param_ownership",
      "allocation_elements","allocation_complete","expanded_kv_buffers","score_probability_buffers","wmma_roles","hip_source",
      "hip_source_sha256","hip_resources","amd_isa_text","amd_isa_sha256","loop_count","barrier_count","synchronization","static_wmma_count",
      "highest_vgpr","highest_sgpr","spill_count","scratch_bytes","lds_bytes","numeric","capture_sha256"}
    if row.get("schema") == ACC_SLICE_CAPTURE_SCHEMA: required.add("acc_slice_pass")
    if row.get("schema") == PHASE_CAPTURE_SCHEMA: required.add("phase_plan")
    if set(row) != required: raise ValueError("shared attention capture fields are malformed")
    numeric = row["numeric"]
    if not isinstance(numeric,Mapping) or set(numeric) != {"max_abs","max_rel","rel_l2","reference_sha256"}:
      raise ValueError("shared attention numeric fields are malformed")
    resources = row["hip_resources"]
    if not isinstance(resources,Mapping): raise ValueError("HIP resources are malformed")
    roles = tuple((int(x[0]),AttentionWMMARole(str(x[1]),int(x[2]))) for x in row["wmma_roles"])
    pass_row=row.get("acc_slice_pass")
    acc_slice_pass=SharedAttentionAccSlicePass.from_json(pass_row) if isinstance(pass_row,Mapping) else None
    phase_row=row.get("phase_plan")
    phase_plan=SharedAttentionPhasePlan.from_json(phase_row) if isinstance(phase_row,Mapping) else None
    return cls(str(row["schema"]),_context_from_json(row["candidate_context"]),str(row["canonical_graph_sha256"]),
      int(row["compute_call_count"]),int(row["copy_call_count"]),tuple((int(x[0]),int(x[1])) for x in row["param_ownership"]),
      tuple(int(x) for x in row["allocation_elements"]),row["allocation_complete"],int(row["expanded_kv_buffers"]),
      int(row["score_probability_buffers"]),roles,str(row["hip_source"]),str(row["hip_source_sha256"]),
      tuple(sorted((str(k),int(v)) for k,v in resources.items())),str(row["amd_isa_text"]),str(row["amd_isa_sha256"]),
      int(row["loop_count"]),int(row["barrier_count"]),SharedAttentionSynchronization.from_json(row["synchronization"]),
      int(row["static_wmma_count"]),int(row["highest_vgpr"]),
      int(row["highest_sgpr"]),int(row["spill_count"]),int(row["scratch_bytes"]),int(row["lds_bytes"]),
      float(numeric["max_abs"]),float(numeric["max_rel"]),float(numeric["rel_l2"]),str(numeric["reference_sha256"]),
      str(row["capture_sha256"]),acc_slice_pass,phase_plan).validate()

def _program_parts(program:UOp) -> tuple[ProgramInfo,UOp,str,bytes|None]:
  if program.op is not Ops.PROGRAM or not isinstance(program.arg,ProgramInfo): raise TypeError("capture requires a final PROGRAM")
  linear = next((u for u in program.src if u.op is Ops.LINEAR),None)
  if linear is None: raise ValueError("capture PROGRAM is missing final LINEAR")
  source = next((u.arg for u in program.src if u.op is Ops.SOURCE),"")
  binary = next((u.arg for u in program.src if u.op is Ops.BINARY),None)
  return program.arg,linear,source,binary

def _highest_register(text:str,prefix:str) -> int:
  values = [int(x) for x in re.findall(rf"(?<![a-zA-Z0-9_]){prefix}(?:\[(\d+)(?::(\d+))?\]|(\d+))",text)
            for x in x if x]
  return max(values,default=-1)+1

def build_shared_attention_compiler_capture(*, schedule:UOp, compute_call:UOp, hip_program:UOp, amd_isa_program:UOp,
                                            output:Any, reference:Any, output_block_base:int=0) -> SharedAttentionCompilerCapture:
  """Construct evidence only from one scheduled call, two final programs, and numeric arrays."""
  import numpy as np
  calls = tuple(u for u in schedule.src if u.op is Ops.CALL)
  if compute_call not in calls or compute_call.op is not Ops.CALL or not compute_call.src: raise TypeError("compute_call is not owned by schedule")
  compute_sink = compute_call.src[0]
  if compute_sink.op is not Ops.SINK or not isinstance(compute_sink.arg,KernelInfo): raise TypeError("compute_call has no scheduled KernelInfo")
  context = compute_sink.arg.candidate_context
  if not isinstance(context,SharedAttentionCandidateContext): raise ValueError("compute call has no shared attention candidate context")
  _validate_candidate_context(context)
  copy_calls = sum(any(x.op in {Ops.COPY,Ops.SLICE} for x in call.src[0].toposort()) for call in calls)
  compute_calls = len(calls)-copy_calls
  params = tuple(sorted((u.arg.slot,u.ptrdtype.size) for u in compute_sink.toposort() if u.op is Ops.PARAM))
  buffers = tuple(sorted(int(u.arg) for u in schedule.toposort() if u.op is Ops.BUFFER and isinstance(u.arg,int)))
  expected_counts = Counter(size for _,size in ((0,context.hq*context.q_tokens*context.hd),(1,context.hq*context.q_tokens*context.hd),
    (2,context.hkv*context.kv_tokens*context.hd),(3,context.hkv*context.kv_tokens*context.hd)))
  actual_counts = Counter(buffers)
  expanded = context.hq*context.kv_tokens*context.hd
  expanded_count = max(0,actual_counts[expanded]-expected_counts[expanded])
  score = context.hq*context.q_tokens*context.kv_tokens
  score_count = max(0,actual_counts[score]-expected_counts[score])
  hip_info,hip_linear,hip_source,hip_binary = _program_parts(hip_program)
  isa_info,isa_linear,_,_ = _program_parts(amd_isa_program)
  if hip_info.candidate_context != context or isa_info.candidate_context != context:
    raise ValueError("scheduled and final PROGRAM candidate contexts differ")
  if sorted(role for _,role in hip_info.wmma_roles.sites) != sorted(role for _,role in isa_info.wmma_roles.sites):
    raise ValueError("HIP and AMD ISA WMMA role sets differ")
  if not hip_source or not isinstance(hip_binary,bytes): raise ValueError("HIP PROGRAM lacks final source/binary")
  from extra.qk.mmq_compile_evidence import parse_amdgpu_metadata
  metadata = parse_amdgpu_metadata(hip_binary)
  hip_resources = tuple(sorted((key,int(metadata[key])) for key in _RESOURCE_FIELDS))
  from tinygrad.helpers import Target
  from tinygrad.renderer.isa.amd import AMDISARenderer
  isa_text = AMDISARenderer(Target.parse("AMD:ISA:gfx1100")).asm_str(list(isa_linear.src),isa_info.name)
  lowered = amd_isa_program.src[0].toposort()
  lds_bytes = max(int(metadata["lds_bytes"]),sum(u.ptrdtype.size*u.ptrdtype.base.itemsize for u in lowered if u.op is Ops.DEFINE_LOCAL))
  out,ref = np.asarray(output,dtype=np.float32),np.asarray(reference,dtype=np.float32)
  if out.shape != ref.shape or out.size != context.hq*context.q_tokens*context.hd: raise ValueError("numeric output/reference shape mismatch")
  diff = np.abs(out-ref); denom = np.maximum(np.abs(ref),1e-3)
  ref_bytes = str(ref.shape).encode()+str(ref.dtype).encode()+ref.tobytes()
  schema, acc_slice_pass = CAPTURE_SCHEMA, None
  if context.acc_blocks == 4:
    logical_context = _context_json(context)
    logical_context["acc_blocks"] = 8
    logical_graph_sha256 = _sha(_canonical({"candidate_context":logical_context,"param_ownership":[list(x) for x in params],
      "qk_tiles":list(range(8)),"pv_output_tiles":list(range(8))}))
    acc_slice_pass = SharedAttentionAccSlicePass(ACC_SLICE_PASS_SCHEMA,output_block_base,context.acc_blocks,
      True,logical_graph_sha256)
    schema = ACC_SLICE_CAPTURE_SCHEMA
  capture = SharedAttentionCompilerCapture(schema,context,compute_sink.replace(arg=None).key.hex(),compute_calls,copy_calls,
    params,buffers,True,expanded_count,score_count,isa_info.wmma_roles.sites,hip_source,_text_sha(hip_source),hip_resources,
    isa_text,_text_sha(isa_text),max(1,isa_text.lower().count("s_cbranch")),isa_text.lower().count("s_barrier"),
    _derive_synchronization(hip_info.local_size,dict(hip_resources)["wavefront_size"],hip_source,isa_text),
    len(isa_info.wmma_roles.sites),_highest_register(isa_text,"v"),_highest_register(isa_text,"s"),
    sum("spill" in str(u.arg).lower() for u in isa_linear.src),sum("scratch" in str(u.arg).lower() for u in isa_linear.src),lds_bytes,
    float(diff.max(initial=0)),float((diff/denom).max(initial=0)),float(np.linalg.norm(diff)/(np.linalg.norm(ref)+1e-12)),_sha(ref_bytes),
    acc_slice_pass=acc_slice_pass)
  return capture.with_hash().validate()

__all__ = ["CAPTURE_SCHEMA","ACC_SLICE_CAPTURE_SCHEMA","ACC_SLICE_PASS_SCHEMA","PHASE_CAPTURE_SCHEMA","PHASE_PLAN_SCHEMA",
           "SharedAttentionStateHandleOwnership","SharedAttentionPhasePlan","SharedAttentionAccSlicePass",
           "SharedAttentionSynchronization","SharedAttentionCompilerCapture","build_shared_attention_compiler_capture"]
