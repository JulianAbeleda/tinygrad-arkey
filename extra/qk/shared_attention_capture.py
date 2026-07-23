"""Content-addressed compiler and numeric evidence for shared prefill attention."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import hashlib, json, math, re
from typing import Any, Mapping

from tinygrad.uop.ops import (AttentionWMMARole, KernelInfo, Ops, ProgramInfo,
  SharedAttentionCandidateContext, UOp)

CAPTURE_SCHEMA = "tinygrad.shared_attention_compiler_capture.v1"
_RESOURCE_FIELDS = ("vgpr", "sgpr", "lds_bytes", "scratch_bytes", "vgpr_spills", "sgpr_spills", "wavefront_size")

def _sha(data:bytes) -> str: return hashlib.sha256(data).hexdigest()
def _text_sha(value:str) -> str: return _sha(value.encode())
def _is_sha(value:Any) -> bool:
  return isinstance(value,str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
def _canonical(value:Mapping[str,Any]) -> bytes:
  return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()

def _context_json(ctx:SharedAttentionCandidateContext) -> dict[str,Any]:
  return {name:getattr(ctx,name) for name in ctx._fields}

def _context_from_json(row:Mapping[str,Any]) -> SharedAttentionCandidateContext:
  if set(row) != set(SharedAttentionCandidateContext._fields): raise ValueError("candidate_context fields are malformed")
  return SharedAttentionCandidateContext(*(row[name] for name in SharedAttentionCandidateContext._fields)).validate()

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

  def _payload(self) -> dict[str,Any]:
    return {"schema":self.schema,"candidate_context":_context_json(self.candidate_context),
      "canonical_graph_sha256":self.canonical_graph_sha256,"compute_call_count":self.compute_call_count,
      "copy_call_count":self.copy_call_count,"param_ownership":[list(x) for x in self.param_ownership],
      "allocation_elements":list(self.allocation_elements),"allocation_complete":self.allocation_complete,
      "expanded_kv_buffers":self.expanded_kv_buffers,"score_probability_buffers":self.score_probability_buffers,
      "wmma_roles":[[site,role.contraction,role.tile] for site,role in self.wmma_roles],
      "hip_source":self.hip_source,"hip_source_sha256":self.hip_source_sha256,
      "hip_resources":{key:value for key,value in self.hip_resources},"amd_isa_text":self.amd_isa_text,
      "amd_isa_sha256":self.amd_isa_sha256,"loop_count":self.loop_count,"barrier_count":self.barrier_count,
      "static_wmma_count":self.static_wmma_count,"highest_vgpr":self.highest_vgpr,"highest_sgpr":self.highest_sgpr,
      "spill_count":self.spill_count,"scratch_bytes":self.scratch_bytes,"lds_bytes":self.lds_bytes,
      "numeric":{"max_abs":self.numeric_max_abs,"max_rel":self.numeric_max_rel,"rel_l2":self.numeric_rel_l2,
                 "reference_sha256":self.reference_sha256}}

  def with_hash(self) -> SharedAttentionCompilerCapture:
    return replace(self,capture_sha256=_sha(_canonical(self._payload())))

  def validate(self) -> SharedAttentionCompilerCapture:
    self.candidate_context.validate()
    if self.schema != CAPTURE_SCHEMA: raise ValueError("shared attention capture schema mismatch")
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
    if self.loop_count < 1 or self.barrier_count < 1 or self.static_wmma_count != 16 or self.spill_count or self.scratch_bytes:
      raise ValueError("AMD ISA loop/WMMA/resource contract failed")
    sites = tuple(site for site,_ in self.wmma_roles)
    if len(self.wmma_roles) != 16 or sites != tuple(sorted(set(sites))): raise ValueError("WMMA role sites are malformed")
    for role in (role for _,role in self.wmma_roles): role.validate()
    for contraction in ("QK","PV"):
      if sorted(role.tile for _,role in self.wmma_roles if role.contraction == contraction) != list(range(8)):
        raise ValueError(f"shared attention capture requires exactly 8 {contraction} roles")
    if not all(isinstance(x,(int,float)) and math.isfinite(x) and x >= 0 for x in
               (self.numeric_max_abs,self.numeric_max_rel,self.numeric_rel_l2)):
      raise ValueError("numeric metrics are malformed")
    return self

  def to_json(self) -> dict[str,Any]: return {**self._payload(),"capture_sha256":self.capture_sha256}

  @classmethod
  def from_json(cls,row:Mapping[str,Any]) -> SharedAttentionCompilerCapture:
    required = {"schema","candidate_context","canonical_graph_sha256","compute_call_count","copy_call_count","param_ownership",
      "allocation_elements","allocation_complete","expanded_kv_buffers","score_probability_buffers","wmma_roles","hip_source",
      "hip_source_sha256","hip_resources","amd_isa_text","amd_isa_sha256","loop_count","barrier_count","static_wmma_count",
      "highest_vgpr","highest_sgpr","spill_count","scratch_bytes","lds_bytes","numeric","capture_sha256"}
    if set(row) != required: raise ValueError("shared attention capture fields are malformed")
    numeric = row["numeric"]
    if not isinstance(numeric,Mapping) or set(numeric) != {"max_abs","max_rel","rel_l2","reference_sha256"}:
      raise ValueError("shared attention numeric fields are malformed")
    resources = row["hip_resources"]
    if not isinstance(resources,Mapping): raise ValueError("HIP resources are malformed")
    roles = tuple((int(x[0]),AttentionWMMARole(str(x[1]),int(x[2]))) for x in row["wmma_roles"])
    return cls(str(row["schema"]),_context_from_json(row["candidate_context"]),str(row["canonical_graph_sha256"]),
      int(row["compute_call_count"]),int(row["copy_call_count"]),tuple((int(x[0]),int(x[1])) for x in row["param_ownership"]),
      tuple(int(x) for x in row["allocation_elements"]),row["allocation_complete"],int(row["expanded_kv_buffers"]),
      int(row["score_probability_buffers"]),roles,str(row["hip_source"]),str(row["hip_source_sha256"]),
      tuple(sorted((str(k),int(v)) for k,v in resources.items())),str(row["amd_isa_text"]),str(row["amd_isa_sha256"]),
      int(row["loop_count"]),int(row["barrier_count"]),int(row["static_wmma_count"]),int(row["highest_vgpr"]),
      int(row["highest_sgpr"]),int(row["spill_count"]),int(row["scratch_bytes"]),int(row["lds_bytes"]),
      float(numeric["max_abs"]),float(numeric["max_rel"]),float(numeric["rel_l2"]),str(numeric["reference_sha256"]),
      str(row["capture_sha256"])).validate()

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
                                            output:Any, reference:Any) -> SharedAttentionCompilerCapture:
  """Construct evidence only from one scheduled call, two final programs, and numeric arrays."""
  import numpy as np
  calls = tuple(u for u in schedule.src if u.op is Ops.CALL)
  if compute_call not in calls or compute_call.op is not Ops.CALL or not compute_call.src: raise TypeError("compute_call is not owned by schedule")
  compute_sink = compute_call.src[0]
  if compute_sink.op is not Ops.SINK or not isinstance(compute_sink.arg,KernelInfo): raise TypeError("compute_call has no scheduled KernelInfo")
  context = compute_sink.arg.candidate_context
  if not isinstance(context,SharedAttentionCandidateContext): raise ValueError("compute call has no shared attention candidate context")
  context.validate()
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
  capture = SharedAttentionCompilerCapture(CAPTURE_SCHEMA,context,compute_sink.replace(arg=None).key.hex(),compute_calls,copy_calls,
    params,buffers,True,expanded_count,score_count,isa_info.wmma_roles.sites,hip_source,_text_sha(hip_source),hip_resources,
    isa_text,_text_sha(isa_text),max(1,isa_text.lower().count("s_cbranch")),isa_text.lower().count("s_barrier"),
    len(isa_info.wmma_roles.sites),_highest_register(isa_text,"v"),_highest_register(isa_text,"s"),
    sum("spill" in str(u.arg).lower() for u in isa_linear.src),sum("scratch" in str(u.arg).lower() for u in isa_linear.src),lds_bytes,
    float(diff.max(initial=0)),float((diff/denom).max(initial=0)),float(np.linalg.norm(diff)/(np.linalg.norm(ref)+1e-12)),_sha(ref_bytes))
  return capture.with_hash().validate()

__all__ = ["CAPTURE_SCHEMA","SharedAttentionCompilerCapture","build_shared_attention_compiler_capture"]
