from dataclasses import FrozenInstanceError, replace
import hashlib

import pytest

from extra.qk.shared_attention_capture import CAPTURE_SCHEMA, SharedAttentionCompilerCapture
from extra.qk.shared_attention_evidence import shared_attention_proof_artifact
from tinygrad.uop.ops import AttentionWMMARole, SharedAttentionCandidateContext

def _sha(value:str) -> str: return hashlib.sha256(value.encode()).hexdigest()

def capture(profile,strategy,start,hq):
  qt,kv=16,start+16
  ctx=SharedAttentionCandidateContext(profile,strategy,qt,kv,start,hq,8,128,True)
  roles=tuple((i,AttentionWMMARole("QK" if i<8 else "PV",i if i<8 else i-8)) for i in range(16))
  hip="// final HIP source\n"; isa="loop:\n"+"s_cbranch_scc1 loop\n"+"s_barrier\n"+("v_wmma_f32_16x16x16_f16 v0, v1, v2\n"*16)
  resources=tuple(sorted({"vgpr":64,"sgpr":16,"lds_bytes":512,"scratch_bytes":0,"vgpr_spills":0,
                          "sgpr_spills":0,"wavefront_size":32}.items()))
  sizes=(hq*qt*128,hq*qt*128,8*kv*128,8*kv*128)
  value=SharedAttentionCompilerCapture(CAPTURE_SCHEMA,ctx,"a"*64,1,0,tuple(enumerate(sizes)),tuple(sorted(sizes)),True,0,0,
    roles,hip,_sha(hip),resources,isa,_sha(isa),1,1,16,64,16,0,0,512,.01,.02,.005,"b"*64)
  return value.with_hash().validate()

def captures():
  return (capture("qwen3_8b_q4k_m_gfx1100","FULL_RESIDENT_OVERLAY",0,32),
          capture("qwen3_8b_q4k_m_gfx1100","FULL_RESIDENT_OVERLAY",16,32),
          capture("qwen3_14b_q4k_m_gfx1100","BOUNDED_PACKED_TILES",0,40),
          capture("qwen3_14b_q4k_m_gfx1100","BOUNDED_PACKED_TILES",16,40))

def test_capture_is_immutable_content_addressed_and_strictly_roundtrips():
  value=captures()[0]
  assert SharedAttentionCompilerCapture.from_json(value.to_json()) == value
  with pytest.raises(FrozenInstanceError): value.loop_count=2

def test_capture_rejects_missing_field_source_hash_and_payload_tamper():
  value=captures()[0]
  missing=value.to_json(); missing.pop("wmma_roles")
  with pytest.raises(ValueError,match="fields"): SharedAttentionCompilerCapture.from_json(missing)
  with pytest.raises(ValueError,match="source/ISA hash"): replace(value,hip_source=value.hip_source+"tamper").validate()
  with pytest.raises(ValueError,match="capture hash"): replace(value,numeric_max_abs=.02).validate()

def test_capture_rejects_missing_role_materialization_and_spill():
  value=captures()[0]
  with pytest.raises(ValueError,match="WMMA role|8 PV"): replace(value,wmma_roles=value.wmma_roles[:-1]).with_hash().validate()
  with pytest.raises(ValueError,match="materialized"): replace(value,score_probability_buffers=1).with_hash().validate()
  with pytest.raises(ValueError,match="resource contract"): replace(value,spill_count=1).with_hash().validate()

def test_proof_requires_only_exact_validated_four_capture_coverage():
  proof=shared_attention_proof_artifact(captures())
  assert proof["status"] == "PASS" and proof["passed"] and len(proof["captures"]) == 4
  with pytest.raises(ValueError,match="exact"): shared_attention_proof_artifact(captures()[:-1])
  with pytest.raises(TypeError,match="immutable"): shared_attention_proof_artifact(("raw source",))
  with pytest.raises(TypeError): shared_attention_proof_artifact(source="raw",isa="raw",ownership={},model_routes={})

def test_constructor_uses_actual_scheduled_call_and_final_hip_amdisa_programs():
  import numpy as np
  from tinygrad import Tensor, dtypes
  from tinygrad.codegen import to_program
  from tinygrad.helpers import Target
  from tinygrad.llm.flash_prefill_attention import shared_prefill_attention
  from tinygrad.renderer.cstyle import HIPRenderer
  from tinygrad.renderer.isa.amd import AMDISARenderer
  from tinygrad.uop.ops import Ops
  from extra.qk.shared_attention_capture import build_shared_attention_compiler_capture
  ctx=SharedAttentionCandidateContext("qwen3_8b_q4k_m_gfx1100","FULL_RESIDENT_OVERLAY",512,512,0,32,8,128,True)
  q=Tensor.empty(1,32,512,128,dtype=dtypes.float16,device="AMD")
  k=Tensor.empty(1,8,512,128,dtype=dtypes.float16,device="AMD")
  v=Tensor.empty(1,8,512,128,dtype=dtypes.float16,device="AMD")
  mask=Tensor.full((1,1,512,512),float("-inf"),dtype=dtypes.float16,buffer=False).triu(1)
  schedule=shared_prefill_attention(q,k,v,mask=mask,candidate_context=ctx).schedule_linear()
  calls=[x for x in schedule.src if x.op is Ops.CALL and getattr(x.src[0].arg,"candidate_context",None)==ctx]
  assert len(calls)==1
  ast=calls[0].src[0]
  assert ast.arg.opts_to_apply is None and ast.arg.required_native_attention.candidate_context==ctx
  hip=to_program(ast,HIPRenderer(Target.parse("AMD:HIP:gfx1100")))
  isa=to_program(ast,AMDISARenderer(Target.parse("AMD:ISA:gfx1100")))
  zeros=np.zeros((1,32,512,128),dtype=np.float32)
  value=build_shared_attention_compiler_capture(schedule=schedule,compute_call=calls[0],hip_program=hip,
    amd_isa_program=isa,output=zeros,reference=zeros)
  assert value.candidate_context==ctx and len(value.wmma_roles)==16
  assert value.hip_source_sha256==_sha(value.hip_source) and value.amd_isa_sha256==_sha(value.amd_isa_text)
