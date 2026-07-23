from dataclasses import FrozenInstanceError, replace
import hashlib

import pytest

from extra.qk.shared_attention_capture import (ACC_SLICE_CAPTURE_SCHEMA, ACC_SLICE_PASS_SCHEMA, CAPTURE_SCHEMA,
  SharedAttentionAccSlicePass, SharedAttentionCompilerCapture, SharedAttentionSynchronization)
from extra.qk.shared_attention_evidence import shared_attention_proof_artifact
from tinygrad.uop.ops import AttentionWMMARole, SharedAttentionCandidateContext

def _sha(value:str) -> str: return hashlib.sha256(value.encode()).hexdigest()

def capture(profile,strategy,start,hq,base=0,blocks=8):
  qt,kv=16,start+16
  ctx=SharedAttentionCandidateContext(profile,strategy,qt,kv,start,hq,8,128,True,base,blocks)
  roles=tuple((i,AttentionWMMARole("QK",i)) for i in range(8))+tuple((8+i,AttentionWMMARole("PV",i)) for i in range(blocks))
  static_wmma_count=8+blocks
  hip="*(buf0+0) = x;\n__builtin_amdgcn_s_barrier();\nhalf val0 = (*(buf0+0));\n"
  isa="loop:\n"+"s_cbranch_scc1 loop\n"+"ds_store_b16\n"+"s_barrier\n"+"ds_load_b16\n"+("v_wmma_f32_16x16x16_f16 v0, v1, v2\n"*static_wmma_count)
  resources=tuple(sorted({"vgpr":64,"sgpr":16,"lds_bytes":512,"scratch_bytes":0,"vgpr_spills":0,
                          "sgpr_spills":0,"wavefront_size":32}.items()))
  sizes=(hq*qt*128,hq*qt*128,8*kv*128,8*kv*128)
  pass_metadata=None
  schema=CAPTURE_SCHEMA
  if blocks == 4:
    schema=ACC_SLICE_CAPTURE_SCHEMA
    pass_metadata=SharedAttentionAccSlicePass(ACC_SLICE_PASS_SCHEMA,base,blocks,True,_sha(f"{profile}:{strategy}:{start}:{hq}"))
  value=SharedAttentionCompilerCapture(schema,ctx,_sha(f"graph:{profile}:{strategy}:{start}:{base}:{blocks}"),1,0,
    tuple(enumerate(sizes)),tuple(sorted(sizes)),True,0,0,roles,hip,_sha(hip),resources,isa,_sha(isa),1,1,
    SharedAttentionSynchronization("workgroup",2,0,1),static_wmma_count,64,16,0,0,512,.01,.02,.005,"b"*64,
    acc_slice_pass=pass_metadata)
  return value.with_hash().validate()

def captures():
  return (capture("qwen3_8b_q4k_m_gfx1100","FULL_RESIDENT_OVERLAY",0,32),
          capture("qwen3_8b_q4k_m_gfx1100","FULL_RESIDENT_OVERLAY",16,32),
          capture("qwen3_14b_q4k_m_gfx1100","BOUNDED_PACKED_TILES",0,40),
          capture("qwen3_14b_q4k_m_gfx1100","BOUNDED_PACKED_TILES",16,40))

def slice_captures():
  return tuple(capture(profile,strategy,start,hq,base,4) for profile,strategy,start,hq in
    (("qwen3_8b_q4k_m_gfx1100","FULL_RESIDENT_OVERLAY",0,32),
     ("qwen3_8b_q4k_m_gfx1100","FULL_RESIDENT_OVERLAY",16,32),
     ("qwen3_14b_q4k_m_gfx1100","BOUNDED_PACKED_TILES",0,40),
     ("qwen3_14b_q4k_m_gfx1100","BOUNDED_PACKED_TILES",16,40)) for base in (0,4))

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

def test_acc_slice_capture_roundtrip_and_two_pass_proof():
  values=slice_captures()
  assert SharedAttentionCompilerCapture.from_json(values[0].to_json()) == values[0]
  proof=shared_attention_proof_artifact(values)
  assert proof["schema"] == "tinygrad.shared_attention_proof.acc_slice_v3"
  assert proof["passed"] and len(proof["captures"]) == 4
  assert all(row["output_blocks"] == list(range(8)) and row["wmma"] == {"qk":16,"pv":8,"qk_recomputed_passes":2}
             for row in proof["captures"])

def test_acc_slice_proof_rejects_overlap_gap_and_mixed_schema():
  values=list(slice_captures())
  overlap=values.copy(); overlap[1]=capture("qwen3_8b_q4k_m_gfx1100","FULL_RESIDENT_OVERLAY",0,32,0,4)
  with pytest.raises(ValueError,match="overlap"): shared_attention_proof_artifact(tuple(overlap))
  gap=values.copy(); gap[0]=capture("qwen3_8b_q4k_m_gfx1100","FULL_RESIDENT_OVERLAY",0,32,4,4)
  with pytest.raises(ValueError,match="gap"): shared_attention_proof_artifact(tuple(gap))
  with pytest.raises(ValueError,match="mixed"): shared_attention_proof_artifact(tuple(values[:-1])+captures()[:1])

def test_acc_slice_proof_rejects_graph_context_numeric_and_resource_mismatch():
  values=list(slice_captures())
  graph=values.copy(); graph[1]=replace(graph[1],acc_slice_pass=replace(graph[1].acc_slice_pass,logical_graph_sha256="c"*64)).with_hash().validate()
  with pytest.raises(ValueError,match="logical graphs"): shared_attention_proof_artifact(tuple(graph))
  context=values.copy(); context[1]=replace(context[1],candidate_context=context[1].candidate_context._replace(causal=False)).with_hash().validate()
  with pytest.raises(ValueError,match="contexts"): shared_attention_proof_artifact(tuple(context))
  numeric=values.copy(); numeric[1]=replace(numeric[1],numeric_max_abs=.011).with_hash().validate()
  with pytest.raises(ValueError,match="numeric"): shared_attention_proof_artifact(tuple(numeric))
  resource=values.copy()
  changed_resources=tuple((name,1024 if name == "lds_bytes" else value) for name,value in resource[1].hip_resources)
  resource[1]=replace(resource[1],hip_resources=changed_resources,lds_bytes=1024).with_hash().validate()
  with pytest.raises(ValueError,match="resource"): shared_attention_proof_artifact(tuple(resource))

def test_acc_slice_capture_rejects_incomplete_role_and_recomputation_metadata():
  value=slice_captures()[0]
  with pytest.raises(ValueError,match="WMMA role|4 PV"): replace(value,wmma_roles=value.wmma_roles[:-1]).with_hash().validate()
  bad_pass=replace(value.acc_slice_pass,qk_recomputed=False)
  with pytest.raises(ValueError,match="QK recomputation"): replace(value,acc_slice_pass=bad_pass).with_hash().validate()

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
