from types import SimpleNamespace
import pytest

from tinygrad.llm.prefill_policy import select_prefill_runtime_policy, bounded_packed_projection_proven_eligible
from extra.qk.shared_attention_evidence import shared_attention_proof_artifact
from tinygrad.llm.prefill_route_observer import PrefillRouteAttachment
from tinygrad.llm.prefill_routes import _attached_production_route
from tinygrad import Tensor


def _base():
  return {"strategy": "DIRECT_PACKED_FALLBACK", "candidate_id": "baseline", "routes": {}}


def _facts(): return SimpleNamespace(backend="AMD", architecture="gfx1100")

def _artifact():
  return {"schema":"tinygrad.shared_attention_proof.v2","status":"PASS","passed":True,"captures":[{} for _ in range(4)]}


def test_shared_attention_is_disabled_without_complete_proof():
  assert not select_prefill_runtime_policy(_base(), scanned_device_facts=_facts(), workload_reuse=False)["prefill_tc_attn"]


def test_shared_attention_requires_every_roofline_proof_field():
  proof = {"status": "PASS", "target": {"backend": "AMD", "architecture": "gfx1100"}, "geometry": {"Bq": 16, "Bkv": 64},
           "correctness": True, "score_resident": True, "qk_wmma": True, "pv_wmma": True,
           "model_8b_prefill": True, "model_14b_prefill": True,
           "decode_nonregression_8b": True, "decode_nonregression_14b": True, "artifact": _artifact()}
  assert select_prefill_runtime_policy({**_base(), "shared_attention_proof": proof}, scanned_device_facts=_facts(), workload_reuse=False)["prefill_tc_attn"]
  proof["pv_wmma"] = False
  assert not select_prefill_runtime_policy({**_base(), "shared_attention_proof": proof}, scanned_device_facts=_facts(), workload_reuse=False)["prefill_tc_attn"]


def test_shared_attention_override_cannot_bypass_incomplete_proof():
  assert not select_prefill_runtime_policy(_base(), scanned_device_facts=_facts(), workload_reuse=False,
                                           tc_attn_override=True)["prefill_tc_attn"]

def test_shared_attention_artifact_rejects_raw_caller_evidence():
  with pytest.raises(TypeError): shared_attention_proof_artifact(source="CALL fused",isa="v_wmma",ownership={},model_routes={})

def test_bounded_packed_projection_requires_all_compiler_numeric_and_owner_facts():
  proof = {"status": "PASS", "target": {"backend": "AMD", "architecture": "gfx1100"},
           "q4_source_owner": "MODEL_PARAMETER", "fused_dequant_wmma": True, "fp16_qkv_outputs": True,
           "numeric_correctness": True, "memory_cap": True, "allocation_owner_identity": "q4k:selected"}
  assert bounded_packed_projection_proven_eligible({"bounded_packed_projection_proof": proof}, _facts())
  proof["numeric_correctness"] = False
  assert not bounded_packed_projection_proven_eligible({"bounded_packed_projection_proof": proof}, _facts())

def test_overlay_and_bounded_projection_attachments_converge_before_shared_attention():
  class Linear:
    out_features=4096; in_features=4096; q4k_storage=object(); bias=None
    def prefill_packed_weight(self): return Tensor.empty(1,device="CPU")
  x=Tensor.empty(1,512,4096,device="CPU")
  overlay=Linear(); overlay._pf16_w=Tensor.empty(1,device="CPU")
  overlay._prefill_route_attachment=PrefillRouteAttachment("i","overlay","q.weight", {"strategy":"FULL_RESIDENT_OVERLAY","candidate_id":"overlay"}, _facts())
  assert _attached_production_route(overlay,x) == "fp16"
  proof={"status":"PASS","target":{"backend":"AMD","architecture":"gfx1100"},"q4_source_owner":"MODEL_PARAMETER",
    "fused_dequant_wmma":True,"fp16_qkv_outputs":True,"numeric_correctness":True,"memory_cap":True,
    "allocation_owner_identity":"q4k:selected"}
  bounded=Linear(); bounded._prefill_route_attachment=PrefillRouteAttachment("i","bounded","q.weight",
    {"strategy":"BOUNDED_PACKED_TILES","candidate_id":"bounded","bounded_packed_projection_proof":proof},_facts(),"q4k:selected")
  assert _attached_production_route(bounded,x) == "bounded_packed"
  proof["memory_cap"]=False
  assert _attached_production_route(bounded,x) is None
