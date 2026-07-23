from types import SimpleNamespace

from tinygrad.llm.prefill_policy import select_prefill_runtime_policy, bounded_packed_projection_proven_eligible
from extra.qk.shared_attention_evidence import shared_attention_proof_artifact


def _base():
  return {"strategy": "DIRECT_PACKED_FALLBACK", "candidate_id": "baseline", "routes": {}}


def _facts(): return SimpleNamespace(backend="AMD", architecture="gfx1100")

def _artifact():
  source = "CALL fused_attention\n// QK WMMA\n// PV WMMA\nSHAPED_WMMA(TILE_GATHER)"
  routes = {name: {"first_chunk": True, "prefix_chunk": True, "shared_boundary": "shared_prefill_attention",
                   "projection_strategies": ("FULL_RESIDENT_OVERLAY", "BOUNDED_PACKED_TILES")}
            for name in ("qwen3_8b_q4k_m_gfx1100", "qwen3_14b_q4k_m_gfx1100")}
  return shared_attention_proof_artifact(source=source, isa="QK: v_wmma\nPV: v_wmma",
    ownership={"authority": "final_regalloc", "operands": ("output", "q", "k", "v"), "grid_owner": "gidx0"}, model_routes=routes)


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

def test_shared_attention_artifact_fails_closed_without_role_attributed_isa():
  bad = shared_attention_proof_artifact(source="CALL fused\n// QK WMMA\n// PV WMMA\nSHAPED_WMMA",
    isa="QK: v_wmma", ownership={"authority": "final_regalloc", "operands": ("output", "q", "k", "v"), "grid_owner": "gidx0"},
    model_routes={})
  assert bad["status"] == "INCOMPLETE"

def test_bounded_packed_projection_requires_all_compiler_numeric_and_owner_facts():
  proof = {"status": "PASS", "target": {"backend": "AMD", "architecture": "gfx1100"},
           "q4_source_owner": "MODEL_PARAMETER", "fused_dequant_wmma": True, "fp16_qkv_outputs": True,
           "numeric_correctness": True, "memory_cap": True, "allocation_owner_identity": "q4k:selected"}
  assert bounded_packed_projection_proven_eligible({"bounded_packed_projection_proof": proof}, _facts())
  proof["numeric_correctness"] = False
  assert not bounded_packed_projection_proven_eligible({"bounded_packed_projection_proof": proof}, _facts())
