from types import SimpleNamespace

from tinygrad.llm.prefill_policy import select_prefill_runtime_policy


def _base():
  return {"strategy": "DIRECT_PACKED_FALLBACK", "candidate_id": "baseline", "routes": {}}


def _facts(): return SimpleNamespace(backend="AMD", architecture="gfx1100")


def test_shared_attention_is_disabled_without_complete_proof():
  assert not select_prefill_runtime_policy(_base(), scanned_device_facts=_facts(), workload_reuse=False)["prefill_tc_attn"]


def test_shared_attention_requires_every_roofline_proof_field():
  proof = {"status": "PASS", "target": {"backend": "AMD", "architecture": "gfx1100"}, "geometry": {"Bq": 16, "Bkv": 64},
           "correctness": True, "score_resident": True, "qk_wmma": True, "pv_wmma": True,
           "model_8b_prefill": True, "model_14b_prefill": True,
           "decode_nonregression_8b": True, "decode_nonregression_14b": True}
  assert select_prefill_runtime_policy({**_base(), "shared_attention_proof": proof}, scanned_device_facts=_facts(), workload_reuse=False)["prefill_tc_attn"]
  proof["pv_wmma"] = False
  assert not select_prefill_runtime_policy({**_base(), "shared_attention_proof": proof}, scanned_device_facts=_facts(), workload_reuse=False)["prefill_tc_attn"]


def test_shared_attention_override_cannot_bypass_incomplete_proof():
  assert not select_prefill_runtime_policy(_base(), scanned_device_facts=_facts(), workload_reuse=False,
                                           tc_attn_override=True)["prefill_tc_attn"]
