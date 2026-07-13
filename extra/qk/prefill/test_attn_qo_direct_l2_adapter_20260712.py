from extra.qk.prefill.attn_qo_direct_l2_adapter_20260712 import prepare_exact_pair


def test_attn_qo_adapter_blocks_without_exact_payloads():
  result = prepare_exact_pair(direct_payload=None, lds_payload=None,
                              direct_binary_sha256=None, lds_binary_sha256=None)
  assert result["status"] == "blocked"
  assert result["dispatch_state"] == "not_attempted"
  assert result["blockers"] == ["exact direct_l2 and lds candidate payloads are required"]


def test_attn_qo_adapter_rejects_missing_binary_identity():
  result = prepare_exact_pair(direct_payload={}, lds_payload={},
                              direct_binary_sha256="0" * 64, lds_binary_sha256=None)
  assert result["status"] == "blocked"
  assert result["blockers"] == ["exact direct_l2 and lds binary SHA-256 identities are required"]


def test_attn_qo_adapter_requires_semantic_pair_key_before_candidate_admission():
  result = prepare_exact_pair(direct_payload={}, lds_payload={},
                              direct_binary_sha256="0" * 64, lds_binary_sha256="1" * 64)
  assert result["status"] == "blocked"
  assert result["blockers"] == ["semantic pair key is required"]
