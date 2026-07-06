import json

import pytest

from extra.qk.q4k_wmma_full_role_contract_gate import build_report
from extra.qk.q4k_wmma_tile_lowering import (
  QWEN3_14B_Q4K_ROLE_SHAPES,
  describe_int8_wmma_tile_lowering,
  describe_qwen3_14b_q4k_full_role_lowering,
)


def test_q4k_wmma_full_role_contract_covers_14b_roles():
  spec = describe_qwen3_14b_q4k_full_role_lowering()
  payload = spec.to_json()
  assert payload["role_count"] == len(QWEN3_14B_Q4K_ROLE_SHAPES)
  assert {r["role"] for r in payload["roles"]} == {"attn_kv", "attn_qo", "ffn_down", "ffn_gate_up"}
  assert payload["max_live_raw_elems"] == 256
  assert payload["max_forbidden_full_raw_elems"] == 1426063360
  assert all(r["bounds"]["bounded_raw_ok"] for r in payload["roles"])
  assert all(r["lowering"]["requires_scheduler_owned_loop"] for r in payload["roles"])
  json.dumps(payload)


def test_q4k_wmma_tile_contract_rejects_unaligned_shapes():
  with pytest.raises(ValueError, match="multiple of Q4_K"):
    describe_int8_wmma_tile_lowering(512, 5120, 5136, role="attn_qo")
  with pytest.raises(ValueError, match="m/n must divide"):
    describe_int8_wmma_tile_lowering(513, 5120, 5120, role="attn_qo")
  with pytest.raises(ValueError, match="unknown wmma_surface"):
    describe_int8_wmma_tile_lowering(512, 5120, 5120, role="attn_qo", wmma_surface="raw")


def test_q4k_wmma_full_role_contract_gate_reports_remaining_scheduler_blocker():
  report = build_report()
  assert report["schema"] == "q4k-wmma-full-role-contract-gate.v1"
  assert report["verdict"] == "Q4K_WMMA_FULL_ROLE_CONTRACT_PASS"
  assert report["evidence"]["surface_ok"] is True
  assert report["evidence"]["lifecycle_ok"] is True
  assert report["evidence"]["no_hand_ok"] is True
  assert report["remaining_blocker"] == "scheduler_owned_tile_loop_missing"
