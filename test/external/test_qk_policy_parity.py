import unittest

from extra.qk_layout import GGML_Q4_K, GGML_Q6_K, GGUFInfo, GGUFMetadata
from extra.qk_policy_parity import compare_policies, summarize

class TestQKPolicyParity(unittest.TestCase):
  def _meta(self):
    return GGUFMetadata(128, [
      GGUFInfo("blk.0.ffn_gate.weight", (4096, 12288), GGML_Q4_K, 0),
      GGUFInfo("blk.0.ffn_down.weight", (12288, 4096), GGML_Q4_K, 0),
      GGUFInfo("blk.0.attn_k.weight", (4096, 1024), GGML_Q4_K, 0),
      GGUFInfo("blk.0.ffn_down.weight", (12288, 4096), GGML_Q6_K, 0),
    ], {})

  def test_matching_generated_policy(self):
    policy = {
      (GGML_Q4_K, 12288, 4096): {"winner": "v1_q4_packed", "parts": 1, "opts": ("LOCAL:0:64",), "family": "q4_k_packed_u32"},
      (GGML_Q4_K, 4096, 12288): {"winner": "v1_q4_packed", "parts": 4, "opts": ("LOCAL:0:32",), "family": "q4_k_packed_u32"},
      (GGML_Q4_K, 1024, 4096): {"winner": "fused_graph", "parts": 0, "opts": (), "family": "fused_graph"},
      (GGML_Q6_K, 4096, 12288): {"winner": "v1_q6_packed", "parts": 1, "opts": ("LOCAL:0:64",), "family": "q6_k_packed_u16"},
    }
    rows = compare_policies(self._meta(), policy)
    self.assertEqual(summarize(rows)["effective_mismatches"], 0)
    self.assertEqual(summarize(rows)["generated_unsupported"], 0)

  def test_missing_fallback_is_effectively_same_but_raw_different(self):
    rows = compare_policies(GGUFMetadata(128, [GGUFInfo("blk.0.attn_k.weight", (4096, 1024), GGML_Q4_K, 0)], {}), {})
    self.assertTrue(rows[0].same_effective)
    self.assertFalse(rows[0].same_raw)
    self.assertEqual(rows[0].generated.reason, "policy_missing")

  def test_wrong_shape_policy_is_effective_mismatch(self):
    policy = {
      (GGML_Q4_K, 12288, 4096): {"winner": "fused_graph", "parts": 0, "opts": (), "family": "fused_graph"},
    }
    rows = compare_policies(GGUFMetadata(128, [GGUFInfo("blk.0.ffn_gate.weight", (4096, 12288), GGML_Q4_K, 0)], {}), policy)
    self.assertFalse(rows[0].same_effective)
    self.assertEqual(summarize(rows)["effective_mismatches"], 1)

  def test_unsupported_generated_winner_is_loud(self):
    policy = {
      (GGML_Q4_K, 12288, 4096): {"winner": "q8_1_q4_packed", "parts": 1, "opts": ("LOCAL:0:64",), "family": "q4_k_q8_1"},
    }
    rows = compare_policies(GGUFMetadata(128, [GGUFInfo("blk.0.ffn_gate.weight", (4096, 12288), GGML_Q4_K, 0)], {}), policy)
    self.assertTrue(rows[0].generated.unsupported)
    self.assertEqual(summarize(rows)["generated_unsupported"], 1)
    self.assertFalse(rows[0].same_effective)

  def test_direct_output_q4_family_is_supported_but_effectively_different(self):
    policy = {
      (GGML_Q4_K, 1024, 4096): {
        "winner": "direct_out", "parts": 1, "opts": ("LOCAL:0:64",), "family": "q4_k_packed_u32_direct",
      },
    }
    rows = compare_policies(GGUFMetadata(128, [GGUFInfo("blk.0.attn_q.weight", (4096, 1024), GGML_Q4_K, 0)], {}), policy)
    self.assertFalse(rows[0].generated.unsupported)
    self.assertEqual(rows[0].generated.effective_winner, "q4_k_packed_u32_direct")
    self.assertEqual(summarize(rows)["generated_unsupported"], 0)
    self.assertFalse(rows[0].same_effective)

  def test_tensor_policy_overrides_shape_policy_and_reports_memory_cap(self):
    policy = {
      "by_shape": {
        (GGML_Q4_K, 12288, 4096): {
          "winner": "v1_q4_packed", "parts": 1, "opts": ("LOCAL:0:64",), "family": "q4_k_packed_u32",
        },
      },
      "by_tensor": {
        ("blk.0.ffn_gate.weight", GGML_Q4_K, 12288, 4096): {
          "winner": "fused_graph", "parts": 0, "opts": (), "family": "fused_graph",
          "policy_reason": "memory_cap_fused_over_budget",
        },
      },
    }
    rows = compare_policies(GGUFMetadata(128, [GGUFInfo("blk.0.ffn_gate.weight", (4096, 12288), GGML_Q4_K, 0)], {}), policy)
    self.assertEqual(rows[0].generated.reason, "policy_memory_cap")
    self.assertFalse(rows[0].same_effective)

if __name__ == "__main__":
  unittest.main()
