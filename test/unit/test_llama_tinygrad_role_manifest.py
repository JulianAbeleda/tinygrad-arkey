import unittest

from extra.llm_research.decode.llama_tinygrad_role_manifest import clean, _quant_type, ROLES, LLAMA_OFFSETS, TG_OFFSETS


class TestRoleManifest(unittest.TestCase):
  def test_role_offsets_are_complete_and_ordered(self):
    self.assertEqual(len(ROLES), 6)
    self.assertEqual(ROLES, ("attn_q", "attn_v", "attn_k", "attn_o", "ffn_gate_up", "ffn_down"))
    self.assertEqual(len(LLAMA_OFFSETS), len(ROLES))
    self.assertEqual(len(TG_OFFSETS), len(ROLES))
    self.assertEqual(sorted(LLAMA_OFFSETS), list(LLAMA_OFFSETS))
    self.assertEqual(sorted(TG_OFFSETS), list(TG_OFFSETS))

  def test_equal_shape_attention_order_is_v_then_k(self):
    # Pinned Qwen3 builds Q/K/V, while build_attn intentionally expands Q/V/K
    # so rope can fuse the K write into the KV cache.  The trace observes graph
    # execution order, hence its first and second 1024 MMVQs are V and K.
    self.assertEqual(ROLES[1:3], ("attn_v", "attn_k"))

  def test_symbol_classification_fails_closed(self):
    self.assertEqual(_quant_type("mul_mat_vec_q<(ggml_type)12, x>"), "Q4_K")
    self.assertEqual(_quant_type("mul_mat_vec_q<(ggml_type)14, x>"), "Q6_K")
    self.assertEqual(_quant_type("mul_mat_vec_q<(ggml_type)9, x>"), "UNKNOWN")

  def test_clean_only_removes_content_suffix(self):
    self.assertEqual(clean("q4k_" + "a"*64), "q4k")
    self.assertEqual(clean("q4k_abc"), "q4k_abc")


if __name__ == "__main__": unittest.main()
