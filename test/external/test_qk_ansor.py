import pathlib, unittest

from extra.qk_ansor import (
  GENERATOR_VERSION, candidate_from_json, candidate_to_json, descriptor_from_info, descriptor_from_json, descriptor_to_json,
  generate_candidates, validate_policy_cache,
)
from extra.qk_layout import GGML_Q4_K, GGML_Q6_K, GGUFInfo, GGUFMetadata

class TestQKAnsor(unittest.TestCase):
  def _meta(self, info:GGUFInfo, data_start:int=128) -> GGUFMetadata:
    return GGUFMetadata(data_start, [info], {"general.architecture": "qwen3", "qwen3.embedding_length": 4096, "qwen3.feed_forward_length": 12288})

  def test_q4_descriptor_and_level0_candidates(self):
    info = GGUFInfo("blk.0.ffn_gate.weight", (4096, 12288), GGML_Q4_K, 64)
    desc = descriptor_from_info(pathlib.Path("/tmp/model.gguf"), self._meta(info), info, device="AMD", arch="gfx1100")
    self.assertEqual(desc.format, "Q4_K")
    self.assertEqual((desc.rows, desc.cols), (12288, 4096))
    self.assertEqual(desc.role, "ffn_gate")
    candidates = generate_candidates(desc, level=0)
    self.assertEqual([c.name for c in candidates], ["fused_graph", "v1_q4_packed"])
    self.assertEqual(candidates[1].opts, ("LOCAL:0:64",))
    self.assertNotIn("v1_q6_packed", [c.name for c in candidates])

  def test_q4_splitk_default_is_shape_derived(self):
    info = GGUFInfo("blk.0.ffn_down.weight", (12288, 4096), GGML_Q4_K, 64)
    desc = descriptor_from_info(pathlib.Path("/tmp/model.gguf"), self._meta(info), info, device="AMD", arch="gfx1100")
    self.assertEqual((desc.rows, desc.cols), (4096, 12288))
    self.assertEqual(generate_candidates(desc, level=0)[1].parts, 4)

  def test_q6_descriptor_and_level0_candidates(self):
    info = GGUFInfo("blk.0.ffn_down.weight", (12288, 4096), GGML_Q6_K, 64)
    desc = descriptor_from_info(pathlib.Path("/tmp/model.gguf"), self._meta(info), info, device="AMD", arch="gfx1100")
    candidates = generate_candidates(desc, level=0)
    self.assertEqual([c.name for c in candidates], ["fused_graph", "v1_q6_packed"])
    self.assertEqual(candidates[1].requires, ("q6k_gemv_partial_kernel", "u16_packed_storage"))

  def test_level2_emits_q8_1_sketch_without_runtime_claim(self):
    info = GGUFInfo("blk.0.ffn_gate.weight", (4096, 12288), GGML_Q4_K, 64)
    desc = descriptor_from_info(pathlib.Path("/tmp/model.gguf"), self._meta(info), info, device="AMD", arch="gfx1100")
    sketch = generate_candidates(desc, level=2)[-1]
    self.assertEqual(sketch.activation, "q8_1")
    self.assertIn("not_implemented", sketch.requires)

  def test_json_round_trip(self):
    info = GGUFInfo("blk.0.ffn_gate.weight", (4096, 12288), GGML_Q4_K, 64)
    desc = descriptor_from_info(pathlib.Path("/tmp/model.gguf"), self._meta(info), info, device="AMD", arch="gfx1100")
    self.assertEqual(descriptor_from_json(descriptor_to_json(desc)), desc)
    cand = generate_candidates(desc, level=0)[1]
    self.assertEqual(candidate_from_json(candidate_to_json(cand)), cand)

  def test_alignment_and_type_errors_are_loud(self):
    q4_bad = GGUFInfo("blk.0.ffn_gate.weight", (4096, 12288), GGML_Q4_K, 66)
    desc = descriptor_from_info(pathlib.Path("/tmp/model.gguf"), self._meta(q4_bad), q4_bad, device="AMD", arch="gfx1100")
    with self.assertRaisesRegex(ValueError, "uint32-aligned"):
      generate_candidates(desc, level=0)

    bad_type = GGUFInfo("blk.0.ffn_gate.weight", (4096, 12288), 8, 64)
    with self.assertRaisesRegex(ValueError, "unsupported"):
      descriptor_from_info(pathlib.Path("/tmp/model.gguf"), self._meta(bad_type), bad_type, device="AMD", arch="gfx1100")

  def test_policy_cache_validation_is_fail_closed(self):
    with self.assertRaisesRegex(ValueError, "generator version"):
      validate_policy_cache({"kind": "qk_generated_policy", "generator_version": GENERATOR_VERSION + 1, "commit": "x"}, pathlib.Path.cwd())

if __name__ == "__main__":
  unittest.main()
