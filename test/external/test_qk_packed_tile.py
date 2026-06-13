import json, pathlib, unittest

from extra.qk_layout import GGML_Q4_K, GGML_Q6_K, GGUFInfo, GGUFMetadata
from extra.qk_packed_tile import (
  Q4_K_FORMAT, Q6_K_FORMAT, format_for_type, load_tile, qk_tile_search_axes, tile_from_info,
  tile_from_semantic_row, tile_summary, tile_to_json,
)

class TestQKPackedTile(unittest.TestCase):
  def _meta(self, info:GGUFInfo, data_start:int=128) -> GGUFMetadata:
    return GGUFMetadata(data_start, [info], {"general.architecture": "qwen3"})

  def test_q4_tile_exposes_scalar_and_aligned_vector_loads(self):
    info = GGUFInfo("blk.0.ffn_gate.weight", (4096, 12288), GGML_Q4_K, 64)
    tile = tile_from_info(self._meta(info), info)
    self.assertEqual(tile.format, Q4_K_FORMAT)
    self.assertEqual((tile.rows, tile.cols), (12288, 4096))
    self.assertEqual(tile.blocks, 196608)
    self.assertEqual(tile.packed_bytes, 28311552)
    self.assertEqual(tile.format.storage_items_per_block, 36)
    self.assertEqual([t.name for t in tile.legal_load_tiles], ["u32_scalar", "u32x4_aligned"])
    vector = load_tile(tile, "u32x4_aligned")
    self.assertEqual(vector.bytes, 16)
    self.assertEqual(vector.q_values_per_load, 32)
    self.assertFalse(vector.requires_tail)
    self.assertEqual(tile_summary(tile)["reference"], "extra.qk_layout.q4_k_reference")

  def test_q4_vector_load_requires_16_byte_alignment(self):
    info = GGUFInfo("blk.0.ffn_gate.weight", (4096, 12288), GGML_Q4_K, 68)
    tile = tile_from_info(self._meta(info), info)
    self.assertTrue(tile.storage_aligned)
    self.assertEqual([t.name for t in tile.legal_load_tiles], ["u32_scalar"])
    with self.assertRaisesRegex(ValueError, "u32x4_aligned"):
      load_tile(tile, "u32x4_aligned")

  def test_q6_tile_is_scalar_halfword_only_for_now(self):
    info = GGUFInfo("blk.0.ffn_down.weight", (12288, 4096), GGML_Q6_K, 64)
    tile = tile_from_info(self._meta(info), info)
    self.assertEqual(tile.format, Q6_K_FORMAT)
    self.assertEqual((tile.rows, tile.cols), (4096, 12288))
    self.assertEqual(tile.format.storage_items_per_block, 105)
    self.assertEqual([t.name for t in tile.legal_load_tiles], ["u16_scalar"])
    self.assertEqual(load_tile(tile, "u16_scalar").storage_dtype, "uint16")
    with self.assertRaisesRegex(ValueError, "u32x4_aligned"):
      load_tile(tile, "u32x4_aligned")

  def test_committed_semantic_descriptor_row_builds_same_tile_shape(self):
    repo = pathlib.Path(__file__).resolve().parents[2]
    descriptor = json.loads((repo / "bench/qk-ansor-transition-20260612/descriptors/8b.json").read_text())
    row = next(r for r in descriptor["descriptors"] if r["format"] == "Q4_K" and r["role"] == "ffn_gate")
    tile = tile_from_semantic_row(row)
    self.assertEqual(tile.tensor, "blk.0.ffn_gate.weight")
    self.assertEqual(tile_summary(tile)["legal_load_tiles"], ["u32_scalar", "u32x4_aligned"])
    axes = qk_tile_search_axes(tile)
    self.assertEqual(axes["semantic_object"], "packed_qk_tile")
    self.assertIn("load_tile", axes["memory_axes"])
    self.assertEqual(tile_to_json(tile)["format"]["name"], "Q4_K")

  def test_bad_format_and_bad_layout_are_loud(self):
    with self.assertRaisesRegex(ValueError, "unsupported packed QK"):
      format_for_type(999)
    row = {
      "tensor": "blk.0.ffn_gate.weight",
      "role": "ffn_gate",
      "format": "Q4_K",
      "ggml_type": GGML_Q4_K,
      "shape": {"rows": 12288, "cols": 4096},
      "layout": {"block_bytes": 140, "block_elems": 256, "byte_start": 128, "packed_bytes": 28311552},
    }
    with self.assertRaisesRegex(ValueError, "layout mismatch"):
      tile_from_semantic_row(row)

if __name__ == "__main__":
  unittest.main()
