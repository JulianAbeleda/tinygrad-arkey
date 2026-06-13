import json, pathlib, tempfile, unittest

import numpy as np

from extra.q4_k_gemv_primitive import _q4k_tile_custom_partial_source
from extra.qk_layout import GGML_Q4_K, GGML_Q6_K, GGUFInfo, GGUFMetadata
from extra.qk_load_width_report import build_report as build_load_width_report
from extra.qk_packed_tile_lowering_analysis import report_markdown as lowering_report_markdown, summarize_runs
from extra.qk_packed_tile import (
  Q4_K_FORMAT, Q6_K_FORMAT, format_for_type, load_tile, qk_tile_search_axes, tile_from_info,
  tile_from_semantic_row, tile_summary, tile_to_json,
)
from extra.qk_packed_tile_consumption_probe import _q4_words_and_x

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

  def test_consumption_probe_oracle_and_load_width_artifact(self):
    words, x, expected = _q4_words_and_x()
    got = np.float32(0.0)
    for lane, word in enumerate(words[4:8]):
      for nib in range(4):
        byte = int((int(word) >> (8*nib)) & 0xff)
        got += np.float32(byte & 0xf) * x[lane*4+nib]
        got += np.float32(byte >> 4) * x[32+lane*4+nib]
    self.assertEqual(got, float(expected))

    repo = pathlib.Path(__file__).resolve().parents[2]
    probe = repo / "bench/qk-packed-tile-consumption-20260613/probe.json"
    if probe.exists():
      summary = json.loads(probe.read_text())["summary"]
      self.assertEqual(summary["decision"], "semantic_custom_op_required")
      self.assertFalse(summary["run_microbench"])
      self.assertFalse(summary["run_full_decode"])

    log = repo / "bench/qk-packed-tile-consumption-20260613/load-width/probe-debug4.log"
    if log.exists():
      report = build_load_width_report([log], repo=repo)
      self.assertTrue(report["summary"]["has_vector_load_evidence"])
      self.assertEqual(report["rows"][0]["mode"], "packed_tile_custom_q4_dot")
      self.assertEqual(report["rows"][0]["load_width_inferred"], "vector_u32x4")

  def test_tile_custom_partial_source_is_classified_as_vector_load(self):
    source = _q4k_tile_custom_partial_source(k_blocks=16, parts=1)
    self.assertIn("tg_uint4 qv = *((tg_uint4*)", source)
    self.assertIn("byte & 15u", source)
    self.assertIn("byte >> 4u", source)

    repo = pathlib.Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as td:
      log = pathlib.Path(td) / "tile-custom-debug4.log"
      log.write_text("q4k_gemv_tile_custom_partial_2_4096_1\n" + source)
      report = build_load_width_report([log], repo=repo)
    self.assertEqual(report["rows"][0]["mode"], "tile_custom_partial")
    self.assertEqual(report["rows"][0]["load_width_inferred"], "vector_u32x4")

  def test_packed_tile_lowering_analysis_decision_gate(self):
    rows = []
    for tensor, gain in (("blk.0.ffn_gate.weight", 7.0), ("blk.0.attn_output.weight", 3.0)):
      for mode, base in (("v1_partial", 100.0), ("tile_custom", 100.0 + gain)):
        for run in range(3):
          rows.append({
            "tensor": tensor,
            "mode": mode,
            "raw_file": f"bench/fake/{tensor}/{mode}/{run}.json",
            "q4_eff_gbs": base + run,
            "ms": 1.0,
            "primitive_gemv_max_abs": 0.001,
          })
    report = summarize_runs(rows)
    self.assertEqual(report["summary"]["decision"], "diagnose_only_not_promoted")
    self.assertAlmostEqual(report["summary"]["max_gain_pct"], 6.93, places=2)
    md = lowering_report_markdown(report)
    self.assertIn("diagnose_only_not_promoted", md)
    self.assertIn("blk.0.ffn_gate.weight", md)

  def test_packed_tile_lowering_analysis_artifact_reproduces(self):
    repo = pathlib.Path(__file__).resolve().parents[2]
    root = repo / "bench/qk-packed-tile-lowering-analysis-20260613"
    if not root.exists(): return

    logs = [root / "source/v1_partial-debug4.log", root / "source/tile_custom-debug4.log"]
    load_report = build_load_width_report(logs, repo=repo)
    self.assertEqual(json.loads((root / "source/load-width-report.json").read_text()), load_report)

    raw_runs = []
    for raw in sorted((root / "raw").rglob("run-*.json")):
      rows = json.loads(raw.read_text())
      matches = [row for row in rows if row.get("name") == "q4k_primitive_gemv"]
      self.assertEqual(len(matches), 1)
      row = matches[0]
      raw_runs.append({
        "tensor": row["tensor"],
        "mode": raw.parent.name,
        "raw_file": str(raw.relative_to(repo)),
        "q4_eff_gbs": row["q4_eff_gbs"],
        "ms": row["ms"],
        "primitive_gemv_max_abs": row["primitive_gemv_max_abs"],
      })

    committed = json.loads((root / "analysis.json").read_text())
    rebuilt = summarize_runs(raw_runs)
    self.assertEqual(rebuilt["summary"], committed["summary"])
    self.assertEqual(rebuilt["comparisons"], committed["comparisons"])
    self.assertEqual((root / "analysis.md").read_text(), lowering_report_markdown(committed))

if __name__ == "__main__":
  unittest.main()
