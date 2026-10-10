"""The block-quant decode GEMV as a search family: rows as schedule fields, the installed rule, fixtures and reference."""
import unittest
import numpy as np

from extra.llm_research import decode_emitter_space as space
from tinygrad.llm.decode_kernels import BLOCK_QUANT_LAYOUTS, block_quant_gemv_geometry
from tinygrad.llm.qk_layout import GGML_TYPE_FORMATS, GGML_TYPE_IDS, QUANT_FORMATS, packed_nbytes


def _schedule(row):
  sched = {"launch": {"threads": 0}, "tile": {"m": 1, "n": 0, "k": 0}, "memory": {"a": {"space": ""}, "b": {"vector_width": 0}},
           "pipeline": {"stage_count": 0}}
  for path, value in row.items():
    cur = sched
    parts = path.split(".")[1:]
    for p in parts[:-1]: cur = cur[p]
    cur[parts[-1]] = value
  return sched


class TestBlockQuantFamily(unittest.TestCase):
  def test_family_describes_layout_facts_and_rule(self):
    fam = space.BY_FAMILY[space.BLOCK_QUANT_FAMILY].to_json()
    self.assertEqual(fam["quants"], ["Q8_0", "Q5_0", "Q4_0"])
    self.assertEqual(fam["code_units_per_block"], {q: BLOCK_QUANT_LAYOUTS[q].units for q in fam["quants"]})
    self.assertEqual(set(fam["installed_rule"]), {"units_per_lane", "max_warps_per_cta", "min_ctas_per_sm"})
    self.assertIsNone(space.installed_row(space.BY_FAMILY[space.BLOCK_QUANT_FAMILY], "X", "y"))

  def test_installed_row_round_trips_to_route_geometry(self):
    facts = {"wave_size": 32, "sm_count": 10, "max_threads": 1024}
    for quant in ("Q8_0", "Q5_0", "Q4_0"):
      for rows, k in ((40, 64), (5000, 96), (7, 32)):
        row = space.installed_block_quant_row(quant, rows, k, facts)
        params = space.BY_FAMILY[space.BLOCK_QUANT_FAMILY].parameters(_schedule(row), rows, k, quant, 32)
        spec = block_quant_gemv_geometry(QUANT_FORMATS[quant], rows, k, wave_size=32, sm_count=10, max_threads=1024)
        self.assertEqual((params["lanes_per_block"], params["warps_per_cta"]), (spec.lanes_per_block, spec.warps_per_cta))
        self.assertEqual(row["schedule.tile.k"], QUANT_FORMATS[quant].block_elems)

  def test_parameters_refuse_a_split_code_unit_and_a_non_warp_cta(self):
    p = space.BY_FAMILY[space.BLOCK_QUANT_FAMILY].parameters
    with self.assertRaises(ValueError): p(_schedule({"schedule.memory.b.vector_width": 3, "schedule.launch.threads": 32, "schedule.tile.n": 1}), 8, 32, "Q8_0", 32)
    with self.assertRaises(ValueError): p(_schedule({"schedule.memory.b.vector_width": 2, "schedule.launch.threads": 48, "schedule.tile.n": 1}), 8, 32, "Q8_0", 32)

  def test_fixture_sizes_and_reference_against_ggml_layout(self):
    for quant in ("Q8_0", "Q5_0", "Q4_0", "Q4_K", "Q6_K"):
      k = QUANT_FORMATS[quant].block_elems * 2
      raw = space.fixture_bytes(quant, 3, k)
      self.assertEqual(raw.nbytes, packed_nbytes(QUANT_FORMATS[quant], 3 * k))
      w = space.reference(quant, raw, 3 * k)
      self.assertEqual(w.size, 3 * k)
      self.assertTrue(np.isfinite(w).all() and np.any(w != 0))
    # Q8_0 by hand: d * int8
    raw = space.fixture_bytes("Q8_0", 1, 32)
    d = raw[0:2].view(np.float16).astype(np.float32)[0]
    np.testing.assert_allclose(space.reference("Q8_0", raw, 32), d * raw[2:34].view(np.int8).astype(np.float32), rtol=1e-6)

  def test_ggml_type_table(self):
    self.assertEqual({t: f.name for t, f in GGML_TYPE_FORMATS.items()}, {2: "Q4_0", 6: "Q5_0", 8: "Q8_0", 12: "Q4_K", 14: "Q6_K"})
    self.assertEqual(GGML_TYPE_IDS["Q8_0"], 8)
    with self.assertRaises(ValueError): packed_nbytes(QUANT_FORMATS["Q8_0"], 33)

  def test_registry_builds_the_emitter(self):
    from extra.llm_research.semantic_kernel_lowering import build_registered_llm_emitter
    fn = build_registered_llm_emitter("block_quant_gemv.v1", {"rows": 8, "k": 64, "quant": "Q5_0", "wave_size": 32,
                                                              "lanes_per_block": 8, "warps_per_cta": 2})
    self.assertTrue(callable(fn))
    with self.assertRaises(ValueError): build_registered_llm_emitter("block_quant_gemv.v1", {"rows": 8})


if __name__ == "__main__":
  unittest.main()
