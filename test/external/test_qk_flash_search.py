"""Synthetic tests for the flash-threshold search (no GPU). `_sweep` is patched with controlled
SDPA/flash curves so crossover detection, AcceptedPolicy emission, and the provenance/portability
guard are exercised deterministically."""
from __future__ import annotations

import contextlib
import json
import pathlib
import tempfile
import unittest
from unittest import mock

from extra import qk_flash_search as fs

# SDPA degrades fast with ctx; flash degrades slowly -> they cross between 256 and 512.
SDPA = {8: 56.0, 256: 46.0, 512: 38.0, 1024: 28.0, 3072: 9.4}
FLASH = {8: 47.0, 256: 44.0, 512: 42.0, 1024: 34.0, 3072: 22.7}


def _fake_sweep(model, flash, buckets, max_context, timeout):
  return dict(FLASH if flash else SDPA)


class TestFlashSearch(unittest.TestCase):
  def test_find_threshold(self):
    threshold, frontier = fs.find_threshold(SDPA, FLASH, 4096)
    self.assertEqual(threshold, 512)                       # first ctx where flash >= SDPA
    self.assertFalse(frontier[0]["flash_wins"])            # ctx 8: flash loses
    self.assertTrue(next(r for r in frontier if r["ctx"] == 3072)["flash_wins"])
    self.assertAlmostEqual(next(r for r in frontier if r["ctx"] == 3072)["speedup"], 2.415, places=2)

  def test_no_crossover_defaults_to_max_context(self):
    # flash never wins -> threshold = max_context (never flash)
    threshold, _ = fs.find_threshold({8: 56, 512: 38}, {8: 40, 512: 30}, 4096)
    self.assertEqual(threshold, 4096)

  @contextlib.contextmanager
  def _run(self):
    with tempfile.TemporaryDirectory() as d, mock.patch.object(fs, "_sweep", _fake_sweep):
      out = pathlib.Path(d)
      yield fs.run_threshold_search("fake.gguf", buckets=[8, 256, 512, 1024, 3072], max_context=4096,
                          timeout=1, out_dir=out), out

  def test_emits_accepted_policy(self):
    with self._run() as (summary, out):
      self.assertEqual(summary["threshold_ctx"], 512)
      self.assertTrue((out / "accepted-flash-threshold.json").exists())
      ap = json.loads((out / "accepted-flash-threshold.json").read_text())
      self.assertEqual(ap["ctx_range"], [512, 4096])
      self.assertEqual(ap["exactness"], "byte-identical")
      self.assertEqual(ap["threshold_ctx"], 512)

  def test_artifacts_portable_and_provenanced(self):
    with self._run() as (_summary, out):
      for f in out.glob("*.json"):
        text = f.read_text()
        self.assertNotIn("/home/", text)
        self.assertNotIn("uncommitted", text)
      s = json.loads((out / "flash-search.json").read_text())
      self.assertNotIn("model", s)
      self.assertIn("model_id", s)
      self.assertIn("gfx", s["hardware"].lower())
      self.assertTrue(s["commit"] and s["commit"] != "uncommitted")

  def test_frontier_md_renders(self):
    with self._run() as (summary, _out):
      md = fs.threshold_frontier_md(summary)
      self.assertIn("crossover at ctx 512", md)
      self.assertIn("YES", md)


if __name__ == "__main__":
  unittest.main()
