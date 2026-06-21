#!/usr/bin/env python3
"""Comparator SSOT drift guard.

extra/qk_harness_contract.py:DECODE_COMPARATOR is a light, importable mirror of the shipped model default
extra/qk_flash_decode.py:FLASH_DECODE_DEFAULT_VARIANT (which imports tinygrad, so light tooling can't import it).
This test fails if the two drift -- i.e. if the shipped decode winner changes but the comparator mirror does not,
which would make every A/B silently compare against a stale winner. No GPU / no tinygrad import (source-parsed).

Run: PYTHONPATH=. python -m unittest test.unit.test_comparator_ssot -v
"""
from __future__ import annotations
import ast
import pathlib
import unittest

from extra.qk_harness_contract import DECODE_COMPARATOR

ROOT = pathlib.Path(__file__).resolve().parents[2]
FLASH_DECODE = ROOT / "extra/qk_flash_decode.py"


def _module_str_constant(path: pathlib.Path, name: str) -> str:
  """Return the string value of a module-level `name = "..."` assignment, without importing the module."""
  tree = ast.parse(path.read_text())
  for node in tree.body:
    if isinstance(node, ast.Assign):
      for t in node.targets:
        if isinstance(t, ast.Name) and t.id == name and isinstance(node.value, ast.Constant):
          return node.value.value
  raise AssertionError(f"{name} not found as a module-level string assignment in {path}")


class TestComparatorSSOT(unittest.TestCase):
  def test_comparator_mirror_matches_shipped_default(self):
    shipped = _module_str_constant(FLASH_DECODE, "FLASH_DECODE_DEFAULT_VARIANT")
    self.assertEqual(DECODE_COMPARATOR, shipped,
                     "qk_harness_contract.DECODE_COMPARATOR drifted from the shipped "
                     "qk_flash_decode.FLASH_DECODE_DEFAULT_VARIANT -- update both to the new winner")

  def test_comparator_is_a_known_variant(self):
    variants = set()
    tree = ast.parse(FLASH_DECODE.read_text())
    for node in tree.body:
      if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "FLASH_DECODE_VARIANTS" for t in node.targets):
        if isinstance(node.value, ast.Tuple):
          variants = {e.value for e in node.value.elts if isinstance(e, ast.Constant)}
    self.assertIn(DECODE_COMPARATOR, variants, "DECODE_COMPARATOR is not in FLASH_DECODE_VARIANTS")


if __name__ == "__main__":
  unittest.main()
