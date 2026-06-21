#!/usr/bin/env python3
"""Child-env SSOT (audit C6).

extra/qk_harness_contract.py:child_env is the single builder for a spawned QK eval subprocess env. This pins the
keys it must set so the evaluator and the lifecycle loop (which both launch decode_eval children) cannot drift.
No GPU / no tinygrad import.

Run: PYTHONPATH=. python -m unittest test.unit.test_child_env_ssot -v
"""
from __future__ import annotations
import os
import unittest

from extra.qk_harness_contract import child_env, DEFAULT_MODEL


class TestChildEnvSSOT(unittest.TestCase):
  def test_required_keys_present(self):
    e = child_env()
    for k in ("DEV", "JIT", "PYTHONPATH", "QK_MODEL"):
      self.assertIn(k, e, f"child_env must set {k}")
    self.assertEqual(e["DEV"], os.environ.get("DEV", "AMD"))
    self.assertEqual(e["JIT"], os.environ.get("JIT", "1"))
    self.assertEqual(e["QK_MODEL"], os.environ.get("QK_MODEL", DEFAULT_MODEL))

  def test_setdefault_respects_existing_dev(self):
    # DEV/JIT use setdefault: an already-set value wins (env-ordering safety).
    orig = os.environ.get("DEV")
    os.environ["DEV"] = "CPU"
    try:
      self.assertEqual(child_env()["DEV"], "CPU")
    finally:
      if orig is None: del os.environ["DEV"]
      else: os.environ["DEV"] = orig

  def test_extra_overrides(self):
    e = child_env({"FLASH_VARIANT": "gqa_coop_vec", "FOO": 1})
    self.assertEqual(e["FLASH_VARIANT"], "gqa_coop_vec")
    self.assertEqual(e["FOO"], "1")  # values are stringified


if __name__ == "__main__":
  unittest.main()
