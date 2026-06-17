#!/usr/bin/env python3
"""Lock the flash-decode auto-select policy (tinygrad.llm.model.should_use_flash_decode). Tests the policy logic
directly -- no model weights. Monkeypatches model.getenv (getenv is cached) to control FLASH_DECODE mode +
threshold. Invariants: decode-only (T==1, symbolic start_pos); force off/on; auto threshold on trace-time context;
conservative SDPA fallback."""
import unittest
from tinygrad import UOp
import tinygrad.llm.model as M

def _sp(pos):  # a bound symbolic start_pos, as the decode JIT passes it
  return UOp.variable("start_pos", 0, 4095).bind(pos)

class _Env:
  def __init__(self, **kv): self.kv = kv
  def __enter__(self):
    self._orig = M.getenv
    M.getenv = lambda k, d=0: self.kv.get(k, d)
    return self
  def __exit__(self, *a): M.getenv = self._orig

class TestFlashDecodePolicy(unittest.TestCase):
  def test_auto_below_threshold_uses_sdpa(self):
    with _Env(FLASH_DECODE="auto", FLASH_DECODE_THRESHOLD=1024):
      self.assertFalse(M.should_use_flash_decode(_sp(500), 1))    # ctx 501 < 1024

  def test_auto_above_threshold_uses_flash(self):
    with _Env(FLASH_DECODE="auto", FLASH_DECODE_THRESHOLD=1024):
      self.assertTrue(M.should_use_flash_decode(_sp(1500), 1))    # ctx 1501 >= 1024

  def test_force_off_always_sdpa(self):
    with _Env(FLASH_DECODE="0", FLASH_DECODE_THRESHOLD=1024):
      self.assertFalse(M.should_use_flash_decode(_sp(2000), 1))   # even long ctx
      self.assertFalse(M.should_use_flash_decode(_sp(2000), 1, use_flash=True))  # 0 overrides programmatic on

  def test_force_on_below_threshold(self):
    with _Env(FLASH_DECODE="1", FLASH_DECODE_THRESHOLD=1024):
      self.assertTrue(M.should_use_flash_decode(_sp(10), 1))      # force on ignores threshold

  def test_programmatic_use_flash(self):
    with _Env(FLASH_DECODE="auto", FLASH_DECODE_THRESHOLD=1024):
      self.assertTrue(M.should_use_flash_decode(_sp(10), 1, use_flash=True))

  def test_decode_only_invariant(self):
    with _Env(FLASH_DECODE="1", FLASH_DECODE_THRESHOLD=1024):
      self.assertFalse(M.should_use_flash_decode(_sp(2000), 2))   # T!=1 (prefill) -> never flash
      self.assertFalse(M.should_use_flash_decode(2000, 1))        # concrete start_pos (not symbolic) -> never flash

  def test_unreadable_context_falls_back_to_sdpa(self):
    with _Env(FLASH_DECODE="auto", FLASH_DECODE_THRESHOLD=1024):
      self.assertFalse(M.should_use_flash_decode(UOp.variable("start_pos", 0, 4095), 1))  # unbound var, no value

  def test_threshold_boundary(self):
    with _Env(FLASH_DECODE="auto", FLASH_DECODE_THRESHOLD=1024):
      self.assertTrue(M.should_use_flash_decode(_sp(1023), 1))    # ctx 1024 >= 1024

  def test_default_threshold_is_512(self):
    # Arc 1: default cutover lowered 1024->512 (measured +12.8% real-generate @ctx520, byte-identical greedy,
    # no regression <512 which stays SDPA). No FLASH_DECODE_THRESHOLD env -> getenv default applies.
    with _Env(FLASH_DECODE="auto"):
      self.assertFalse(M.should_use_flash_decode(_sp(255), 1))    # ctx 256 < 512 -> SDPA (flash regresses here)
      self.assertFalse(M.should_use_flash_decode(_sp(510), 1))    # ctx 511 < 512 -> SDPA
      self.assertTrue(M.should_use_flash_decode(_sp(511), 1))     # ctx 512 >= 512 -> flash
      self.assertTrue(M.should_use_flash_decode(_sp(1022), 1))    # ctx 1023 >= 512 -> flash (long ctx preserved)

if __name__ == "__main__":
  unittest.main()
