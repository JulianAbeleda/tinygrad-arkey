"""The QK primitive/generated paths are AMD-targeted; from_gguf must fail fast on
another backend instead of failing obscurely later in the kernels. Simulated by
stubbing the module-level Device (no GPU); the guard fires before any gguf load."""
import os, unittest
from unittest import mock

import tinygrad.llm.model as model
from tinygrad import getenv
from tinygrad.llm.model import Transformer

_KEYS = ("Q4K_PRIMITIVE", "Q6K_PRIMITIVE", "QK_GENERATED_POLICY")

class _StubDevice:
  DEFAULT = "CPU"

class TestQKAmdGuard(unittest.TestCase):
  def setUp(self):
    self._saved = {k: os.environ.get(k) for k in _KEYS}
    for k in _KEYS: os.environ.pop(k, None)
    getenv.cache_clear()

  def tearDown(self):
    for k, v in self._saved.items():
      if v is None: os.environ.pop(k, None)
      else: os.environ[k] = v
    getenv.cache_clear()

  def _expect_amd_error(self):
    with mock.patch.object(model, "Device", _StubDevice):
      getenv.cache_clear()
      with self.assertRaisesRegex(ValueError, "DEV=AMD"):
        Transformer.from_gguf("nonexistent.gguf")  # guard raises before the file is touched

  def test_explicit_q4k_primitive_on_non_amd_raises(self):
    os.environ["Q4K_PRIMITIVE"] = "1"
    self._expect_amd_error()

  def test_explicit_q6k_primitive_on_non_amd_raises(self):
    os.environ["Q6K_PRIMITIVE"] = "1"
    self._expect_amd_error()

  def test_generated_policy_on_non_amd_raises(self):
    os.environ["QK_GENERATED_POLICY"] = "some/policy.json"
    self._expect_amd_error()

  def test_no_primitives_does_not_hit_amd_guard(self):
    # Q4K_PRIMITIVE=0 -> no primitive path -> the AMD guard must NOT fire (a different error,
    # from trying to load the missing file, is expected instead).
    os.environ["Q4K_PRIMITIVE"] = "0"
    with mock.patch.object(model, "Device", _StubDevice):
      getenv.cache_clear()
      with self.assertRaises(Exception) as ctx:
        Transformer.from_gguf("nonexistent.gguf")
      self.assertNotIn("DEV=AMD", str(ctx.exception))

if __name__ == "__main__":
  unittest.main()
