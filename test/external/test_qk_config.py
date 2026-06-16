"""Byte-proof for QKConfig.from_env (tinygrad/llm/model.py).

QKConfig centralizes the QK primitive *install* env reads that from_gguf used to do
inline. This pins that each field reproduces the exact original getenv expression
across an env matrix, plus the validation raises. Pure CPU, no model load / no GPU.
"""
import os, unittest
from contextlib import contextmanager

from tinygrad import getenv
from tinygrad.llm.model import QKConfig, _qk_storage_cap_from_env, _qk_storage_mode_from_env, _q6k_effective_storage_mode

QK_KEYS = ("QK_GENERATED_POLICY_STRICT", "QK_PRIMITIVE_MAX_STORAGE_MB", "QK_PRIMITIVE_STORAGE",
           "QK_GENERATED_POLICY_DEBUG", "Q4K_PRIMITIVE_DEBUG", "Q6K_PRIMITIVE_DEBUG", "Q6K_DEMOTE_FFNDOWN", "Q4K_FUSE")

@contextmanager
def _env(**overrides):
  """Apply env overrides (None deletes), clear the getenv cache so reads are fresh, restore on exit."""
  saved = {k: os.environ.get(k) for k in QK_KEYS}
  try:
    for k in QK_KEYS: os.environ.pop(k, None)
    for k, v in overrides.items():
      if v is not None: os.environ[k] = v
    getenv.cache_clear()
    yield
  finally:
    for k, v in saved.items():
      if v is None: os.environ.pop(k, None)
      else: os.environ[k] = v
    getenv.cache_clear()

def _original(storage_default):
  """The exact scattered expressions QKConfig.from_env replaced, recomputed live."""
  storage_mode = _qk_storage_mode_from_env(storage_default)
  return dict(
    generated_policy_strict=bool(getenv("QK_GENERATED_POLICY_STRICT", 0)),
    max_storage_bytes=_qk_storage_cap_from_env(),
    storage_mode=storage_mode,
    q6_storage_mode=_q6k_effective_storage_mode(storage_mode),
    policy_debug=bool(getenv("QK_GENERATED_POLICY_DEBUG", 0)),
    storage_debug=bool(getenv("QK_GENERATED_POLICY_DEBUG", getenv("Q4K_PRIMITIVE_DEBUG", getenv("Q6K_PRIMITIVE_DEBUG", 0)))),
    demote_q6k_ffndown=bool(getenv("Q6K_DEMOTE_FFNDOWN")),
    fuse_q4k=bool(getenv("Q4K_FUSE")))

class TestQKConfig(unittest.TestCase):
  def _assert_matches(self, storage_default, **env):
    with _env(**env):
      cfg = QKConfig.from_env(storage_default=storage_default)
      expected = _original(storage_default)  # same (freshly-cleared) getenv cache -> identical reads
    self.assertEqual({f: getattr(cfg, f) for f in expected}, expected)

  def test_all_unset_sidecar_default(self):
    self._assert_matches("sidecar")

  def test_all_unset_shared_default(self):
    self._assert_matches("shared")

  def test_storage_mode_override_wins_over_default(self):
    self._assert_matches("sidecar", QK_PRIMITIVE_STORAGE="shared")
    self._assert_matches("shared", QK_PRIMITIVE_STORAGE="sidecar")

  def test_q6_effective_tracks_q4(self):
    with _env(QK_PRIMITIVE_STORAGE="q4_ondemand"):
      cfg = QKConfig.from_env(storage_default="sidecar")
    self.assertEqual((cfg.storage_mode, cfg.q6_storage_mode), ("q4_ondemand", "sidecar"))

  def test_flags_and_cap(self):
    self._assert_matches("sidecar", QK_GENERATED_POLICY_STRICT="1", Q4K_FUSE="1", Q6K_DEMOTE_FFNDOWN="1",
                         QK_PRIMITIVE_MAX_STORAGE_MB="512")

  def test_storage_debug_fallback_chain(self):
    # storage_debug = QK_GENERATED_POLICY_DEBUG || Q4K_PRIMITIVE_DEBUG || Q6K_PRIMITIVE_DEBUG
    self._assert_matches("sidecar", Q6K_PRIMITIVE_DEBUG="1")
    self._assert_matches("sidecar", Q4K_PRIMITIVE_DEBUG="1")
    self._assert_matches("sidecar", QK_GENERATED_POLICY_DEBUG="1")

  def test_cap_resolves_to_bytes(self):
    with _env(QK_PRIMITIVE_MAX_STORAGE_MB="2"):
      cfg = QKConfig.from_env(storage_default="sidecar")
    self.assertEqual(cfg.max_storage_bytes, 2 * 1024 * 1024)

  def test_invalid_storage_mode_raises(self):
    with _env(QK_PRIMITIVE_STORAGE="bogus"):
      with self.assertRaises(ValueError): QKConfig.from_env(storage_default="sidecar")

  def test_negative_cap_raises(self):
    with _env(QK_PRIMITIVE_MAX_STORAGE_MB="-1"):
      with self.assertRaises(ValueError): QKConfig.from_env(storage_default="sidecar")

if __name__ == "__main__":
  unittest.main()
