"""CPU-only contract checks for the matched Q6_K role diagnostic."""
import importlib.util
from pathlib import Path


def _module():
  path = Path(__file__).parents[2] / "scratchpad/q6k_matched_tinygrad_role_benchmark.py"
  spec = importlib.util.spec_from_file_location("q6k_matched_role", path)
  assert spec is not None and spec.loader is not None
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def test_matched_role_is_pinned_to_observed_kv_shape():
  mod = _module()
  try:
    mod.live(512, 4096, 1, 1, 1, Path("/missing"))
  except ValueError as exc:
    assert "1024x4096" in str(exc)
  else: assert False, "unexpected shape must fail before any GPU/library access"


def test_matched_role_documents_q8_to_fp16_bridge():
  text = (Path(__file__).parents[2] / "scratchpad/q6k_matched_tinygrad_role_benchmark.py").read_text()
  assert "same Q8_1 payload independently decoded then rounded to fp16" in text
  assert "partials.sum(axis=1).contiguous()" in text
