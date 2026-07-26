"""Contract for extra/qk/bench.py's throughput scan.

The entry-point hardening treats "no parsable throughput number" as a failure even when the child exits 0. That is
right, but it only works if the scan looks where the measurement authorities actually print. decode_runtime_overhead.py
prints its `ctx N: W ..ms (.. tok/s)` rows to stderr, so a stdout-only scan failed every successful decode run and
silently discarded valid measurements. These tests pin both directions: a real number must be found on either stream,
and a genuinely empty run must still fail loudly.
"""
import re
import pytest

from extra.qk.bench import _decode_ckpts, _run, BelowPerfFloor, NoThroughputProduced

DECODE_RE = re.compile(r"ctx\s*\d+:\s*W\s*[\d.]+ms\s*\(([\d.]+)\s*tok/s\)")
ROW = "ctx   512: W   8.82ms (113.35 tok/s) | D   9.08ms (110.19 tok/s)"

def _emit(tmp_path, stream: str, text: str) -> list[str]:
  """A real script file: _run verifies its dispatch target exists before executing it."""
  script = tmp_path / "emit_probe.py"
  script.write_text(f"import sys\nprint({text!r}, file=sys.{stream})\n")
  return [str(script)]

def test_throughput_is_found_on_stdout(tmp_path, capfd):
  assert _run("probe", _emit(tmp_path, "stdout", ROW), {}, throughput_re=DECODE_RE) == 0

def test_throughput_is_found_on_stderr(tmp_path, capfd):
  """The decode authority prints here; a stdout-only scan regressed this into a guaranteed failure."""
  assert _run("probe", _emit(tmp_path, "stderr", ROW), {}, throughput_re=DECODE_RE) == 0

def test_a_run_that_produces_no_number_still_fails_loudly(tmp_path, capfd):
  with pytest.raises(NoThroughputProduced):
    _run("probe", _emit(tmp_path, "stdout", "no measurement here"), {}, throughput_re=DECODE_RE)

def test_min_value_is_enforced_against_the_scanned_number(tmp_path, capfd):
  """Also proves the floor is checked against the number scanned off stderr, not a default."""
  with pytest.raises(BelowPerfFloor, match="113.35"):
    _run("probe", _emit(tmp_path, "stderr", ROW), {}, throughput_re=DECODE_RE, min_value=1e6)


def test_decode_checkpoint_defaults_are_model_aware():
  assert _decode_ckpts(None, "qwen3_8b_q4k_m_gfx1100") is None
  assert _decode_ckpts(None, "qwen3_14b_q4k_m_gfx1100") == (512, 1024, 4096)
  assert _decode_ckpts("128", "qwen3_14b_q4k_m_gfx1100") == (128,)
