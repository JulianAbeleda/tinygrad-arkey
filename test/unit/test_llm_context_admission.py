import pytest

from tinygrad.llm.admission import AUTO_MAX_CONTEXT, _resolve_max_context_admission
from tinygrad.llm.memory_ledger import AllocationProvenance, ScannedMemoryBudget

_TEST_ADMIT_FRACTION = 0.8

def _budget(free):
  reserve = None if free is None else int(free * (1.0-_TEST_ADMIT_FRACTION))
  return ScannedMemoryBudget(free, reserve, AllocationProvenance("unit test", "private fixed-fraction test fixture"))

def _admit(requested, *, trained_ctx=32768, free=24_000_000_000, weights=8_000_000_000,
           kv_per_tok=1_000_000, prefill_per_tok=65_536, scratch=1_000_000, stream="auto", ring_supported=False,
           kv_quant_supported=False, scale_per_tok=0):
  return _resolve_max_context_admission(requested, trained_ctx, _budget(free), weights, kv_per_tok, prefill_per_tok,
                                        scratch, "test-model", kv_quant_supported=kv_quant_supported,
                                        scale_per_tok=scale_per_tok, stream=stream, ring_supported=ring_supported)

def test_stream_on_forces_ring_even_when_lossless_fits():
  # huge free -> fp16 would admit trained ctx; stream=on forces the lossy ring instead (unbounded generation)
  ctx, kv_quant, report = _admit(AUTO_MAX_CONTEXT, free=128_000_000_000, trained_ctx=8192, stream="on", ring_supported=True)
  assert report["mode"] == "ring" and report["ring"] is True and kv_quant is False
  assert ctx == 8192 and "banner" in report

def test_auto_falls_to_ring_only_when_no_exact_candidate_fits():
  # Neither exact candidate reaches the trained context, so the positive
  # candidate-local fp16 window is selected as the lossy fallback.
  ctx, kv_quant, report = _admit(AUTO_MAX_CONTEXT, free=13_000_000_000, weights=9_000_000_000,
                                 kv_per_tok=900_000, scale_per_tok=200_000, kv_quant_supported=True, ring_supported=True)
  assert report["mc_fp16"] < report["trained_ctx"] and report["mc_q8"] < report["trained_ctx"]
  assert report["mode"] == "ring" and report["ring"] is True and ctx == report["mc_ring"] and ctx > 0 and kv_quant is False

def test_stream_off_refuses_instead_of_ring():
  with pytest.raises(RuntimeError, match="Refusing|needs"):
    _admit(AUTO_MAX_CONTEXT, free=10_000_000_000, weights=9_000_000_000, kv_per_tok=2_000_000,
           kv_quant_supported=True, ring_supported=True, stream="off")

def test_stream_on_accepts_any_positive_structural_window():
  ctx, _, report = _admit(AUTO_MAX_CONTEXT, free=12_000_000_000, weights=9_000_000_000,
                          kv_per_tok=8_000_000, ring_supported=True, stream="on")
  assert ctx == report["mc_ring"] and ctx > 0

def test_explicit_int_never_rescued_by_ring():
  # explicit --max_context that doesn't fit fp16/Q8 refuses even with ring_supported (N is physical); hint mentions --stream
  with pytest.raises(RuntimeError, match="Largest admissible"):
    _admit(30000, free=10_000_000_000, weights=9_000_000_000, kv_per_tok=2_000_000, kv_quant_supported=True,
           ring_supported=True, stream="auto")

def test_auto_context_uses_memory_cap_but_never_exceeds_trained_ctx():
  ctx, kv_quant, report = _admit(AUTO_MAX_CONTEXT, free=128_000_000_000, trained_ctx=8192)
  assert ctx == 8192
  assert kv_quant is False
  assert report["mode"] == "auto"
  assert report["trained_ctx"] == 8192

def test_auto_context_uses_admissible_memory_cap():
  free, weights, kv_per_tok, prefill_per_tok, scratch = 24_000_000_000, 8_000_000_000, 1_000_000, 65_536, 1_000_000
  expected = int((free * _TEST_ADMIT_FRACTION - weights - scratch) // (kv_per_tok + prefill_per_tok))
  ctx, kv_quant, report = _admit(AUTO_MAX_CONTEXT, free=free, weights=weights, kv_per_tok=kv_per_tok,
                                 prefill_per_tok=prefill_per_tok, scratch=scratch, trained_ctx=8192)
  assert ctx == 8192
  assert kv_quant is False
  assert report["mc_mem"] == expected

def test_auto_context_refuses_when_exact_arithmetic_has_no_capacity():
  with pytest.raises(RuntimeError, match="auto-scan needs"):
    _admit(AUTO_MAX_CONTEXT, free=9_000_000_000, weights=9_000_000_000, kv_per_tok=2_000_000)

def test_explicit_context_is_admission_checked():
  ctx, kv_quant, report = _admit(4096)
  assert ctx == 4096
  assert kv_quant is False
  assert report["mode"] == "explicit"
  with pytest.raises(RuntimeError, match="needs 16384 tokens"):
    _admit(16384)

def test_no_probe_fails_closed_for_auto_and_explicit_context():
  with pytest.raises(RuntimeError, match="requires scanned total/free VRAM and allocator granularity"):
    _admit(AUTO_MAX_CONTEXT, free=None)
  with pytest.raises(RuntimeError, match="requires scanned total/free VRAM and allocator granularity"):
    _admit(65536, free=None, trained_ctx=32768)

def test_auto_context_escalates_to_q8_when_supported():
  ctx, kv_quant, report = _resolve_max_context_admission(AUTO_MAX_CONTEXT, 2000, _budget(14_000_000_000),
                                                         9_000_000_000, 2_000_000, 1_000, 0, "test-model",
                                                         kv_quant_supported=True, scale_per_tok=32_000)
  assert ctx == 2000
  assert kv_quant is True
  assert report["mode"] == "auto+q8"

def test_environment_cannot_change_admission(monkeypatch):
  kwargs = dict(free=13_000_000_000, weights=9_000_000_000, kv_per_tok=900_000,
                ring_supported=True, kv_quant_supported=True, scale_per_tok=200_000)
  baseline = _admit(AUTO_MAX_CONTEXT, **kwargs)
  monkeypatch.setenv("MIN_USABLE_CTX", "999999999")
  monkeypatch.setenv("MIN_RING_WINDOW", "999999999")
  assert _admit(AUTO_MAX_CONTEXT, **kwargs) == baseline
