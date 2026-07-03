import pytest

from tinygrad.llm.admission import AUTO_MAX_CONTEXT, MIN_USABLE_CTX, VRAM_ADMIT_FRACTION, resolve_max_context_admission

def _admit(requested, *, trained_ctx=32768, free=24_000_000_000, weights=8_000_000_000,
           kv_per_tok=1_000_000, prefill_per_tok=65_536, scratch=1_000_000):
  return resolve_max_context_admission(requested, trained_ctx, free, weights, kv_per_tok, prefill_per_tok, scratch, "test-model")

def test_auto_context_uses_memory_cap_but_never_exceeds_trained_ctx():
  ctx, kv_quant, report = _admit(AUTO_MAX_CONTEXT, free=128_000_000_000, trained_ctx=8192)
  assert ctx == 8192
  assert kv_quant is False
  assert report["mode"] == "auto"
  assert report["trained_ctx"] == 8192

def test_auto_context_uses_admissible_memory_cap():
  free, weights, kv_per_tok, prefill_per_tok, scratch = 24_000_000_000, 8_000_000_000, 1_000_000, 65_536, 1_000_000
  expected = int((free * VRAM_ADMIT_FRACTION - weights - scratch) // (kv_per_tok + prefill_per_tok))
  ctx, kv_quant, report = _admit(AUTO_MAX_CONTEXT, free=free, weights=weights, kv_per_tok=kv_per_tok,
                                 prefill_per_tok=prefill_per_tok, scratch=scratch)
  assert ctx == expected
  assert kv_quant is False
  assert report["mc_mem"] == expected

def test_auto_context_refuses_below_min_usable_context():
  with pytest.raises(RuntimeError, match="auto-scan needs"):
    _admit(AUTO_MAX_CONTEXT, free=10_000_000_000, weights=9_000_000_000, kv_per_tok=2_000_000)

def test_explicit_context_is_admission_checked():
  ctx, kv_quant, report = _admit(4096)
  assert ctx == 4096
  assert kv_quant is False
  assert report["mode"] == "explicit"
  with pytest.raises(RuntimeError, match="needs 16384 tokens"):
    _admit(16384)

def test_no_probe_auto_fails_but_explicit_uses_trained_clamp():
  with pytest.raises(RuntimeError, match="auto needs a VRAM free-probe"):
    _admit(AUTO_MAX_CONTEXT, free=None)
  ctx, kv_quant, report = _admit(65536, free=None, trained_ctx=32768)
  assert ctx == 32768
  assert kv_quant is False
  assert report == {"mode": "explicit_no_probe", "max_context": 32768, "trained_ctx": 32768}

def test_auto_context_escalates_to_q8_when_supported():
  ctx, kv_quant, report = resolve_max_context_admission(AUTO_MAX_CONTEXT, 32768, 14_000_000_000, 9_000_000_000,
                                                        2_000_000, 1_000, 0, "test-model",
                                                        kv_quant_supported=True, scale_per_tok=32_000)
  assert ctx >= MIN_USABLE_CTX
  assert kv_quant is True
  assert report["mode"] == "auto+q8"

def test_min_usable_context_default_documents_guardrail():
  assert MIN_USABLE_CTX == 2048
