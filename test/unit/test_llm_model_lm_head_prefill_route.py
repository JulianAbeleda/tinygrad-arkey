"""Host-side (no GPU) proof that Transformer.logits() selects the packed prefill
route for self.output ONLY on a concrete prefill-v2 batch (e.g. T=512), and take the byte-for-byte unchanged
decode (T==1) path otherwise -- the wiring described in the LM-head prefill-route task.

This tests ROUTE SELECTION only (which callable gets invoked for the LM head), never a real kernel: `_pf16` and
`is_direct_packed_prefill_linear` are monkeypatched to recorder stubs, so no CustomKernel/GPU device is touched.
"""
from types import SimpleNamespace

from tinygrad.llm.model import Transformer


class _XStub:
  """Minimal activation-tensor stand-in for the chained calls logits() makes on `x`."""
  device = "CPU"
  def float(self): return self
  def contiguous(self): return self
  def realize(self): return self


class _FakeBlock:
  """Stands in for a real FFNBlock/MLATransformerBlock. Only `_prefill_v2` matters here -- it is the same
  phase flag the per-block linears already gate on (see model.py:401's `getattr(self, '_prefill_v2', False)`),
  set by Transformer.__call__ before dispatch: True only for a concrete-batch prefill-v2 forward, never decode."""
  def __init__(self, prefill_v2:bool): self._prefill_v2 = prefill_v2
  def __call__(self, x, start_pos): return x
  def _init_state(self, x): pass


class _FakeOutputLinear:
  """Stands in for self.output: either a plain nn.Linear (primitives not installed) or an installed Q4_K/Q6_K
  direct-packed primitive, toggled via `_is_direct_packed` and read by the monkeypatched
  `is_direct_packed_prefill_linear`."""
  def __init__(self, is_direct_packed:bool):
    self._is_direct_packed = is_direct_packed
  def __call__(self, x):
    return ("plain_output_call", x)


class _FakeTransformer:
  # Reuse the real gate + logits implementations under test; only construct lightweight stand-ins for the
  # attributes/collaborators they touch, so this never needs a real GGUF-loaded model or a GPU device.
  _lm_head_wants_pf16 = Transformer._lm_head_wants_pf16
  logits = Transformer.logits

  def __init__(self, *, prefill_v2:bool, output_is_direct_packed:bool):
    self.config = SimpleNamespace(lm_head_route="lazy")
    self.blk = [_FakeBlock(prefill_v2)]
    self.output = _FakeOutputLinear(output_is_direct_packed)
    self.output_norm = lambda x: x
    self.token_embd = lambda tokens: _XStub()


def _install_stubs(monkeypatch):
  from tinygrad.llm import model as model_mod
  calls = []

  class _PF16Result:
    def __init__(self, lin, x): self.lin, self.x = lin, x
    def contiguous(self): return self

  def fake_pf16(lin, x):
    calls.append((lin, x))
    return _PF16Result(lin, x)

  monkeypatch.setattr(model_mod, "_pf16", fake_pf16)
  monkeypatch.setattr(model_mod, "_prefill_semantic", lambda _enabled, _mark, value, _runtime_mark=None: value)
  monkeypatch.setattr(model_mod, "is_direct_packed_prefill_linear", lambda lin: bool(getattr(lin, "_is_direct_packed", False)))
  return calls


def test_logits_keeps_lm_head_lazy_by_default_for_t512_prefill_v2_batch(monkeypatch):
  calls = _install_stubs(monkeypatch)
  fake = _FakeTransformer(prefill_v2=True, output_is_direct_packed=True)
  tokens = SimpleNamespace(shape=(1, 512))

  out = fake.logits(tokens, 0)

  assert calls == []
  assert out[0] == "plain_output_call"


def test_logits_routes_lm_head_through_pf16_when_full_sequence_route_is_explicit(monkeypatch):
  calls = _install_stubs(monkeypatch)
  fake = _FakeTransformer(prefill_v2=True, output_is_direct_packed=True)
  fake.config.lm_head_route = "resident_fp16"
  tokens = SimpleNamespace(shape=(1, 512))

  out = fake.logits(tokens, 0)

  assert len(calls) == 1, "self.output must be routed through _pf16 exactly once for the T=512 prefill-v2 batch"
  routed_lin, _routed_x = calls[0]
  assert routed_lin is fake.output
  assert not isinstance(out, tuple), "output must come from the _pf16 route, not the plain self.output(x) call"


def test_logits_keeps_decode_t1_path_unchanged(monkeypatch):
  calls = _install_stubs(monkeypatch)
  fake = _FakeTransformer(prefill_v2=False, output_is_direct_packed=True)
  tokens = SimpleNamespace(shape=(1, 1))

  out = fake.logits(tokens, 5)

  assert calls == [], "decode (T==1, _prefill_v2=False) must never call _pf16 for the LM head"
  assert out[0] == "plain_output_call", "decode must keep calling self.output(x) exactly as before this change"


def test_logits_keeps_plain_linear_path_when_primitives_not_installed(monkeypatch):
  # Even on a T=512 prefill-v2 batch, a plain nn.Linear self.output (primitives never installed, or this
  # gguf's output.weight isn't Q4_K/Q6_K) must keep calling self.output(x) exactly as before this change.
  calls = _install_stubs(monkeypatch)
  fake = _FakeTransformer(prefill_v2=True, output_is_direct_packed=False)
  tokens = SimpleNamespace(shape=(1, 512))

  out = fake.logits(tokens, 0)

  assert calls == [], "plain nn.Linear LM head must not be routed through _pf16"
  assert out[0] == "plain_output_call"
