"""Regression tests for the recompute-hostile-producer COST GATE in remove_bufferize
(tinygrad/schedule/rangeify.py, _RECOMPUTE_HOSTILE_OPS / _consumer_parallelism_and_trip).

The gate refuses to inline (fuse) a producer containing a transcendental op into a consumer
when that consumer's iteration space has too few independent outputs to hide the transcendental's
latency (low "parallelism") behind a long serialized reduce ("trip"). It must fire on wide-vocab
Gumbel-max sampling (argmax lowers to a near-single-workgroup reduce) and must NOT fire on prefill
attention softmax (softmax's reduce is spread across many independent rows/heads).

A prior version of this gate keyed purely on raw producer element count (prod(buf.shape)), which
does not distinguish these two cases -- both producers are "wide" -- and cost prefill ~2.5%
throughput by refusing to fuse the softmax exp into its consumers.

These tests instrument remove_bufferize directly (rather than asserting on final kernel counts)
because that is the exact function the cost gate lives in and the exact quantities the task is
about. Two method pitfalls this file works around, both hit for real while writing it:

  1. PatternMatcher captures its callback at construction time. Monkeypatching
     rangeify.remove_bufferize alone instruments nothing -- pm_remove_bufferize must be rebuilt
     to reference the wrapper.
  2. Compiled PatternMatchers reject closures. pm_remove_bufferize gets re-combined with other
     matchers via `+` (tinygrad/schedule/rangeify.py:1029, `graph_rewrite(..., ...+pm_remove_bufferize,
     ...)`), and PatternMatcher.__add__ recompiles with UPAT_COMPILE's default (compiled=True),
     which ignores any compiled=False passed when constructing pm_remove_bufferize standalone. So
     the wrapper must be a plain module-level function using module-level state, not a closure.
"""
import os
os.environ.setdefault("SCACHE", "0")

from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import Ops, UPat, PatternMatcher
import tinygrad.schedule.rangeify as rangeify

# module-level (non-closure) recording state, toggled per-test via _instrument/_restore
_records: list[tuple[tuple, bool]] = []
_orig_remove_bufferize = rangeify.remove_bufferize


def _wrapped_remove_bufferize(src, buf, idx):
  ret = _orig_remove_bufferize(src, buf, idx)
  if any(u.op in rangeify._RECOMPUTE_HOSTILE_OPS for u in src.toposort()):
    _records.append((buf.shape, ret is None))
  return ret


def _instrument():
  _records.clear()
  rangeify.remove_bufferize = _wrapped_remove_bufferize
  rangeify.pm_remove_bufferize = PatternMatcher([
    (UPat.var("src").f(Ops.STAGE, allow_any_len=True, name="buf").f(Ops.INDEX, allow_any_len=True, name="idx"), _wrapped_remove_bufferize),
  ] + rangeify.pm_remove_bufferize.patterns[1:])


def _restore():
  rangeify.remove_bufferize = _orig_remove_bufferize
  rangeify.pm_remove_bufferize = PatternMatcher([
    (UPat.var("src").f(Ops.STAGE, allow_any_len=True, name="buf").f(Ops.INDEX, allow_any_len=True, name="idx"), _orig_remove_bufferize),
  ] + rangeify.pm_remove_bufferize.patterns[1:])


def test_wide_vocab_gumbel_argmax_gates():
  """Gumbel-max sampling over a full vocab: two LOG2 calls feeding TWO argmax reductions. argmax
  lowers to a low-parallelism reduce (measured: 128, then 1, independent outputs), so the gate must
  refuse to inline the hostile producer -- this is the pathology commit 04f7e3f1b fixed."""
  _instrument()
  try:
    VOCAB = 151936
    logits = Tensor.empty(VOCAB, dtype=dtypes.float32)
    u = Tensor.empty(VOCAB, dtype=dtypes.float32)
    gumbel = -((-u.log()).log())
    y = logits + gumbel
    idx1 = y.argmax()
    idx2 = (y * 2).argmax()
    (idx1 + idx2).schedule_linear()
    records = list(_records)
  finally:
    _restore()

  assert records, "no hostile-op call sites were observed; instrumentation did not fire"
  assert any(gated for _, gated in records), f"expected at least one gated call site, got {records}"


def test_prefill_attention_softmax_does_not_gate():
  """softmax(-1) inside scaled-dot-product attention: EXP2 feeding a reduce whose independent-output
  count (heads x query positions, measured 131072) is orders of magnitude above the pathological
  case. Fusion here is fine and the gate must not block it -- this is the prefill regression this
  predicate revision fixes (commit 04f7e3f1b's raw-width gate refused it and cost ~2.5% throughput)."""
  _instrument()
  try:
    H, S, D = 32, 4096, 128
    q = Tensor.empty(1, H, S, D, dtype=dtypes.float16)
    k = Tensor.empty(1, H, S, D, dtype=dtypes.float16)
    v = Tensor.empty(1, H, S, D, dtype=dtypes.float16)
    scores = q @ k.transpose(-2, -1)
    sm = scores.softmax(-1)
    (sm @ v).schedule_linear()
    records = list(_records)
  finally:
    _restore()

  assert records, "no hostile-op call sites were observed; instrumentation did not fire"
  assert not any(gated for _, gated in records), f"attention softmax must not be gated, got {records}"


if __name__ == "__main__":
  test_wide_vocab_gumbel_argmax_gates()
  test_prefill_attention_softmax_does_not_gate()
  print("OK")
