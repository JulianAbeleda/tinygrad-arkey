"""Per-site REDUCE_OUTPUT decision flags for the decode norm call sites.

The fp32 q/k site is promoted by the global
``_decode_reduce_output_rmsnorm_promoted`` route in production.  The FFN-norm
site has an independent knob and is CLOSED by default (its GPU wall bracket is
net-negative), so a census can keep the fp32 q/k site promoted while the FFN
site stays off.
"""

from types import SimpleNamespace

from tinygrad.llm.model import _decode_reduce_output_norm_flags
from tinygrad.llm.shared_q8_attention import SharedQ8AttentionAdmission


def _block(**kwargs):
  return SimpleNamespace(**kwargs)


def test_global_route_promotes_both_sites_by_default():
  # No per-site knob: the FFN site follows the global route, matching the
  # pre-split production behavior.
  assert _decode_reduce_output_norm_flags(_block(_decode_reduce_output_rmsnorm_promoted=True), False) == (True, True)


def test_ffn_knob_closes_ffn_site_independently():
  # The new measurement arm: q/k stays promoted, FFN-norm closes.
  block = _block(_decode_reduce_output_rmsnorm_promoted=True, _decode_reduce_output_ffn_rmsnorm_promoted=False)
  assert _decode_reduce_output_norm_flags(block, False) == (True, False)


def test_ffn_knob_can_promote_without_global_route():
  block = _block(_decode_reduce_output_rmsnorm_promoted=False, _decode_reduce_output_ffn_rmsnorm_promoted=True)
  assert _decode_reduce_output_norm_flags(block, False) == (False, True)


def test_prefill_always_closes_both_sites():
  block = _block(_decode_reduce_output_rmsnorm_promoted=True, _decode_reduce_output_ffn_rmsnorm_promoted=True)
  assert _decode_reduce_output_norm_flags(block, True) == (False, False)


def test_shared_q8_attention_lease_promotes_attention_only():
  block = _block(
    _decode_reduce_output_rmsnorm_promoted=False,
    _decode_reduce_output_ffn_rmsnorm_promoted=False,
    _shared_q8_attention_admission=SharedQ8AttentionAdmission(0),
    _decode_reduce_output_attn_rmsnorm_promoted=True,
  )
  assert _decode_reduce_output_norm_flags(block, False) == (True, False)

def test_native_attention_norm_bypasses_global_reduce_output_route():
  block = _block(_decode_reduce_output_rmsnorm_promoted=True,
    attn_norm=SimpleNamespace(_rmsnorm_native_promoted=True))
  assert _decode_reduce_output_norm_flags(block, False) == (False, True)

def test_shared_q8_provider_retains_reduce_output_ownership_over_native_site():
  block = _block(_decode_reduce_output_rmsnorm_promoted=True,
    attn_norm=SimpleNamespace(_rmsnorm_native_promoted=True),
    _shared_q8_attention_admission=SharedQ8AttentionAdmission(0),
    _decode_reduce_output_attn_rmsnorm_promoted=True)
  assert _decode_reduce_output_norm_flags(block, False) == (True, True)
