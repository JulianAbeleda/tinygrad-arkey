"""Centralized KV-cache element load for the generated flash-decode tile kernel.

The tile kernel reads K/V cache elements at several sites (K stage, V stage, K_ONLY V read). Each read may apply the
same two in-register transforms, and duplicating them inline bloats the kernel. This module owns those transforms in
ONE place:

  * dequant  (quant=True):  cache is int8; recover fp16 as int8 * per-(K|V,head,token) fp16 `scale`.
  * rope-at-read (rope=True, K only): cache holds UN-roped K; rotate in-register (half-split / rotate_half convention,
    matching tinygrad.llm.model.apply_rope) using `freqs` [MAXC, Hd] (cols 0:Hd/2 = cos, Hd/2:Hd = sin) at the token's
    position. `pos_of` maps a cache token index -> rotary position (identity for absolute positions; a slot-relative
    map for the StreamingLLM ring). Rope never applies to V.

Neither transform materializes an fp16 KV buffer -- everything is fused at the load site, so the resident cache stays
int8 (quant) and un-roped (rope). Model-agnostic: keyed off the passed shapes/flags, no model-name logic.
"""
from __future__ import annotations
from tinygrad import dtypes
from tinygrad.uop.ops import UOp

def make_kv_element_loader(cache:UOp, Hd:int, kvscale:UOp|None=None, freqs:UOp|None=None, pos_of=None):
  """Return load(which, kvh, tok, e) -> half UOp for cache[which,0,kvh,tok,e] with the configured transforms.
  which: 0=K, 1=V. quant is on iff kvscale is given; rope is on iff freqs is given (K only)."""
  quant, rope = kvscale is not None, freqs is not None
  _Hh = Hd // 2
  if pos_of is None: pos_of = lambda tok: tok            # absolute position = token index (Phase 1)

  def _raw(which, kvh, tok, e):                          # dequant-aware raw element (no rope)
    val = cache[which, 0, kvh, tok, e].cast(dtypes.half)
    if quant: val = val * kvscale[which, 0, kvh, tok].cast(dtypes.half)
    return val

  def load(which, kvh, tok, e):
    val = _raw(which, kvh, tok, e)
    if rope and which == 0:                              # rotate K only
      _pos = pos_of(tok)
      _jr, _low = e % _Hh, e < _Hh
      _pair = _raw(0, kvh, tok, _low.where(e + _Hh, e - _Hh))
      _cos, _sin = freqs[_pos, _jr], freqs[_pos, _Hh + _jr]
      val = val * _cos + _low.where(-(_pair * _sin), _pair * _sin)
    return val
  return load
