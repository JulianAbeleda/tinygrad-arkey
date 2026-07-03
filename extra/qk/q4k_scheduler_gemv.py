#!/usr/bin/env python3
"""Word-structured fused Q4_K dequant + scheduler GEMV (M6 follow-on, research/default-off).

Goal: a PURE tinygrad-ops Q4_K GEMV whose packed-weight load unit is the uint32 WORD, structured so the matvec's
K-reduce maps the within-block word-column (pos//4, 0..7) to the fast/lane dim -> adjacent lanes read adjacent
packed words (the owned warp kernel's coalescing, extra/qk/quant/q4_k_gemv_primitive.py:53-62,480-492), but expressed in
Tensor ops (no CUSTOM). Tests whether the scheduler/lowerer can coalesce a word-granular fused-dequant matvec.

Dequant math mirrors the owned helpers / gguf.ggml_data_to_tensor (type 12); validated synthetically against the
gguf reference. words layout per block (Q4K_WORDS_PER_BLOCK=36 uint32): word0 = d(low16)|dmin(high16);
words1-3 = 6-bit scales[0-7]/mins[0-7]; words4-35 = 32 quant-words (8 nibbles each).
"""
from __future__ import annotations
from tinygrad import Tensor, dtypes
from extra.qk.gemv_g2_lanemap import Q4KGateUpLaneMap

U16, U32, F16, F32 = dtypes.uint16, dtypes.uint32, dtypes.float16, dtypes.float32

def _f16_lo(w:Tensor) -> Tensor: return w.bitwise_and(0xffff).cast(U16).bitcast(F16).cast(F32)
def _f16_hi(w:Tensor) -> Tensor: return w.rshift(16).bitwise_and(0xffff).cast(U16).bitcast(F16).cast(F32)

def q4k_dequant_words(words:Tensor) -> Tensor:
  """words: [..., n_blocks, 36] uint32 -> dequantized weight [..., n_blocks, 256] f32, standard (group,pos) order
  (matches gguf.ggml_data_to_tensor type 12 / qk_layout.q4_k_reference)."""
  *lead, nb, w = words.shape
  assert w == 36, f"expected 36 uint32 words/block, got {w}"
  w0 = words[..., 0]                                   # [..., nb]
  d, dmin = _f16_lo(w0), _f16_hi(w0)                   # [..., nb]
  sw = words[..., 1:4]                                 # [..., nb, 3]  scale/min words
  def sbyte(idx:int) -> Tensor: return sw[..., idx//4].rshift((idx%4)*8).bitwise_and(0xff)  # [..., nb]
  sc_l, mn_l = [], []
  for g in range(8):
    if g < 4:
      sc_l.append(sbyte(g).bitwise_and(63)); mn_l.append(sbyte(4+g).bitwise_and(63))
    else:
      high = sbyte(8+g-4)
      sc_l.append(high.bitwise_and(0xf).bitwise_or(sbyte(g-4).rshift(6).lshift(4)))
      mn_l.append(high.rshift(4).bitwise_or(sbyte(4+g-4).rshift(6).lshift(4)))
  qw = words[..., 4:36]                                # [..., nb, 32] quant words
  out_g = []
  for g in range(8):
    qg = qw[..., (g//2)*8:(g//2)*8+8]                  # [..., nb, 8]  the 8 word-cols for this group-pair
    nibs = [qg.rshift(nib*8 + (g%2)*4).bitwise_and(0xf).cast(F32) for nib in range(4)]   # 4x [..., nb, 8]
    q = Tensor.stack(*nibs, dim=-1)                    # [..., nb, 8(wordcol), 4(nib)]
    dsc = (d * sc_l[g].cast(F32)).reshape(*lead, nb, 1, 1)
    dmn = (dmin * mn_l[g].cast(F32)).reshape(*lead, nb, 1, 1)
    out_g.append((dsc * q - dmn).reshape(*lead, nb, 32))   # p = wordcol*4 + nib
  return Tensor.stack(*out_g, dim=-2).reshape(*lead, nb, 256)   # [..., nb, 8, 32] -> [..., nb, 256]

def q4k_scheduler_matvec(words:Tensor, x:Tensor, out_features:int, in_features:int) -> Tensor:
  """Decode GEMV out[out_features] = W @ x, W dequantized from packed uint32 `words` via tinygrad ops (fused).
  words: flat or [out_features, n_blocks*36] uint32; x: [in_features]. The dequant load unit is the uint32 word."""
  nb = in_features // 256
  W = q4k_dequant_words(words.reshape(out_features, nb, 36))     # [out, nb, 256], lazy/fused
  xr = x.reshape(nb, 256).cast(F32)                              # [nb, 256]
  return (W * xr).sum(axis=(1, 2))                               # [out]

def q4k_scheduler_matvec_wordlane(words:Tensor, x:Tensor, out_features:int, in_features:int) -> Tensor:
  """Mode 3: same value as q4k_scheduler_matvec, but the K-reduce is restructured so the 32-quant-WORD axis is the
  FIRST reduce axis. With MV_DEQUANT GROUP(0,32) the word axis lands on the wave -> lane w reads word (4+w), 32
  adjacent lanes read 32 adjacent (contiguous) packed words -> coalesced (the owned kernel's lane4 access, no
  reshuffle needed since a block's 32 quant words are already contiguous). 256 = g(8)xpos(32); g=(gpair4,gbit2),
  pos=(wordcol8,p4 4); word index 4+w with w=(gpair,wordcol); the word's 8 nibbles = n=(p4,gbit)."""
  nb = in_features // 256
  W = q4k_dequant_words(words.reshape(out_features, nb, 36))                       # [out, nb, 256]
  W = W.reshape(out_features, nb, 4, 2, 8, 4).permute(0, 2, 4, 1, 5, 3).reshape(out_features, 32, nb, 8)  # [out, w, nb, n]
  xr = x.reshape(nb, 4, 2, 8, 4).permute(1, 3, 0, 4, 2).reshape(32, nb, 8).cast(F32)                      # [w, nb, n]
  return (W * xr).sum(axis=(1, 2, 3))                                             # [out], w(32) is first reduce axis

def q4k_scheduler_matvec_lanemap(words:Tensor, x:Tensor, out_features:int, in_features:int) -> Tensor:
  """Mode 5 / G2.3: generated Tensor route bound to the bridge-independent Q4_K LaneMap.

  This is intentionally a runtime/codegen binding probe, not a promotion candidate.  G2.0-G2.2 proved the lane map
  and packed-word address expression are representable; this arm consumes that representation in the generated
  scheduler route and stays route-clean: no owned warp custom_kernel and no lane-partition custom bridge.
  """
  lm = Q4KGateUpLaneMap(k=in_features, n=out_features)
  lm.validate()
  nb = lm.k_blocks
  W = q4k_dequant_words(words.reshape(lm.n, nb, lm.q4k_words_per_block))
  # Layout follows the G2 LaneMap: word_col is the contiguous/coalesced word axis inside each group pair.  The current
  # Tensor scheduler still materializes this as graph algebra, so this arm proves route cleanliness and correctness;
  # W==D decides whether codegen can exploit the representation without a custom bridge.
  W = W.reshape(lm.n, nb, 4, 2, lm.words_per_group, 4).permute(0, 2, 4, 1, 5, 3).reshape(lm.n, lm.group_pairs*lm.words_per_group, nb, 8)
  xr = x.reshape(nb, 4, 2, lm.words_per_group, 4).permute(1, 3, 0, 4, 2).reshape(lm.group_pairs*lm.words_per_group, nb, 8).cast(F32)
  return (W * xr).sum(axis=(1, 2, 3))
