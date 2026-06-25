#!/usr/bin/env python3
"""Word-structured fused Q4_K dequant + scheduler GEMV (M6 follow-on, research/default-off).

Goal: a PURE tinygrad-ops Q4_K GEMV whose packed-weight load unit is the uint32 WORD, structured so the matvec's
K-reduce maps the within-block word-column (pos//4, 0..7) to the fast/lane dim -> adjacent lanes read adjacent
packed words (the owned warp kernel's coalescing, extra/q4_k_gemv_primitive.py:53-62,480-492), but expressed in
Tensor ops (no CUSTOM). Tests whether the scheduler/lowerer can coalesce a word-granular fused-dequant matvec.

Dequant math mirrors the owned helpers / gguf.ggml_data_to_tensor (type 12); validated synthetically against the
gguf reference. words layout per block (Q4K_WORDS_PER_BLOCK=36 uint32): word0 = d(low16)|dmin(high16);
words1-3 = 6-bit scales[0-7]/mins[0-7]; words4-35 = 32 quant-words (8 nibbles each).
"""
from __future__ import annotations
from tinygrad import Tensor, dtypes

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
