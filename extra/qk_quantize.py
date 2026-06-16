#!/usr/bin/env python3
"""fp16/fp32 weights -> Q4_K block bytes (the inverse of q4_k_reference). Port of llama.cpp's
quantize_row_q4_K (make_qkx2 search, no-imatrix weights). Offline numpy, not a hot path.

Q4_K block = 144 bytes / 256 weights = 36 uint32 words: word[0]=d|dmin (fp16 each),
word[1..3]=12 bytes of packed 6-bit (scale,min) x8, word[4..35]=256x4-bit quants.
Layout exactly matches extra/q4_k_gemv_primitive._q4k_group_params / _q4k_quant.
"""
import numpy as np

QK_K = 256

def _make_qkx2(x):  # x: [M,32] -> (scale[M], the_min[M], L[M,32]) ; reconstruction x ~= scale*L - the_min
  M = x.shape[0]
  av = np.sqrt((x*x).mean(1, keepdims=True))
  w = av + np.abs(x)                                   # llama no-imatrix weights
  sum_w = w.sum(1); sum_x = (w*x).sum(1)
  mn = np.minimum(x.min(1), 0.0); mx = x.max(1)
  scale = np.zeros(M); the_min = -mn.copy(); L = np.zeros((M, 32), np.int32)
  rng = mx - mn
  ok = rng > 0
  iscale = np.where(ok, 15.0/np.where(ok, rng, 1), 0.0)
  Lcur = np.clip(np.round(iscale[:, None]*(x - mn[:, None])), 0, 15)
  sc0 = np.where(ok, 1.0/np.where(ok, iscale, 1), 0.0)
  best = (w*(sc0[:, None]*Lcur + mn[:, None] - x)**2).sum(1)
  scale = sc0.copy(); minv = mn.copy(); L = Lcur.astype(np.int32)
  for is_ in range(9):                                 # nstep=8 search steps (load-speed vs quality; round-trip stays ~exact)
    isc = np.where(ok, (-1.0 + 0.1*is_ + 15.0)/np.where(ok, rng, 1), 0.0)
    La = np.clip(np.round(isc[:, None]*(x - mn[:, None])), 0, 15)
    sl = (w*La).sum(1); sl2 = (w*La*La).sum(1); sxl = (w*La*x).sum(1)
    D = sum_w*sl2 - sl*sl
    good = D > 0
    ts = np.where(good, (sum_w*sxl - sum_x*sl)/np.where(good, D, 1), 0.0)
    tm = np.where(good, (sl2*sum_x - sl*sxl)/np.where(good, D, 1), 0.0)
    posm = tm > 0
    ts = np.where(posm & good, np.where(sl2 > 0, sxl/np.where(sl2 > 0, sl2, 1), ts), ts)
    tm = np.where(posm, 0.0, tm)
    mad = (w*(ts[:, None]*La + tm[:, None] - x)**2).sum(1)
    upd = good & (mad < best)
    scale = np.where(upd, ts, scale); minv = np.where(upd, tm, minv)
    best = np.where(upd, mad, best); L = np.where(upd[:, None], La.astype(np.int32), L)
  return scale, -minv, L  # the_min = -min (>=0)

def quantize_q4_k(weight: np.ndarray) -> np.ndarray:
  """weight: [rows, k] float -> uint32 words [rows * k//256 * 36]."""
  rows, k = weight.shape
  assert k % QK_K == 0
  nb = k // QK_K
  x = weight.astype(np.float32).reshape(rows*nb, QK_K)            # [B, 256] super-blocks
  B = x.shape[0]
  sub = x.reshape(B*8, 32)                                        # [B*8, 32] sub-blocks
  scales, mins, L = _make_qkx2(sub)
  scales = scales.reshape(B, 8); mins = mins.reshape(B, 8); L = L.reshape(B, 8, 32)
  max_scale = scales.max(1); max_min = mins.max(1)
  d = np.where(max_scale > 0, max_scale/63.0, 0.0)
  dmin = np.where(max_min > 0, max_min/63.0, 0.0)
  inv_s = np.where(max_scale > 0, 63.0/np.where(max_scale > 0, max_scale, 1), 0.0)
  inv_m = np.where(max_min > 0, 63.0/np.where(max_min > 0, max_min, 1), 0.0)
  ls = np.clip(np.round(inv_s[:, None]*scales), 0, 63).astype(np.int32)   # [B,8] 6-bit
  lm = np.clip(np.round(inv_m[:, None]*mins), 0, 63).astype(np.int32)
  # re-quantize each weight with the QUANTIZED scales: q = round((x + dmin*lm)/(d*ls))
  dj = (d[:, None]*ls).reshape(B, 8, 1); dmj = (dmin[:, None]*lm).reshape(B, 8, 1)
  xr = x.reshape(B, 8, 32)
  q = np.where(dj != 0, np.clip(np.round((xr + dmj)/np.where(dj != 0, dj, 1)), 0, 15), 0).astype(np.uint32)  # [B,8,32]
  # pack into 36 uint32 words/block
  words = np.zeros((B, 36), np.uint32)
  dh = d.astype(np.float16).view(np.uint16).astype(np.uint32)
  dmh = dmin.astype(np.float16).view(np.uint16).astype(np.uint32)
  words[:, 0] = dh | (dmh << 16)
  # 12 scale bytes (word[1..3]): the Q4_K 6-bit packing (inverse of _q4k_group_params)
  sb = np.zeros((B, 12), np.uint32)
  for j in range(4):
    sb[:, j]   = (ls[:, j] & 63) | ((ls[:, j+4] >> 4) << 6)
    sb[:, 4+j] = (lm[:, j] & 63) | ((lm[:, j+4] >> 4) << 6)
    sb[:, 8+j] = (ls[:, j+4] & 0xf) | ((lm[:, j+4] & 0xf) << 4)
  for w_i in range(3):
    words[:, 1+w_i] = sb[:, w_i*4] | (sb[:, w_i*4+1] << 8) | (sb[:, w_i*4+2] << 16) | (sb[:, w_i*4+3] << 24)
  # 128 qs bytes (word[4..35]): q[grp,pos] -> word (grp//2)*8 + pos//4, bit (pos%4)*8 + (grp%2)*4
  for grp in range(8):
    for pos in range(32):
      wi = 4 + (grp//2)*8 + pos//4
      sh = (pos % 4)*8 + (grp % 2)*4
      words[:, wi] |= q[:, grp, pos] << sh
  return words.reshape(rows*nb*36)
