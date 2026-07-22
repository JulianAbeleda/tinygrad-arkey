# Piece 2-A Probe: REDUCE-preserving fusion breakpoint in attention

Date: 2026-07-21
Parent: Piece 1 fix (cb6e760e0) — WMMA survives fused matmul+epilogue-reduce
Probes at: `TC_OPT=2`, fp16, 512×512, DEV=AMD gfx1100

## 1. Incremental expression table

| # | Expression | Kernels (total) | Kernels (compute) | WMMA calls | Score spilled? |
|---|-----------|-----------------|-------------------|------------|----------------|
| 1 | `(a@b).max(-1)` | 9 | **1** | 3 | No — fused |
| 2 | `(a@b).sum(-1)` | 9 | **1** | 3 | No — fused |
| 3 | `(a@b) - (a@b).max(-1,keepdim=True)` | 11 | **3** | 3 | **Yes — 512×512 HBM buffer** |
| 4 | `((a@b) - (a@b).max(-1,keepdim=True)).exp().sum(-1)` | 11 | **3** | 3 | Yes |
| 5 | `(a@b).softmax(-1)` | 12 | **4** | 3 | Yes |
| 6 | `(a@b).softmax(-1) @ c` | 15 | **4** | 6 | Yes |

Init kernels: 5 (a.realize) + 3 (b.realize) = 8 for probes 1–5.
Probe 6 adds 3 more init kernels for c.realize (cache-miss variant), total 11 init.

## 2. Breakpoint: Expression 3

REDUCE-preserving fusion holds for expressions 1 and 2 (single-consumer reduce after matmul). It **breaks at expression 3** when the broadcast-subtract op makes the matmul output feed **two consumers**:

1. `max(-1, keepdim=True)` — a reduce consumer
2. `-` (broadcast subtract) — an elementwise consumer of the original score

The scheduler cannot fuse a kernel whose output has multiple consumers of different shapes. The 512×512 matmul score is materialized to HBM, then the max-reduce kernel and the subtract kernel read it separately.

## 3. Kernel signatures at breakpoint (Expression 3)

```
TC(0): [(1, 512)] [(2, 512)] [(0, 512)]

r_16_8_32_4_2_2_2_2_32_2_...  opts: (TC, UPCAST(2), LOCAL(4))
  → matmul a@b → 512×512 score buffer (WMMA)

r_512_16_32_bab34e...  opts: (GROUPTOP(16),)
  → max reduction over last axis → 512 output

E_64_8_8_16_4_...  opts: (UPCAST(4), LOCAL(8), LOCAL(16))
  → elementwise subtract: score - broadcast(max_result)
```

The matmul kernel (`r_...` with TC opts) writes the full 512×512 fp16 score (512 KB) to HBM. The subsequent `r_...` reduce kernel reads it back for the max. The `E_...` elementwise kernel reads both the score and the broadcast max to compute the numerator.

## 4. Softmax adds one more kernel (Expression 5)

```
Softmax compute kernels (4 total):
1. r_... (TC)         — matmul a@b → 512×512 score (WMMA)
2. r_512_16_32_...     — max reduction
3. r_512_16_32_...     — exp + sum reduction (softmax denominator)
4. E_64_8_8_16_4_...   — normalize: (score - max).exp() / sum
```

The `exp().sum()` after the score-minus-max adds a third reduce kernel (softmax's denominator reduction), making 4 compute kernels total for the full softmax. The score remains spilled.

## 5. Full attention (Expression 6)

```
Probe 6 compute kernels (4 total):
1. r_... (TC)         — QKᵀ matmul → 512×512 score (WMMA) 
2. r_512_16_32_...     — max reduction
3. r_512_16_32_...     — exp + sum reduction
4. r_... (TC)         — PV matmul (WMMA)
```

Both matmuls get WMMA (TC appears twice, WMMA=6). The score is spilled between QKᵀ and softmax, and the probabilities are spilled between softmax and PV. No kernels fused across the score boundary.
