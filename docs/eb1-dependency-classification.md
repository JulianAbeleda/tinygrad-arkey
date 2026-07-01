# EB1 — Dependency Classification for E_49152_32_3

Date: 2026-07-01. Follows EB0.

## Classification: Case A — Real sequential dependency (V from persistent KV cache), BUT cache-warming beneficial

The E_49152_32_3 kernel is NOT a pure scheduling artifact. It serves two roles simultaneously:
1. **Ordering enforcer**: tinygrad's `callify` uses it to sequence the KV store before the flash V read.
2. **Cache warmer**: the copy writes V into a fresh L2-warm buffer, improving flash_partial's V read latency.

Role (2) was discovered empirically (EB4 W==D showed -9.6% regression when E_49152 is bypassed). It is structural
— the flash_partial_coop_vec kernel reads V many times (once per split × per G group), and having V warm in L2
(from the just-executed copy) is materially faster than reading V cold from the persistent KV cache.

## Source location

`tinygrad/llm/model.py:1178`:
```python
out = flash_decode_attention(q.reshape(Hq, Hd), assigned_kv[0, 0], assigned_kv[1, 0], ...)
```

Inside `flash_decode_attention` in `extra/qk_flash_decode.py:1338-1350`:
```python
vc_f = v_full.reshape(Hkv * MAXC * Hd)   # v_full = assigned_kv[1, 0]
po = Tensor.empty(...).custom_kernel(prob, vc_f, fxn=flash_partial_coop_vec_kernel(...))[0]
```

The `[1, 0]` indexing of `assigned_kv` creates a non-trivial stride chain that tinygrad's callify cannot alias
to the underlying `cache_kv` buffer, forcing the copy kernel.

## Experiment (EB2-EB4)

Implemented `DECODE_BYPASS_KV_SLICE=0` (default-off) in:
- `extra/qk_flash_decode.py`: `flash_partial_coop_vec_kv_flat_kernel` + `flash_decode_attention_kv_flat`
- `tinygrad/llm/model.py:1177`: bypass guard that passes `assigned_kv.reshape(2*Hkv*MAXC*Hd)` directly

The bypass passes the combined flat KV buffer (`assigned_kv.reshape(2*Hkv*MAXC*Hd)`) to flash_partial instead
of the `[1, 0]`-indexed V slice, eliminating the E_49152 copy.

**EB3 correctness**: CORRECTNESS_PASS (rel_rmse=0.00e+00, top1 identical across 5 steps, FLASH_DECODE=1).

**EB4 W==D**:
| ctx | baseline | bypass | delta |
|-----|----------|--------|-------|
| 128 | 52.3 tok/s | 52.3 tok/s | 0.0 |
| 512 | 50.1 tok/s | 45.3 tok/s | **-4.8 (-9.6%)** |

**Verdict: EB4_REFUTED_WD_REGRESSION**

## Root cause of regression

The bypass eliminates the E_49152 copy, but flash_partial then reads V COLD from the persistent KV cache.
The KV cache at MAXC=4608 is 8 × 4608 × 128 × 2 = ~9.4 MB for V. At ctx512, flash_partial_coop_vec
uses S = ceil(512/128) = 4 splits × Hkv=8 × G=5 = 160 workgroups. Each workgroup reads V[kv_head, 0..L, :].
With the copy (E_49152), V is warm in L2 (just written). Without the copy, V reads are cold HBM loads.

The 9.6% regression (compared to E_49152's 6.69% share) confirms that E_49152's cache-warming effect provides
MORE value than its execution time costs. The net effect of removing it is negative.

## What would reopen this

A flash_partial variant that explicitly prefetches V into LDS (Local Data Share, on-chip shared memory) at the
start of each workgroup. This would achieve cache-warming without the global copy. Requires:
- LDS-staged flash_partial_coop_vec: each workgroup loads its V chunk into LDS, then computes the partial sum.
- This is a handwritten or carefully-generated kernel with LDS allocation and barrier synchronization.
- Currently NOT representable as a generated UOp path (LDS alloc is missing from tinygrad's emitter).
