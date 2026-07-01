# EB5 — Emitter Blocked Ledger Report

Date: 2026-07-01.

## Candidate: decode_bypass_kv_slice

**Fragment targeted:** E_49152_32_3 (6.69% GPU at ctx512, 40 calls/step, V cache copy before flash_partial)

**EB0 boundary contract:** V slice [Hkv=8, MAXC=4608, Hd=128] materialized by tinygrad callify because
`assigned_kv[1, 0]` is a non-trivially-strided view. 8B path avoids this via KV_IDENTITY (passes whole
assigned_kv). 14B gqa_coop_vec path does not.

**EB1 classification:** Case A (real KV write dependency) + cache-warming role. The copy is not dead work.

**EB2 implementation:** `DECODE_BYPASS_KV_SLICE=0` (default-off):
- `flash_partial_coop_vec_kv_flat_kernel`: accesses V at offset `Hkv*MAXC*Hd` in flat combined buffer
- `flash_decode_attention_kv_flat`: passes `assigned_kv.reshape(2*Hkv*MAXC*Hd)` instead of `[1,0]` slice
- `model.py:1177`: bypass guard for `B==1` and `FLASH_VARIANT==gqa_coop_vec`

**EB3:** CORRECTNESS_PASS (rel_rmse=0.00e+00, 5 steps, FLASH_DECODE=1 forced)

**EB4:** ctx512 50.1 → 45.3 tok/s **(-4.8 tok/s, -9.6%)** — REFUTED

## Why the bypass regresses

The E_49152 copy kernel is not pure overhead. It writes V into a fresh L2-warm buffer that flash_partial_coop_vec
then reads sequentially. Without the copy:
- flash_partial reads V cold from `cache_kv` (MAXC=4608 ≫ L2 capacity)
- 160 workgroups (Hkv=8 × S=4 × G=5) compete for cold V reads
- The regression (-9.6%) exceeds E_49152's own cost (6.69%) — the copy was more than paying for itself

## Decision

**DEFER_NOT_DEFAULT_ON.** The DECODE_BYPASS_KV_SLICE code is kept (default-off, correct), but the bypass is
refuted as a performance improvement. `do_not_retry=False`.

## Reopen condition

LDS-staged flash_partial_coop_vec: prefetch V chunk into on-chip shared memory at workgroup start, then accumulate
from LDS. This achieves the same cache-warming effect without the global copy. Requires:
1. LDS-alloc UOp in tinygrad's emitter (`PRIMITIVE_MISSING`)
2. A new generated flash_partial variant with explicit LDS staging

Classification updated in BoltBeam: `decode_bypass_kv_slice` → refuted, `decode_bypass_kv_slice_lds` → PRIMITIVE_MISSING.

## Key insight for future work

E_49152_32_3 is a **BENEFICIAL_CACHE_WARM** kernel, not a pure scheduling artifact. The EMITTER_BLOCKED
classification in SF1 was correct about the scheduling mechanism but missed the performance function.
Any future attempt to remove this copy must provide an equivalent L2 warm-up mechanism for V reads in flash_partial.
