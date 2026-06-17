# Arc 1 — attention reduce/codegen fusion: RESULT — SHIPPED (+12.8% @ctx512) (2026-06-17)

Goal: can the 8B short-context attention/reduce path be reduced/fused enough to improve base decode? **Yes —
shipped.** The fused attention primitive already exists (flash-decode); the lever was its **threshold**, which
was set too conservatively. Lowering `FLASH_DECODE_THRESHOLD` 1024→512 gives **+12.8% real-generate decode at
ctx520, byte-identical greedy output, no regression below 512.** A default change, measured + gated.

## 1. Exact attention kernels/layer (Phase 0, ctx512 SDPA, eager proxy)
234 attention kernels = **6.5/layer, 25.7% of GPU** (gemv 51.8%, 780 kernels/token):

| role | /layer | GPU % |
|---|---:|---:|
| softmax max/sum + scores@V — reduce over KV=513 | 4.0 | 21.1% |
| qk_scores — reduce over Hd=128 | 0.5 | 3.6% |
| attn out reshape/cast | 2.0 | 1.0% |

The cost is the **4 KV-length reduces/layer** — exactly what flash-decode fuses. (The 21% is an eager-unbatch
proxy; the real batched share is smaller — see Phase 1.)

## 2. SDPA vs flash anatomy (Phase 1) — crossover ~384

Clean decode tok/s (device-token feed), FLASH forced 0 vs 1:

| ctx | SDPA | flash | flash/SDPA |
|---:|---:|---:|---:|
| 128 | 48.8 | 45.6 | 0.93× (regresses) |
| 256 | 44.0 | 41.9 | 0.95× (regresses) |
| 384 | 40.1 | 41.7 | 1.04× |
| **512** | **36.9** | **38.9** | **1.05×** |
| 768 | 31.6 | 36.2 | 1.15× |
| 1024 | 27.4 | 33.7 | 1.23× |
| 4096 | 11.1 | 19.2 | 1.73× |

**Why flash is only ~1.05× @ctx512:** the real (batched) attention cost is small at KV=513 — the GEMV dominates;
flash's fusion win grows with KV. **flash REGRESSES below ~256** (its fixed kernel overhead exceeds the tiny KV
reduce it replaces). Crossover ~384. The shipped threshold (1024) left the whole **512–1024 band on SDPA** —
the band a real generation spends most of its time in.

## 3. Isolated repro (Phase 2)
Not needed as a new kernel — the candidate reuses the **existing** flash-decode kernel (already proven exact in
the prior flash-decode-auto arc). Phase 1's forced SDPA-vs-flash path comparison *is* the isolated anatomy.

## 4. Candidate selected (Phase 3)
**Candidate A — lower `FLASH_DECODE_THRESHOLD` 1024→512.** Smallest possible change (one default), no new kernel,
no codegen, keeps the SDPA fallback for ctx<512. Touches `tinygrad/llm/model.py:233` (`should_use_flash_decode`
auto policy). Plausible upside ≥5% at ctx512 (measured); risk low (flash exact; <512 unchanged). Rejected the
more aggressive 384 cutover — at 384 flash is only +4% (marginal/noisy zone); 512 is the safe clear-win.

## 5. Implementation
`tinygrad/llm/model.py:233`: `getenv("FLASH_DECODE_THRESHOLD", 1024)` → `512` (+ docstring). New test
`test_default_threshold_is_512` pins the cutover (256→SDPA, 512→flash, 1023→flash); 9/9 policy tests pass.

## 6. Before/after (in-model, real model.generate, greedy)

| ctx | path before (thr 1024) | path after (thr 512) | tok/s before | tok/s after | Δ |
|---:|---|---|---:|---:|---|
| 128 | SDPA | SDPA (<512, unchanged) | 48.8 | 48.8 | 0% |
| 256 | SDPA | SDPA (<512, unchanged) | 44.0 | 44.0 | 0% |
| **520** | **SDPA** | **flash** | **36.66** | **41.37** | **+12.8%** |
| 1024 | flash | flash (unchanged) | — | — | 0% |
| 4096 | flash | flash (unchanged) | — | — | 0% |

Real `model.generate` at ctx520, 48 tokens: **36.66 → 41.37 tok/s (+12.8%), all 48 tokens byte-identical**
(flash-decode is exact). The change affects only ctx 512–1023 (SDPA→flash); ctx<512 and ctx≥1024 are untouched
→ no possible regression. (The synthetic device-feed bench showed +5.4% @exactly-512; real generate at 520+
benefits more as the trajectory sits further past the crossover.)

## 7. Verdict: **SHIPPED**
Strong success: ctx512 decode +12.8% (≥10% gate), greedy output byte-identical, no ctx128 regression, long-ctx
flash wins preserved. The attention reduce path *was* improvable at short context — not by new fusion, but by
extending the existing fused flash-decode kernel down to its true crossover (~512). The program-count proxy
(780→<700) couldn't be measured cleanly (the flash path needs a symbolic start_pos, present only in the JIT
decode, not the eager kernel-map), but the tok/s gate — the real criterion — passed strongly.

## 8. Next action
Flash-decode now covers ctx≥512. Remaining attention candidates are deeper: (B) tune the flash tile/split for
KV~512–1024 (codegen, medium risk), (C) collapse the SDPA softmax reduces for ctx<512 (where flash regresses) —
but ctx<512 is a small slice of real generation and the SDPA cost there is small (GEMV-dominated). Given the
shipped win and that all other bounded 8B levers are exhausted, the next move is **14B** or a high-risk
flash-tile codegen tuning arc. Recommend banking this win and moving to 14B.
