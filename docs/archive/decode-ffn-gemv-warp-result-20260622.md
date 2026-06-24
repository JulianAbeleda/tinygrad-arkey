# Decode FFN-GEMV `q4k_gemv_warp` — Result

Date: 2026-06-22

Executes `docs/decode-ffn-gemv-scheduler-implementation-scope-20260622.md` (the bounded lossless work-decomposition
lever from `FFN_GEMV_DIAGNOSTIC_BOUNDED_SCHEDULE_SCOPE_READY` / `GEMV_SCHEDULE_BOUND`). Lossless FP only; default-off;
W==D-gated. No int-dot/MMVQ reopen, no q8, no attention, no backend codegen, no default change.

## Verdict: **`Q4K_GEMV_WARP_WD_PASS`** — the lossless work-decomposition GEMV clears the W==D gate handily and BYTE-IDENTICALLY. Gate/up+down: **+9.78%@ctx1024 / +8.71%@ctx4096 / +9.83%@ctx512** (decode 66.7→73.9 / 61.0→66.8 / 68.3→75.7 tok/s). It is **lossless** (greedy byte-identical) → **default-eligible** (owner call); kept default-off per the task boundary.

## The kernel (`extra/q4_k_gemv_primitive.py::q4k_gemv_warp_kernel`)
llama's MMVQ *work decomposition* in lossless FP: **one workgroup per row = 32 threads (one gfx1100 wave)**; `lane =
block_group(0..3)·8 + lane4(0..7)` — `lane4` = within-block word index (8 adjacent lanes read 8 adjacent packed words
→ **coalesced**), `block_group` splits the row's K-blocks into 4 **K-parallel** chunks across the wave. Each lane
FP-accumulates its blocks in a register, then an **in-kernel `warp_reduce_sum` (ds_bpermute)** → **one output store**
(no stage-2 `.sum` partials buffer). Decode/math identical to the default `_q4k_block_dot_packed_load` → exact up to fp
reassoc. (Bug found+fixed: `warp_reduce_sum` needs `UOp.special(32,"lidx0")`, not an `AxisType.LOCAL` range.)

## I0 — baseline reproduced
gate/up `q4k_gemv_partial_12288_4096` 488 GB/s / 51% peak; down 47–57%; no drift.

## I1 — local A/B (`extra/qk_ffn_gemv_warp_ab.py`, synthetic Q4_K, DEBUG=2 device time)
| role | shape | opted default | coop (8-lane) | **warp (32-lane)** | warp/opted | correct |
|---|---|---|---|---|---|---|
| gate/up | 12288×4096 | 78.6µs (38%) | 67.1µs (44%) | **59.8µs (49%)** | **1.31×** | rel 2e-6 ✓ |
| down (Q4_K) | 4096×12288 | 90.6µs (33%) | 79.5µs (37%) | **65.9µs (45%)** | **1.37×** | rel 5e-6 ✓ |

The warp beats both the opted default (`LOCAL:0:64/32`) and the coop (8-lane + stage-2 sum), losslessly. (Synthetic
% peak runs a few points below in-model; W==D is the arbiter — and it over-delivered.)

## I2/I3/I4 — env-gated route + W==D (`extra/qk_ffn_gemv_warp_wd.py`, in-process interleaved A/B)
Route in `model.py` (`Q4K_GEMV_WARP=1` gate/up; `+Q4K_GEMV_WARP_DOWN=1` Q4_K down), shape/arch-guarded (gfx1100,
in/out 4096↔12288, k_blocks%4==0), fallback to the default `q4k_gemv_partial` on any mismatch. **Default OFF.** Route
fires (72 gate/up / 90 +down kernels), greedy byte-identical.

| scope | ctx512 | ctx1024 | ctx4096 | tokens |
|---|---|---|---|---|
| gate/up only | +8.69% | +8.58% | +7.56% | identical |
| **gate/up + Q4_K down** | **+9.83%** | **+9.78%** | **+8.71%** | identical |

**Gate `(d1024 ≥ +5% OR d4096 ≥ +7%), no ctx512 regress, tokens match` → PASS** (gate/up alone already passes;
+down is strictly better, no regression). Exceeds the diagnostic's projected +6.5% gate/up / ~+11% combined — the
work-decomposition lever **transfers** (the FFN GEMV is on the critical path, as q8 predicted; the opposite of
attention's B5 saturation).

## Correctness / losslessness
Standalone warp-vs-default `rel ≤ 5e-6`; in-model greedy **0/40 mismatches** (byte-identical). The kernel runs the
**same** Q4_K dequant + dot as the default, only reassociated (warp tree-sum vs serial sum) — exactly the "lossless up
to fp reassoc" class already accepted for the coop/flash variants. **Not q8, no quant change, no quality loss.**

## Lifecycle
Registered decode_eval candidate `q4k_gemv_warp_ffn` (`bench/qk-decode-eval/candidates.json`), comparator the default
`q4k_gemv_partial`, **lossless**, W==D PASS. **`default_eligible=true`** in principle (lossless + passes + no
regression) — but kept **default-off** (`Q4K_GEMV_WARP` / `Q4K_GEMV_WARP_DOWN`) pending owner approval per the task's
no-default-change boundary. Q6_K down (the other half of FFN down, ×18 layers) is a bounded follow-on (needs a Q6_K
warp variant) for further headroom.

## What this means
First decode primitive to **clear the W==D promotion gate** since the attention arc: a **lossless ~+9.8%@ctx1024 /
~+8.7%@ctx4096** whole-decode gain on the FFN Q4_K weight GEMV (decode moves from ~67% → ~73% of llama). It validates
the time-tax → diagnostic → bounded-scope chain: the named `GEMV_SCHEDULE_BOUND` work-decomposition lever was real,
bounded, lossless, and **transferred**.

## Deliverables
`q4k_gemv_warp_kernel` in `extra/q4_k_gemv_primitive.py`; `extra/qk_ffn_gemv_warp_ab.py` (local A/B);
`extra/qk_ffn_gemv_warp_wd.py` (W==D); env-gated route in `tinygrad/llm/model.py` (default-off);
`bench/qk-ffn-gemv-warp/{latest,wd}.json`; this doc.

## Boundaries honored
Lossless FP only (no int-dot/MMVQ reopen, no q8, no quant change). No attention, no backend/Route-A codegen. Default
unchanged (route default-off). `q4k_gemv_partial`/llama comparators. W==D-gated (not shipped on % peak). Bench
artifacts gitignored.
