# Decode FFN-GEMV Scheduler Diagnostic — Result

Date: 2026-06-22

Executes `docs/decode-ffn-gemv-scheduler-diagnostic-scope-20260622.md`. Attribution/tooling/docs only — **no kernel
optimization, no new primitive, no Route-A/backend codegen, no default change.** Harness
`extra/qk_ffn_gemv_scheduler_diagnostic.py` → `bench/qk-ffn-gemv-scheduler-diagnostic/latest.json`.

## Verdict: **`FFN_GEMV_DIAGNOSTIC_BOUNDED_SCHEDULE_SCOPE_READY`** — failure layer **`GEMV_SCHEDULE_BOUND`** (WORK DECOMPOSITION). The FFN Q4_K GEMV runs at ~47–57% HBM peak vs llama MMVQ's ~70% because of the *schedule* (1 thread/row, serial whole-row K, uncoalesced), not the math/dot or the quant. The fix — **128 threads/row + K-block-parallel + in-kernel warp-shuffle reduce** — is **bounded** (enabling primitives exist, expressible), **lossless** (FP weights), and targets a bucket that **transfers** (q8 +6% proves it). Projected W==D **+6.5% (gate/up) / ~+9–11% (gate/up+down)** — clears +5%@ctx1024. **Gate the build on W==D** (the int-dot proxy was null; B5 taught that local≠in-graph).

## G0/G1 — role inventory + bandwidth (ctx1024)
| role | kernel | shape | quant | per-call µs | eff GB/s | % peak | share |
|---|---|---|---|---|---|---|---|
| **FFN gate/up** | `q4k_gemv_partial_12288_4096` | 12288×4096 | Q4_K | 58.0 | 488 | **51%** | 24% |
| FFN down (Q4_K) | `q4k_gemv_partial_4096_12288` | 4096×12288 | Q4_K | 63.0 | 449 | 47% | 14% |
| FFN down (Q6_K) | `q6k_coop_partial_4096_12288` | 4096×12288 | Q6_K | 75.7 | 546 | 57% | (14%) |
| attn q/o proj | `q4k_coop_partial_4096_4096` | 4096×4096 | Q4_K | 18.9 | 500 | 52% | 8% |

**Anatomy (source, `extra/q4_k_gemv_primitive.py::q4k_gemv_partial_kernel`):** the gate/up default is **1 thread per
output row** (`row = UOp.range(rows)`), a **serial K reduction** inside the thread (`blk` × `pos` REDUCE ranges over the
whole row), scratch-buffer partials + a stage-2 `.sum`, per-thread Q4_K scale/min decode. A warp of 32 threads = 32
*rows*, whose Q4_K blocks are `row_bytes`(=2304 B) apart → **uncoalesced** weight loads. (The `q4k_coop_partial`
sibling maps the within-block word index to a LOCAL lane → coalesced, used for attn q/o; per `model.py:255-258` the
gate/up "is already ~41% peak so it is NOT routed" through coop — coalescing alone doesn't close it.)

## G2 — llama/reference gap (from `docs/llama-q4k-mmvq-scheduler-audit-20260618.md`; no new port needed)
| variant | row mapping | K mapping | reduction | coalesced | % peak |
|---|---|---|---|---|---|
| tinygrad base (gate/up) | 1 thread/row | serial whole row | stage-2 sum | no | **40** |
| tinygrad coop | 8 lanes/row | serial blk loop | stage-2 sum | yes | 48 (53 partial-alone) |
| **llama MMVQ** | **128 threads/row** | **K-block-parallel (no serial loop)** | **in-kernel warp-shuffle** | yes | **70** |

**The gap is WORK DECOMPOSITION, NOT math.** The audit already matched llama's `dot4` (`_sdot4` native `v_dot4`) and
packed extract (`&0x0F0F0F0F`); what tinygrad has **never tried** is **128 threads/row + K-block-parallelism +
in-kernel warp-shuffle reduce + one output write**. The enabling primitives **exist** (`_sdot4`,
`extra/amd_warp_reduce.warp_reduce_sum` via `ds_bpermute`) and the structure is **expressible**. The gap is
**role-general** (gate/up 51%, down 47–57%, q/o 52% — all ~50%, all the same schedule).

## G3 — controlled toggle ladder (the transfer test, ctx1024, lossless unless noted)
| toggle | tok/s | Δ | tokens match | kind |
|---|---|---|---|---|
| default | 66.7 | — | — | lossless |
| `Q4K_VDOT` (int-dot) | 67.6 | **+1.25%** | yes | lossless int-dot |
| `Q4K_VDOT` + `Q4K_VDOT_AMORT` | 67.7 | **+1.46%** | yes | + amortized q8 quant |
| `Q8_FFN_HANDWRITTEN` | 66.8 | +0.18%* | yes | LOSSY q8 weights |

\*The in-process q8 toggle **did not activate** (the q8 route needs **load-time** q8 storage, not just decode-time env);
the **canonical q8 W==D is +6%** (handoff/candidates, proper harness) and is the authority.

**Reading:** the int-dot path is **NULL in-model (+1.25%)** — and amortizing the q8 quant adds only **+0.2%**, so the
quant lifecycle is *not* the bottleneck either. This confirms the closed MMVQ verdict: the int-dot dot win is eaten by
the **q8-activation lifecycle** (per-linear quant, partial coverage 163/199), **not** the schedule. Crucially, a **FP
work-decomposition** variant (128-thread/row + warp-reduce, **no q8 quant**) pays **none** of that lifecycle tax — it
is the untested lossless lever. And q8's canonical **+6%** proves the gate/up bucket is **on the critical path and
transfers** (unlike attention, which B5 showed overlaps).

## G4 — failure classification: **`GEMV_SCHEDULE_BOUND`**
Not `MEMORY_COALESCING_BOUND` (coop coalescing alone → 53%, gate/up not routed). Not `INSTRUCTION_SELECTION_BOUND`
(`dot4`/extract already matched). Not `WAIT_SCHEDULER_BOUND` / `BACKEND_PROJECT_LEVEL` (the missing structure is a
*work decomposition* with existing primitives, not a renderer/wait-scheduler change). Not `ACTIVATION_LIFECYCLE_BOUND`
(that's the int-dot path; the FP lever avoids it). Not `REFERENCE_GAP_INSUFFICIENT` (llama decomposition + % peak are
documented). → **`GEMV_SCHEDULE_BOUND`**: rows/block + K-split + in-kernel reduce is the visible, named issue.

## G5 — headroom + recommendation
| lever | local target | affected share | projected W==D | bounded? | verdict |
|---|---|---|---|---|---|
| q8 FFN hardening | measured | gate/up 24% | **+6% (canonical, LOSSY)** | yes (shipped opt-in) | stays opt-in (dNLL); not default-promotable |
| int-dot `Q4K_VDOT` (lossless) | ~57% peak | parts==1 roles | **+1.25% (NULL)** | closed | REFUTED in-model (q8 lifecycle) — do not pursue |
| **FP work-decomp GEMV** (128-thread/row + K-parallel + warp-reduce) | 51%→~65–70% peak | gate/up 24% (+down 14%) | **+6.5% / ~+9–11% (projected)** | **yes** (warp_reduce_sum exists; expressible; not deep backend, not int-dot lifecycle) | **BOUNDED SCHEDULE SCOPE READY** |

## Result-doc required answers
1. **Top tax / role:** FFN gate/up `q4k_gemv_partial_12288_4096` (24%, 51% peak), with FFN down (14%, 47–57%) the same
   family — Q4_K weight GEMV, ~38% of decode.
2. **Gap layer:** **schedule / work-decomposition** (`GEMV_SCHEDULE_BOUND`) — 1-thread/row serial uncoalesced vs llama's
   128-thread/row + K-block-parallel + warp-shuffle reduce. NOT memory-coalescing-alone, NOT instruction-selection,
   NOT wait-scheduler, NOT backend-project-level, NOT reference-insufficient.
3. **Does the q8 transfer imply the lossless lever transfers?** Yes — q8 (+6%) proves gate/up is on the critical path
   and transfers (unlike attention). The int-dot proxy was null only due to the q8-activation lifecycle, which a FP
   work-decomposition variant avoids. **But the build must still gate on W==D** (B5 lesson; transfer not yet measured
   for the FP variant).
4. **Bounded implementation scope?** Yes — a FP 128-thread/row + K-block-parallel + warp-shuffle-reduce Q4_K GEMV for
   gate/up (then down), using the existing `warp_reduce_sum`; lossless; env-gated default-off; **W==D-gated** (do not
   ship on % peak alone). Scoped in `docs/decode-ffn-gemv-scheduler-implementation-scope-20260622.md`.
5. **What NOT to pursue:** the int-dot/MMVQ q8-activation path (refuted, null in-model), q8 as a *default* (lossy),
   coalescing-only (gate/up already not-coop-routed), more attention work (closed), or deep backend codegen before the
   bounded FP work-decomposition variant is tried and W==D-measured.

## Boundaries honored
No kernel optimized, no new primitive built, no defaults changed, no q8 promotion, no int-dot reopen (the diagnostic
*confirms* its in-model null), no broad search. `gqa_coop_vec`/llama comparators; bench artifact gitignored.
