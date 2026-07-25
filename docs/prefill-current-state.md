# Prefill Current State (8B + 14B)

This is the compact authority for the shipped Qwen3 gfx1100 prefill routes. **14B is live again as of 2026-07-24
(`7463a6774`)** -- it had been stuck on a slow fallback and was widely believed unrunnable; see the 14B
section below. Historical scopes and failed benchmark
banks live in Git history, not on the active repository surface.

Last updated: 2026-07-24 (late): PREFILL_SOFTMAX_REDUCE_FUSE promoted default-ON; 14B packed-WMMA load
fault fixed; llama.cpp re-measured SAME-SESSION, superseding every earlier cross-session llama figure.

## Shipped route

- Route: `prefill_wmma_lds_dbuf_generated`.
- Status: promoted default for the exact pp512 roles `attn_qo`, `attn_kv`, `ffn_down`, and `ffn_gate_up`.
- Ownership: ordinary tinygrad matmul lowered from a typed `KernelCandidateContext`; there is no hand-emitted pipe/LDS2
  kernel on the runtime path.
- Pipeline: fp16 operands, fp32 accumulation, two LDS slots, 256 threads, and exact role/shape/target admission from the
  canonical candidate set.
- Device: normal `DEV=AMD` / HIP renderer. `DEV=AMD:ISA` is optional compiler-analysis tooling and is not imported or
  selected by production execution.
- Rollback: `PREFILL_GRAPH_GEMM=0` returns to the ordinary tinygrad scheduler. Retired raw-kernel selectors fail loud.

## Durable evidence

The only promotion evidence retained in-tree is under
`bench/prefill-pure-full-kernel/multirole-buffer2-candidate-set-v1/`:

- `candidate-set.json`: four exact admitted payloads and canonical identities.
- `whole-model-quality.json`: PASS, three deterministic greedy cases, baseline/candidate token parity, route bound, and
  healthy GPU after both isolated children.
- `whole-prefill-pinned.json`: clean authority at commit `8045efcef`, pinned pp512 `3561.32 tok/s`, all four candidate
  identities observed, no missing or unexpected bindings.

The current raw-versus-practical placement is owned by BoltBeam in
`BoltBeam/docs/qwen3-8b-current-dual-roofline-20260713.md`. The retired
`bench/qk-prefill-theoretical-ceiling/latest.json` missing-evidence placeholder is intentionally removed.

## Aligned benchmark: fused vs pre-fused baseline + vs llama.cpp (2026-07-24)

Same device (RX 7900 XTX / gfx1100, ROCm), **unpinned**, guard-serialized, best-of-bursts.
8B only (14B tinygrad blocked by the packed-WMMA canary GPU HW-fault, see
`BOLTBEAM_GPU_HANG_DIAGNOSIS_HANDOFF_20260724.md`; run under
`TINYGRAD_PREFILL_PACKED_WMMA=0`, keeping 8B on graph-GEMM). The correct "what did fused
add" baseline is the **pre-fused commit `533c0aa00` (07-21, last before the flash
campaign), measured fresh on the same hardware** — NOT a fused-off patch. (An earlier
draft of this section used `_should_use_custom_kernel_prefill_attn=False`, which lands on a
pathologically slow SDPA path that collapses to DNF at long context and grossly INFLATED
the fused contribution to "+61% → +319%". That column is RETRACTED as a measurement error.)

| pp   | 07-21 pre-fused | current fused | fused gain | llama (best) | fused vs llama |
|------|-----------------|---------------|------------|--------------|----------------|
| 512  | 3461            | 3623          | +4.7%      | 3571         | +1.5%          |
| 1024 | 3223            | 3515          | +9.1%      | 3470         | +1.3%          |
| 2048 | 2819            | 3314          | +17.6%     | 3338         | −0.7%          |
| 4096 | 2213            | 3026          | +36.7%     | 3160         | −4.4%          |

Findings (honest, corrected):
1. **No regression.** Current fused ≥ the pre-fused baseline at every context (07-21 3461 →
   fused 3623 at pp512, +4.7%). At pp512 attention is not the bottleneck, so fused only
   nudges the whole-model number (~+5%) — which is why "the pp512 number barely moved" is
   the correct observation, not a regression.
2. **Fused's real contribution grows with context: +5% (pp512) → +37% (pp4096)** as the
   pre-fused attention decays (3461 → 2213, −36%). Real flash-attention thesis, but MODEST.
   The retracted "+61% → +319%" was measured against a crippled fused-off SDPA arm and is
   WRONG.
3. **Fused's biggest actual value is closing the long-context gap to llama.** Pre-fused was
   **−30% behind llama** at pp4096 (2213 vs 3160); fused pulls that to **−4.4%** (3026).
   Short context we edge llama (+1.5%/+1.3%); long context llama still edges us
   (−0.7%/−4.4%) — a crossover.
   **CORRECTION (2026-07-24, per-kernel trace):** the earlier claim here — "the residual
   long-context deficit is the graph-GEMM / per-chunk path, not attention" — is WRONG for
   the *intra-model* context decay. A `PROFILE=1` per-kernel trace (8B, gfx1100, start_pos
   0→3584; artifact `/tmp/...tax_trace_final.json`, method below) shows the per-chunk
   decomposition is **GEMM-bound and flat with context** (QKV/O + FFN ≈ 470 ms/chunk,
   93%→74% of per-chunk time) while **fused attention is the ONLY component that grows with
   KV** — 4% (pp512) → 23% (pp4096), ~8×. So our own throughput decay (3564→2976 tok/s,
   −16.5%) is **entirely attention's KV-length scaling**, not GEMM. GEMM owns the *baseline*
   ceiling (~3564 tok/s); attention owns the *long-context* decay.
   **vs-llama attribution (settled — llama.cpp `flash_attn_ext` per-kernel trace under
   rocprofv3, `-fa 1 -ngl 99`, same 8B model):** llama's prefill is the SAME qualitative
   shape — GEMM-flat, attention-grows. But llama's attention plateaus lower: its attention
   bucket is **6.7% (pp512) → 15.1% (pp4096)** (flash-attn core 4.5% → 13%), vs **ours 4% →
   23%**. Both attention cores scale ~super-linearly (llama's `flash_attn_ext` grows 22.6×
   over the sweep, comparable to ours) with GEMM/overhead shares tracking each other. So at
   pp4096 our fused attention burns **~8 extra percentage points** of GPU budget vs llama
   (23% vs 15%, ~1.5–1.8× less efficient) — and that IS the vs-llama gap. Throughput confirms
   the crossover: we edge llama short (3564 vs llama 3306 at pp512) but fall behind long
   (2976 vs 3214 at pp4096); we lose ~16% over the sweep, llama ~3%. **The long-context
   deficit — intra-model AND vs-llama — is our fused attention's KV-scaling efficiency, NOT
   GEMM or overhead. This fully retracts the earlier "not attention" claim.**
   **Next levers, ranked:** (1) fused-attention KV-scaling (owns the long-context deficit);
   (2) per-layer GEMM efficiency (owns the ~3564 baseline ceiling).
   *Trace method:* `PROFILE=1 TINYGRAD_PREFILL_PACKED_WMMA=0 prefill_whole_synced.py --mode
   authority --logits-only` populates `device_profile[sp].by_name` (per-kernel `device_ms`);
   exclude the `r_16_2374_*` kernel (the 512×151936 LM-head is a `--logits-only` artifact,
   absent from real generation). Non-`--logits-only` gives representative throughput but the
   profiler returns 0 events on the argmax path.
4. **Reconciliation:** unpinned-best fused 3623 ≈ pinned 3561 ≈ llama-best 3571 (all
   best-case ~3.6k). The historical "≈145% of llama" was an artifact of the invalid
   ~4408/4413 tinygrad number (see below) plus an un-artifacted "~3050" llama estimate; the
   real aligned figure is near-parity-to-slightly-ahead short, slightly behind long.

## AUTHORITATIVE NUMBERS (2026-07-24 late, same-session llama, supersedes the section above)

Everything below is paired, same-session, `--mode authority`, `flock`-serialized. **Every llama figure earlier
in this doc and in the sibling docs was CROSS-SESSION and is superseded** -- llama.cpp re-run tonight
(`llama-bench -fa 1 -ngl 99`, same GPU state) gives materially different short-context numbers.

| | pp | ours | llama (same session) | margin |
|---|------|------|------|------|
| 8B  | 512  | 3727 | 3347 +/- 242 | +11.4% (llama's own noisiest point, 7% stdev -- treat as soft) |
| 8B  | 4096 | **3262** | 3158 +/- 17 | **+3.3%** |
| 14B | 512  | **1948** | 1845 +/- 86 | **+5.6%** |
| 14B | 4096 | **1787** | 1642 +/- 9 | **+8.8%** |

Ahead on both models at both ends, on a valid comparison for the first time.

### CORRECTION: the 8B decay-parity claim in the section above is WRONG
That section claims the T6 fix brought our 8B decay to near-parity with llama (-12.08% vs -11.51%, "gap
closed from 5.25pp to 0.57pp"). llama's -11.51% was cross-session. **Same-session llama 8B decay is
-5.62%**, because its pp512 came in at 3347 tonight rather than 3571. Ours is -12.48%, so we are at
**~2.2x llama's 8B decay, NOT at parity.**

14B is the opposite and genuinely good: **ours -8.24% vs llama -10.99%**, i.e. our context-scaling on the
larger model beats llama's and sits closer to the -6.14% structural floor
(`docs/prefill-roofline-first-principles-20260724.md`).

Drift note, now localized: llama 8B pp4096 reproduced to **-0.1%** across sessions and pp2048 to -0.6%, but
pp512 drifted **-6.3%**. So "the box drifts ~5%" is right in magnitude but wrong in location -- it is
SHORT-context variance, not a global session shift. Deep-context cross-session comparisons were roughly fair;
short-context ones were not.

## 14B route (live again, 2026-07-24)

- Route: packed-WMMA prefill candidates (`TINYGRAD_PREFILL_PACKED_WMMA`, **default ON**), Q4_K/Q6_K,
  6/6 correctness-gated combos at `max_abs 0.0`.
- **pp512 1948, pp4096 1787 tok/s** (was 355-364 on the direct-packed fallback) = **5.0-5.4x**.
- **CORRECTION (same day, after the fact): the ordering fix `7463a6774` did NOT cause this recovery.**
  A control run at `7617ff284` -- i.e. WITHOUT the ordering change -- gives **1937 tok/s**. 14B was
  recoverable at any point simply by NOT setting `TINYGRAD_PREFILL_PACKED_WMMA=0`. The 5.0-5.4x is real but
  it is the packed-WMMA route vs the direct-packed fallback, not a fix-vs-broken delta. I claimed causation
  from a before/after in which the "before" arm was never run.
- The VRAM-starvation mechanism **cannot apply to 14B at all**: 14B logs `weights 9.0GB` (Q4_K only, 8.38
  GiB) and never realizes the fp16 overlay -- `realize_prefill_v2_weights()` returns early unless the policy
  admits FULL_RESIDENT_OVERLAY, which 14B's ~29GB fp16 footprint cannot. Only 8B realizes it (`weights
  18.9GB`), and 8B does not use packed-WMMA. So the documented 15:15 fault was NOT reproducible at HEAD
  tonight and its true trigger remains **unexplained** -- most likely session/environment state (the 18GB
  compile cache and ~20 prior model loads that the original handoff blamed before its own correction
  retracted them as "red herrings"). Treat that retraction as itself unproven.
- The ordering change is retained as defensible robustness -- a spawning canary should not run after a large
  allocation -- and is measured harmless (8B 3727/3262, 14B 1948/1787). It is NOT a fix for a live fault.
- The fault that made 14B "unrunnable" all day was NOT the kernel. Its code objects are byte-identical to
  the 6/6-gated 07-21 state (bisected compile-only,
  `docs/packed-wmma-14b-codegen-transition-bisect-20260724.md`). `6ca798568` flipped the enable default
  0->1 with gates captured in the OPT-IN configuration; that made the load-time correctness canary run on
  every default load, and `from_gguf` ran it AFTER `realize_prefill_v2_weights()` had taken the ~19GB fp16
  overlay, so the spawned canary child started VRAM-starved. Fix = run the canary BEFORE the overlay
  (`7463a6774`). Full trace: `docs/packed-wmma-14b-fault-trace-20260724.md`.
- `TINYGRAD_PREFILL_PACKED_WMMA=0` is a **rollback to the pre-`6ca798568` default**, not a workaround.
- OPEN: 14B end-to-end token parity has NOT been run with packed-WMMA live. A passing canary plus
  throughput is not the same as matching tokens. `extra/qk/prefill_flash_e2e_parity.py --only 14B`.

## Shipped default flags (2026-07-24 late)

| flag | default | worth |
|---|---|---|
| `PREFILL_SOFTMAX_REDUCE_FUSE` | **ON** (`24c8c0d94`) | 8B +1.37/+2.17/+3.71/+6.72% at pp512/1024/2048/4096; decode codegen proven byte-identical |
| `TINYGRAD_PREFILL_PACKED_WMMA` | **ON** | 14B 5.0-5.4x; 8B unaffected (graph-GEMM) |
| `PREFILL_CAUSAL_TILE_SKIP` | **OFF** (`c44905a18`) | +1.77/+1.66% measured, 3 pairs; its promotion gate correctly FAILS pending 14B evidence |
| `PREFILL_V_TRANSPOSED` | **OFF** (`fd654024e`) | REFUTED: mechanism proven (VMEM 65%->37%) but -3.4% whole-model |

## Closed branches

- Historical `~4413` and recreated `~4099` pipe results were invalid: leaked LDS geometry launched only 1/16 of the
  pipe-owned output. Correcting geometry and buffer effects produced parity but only `2095.70 tok/s`.
- The raw LDS2 oracle hung the GPU and is not benchmark-eligible.
- The old S9/S10, hybrid, raw pipe/LDS2, single-buffer, and environment-driven local-stage experiments are deleted from
  the active tree. Their conclusions remain in `docs/prefill-lessons-ledger.md` and Git history.

## Change rule

A replacement must supply an exact typed candidate, isolated correctness and GPU-health evidence, whole-model parity,
route-census identity, and pinned timing before it can replace the current default. A faster incomplete or unverified
kernel is not a performance authority.
